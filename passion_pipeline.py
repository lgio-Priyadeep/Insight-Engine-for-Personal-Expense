"""
Memory budget: process_pipeline performs deep copies to ensure the input DataFrame
is never mutated (defensive-copy ownership).
Peak memory may exceed 5x debits DataFrame size during integration because of work_df,
enrich_df, detect_df/output_df, PassionResult copy, and PipelineResult replacement copy.
For DataFrames over 500MB, consider enabling pandas Copy-on-Write mode
and testing under production memory limits.
If a cooperative stage deadline occurs, the core pipeline result is returned unchanged.
The passion budget guard checks between substages and selected internal checkpoints. It is not a process-level kill switch. Large-input protection depends on INSIGHT_ENGINE_PASSION_MAX_ROWS and avoiding unbounded regex/vectorized joins inside substages.
"""
import os as _os
import re as _re
import time
import math
import datetime
import threading
import random
import pandas as pd
import numpy as np
from numbers import Integral, Real
from typing import Any

from schema import Col
from config_passion import (
    MAX_SPIKE_CANDIDATES, PIPELINE_BUDGET_MS, PIPELINE_TOP_N,
    PIPELINE_HARD_TIMEOUT_MS,
    # D5: Moved from inline try/except inside process_pipeline — dead code removed.
    # config_passion is already a hard dependency; if it fails, this module never loads.
    PASSION_MIN_DEBIT_ROWS,
)
from passion_utils import assert_columns_exist, coerce_bool_column, _safe_isna, safe_numeric
from candidate import Candidate
from banned_content import contains_banned_content
from logger_factory import get_logger

# FIX C1: Import at module top level instead of inside early return block.
from pipeline_result import PassionResult

logger = get_logger(__name__)
# Standard Python logger for pytest caplog interception (caplog cannot see
# the custom JSON-to-stderr emitter used by get_logger).
# IMPORTANT: must NOT be under __name__ namespace — get_logger(__name__) sets
# propagate=False on that logger (and all children), making it invisible to caplog.
# 'insight_engine.caplog_bridge' is independent: its parent is root (propagate=True).
import logging as _std_logging
_std_logger = _std_logging.getLogger("insight_engine.caplog_bridge")

__all__ = ["process_pipeline"]

_init_lock = threading.Lock()
_init_complete = threading.Event()
_init_failed = threading.Event()
_init_in_progress = False

# FIX H2: _PERMANENT_STARTUP_ERRORS. OSError is NOT included — it is retryable
# with bounded retry and backoff (D5). One transient OSError must never permanently
# poison the process.
_PERMANENT_STARTUP_ERRORS = (
    RuntimeError,
    TypeError,
    ValueError,
    ImportError,
    ModuleNotFoundError,
    AttributeError,
)

# D2: Renamed from _empty_passion_result → _neutral_passion_result.
# Returns the original input rows with neutral/default passion columns — NOT an empty DataFrame.
# "empty" was misleading; the result has the same row count as input, with zeroed/NA passion fields.
def _neutral_passion_result(df: pd.DataFrame, reason: str = "empty_input") -> "PassionResult":
    level = logger.warning if reason == "substage_failure" else logger.info
    level("passion_neutral_result", extra={"reason": reason})
    neutral_debits = df.copy(deep=True)
    if Col.INFERRED_SUBCATEGORY not in neutral_debits.columns:
        neutral_debits[Col.INFERRED_SUBCATEGORY] = pd.Series(pd.NA, dtype="object", index=neutral_debits.index)
    else:
        neutral_debits[Col.INFERRED_SUBCATEGORY] = pd.NA

    if Col.SUBCATEGORY_CONFIDENCE not in neutral_debits.columns:
        neutral_debits[Col.SUBCATEGORY_CONFIDENCE] = pd.Series(0.0, dtype="float64", index=neutral_debits.index)
    else:
        neutral_debits[Col.SUBCATEGORY_CONFIDENCE] = 0.0

    return PassionResult(
        debits=neutral_debits,
        candidates=(),
        insights=(),
        passion_signals=(),
    )

# Fix 14: Waiting threads use Event.wait(timeout) instead of busy-spinning
# with time.sleep(0.1). This eliminates CPU spin while another thread initializes
# and raises TimeoutError if initialization takes longer than the configured limit.
PASSION_INIT_WAIT_TIMEOUT_SECONDS: float = 30.0


def _ensure_initialized() -> None:
    """
    Lazy startup initialization. Thread-safe via _init_lock + Events.

    WARNING: This function is synchronous and performs blocking File-IO
    (loading subcategory datasets). Calling it from within an asynchronous
    event loop (e.g. async/await context) will block the loop, causing thread
    starvation and substantial latency spikes. It MUST NOT be called from
    an async event loop.
    """
    global _init_in_progress
    if _init_complete.is_set():
        return
    if _init_failed.is_set():
        raise RuntimeError("Passion engine startup checks previously failed. Check startup logs.")

    # Hold lock only to atomically check/set _init_in_progress.
    # All initialization work runs OUTSIDE the lock to prevent deadlock
    # (inner `with _init_lock:` calls in the try block need to re-acquire).
    should_run = False
    with _init_lock:
        if _init_complete.is_set():
            return
        if _init_failed.is_set():
            raise RuntimeError("Passion engine startup checks previously failed. Check startup logs.")
        if not _init_in_progress:
            _init_in_progress = True
            should_run = True

    if not should_run:
        # Fix 14: Another thread is initializing — wait on the event instead of busy-spinning.
        completed = _init_complete.wait(timeout=PASSION_INIT_WAIT_TIMEOUT_SECONDS)
        if not completed:
            raise TimeoutError(
                f"Passion engine initialization did not complete within "
                f"{PASSION_INIT_WAIT_TIMEOUT_SECONDS}s."
            )
        if _init_failed.is_set():
            raise RuntimeError("Passion engine startup checks previously failed. Check startup logs.")
        return

    try:
        skip = _os.environ.get("INSIGHT_ENGINE_SKIP_STARTUP_CHECKS", "").lower() == "true"
        if skip:
            _env = _os.environ.get("ENV", "development").strip().lower()
            if _env in ("production", "prod", "staging"):
                with _init_lock:
                    _init_failed.set()
                    _init_in_progress = False
                raise RuntimeError(
                    "INSIGHT_ENGINE_SKIP_STARTUP_CHECKS cannot be set in production/staging."
                )
            with _init_lock:
                _init_complete.set()
                _init_in_progress = False
            return
        else:
            retries = 3
            for attempt in range(1, retries + 1):
                try:
                    from bootstrap import run_startup_checks as _run_startup_checks
                    _run_startup_checks(env=_os.environ.get("ENV", "development").strip().lower())
                    with _init_lock:
                        _init_complete.set()
                        _init_in_progress = False
                    return
                except _PERMANENT_STARTUP_ERRORS:
                    with _init_lock:
                        _init_failed.set()
                        _init_in_progress = False
                    raise
                except OSError as e:
                    if attempt == retries:
                        with _init_lock:
                            _init_failed.set()
                            _init_in_progress = False
                        raise RuntimeError(f"Startup checks failed after {retries} retries: {e}") from e
                    # Retry backoff (outside lock — no deadlock risk here).
                    time.sleep(0.5 * (2 ** (attempt - 1)))
                except Exception:
                    with _init_lock:
                        _init_failed.set()
                        _init_in_progress = False
                    raise
    finally:
        # Fix 14: Guarantee _init_in_progress is cleared on ANY exit path
        # (exception, unexpected return, or KeyboardInterrupt).
        # Inner branches already clear it; this is the last-resort safety net.
        with _init_lock:
            _init_in_progress = False




# Blocker 3: Handle TimeoutError explicitly, restrict _FATAL_EXCEPTIONS.
_FATAL_EXCEPTIONS = (
    MemoryError,
    RecursionError,
    KeyboardInterrupt,
    SystemExit,
)

def _should_reraise(exc: Exception, strict_mode: bool) -> bool:
    """Return True if this exception must propagate regardless of strict_mode."""
    return strict_mode or isinstance(exc, _FATAL_EXCEPTIONS) or isinstance(exc, TimeoutError)


# FIX 23: Allowlist clarity
PASSION_OWNED_OUTPUT_COLUMNS = (Col.INFERRED_SUBCATEGORY, Col.SUBCATEGORY_CONFIDENCE)

def safe_assign_new_columns(original: pd.DataFrame, updated: pd.DataFrame, strict_mode: bool = False) -> pd.DataFrame:
    result = original.copy(deep=True)
    if not result.index.equals(updated.index):
        raise ValueError(
            f"safe_assign_new_columns: index mismatch. "
            f"original shape={original.shape}, updated shape={updated.shape}. "
            f"Adapter must preserve DataFrame index."
        )

    unexpected = set(updated.columns) - set(original.columns) - set(PASSION_OWNED_OUTPUT_COLUMNS)
    if unexpected:
        if strict_mode:
            raise ValueError(f"safe_assign_new_columns: unexpected columns: {list(unexpected)}")
        logger.warning("passion_unexpected_columns", extra={"columns": list(unexpected)})

    for col in PASSION_OWNED_OUTPUT_COLUMNS:
        if col in updated.columns:
            result[col] = updated[col]
    return result




def _looks_like_compact_yyyymmdd(v: Any) -> bool:
    if isinstance(v, (bool, np.bool_)) or _safe_isna(v):
        return False
    if isinstance(v, str):
        s = v.strip()
        return len(s) == 8 and s.isdigit()
    if isinstance(v, (Integral, np.integer)):
        return 10_000_000 <= int(v) <= 99_999_999
    # Fix 12: Handle float YYYYMMDD (e.g. 20230101.0 from CSV numeric parsing).
    # Only accept whole-number floats in the valid YYYYMMDD range.
    if isinstance(v, (float, np.floating)) and not math.isnan(v) and not math.isinf(v):
        if float(v).is_integer():
            i = int(v)
            return 10_000_000 <= i <= 99_999_999
    return False

# Blocker 7: Abs-magnitude _normalize_ts
def _normalize_ts(val: Any, allow_yyyymmdd: bool = False) -> Any:
    if isinstance(val, (bool, np.bool_)) or _safe_isna(val) or isinstance(val, datetime.time):
        return np.nan
    if isinstance(val, Real) and np.isinf(val):
        return np.nan

    if _looks_like_compact_yyyymmdd(val):
        if not allow_yyyymmdd:
            # FIX 9: return np.nan if compact YYYYMMDD but allow_yyyymmdd=False
            return np.nan
        try:
            s = str(int(val)) if not isinstance(val, str) else val.strip()
            ts = pd.Timestamp(s)
            if 1900 <= ts.year <= 2099:
                return int(ts.timestamp())
            return np.nan
        except (ValueError, TypeError, OverflowError, pd.errors.OutOfBoundsDatetime):
            return np.nan

    if isinstance(val, (Integral, Real)):
        try:
            val_int = int(val)
            abs_val = abs(val_int)
        except (ValueError, TypeError, OverflowError):
            return np.nan

        if abs_val >= 100_000_000_000_000_000:
            return val_int // 1_000_000_000
        elif abs_val >= 100_000_000_000_000:
            return val_int // 1_000_000
        elif abs_val >= 100_000_000_000:
            return val_int // 1000

        return val_int

    try:
        ts = pd.Timestamp(val)
        if ts is pd.NaT:
            return np.nan
        # C3: Always normalize to UTC before calling .timestamp().
        # pd.Timestamp("2023-01-01").timestamp() uses the LOCAL system timezone
        # for naive timestamps, producing different Unix epoch seconds on machines
        # in different timezones. This causes non-deterministic sort keys and
        # broken _is_non_declining bucketing on developer/CI systems.
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        return int(ts.timestamp())
    except (ValueError, TypeError, OverflowError, SyntaxError, pd.errors.OutOfBoundsDatetime):
        return np.nan


# FIX C7: Removed _to_datetime_safe as it was brittle with object-dtypes.
# Instead, _normalize_ts is universally applied to guarantee uniform integer timestamps.


_CURRENCY_RE_LOCAL = _re.compile(r'[$\€\£\¥\u20A0-\u20CF\u20B9]')
_INR_RE_LOCAL = _re.compile(r'^(rs\.?\s*|inr\s+)', _re.IGNORECASE)

def _is_unparseable_amount(val: Any) -> bool:
    try:
        parsed = safe_numeric(val, default=np.nan)
        return _safe_isna(parsed) or math.isinf(float(parsed)) or math.isnan(float(parsed))
    except (ValueError, TypeError, OverflowError):
        return True


from config_passion import PIPELINE_HARD_TIMEOUT_MS as _DEFAULT_HARD_TIMEOUT_MS

class _StepBudgetGuard:
    """
    Cooperative deadline guard. Checks budget only between stages.
    Cannot interrupt a hung pandas, regex, or model call.
    Not a true watchdog; if true hard timeout is required, use process isolation.
    """
    __slots__ = ('_budget_s', '_start', '_hard_deadline', '_step_log')

    def __init__(self, budget_ms: float, hard_timeout_ms: float = _DEFAULT_HARD_TIMEOUT_MS):
        self._budget_s = budget_ms / 1000.0
        self._start = time.monotonic()
        self._hard_deadline = self._start + hard_timeout_ms / 1000.0
        self._step_log: list[tuple[str, float]] = []

    def check(self, step_name: str) -> None:
        now = time.monotonic()
        elapsed = now - self._start
        last_t = self._step_log[-1][1] if self._step_log else 0.0
        stage_elapsed_ms = round((elapsed - last_t) * 1000, 2)
        self._step_log.append((step_name, round(elapsed, 3)))
        # E5: Per-checkpoint logs are DEBUG to avoid INFO flooding in production.
        # One INFO summary is emitted after successful pipeline completion (see process_pipeline).
        logger.debug("passion_stage_complete", extra={"stage": step_name, "elapsed_ms": stage_elapsed_ms})
        breakdown = ", ".join(f"{s}={t}s" for s, t in self._step_log)
        # FIX-23: Distinguish soft budget exceeded from hard deadline exceeded for better observability.
        if now > self._hard_deadline:
            raise TimeoutError(f"Hard deadline exceeded at '{step_name}': {breakdown}")
        if elapsed > self._budget_s:
            raise TimeoutError(f"Budget exceeded at '{step_name}': {breakdown}")

    # FIX-2: Add summary() to avoid direct private attribute access in process_pipeline.
    def summary(self) -> dict[str, Any]:
        now = time.monotonic()
        total_elapsed_ms = round((now - self._start) * 1000, 2)
        stage_breakdown = {s: round(t * 1000, 2) for s, t in self._step_log}
        return {
            "total_elapsed_ms": total_elapsed_ms,
            "stages": stage_breakdown,
        }




def process_pipeline(
    df_raw: pd.DataFrame,
    strict_mode: bool = True,
    rng: random.Random | None = None,
    allow_yyyymmdd_dates: bool = False,
) -> "PassionResult":
    """
    Process debits to generate passion insights.

    Behavior Notes:
    - strict_mode=True (integration default): any internal substage failure raises
      immediately. The caller (_attach_passion_results) catches and returns the
      unchanged core PipelineResult.
    - strict_mode=False (soft/standalone mode): the FIRST substage failure causes
      an immediate return of _neutral_passion_result (original rows + neutral columns).
      Subsequent substages are NOT executed — fail-fast, no wasted work. (D1)

    D3 — TimeoutError Propagation Contract:
      TimeoutError raised by _StepBudgetGuard ALWAYS propagates out of process_pipeline
      regardless of strict_mode. _should_reraise() returns True for TimeoutError
      unconditionally. The swallowing decision for TimeoutError lives ONLY in
      _attach_passion_results (governed by strict_attach, not strict_mode).
      Standalone callers (not via _attach_passion_results) will always see TimeoutError.
    """
    if not isinstance(df_raw, pd.DataFrame):
        raise TypeError(f"df_raw must be pd.DataFrame, got {type(df_raw).__name__}")

    _ensure_initialized()

    if df_raw.index.duplicated().any():
        raise ValueError("Input DataFrame has duplicate index values")
    if df_raw.columns.duplicated().any():
        raise ValueError("Duplicate column names in input DataFrame")

    assert_columns_exist(
        df_raw,
        [Col.AMOUNT, Col.CLEANED_REMARKS, Col.DATE, Col.PREDICTED_CATEGORY],
        "process_pipeline input",
    )

    # D5: PASSION_MIN_DEBIT_ROWS is imported at module top level (no try/except guard needed).
    # The old inline try/except ImportError guard was dead code — config_passion is a hard
    # dependency; if it fails to import, this module never loads.

    # FIX 7: Validation happens before neutral return unless DataFrame is completely empty.
    # D2: Renamed from _empty_passion_result → _neutral_passion_result.
    if df_raw.empty:
        return _neutral_passion_result(df_raw, reason="empty_input")

    # Fix 19: max_rows guard — prevents memory blowup on unexpectedly large inputs.
    # Controlled by INSIGHT_ENGINE_PASSION_MAX_ROWS env var (default: 100 000).
    max_rows = int(_os.environ.get("INSIGHT_ENGINE_PASSION_MAX_ROWS", "100000"))
    if len(df_raw) > max_rows:
        logger.warning(
            "process_pipeline exceeds_max_rows",
            extra={"row_count": len(df_raw), "max_rows": max_rows},
        )
        if strict_mode:
            raise ValueError(
                f"Input exceeds max rows: {len(df_raw)} > {max_rows}. "
                "Set INSIGHT_ENGINE_PASSION_MAX_ROWS to override."
            )
        return _neutral_passion_result(df_raw, reason="exceeds_max_rows")

    if len(df_raw) < PASSION_MIN_DEBIT_ROWS:
        return _neutral_passion_result(df_raw, reason="below_min_rows")


    if rng is None:
        # FIX-9: RNG seed 0 is intentional. Output must be deterministic per input.
        # Template rotation is not part of this phase.
        rng = random.Random(0)

    budget = _StepBudgetGuard(
        budget_ms=PIPELINE_BUDGET_MS,
        hard_timeout_ms=PIPELINE_HARD_TIMEOUT_MS,
    )

    # Blocker 4: Use work_df for parsing/detection, return output_df based on original df_raw.
    # Blocker 1: Do not sort work_df here, as it breaks assignment back to df_raw later.
    work_df = df_raw.copy(deep=True)

    # FIX 30: Assign _detection_dates directly (Col.DATE guaranteed by assert_columns_exist)
    if not allow_yyyymmdd_dates and work_df[Col.DATE].map(_looks_like_compact_yyyymmdd).any():
        raise ValueError(
            "Compact YYYYMMDD dates require allow_yyyymmdd_dates=True"
        )
    _detection_dates = work_df[Col.DATE].map(
        lambda v: _normalize_ts(v, allow_yyyymmdd=allow_yyyymmdd_dates)
    )
    _detection_dates = pd.to_datetime(_detection_dates, unit="s", errors="coerce")
    # work_df[Col.DATE] is left unchanged — original values are preserved in output.
    # FIX-20: Add budget checkpoint to isolate date-normalization time
    budget.check("date_normalization")

    # Amount processing (integrated mode logic: do not mutate amount, just get masks)
    if Col.AMOUNT in work_df.columns:
        bad_mask = work_df[Col.AMOUNT].apply(_is_unparseable_amount)
        valid_amount_mask = ~bad_mask
        amount_numeric = work_df[Col.AMOUNT].map(safe_numeric).astype(float)
    else:
        valid_amount_mask = pd.Series(True, index=work_df.index, dtype=bool)
        amount_numeric = pd.Series(0.0, index=work_df.index, dtype=float)
    # FIX-20: Add budget checkpoint to isolate amount-preparation time
    budget.check("amount_preparation")

    if Col.IS_KNOWN_PERSON in work_df.columns:
        known_mask = coerce_bool_column(work_df[Col.IS_KNOWN_PERSON].fillna(False))
    else:
        # FIX 8: Required in strict_mode (integration), fallback only in standalone/soft mode
        if strict_mode:
            raise ValueError(f"Missing required column: {Col.IS_KNOWN_PERSON}")
        known_mask = pd.Series(False, index=work_df.index, dtype=bool)

    spend_mask = ~known_mask & valid_amount_mask

    if Col.INFERRED_SUBCATEGORY not in work_df.columns:
        work_df[Col.INFERRED_SUBCATEGORY] = pd.NA

    if Col.SUBCATEGORY_CONFIDENCE not in work_df.columns:
        work_df[Col.SUBCATEGORY_CONFIDENCE] = 0.0

    # Enrich subcategories
    try:
        from marketplace_subcategory import enrich_subcategories
        # Pass a copy with clean numeric amounts for enrichment logic
        enrich_df = work_df.copy(deep=True)
        enrich_df[Col.AMOUNT] = amount_numeric
        _result = enrich_subcategories(enrich_df)
        work_df = safe_assign_new_columns(work_df, _result, strict_mode=strict_mode)

        # Blocker 2: Force ineligible rows after enrichment
        ineligible_mask = known_mask | ~valid_amount_mask
        if Col.INFERRED_SUBCATEGORY in work_df.columns:
            work_df.loc[ineligible_mask, Col.INFERRED_SUBCATEGORY] = pd.NA
        if Col.SUBCATEGORY_CONFIDENCE in work_df.columns:
            work_df.loc[ineligible_mask, Col.SUBCATEGORY_CONFIDENCE] = 0.0

    except Exception as e:
        if _should_reraise(e, strict_mode):
            raise
        logger.warning("subcategory_enrichment_failed", extra={"error_type": type(e).__name__, "stage": "subcategory_enrichment"})
        # D1: Fail-fast in soft mode — return immediately, do not run downstream stages.
        return _neutral_passion_result(df_raw, reason="substage_failure")

    budget.check("subcategory_enrichment")

    try:
        from marketplace_subcategory import resolve_merchant_vectorized
        from passion_detector import detect_passions

        resolved_merchants = resolve_merchant_vectorized(
            work_df.loc[spend_mask, Col.CLEANED_REMARKS]
        )

        # detect_passions needs clean numeric amounts.
        detect_df = work_df.copy(deep=True)
        detect_df[Col.AMOUNT] = amount_numeric
        # P3-7: Use pre-normalized dates for detection, avoiding duplicate conversion.
        detect_df[Col.DATE] = _detection_dates
        detect_df = detect_df.sort_index()
        # Sort the masks to match detect_df
        sorted_spend_mask = spend_mask.loc[detect_df.index]
        # P0-1 FIX: Reindex resolved_merchants to match detect_df's full index.
        sorted_resolved = resolved_merchants.reindex(detect_df.index)

        passion_signals = detect_passions(detect_df, sorted_spend_mask, sorted_resolved)
    except Exception as e:
        if _should_reraise(e, strict_mode):
            raise
        logger.warning("passion_detection_failed", extra={"error_type": type(e).__name__, "stage": "passion_detection"})
        # D1: Fail-fast in soft mode — return immediately, do not run downstream stages.
        return _neutral_passion_result(df_raw, reason="substage_failure")

    budget.check("passion_detection")

    # Wire PassionSignals to insight generation
    def _signal_to_candidate(sig) -> Candidate:
        return Candidate.passion(
            signal=sig,
            sort_key_ts=getattr(sig, "latest_ts", 0),
            normalized_score=sig.spend_share,
            original_index=getattr(sig, "original_index", 0),
        )

    signal_candidates = [
        _signal_to_candidate(sig)
        for sig in passion_signals
        if not getattr(sig, "is_suppressed", False)
    ]

    all_candidates = signal_candidates
    all_candidates.sort()

    # Generate insights
    try:
        from passion_insight_generator import generate_passion_insights
        insights = generate_passion_insights(
            all_candidates, top_n=PIPELINE_TOP_N, strict_mode=strict_mode, rng=rng,
        )
    except Exception as e:
        if _should_reraise(e, strict_mode):
            raise
        logger.warning("insight_generation_failed", extra={"error_type": type(e).__name__, "stage": "insight_generation"})
        # D1: Fail-fast in soft mode — return immediately, do not run downstream stages.
        return _neutral_passion_result(df_raw, reason="substage_failure")

    budget.check("insight_generation")

    # D1: pipeline_failed flag removed — soft mode now returns immediately at the
    # failing substage. This block is only reached when all substages succeeded.

    # Blocker 4: Return output_df based on original df_raw plus only passion-owned columns.
    output_df = safe_assign_new_columns(df_raw, work_df, strict_mode=strict_mode)

    # P3-4: Filter suppressed signals — only active signals reach PipelineResult.
    active_signals = tuple(s for s in passion_signals if not s.is_suppressed)
    suppressed_count = len(passion_signals) - len(active_signals)
    if suppressed_count > 0:
        logger.info(
            "passion_signals_suppressed",
            extra={"suppressed_count": suppressed_count},
        )
        # Standard logging so pytest caplog can capture suppressed_count attribute.
        _std_logger.info(
            "passion_pipeline_complete",
            extra={"suppressed_count": suppressed_count},
        )

    # E5: Emit one INFO-level timing summary at successful pipeline completion.
    # Per-stage detail is already logged at DEBUG level by _StepBudgetGuard.check().
    # FIX-2: Access timing metrics cleanly via budget.summary().
    budget_summary = budget.summary()
    logger.info(
        "passion_stage_timings",
        extra={
            "total_elapsed_ms": budget_summary["total_elapsed_ms"],
            "stages": budget_summary["stages"],
            "signal_count": len(active_signals),
            "insight_count": len(insights),
        },
    )

    return PassionResult(
        debits=output_df,
        candidates=tuple(all_candidates),
        insights=tuple(insights),
        passion_signals=active_signals,
        passion_debits=detect_df.copy(deep=True),
    )
