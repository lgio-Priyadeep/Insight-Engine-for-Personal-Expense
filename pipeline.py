"""
pipeline.py — Central Orchestrator
====================================
Defines the complete Insight Engine pipeline as a single, importable
entry point. All module wiring is centralised here — no test file
should need to know the order of operations.

Usage:
    from pipeline import run_pipeline, PipelineResult

    result = run_pipeline(raw_df)
    for insight in result.insights:
        print(insight)
"""

import os
import hashlib
import json
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from sklearn.pipeline import Pipeline as SklearnPipeline

from logger_factory import get_logger, generate_new_run_id, pipeline_run_id_ctx
from model_state import InsightModelState
import config
from schema import Col
from preprocessor import preprocess
from feature_engineer import engineer_features, engineer_features_inference
from seed_labeler import label_debits, label_credits
from categorization_model import train_categorization_model, predict_categories
from expected_spend_model import train_expected_spend_model, predict_expected_spend
from anomaly_detector import detect_anomalies
from recurring_detector import find_recurring_transactions
from insight_model import load_insight_ranker, predict_insight_scores
from insight_generator import generate_human_insights
from known_persons import (
    tag_known_persons, 
    _enforce_known_person_schema, 
    log_unmatched_recurring_transfers, 
    detect_personal_patterns
)

logger = get_logger(__name__)
# FIX #1: passion_logger MUST be at module scope before _attach_passion_results.
# Do NOT shadow the existing pipeline-level logger — use a distinct name.
from logger_factory import get_logger
passion_logger = get_logger("passion.engine")
# Standard Python logger for pytest caplog interception (caplog cannot see
# the custom JSON-to-stderr emitter used by passion_logger / get_logger).
# IMPORTANT: must NOT be under __name__ namespace — get_logger(__name__) sets
# propagate=False on that logger (and all children), making it invisible to caplog.
# 'insight_engine.caplog_bridge' is independent: its parent is root (propagate=True).
import logging as _std_logging
_std_logger = _std_logging.getLogger("insight_engine.caplog_bridge")


def _compute_config_hash(kp: dict, sa: dict) -> str:
    """Deterministic fingerprint of the known-persons config.
    
    Used to detect config drift between training and inference.
    Sorted serialization ensures dict ordering doesn't matter.
    """
    payload = json.dumps({"kp": kp, "sa": sa}, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]

def _validate_state_version(
    state: InsightModelState,
    known_persons: dict = None,
    self_accounts: dict = None,
) -> None:
    """Must be the FIRST step in run_inference(). No exceptions."""
    kp = known_persons if known_persons is not None else config.KNOWN_PERSONS
    sa = self_accounts if self_accounts is not None else config.SELF_ACCOUNTS
    
    has_known_persons = bool(kp) or bool(sa)
    if has_known_persons and state.stats_version == "v1_raw":
        raise ValueError(
            "InsightModelState was trained without known-person exclusion "
            "(stats_version='v1_raw'), but KNOWN_PERSONS/SELF_ACCOUNTS is "
            "now configured. Run full run_pipeline() first to retrain "
            "models with filtered stats."
        )
    
    current_hash = _compute_config_hash(kp, sa)
    if has_known_persons and state.kp_config_hash != current_hash:
        raise ValueError(
            f"KNOWN_PERSONS/SELF_ACCOUNTS config has changed since "
            f"last training (stored hash: '{state.kp_config_hash}', "
            f"current hash: '{current_hash}'). Run full run_pipeline() "
            f"to retrain with updated exclusion list."
        )


@dataclass(frozen=True, kw_only=True)
class PipelineResult:
    """
    D4 — passion_debits vs debits Contract
    ──────────────────────────────────────
    result.debits:
        Core pipeline output. Contains all ML-enriched columns produced by
        run_pipeline / run_inference (predicted_category, insight_score,
        is_anomaly, is_recurring, etc.). This field is NEVER mutated by the
        passion sidecar. Downstream consumers that only need core ML output
        should read result.debits.

    result.passion_debits:
        passion_debits is produced by running the passion sidecar against
        result.debits and returning a defensive-copy DataFrame with the
        same rows plus passion-owned columns added:
          - Col.INFERRED_SUBCATEGORY ("inferred_subcategory")
          - Col.SUBCATEGORY_CONFIDENCE ("subcategory_confidence")
        Populated only when INSIGHT_ENGINE_PASSION_ENABLED=true and the
        passion engine runs successfully. Defaults to empty DataFrame.
        Downstream consumers needing subcategory enrichment MUST read
        passion_debits, not debits.

    DESIGN NOTE: If product requirements change to expect enriched subcategory
    data in result.debits (not passion_debits), that is a deliberate design
    change requiring a separate migration — it must NOT be done by mutating
    result.debits inside _attach_passion_results.
    """
    debits: pd.DataFrame
    credits: pd.DataFrame
    insights: List[str] = field(default_factory=list)
    cat_pipeline: Optional[SklearnPipeline] = None
    spend_pipeline: Optional[SklearnPipeline] = None
    ranker_pipeline: Optional[SklearnPipeline] = None
    global_mean: float = 0.0
    global_std: float = 0.0
    raw_global_mean: float = 0.0
    raw_global_std: float = 0.0
    stats_version: str = "v1_raw"
    personal_summary: dict = field(default_factory=dict)
    stats: dict = field(default_factory=dict)
    passion_debits: pd.DataFrame = field(default_factory=pd.DataFrame)
    passion_insights: tuple = field(default=())
    passion_signals: tuple = field(default=())
    transfer_patterns: list = field(default_factory=list)
    exclusion_stats: dict = field(default_factory=dict)
    kp_config_hash: str = ""
    personal_debits: pd.DataFrame = field(default_factory=pd.DataFrame)
    personal_credits: pd.DataFrame = field(default_factory=pd.DataFrame)

    def __post_init__(self) -> None:
        if hasattr(self, "passion_debits") and isinstance(self.passion_debits, pd.DataFrame):
            object.__setattr__(self, "passion_debits", self.passion_debits.copy(deep=True))

        if hasattr(self, "stats") and isinstance(self.stats, dict):
            # FIX-5: Validate that all stats dictionary keys and values are scalars to prevent deep nested mutation.
            import numpy as np
            _allowed_scalar_types = (str, int, float, bool, bytes, type(None), np.generic)
            for k, v in self.stats.items():
                if not isinstance(k, _allowed_scalar_types):
                    raise TypeError(f"stats key must be scalar, got {type(k).__name__}")
                if not isinstance(v, _allowed_scalar_types):
                    raise TypeError(f"stats value for key '{k}' must be scalar, got {type(v).__name__}")
            object.__setattr__(self, "stats", dict(self.stats))

        # P2-1: Passion field normalization and type validation
        object.__setattr__(self, "passion_insights",
            tuple(self.passion_insights) if self.passion_insights is not None else ()
        )
        object.__setattr__(self, "passion_signals",
            tuple(self.passion_signals) if self.passion_signals is not None else ()
        )

        # B1: Do NOT import passion_models unconditionally here.
        # An unconditional 'from passion_models import PassionSignal' makes the
        # optional Passion sidecar structurally mandatory for every core pipeline
        # import. Use duck-typing instead: check for required PassionSignal
        # attributes only when passion_signals is non-empty.
        for s in self.passion_signals:
            if not (
                hasattr(s, 'category') and
                hasattr(s, 'spend_share') and
                hasattr(s, 'is_suppressed') and
                hasattr(s, 'merchant_list')
            ):
                raise TypeError(
                    f"passion_signals elements must be PassionSignal-like "
                    f"(have category, spend_share, is_suppressed, merchant_list), "
                    f"got {type(s)}"
                )

        for t in self.passion_insights:
            if not isinstance(t, str):
                raise TypeError(f"passion_insights must contain str, got {type(t)}")

    def replace(self, **kwargs) -> "PipelineResult":
        import dataclasses
        return dataclasses.replace(self, **kwargs)


def finalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure explicitly required float constraints map consistently post analysis"""
    df = df.copy()
    if Col.RECURRING_SCORE in df.columns:
        df[Col.RECURRING_SCORE] = df[Col.RECURRING_SCORE].fillna(0.0)
    return df

def _optimize_memory_footprint(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast flag variables safely AFTER the ML pipeline finishes predicting."""
    df = df.copy()
    if Col.PREDICTED_CATEGORY in df.columns:
        df[Col.PREDICTED_CATEGORY] = df[Col.PREDICTED_CATEGORY].astype('category')
    if Col.IS_WEEKEND in df.columns:
        df[Col.IS_WEEKEND] = df[Col.IS_WEEKEND].astype(bool)
    return df


def train_models(
    debits: pd.DataFrame, 
    label_col: str, 
    target_col: str,
    stats_version: str,
    kp_config_hash: str,
) -> InsightModelState:
    cat_pipeline = train_categorization_model(debits, label_col=label_col)
    debits = predict_categories(cat_pipeline, debits)
    spend_pipeline = train_expected_spend_model(debits, target_col=target_col)
    ranker_pipeline = load_insight_ranker()
    return InsightModelState(
        pipeline_version="1.0.0",
        cat_pipeline=cat_pipeline,
        spend_pipeline=spend_pipeline,
        ranker_pipeline=ranker_pipeline,
        global_mean=debits[Col.SIGNED_AMOUNT].mean(),
        global_std=debits[Col.SIGNED_AMOUNT].std(),
        stats_version=stats_version,
        kp_config_hash=kp_config_hash,
    )

def _pre_initialize_ml_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Initialize ML columns BEFORE mask-gated `.loc` writes."""
    df = df.copy()
    for col in Col.ml_output_columns():
        if col not in df.columns:
            df[col] = pd.NA
    return df

# ── Passion Engine Integration Helpers ────────────────────────────────
# FIX-01: Shared helper for both run_pipeline and run_inference.
# FIX-33: Checks PASSION_ENGINE_ENABLED kill switch at runtime.
# FIX C9/C10: Lazy-import patching to prevent circular dependencies.

def _write_crash_dumps(
    *,
    debits,
    credits,
    crash_dump_dir,
    passion_debits=None,
    passion_insights=(),
    passion_signals=(),
    run_id=None,
) -> bool:
    """Safely write crash dump files containing debits, credits, and passion data."""
    import os
    import json
    import numpy as np
    import pandas as pd
    import config
    from logger_factory import get_logger, pipeline_run_id_ctx
    from schema import Col

    logger = get_logger(__name__)

    if not run_id:
        run_id = pipeline_run_id_ctx.get() or "unknown"

    os.makedirs(crash_dump_dir, exist_ok=True)
    wrote_any = False

    if debits is not None and not isinstance(debits, pd.DataFrame):
        logger.warning(
            "Unexpected type for debits: %s",
            type(debits).__name__,
            extra={"event_type": "data_corruption", "stage": "crash_handler"}
        )
    safe_debits = debits.head(1000) if isinstance(debits, pd.DataFrame) else pd.DataFrame()

    if credits is not None and not isinstance(credits, pd.DataFrame):
        logger.warning(
            "Unexpected type for credits: %s",
            type(credits).__name__,
            extra={"event_type": "data_corruption", "stage": "crash_handler"}
        )
    safe_credits = credits.head(1000) if isinstance(credits, pd.DataFrame) else pd.DataFrame()

    # Safely handle new passion fields
    safe_passion_debits = pd.DataFrame()
    passion_summary = {}

    if passion_debits is not None and isinstance(passion_debits, pd.DataFrame):
        safe_passion_debits = passion_debits.head(1000)

    if passion_insights is not None:
        _raw_insights = list(passion_insights)[:100]
        if getattr(config, "ENABLE_PII_DEBUG_LOGS", False):
            passion_summary["insights"] = _raw_insights
        else:
            from log_utils import log_safe_text
            passion_summary["insights"] = [log_safe_text(str(x)) for x in _raw_insights]

    if passion_signals is not None:
        signals_serial = []
        from log_utils import log_safe_merchant
        for sig in list(passion_signals)[:100]:
            masked_merchants = [
                log_safe_merchant(m) if not getattr(config, "ENABLE_PII_DEBUG_LOGS", False) else m
                for m in sig.merchant_list
            ]
            signals_serial.append({
                "category": sig.category,
                "total_spend": sig.total_spend,
                "merchant_count": sig.merchant_count,
                "spend_share": sig.spend_share,
                "trend_direction": sig.trend_direction,
                "is_suppressed": sig.is_suppressed,
                "suppression_reason": sig.suppression_reason,
                "latest_ts": sig.latest_ts,
                "merchant_list": masked_merchants,
            })
        passion_summary["signals"] = signals_serial

    # Atomicity Note: Guaranteed on POSIX systems; best-effort on Windows.
    if not safe_debits.empty:
        wrote_any = True
        tmp_path = os.path.join(crash_dump_dir, f"{run_id}_debits.csv.tmp")
        final_path = os.path.join(crash_dump_dir, f"{run_id}_debits.csv")
        safe_debits.to_csv(tmp_path, index=False)
        os.replace(tmp_path, final_path)

    if not safe_credits.empty:
        wrote_any = True
        tmp_path = os.path.join(crash_dump_dir, f"{run_id}_credits.csv.tmp")
        final_path = os.path.join(crash_dump_dir, f"{run_id}_credits.csv")
        safe_credits.to_csv(tmp_path, index=False)
        os.replace(tmp_path, final_path)

    if not safe_passion_debits.empty:
        wrote_any = True
        tmp_path = os.path.join(crash_dump_dir, f"{run_id}_passion_debits.csv.tmp")
        final_path = os.path.join(crash_dump_dir, f"{run_id}_passion_debits.csv")
        from log_utils import log_safe_merchant
        pii_safe_passion_debits = safe_passion_debits.copy()
        if Col.CLEANED_REMARKS in pii_safe_passion_debits.columns and not getattr(config, "ENABLE_PII_DEBUG_LOGS", False):
            pii_safe_passion_debits[Col.CLEANED_REMARKS] = pii_safe_passion_debits[Col.CLEANED_REMARKS].map(log_safe_merchant)
        pii_safe_passion_debits.to_csv(tmp_path, index=False)
        os.replace(tmp_path, final_path)

    if passion_summary:
        wrote_any = True
        tmp_path = os.path.join(crash_dump_dir, f"{run_id}_passion_summary.json.tmp")
        final_path = os.path.join(crash_dump_dir, f"{run_id}_passion_summary.json")

        def _json_safe(obj):
            import math
            import pandas as _pd2
            if isinstance(obj, np.generic):
                return obj.item()
            if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
                return None
            try:
                if _pd2.isna(obj):
                    return None
            except (TypeError, ValueError):
                pass
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return str(obj)

        with open(tmp_path, "w") as f:
            json.dump(passion_summary, f, indent=2, default=_json_safe)
        os.replace(tmp_path, final_path)

    return wrote_any


def _resolve_passion_crash_fields(result=None, locals_snapshot=None):
    """
    Resolve passion fields for crash dumps without relying on inline locals()
    expressions in the exception handler.
    """
    snap = locals_snapshot or {}

    if "passion_debits" in snap:
        passion_debits = snap.get("passion_debits")
    else:
        passion_debits = getattr(result, "passion_debits", None) if result is not None else None

    if "passion_insights" in snap:
        passion_insights = snap.get("passion_insights") or ()
    else:
        passion_insights = getattr(result, "passion_insights", ()) if result is not None else ()

    if "passion_signals" in snap:
        passion_signals = snap.get("passion_signals") or ()
    else:
        passion_signals = getattr(result, "passion_signals", ()) if result is not None else ()

    return passion_debits, passion_insights, passion_signals


def _attach_passion_results(
    result: "PipelineResult",
    process_fn=None,
    replace_fn=None,
    fields_fn=None,
    rng=None,
    strict_attach: bool = False,
) -> "PipelineResult":
    """
    Attempt to run the passion pipeline and attach results to PipelineResult.

    P2-3: MemoryError and RecursionError always propagate.
    TimeoutError cancels the entire Passion sidecar.
    Core PipelineResult is returned unchanged on any other failure.
    strict_attach=True (set via env INSIGHT_ENGINE_PASSION_STRICT_ATTACH=true in CI)
    makes all exceptions propagate.

    Args:
        result: The existing PipelineResult from the core pipeline.
        debits: The debits DataFrame to analyze (will be deep-copied internally).
        strict_attach: If True, all non-fatal exceptions propagate. Default False.

    Returns:
        A new PipelineResult with passion fields populated, or the original
        result unchanged if the passion engine fails or is disabled.
    """
    def _with_passion_status(res: "PipelineResult", status: str) -> "PipelineResult":
        import dataclasses as _dc
        r_fn = replace_fn or _dc.replace
        current_stats = getattr(res, "stats", {})
        if not isinstance(current_stats, dict):
            current_stats = {}
        new_stats = {**current_stats, "passion_status": status}
        return r_fn(res, stats=new_stats)

    # FIX-33: Runtime kill switch — check on every call so env var changes
    # take effect without restart. Default is now false!
    import os as _os
    enabled = _os.environ.get(
        "INSIGHT_ENGINE_PASSION_ENABLED", "false"
    ).lower() == "true"

    if not enabled:
        passion_logger.info("passion_engine_config", extra={"enabled": "false", "reason": "INSIGHT_ENGINE_PASSION_ENABLED=false"})
        return _with_passion_status(result, "disabled")

    if not strict_attach:
        strict_attach = _os.environ.get(
            "INSIGHT_ENGINE_PASSION_STRICT_ATTACH", "false"
        ).lower() == "true"

    try:
        import dataclasses as _dc
        import random
        import pandas as pd
        from schema import Col

        if process_fn is None:
            from passion_pipeline import process_pipeline as process_fn

        replace_fn = replace_fn or _dc.replace
        fields_fn = fields_fn or _dc.fields
        rng = rng or random.Random(0)

        # FIX 15: Extract debits from result to make contract robust to misuse
        debits = result.debits if hasattr(result, "debits") else pd.DataFrame()

        # P2-3: Preflight checks before calling process_fn
        if not isinstance(debits, pd.DataFrame):
            passion_logger.warning("passion_skip", extra={"reason": "non_dataframe"})
            return _with_passion_status(result, "skipped")
        # FIX-21: Configurable max rows limit to prevent memory blowup (default 100,000 rows)
        max_rows = int(_os.environ.get("INSIGHT_ENGINE_PASSION_MAX_ROWS", "100000"))
        if len(debits) > max_rows:
            passion_logger.warning("passion_skip", extra={"reason": "exceeds_max_rows", "row_count": len(debits), "max_rows": max_rows})
            _std_logger.warning("passion_skip exceeds_max_rows", extra={"row_count": len(debits), "max_rows": max_rows})
            return _with_passion_status(result, "skipped")
        if debits.empty:
            passion_logger.info("passion_skip", extra={"reason": "empty_debits"})
            return _with_passion_status(result, "skipped")
        if debits.index.duplicated().any():
            passion_logger.warning("passion_skip", extra={"reason": "duplicate_index"})
            return _with_passion_status(result, "skipped")
        # FIX 8: IS_KNOWN_PERSON is required
        required = {Col.DATE, Col.AMOUNT, Col.PREDICTED_CATEGORY, Col.CLEANED_REMARKS, Col.IS_KNOWN_PERSON}
        missing = required - set(debits.columns)
        if missing:
            passion_logger.warning("passion_skip", extra={"reason": "missing_columns", "missing": sorted(missing)})
            return _with_passion_status(result, "skipped")
        if debits.columns.duplicated().any():
            passion_logger.warning("passion_skip", extra={"reason": "duplicate_columns"})
            return _with_passion_status(result, "skipped")

        # FIX C3 / L3: Preflight check — verify PipelineResult has the passion fields and stats
        required_fields = {
            "passion_debits", "passion_insights", "passion_signals", "stats"
        }
        actual_fields = {f.name for f in fields_fn(result)}
        if missing := required_fields - actual_fields:
            passion_logger.warning("passion_fields_missing", extra={"missing_fields": list(missing)})
            # Fix #4a: Return with passion_status="missing_fields" — not bare result.
            # A bare return loses the status audit trail needed for kill-switch monitoring.
            return _with_passion_status(result, "missing_fields")

        # A2 / L2: pass result and rng
        passion_result = process_fn(
            df_raw=debits,
            strict_mode=True,
            rng=rng,
        )

        # Blocker 14: Improve structured logs with required metrics
        passion_logger.info("passion_engine_success", extra={
            "outcome": "success",
            "row_count": len(debits),
            "candidate_count": len(passion_result.candidates),
            "insight_count": len(passion_result.insights),
            "signal_count": len(passion_result.passion_signals),
        })

        # Fix #4b: Single replace_fn call — _with_passion_status already calls replace_fn
        # internally, so wrapping its result in a second replace_fn is a double-replace.
        # Merge stats directly and replace all passion fields in one atomic call.
        new_stats = {**getattr(result, "stats", {}), "passion_status": "success"}
        return replace_fn(
            result,
            stats=new_stats,
            passion_debits=passion_result.debits,
            passion_insights=tuple(passion_result.insights),
            passion_signals=tuple(passion_result.passion_signals),
        )

    except MemoryError:
        raise
    except RecursionError:
        raise
    except SystemExit:
        raise
    except KeyboardInterrupt:
        raise
    except TimeoutError as e:
        passion_logger.warning("passion_engine_timeout", extra={"error_type": "TimeoutError", "stage": "attach_results", "safe_message": "Hard timeout or budget exceeded"})
        _std_logger.warning("passion_engine_timeout", extra={"stage": "attach_results"})
        if strict_attach:
            raise
        return _with_passion_status(result, "timeout")
    except Exception as e:
        passion_logger.warning(
            "passion_engine_failed",
            extra={
                "error_type": type(e).__name__,
                "stage": "attach_results",
                "row_count": len(debits) if 'debits' in locals() and isinstance(debits, pd.DataFrame) else 0,
                "column_count": len(debits.columns) if 'debits' in locals() and isinstance(debits, pd.DataFrame) else 0,
            },
            exc_info=True,
        )
        _std_logger.error("passion_engine_failed", extra={"error_type": type(e).__name__})
        if strict_attach:
            raise
        return _with_passion_status(result, "failure")


def run_pipeline(
    raw_df: pd.DataFrame,
    zscore_threshold: float = 3.0,
    pct_dev_threshold: float = 0.5,
    label_col: str = Col.PSEUDO_LABEL,
    target_col: str = Col.AMOUNT,
    state: Optional[InsightModelState] = None,
    known_persons: dict = None,
    self_accounts: dict = None,
) -> PipelineResult:
    debits: pd.DataFrame | None = None
    credits: pd.DataFrame | None = None
    
    run_id = generate_new_run_id()
    token = pipeline_run_id_ctx.set(run_id)
    
    try:
        
        logger.info("=" * 60)
        logger.info("  INSIGHT ENGINE — Pipeline Start", extra={"event_type": "pipeline_start"})
        logger.info("=" * 60)
        logger.info(
            "Pipeline mode initialized", 
            extra={"event_type": "pipeline_mode", "metrics": {"mode": "training+inference" if state is None else "inference-only"}}
        )

        kp = known_persons if known_persons is not None else config.KNOWN_PERSONS
        sa = self_accounts if self_accounts is not None else config.SELF_ACCOUNTS

        if state is not None:
            _validate_state_version(state, known_persons=kp, self_accounts=sa)

        # ── PHASE 1: Preprocessing ────────────────────────────────────────────
        logger.info("[Phase 1] Preprocessing...", extra={"event_type": "phase_start", "metrics": {"phase": 1}})
        debits, credits = preprocess(raw_df)

        # ── PHASE 1.5: Known Persons Tagging ──────────────────────────────────
        debits = tag_known_persons(debits, known_persons=kp, self_accounts=sa)
        credits = tag_known_persons(credits, known_persons=kp, self_accounts=sa)
        
        log_unmatched_recurring_transfers(debits)
        log_unmatched_recurring_transfers(credits)
        
        debits = _pre_initialize_ml_columns(debits)
        credits = _pre_initialize_ml_columns(credits)
        
        spend_mask = ~debits[Col.IS_KNOWN_PERSON]
        credit_spend_mask = ~credits[Col.IS_KNOWN_PERSON]

        has_known_persons = bool(kp) or bool(sa)
        stats_version = "v2_filtered" if has_known_persons else "v1_raw"
        current_hash = _compute_config_hash(kp, sa)

        raw_global_mean = debits[Col.SIGNED_AMOUNT].mean()
        raw_global_std = debits[Col.SIGNED_AMOUNT].std()

        # ── PHASE 2: Seed Labeling ────────────────────────────────────────────
        logger.info("[Phase 2] Seed labeling...", extra={"event_type": "phase_start", "metrics": {"phase": 2}})
        if spend_mask.any():
            spend_debits = debits.loc[spend_mask].copy()
            spend_debits = label_debits(spend_debits, label_col=label_col)
            debits.loc[spend_mask, spend_debits.columns] = spend_debits
            
        if credit_spend_mask.any():
            spend_credits = credits.loc[credit_spend_mask].copy()
            spend_credits = label_credits(spend_credits, label_col=label_col)
            credits.loc[credit_spend_mask, spend_credits.columns] = spend_credits
            
        debits = _enforce_known_person_schema(debits)
        credits = _enforce_known_person_schema(credits)

        # ── PHASE 3: Feature Engineering ──────────────────────────────────────
        logger.info("[Phase 3] Feature engineering...", extra={"event_type": "phase_start", "metrics": {"phase": 3}})
        filtered_global_mean = debits.loc[spend_mask, Col.SIGNED_AMOUNT].mean() if spend_mask.any() else raw_global_mean
        filtered_global_std = debits.loc[spend_mask, Col.SIGNED_AMOUNT].std() if spend_mask.any() else raw_global_std

        if spend_mask.any():
            spend_debits = debits.loc[spend_mask].copy()
            spend_debits = engineer_features(
                spend_debits,
                global_mean=filtered_global_mean,
                global_std=filtered_global_std,
                amount_col=target_col,
            )
            debits.loc[spend_mask, spend_debits.columns] = spend_debits
        debits = _enforce_known_person_schema(debits)

        # ── PHASE 4: ML Models ────────────────────────────────────────────────
        logger.info("[Phase 4] Machine Learning Models...", extra={"event_type": "phase_start", "metrics": {"phase": 4}})
        if state is None:
            logger.info(f"Training models explicitly on {spend_mask.sum()} spend rows.", extra={"event_type": "training_triggered", "metrics": {"row_count": int(spend_mask.sum())}})
            if spend_mask.any():
                new_state = train_models(
                    debits.loc[spend_mask].copy(), 
                    label_col, 
                    target_col,
                    stats_version=stats_version,
                    kp_config_hash=current_hash
                )
                cat_pipeline, spend_pipeline, ranker_pipeline = new_state.cat_pipeline, new_state.spend_pipeline, new_state.ranker_pipeline
            else:
                raise ValueError("No spend rows available for training.")
        else:
            cat_pipeline, spend_pipeline, ranker_pipeline = state.cat_pipeline, state.spend_pipeline, state.ranker_pipeline

        if spend_mask.any():
            spend_debits = debits.loc[spend_mask].copy()
            spend_debits = predict_categories(cat_pipeline, spend_debits)
            spend_debits = predict_expected_spend(spend_pipeline, spend_debits)
            debits.loc[spend_mask, spend_debits.columns] = spend_debits
        debits = _enforce_known_person_schema(debits)

        # ── PHASE 5: Signal Detection ─────────────────────────────────────────
        logger.info("[Phase 5] Signal detection...", extra={"event_type": "phase_start", "metrics": {"phase": 5}})
        if spend_mask.any():
            spend_debits = debits.loc[spend_mask].copy()
            spend_debits = detect_anomalies(
                spend_debits,
                zscore_threshold=zscore_threshold,
                pct_dev_threshold=pct_dev_threshold,
            )
            spend_debits = find_recurring_transactions(spend_debits, group_col=Col.CLEANED_REMARKS)
            debits.loc[spend_mask, spend_debits.columns] = spend_debits
        debits = _enforce_known_person_schema(debits)

        # ── PHASE 5.5: ML Insight Ranking ─────────────────────────────────────
        logger.info("[Phase 5.5] Ranking candidate insights...", extra={"event_type": "phase_start", "metrics": {"phase": "5.5"}})
        if spend_mask.any():
            spend_debits = debits.loc[spend_mask].copy()
            spend_debits = predict_insight_scores(ranker_pipeline, spend_debits)
            debits.loc[spend_mask, spend_debits.columns] = spend_debits
        debits = _enforce_known_person_schema(debits)

        # ── PHASE 6: Insight Generation ───────────────────────────────────────
        logger.info("[Phase 6] Generating insights...", extra={"event_type": "phase_start", "metrics": {"phase": 6}})
        debits = finalize_df(debits)
        credits = finalize_df(credits)
        
        debits = _optimize_memory_footprint(debits)
        credits = _optimize_memory_footprint(credits)
        
        insights = []
        if spend_mask.any():
            insights = generate_human_insights(debits.loc[spend_mask])

        # Personal Patterns
        personal_mask = debits[Col.IS_KNOWN_PERSON].fillna(False)
        personal_insights, personal_summary = detect_personal_patterns(debits.loc[personal_mask])
        insights.extend(personal_insights)
        
        # Calculate exclusion stats automatically to provide transparency
        total_debits = len(debits)
        excluded_debits = int(personal_mask.sum())
        exclusion_stats = {
            "total_transactions": total_debits,
            "excluded_transactions": excluded_debits,
            "exclusion_rate": float(excluded_debits / total_debits) if total_debits > 0 else 0.0,
            "raw_global_mean": raw_global_mean,
            "filtered_global_mean": filtered_global_mean,
            "mean_distortion_pct": float(abs((raw_global_mean - filtered_global_mean) / filtered_global_mean)) if filtered_global_mean > 0 else 0.0
        }
        
        if has_known_persons and total_debits > 0 and excluded_debits == 0:
            logger.info(
                "Configuration provided but no matches found. "
                "v2_filtered stats are identical to v1_raw. This is expected if the "
                "dataset contains no transfers to configured persons.",
                extra={"event_type": "known_persons_zero_matches"}
            )

        logger.info("=" * 60)
        logger.info(f"  Pipeline complete. Generated {len(insights)} insights.", extra={"event_type": "pipeline_complete", "metrics": {"insights_count": len(insights)}})
        logger.info("=" * 60)

        result = PipelineResult(
            debits=debits,
            credits=credits,
            insights=insights,
            cat_pipeline=cat_pipeline,
            spend_pipeline=spend_pipeline,
            ranker_pipeline=ranker_pipeline,
            global_mean=filtered_global_mean,
            global_std=filtered_global_std,
            raw_global_mean=raw_global_mean,
            raw_global_std=raw_global_std,
            stats_version=stats_version,
            personal_summary=personal_summary,
            transfer_patterns=personal_insights,
            exclusion_stats=exclusion_stats,
            kp_config_hash=current_hash,
            personal_debits=debits.loc[personal_mask].copy(),
            personal_credits=credits.loc[credits[Col.IS_KNOWN_PERSON].fillna(False)].copy()
        )

        # Phase 7: Passion Engine (optional — errors are swallowed)
        result = _attach_passion_results(result)

        import os as _os
        if _os.environ.get("INSIGHT_ENGINE_CRASH_TEST", "false").lower() == "true":
            raise ValueError("Simulated post-passion crash")

        return result
    except Exception:
        logger.critical(
            "An unhandled exception crashed the pipeline core execution.",
            extra={"event_type": "pipeline_crash", "stage": "pipeline_core"},
            exc_info=True
        )

        if config.ENABLE_CRASH_DUMPS:
            try:
                _snapshot = dict(locals())
                _passion_debits, _passion_insights, _passion_signals = _resolve_passion_crash_fields(
                    result=_snapshot.get("result"),
                    locals_snapshot=_snapshot,
                )

                _wrote_any = _write_crash_dumps(
                    debits=_snapshot.get("debits"),
                    credits=_snapshot.get("credits"),
                    crash_dump_dir=config.CRASH_DUMP_DIR,
                    passion_debits=_passion_debits,
                    passion_insights=_passion_insights,
                    passion_signals=_passion_signals,
                    run_id=_snapshot.get("run_id"),
                )

                if _wrote_any:
                    logger.info(
                        "Crash state snapshots written.",
                        extra={"event_type": "crash_dump_success", "stage": "crash_handler"}
                    )
                else:
                    logger.info(
                        "No crash data available to persist.",
                        extra={"event_type": "crash_dump_empty", "stage": "crash_handler"}
                    )
            except Exception:
                logger.warning(
                    "Failed to write state dump to CSV during crash handling sequence.",
                    extra={"event_type": "crash_dump_failed", "stage": "crash_handler"},
                    exc_info=True
                )

        raise
    finally:
        pipeline_run_id_ctx.reset(token)


def run_inference(
    new_txn: pd.DataFrame,
    state: InsightModelState,
    history_df: pd.DataFrame,
    zscore_threshold: float = 3.0,
    pct_dev_threshold: float = 0.5,
    known_persons: dict = None,
    self_accounts: dict = None,
) -> PipelineResult:
    run_id = generate_new_run_id()
    token = pipeline_run_id_ctx.set(run_id)
    
    try:
        logger.info("Running inference on new transaction(s)...", extra={"event_type": "pipeline_inference_start"})

        kp = known_persons if known_persons is not None else config.KNOWN_PERSONS
        sa = self_accounts if self_accounts is not None else config.SELF_ACCOUNTS
        
        _validate_state_version(state, known_persons=kp, self_accounts=sa)

        # ── Phase 1: Preprocess ──
        debits, credits = preprocess(new_txn)
        
        # ── Phase 1.5: Known Persons ──
        debits = tag_known_persons(debits, known_persons=kp, self_accounts=sa)
        credits = tag_known_persons(credits, known_persons=kp, self_accounts=sa)
        history_df = tag_known_persons(history_df, known_persons=kp, self_accounts=sa)

        debits = _pre_initialize_ml_columns(debits)
        credits = _pre_initialize_ml_columns(credits)
        
        spend_mask = ~debits[Col.IS_KNOWN_PERSON]
        credit_spend_mask = ~credits[Col.IS_KNOWN_PERSON]
        history_spend_mask = ~history_df[Col.IS_KNOWN_PERSON]

        # ── Phase 2: Seed Labeling ────────────────────────────────────────────
        if spend_mask.any():
            spend_debits = debits.loc[spend_mask].copy()
            spend_debits = label_debits(spend_debits, label_col=Col.PSEUDO_LABEL)
            debits.loc[spend_mask, spend_debits.columns] = spend_debits
            
        if credit_spend_mask.any():
            spend_credits = credits.loc[credit_spend_mask].copy()
            spend_credits = label_credits(spend_credits, label_col=Col.PSEUDO_LABEL)
            credits.loc[credit_spend_mask, spend_credits.columns] = spend_credits
            
        debits = _enforce_known_person_schema(debits)
        credits = _enforce_known_person_schema(credits)

        # ── Phase 3: Feature Engineering (history-aware) ──
        if spend_mask.any():
            spend_debits = debits.loc[spend_mask].copy()
            spend_debits = engineer_features_inference(
                spend_debits,
                history_df=history_df.loc[history_spend_mask].copy(),
                global_mean=state.global_mean,
                global_std=state.global_std,
            )
            debits.loc[spend_mask, spend_debits.columns] = spend_debits
        debits = _enforce_known_person_schema(debits)

        # ── Phase 4: ML Prediction (pre-trained models) ──
        cat_pipeline, spend_pipeline, ranker_pipeline = (
            state.cat_pipeline, state.spend_pipeline, state.ranker_pipeline,
        )
        if spend_mask.any():
            spend_debits = debits.loc[spend_mask].copy()
            spend_debits = predict_categories(cat_pipeline, spend_debits)
            spend_debits = predict_expected_spend(spend_pipeline, spend_debits)
            debits.loc[spend_mask, spend_debits.columns] = spend_debits
        debits = _enforce_known_person_schema(debits)

        # ── Phase 5: Signal Detection ──
        if spend_mask.any():
            spend_debits = debits.loc[spend_mask].copy()
            spend_debits = detect_anomalies(
                spend_debits,
                zscore_threshold=zscore_threshold,
                pct_dev_threshold=pct_dev_threshold,
            )
            spend_debits = find_recurring_transactions(spend_debits, group_col=Col.CLEANED_REMARKS)
            debits.loc[spend_mask, spend_debits.columns] = spend_debits
        debits = _enforce_known_person_schema(debits)

        # ── Phase 5.5: ML Insight Ranking ──
        if spend_mask.any():
            spend_debits = debits.loc[spend_mask].copy()
            spend_debits = predict_insight_scores(ranker_pipeline, spend_debits)
            debits.loc[spend_mask, spend_debits.columns] = spend_debits
        debits = _enforce_known_person_schema(debits)

        # ── Phase 6: Finalize + Insight Generation ──
        debits = finalize_df(debits)
        credits = finalize_df(credits)
        
        debits = _optimize_memory_footprint(debits)
        credits = _optimize_memory_footprint(credits)
        
        insights = []
        if spend_mask.any():
            insights = generate_human_insights(debits.loc[spend_mask])

        result = PipelineResult(
            debits=debits,
            credits=credits,
            insights=insights,
            cat_pipeline=cat_pipeline,
            spend_pipeline=spend_pipeline,
            ranker_pipeline=ranker_pipeline,
            global_mean=state.global_mean,
            global_std=state.global_std,
            stats_version=state.stats_version,
            kp_config_hash=state.kp_config_hash,
            personal_debits=debits.loc[spend_mask == False].copy(),
            personal_credits=credits.loc[credits[Col.IS_KNOWN_PERSON].fillna(False)].copy()
        )

        # Phase 7: Passion Engine (optional — errors are swallowed)
        result = _attach_passion_results(result)
        return result
    except Exception:
        logger.critical(
            "Inference crashed (no crash dump available).", 
            extra={"event_type": "inference_crash", "stage": "inference_core"}, 
            exc_info=True
        )
        raise
    finally:
        # Safe resolution explicitly without masking NameErrors
        pipeline_run_id_ctx.reset(token)