"""
llm_export.py — Public LLM Export API
======================================
Public API module for the LLM JSON export interface.

Provides:
  build_llm_context(result) -> dict
      Derive a PII-safe, JSON-serializable, LLM-ready context dict from a
      completed PipelineResult. Pure read — never mutates result.

  export_to_json(result, output_path=None) -> str
      Build LLM context and write it atomically to disk as JSON.

Design decisions:
  - run_id is NOT included in the dict returned by build_llm_context (preserves
    determinism for testing). export_to_json generates a fresh UUID, injects it
    as context["run_id"], then writes to disk.
  - DO NOT use pipeline_run_id_ctx — the ContextVar is reset in the pipeline's
    finally: block (pipeline.py:814, :946) and returns its pre-call default by
    the time export_to_json is called.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import TYPE_CHECKING
from uuid import uuid4

import pandas as pd

import config
from schema import Col
from insight_generator import _generate_insight_records
from llm_serializer import write_json_atomic, get_pii_safe_flag
from llm_export_aggregators import (
    build_period,
    build_spend_profile,
    build_anomaly_summary,
    build_recurring_subscriptions,
    build_personal_transfers,
    build_budget_health,
)
from llm_export_passion import build_passion_context

if TYPE_CHECKING:
    from pipeline import PipelineResult

logger = logging.getLogger(__name__)

__all__ = ["build_llm_context", "export_to_json"]


def build_llm_context(result: "PipelineResult") -> dict:
    """
    Derive a JSON-serializable, PII-safe, LLM-ready context dict from a
    completed PipelineResult.

    Pure read — never mutates result.
    All fields pre-aggregated so the LLM does not parse raw DataFrames.

    IMPORTANT: The returned dict does NOT contain "run_id". The run_id is
    injected by export_to_json() after building this dict, to keep this
    function deterministic (same input → same output).

    Args:
        result: A completed PipelineResult from run_pipeline() or run_inference().

    Returns:
        dict — InsightEngineContext conforming to schema_version "1.0".

    Raises:
        TypeError:  if result is not a PipelineResult-like object.
        ValueError: if result.debits is empty or missing required columns.
    """
    # ── Validate input ─────────────────────────────────────────────────────────
    if not hasattr(result, "debits") or not hasattr(result, "credits"):
        raise TypeError(
            f"result must be a PipelineResult instance (has 'debits' and 'credits' attrs), "
            f"got {type(result).__name__}"
        )

    debits: pd.DataFrame = result.debits
    credits: pd.DataFrame = result.credits

    if not isinstance(debits, pd.DataFrame) or debits.empty:
        raise ValueError(
            "result.debits must be a non-empty DataFrame. "
            "Ensure the pipeline ran successfully before calling build_llm_context()."
        )

    if Col.AMOUNT not in debits.columns:
        raise ValueError(
            f"result.debits is missing required column '{Col.AMOUNT}'. "
            "Ensure the pipeline ran successfully."
        )

    # ── PII gate ───────────────────────────────────────────────────────────────
    pii_safe = get_pii_safe_flag()

    # ── Spend-only debits (exclude known-person rows) ──────────────────────────
    if Col.IS_KNOWN_PERSON in debits.columns:
        spend_debits = debits[~debits[Col.IS_KNOWN_PERSON].fillna(False)]
    else:
        spend_debits = debits

    # ── Global stats from result ───────────────────────────────────────────────
    global_mean = float(getattr(result, "global_mean", 0.0) or 0.0)
    global_std = float(getattr(result, "global_std", 0.0) or 0.0)
    passion_signals = getattr(result, "passion_signals", ()) or ()

    # ── Build sections ─────────────────────────────────────────────────────────
    period = build_period(debits, credits)

    spend_profile = build_spend_profile(spend_debits, global_mean, global_std)

    anomaly_summary = build_anomaly_summary(spend_debits, pii_safe=pii_safe)

    recurring_subscriptions = build_recurring_subscriptions(spend_debits, pii_safe=pii_safe)

    personal_transfers = build_personal_transfers(result)

    budget_health = build_budget_health(debits, credits, passion_signals=passion_signals)

    passion_insights_section = build_passion_context(result, pii_safe=pii_safe)

    # ── Structured expense insight records ─────────────────────────────────────
    insight_records: list = []
    if not spend_debits.empty:
        try:
            insight_records = _generate_insight_records(spend_debits)
        except Exception as exc:
            logger.warning(
                "build_llm_context_insight_records_failed",
                extra={"error": str(exc)},
            )
            insight_records = []

    # ── Assemble context dict (NO run_id — injected by export_to_json) ─────────
    context: dict = {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "period": period,
        "spend_profile": spend_profile,
        "anomaly_summary": anomaly_summary,
        "recurring_subscriptions": recurring_subscriptions,
        "personal_transfers": personal_transfers,
        "budget_health": budget_health,
        "passion_insights": passion_insights_section,
        "expense_insights": insight_records,
    }

    logger.debug(
        "build_llm_context_complete",
        extra={
            "spend_categories": len(spend_profile),
            "anomaly_count": anomaly_summary.get("total_count", 0),
            "recurring_count": len(recurring_subscriptions),
            "passion_signals": passion_insights_section.get("signal_count", 0),
            "insight_records": len(insight_records),
        },
    )
    return context


def export_to_json(
    result: "PipelineResult",
    output_path: str | None = None,
    *,
    user_id: str | None = None,
    run_id: str | None = None,
) -> str:
    """
    Build LLM context and write it atomically to disk as JSON.

    Internal flow:
      1. export_id = uuid4().hex[:12]   (fresh UUID — see plan note C10)
      2. context = build_llm_context(result)
      3. context["run_id"] = export_id  (injected HERE, not in build_llm_context)
      4. if output_path is None:
             output_path = <LLM_EXPORT_OUTPUT_DIR>/<export_id>_llm_context.json
      5. write_json_atomic(context, output_path)

    DO NOT use pipeline_run_id_ctx — the ContextVar is reset in the
    pipeline's finally: block (pipeline.py:814, :946) and returns its
    pre-call default by the time export_to_json is called.

    Args:
        result:      A completed PipelineResult.
        output_path: Optional explicit output file path. If None, a path is
                     auto-generated under LLM_EXPORT_OUTPUT_DIR.

    Returns:
        str — absolute path to the written JSON file.

    Raises:
        TypeError:  if result is not a PipelineResult-like object.
        ValueError: if result.debits is empty or missing required columns.
        OSError:    on file system write failure.
    """
    export_id = run_id if run_id is not None else uuid4().hex[:12]

    context = build_llm_context(result)
    context["run_id"] = export_id  # Inject run_id into the in-memory dict before serializing

    if output_path is None:
        base_dir   = getattr(config, "LLM_EXPORT_OUTPUT_DIR", "output/llm_context")
        output_dir = os.path.join(base_dir, user_id) if user_id is not None else base_dir
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{export_id}_llm_context.json")

    write_json_atomic(context, output_path)

    logger.info(
        "llm_context_exported",
        extra={
            "export_id": export_id,
            "path": os.path.abspath(output_path),
        },
    )
    return os.path.abspath(output_path)
