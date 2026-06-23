"""
llm_export_passion.py — Passion Signal Serialization for LLM Export
====================================================================
Responsibilities:
  - _serialize_passion_signal : Convert a PassionSignal to a JSON-ready dict.
  - build_passion_context     : Build the full passion insights section for LLM context.

Field derivation rules (per plan U2):
  - display_label : sig.subcategory.replace("_", " ").title() if subcategory else
                    sig.category.replace("_", " ").title()
  - narrative     : Synthesized directly from signal fields — NOT from result.passion_insights.
                    (passion_insight_generator applies banned-content filtering and deduplication
                    that breaks index correlation with result.passion_signals.)
  - Suppressed signals are excluded from export (only is_suppressed=False exported).
  - Capped at config.LLM_EXPORT_MAX_PASSION_SIGNALS.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Sequence

import config
from llm_serializer import _mask_merchant

if TYPE_CHECKING:
    from passion_models import PassionSignal
    from pipeline import PipelineResult

logger = logging.getLogger(__name__)


def _serialize_passion_signal(sig: "PassionSignal", pii_safe: bool) -> dict:
    """
    Serialize a single PassionSignal to a JSON-ready dict.

    Args:
        sig:      PassionSignal instance (is_suppressed=False enforced by caller).
        pii_safe: True → mask merchant names via HMAC token.

    Returns:
        dict with keys: category, subcategory, display_label, total_spend,
        spend_share_pct, merchant_count, active_months, trend, is_suppressed,
        suppression_reason, narrative, tip.
    """
    # display_label: prefer subcategory, fall back to category
    if sig.subcategory:
        display_label = sig.subcategory.replace("_", " ").title()
    else:
        display_label = sig.category.replace("_", " ").title()

    # narrative: synthesized from signal fields — do NOT use result.passion_insights
    narrative = (
        f"{display_label}: \u20b9{sig.total_spend:.0f} across {sig.merchant_count} "
        f"merchant(s), {sig.active_months} month(s) active. "
        f"Trend: {sig.trend_direction.replace('_', ' ')}."
    )

    # spend_share is 0–1 float; convert to percentage for LLM readability
    spend_share_pct = round(float(sig.spend_share) * 100, 2)

    return {
        "category": sig.category,
        "subcategory": sig.subcategory,
        "display_label": display_label,
        "total_spend": float(sig.total_spend),
        "spend_share_pct": spend_share_pct,
        "merchant_count": int(sig.merchant_count),
        "active_months": int(sig.active_months),
        "trend": sig.trend_direction,
        "is_suppressed": bool(sig.is_suppressed),
        "suppression_reason": sig.suppression_reason or "",
        "narrative": narrative,
        "tip": sig.tip,
    }


def build_passion_context(result: "PipelineResult", pii_safe: bool) -> dict:
    """
    Build the passion_insights section of the LLM context dict.

    Rules:
        - passion feature must be enabled (env: INSIGHT_ENGINE_PASSION_ENABLED=true).
        - Only non-suppressed signals are exported.
        - Capped at config.LLM_EXPORT_MAX_PASSION_SIGNALS.
        - If passion is disabled or no signals exist, returns {"enabled": False, "signals": []}.

    Args:
        result:   Completed PipelineResult (read-only).
        pii_safe: PII masking flag — passed through to _serialize_passion_signal.

    Returns:
        dict with keys: enabled, signal_count, signals.
    """
    passion_enabled = os.environ.get("INSIGHT_ENGINE_PASSION_ENABLED", "true").lower() == "true"

    if not passion_enabled:
        logger.debug("Passion feature disabled — returning empty passion context.")
        return {"enabled": False, "signal_count": 0, "signals": []}

    raw_signals: Sequence = getattr(result, "passion_signals", ()) or ()

    # Filter suppressed signals
    active_signals = [sig for sig in raw_signals if not sig.is_suppressed]

    if not active_signals:
        logger.debug("No active (non-suppressed) passion signals to export.")
        return {"enabled": True, "signal_count": 0, "signals": []}

    # Sort by spend descending for deterministic output, then cap
    active_signals_sorted = sorted(
        active_signals, key=lambda s: float(s.total_spend), reverse=True
    )
    max_signals = getattr(config, "LLM_EXPORT_MAX_PASSION_SIGNALS", 5)
    capped = active_signals_sorted[:max_signals]

    serialized = []
    for sig in capped:
        try:
            serialized.append(_serialize_passion_signal(sig, pii_safe=pii_safe))
        except Exception as exc:
            logger.warning(
                "passion_signal_serialization_failed",
                extra={
                    "category": getattr(sig, "category", "unknown"),
                    "error": str(exc),
                },
            )
            # Skip malformed signal — do not crash the entire export

    logger.debug(
        f"Passion context built: {len(serialized)} signals "
        f"(from {len(raw_signals)} total, {len(active_signals)} active)."
    )
    return {
        "enabled": True,
        "signal_count": len(serialized),
        "signals": serialized,
    }
