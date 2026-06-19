"""
llm_export_aggregators.py — Aggregation Functions for LLM Export
================================================================
Six pure aggregation functions that read from PipelineResult (or sub-DataFrames)
and return JSON-ready dicts. All functions are read-only — they never mutate result.

Functions:
  build_period               : Period summary (dates, transaction counts, totals).
  build_spend_profile        : Per-category spend breakdown.
  build_anomaly_summary      : Anomaly summary with top N anomalies.
  build_recurring_subscriptions : Recurring subscription list.
  build_personal_transfers   : Personal transfer breakdown with per-alias patterns.
  build_budget_health        : Budget health metrics including savings rate.

All 6 conflict resolutions from plan § "Conflict Analysis" are applied.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Sequence, Tuple

import pandas as pd

import config
from schema import Col
from llm_serializer import _mask_merchant

if TYPE_CHECKING:
    from pipeline import PipelineResult

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _safe_date_str(dt) -> str | None:
    """Convert a date/datetime/Timestamp to ISO-8601 string, or None on failure."""
    if dt is None or (isinstance(dt, float) and pd.isna(dt)):
        return None
    try:
        if hasattr(dt, "strftime"):
            return dt.strftime("%Y-%m-%d")
        return str(dt)
    except Exception:
        return None


def _safe_float(val, default: float = 0.0) -> float:
    """Coerce value to float; return default on NaN/Inf/error."""
    import math
    try:
        v = float(val)
        return default if (math.isnan(v) or math.isinf(v)) else v
    except (TypeError, ValueError):
        return default


# ──────────────────────────────────────────────────────────────────────────────
# 1. build_period
# ──────────────────────────────────────────────────────────────────────────────

def build_period(debits: pd.DataFrame, credits: pd.DataFrame) -> dict:
    """
    Build the period summary section.

    Args:
        debits:  Core debits DataFrame.
        credits: Core credits DataFrame.

    Returns:
        dict with keys: start, end, total_transactions, total_spend_inr,
        total_inflow_inr, net_cashflow_inr, data_completeness_pct.
    """
    start = None
    end = None
    if not debits.empty and Col.DATE in debits.columns:
        dates = debits[Col.DATE].dropna()
        if not dates.empty:
            start = _safe_date_str(dates.min())
            end = _safe_date_str(dates.max())

    total_transactions = len(debits) + len(credits)
    total_spend_inr = _safe_float(debits[Col.AMOUNT].sum()) if Col.AMOUNT in debits.columns else 0.0
    total_inflow_inr = _safe_float(credits[Col.AMOUNT].sum()) if Col.AMOUNT in credits.columns else 0.0
    net_cashflow_inr = total_inflow_inr - total_spend_inr

    return {
        "start": start,
        "end": end,
        "total_transactions": total_transactions,
        "total_spend_inr": total_spend_inr,
        "total_inflow_inr": total_inflow_inr,
        "net_cashflow_inr": net_cashflow_inr,
        "data_completeness_pct": 100.0,  # KC-01: v1 — no cross-run comparison
    }


# ──────────────────────────────────────────────────────────────────────────────
# 2. build_spend_profile
# ──────────────────────────────────────────────────────────────────────────────

def build_spend_profile(
    debits: pd.DataFrame,
    global_mean: float,
    global_std: float,
) -> list:
    """
    Build the per-category spend profile.

    Excludes known-person debits (Col.IS_KNOWN_PERSON == True).
    Groups by predicted_category, computes per-category stats.
    Sorted by total_spend descending, capped at LLM_EXPORT_MAX_CATEGORIES.

    Args:
        debits:      Full debits DataFrame.
        global_mean: Filtered global mean from PipelineResult.
        global_std:  Filtered global std from PipelineResult.

    Returns:
        List of category dicts with keys: category, total_spend, transaction_count,
        share_of_total_pct, avg_per_transaction, month_over_month_change_pct,
        anomaly_count, has_recurring.
    """
    required = {Col.PREDICTED_CATEGORY, Col.AMOUNT}
    if not required.issubset(set(debits.columns)):
        logger.warning("build_spend_profile: missing required columns; returning empty list.")
        return []

    # Exclude known-person rows
    if Col.IS_KNOWN_PERSON in debits.columns:
        spend_df = debits[~debits[Col.IS_KNOWN_PERSON].fillna(False)].copy()
    else:
        spend_df = debits.copy()

    if spend_df.empty:
        return []

    grand_total = _safe_float(spend_df[Col.AMOUNT].sum())
    max_cats = getattr(config, "LLM_EXPORT_MAX_CATEGORIES", 15)

    rows = []
    for cat, grp in spend_df.groupby(Col.PREDICTED_CATEGORY):
        total_spend = _safe_float(grp[Col.AMOUNT].sum())
        count = len(grp)
        share = round((total_spend / grand_total * 100), 2) if grand_total > 0 else 0.0
        avg = round(total_spend / count, 2) if count > 0 else 0.0
        anomaly_count = int(grp[Col.IS_ANOMALY].sum()) if Col.IS_ANOMALY in grp.columns else 0
        has_recurring = bool(grp[Col.IS_RECURRING].any()) if Col.IS_RECURRING in grp.columns else False

        rows.append({
            "category": str(cat),
            "total_spend": total_spend,
            "transaction_count": count,
            "share_of_total_pct": share,
            "avg_per_transaction": avg,
            "month_over_month_change_pct": None,  # KC-01: v1 — requires cross-run history
            "anomaly_count": anomaly_count,
            "has_recurring": has_recurring,
        })

    rows.sort(key=lambda r: r["total_spend"], reverse=True)
    return rows[:max_cats]


# ──────────────────────────────────────────────────────────────────────────────
# 3. build_anomaly_summary
# ──────────────────────────────────────────────────────────────────────────────

def build_anomaly_summary(debits: pd.DataFrame, pii_safe: bool) -> dict:
    """
    Build the anomaly summary section.

    Args:
        debits:   Full debits DataFrame.
        pii_safe: PII masking flag.

    Returns:
        dict with keys: total_count, total_excess_spend, top_anomalies (list).
    """
    required = {Col.IS_ANOMALY, Col.AMOUNT}
    if not required.issubset(set(debits.columns)):
        logger.warning("build_anomaly_summary: missing required columns; returning empty summary.")
        return {"total_count": 0, "total_excess_spend": 0.0, "top_anomalies": []}

    anomalies = debits[debits[Col.IS_ANOMALY].fillna(False)]
    total_count = len(anomalies)

    # total_excess_spend = sum(amount - expected_amount) for anomalous rows
    if Col.EXPECTED_AMOUNT in anomalies.columns:
        excess = (anomalies[Col.AMOUNT] - anomalies[Col.EXPECTED_AMOUNT]).clip(lower=0)
        total_excess_spend = _safe_float(excess.sum())
    else:
        total_excess_spend = 0.0

    max_anomalies = getattr(config, "LLM_EXPORT_MAX_ANOMALIES", 10)

    # Sort by insight_score descending
    if Col.INSIGHT_SCORE in anomalies.columns:
        anomalies_sorted = anomalies.sort_values(by=Col.INSIGHT_SCORE, ascending=False)
    else:
        anomalies_sorted = anomalies

    top_anomalies = []
    for _, row in anomalies_sorted.head(max_anomalies).iterrows():
        merchant_raw = str(row.get(Col.CLEANED_REMARKS, "")) if Col.CLEANED_REMARKS in anomalies.columns else ""
        top_anomalies.append({
            "date": _safe_date_str(row.get(Col.DATE)),
            "merchant": _mask_merchant(merchant_raw, pii_safe=pii_safe),
            "category": str(row.get(Col.PREDICTED_CATEGORY, "uncategorized")) if Col.PREDICTED_CATEGORY in anomalies.columns else "uncategorized",
            "amount": _safe_float(row.get(Col.AMOUNT, 0.0)),
            "expected_amount": _safe_float(row.get(Col.EXPECTED_AMOUNT, 0.0)) if Col.EXPECTED_AMOUNT in anomalies.columns else None,
            "percent_deviation": _safe_float(row.get(Col.PERCENT_DEVIATION, 0.0)) if Col.PERCENT_DEVIATION in anomalies.columns else None,
            "insight_score": _safe_float(row.get(Col.INSIGHT_SCORE, 0.0)) if Col.INSIGHT_SCORE in anomalies.columns else None,
        })

    return {
        "total_count": total_count,
        "total_excess_spend": total_excess_spend,
        "top_anomalies": top_anomalies,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 4. build_recurring_subscriptions
# ──────────────────────────────────────────────────────────────────────────────

def build_recurring_subscriptions(debits: pd.DataFrame, pii_safe: bool) -> list:
    """
    Build the recurring subscriptions list.

    Groups IS_RECURRING rows by cleaned_remarks.
    Capped at LLM_EXPORT_MAX_SUBSCRIPTIONS.

    Args:
        debits:   Full debits DataFrame.
        pii_safe: PII masking flag.

    Returns:
        List of subscription dicts with keys: merchant, frequency,
        avg_amount, total_paid_in_period, insight_score.
    """
    required = {Col.IS_RECURRING, Col.CLEANED_REMARKS, Col.AMOUNT}
    if not required.issubset(set(debits.columns)):
        logger.warning("build_recurring_subscriptions: missing required columns; returning empty list.")
        return []

    recurring = debits[debits[Col.IS_RECURRING].fillna(False)]
    if recurring.empty:
        return []

    max_subs = getattr(config, "LLM_EXPORT_MAX_SUBSCRIPTIONS", 10)

    rows = []
    for name, grp in recurring.groupby(Col.CLEANED_REMARKS):
        freq = str(grp[Col.RECURRING_FREQUENCY].iloc[-1]) if Col.RECURRING_FREQUENCY in grp.columns else "unknown"
        avg_amount = _safe_float(grp[Col.AMOUNT].mean())
        total_paid = _safe_float(grp[Col.AMOUNT].sum())
        score = _safe_float(grp[Col.INSIGHT_SCORE].max()) if Col.INSIGHT_SCORE in grp.columns else 0.0

        rows.append({
            "merchant": _mask_merchant(str(name), pii_safe=pii_safe),
            "frequency": freq,
            "avg_amount": avg_amount,
            "total_paid_in_period": total_paid,
            "insight_score": score,
        })

    # Sort by total_paid descending
    rows.sort(key=lambda r: r["total_paid_in_period"], reverse=True)
    return rows[:max_subs]


# ──────────────────────────────────────────────────────────────────────────────
# 5. build_personal_transfers
# ──────────────────────────────────────────────────────────────────────────────

def build_personal_transfers(result: "PipelineResult") -> dict:
    """
    Build the personal_transfers section of the LLM context.

    All 6 conflict resolutions from plan § "Conflict Analysis" are applied:
      - Conflict 1: personal_summary absent (run_inference) → pattern fields are null.
      - Conflict 2: Self:* aliases → always pattern=null regardless of personal_summary.
      - Conflict 3: Col.TRANSFER_CLASS absent → transfer_class_summary=null, warning logged.
      - Conflict 4: exclusion_stats absent → fallback formula used.
      - Conflict 5: KNOWN_PERSON_ALIAS NA → dropped before groupby, warning logged.
      - Conflict 6: transfer_external always present in summary with zeros as floor.

    Args:
        result: PipelineResult (read-only).

    Returns:
        dict — personal_transfers section.
    """
    personal_debits: pd.DataFrame = getattr(result, "personal_debits", pd.DataFrame())
    personal_credits: pd.DataFrame = getattr(result, "personal_credits", pd.DataFrame())
    personal_summary: dict = getattr(result, "personal_summary", {}) or {}
    transfer_patterns: list = getattr(result, "transfer_patterns", []) or []
    exclusion_stats: dict = getattr(result, "exclusion_stats", {}) or {}
    debits: pd.DataFrame = getattr(result, "debits", pd.DataFrame())

    # Totals
    total_sent = _safe_float(personal_debits[Col.AMOUNT].sum()) if Col.AMOUNT in personal_debits.columns and not personal_debits.empty else 0.0
    total_received = _safe_float(personal_credits[Col.AMOUNT].sum()) if Col.AMOUNT in personal_credits.columns and not personal_credits.empty else 0.0
    net_inr = total_received - total_sent
    transfer_count = len(personal_debits) + len(personal_credits)

    # Conflict 4: exclusion_rate_pct fallback
    if exclusion_stats and "exclusion_rate" in exclusion_stats:
        exclusion_rate_pct = round(_safe_float(exclusion_stats["exclusion_rate"]) * 100, 2)
    else:
        # Fallback: compute from DataFrame lengths
        denom = max(len(debits), 1)
        exclusion_rate_pct = round(len(personal_debits) / denom * 100, 2)

    # ── by_alias ──────────────────────────────────────────────────────────────
    by_alias = []
    if not personal_debits.empty and Col.KNOWN_PERSON_ALIAS in personal_debits.columns:
        # Conflict 5: drop NA aliases before groupby
        clean_debits = personal_debits.copy()
        na_mask = clean_debits[Col.KNOWN_PERSON_ALIAS].isna()
        if na_mask.any():
            logger.warning(
                "personal_transfers_na_aliases_dropped",
                extra={"count": int(na_mask.sum())},
            )
            clean_debits = clean_debits[~na_mask]

        alias_groups = clean_debits.groupby(Col.KNOWN_PERSON_ALIAS)
        for alias, grp in alias_groups:
            alias_str = str(alias)

            # transfer_class for this alias
            if Col.TRANSFER_CLASS in grp.columns:
                transfer_class = str(grp[Col.TRANSFER_CLASS].iloc[0]) if not grp.empty else None
            else:
                transfer_class = None

            alias_sent = _safe_float(grp[Col.AMOUNT].sum()) if Col.AMOUNT in grp.columns else 0.0

            # received for this alias from personal_credits
            alias_received = 0.0
            if not personal_credits.empty and Col.KNOWN_PERSON_ALIAS in personal_credits.columns:
                credit_rows = personal_credits[personal_credits[Col.KNOWN_PERSON_ALIAS] == alias]
                alias_received = _safe_float(credit_rows[Col.AMOUNT].sum()) if Col.AMOUNT in credit_rows.columns else 0.0

            tx_count = len(grp)

            # Conflict 1 & 2: pattern fields
            # Conflict 2: Self:* aliases ALWAYS get null pattern
            is_self = alias_str.startswith("Self:")
            if is_self or not personal_summary:
                pattern = None
                avg_amount = None
                frequency_days = None
            else:
                pdata = personal_summary.get(alias_str, {})
                pattern = pdata.get("pattern") if pdata else None
                avg_amount = pdata.get("avg_amount") if pdata else None
                frequency_days = pdata.get("frequency_days") if pdata else None

            by_alias.append({
                "alias": alias_str,
                "transfer_class": transfer_class,
                "total_sent_inr": alias_sent,
                "total_received_inr": alias_received,
                "transaction_count": tx_count,
                "pattern": pattern,
                "avg_amount": avg_amount,
                "frequency_days": frequency_days,
            })

    # ── transfer_class_summary ─────────────────────────────────────────────────
    # Conflict 6: pre-initialise all 3 classes with zeros as floor
    # Conflict 3: if Col.TRANSFER_CLASS absent, return null
    ALL_TRANSFER_CLASSES = ["transfer_self", "transfer_known", "transfer_external"]
    if not personal_debits.empty and Col.TRANSFER_CLASS in personal_debits.columns:
        transfer_class_summary = {cls: {"count": 0, "total_inr": 0.0} for cls in ALL_TRANSFER_CLASSES}
        for cls, grp in personal_debits.groupby(Col.TRANSFER_CLASS):
            cls_str = str(cls)
            if cls_str in transfer_class_summary:
                transfer_class_summary[cls_str] = {
                    "count": len(grp),
                    "total_inr": _safe_float(grp[Col.AMOUNT].sum()) if Col.AMOUNT in grp.columns else 0.0,
                }
    elif not personal_debits.empty and Col.TRANSFER_CLASS not in personal_debits.columns:
        # Conflict 3: TRANSFER_CLASS column absent — log warning and return null
        logger.warning(
            "personal_transfers_transfer_class_missing",
            extra={"columns": list(personal_debits.columns)},
        )
        transfer_class_summary = None
    else:
        transfer_class_summary = {cls: {"count": 0, "total_inr": 0.0} for cls in ALL_TRANSFER_CLASSES}

    return {
        "total_sent_inr": total_sent,
        "total_received_inr": total_received,
        "net_inr": net_inr,
        "transfer_count": transfer_count,
        "exclusion_rate_pct": exclusion_rate_pct,
        "by_alias": by_alias,
        "transfer_class_summary": transfer_class_summary,
        "transfer_insights": list(transfer_patterns),
    }


# ──────────────────────────────────────────────────────────────────────────────
# 6. build_budget_health
# ──────────────────────────────────────────────────────────────────────────────

def build_budget_health(
    debits: pd.DataFrame,
    credits: pd.DataFrame,
    passion_signals: tuple = (),
) -> dict:
    """
    Build the budget health section.

    Args:
        debits:          Full debits DataFrame.
        credits:         Full credits DataFrame.
        passion_signals: Tuple of PassionSignal instances (default empty tuple).

    Returns:
        dict with keys: savings_rate_pct, high_spend_categories,
        declining_categories, risk_flags.
    """
    total_inflow = _safe_float(credits[Col.AMOUNT].sum()) if Col.AMOUNT in credits.columns else 0.0
    total_spend = _safe_float(debits[Col.AMOUNT].sum()) if Col.AMOUNT in debits.columns else 0.0

    # savings_rate = (total_inflow - total_spend) / total_inflow * 100
    if total_inflow > 0:
        savings_rate_pct = round((total_inflow - total_spend) / total_inflow * 100, 2)
    else:
        savings_rate_pct = None  # Undefined when there is no inflow

    # high_spend_categories: top 3 categories by spend share
    high_spend_categories = []
    if not debits.empty and Col.PREDICTED_CATEGORY in debits.columns and Col.AMOUNT in debits.columns:
        if Col.IS_KNOWN_PERSON in debits.columns:
            spend_df = debits[~debits[Col.IS_KNOWN_PERSON].fillna(False)]
        else:
            spend_df = debits

        grand_total = _safe_float(spend_df[Col.AMOUNT].sum())
        if grand_total > 0:
            cat_totals = (
                spend_df.groupby(Col.PREDICTED_CATEGORY)[Col.AMOUNT]
                .sum()
                .sort_values(ascending=False)
            )
            for cat, total in cat_totals.head(3).items():
                high_spend_categories.append({
                    "category": str(cat),
                    "share_pct": round(_safe_float(total) / grand_total * 100, 2),
                })

    # declining_categories: from passion_signals where trend is "declining" and not suppressed
    declining_categories = [
        sig.subcategory if sig.subcategory else sig.category
        for sig in (passion_signals or ())
        if getattr(sig, "trend_direction", "") == "declining" and not getattr(sig, "is_suppressed", True)
    ]

    # risk_flags: categories exceeding 30% of total spend
    risk_flags = []
    for entry in high_spend_categories:
        if entry["share_pct"] > 30.0:
            risk_flags.append(
                f"{entry['category']} represents {entry['share_pct']}% of total spend — exceeds 30% risk threshold."
            )

    return {
        "savings_rate_pct": savings_rate_pct,
        "high_spend_categories": high_spend_categories,
        "declining_categories": declining_categories,
        "risk_flags": risk_flags,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 7. build_spend_insights
# ──────────────────────────────────────────────────────────────────────────────

def build_spend_insights(
    insight_records: list,
    exclusion_stats: dict,
) -> dict:
    """
    Build the spend_insights section of the API response.

    Uses paired objects — text + tip co-located per record — to avoid
    parallel-array index-alignment breakage (conflict C6).

    Source MUST be _generate_insight_records(), NOT result.insights.
    result.insights merges personal-transfer strings (pipeline.py:716)
    that are not spend anomalies and would render incorrectly as spend cards.

    Args:
        insight_records: Output of _generate_insight_records() —
                         list of {type, text, tip, score} dicts.
        exclusion_stats: result.exclusion_stats dict.

    Returns:
        dict with keys:
            insights : list of {text, tip, type, score}
                       tip is "" when no tip exists — never omitted.
            stats    : {total_transactions, excluded_transactions, exclusion_rate}
    """
    paired = [
        {
            "text":  str(r.get("text", "")),
            "tip":   str(r.get("tip", "")),
            "type":  str(r.get("type", "")),
            "score": _safe_float(r.get("score", 0.0)),
        }
        for r in (insight_records or [])
        if r.get("text")  # guard: skip malformed empty-text records
    ]

    stats = {
        "total_transactions":   int(exclusion_stats.get("total_transactions", 0)),
        "excluded_transactions": int(exclusion_stats.get("excluded_transactions", 0)),
        "exclusion_rate":       _safe_float(exclusion_stats.get("exclusion_rate", 0.0)),
    }

    return {"insights": paired, "stats": stats}
