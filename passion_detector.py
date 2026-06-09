import numpy as np
import re
import math
import pandas as pd
from schema import Col
from config_passion import (
    DISTRESS_FEES_THRESHOLD,
    PASSION_MERCHANT_COUNT_MIN,
    PASSION_ANOMALY_SUPPRESSION_THRESHOLD,
    PASSION_MIN_MONTHS,
    PASSION_SPEND_SHARE_THRESHOLD,
)
from passion_utils import assert_columns_exist, _safe_isna, sanitize_mask

# FIX-05: Use safe_numeric from passion_utils instead of pd.to_numeric for
# amount handling. safe_numeric handles currency prefixes ("Rs", "INR", "₹")
# that pd.to_numeric silently coerces to NaN.
from passion_utils import safe_numeric

from marketplace_subcategory import resolve_merchant_vectorized
from passion_models import PassionSignal

# Step 4b: Guarded import for per-subcategory merchant count overrides.
# Falls back to empty dict if SUBCATEGORY_MERCHANT_COUNT_MIN not yet deployed.
try:
    from config_passion import SUBCATEGORY_MERCHANT_COUNT_MIN as _SUBCATEGORY_MIN
except ImportError:
    _SUBCATEGORY_MIN: dict[str, int] = {}

# Guarded import for blocked categories (ATM, finance, utilities are never passions).
try:
    from config_passion import PASSION_BLOCKED_CATEGORIES as _BLOCKED_CATEGORIES
except ImportError:
    _BLOCKED_CATEGORIES: frozenset[str] = frozenset()

# FIX H11: Use structured logger from logger_factory instead of plain logging.
from logger_factory import get_logger

__all__ = ["detect_passions"]

logger = get_logger(__name__)

_FEE_KEYWORDS = frozenset({
    "fee", "charge", "penalty", "interest", "overdue",
    "late payment", "bounce", "insufficient", "overlimit",
})

# P1.3: Use (?<!\w)/(?!\w) instead of \b.
_FEE_PATTERN = re.compile(
    r'(?<!\w)(?:' + '|'.join(
        re.escape(kw) for kw in sorted(_FEE_KEYWORDS, key=len, reverse=True)
    ) + r')(?!\w)',
    flags=re.IGNORECASE,
)

# Step 4c (D2 approved): Education-specific exemption.
# These remarks contain "fee" or "charge" but represent legitimate passion spending,
# not financial distress. The lookbehind (?<!\w) and lookahead (?!\w) prevent partial
# matches. fees? matches both singular ("fee") and plural ("fees") — Indian transaction
# data commonly uses plural forms ("tuition fees", "exam fees").
_EDUCATION_EXEMPTION_PATTERN = re.compile(
    r'(?<!\w)(?:tuition|course|exam|admission|registration|class|training)\s*fees?(?!\w)',
    re.IGNORECASE,
)

# Non-passion merchant pattern: regex matching descriptions that represent structural
# (non-discretionary) spend regardless of predicted category.
# Common in Indian UPI data: rent via "landlord <name> oid", maid/cook payments,
# society maintenance, etc. These must be excluded from unique_merchants before gate
# evaluation to prevent them from inflating passion merchant counts.
_NON_PASSION_MERCHANT_PATTERN = re.compile(
    r'(?<!\w)(?:landlord|rent|maid|cook|society|maintenance|servant|househelp)\b',
    re.IGNORECASE,
)


# Fix 17: _BOOL_MAP and _safe_coerce_anomaly handle IS_ANOMALY column values that
# arrive as strings ('True', '1', 'yes', 'false', etc.) due to CSV round-trips
# or upstream coercion bugs. coerce_bool_column alone does not handle all variants.
_BOOL_MAP: dict[str, bool] = {
    "true": True, "1": True, "yes": True,
    "false": False, "0": False, "no": False,
}


def _safe_coerce_anomaly(col: "pd.Series") -> "pd.Series":
    """Fix 17: Coerce IS_ANOMALY column to bool robustly, matching fixlist spec.
    - bool dtype   : fillna(False) fast-path, no allocation
    - string/object: vectorized _BOOL_MAP lookup; logs invalid-value count
    - numeric/other: delegates to coerce_bool_column (existing path)
    """
    from passion_utils import coerce_bool_column

    if pd.api.types.is_bool_dtype(col):
        return col.fillna(False)

    if pd.api.types.is_string_dtype(col) or col.dtype == object:
        mapped = col.map(
            lambda v: _BOOL_MAP.get(str(v).strip().lower(), None)
            if pd.notna(v)
            else False
        )
        invalid = mapped.isna() & col.notna()
        if invalid.any():
            logger.warning(
                "anomaly_coerce_invalid_values",
                extra={"count": int(invalid.sum())},
            )
        return mapped.fillna(False).astype(bool)

    return coerce_bool_column(col.fillna(False))


def _check_distress_gate(cat_df: pd.DataFrame) -> bool:
    """
    Returns True if the category data is dominated by fee-like keywords,
    indicating financial distress rather than genuine passion spending.
    """
    if cat_df.empty:
        return False
    # P3-5: Use total row count as denominator, not non-null remark count.
    total_rows = len(cat_df)
    remarks = cat_df[Col.CLEANED_REMARKS].fillna("").astype(str)
    # Step 4c: Exclude education-exempt remarks from fee counting.
    # "tuition fee", "course fee", "exam fee" etc. contain fee/charge keywords but
    # represent legitimate passion spending, not financial distress.
    non_exempt = remarks[~remarks.str.contains(_EDUCATION_EXEMPTION_PATTERN, regex=True, na=False)]
    fee_count = non_exempt.str.contains(_FEE_PATTERN, regex=True, na=False).sum()
    ratio = fee_count / total_rows if total_rows > 0 else 0.0
    return bool(ratio > DISTRESS_FEES_THRESHOLD)


def _check_anomaly_suppression(cat_df: pd.DataFrame) -> bool:
    # Fix 17: Use _safe_coerce_anomaly instead of coerce_bool_column.
    # coerce_bool_column does not handle string variants ('1', 'true', 'yes')
    # that appear after CSV round-trips or upstream serialization.
    if Col.IS_ANOMALY not in cat_df.columns:
        return False
    anomaly_series = _safe_coerce_anomaly(cat_df[Col.IS_ANOMALY].fillna(False))
    anomaly_share = float(anomaly_series.mean())
    # I7: Use PASSION_ANOMALY_SUPPRESSION_THRESHOLD
    return anomaly_share > PASSION_ANOMALY_SUPPRESSION_THRESHOLD


def _parse_dates_safe(series: pd.Series) -> pd.Series:
    """
    Shared local helper for date parsing.
    Handles datetime, integer epochs (using unit="s"), and string/others.
    Ensures output is localized/converted to UTC.
    """
    try:
        if pd.api.types.is_datetime64_any_dtype(series):
            dates = series
            if dates.dt.tz is None:
                return dates.dt.tz_localize("UTC")
            else:
                return dates.dt.tz_convert("UTC")
        if pd.api.types.is_integer_dtype(series) or pd.api.types.is_numeric_dtype(series):
            return pd.to_datetime(series, unit="s", errors="coerce", utc=True)
        return pd.to_datetime(series, errors="coerce", utc=True)
    except (ValueError, TypeError, OverflowError, pd.errors.OutOfBoundsDatetime):
        return pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns, UTC]")


def _is_non_declining(cat_df: pd.DataFrame, window: int = PASSION_MIN_MONTHS) -> bool:
    """
    Checks whether the monthly spend trend is non-declining.
    D4: Missing purchase months are absent observations, not zero-spend.
    No resample. No fillna(0). Active transaction months only.

    C4: The integer-epoch-second dtype guard lives HERE, not upstream.
    _is_non_declining is reachable from direct API tests and standalone callers
    who pass raw epoch-second integers without any prior pipeline normalization.
    pd.to_datetime(integers) interprets them as nanoseconds by default, which
    maps year-2023 epoch seconds to year-1970, breaking monthly bucketing.
    Use unit="s" when the column dtype is integral.
    """
    dates = _parse_dates_safe(cat_df[Col.DATE])
    if dates.isna().all():
        logger.warning("_is_non_declining: DATE column parse failed, treating as declining")
        return False


    amounts = cat_df[Col.AMOUNT].replace([np.inf, -np.inf], np.nan).map(
        lambda v: safe_numeric(v, default=np.nan)
    ).astype(float)

    temp = pd.DataFrame({"date": dates, "amount": amounts}).dropna(subset=["date", "amount"])
    if temp.empty:
        return False

    # P3-3: Convert to tz-naive before to_period to avoid pandas 3.x issues
    temp["_month"] = temp["date"].dt.tz_convert(None).dt.to_period("M")
    monthly = temp.groupby("_month")["amount"].sum()

    if monthly.sum() <= 0:
        return False

    # PASSION_MIN_MONTHS check happens before this function is called.
    if len(monthly) < 2:
        return False

    diffs = monthly.diff().dropna()
    return bool((diffs >= 0).all())


def detect_passions(
    df: pd.DataFrame,
    spend_mask: pd.Series,
    resolved_merchants: pd.Series | None = None,
) -> list[PassionSignal]:
    if not isinstance(spend_mask, pd.Series):
        raise TypeError(f"spend_mask must be pd.Series, got {type(spend_mask)}")
    # I2: assert_columns_exist for required columns
    assert_columns_exist(
        df,
        [Col.PREDICTED_CATEGORY, Col.CLEANED_REMARKS, Col.AMOUNT, Col.DATE],
        "detect_passions",
    )

    spend_mask = sanitize_mask(spend_mask, df.index, "detect_passions")

    # P3-6: Exclude NA/blank PREDICTED_CATEGORY before groupby.
    na_mask = (
        df[Col.PREDICTED_CATEGORY].isna() |
        (df[Col.PREDICTED_CATEGORY].astype(str).str.strip() == "")
    )
    na_count = int(na_mask.sum())
    if na_count > 0:
        logger.info("passion_skipped_uncategorized", extra={"count": na_count})

    # I6: Exclude invalid amounts from eligibility (do not drop from df completely)
    amounts_raw = df[Col.AMOUNT].replace([np.inf, -np.inf], np.nan)
    numeric_amounts = amounts_raw.map(
        lambda v: safe_numeric(v, default=np.nan)
    ).astype(float)
    valid_amount_mask = numeric_amounts.notna()

    spend_df = df.loc[spend_mask & valid_amount_mask & ~na_mask]
    if spend_df.empty:
        return []

    # I9: deterministic, total_spend calculated over valid rows excluding NA categories
    total_spend = numeric_amounts[spend_mask & valid_amount_mask & ~na_mask].sum()
    if total_spend <= 0:
        return []

    # Step 4a: Ensure INFERRED_SUBCATEGORY column is present with a safe default.
    # This guards against callers that built spend_df without running enrich_subcategories.
    # Single copy — branch on column presence, not twice.
    spend_df = spend_df.copy()
    if Col.INFERRED_SUBCATEGORY not in spend_df.columns:
        spend_df[Col.INFERRED_SUBCATEGORY] = ""
    else:
        # fillna → astype(str) converts pd.NA to the string "nan".
        # The subsequent .replace({"nan": ""}) cleans those up.
        # Order matters: fillna first, then astype, then strip, then replace.
        spend_df[Col.INFERRED_SUBCATEGORY] = (
            spend_df[Col.INFERRED_SUBCATEGORY]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
            .replace({"nan": ""})
        )

    signals: list[PassionSignal] = []

    # Pre-compute per-category spend totals for subcategory-relative share calculation.
    # When a subcategory group is evaluated, spend_share is computed against the parent
    # category total (not global total) — otherwise each subcategory fragment appears tiny.
    # e.g. Myntra (fashion) = 100% of shopping spend, not 13% of all spend.
    category_spend_totals: dict[str, float] = (
        numeric_amounts.groupby(spend_df[Col.PREDICTED_CATEGORY]).sum().to_dict()
    )

    for (category, subcategory), cat_df in spend_df.groupby(
        [Col.PREDICTED_CATEGORY, Col.INFERRED_SUBCATEGORY],
        dropna=False,
    ):
        subcategory = (
            ""
            if (subcategory is None or str(subcategory).strip().lower() in ("nan", "none", "<na>", ""))
            else str(subcategory).strip().lower()
        )
        if not isinstance(category, str) or not category.strip():
            continue
        # Block structural categories that can never be lifestyle passions.
        if category.strip().lower() in _BLOCKED_CATEGORIES:
            continue

        # I5: Use resolved_merchants if provided, else use CLEANED_REMARKS.
        # FIX 14 contract: resolved_merchants may contain NaN for rows outside spend_mask.
        # Inside detect_passions, we only read resolved_merchants for rows included by spend_mask/category group (cat_df.index is fully contained in spend_mask).
        if resolved_merchants is not None:
            merchants = resolved_merchants.loc[cat_df.index]
        else:
            merchants = cat_df[Col.CLEANED_REMARKS]

        # Drop/ignore NaN merchants and structural non-passion merchant patterns.
        # _NON_PASSION_MERCHANT_PATTERN filters rent, landlord, maid etc. regardless of
        # predicted category (the ML model can mis-assign rent to education/food/shopping).
        unique_merchants = tuple(sorted(set(
            str(m) for m in merchants
            if (not _safe_isna(m) and str(m).strip() != "" and str(m).lower() != "nan"
                and not _NON_PASSION_MERCHANT_PATTERN.search(str(m)))
        )))

        # Step 4b: Subcategory-aware merchant count gate.
        # Uses _SUBCATEGORY_MIN (imported with fallback) — safe if D1 not yet applied.
        effective_min = _SUBCATEGORY_MIN.get(
            subcategory, PASSION_MERCHANT_COUNT_MIN
        ) if subcategory else PASSION_MERCHANT_COUNT_MIN
        if len(unique_merchants) < effective_min:
            continue

        cat_spend = numeric_amounts.loc[cat_df.index].sum()
        if cat_spend <= 0:
            continue

        # Gate uses parent-category share for subcategory groups to prevent fragmentation dilution.
        # e.g. Myntra (fashion) = 35% of shopping spend → valid passion even if only 13% of total.
        # Display always uses global total_spend so reported percentages are honest and comparable.
        if subcategory:
            parent_total = category_spend_totals.get(str(category), total_spend)
            gate_denominator = parent_total if parent_total > 0 else total_spend
        else:
            gate_denominator = total_spend
        gate_share = cat_spend / gate_denominator if gate_denominator > 0 else 0.0
        # Display share: always relative to global total spend for honest user-facing numbers.
        display_share = cat_spend / total_spend if total_spend > 0 else 0.0

        if gate_share < PASSION_SPEND_SHARE_THRESHOLD:
            continue

        # FIX 10 & 19: Calculate dates, active_months, latest_ts, and original_index FIRST
        # Fix #5: Narrow exception scope.
        # The outer except covers fundamental date-parse failures (resets all three).
        # The inner except covers only get_indexer label lookup failure (resets original_index only).
        # This prevents a MultiIndex mismatch from silently zeroing out active_months/latest_ts
        # that were already computed correctly from valid_dates.
        try:
            dates = _parse_dates_safe(cat_df[Col.DATE])
            valid_amounts = numeric_amounts.loc[cat_df.index].notna()
            valid_dates_mask = dates.notna() & valid_amounts
            valid_dates = dates[valid_dates_mask]

            if valid_dates.empty:
                active_months = 0
                latest_ts = 0
                original_index = 0
            else:
                valid_dates_naive = valid_dates.dt.tz_convert(None)
                active_months = valid_dates_naive.dt.to_period("M").nunique()
                latest_ts = int(valid_dates.max().timestamp())
                # Use the original index of the max date row for deterministic sort.
                # Fix #5: Narrow try/except — only original_index resets on get_indexer failure.
                latest_idx_label = valid_dates.idxmax()
                try:
                    pos = df.index.get_indexer([latest_idx_label])[0]
                    original_index = int(pos) if pos >= 0 else 0
                except Exception:
                    original_index = 0
        except Exception:
            active_months = 0
            latest_ts = 0
            original_index = 0

        if _check_distress_gate(cat_df):
            signals.append(PassionSignal(
                category=str(category),
                merchant_list=unique_merchants,
                total_spend=float(cat_spend),
                merchant_count=len(unique_merchants),
                spend_share=float(display_share),
                trend_direction="suppressed",
                is_suppressed=True,
                suppression_reason="distress_gate",
                latest_ts=latest_ts,
                original_index=original_index,
                active_months=active_months,
                subcategory=subcategory,
            ))
            continue

        if _check_anomaly_suppression(cat_df):
            signals.append(PassionSignal(
                category=str(category),
                merchant_list=unique_merchants,
                total_spend=float(cat_spend),
                merchant_count=len(unique_merchants),
                spend_share=float(display_share),
                trend_direction="suppressed",
                is_suppressed=True,
                suppression_reason="anomaly_suppression",
                latest_ts=latest_ts,
                original_index=original_index,
                active_months=active_months,
                subcategory=subcategory,
            ))
            continue

        # I4: One or two active months returns "insufficient_history"
        if active_months < PASSION_MIN_MONTHS:
            trend = "insufficient_history"
        else:
            trend = "non_declining" if _is_non_declining(cat_df) else "declining"

        signals.append(PassionSignal(
            category=str(category),
            merchant_list=unique_merchants,
            total_spend=float(cat_spend),
            merchant_count=len(unique_merchants),
            spend_share=float(display_share),
            trend_direction=trend,
            is_suppressed=False,
            latest_ts=latest_ts,
            original_index=original_index,
            active_months=active_months,
            subcategory=subcategory,
        ))

    return signals
