import re
import pandas as pd
import numpy as np
from collections import defaultdict
from schema import Col
from config_passion import (
    PASSION_MERCHANT_ALIASES,
    GENERALIST_CANONICALS,
    MARKETPLACE_HIGH_AMOUNT_THRESHOLD,
    MARKETPLACE_HIGH_CONFIDENCE,
    MARKETPLACE_LOW_CONFIDENCE,
    ELECTRONICS_ALLOWED_CATEGORIES,
)
from passion_utils import assert_columns_exist, safe_numeric, coerce_bool_column
from logger_factory import get_logger

__all__ = ["resolve_merchant_vectorized", "enrich_subcategories"]

logger = get_logger(__name__)

# P3-2: Derive categories directly from config_passion.py to remove DEBIT_CATEGORIES assumption

from types import MappingProxyType

# Fix #9: Build canonical lookup from PASSION_MERCHANT_ALIASES.
# resolve_merchant_vectorized returns the canonical name DIRECTLY (not a rewritten phrase).
# Use (?<!\w)/(?!\w) lookaround boundaries — \b breaks on special-char adjacent inputs like |amzn|.
_ALIAS_TO_CANONICAL: dict[str, str] = {
    str(alias).strip().lower(): str(canonical).strip().lower()
    for alias, canonical in PASSION_MERCHANT_ALIASES.items()
}
_ALIAS_PATTERN = re.compile(
    r"(?<!\w)("
    + "|".join(re.escape(a) for a in sorted(_ALIAS_TO_CANONICAL, key=len, reverse=True))
    + r")(?!\w)",
    re.IGNORECASE,
) if _ALIAS_TO_CANONICAL else None


def resolve_merchant_vectorized(series: pd.Series) -> pd.Series:
    """Map merchant descriptions to canonical names.

    Fix #9: Returns the canonical name directly — not a rewritten phrase with the alias
    substituted in context. If no alias matches, returns the normalised lowercased input.
    NaN / empty inputs return empty string "".
    """
    if not isinstance(series, pd.Series):
        raise TypeError(
            f"resolve_merchant_vectorized expects pd.Series, got {type(series).__name__}"
        )

    def _resolve(value: object) -> str:
        if pd.isna(value):
            return ""
        text = re.sub(r"\s+", " ", str(value).lower()).strip()
        if not text:
            return ""
        # Exact match first (O(1), cheapest path)
        canonical = _ALIAS_TO_CANONICAL.get(text)
        if canonical is not None:
            return canonical
        # Longest-match regex path
        if _ALIAS_PATTERN:
            m = _ALIAS_PATTERN.search(text)
            if m:
                return _ALIAS_TO_CANONICAL[m.group(1).lower()]
        return text

    return series.map(_resolve)


def enrich_subcategories(df: pd.DataFrame) -> pd.DataFrame:
    assert_columns_exist(df, [Col.AMOUNT, Col.CLEANED_REMARKS, Col.PREDICTED_CATEGORY], "enrich_subcategories")
    result = df.copy()

    # G1: Output adds ONLY INFERRED_SUBCATEGORY and SUBCATEGORY_CONFIDENCE.
    # G4: Unknown classification gets 0.0.
    result[Col.INFERRED_SUBCATEGORY] = pd.Series(
        [pd.NA] * len(result),
        index=result.index,
        dtype="object",
    )
    result[Col.SUBCATEGORY_CONFIDENCE] = pd.Series(
        0.0,
        index=result.index,
        dtype="float64",
    )

    # G2: Skip known-person rows
    if Col.IS_KNOWN_PERSON in result.columns:
        kp_mask = coerce_bool_column(result[Col.IS_KNOWN_PERSON].fillna(False))
    else:
        kp_mask = pd.Series(False, index=result.index)

    # G3: Invalid or unparseable amount rows get NA/0.0
    amounts_raw = result[Col.AMOUNT].replace([np.inf, -np.inf], np.nan)
    numeric_amounts = amounts_raw.map(
        lambda v: safe_numeric(v, default=np.nan)
    ).astype(float)
    valid_amount_mask = numeric_amounts.notna()

    eligible_mask = ~kp_mask & valid_amount_mask

    # Process only eligible rows
    if not eligible_mask.any():
        return result

    # P3-2: Resolve merchants first via resolve_merchant_vectorized, then
    # check resolved canonical against GENERALIST_CANONICALS.
    resolved = resolve_merchant_vectorized(result.loc[eligible_mask, Col.CLEANED_REMARKS])
    is_generalist = resolved.isin(GENERALIST_CANONICALS)

    if not is_generalist.any():
        return result

    # P3-2: Category-aware enrichment. Check PREDICTED_CATEGORY against
    # allowed set before assigning electronics subcategory.
    eligible_amounts = numeric_amounts[eligible_mask]

    for idx in is_generalist[is_generalist].index:
        canonical = resolved[idx]
        if canonical not in GENERALIST_CANONICALS:
            continue
        # P3-2: Gate on PREDICTED_CATEGORY
        pred_cat = ""
        if Col.PREDICTED_CATEGORY in result.columns:
            raw_cat = result.at[idx, Col.PREDICTED_CATEGORY]
            pred_cat = str(raw_cat).strip().lower() if pd.notna(raw_cat) and str(raw_cat).strip() else ""

        amount = eligible_amounts.get(idx, 0.0)
        if pred_cat in ELECTRONICS_ALLOWED_CATEGORIES and amount >= MARKETPLACE_HIGH_AMOUNT_THRESHOLD:
            result.at[idx, Col.INFERRED_SUBCATEGORY] = "electronics"
            result.at[idx, Col.SUBCATEGORY_CONFIDENCE] = MARKETPLACE_HIGH_CONFIDENCE
        else:
            result.at[idx, Col.INFERRED_SUBCATEGORY] = "general_purchase"
            result.at[idx, Col.SUBCATEGORY_CONFIDENCE] = MARKETPLACE_LOW_CONFIDENCE

    # Blocker 5: Prevent invalid/known rows from receiving subcategory enrichment.
    ineligible_idx = result[~eligible_mask].index
    if not ineligible_idx.empty:
        result.loc[ineligible_idx, Col.INFERRED_SUBCATEGORY] = pd.NA
        result.loc[ineligible_idx, Col.SUBCATEGORY_CONFIDENCE] = 0.0

    return result
