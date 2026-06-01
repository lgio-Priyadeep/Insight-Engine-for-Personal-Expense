"""
passion_utils.py — Shared utility functions for the Passion Detection Engine.

NOTE: assert_columns_exist raises ValueError (not KeyError) for missing columns.
This matches schema.require_columns(). All callers should catch ValueError.
"""
import re as _re
import math
import pandas as pd
import numpy as np
from decimal import Decimal
from typing import Any

# FIX H11: Use structured logger from logger_factory instead of plain logging.
from logger_factory import get_logger

__all__ = [
    "to_bool_strict", "coerce_bool_column", "sanitize_mask",
    "safe_last_nonnull", "validate_template_values", "_safe_isna",
    "assert_columns_exist", "safe_numeric",
]

logger = get_logger(__name__)


def assert_columns_exist(df: pd.DataFrame, columns: list[str], context: str) -> None:
    """
    Validates that a DataFrame contains all required columns.

    Raises:
        ValueError: If any required columns are missing.
    """
    if missing := [c for c in columns if c not in df.columns]:
        raise ValueError(f"Missing columns in {context}: {missing}")


# FIX H7: _safe_isna array truth-value crash.
# pd.isna on an array returns an array. bool() on a multi-element array raises ValueError.
def _safe_isna(val: Any) -> bool:
    try:
        result = pd.isna(val)
        if isinstance(result, (np.ndarray, pd.Series)):
            return bool(np.all(result))
        return bool(result)
    except (TypeError, ValueError):
        return False


# FIX M10 + C2: Refactored to_bool_strict with explicit np.bool_ guard.
# C2: In NumPy 2.x, np.bool_ is NO LONGER a subclass of Python bool.
# The match/case bool(): pattern therefore does not catch np.bool_ values,
# causing them to fall through to the np.generic() case, which works but is
# less explicit. Add an isinstance guard BEFORE match/case for safety across
# all NumPy versions.
def to_bool_strict(val):
    if _safe_isna(val):
        return False
    # C2: Explicit guard — must precede match/case because np.bool_ is no longer
    # a subclass of bool in NumPy 2.x.
    if isinstance(val, (bool, np.bool_)):
        return bool(val)
    match val:
        case int() if val in (0, 1):
            return bool(val)
        case float() if val in (0.0, 1.0):
            return bool(int(val))
        case np.generic():
            num = val.item()
            if num in (0, 1):
                return bool(num)
            raise ValueError(f"Cannot coerce numpy scalar {num!r} to bool")
        case _:
            raise TypeError(f"Cannot coerce {type(val).__name__} to bool")


def coerce_bool_column(col: pd.Series) -> pd.Series:
    if col.dtype == bool:
        return col
    if isinstance(col.dtype, pd.BooleanDtype):
        return col.fillna(False).astype(bool)
    if pd.api.types.is_numeric_dtype(col):
        unique_vals = set(col.dropna().unique()) - {0, 1, 0.0, 1.0}
        if unique_vals:
            raise ValueError(
                f"Numeric column contains non-boolean values: {sorted(str(v) for v in unique_vals)}"
            )
        return col.fillna(0).eq(1).astype(bool)
    if pd.api.types.is_string_dtype(col):
        raise TypeError("String column cannot be coerced to bool")
    if col.dtype == object:
        if col.dropna().apply(lambda v: isinstance(v, str)).any():
            raise TypeError("Object column contains unconvertible string values")
        try:
            return col.map(to_bool_strict).astype(bool)
        except TypeError as e:
            raise TypeError(f"Object column contains unconvertible values: {e}") from e
    return col.map(to_bool_strict).astype(bool)


# FIX L1: sanitize_mask now logs when rows are silently added/dropped
def sanitize_mask(
    mask: Any, target_index: pd.Index, context: str
) -> pd.Series:
    if not target_index.is_unique:
        raise ValueError(f"Non-unique DataFrame index in {context}")
    if isinstance(mask, pd.Series):
        if hasattr(mask, 'cat'):
            mask = mask.astype(mask.cat.categories.dtype)
        if mask.dtype == "string" or (mask.dtype == object and mask.dropna().apply(lambda v: isinstance(v, str)).any()):
            raise TypeError(f"Mask contains string values in {context}")
        if not mask.index.is_unique:
            raise ValueError(f"Mask has non-unique index in {context}")
        if len(mask) != len(target_index):
            raise ValueError(
                f"Mask length mismatch in {context}: "
                f"mask={len(mask)}, target={len(target_index)}"
            )

        dropped = mask.index.difference(target_index)
        added = target_index.difference(mask.index)
        if len(dropped) or len(added):
            logger.debug(
                "sanitize_mask index mismatch in %s: dropped=%d, added=%d (filled False)",
                context, len(dropped), len(added),
            )

        return coerce_bool_column(mask.reindex(target_index, fill_value=False))
    elif isinstance(mask, (list, np.ndarray)):
        if len(mask) != len(target_index):
            raise ValueError(
                f"Mask length mismatch in {context}: "
                f"mask={len(mask)}, target={len(target_index)}"
            )
        return coerce_bool_column(pd.Series(mask, index=target_index, dtype=object))
    else:
        raise TypeError(f"Unsupported mask type in {context}: {type(mask)}")


def safe_last_nonnull(seq: list) -> str | None:
    for val in reversed(seq):
        if _safe_isna(val):
            continue
        try:
            return str(val)
        except (TypeError, ValueError):
            continue
    return None


_ALLOWED_TEMPLATE_SCALAR_TYPES = (str, int, float, bool, Decimal, np.integer, np.floating)

def validate_template_values(values: dict[str, Any]) -> None:
    for k, v in values.items():
        if not isinstance(v, _ALLOWED_TEMPLATE_SCALAR_TYPES):
            raise TypeError(
                f"Template value '{k}' must be a scalar (str/int/float/bool/Decimal), "
                f"got {type(v).__name__}."
            )


_CURRENCY_RE = _re.compile(r'[$\€\£\¥\u20A0-\u20CF\u20B9]')
_INR_RE = _re.compile(r'^(rs\.?\s*|inr\s+)', _re.IGNORECASE)

# FIX H5: Allow custom default and raise_on_invalid.
# Previously returned 0.0 unconditionally, masking parse errors as zero-spend.
def safe_numeric(val: Any, default: float = 0.0, *, raise_on_invalid: bool = False) -> float:
    """Convert a value to float, stripping currency symbols, INR/Rs prefixes, and commas."""
    if _safe_isna(val):
        return default
    try:
        if isinstance(val, str):
            val = _INR_RE.sub("", val)
            val = _CURRENCY_RE.sub("", val).replace(",", "").strip()
        f = float(val)
    except (ValueError, TypeError):
        if raise_on_invalid:
            raise ValueError(f"Cannot parse numeric value: {val!r}")
        return default
    return default if (math.isinf(f) or math.isnan(f)) else f
