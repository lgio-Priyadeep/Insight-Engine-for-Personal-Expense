"""
preprocessor.py — Data Cleaning & Normalization
=================================================
Responsibilities:
  1. Schema validation
  2. Date parsing + chronological sort
  3. amount_flag normalization (DR/CR variants → uppercase)
  4. signed_amount computation (debits negative, credits positive)
  5. Remarks text cleaning (UPI IDs, noise tokens, special chars)
  6. Zero-amount row removal
  7. Duplicate detection
  8. Debit / credit split

Output: (debit_df, credit_df) — both cleaned DataFrames
"""

import re
from typing import Optional, Tuple

import pandas as pd
import numpy as np

from config import NOISE_TOKENS, MERCHANT_ALIASES
from schema import Col, require_columns, coerce_and_validate_types
from logger_factory import get_logger

logger = get_logger(__name__)

# ── Schema ────────────────────────────────────────────────────────────────────

# Regex: any run of 4+ digits (accounts, phones, UPI)
_LONG_DIGIT_PATTERN = re.compile(r"\d{4,}")
# Regex: typical email patterns
_EMAIL_PATTERN = re.compile(r"\S+@\S+")
# Regex: anything that is not a letter, digit, or space
_SPECIAL_CHAR_PATTERN = re.compile(r"[^a-z0-9\s]")
# Regex: collapse multiple spaces
_MULTI_SPACE_PATTERN = re.compile(r"\s+")

# Pre-compiled merchant alias patterns (avoids re-compilation per transaction)
_COMPILED_ALIASES: list[tuple[re.Pattern, str]] = [
    (re.compile(pattern), alias) for pattern, alias in MERCHANT_ALIASES.items()
]


# ── Validation ────────────────────────────────────────────────────────────────

def validate_schema(df: pd.DataFrame) -> None:
    """
    Assert all required columns are present.
    Raises ValueError with a clear message listing the missing columns.
    """
    require_columns(df, Col.raw_input(), "preprocessor")
    logger.info("Schema validation passed.", extra={"event_type": "schema_validation"})


# ── Date Handling ─────────────────────────────────────────────────────────────

def _parse_and_sort_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse 'date' column to datetime and sort chronologically.
    Raises on completely unparseable date columns.
    """
    df = df.copy()
    try:
        df[Col.DATE] = pd.to_datetime(df[Col.DATE], format="%Y-%m-%d")
    except Exception as exc:
        raise ValueError(f"Failed to parse 'date' column. Must be strictly ISO 8601 (YYYY-MM-DD): {exc}") from exc

    df = df.sort_values(Col.DATE).reset_index(drop=True)
    logger.info(
        "Parsed date range.",
        extra={
            "event_type": "date_parsing",
            "metrics": {
                "min_date": str(df[Col.DATE].min().date()),
                "max_date": str(df[Col.DATE].max().date())
            }
        }
    )
    return df


# ── Amount Flag Normalization ─────────────────────────────────────────────────

def _normalize_flag(flag) -> Optional[str]:
    """
    Normalize a single amount_flag value to 'DR' or 'CR'.
    Accepts case-insensitive strings with leading/trailing whitespace.

    Returns None for non-string or unrecognized flags. The caller
    (_compute_signed_amount) converts None → NaN via .apply(), then
    defaults NaN rows to 'DR' for defensive processing.
    """
    if not isinstance(flag, str):
        return None
    cleaned = flag.strip().upper()
    if cleaned not in ("DR", "CR"):
        return None
    return cleaned


def _compute_signed_amount(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply _normalize_flag to every row and derive signed_amount:
        DR  →  -abs(amount)
        CR  →  +abs(amount)
    Defaults invalid flags to 'DR' rather than dropping rows.
    """
    df = df.copy()

    df[Col.AMOUNT_FLAG] = df[Col.AMOUNT_FLAG].apply(_normalize_flag)
    invalid_mask = df[Col.AMOUNT_FLAG].isna()
    if invalid_mask.any():
        logger.warning(
            "Defaulting invalid amount_flag row(s) to 'DR'.",
            extra={
                "event_type": "flag_normalization",
                "metrics": {"invalid_count": int(invalid_mask.sum())}
            }
        )
        df.loc[invalid_mask, Col.AMOUNT_FLAG] = "DR"

    df[Col.SIGNED_AMOUNT] = np.where(
        df[Col.AMOUNT_FLAG] == "DR",
        -df[Col.AMOUNT].abs(),
        df[Col.AMOUNT].abs(),
    )
    return df


# ── Remarks Cleaning ──────────────────────────────────────────────────────────

def normalize(text: str) -> str:
    """
    Normalize text into pure alphanumeric + space, ready for regex \\b bounds.
    Strips ALL special characters (including @ and &) to guarantee safe regex matching.
    """
    if not isinstance(text, str):
        return ""
    normal = re.sub(r'[^a-z0-9]+', ' ', text.lower())
    return re.sub(r'\s+', ' ', normal).strip()

def clean_remark(remark) -> str:
    """
    Clean a single remark string:
      1. Guard against non-string / empty input
      2. Lowercase
      3. Strip UPI/NEFT reference numbers (10+ digit runs)
      4. Remove special characters
      5. Remove noise tokens
      6. Collapse whitespace
    Returns empty string if nothing meaningful remains.
    """
    if not isinstance(remark, str) or not remark.strip():
        return ""

    text = remark.lower()
    
    # ── Merchant Alias Normalisation ──
    # Map the raw Indian routing string to our explicit Regex mappings
    matched_aliases = []
    generic_patterns_matched = []
    
    # Generic routers that should be STRIPPED from string, rather than overriding the real merchant name!
    generics = {
        "upi transfer", "bank transfer", "paytm", "phonepe", "google pay",
        "bhim", "amazon pay", "cred", "mobikwik", "payzapp", "freecharge",
        "razorpay", "payu", "cashfree"
    }

    for compiled_re, alias in _COMPILED_ALIASES:
        if compiled_re.search(text):
            alias_lower = alias.lower()
            if alias_lower not in generics:
                # Specific merchant found — return immediately.
                # NOTE: relies on stable dict insertion order (Python 3.7+).
                # If MERCHANT_ALIASES is ever loaded from JSON or regenerated
                # dynamically, order must be preserved to maintain determinism.
                return alias_lower
            else:
                matched_aliases.append(alias_lower)
                generic_patterns_matched.append(compiled_re)
    
    if matched_aliases:
        # ONLY generic routing tags found! (e.g., 'UPI Transfer')
        # DO NOT return! Otherwise, unmapped uniquely Indian merchants are violently overwritten.
        # Instead, dynamically strip out the generic routing text (e.g. "UPI/98293") from the string natively!
        for gp in generic_patterns_matched:
            text = gp.sub(" ", text)
        # Text now safely falls entirely through to standard deduplication!

    # ── Standard Deduplication Fallback ──
    text = _EMAIL_PATTERN.sub(" ", text)
    text = _LONG_DIGIT_PATTERN.sub(" ", text)
    text = _SPECIAL_CHAR_PATTERN.sub(" ", text)
    text = _MULTI_SPACE_PATTERN.sub(" ", text).strip()

    # Filter out noise tokens; keep tokens with length > 1
    tokens = [
        t for t in text.split()
        if t not in NOISE_TOKENS and len(t) > 1
    ]
    return " ".join(tokens)


# ── Row-level Filtering ───────────────────────────────────────────────────────

def _drop_zero_amount(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows where the raw amount is zero (pass-through entries)."""
    before = len(df)
    df = df[df[Col.AMOUNT] != 0].copy()
    dropped = before - len(df)
    if dropped:
        logger.warning(
            "Dropped zero-amount row(s).",
            extra={
                "event_type": "drop_zero_amount",
                "metrics": {"dropped_count": dropped}
            }
        )
    return df


def _deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop exact duplicates on (date, amount, remarks, amount_flag).
    Keeps the first occurrence safely.
    """
    before = len(df)
    subset = [Col.DATE, Col.AMOUNT, Col.REMARKS, Col.AMOUNT_FLAG]
        
    df = df.drop_duplicates(subset=subset, keep="first").copy()
    dropped = before - len(df)
    if dropped:
        logger.warning(
            "Dropped duplicate transaction(s).",
            extra={
                "event_type": "deduplication",
                "metrics": {"dropped_count": dropped}
            }
        )
    return df


# ── Debit / Credit Split ──────────────────────────────────────────────────────

def _split_debit_credit(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the cleaned DataFrame into debit and credit sub-DataFrames.
    Both are reset-indexed independently.
    """
    debits  = df[df[Col.AMOUNT_FLAG] == "DR"].copy().reset_index(drop=True)
    credits = df[df[Col.AMOUNT_FLAG] == "CR"].copy().reset_index(drop=True)
    logger.info(
        "Split into debits and credits.",
        extra={
            "event_type": "split_debit_credit",
            "metrics": {"debits_count": len(debits), "credits_count": len(credits)}
        }
    )
    return debits, credits


# ── Public API ────────────────────────────────────────────────────────────────

def preprocess(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full preprocessing pipeline.

    Steps (in order):
        1. Schema validation
        2. Date parsing + chronological sort
        3. amount_flag normalization
        4. signed_amount computation
        5. Zero-amount row removal
        6. Deduplication
        7. Remarks cleaning  →  'cleaned_remarks' column
        8. Debit / credit split

    Args:
        df: Raw transaction DataFrame loaded from CSV.

    Returns:
        (debit_df, credit_df): Two independently indexed DataFrames.

    Raises:
        ValueError: on schema mismatch, bad amount_flag values, or
                    unparseable dates.
    """
    validate_schema(df)
    df = coerce_and_validate_types(df)
    df = _parse_and_sort_dates(df)
    df = _compute_signed_amount(df)
    df = _drop_zero_amount(df)
    df = _deduplicate(df)
    df[Col.CLEANED_REMARKS] = df[Col.REMARKS].apply(clean_remark)

    debits, credits = _split_debit_credit(df)

    # Sanity log
    logger.info(
        "Preprocessing complete.",
        extra={
            "event_type": "preprocessing_complete",
            "metrics": {"debits_count": len(debits), "credits_count": len(credits)}
        }
    )
    return debits, credits
