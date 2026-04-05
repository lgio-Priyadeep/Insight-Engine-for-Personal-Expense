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
import logging
from typing import Tuple

import pandas as pd
import numpy as np

from config import NOISE_TOKENS, MERCHANT_ALIASES
from schema import Col, require_columns

logger = logging.getLogger(__name__)

# ── Schema ────────────────────────────────────────────────────────────────────

REQUIRED_COLUMNS: set[str] = {
    "date", "amount", "amount_flag", "remarks"
}

# Regex: any run of 4+ digits (accounts, phones, UPI)
_LONG_DIGIT_PATTERN = re.compile(r"\d{4,}")
# Regex: typical email patterns
_EMAIL_PATTERN = re.compile(r"\S+@\S+")
# Regex: anything that is not a letter, digit, or space
_SPECIAL_CHAR_PATTERN = re.compile(r"[^a-z0-9\s]")
# Regex: collapse multiple spaces
_MULTI_SPACE_PATTERN = re.compile(r"\s+")


# ── Validation ────────────────────────────────────────────────────────────────

def validate_schema(df: pd.DataFrame) -> None:
    """
    Assert all required columns are present.
    Raises ValueError with a clear message listing the missing columns.
    """
    require_columns(df, Col.raw_input(), "preprocessor")
    logger.info("Schema validation passed.")


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
        f"Date range: {df['date'].min().date()} → {df['date'].max().date()}"
    )
    return df


# ── Amount Flag Normalization ─────────────────────────────────────────────────

def _normalize_flag(flag) -> str:
    """
    Normalize a single amount_flag value to 'DR' or 'CR'.
    Accepts case-insensitive strings with leading/trailing whitespace.
    Returns None for invalid flags to be filtered out gracefully.
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
        logger.warning(f"Defaulting {invalid_mask.sum()} invalid amount_flag row(s) to 'DR'.")
        df.loc[invalid_mask, Col.AMOUNT_FLAG] = "DR"

    df[Col.SIGNED_AMOUNT] = df.apply(
        lambda row: -abs(row[Col.AMOUNT]) if row[Col.AMOUNT_FLAG] == "DR"
                    else abs(row[Col.AMOUNT]),
        axis=1,
    )
    return df


# ── Remarks Cleaning ──────────────────────────────────────────────────────────

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

    for pattern, alias in MERCHANT_ALIASES.items():
        if re.search(pattern, text):
            alias_lower = alias.lower()
            matched_aliases.append(alias_lower)
            if alias_lower in generics:
                generic_patterns_matched.append(pattern)
    
    if matched_aliases:
        specifics = [a for a in matched_aliases if a not in generics]
        
        # If a specific merchant was mapped (e.g. Swiggy), prioritize it over standard UPI noise!
        if specifics:
            return specifics[0]
        else:
            # ONLY generic routing tags found! (e.g., 'UPI Transfer')
            # DO NOT return! Otherwise, unmapped uniquely Indian merchants are violently overwritten.
            # Instead, dynamically strip out the generic routing text (e.g. "UPI/98293") from the string natively!
            for gp in generic_patterns_matched:
                text = re.sub(gp, " ", text)
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
        logger.warning(f"Dropped {dropped} zero-amount row(s).")
    return df


def _deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop exact duplicates on (date, amount, remarks, amount_flag).
    Keeps the first occurrence safely.
    """
    before = len(df)
    subset = [Col.DATE, Col.AMOUNT, Col.REMARKS]
    if Col.AMOUNT_FLAG in df.columns:
        subset.append(Col.AMOUNT_FLAG)
        
    df = df.drop_duplicates(subset=subset, keep="first").copy()
    dropped = before - len(df)
    if dropped:
        logger.warning(f"Dropped {dropped} duplicate transaction(s).")
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
    logger.info(f"Split → {len(debits)} debits | {len(credits)} credits")
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
    df = _parse_and_sort_dates(df)
    df = _compute_signed_amount(df)
    df = _drop_zero_amount(df)
    df = _deduplicate(df)
    df[Col.CLEANED_REMARKS] = df[Col.REMARKS].apply(clean_remark)

    debits, credits = _split_debit_credit(df)

    # Sanity log
    logger.info(
        f"Preprocessing complete. "
        f"Debits: {len(debits)} rows | Credits: {len(credits)} rows"
    )
    return debits, credits
