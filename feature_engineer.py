"""
feature_engineer.py — Leak-Free Feature Engineering
=====================================================
Produces the following feature groups:

  Time Features:
    is_weekend          — 1 if Saturday or Sunday, else 0
    week_of_month       — week number within the month (1–5)
    month_sin/cos       — cyclical encoding of month (prevents Jan↔Dec cliff)
    dow_sin/cos         — cyclical encoding of day-of-week

  Rolling Features  ← LEAKAGE-SAFE (shift(1) applied before .rolling())
    rolling_7d_mean     — mean of signed_amount over the past 7 rows
    rolling_30d_mean    — mean over the past 30 rows
    rolling_7d_std      — std over the past 7 rows

  Amount Features:
    amount_log          — log1p(|signed_amount|)
    amount_zscore       — (signed_amount − rolling_mean) / rolling_std
                          clipped to [−5, 5]; safe against zero-std

Leakage Prevention
──────────────────
Rolling stats are computed on  .shift(1)  of the amount column.
This guarantees row i's rolling window contains only rows 0 … i-1.
NaNs introduced by shift/short windows are filled with pre-computed
TRAINING-SET statistics (passed in as arguments, not re-derived here).
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from schema import Col, require_columns

logger = logging.getLogger(__name__)

# Clipping bound for z-score (handles extreme outliers from short windows)
ZSCORE_CLIP = 5.0


# ── Time Features ─────────────────────────────────────────────────────────────

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add calendar-based time features.
    Requires 'date' column of dtype datetime64.
    """
    df = df.copy()

    if not pd.api.types.is_datetime64_any_dtype(df[Col.DATE]):
        raise TypeError("'date' column must be datetime64. Run preprocess() first.")

    df[Col.IS_WEEKEND]    = df[Col.DATE].dt.dayofweek.isin([5, 6]).astype(int)
    df[Col.WEEK_OF_MONTH] = df[Col.DATE].apply(lambda d: (d.day - 1) // 7 + 1)

    # Cyclical month encoding (12-period cycle)
    month = df[Col.DATE].dt.month
    df[Col.MONTH_SIN] = np.sin(2 * np.pi * month / 12)
    df[Col.MONTH_COS] = np.cos(2 * np.pi * month / 12)

    # Cyclical day-of-week encoding (7-period cycle)
    dow = df[Col.DATE].dt.dayofweek
    df[Col.DOW_SIN] = np.sin(2 * np.pi * dow / 7)
    df[Col.DOW_COS] = np.cos(2 * np.pi * dow / 7)

    return df


# ── Rolling Features ──────────────────────────────────────────────────────────

def add_rolling_features(
    df: pd.DataFrame,
    amount_col: str = "amount",
) -> pd.DataFrame:
    """
    Add rolling statistics.  LEAKAGE-SAFE via shift(1).

    The series is first shifted by 1 position so that the rolling window
    for row i contains values from rows [0 … i-1] only — never the
    current row's value.

    Args:
        df:         DataFrame sorted chronologically (required).
        amount_col: Column name to compute rolling stats on.

    Returns:
        DataFrame with rolling_7d_mean, rolling_30d_mean, rolling_7d_std.
        NaNs are intentionally left for the caller to fill with training stats.
    """
    df = df.copy().sort_values(Col.DATE)

    if len(df) <= 1:
        df[Col.ROLLING_7D_MEAN] = np.nan
        df[Col.ROLLING_30D_MEAN] = np.nan
        df[Col.ROLLING_7D_STD] = np.nan
        return df

    # Shift FIRST — this is the core of the leakage prevention
    shifted = df[amount_col].shift(1)

    df[Col.ROLLING_7D_MEAN]  = shifted.rolling(window=7,  min_periods=1).mean()
    df[Col.ROLLING_30D_MEAN] = shifted.rolling(window=30, min_periods=1).mean()

    # std requires at least 2 observations; short windows remain NaN
    df[Col.ROLLING_7D_STD]   = shifted.rolling(window=7,  min_periods=2).std()

    return df


def fill_rolling_nulls(
    df: pd.DataFrame,
    global_mean: float,
    global_std: float,
) -> pd.DataFrame:
    """
    Fill NaNs introduced by rolling windows.

    IMPORTANT: global_mean and global_std MUST be computed on the
    training partition only (before any train/test split) and passed in.
    Deriving them here from `df` would cause leakage on the test set.

    Args:
        df:          DataFrame after add_rolling_features().
        global_mean: Training-set mean of signed_amount.
        global_std:  Training-set std  of signed_amount.
    """
    df = df.copy()

    for col_name in [Col.ROLLING_7D_MEAN, Col.ROLLING_30D_MEAN]:
        null_count = df[col_name].isna().sum()
        if null_count:
            df[col_name] = df[col_name].fillna(global_mean)
            logger.debug(f"Filled {null_count} NaN(s) in {col_name} with global_mean={global_mean:.2f}")

    null_count = df[Col.ROLLING_7D_STD].isna().sum()
    if null_count:
        df[Col.ROLLING_7D_STD] = df[Col.ROLLING_7D_STD].fillna(
            global_std if global_std and global_std > 0 else 1.0
        )
        logger.debug(f"Filled {null_count} NaN(s) in {Col.ROLLING_7D_STD} with global_std={global_std:.2f}")
    return df


# ── Amount Features ───────────────────────────────────────────────────────────

def add_amount_features(
    df: pd.DataFrame,
    amount_col: str = "amount",
) -> pd.DataFrame:
    """
    Add derived amount features.
    Requires rolling_7d_mean and rolling_7d_std to be present (post-fill).

    amount_log:
        log1p of absolute value — handles negatives, compresses scale.

    amount_zscore:
        (amount − rolling_mean) / rolling_std
        Clipped to [−ZSCORE_CLIP, +ZSCORE_CLIP].
        Zero-std rows are treated as std=1 to avoid ±inf.
    """
    df = df.copy()

    # amount_log: safe for negative signed amounts
    df[Col.AMOUNT_LOG] = np.log1p(df[amount_col].abs())

    # Replace zero std with 1.0 to avoid division by zero
    safe_std = df[Col.ROLLING_7D_STD].copy()
    zero_std_mask = safe_std == 0
    if zero_std_mask.any():
        logger.warning(
            f"{zero_std_mask.sum()} row(s) had rolling_7d_std=0; "
            "replaced with 1.0 for z-score computation."
        )
    safe_std = safe_std.replace(0, np.nan).fillna(1.0)

    df[Col.AMOUNT_ZSCORE] = (
        (df[amount_col] - df[Col.ROLLING_7D_MEAN]) / safe_std
    ).clip(-ZSCORE_CLIP, ZSCORE_CLIP)

    return df


# ── Public API ────────────────────────────────────────────────────────────────

def engineer_features(
    df: pd.DataFrame,
    global_mean: Optional[float] = None,
    global_std: Optional[float] = None,
    amount_col: str = "amount",
) -> pd.DataFrame:
    """
    Full feature engineering pipeline.

    For TRAINING:
        Compute global_mean / global_std from the training partition,
        then pass them here.  Never pass None for training data.

    For INFERENCE / TEST:
        Pass the training-set statistics so the test set is filled
        with the same reference values.

    Args:
        df:          Cleaned DataFrame (output of preprocess()).
        global_mean: Training-set signed_amount mean for NaN fill.
        global_std:  Training-set signed_amount std  for NaN fill.
        amount_col:  Column to engineer features from.

    Returns:
        Feature-enriched DataFrame.  Original columns are preserved.
    """
    if global_mean is None:
        global_mean = df[amount_col].mean()
        logger.warning(
            "global_mean not provided — computed on the full df passed in. "
            "This is acceptable for inference but NOT for training splits."
        )
    if global_std is None:
        global_std = df[amount_col].std()
        logger.warning(
            "global_std not provided — computed on the full df passed in. "
            "This is acceptable for inference but NOT for training splits."
        )

    # Formally enforce chronological sort at the root to prevent any downstream hidden coupling/bleed
    df = df.copy().sort_values(Col.DATE)

    df = add_time_features(df)
    df = add_rolling_features(df, amount_col=amount_col)
    df = fill_rolling_nulls(df, global_mean=global_mean, global_std=global_std)
    df = add_amount_features(df, amount_col=amount_col)

    logger.info(
        f"Feature engineering complete. "
        f"Output shape: {df.shape}. "
        f"Columns added: is_weekend, week_of_month, month_sin/cos, "
        f"dow_sin/cos, rolling_7d/30d_mean, rolling_7d_std, "
        f"amount_log, amount_zscore."
    )
    return df


def engineer_features_inference(
    new_txn: pd.DataFrame,
    history_df: pd.DataFrame,
    global_mean: float,
    global_std: float,
    amount_col: str = "amount",
) -> pd.DataFrame:
    """
    Stateless Live-Inference Entrypoint.
    Binds a new live transaction against the user's recent history to correctly
    compute rolling features. This prevents single-row inputs from defaulting
    to the global mean, treating the live transaction accurately in the context
    of their last 30 transactions.

    Uses tag-based row tracking (not tail()) to reliably extract only the
    new transactions from the combined+sorted result, even when new_txn
    contains backdated entries older than the history.

    Args:
        new_txn: DataFrame containing the new real-time transaction(s).
        history_df: DataFrame containing the user's past transactions (at least 30).
        global_mean: Training set global mean.
        global_std: Training set global std.
        amount_col: The column containing numeric target values.

    Returns:
        DataFrame enriched with features matching ONLY the length of `new_txn`.
    """
    _TAG = "__is_new_txn__"

    if history_df is None or history_df.empty:
        logger.warning("No history provided for inference; rolling features will heavily default to global_mean.")
        combined = new_txn.copy()
        combined[_TAG] = True
    else:
        hist = history_df.copy()
        new = new_txn.copy()
        hist[_TAG] = False
        new[_TAG] = True
        combined = pd.concat([hist, new], ignore_index=True)

    # engineer_features internally sorts by date — that's fine.
    # The _TAG column survives through all transformations.
    engineered = engineer_features(
        combined,
        global_mean=global_mean,
        global_std=global_std,
        amount_col=amount_col
    )

    # Extract only the new rows using the tag (not position-based tail)
    result = engineered[engineered[_TAG]].drop(columns=[_TAG]).reset_index(drop=True)

    # No cleanup needed: 'engineered' is local and goes out of scope.
    # The tag column was already stripped from 'result' above.

    return result

