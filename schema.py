"""
schema.py — DataFrame Column Contract Registry
=================================================
Central source of truth for all DataFrame column names used across the
Insight Engine pipeline. Every module that reads or writes DataFrame
columns MUST reference constants from this module.

This eliminates silent breakage from column renames: change the string
once here, and every consumer picks it up via the import.

Usage:
    from schema import Col, require_columns

    # Reference a column
    df[Col.AMOUNT_ZSCORE]

    # Validate input contract
    require_columns(df, Col.REQUIRED_ANOMALY_DETECTOR, "anomaly_detector")
"""

import pandas as pd
from typing import FrozenSet
from logger_factory import get_logger

logger = get_logger(__name__)


class Col:
    """
    All DataFrame column names used across the pipeline.

    Naming convention: UPPER_SNAKE for this registry,
    values are the actual DataFrame column name strings.
    """

    # ── Raw Input Columns (from bank statement CSV) ───────────────────────
    DATE = "date"
    AMOUNT = "amount"
    AMOUNT_FLAG = "amount_flag"
    REMARKS = "remarks"
    BALANCE = "balance"

    # ── Preprocessor Output ───────────────────────────────────────────────
    SIGNED_AMOUNT = "signed_amount"
    CLEANED_REMARKS = "cleaned_remarks"

    # ── Feature Engineer Output ───────────────────────────────────────────
    DOW_SIN = "dow_sin"
    DOW_COS = "dow_cos"
    MONTH_SIN = "month_sin"
    MONTH_COS = "month_cos"
    IS_WEEKEND = "is_weekend"
    WEEK_OF_MONTH = "week_of_month"
    ROLLING_7D_MEAN = "rolling_7d_mean"
    ROLLING_7D_STD = "rolling_7d_std"
    ROLLING_30D_MEAN = "rolling_30d_mean"
    AMOUNT_LOG = "amount_log"
    AMOUNT_ZSCORE = "amount_zscore"

    # ── Phase 1.5: Known Persons Output ───────────────────────────────────
    IS_KNOWN_PERSON = "is_known_person"          # bool — drives all masking decisions
    KNOWN_PERSON_ALIAS = "known_person_alias"    # "Mom", "Self:HDFC_Savings", None
    TRANSFER_CLASS = "transfer_class"            # "transfer_self", "transfer_known", "transfer_external", None
    MATCH_SCORE = "match_score"                  # int (>= 2 for classification)
    MATCH_SIGNALS = "match_signals"              # List[str] reasoning tokens

    # ── Seed Labeler Output ───────────────────────────────────────────────
    PSEUDO_LABEL = "pseudo_label"
    LABEL_REASON = "label_reason"
    LABEL_KEYWORD = "label_keyword"
    LABEL_KEYWORD_NORM = "label_keyword_norm"
    LABEL_CONFIDENCE = "label_confidence"

    # ── Categorization Model Output ───────────────────────────────────────
    PREDICTED_CATEGORY = "predicted_category"

    # ── Expected Spend Model Output ───────────────────────────────────────
    EXPECTED_AMOUNT = "expected_amount"
    RESIDUAL = "residual"
    PERCENT_DEVIATION = "percent_deviation"

    # ── Anomaly Detector Output ───────────────────────────────────────────
    IS_ANOMALY = "is_anomaly"

    # ── Recurring Detector Output ─────────────────────────────────────────
    IS_RECURRING = "is_recurring"
    RECURRING_FREQUENCY = "recurring_frequency"
    RECURRING_CONFIDENCE = "recurring_confidence"
    RECURRING_SCORE = "recurring_score"

    # ── ML Insight Engine (benchmark / training) ──────────────────────────
    CATEGORY_CONFIDENCE = "category_confidence"
    INSIGHT_TYPE = "insight_type"
    TIP_ID = "tip_id"
    INSIGHT_SCORE = "insight_score"

    # ══════════════════════════════════════════════════════════════════════
    # Required Column Sets — Module Input Contracts
    # ══════════════════════════════════════════════════════════════════════

    @staticmethod
    def raw_input() -> FrozenSet[str]:
        """Columns required in the raw bank statement CSV."""
        return frozenset({Col.DATE, Col.AMOUNT, Col.AMOUNT_FLAG, Col.REMARKS})

    @staticmethod
    def feature_engineer_input() -> FrozenSet[str]:
        """Columns required before feature engineering."""
        return frozenset({Col.DATE, Col.AMOUNT, Col.SIGNED_AMOUNT})

    @staticmethod
    def seed_labeler_input() -> FrozenSet[str]:
        """Columns required for seed labeling."""
        return frozenset({Col.CLEANED_REMARKS})

    @staticmethod
    def categorization_model_input() -> FrozenSet[str]:
        """Columns required for categorization training/prediction."""
        return frozenset({Col.CLEANED_REMARKS, Col.AMOUNT_LOG})

    @staticmethod
    def expected_spend_input() -> FrozenSet[str]:
        """Columns required for expected spend model."""
        return frozenset({Col.AMOUNT, Col.PREDICTED_CATEGORY, Col.ROLLING_7D_MEAN})

    @classmethod
    def seed_labeler_output(cls) -> set[str]:
        return {cls.PSEUDO_LABEL, cls.LABEL_REASON, cls.LABEL_KEYWORD, cls.LABEL_KEYWORD_NORM, cls.LABEL_CONFIDENCE}

    @classmethod
    def categorization_output(cls) -> set[str]:
        return {cls.PREDICTED_CATEGORY}

    @classmethod
    def expected_spend_output(cls) -> set[str]:
        return {cls.EXPECTED_AMOUNT, cls.RESIDUAL, cls.PERCENT_DEVIATION}

    @classmethod
    def anomaly_output(cls) -> set[str]:
        return {cls.IS_ANOMALY}

    @classmethod
    def recurring_output(cls) -> set[str]:
        return {
            cls.IS_RECURRING, cls.RECURRING_FREQUENCY,
            cls.RECURRING_CONFIDENCE, cls.RECURRING_SCORE,
        }

    @classmethod
    def known_persons_input(cls) -> set[str]:
        # Required for tag_known_persons()
        return {cls.DATE, cls.AMOUNT, cls.REMARKS, cls.CLEANED_REMARKS}

    @classmethod
    def ml_output_columns(cls) -> set[str]:
        """
        All columns generated by Phase 3 through 6.
        These must be pd.NA for personal transfers.
        """
        return (
            cls.seed_labeler_output()
            | cls.categorization_output()
            | cls.expected_spend_output()
            | cls.anomaly_output()
            | cls.recurring_output()
        )

    @staticmethod
    def anomaly_detector_input() -> FrozenSet[str]:
        """Columns required for anomaly detection."""
        return frozenset({Col.AMOUNT_ZSCORE, Col.PERCENT_DEVIATION, Col.AMOUNT})

    @staticmethod
    def recurring_detector_input() -> FrozenSet[str]:
        """Columns required for recurring transaction detection."""
        return frozenset({Col.CLEANED_REMARKS, Col.DATE, Col.AMOUNT})

    @staticmethod
    def insight_generator_input() -> FrozenSet[str]:
        """Columns required for insight generation."""
        return frozenset({Col.IS_ANOMALY, Col.IS_RECURRING})

    @staticmethod
    def insight_ranker_input() -> FrozenSet[str]:
        """Columns required for training/scoring the Insight Ranker."""
        return frozenset({
            Col.AMOUNT, Col.AMOUNT_ZSCORE, Col.PERCENT_DEVIATION, Col.CATEGORY_CONFIDENCE,
            Col.IS_ANOMALY, Col.IS_RECURRING, Col.IS_WEEKEND,
            Col.ROLLING_7D_MEAN, Col.ROLLING_30D_MEAN, Col.ROLLING_7D_STD,
            Col.MONTH_SIN, Col.MONTH_COS, Col.AMOUNT_LOG, Col.PREDICTED_CATEGORY
        })


def require_columns(
    df: pd.DataFrame,
    required: FrozenSet[str],
    module_name: str,
) -> None:
    """
    Validate that a DataFrame contains all required columns.

    Args:
        df:          The DataFrame to validate.
        required:    Set of required column names.
        module_name: Name of the calling module (for error messages).

    Raises:
        ValueError: with a clear message listing missing columns.
    """
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"[{module_name}] Missing required columns: {sorted(missing)}. "
            f"Available: {sorted(df.columns)}"
        )

def coerce_and_validate_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Safely coerce input columns to their expected types.
    Strictly applies 'DROP + LOG' philosophy for unparseable garbage inputs.
    """
    df = df.copy()

    # 1. Coerce Amount to float explicitly
    if Col.AMOUNT in df.columns:
        # errors='coerce' will turn garbage strings like "100.00xyz" into NaN
        df[Col.AMOUNT] = pd.to_numeric(df[Col.AMOUNT], errors="coerce")
        
        # Validation + Drop
        invalid_mask = df[Col.AMOUNT].isna()
        if invalid_mask.any():
            dropped_count = int(invalid_mask.sum())
            logger.warning(
                "Unparseable garbage detected in amount column. Dropping rows to preserve financial precision.",
                extra={
                    "event_type": "coercion_failure_drop",
                    "metrics": {"dropped_count": dropped_count}
                }
            )
            df = df.loc[~invalid_mask].reset_index(drop=True)

    return df
