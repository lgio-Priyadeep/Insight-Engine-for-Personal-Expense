"""
expected_spend_model.py — Expected Transaction Amount Regressor
================================================================
Trains a regression model to estimate the normal expected spending
amount for a transaction, given historical context and categories.
Later, this will act as the baseline for Anomaly Detection.

Features:
  - Time attributes: is_weekend, dow_sin, dow_cos, month_sin, month_cos
  - Historic context: rolling_7d_mean, rolling_30d_mean, rolling_7d_std
  - Transaction category: predicted_category

Model:
  - RidgeCV (linear baseline to support extrapolation bounds)
"""

import logging
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import RidgeCV

from schema import Col, require_columns

logger = logging.getLogger(__name__)


def build_spend_pipeline() -> Pipeline:
    """Builds the scikit-learn pipeline for expected spend regression."""

    numeric_features = [
        Col.IS_WEEKEND, Col.MONTH_SIN, Col.MONTH_COS, Col.DOW_SIN, Col.DOW_COS,
        Col.WEEK_OF_MONTH, Col.ROLLING_7D_MEAN, Col.ROLLING_30D_MEAN, Col.ROLLING_7D_STD
    ]

    categorical_features = [Col.PREDICTED_CATEGORY]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ],
        remainder="drop"
    )

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RidgeCV(
            alphas=[0.1, 1.0, 10.0, 100.0]
        )),
    ])

    return model


def train_expected_spend_model(
    df: pd.DataFrame, 
    target_col: str = "amount"
) -> Pipeline:
    """
    Trains the regression model on the numeric/time history of transactions.
    """
    logger.info(f"Training expected spend model on {len(df)} samples...")

    # Drop missing essential columns
    # We require predicted_category to be present
    require_columns(df, Col.expected_spend_input(), "expected_spend_model")

    train_df = df.dropna(subset=[
        target_col, Col.PREDICTED_CATEGORY, Col.ROLLING_7D_MEAN, Col.AMOUNT
    ]).copy()

    if len(train_df) < len(df):
        logger.warning(
            f"Dropped {len(df) - len(train_df)} rows during training due to missing "
            "predicted_category or target/rolling feature."
        )

    # Note: Using absolute amount. If signed amounts are strictly needed, change target_col.
    y = train_df[target_col]

    pipeline = build_spend_pipeline()
    pipeline.fit(train_df, y)

    # Basic train metrics (R^2 Score)
    r2 = pipeline.score(train_df, y)
    logger.info(f"Expected spend model trained. Train R²: {r2:.3f}")

    return pipeline


def predict_expected_spend(
    pipeline: Pipeline,
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Predicts the expected amount and adds residual features to the DataFrame.
    """
    df = df.copy()

    # Build prediction-safe input without a full copy — fill NaN only on
    # the columns the model actually reads.
    pred_cols = [
        Col.IS_WEEKEND, Col.MONTH_SIN, Col.MONTH_COS, Col.DOW_SIN, Col.DOW_COS,
        Col.WEEK_OF_MONTH, Col.ROLLING_7D_MEAN, Col.ROLLING_30D_MEAN, Col.ROLLING_7D_STD
    ]
    df_pred = df[pred_cols + [Col.PREDICTED_CATEGORY]].copy()
    for col in pred_cols:
        df_pred[col] = df_pred[col].fillna(0.0)
    df_pred[Col.PREDICTED_CATEGORY] = df_pred[Col.PREDICTED_CATEGORY].fillna("uncategorized")

    # Predict expected amount
    df[Col.EXPECTED_AMOUNT] = pipeline.predict(df_pred)
    
    # Calculate residual as actual minus predicted
    if Col.AMOUNT in df.columns:
        df[Col.RESIDUAL] = df[Col.AMOUNT] - df[Col.EXPECTED_AMOUNT]
        # percent deviation: residual / |expected|
        # Using abs() + clip(lower=1.0) to prevent:
        #   1. Division by zero when expected ≈ 0
        #   2. Sign inversion when expected < 0 (possible with RidgeCV extrapolation)
        safe_expected = df[Col.EXPECTED_AMOUNT].abs().clip(lower=1.0)
        clipped_count = (df[Col.EXPECTED_AMOUNT].abs() < 1.0).sum()
        if clipped_count:
            logger.warning(
                f"{clipped_count} row(s) had |expected_amount| < 1.0; "
                "clipped to 1.0 for percent_deviation computation."
            )
        df[Col.PERCENT_DEVIATION] = df[Col.RESIDUAL] / safe_expected

    return df
