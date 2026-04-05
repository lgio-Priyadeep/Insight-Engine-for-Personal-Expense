"""
categorization_model.py — ML-Based Transaction Categorization
=============================================================
Trains a classifier on the keyword-based pseudo-labels to generalize
to unseen or unstructured transaction remarks.

Features:
  - TF-IDF on `cleaned_remarks`
  - StandardScaler on `amount_log`

Model:
  - Logistic Regression (class_weight='balanced') to ensure minority
    classes (or uncategorized items) are treated fairly.
"""

import logging
import pandas as pd
from typing import Tuple

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from config import FALLBACK_DEBIT_LABEL, FALLBACK_CREDIT_LABEL
from schema import Col, require_columns

logger = logging.getLogger(__name__)


def build_categorization_pipeline() -> Pipeline:
    """Builds the scikit-learn pipeline for transaction categorization."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("text", TfidfVectorizer(ngram_range=(1, 2), max_features=2000), Col.CLEANED_REMARKS),
            # StandardScaler(with_mean=False) prevents sparse->dense explosion
            ("num", StandardScaler(with_mean=False), ["amount_log"])
        ],
        remainder="drop",
        sparse_threshold=1.0  # Force output to remain sparse to prevent OOM DDoSing
    )

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            # multi_class='multinomial' is default for LR in recent sklearn
            # if y is multiclass
        )),
    ])
    
    return model


def train_categorization_model(
    df: pd.DataFrame, 
    label_col: str = "pseudo_label"
) -> Pipeline:
    """
    Trains the categorization model.
    The input DataFrame should already have 'cleaned_remarks', 'amount_log',
    and the target 'label_col'.
    """
    logger.info(f"Training categorization model on {len(df)} samples...")
    
    # Ensure no missing values in critical columns
    require_columns(df, Col.categorization_model_input(), "categorization_model")

    train_df = df.dropna(subset=[Col.CLEANED_REMARKS, Col.AMOUNT_LOG, label_col]).copy()
    
    # Remove ground truth bias by ignoring fallback labels
    fallbacks = {FALLBACK_DEBIT_LABEL, FALLBACK_CREDIT_LABEL}
    valid_mask = ~train_df[label_col].isin(fallbacks)
    train_df = train_df[valid_mask].copy()

    if len(train_df) < len(df):
        logger.warning(
            f"Dropped {len(df) - len(train_df)} rows during training "
            "(due to missing values or being uncategorized fallbacks)."
        )

    X = train_df[[Col.CLEANED_REMARKS, Col.AMOUNT_LOG]]
    y = train_df[label_col]

    pipeline = build_categorization_pipeline()
    pipeline.fit(X, y)
    
    # Calculate simple training accuracy
    acc = pipeline.score(X, y)
    logger.info(f"Training complete. Training Accuracy: {acc:.2%}")
    
    return pipeline


def predict_categories(
    pipeline: Pipeline, 
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Predicts categories and confidence scores for a given DataFrame.
    Returns a copy of the DataFrame with 'predicted_category' and 
    'category_confidence' columns.
    """
    df = df.copy()
    
    # Handle potentially missing values before prediction
    # We fill missing strings with empty strings, and missing amounts with 0 (or median, but 0 log implies 0 amount)
    require_columns(df, Col.categorization_model_input(), "categorization_model.predict")

    X = df[[Col.CLEANED_REMARKS, Col.AMOUNT_LOG]].copy()
    X[Col.CLEANED_REMARKS] = X[Col.CLEANED_REMARKS].fillna("")
    X["amount_log"] = X["amount_log"].fillna(0.0)

    # Predict class and probabilities
    preds = pipeline.predict(X)
    probs = pipeline.predict_proba(X)
    
    # Extract confidence (max probability for the predicted class)
    confidences = probs.max(axis=1)

    df[Col.PREDICTED_CATEGORY] = preds
    df["category_confidence"] = confidences

    return df
