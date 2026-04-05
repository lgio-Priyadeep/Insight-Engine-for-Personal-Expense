"""
insight_model.py — ML Insight Ranking Module
=============================================
Loads the pre-trained LightGBM model to score and rank transactions based
on their likelihood of being a valuable insight.
"""

import logging
import os
import pickle
from typing import Optional

import pandas as pd
from sklearn.pipeline import Pipeline

from schema import Col, require_columns

logger = logging.getLogger(__name__)

# Constants defining the expected feature set
NUMERIC_FEATURES = [
    Col.AMOUNT, Col.AMOUNT_ZSCORE, Col.PERCENT_DEVIATION, Col.CATEGORY_CONFIDENCE,
    Col.IS_ANOMALY, Col.IS_RECURRING, Col.IS_WEEKEND,
    Col.ROLLING_7D_MEAN, Col.ROLLING_30D_MEAN, Col.ROLLING_7D_STD,
    Col.MONTH_SIN, Col.MONTH_COS, Col.AMOUNT_LOG,
]
CATEGORICAL_FEATURES = [Col.PREDICTED_CATEGORY]


def load_insight_ranker(model_path: str = "models/insight_ranker.pkl") -> Optional[Pipeline]:
    """
    Loads the pre-trained LightGBM Insight Ranker from disk.
    
    Returns None if the file is not found (allowing graceful degradation).
    """
    if not os.path.exists(model_path):
        logger.warning(f"Insight ranker model not found at '{model_path}'. "
                       "Run train_and_save_models.py to generate it if required.")
        return None
        
    try:
        with open(model_path, "rb") as f:
            pipeline = pickle.load(f)
        logger.info(f"Loaded Insight Ranker from {model_path}")
        return pipeline
    except Exception as e:
        logger.error(f"Failed to load Insight Ranker: {e}")
        return None


def predict_insight_scores(pipeline: Optional[Pipeline], df: pd.DataFrame) -> pd.DataFrame:
    """
    Scores the DataFrame using the loaded pipeline.
    
    Appends the `Col.INSIGHT_SCORE` column. If no pipeline is provided,
    defaults all scores to 0.0, which falls back to rule-based prioritization.
    """
    ret_df = df.copy()
    
    if pipeline is None:
        logger.warning(
            "No insight ranker pipeline provided. "
            "Insight scoring falls back to default 0.0 baseline."
        )
        ret_df[Col.INSIGHT_SCORE] = 0.0
        return ret_df
        
    require_columns(ret_df, Col.insight_ranker_input(), "insight_model")
    
    # Defensive data preparation (fills missing values for inference)
    X = ret_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    for col in NUMERIC_FEATURES:
        if X[col].isna().any():
            X[col] = X[col].fillna(0.0)
            
    for col in CATEGORICAL_FEATURES:
        if X[col].isna().any():
            X[col] = X[col].fillna("unknown")
            
    import warnings
    
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, message=".*valid feature names.*")
            probs = pipeline.predict_proba(X)
            
        classes = list(pipeline.classes_)
        
        # 'no_action' means zero insight value. We take 1.0 - P(no_action)
        if "no_action" in classes:
            idx = classes.index("no_action")
            scores = 1.0 - probs[:, idx]
        else:
            scores = probs.max(axis=1)  # Fallback
            
        ret_df[Col.INSIGHT_SCORE] = scores
        logger.debug(f"Computed Insight Scores. Mean score: {scores.mean():.3f}")
        
    except Exception as e:
        logger.error(f"Error during Insight Ranker prediction: {e}")
        ret_df[Col.INSIGHT_SCORE] = 0.0
        
    return ret_df
