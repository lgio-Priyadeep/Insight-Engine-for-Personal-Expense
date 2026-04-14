"""
insight_model.py — ML Insight Ranking Module
=============================================
Loads the pre-trained LightGBM model to score and rank transactions based
on their likelihood of being a valuable insight.

Security:
    - Model files are verified via SHA-256 checksum before deserialization.
    - Path traversal attacks are prevented via canonicalization.
    - Unsigned models are refused by default.
"""

import hashlib
import logging
import os
import pickle
import warnings
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

# Expected models directory (relative to project root)
_MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


class ModelSecurityError(Exception):
    """Raised when model integrity verification fails."""
    pass


def _compute_checksum(file_path: str) -> str:
    """
    Compute SHA-256 hex digest of a file.

    Args:
        file_path: Absolute path to the file.

    Returns:
        Hex digest string.
    """
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def _verify_checksum(model_path: str, checksum_path: str) -> bool:
    """
    Verify a model file against its companion checksum file.

    Args:
        model_path:    Path to the serialized model.
        checksum_path: Path to the .sha256 checksum file.

    Returns:
        True if checksum matches.

    Raises:
        ModelSecurityError: If checksum does not match.
    """
    with open(checksum_path, "r") as f:
        expected = f.read().strip()
    actual = _compute_checksum(model_path)
    if actual != expected:
        raise ModelSecurityError(
            f"Model integrity check FAILED for '{model_path}'. "
            f"Expected SHA-256: {expected}, Got: {actual}. "
            "The model file may have been tampered with. "
            "Re-run train_and_save_models.py to regenerate."
        )
    return True


def _validate_model_path(model_path: str) -> str:
    """
    Canonicalize and validate that the model path is within the
    expected models directory. Prevents path traversal attacks.

    Args:
        model_path: Raw model path (may be relative).

    Returns:
        Canonicalized absolute path.

    Raises:
        ModelSecurityError: If path escapes the models directory.
    """
    canonical = os.path.realpath(model_path)
    models_dir = os.path.realpath(_MODELS_DIR)
    if not canonical.startswith(models_dir + os.sep) and canonical != models_dir:
        raise ModelSecurityError(
            f"Model path '{model_path}' resolves to '{canonical}' which is "
            f"outside the allowed models directory '{models_dir}'. "
            "This may be a path traversal attack."
        )
    return canonical


def load_insight_ranker(model_path: str = "models/insight_ranker.pkl") -> Optional[Pipeline]:
    """
    Loads the pre-trained LightGBM Insight Ranker from disk.

    Security checks performed before deserialization:
        1. Path canonicalization (prevents symlink/traversal attacks)
        2. SHA-256 checksum verification (prevents tampering)

    Returns None if the file is not found (allowing graceful degradation).
    Returns None if the checksum file is missing (unsigned model).
    Raises ModelSecurityError if checksum verification fails.
    """
    if not os.path.exists(model_path):
        logger.warning(f"Insight ranker model not found at '{model_path}'. "
                       "Run train_and_save_models.py to generate it if required.")
        return None

    # Security: validate path is within models directory
    canonical_path = _validate_model_path(model_path)

    # Security: verify checksum before deserialization
    checksum_path = canonical_path + ".sha256"
    if not os.path.exists(checksum_path):
        logger.warning(
            f"Checksum file not found at '{checksum_path}'. "
            "Refusing to load unsigned model. "
            "Re-run train_and_save_models.py to generate a signed model."
        )
        return None

    try:
        _verify_checksum(canonical_path, checksum_path)
    except ModelSecurityError as e:
        logger.error(str(e), extra={"event_type": "model_load_failure", "reason": "checksum mismatch"})
        raise
        
    logger.info(f"Model integrity verified (SHA-256 match) for {canonical_path}")

    try:
        with open(canonical_path, "rb") as f:
            pipeline = pickle.load(f)
        logger.info(f"Loaded Insight Ranker from {canonical_path}")
        return pipeline
    except Exception as e:
        logger.error(f"Failed to load Insight Ranker: {e}", extra={"event_type": "model_load_failure", "reason": "load crash"})
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
            
    # Structurally enforce expected column order if pipeline exposes it
    if hasattr(pipeline, "feature_names_in_"):
        X = X[list(pipeline.feature_names_in_)]
    else:
        logger.warning(
            "Insight ranker pipeline does not expose feature_names_in_. "
            "Proceeding with default column ordering, which may be fragile."
        )

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
