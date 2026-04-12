"""
train_and_save_models.py
========================
Trains the LightGBM Insight Ranker on synthetic benchmark data
and serialises it to disk for offline inference in the main pipeline.
"""

import os
import pickle
import logging

from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from training_data_generator import generate_insight_dataset
from insight_model import _compute_checksum
from schema import Col

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

NUMERIC_FEATURES = [
    Col.AMOUNT, Col.AMOUNT_ZSCORE, Col.PERCENT_DEVIATION, Col.CATEGORY_CONFIDENCE,
    Col.IS_ANOMALY, Col.IS_RECURRING, Col.IS_WEEKEND,
    Col.ROLLING_7D_MEAN, Col.ROLLING_30D_MEAN, Col.ROLLING_7D_STD,
    Col.MONTH_SIN, Col.MONTH_COS, Col.AMOUNT_LOG,
]
CATEGORICAL_FEATURES = [Col.PREDICTED_CATEGORY]

def train_and_save():
    logger.info("Generating synthetic training data...")
    X_train, X_test, y_train, y_test = generate_insight_dataset(
        n_samples=5000, n_edge_cases=500, test_size=0.1, random_state=42
    )
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LGBMClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            class_weight="balanced",
            random_state=42,
            verbose=-1,
            n_jobs=-1,
        )),
    ])
    
    logger.info("Training LightGBM Insight Ranker...")
    pipeline.fit(X_train, y_train["insight_type"].values)
    
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", "insight_ranker.pkl")
    
    with open(model_path, "wb") as f:
        pickle.dump(pipeline, f)
    
    # Security: write SHA-256 checksum for integrity verification on load
    checksum = _compute_checksum(model_path)
    checksum_path = model_path + ".sha256"
    with open(checksum_path, "w") as f:
        f.write(checksum)
    
    logger.info(
        f"Model saved successfully to {model_path} "
        f"({os.path.getsize(model_path) / 1024:.1f} KB) "
        f"SHA-256: {checksum}"
    )

if __name__ == "__main__":
    train_and_save()
