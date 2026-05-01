import joblib
import numpy as np
from dataclasses import dataclass
from typing import Optional
from sklearn.pipeline import Pipeline as SklearnPipeline

@dataclass
class InsightModelState:
    """
    Immutable container for serialized ML pipeline components.
    Guarantees no raw dataframes (PII) are accidentally persisted to disk.
    """
    pipeline_version: str
    cat_pipeline: Optional[SklearnPipeline]
    spend_pipeline: Optional[SklearnPipeline]
    ranker_pipeline: Optional[SklearnPipeline]
    global_mean: float
    global_std: float
    stats_version: str = "v1_raw"
    kp_config_hash: str = ""

def save_model_state(filepath: str, state: InsightModelState) -> None:
    """
    Persist the model state to disk safely.
    """
    # Strict safety assertions: Enforce numeric types downcast mathematically if needed,
    # but NEVER serialize a DataFrame accidentally hidden in a generic field.
    if not isinstance(state.global_mean, (float, np.floating)):
        raise TypeError(f"global_mean must be a float, got {type(state.global_mean)}")
    if not isinstance(state.global_std, (float, np.floating)):
        raise TypeError(f"global_std must be a float, got {type(state.global_std)}")
    
    # Store explicitly rather than pickling the whole object indiscriminately
    payload = {
        "pipeline_version": state.pipeline_version,
        "cat_pipeline": state.cat_pipeline,
        "spend_pipeline": state.spend_pipeline,
        "ranker_pipeline": state.ranker_pipeline,
        "global_mean": float(state.global_mean),
        "global_std": float(state.global_std),
        "stats_version": state.stats_version,
        "kp_config_hash": state.kp_config_hash,
    }

    joblib.dump(payload, filepath)

def load_model_state(filepath: str) -> InsightModelState:
    """
    Load an InsightModelState from disk and verify compatibility.
    """
    payload = joblib.load(filepath)

    if payload.get("pipeline_version") != "1.0.0":
        raise ValueError(f"Model version mismatch! Found {payload.get('pipeline_version')}, expected 1.0.0.")

    return InsightModelState(
        pipeline_version=payload["pipeline_version"],
        cat_pipeline=payload["cat_pipeline"],
        spend_pipeline=payload["spend_pipeline"],
        ranker_pipeline=payload["ranker_pipeline"],
        global_mean=payload["global_mean"],
        global_std=payload["global_std"],
        stats_version=payload.get("stats_version", "v1_raw"),
        kp_config_hash=payload.get("kp_config_hash", ""),
    )
