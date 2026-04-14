"""
pipeline.py — Central Orchestrator
====================================
Defines the complete Insight Engine pipeline as a single, importable
entry point. All module wiring is centralised here — no test file
should need to know the order of operations.

Usage:
    from pipeline import run_pipeline, PipelineResult

    result = run_pipeline(raw_df)
    for insight in result.insights:
        print(insight)
"""

from logger_factory import get_logger, generate_new_run_id
from model_state import InsightModelState
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import pandas as pd
from sklearn.pipeline import Pipeline as SklearnPipeline

from schema import Col
from preprocessor import preprocess
from feature_engineer import engineer_features, engineer_features_inference
from seed_labeler import label_debits, label_credits
from categorization_model import train_categorization_model, predict_categories
from expected_spend_model import train_expected_spend_model, predict_expected_spend
from anomaly_detector import detect_anomalies
from recurring_detector import find_recurring_transactions
from insight_model import load_insight_ranker, predict_insight_scores
from insight_generator import generate_human_insights

logger = get_logger(__name__)


@dataclass(frozen=True)
class PipelineResult:
    """
    Immutable container for the full pipeline output.
    """
    debits: pd.DataFrame
    credits: pd.DataFrame
    insights: List[str] = field(default_factory=list)
    cat_pipeline: Optional[SklearnPipeline] = None
    spend_pipeline: Optional[SklearnPipeline] = None
    ranker_pipeline: Optional[SklearnPipeline] = None
    global_mean: float = 0.0
    global_std: float = 0.0

    def replace(self, **kwargs) -> "PipelineResult":
        import dataclasses
        return dataclasses.replace(self, **kwargs)


def finalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure explicitly required float constraints map consistently post analysis"""
    df = df.copy()
    if Col.RECURRING_SCORE in df.columns:
        df[Col.RECURRING_SCORE] = df[Col.RECURRING_SCORE].fillna(0.0)
    return df

def _optimize_memory_footprint(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast flag variables safely AFTER the ML pipeline finishes predicting."""
    df = df.copy()
    if Col.PREDICTED_CATEGORY in df.columns:
        df[Col.PREDICTED_CATEGORY] = df[Col.PREDICTED_CATEGORY].astype('category')
    if Col.IS_WEEKEND in df.columns:
        df[Col.IS_WEEKEND] = df[Col.IS_WEEKEND].astype(bool)
    return df


def train_models(
    debits: pd.DataFrame, 
    label_col: str, 
    target_col: str
) -> InsightModelState:
    cat_pipeline = train_categorization_model(debits, label_col=label_col)
    debits = predict_categories(cat_pipeline, debits)
    spend_pipeline = train_expected_spend_model(debits, target_col=target_col)
    ranker_pipeline = load_insight_ranker()
    return InsightModelState(
        pipeline_version="1.0.0",
        cat_pipeline=cat_pipeline,
        spend_pipeline=spend_pipeline,
        ranker_pipeline=ranker_pipeline,
        global_mean=debits[Col.SIGNED_AMOUNT].mean(),
        global_std=debits[Col.SIGNED_AMOUNT].std(),
    )


def run_pipeline(
    raw_df: pd.DataFrame,
    zscore_threshold: float = 3.0,
    pct_dev_threshold: float = 0.5,
    label_col: str = Col.PSEUDO_LABEL,
    target_col: str = Col.AMOUNT,
    state: Optional[InsightModelState] = None,
) -> PipelineResult:
    generate_new_run_id()
    
    logger.info("=" * 60)
    logger.info("  INSIGHT ENGINE — Pipeline Start", extra={"event_type": "pipeline_start"})
    logger.info("=" * 60)
    logger.info(
        "Pipeline mode initialized", 
        extra={"event_type": "pipeline_mode", "metrics": {"mode": "training+inference" if state is None else "inference-only"}}
    )

    # ── PHASE 1: Preprocessing ────────────────────────────────────────────
    logger.info("[Phase 1] Preprocessing...", extra={"event_type": "phase_start", "metrics": {"phase": 1}})
    debits, credits = preprocess(raw_df)

    # ── PHASE 2: Seed Labeling ────────────────────────────────────────────
    logger.info("[Phase 2] Seed labeling...", extra={"event_type": "phase_start", "metrics": {"phase": 2}})
    debits = label_debits(debits, label_col=label_col)
    credits = label_credits(credits, label_col=label_col)

    # ── PHASE 3: Feature Engineering ──────────────────────────────────────
    logger.info("[Phase 3] Feature engineering...", extra={"event_type": "phase_start", "metrics": {"phase": 3}})
    global_mean = debits[Col.SIGNED_AMOUNT].mean()
    global_std = debits[Col.SIGNED_AMOUNT].std()

    debits = engineer_features(
        debits,
        global_mean=global_mean,
        global_std=global_std,
        amount_col=target_col,
    )

    # ── PHASE 4: ML Models ────────────────────────────────────────────────
    logger.info("[Phase 4] Machine Learning Models...", extra={"event_type": "phase_start", "metrics": {"phase": 4}})
    if state is None:
        logger.info(f"Training models explicitly on {len(debits)} rows.", extra={"event_type": "training_triggered", "metrics": {"row_count": len(debits)}})
        new_state = train_models(debits, label_col, target_col)
        cat_pipeline, spend_pipeline, ranker_pipeline = new_state.cat_pipeline, new_state.spend_pipeline, new_state.ranker_pipeline
    else:
        cat_pipeline, spend_pipeline, ranker_pipeline = state.cat_pipeline, state.spend_pipeline, state.ranker_pipeline

    debits = predict_categories(cat_pipeline, debits)
    debits = predict_expected_spend(spend_pipeline, debits)

    # ── PHASE 5: Signal Detection ─────────────────────────────────────────
    logger.info("[Phase 5] Signal detection...", extra={"event_type": "phase_start", "metrics": {"phase": 5}})
    debits = detect_anomalies(
        debits,
        zscore_threshold=zscore_threshold,
        pct_dev_threshold=pct_dev_threshold,
    )
    debits = find_recurring_transactions(debits, group_col=Col.CLEANED_REMARKS)

    # ── PHASE 5.5: ML Insight Ranking ─────────────────────────────────────
    logger.info("[Phase 5.5] Ranking candidate insights...", extra={"event_type": "phase_start", "metrics": {"phase": "5.5"}})
    debits = predict_insight_scores(ranker_pipeline, debits)

    # ── PHASE 6: Insight Generation ───────────────────────────────────────
    logger.info("[Phase 6] Generating insights...", extra={"event_type": "phase_start", "metrics": {"phase": 6}})
    debits = finalize_df(debits)
    credits = finalize_df(credits)
    
    # Safely downcast flag variables to optimize memory before giving results
    debits = _optimize_memory_footprint(debits)
    credits = _optimize_memory_footprint(credits)
    
    insights = generate_human_insights(debits)

    logger.info("=" * 60)
    logger.info(f"  Pipeline complete. Generated {len(insights)} insights.", extra={"event_type": "pipeline_complete", "metrics": {"insights_count": len(insights)}})
    logger.info("=" * 60)

    return PipelineResult(
        debits=debits,
        credits=credits,
        insights=insights,
        cat_pipeline=cat_pipeline,
        spend_pipeline=spend_pipeline,
        ranker_pipeline=ranker_pipeline,
        global_mean=global_mean,
        global_std=global_std,
    )


def run_inference(
    new_txn: pd.DataFrame,
    state: InsightModelState,
    history_df: pd.DataFrame,
    zscore_threshold: float = 3.0,
    pct_dev_threshold: float = 0.5,
) -> PipelineResult:
    """
    Run inference on new transaction(s) using pre-trained models from a
    prior ``run_pipeline`` result.

    Unlike ``run_pipeline``, this function correctly stitches new
    transactions against historical data via ``engineer_features_inference``
    so that rolling features reflect the user's real spending history.

    PERFORMANCE WARNING:
    This function recomputes historical rolling windows across `history_df`. 
    Calling this sequentially in a tight loop for single-row DataFrames will 
    bottleneck the system. Always pass a batch (e.g., a multi-row DataFrame) 
    to `new_txn` for optimal rate efficiency instead of iterating.

    Args:
        new_txn:            Raw DataFrame matching the schema's raw_input() contract.
        state:              A previous InsightModelState.
        history_df:         Historical baseline DataFrame (previously processed).
        zscore_threshold:   Z-score gate for anomaly detection.
        pct_dev_threshold:  Percent-deviation gate for anomaly detection.
    """
    generate_new_run_id()
    logger.info("Running inference on new transaction(s)...", extra={"event_type": "pipeline_inference_start"})

    # ── Phase 1: Preprocess ──
    debits, credits = preprocess(new_txn)
    debits = label_debits(debits, label_col=Col.PSEUDO_LABEL)
    credits = label_credits(credits, label_col=Col.PSEUDO_LABEL)

    # ── Phase 2: Feature Engineering (history-aware) ──
    debits = engineer_features_inference(
        debits,
        history_df=history_df,
        global_mean=state.global_mean,
        global_std=state.global_std,
    )

    # ── Phase 3: ML Prediction (pre-trained models) ──
    cat_pipeline, spend_pipeline, ranker_pipeline = (
        state.cat_pipeline, state.spend_pipeline, state.ranker_pipeline,
    )
    debits = predict_categories(cat_pipeline, debits)
    debits = predict_expected_spend(spend_pipeline, debits)

    # ── Phase 4: Signal Detection ──
    debits = detect_anomalies(
        debits,
        zscore_threshold=zscore_threshold,
        pct_dev_threshold=pct_dev_threshold,
    )
    debits = find_recurring_transactions(debits, group_col=Col.CLEANED_REMARKS)

    # ── Phase 5: ML Insight Ranking ──
    debits = predict_insight_scores(ranker_pipeline, debits)

    # ── Phase 6: Finalize + Insight Generation ──
    debits = finalize_df(debits)
    credits = finalize_df(credits)
    
    debits = _optimize_memory_footprint(debits)
    credits = _optimize_memory_footprint(credits)
    
    insights = generate_human_insights(debits)

    return PipelineResult(
        debits=debits,
        credits=credits,
        insights=insights,
        cat_pipeline=cat_pipeline,
        spend_pipeline=spend_pipeline,
        ranker_pipeline=ranker_pipeline,
        global_mean=state.global_mean,
        global_std=state.global_std,
    )
