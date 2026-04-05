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

import logging
from dataclasses import dataclass, field
from typing import List, Optional

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

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """
    Immutable container for the full pipeline output.

    Attributes:
        debits:             Fully analysed debit DataFrame.
        credits:            Labeled credit DataFrame.
        insights:           List of human-readable insight strings.
        cat_pipeline:       Trained categorization model (for inference reuse).
        spend_pipeline:     Trained expected spend model (for inference reuse).
        global_mean:        Training-set mean (for inference reuse).
        global_std:         Training-set std (for inference reuse).
    """
    debits: pd.DataFrame
    credits: pd.DataFrame
    insights: List[str] = field(default_factory=list)
    cat_pipeline: Optional[SklearnPipeline] = None
    spend_pipeline: Optional[SklearnPipeline] = None
    ranker_pipeline: Optional[SklearnPipeline] = None
    global_mean: float = 0.0
    global_std: float = 0.0


def run_pipeline(
    raw_df: pd.DataFrame,
    zscore_threshold: float = 3.0,
    pct_dev_threshold: float = 0.5,
    amount_tolerance: float = 0.05,
    label_col: str = Col.PSEUDO_LABEL,
    target_col: str = Col.AMOUNT,
) -> PipelineResult:
    """
    Execute the full Insight Engine pipeline end-to-end.

    Phases:
        1. Preprocessing  — schema validation, cleaning, debit/credit split
        2. Labeling       — rule-based pseudo-labeling for model training
        3. Features       — time, rolling, and amount feature engineering
        4. ML Models      — categorization model + expected spend model
        5. Signals        — anomaly detection + recurring transaction flagging
        6. Insights       — natural language insight generation

    Args:
        raw_df:             Raw bank statement DataFrame.
        zscore_threshold:   Z-score threshold for anomaly detection.
        pct_dev_threshold:  Percent deviation threshold for anomaly detection.
        amount_tolerance:   Amount variation tolerance for recurring detection.
        label_col:          Name of the pseudo-label column.
        target_col:         Name of the amount column for spend modelling.

    Returns:
        PipelineResult with fully analysed debits, credits, and insights.

    Raises:
        ValueError: on schema violations or missing columns.
    """
    logger.info("=" * 60)
    logger.info("  INSIGHT ENGINE — Pipeline Start")
    logger.info("=" * 60)

    # ── PHASE 1: Preprocessing ────────────────────────────────────────────
    logger.info("[Phase 1] Preprocessing...")
    debits, credits = preprocess(raw_df)

    # ── PHASE 2: Seed Labeling ────────────────────────────────────────────
    logger.info("[Phase 2] Seed labeling...")
    debits = label_debits(debits, label_col=label_col)
    credits = label_credits(credits, label_col=label_col)

    # ── PHASE 3: Feature Engineering ──────────────────────────────────────
    logger.info("[Phase 3] Feature engineering...")
    global_mean = debits[Col.SIGNED_AMOUNT].mean()
    global_std = debits[Col.SIGNED_AMOUNT].std()

    debits = engineer_features(
        debits,
        global_mean=global_mean,
        global_std=global_std,
        amount_col=target_col,
    )

    # ── PHASE 4: ML Models ────────────────────────────────────────────────
    logger.info("[Phase 4] Training ML models...")

    cat_pipeline = train_categorization_model(debits, label_col=label_col)
    debits = predict_categories(cat_pipeline, debits)

    spend_pipeline = train_expected_spend_model(debits, target_col=target_col)
    debits = predict_expected_spend(spend_pipeline, debits)

    # ── PHASE 5: Signal Detection ─────────────────────────────────────────
    logger.info("[Phase 5] Signal detection...")

    debits = detect_anomalies(
        debits,
        zscore_threshold=zscore_threshold,
        pct_dev_threshold=pct_dev_threshold,
    )

    debits = find_recurring_transactions(
        debits,
        group_col=Col.CLEANED_REMARKS,
        amount_tolerance=amount_tolerance,
    )

    # ── PHASE 5.5: ML Insight Ranking ─────────────────────────────────────
    logger.info("[Phase 5.5] Ranking candidate insights...")
    ranker_pipeline = load_insight_ranker()
    debits = predict_insight_scores(ranker_pipeline, debits)

    # ── PHASE 6: Insight Generation ───────────────────────────────────────
    logger.info("[Phase 6] Generating insights...")
    insights = generate_human_insights(debits)

    logger.info("=" * 60)
    logger.info(f"  Pipeline complete. Generated {len(insights)} insights.")
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
    result: PipelineResult,
) -> PipelineResult:
    """
    Run inference on new transactions using pre-trained models from a
    previous pipeline run.

    Args:
        new_txn:    New raw transaction(s) to analyse.
        result:     PipelineResult from a previous run_pipeline() call.

    Returns:
        PipelineResult for the new transactions.
    """
    logger.info("Running inference on new transaction(s)...")

    debits, credits = preprocess(new_txn)

    debits = label_debits(debits)

    debits = engineer_features_inference(
        debits,
        history_df=result.debits,
        global_mean=result.global_mean,
        global_std=result.global_std,
    )

    debits = predict_categories(result.cat_pipeline, debits)
    debits = predict_expected_spend(result.spend_pipeline, debits)
    debits = detect_anomalies(debits)
    debits = find_recurring_transactions(debits)
    debits = predict_insight_scores(result.ranker_pipeline, debits)

    insights = generate_human_insights(debits)

    return PipelineResult(
        debits=debits,
        credits=credits,
        insights=insights,
        cat_pipeline=result.cat_pipeline,
        spend_pipeline=result.spend_pipeline,
        ranker_pipeline=result.ranker_pipeline,
        global_mean=result.global_mean,
        global_std=result.global_std,
    )
