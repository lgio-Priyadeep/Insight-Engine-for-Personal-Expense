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

import hashlib
import json
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from sklearn.pipeline import Pipeline as SklearnPipeline

from logger_factory import get_logger, generate_new_run_id
from model_state import InsightModelState
import config
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
from known_persons import (
    tag_known_persons, 
    _enforce_known_person_schema, 
    log_unmatched_recurring_transfers, 
    detect_personal_patterns
)

logger = get_logger(__name__)


def _compute_config_hash(kp: dict, sa: dict) -> str:
    """Deterministic fingerprint of the known-persons config.
    
    Used to detect config drift between training and inference.
    Sorted serialization ensures dict ordering doesn't matter.
    """
    payload = json.dumps({"kp": kp, "sa": sa}, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]

def _validate_state_version(
    state: InsightModelState,
    known_persons: dict = None,
    self_accounts: dict = None,
) -> None:
    """Must be the FIRST step in run_inference(). No exceptions."""
    kp = known_persons if known_persons is not None else config.KNOWN_PERSONS
    sa = self_accounts if self_accounts is not None else config.SELF_ACCOUNTS
    
    has_known_persons = bool(kp) or bool(sa)
    if has_known_persons and state.stats_version == "v1_raw":
        raise ValueError(
            "InsightModelState was trained without known-person exclusion "
            "(stats_version='v1_raw'), but KNOWN_PERSONS/SELF_ACCOUNTS is "
            "now configured. Run full run_pipeline() first to retrain "
            "models with filtered stats."
        )
    
    current_hash = _compute_config_hash(kp, sa)
    if has_known_persons and state.kp_config_hash != current_hash:
        raise ValueError(
            f"KNOWN_PERSONS/SELF_ACCOUNTS config has changed since "
            f"last training (stored hash: '{state.kp_config_hash}', "
            f"current hash: '{current_hash}'). Run full run_pipeline() "
            f"to retrain with updated exclusion list."
        )


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
    raw_global_mean: float = 0.0
    raw_global_std: float = 0.0
    stats_version: str = "v1_raw"
    personal_summary: dict = field(default_factory=dict)
    transfer_patterns: list = field(default_factory=list)
    exclusion_stats: dict = field(default_factory=dict)
    kp_config_hash: str = ""
    personal_debits: pd.DataFrame = field(default_factory=pd.DataFrame)
    personal_credits: pd.DataFrame = field(default_factory=pd.DataFrame)

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
    target_col: str,
    stats_version: str,
    kp_config_hash: str,
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
        stats_version=stats_version,
        kp_config_hash=kp_config_hash,
    )

def _pre_initialize_ml_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Initialize ML columns BEFORE mask-gated `.loc` writes."""
    df = df.copy()
    for col in Col.ml_output_columns():
        if col not in df.columns:
            df[col] = pd.NA
    return df

def run_pipeline(
    raw_df: pd.DataFrame,
    zscore_threshold: float = 3.0,
    pct_dev_threshold: float = 0.5,
    label_col: str = Col.PSEUDO_LABEL,
    target_col: str = Col.AMOUNT,
    state: Optional[InsightModelState] = None,
    known_persons: dict = None,
    self_accounts: dict = None,
) -> PipelineResult:
    generate_new_run_id()
    
    logger.info("=" * 60)
    logger.info("  INSIGHT ENGINE — Pipeline Start", extra={"event_type": "pipeline_start"})
    logger.info("=" * 60)
    logger.info(
        "Pipeline mode initialized", 
        extra={"event_type": "pipeline_mode", "metrics": {"mode": "training+inference" if state is None else "inference-only"}}
    )

    kp = known_persons if known_persons is not None else config.KNOWN_PERSONS
    sa = self_accounts if self_accounts is not None else config.SELF_ACCOUNTS

    if state is not None:
        _validate_state_version(state, known_persons=kp, self_accounts=sa)

    # ── PHASE 1: Preprocessing ────────────────────────────────────────────
    logger.info("[Phase 1] Preprocessing...", extra={"event_type": "phase_start", "metrics": {"phase": 1}})
    debits, credits = preprocess(raw_df)

    # ── PHASE 1.5: Known Persons Tagging ──────────────────────────────────
    debits = tag_known_persons(debits, known_persons=kp, self_accounts=sa)
    credits = tag_known_persons(credits, known_persons=kp, self_accounts=sa)
    
    log_unmatched_recurring_transfers(debits)
    log_unmatched_recurring_transfers(credits)
    
    debits = _pre_initialize_ml_columns(debits)
    credits = _pre_initialize_ml_columns(credits)
    
    spend_mask = ~debits[Col.IS_KNOWN_PERSON]
    credit_spend_mask = ~credits[Col.IS_KNOWN_PERSON]

    has_known_persons = bool(kp) or bool(sa)
    stats_version = "v2_filtered" if has_known_persons else "v1_raw"
    current_hash = _compute_config_hash(kp, sa)

    raw_global_mean = debits[Col.SIGNED_AMOUNT].mean()
    raw_global_std = debits[Col.SIGNED_AMOUNT].std()

    # ── PHASE 2: Seed Labeling ────────────────────────────────────────────
    logger.info("[Phase 2] Seed labeling...", extra={"event_type": "phase_start", "metrics": {"phase": 2}})
    if spend_mask.any():
        spend_debits = debits.loc[spend_mask].copy()
        spend_debits = label_debits(spend_debits, label_col=label_col)
        debits.loc[spend_mask, spend_debits.columns] = spend_debits
        
    if credit_spend_mask.any():
        spend_credits = credits.loc[credit_spend_mask].copy()
        spend_credits = label_credits(spend_credits, label_col=label_col)
        credits.loc[credit_spend_mask, spend_credits.columns] = spend_credits
        
    debits = _enforce_known_person_schema(debits)
    credits = _enforce_known_person_schema(credits)

    # ── PHASE 3: Feature Engineering ──────────────────────────────────────
    logger.info("[Phase 3] Feature engineering...", extra={"event_type": "phase_start", "metrics": {"phase": 3}})
    filtered_global_mean = debits.loc[spend_mask, Col.SIGNED_AMOUNT].mean() if spend_mask.any() else raw_global_mean
    filtered_global_std = debits.loc[spend_mask, Col.SIGNED_AMOUNT].std() if spend_mask.any() else raw_global_std

    if spend_mask.any():
        spend_debits = debits.loc[spend_mask].copy()
        spend_debits = engineer_features(
            spend_debits,
            global_mean=filtered_global_mean,
            global_std=filtered_global_std,
            amount_col=target_col,
        )
        debits.loc[spend_mask, spend_debits.columns] = spend_debits
    debits = _enforce_known_person_schema(debits)

    # ── PHASE 4: ML Models ────────────────────────────────────────────────
    logger.info("[Phase 4] Machine Learning Models...", extra={"event_type": "phase_start", "metrics": {"phase": 4}})
    if state is None:
        logger.info(f"Training models explicitly on {spend_mask.sum()} spend rows.", extra={"event_type": "training_triggered", "metrics": {"row_count": int(spend_mask.sum())}})
        if spend_mask.any():
            new_state = train_models(
                debits.loc[spend_mask].copy(), 
                label_col, 
                target_col,
                stats_version=stats_version,
                kp_config_hash=current_hash
            )
            cat_pipeline, spend_pipeline, ranker_pipeline = new_state.cat_pipeline, new_state.spend_pipeline, new_state.ranker_pipeline
        else:
            raise ValueError("No spend rows available for training.")
    else:
        cat_pipeline, spend_pipeline, ranker_pipeline = state.cat_pipeline, state.spend_pipeline, state.ranker_pipeline

    if spend_mask.any():
        spend_debits = debits.loc[spend_mask].copy()
        spend_debits = predict_categories(cat_pipeline, spend_debits)
        spend_debits = predict_expected_spend(spend_pipeline, spend_debits)
        debits.loc[spend_mask, spend_debits.columns] = spend_debits
    debits = _enforce_known_person_schema(debits)

    # ── PHASE 5: Signal Detection ─────────────────────────────────────────
    logger.info("[Phase 5] Signal detection...", extra={"event_type": "phase_start", "metrics": {"phase": 5}})
    if spend_mask.any():
        spend_debits = debits.loc[spend_mask].copy()
        spend_debits = detect_anomalies(
            spend_debits,
            zscore_threshold=zscore_threshold,
            pct_dev_threshold=pct_dev_threshold,
        )
        spend_debits = find_recurring_transactions(spend_debits, group_col=Col.CLEANED_REMARKS)
        debits.loc[spend_mask, spend_debits.columns] = spend_debits
    debits = _enforce_known_person_schema(debits)

    # ── PHASE 5.5: ML Insight Ranking ─────────────────────────────────────
    logger.info("[Phase 5.5] Ranking candidate insights...", extra={"event_type": "phase_start", "metrics": {"phase": "5.5"}})
    if spend_mask.any():
        spend_debits = debits.loc[spend_mask].copy()
        spend_debits = predict_insight_scores(ranker_pipeline, spend_debits)
        debits.loc[spend_mask, spend_debits.columns] = spend_debits
    debits = _enforce_known_person_schema(debits)

    # ── PHASE 6: Insight Generation ───────────────────────────────────────
    logger.info("[Phase 6] Generating insights...", extra={"event_type": "phase_start", "metrics": {"phase": 6}})
    debits = finalize_df(debits)
    credits = finalize_df(credits)
    
    debits = _optimize_memory_footprint(debits)
    credits = _optimize_memory_footprint(credits)
    
    insights = []
    if spend_mask.any():
        insights = generate_human_insights(debits.loc[spend_mask])

    # Personal Patterns
    personal_mask = debits[Col.IS_KNOWN_PERSON].fillna(False)
    personal_insights, personal_summary = detect_personal_patterns(debits.loc[personal_mask])
    insights.extend(personal_insights)
    
    # Calculate exclusion stats automatically to provide transparency
    total_debits = len(debits)
    excluded_debits = int(personal_mask.sum())
    exclusion_stats = {
        "total_transactions": total_debits,
        "excluded_transactions": excluded_debits,
        "exclusion_rate": float(excluded_debits / total_debits) if total_debits > 0 else 0.0,
        "raw_global_mean": raw_global_mean,
        "filtered_global_mean": filtered_global_mean,
        "mean_distortion_pct": float(abs((raw_global_mean - filtered_global_mean) / filtered_global_mean)) if filtered_global_mean > 0 else 0.0
    }
    
    if has_known_persons and total_debits > 0 and excluded_debits == 0:
        logger.info(
            "Configuration provided but no matches found. "
            "v2_filtered stats are identical to v1_raw. This is expected if the "
            "dataset contains no transfers to configured persons.",
            extra={"event_type": "known_persons_zero_matches"}
        )

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
        global_mean=filtered_global_mean,
        global_std=filtered_global_std,
        raw_global_mean=raw_global_mean,
        raw_global_std=raw_global_std,
        stats_version=stats_version,
        personal_summary=personal_summary,
        transfer_patterns=personal_insights,
        exclusion_stats=exclusion_stats,
        kp_config_hash=current_hash,
        personal_debits=debits.loc[personal_mask].copy(),
        personal_credits=credits.loc[credits[Col.IS_KNOWN_PERSON].fillna(False)].copy()
    )


def run_inference(
    new_txn: pd.DataFrame,
    state: InsightModelState,
    history_df: pd.DataFrame,
    zscore_threshold: float = 3.0,
    pct_dev_threshold: float = 0.5,
    known_persons: dict = None,
    self_accounts: dict = None,
) -> PipelineResult:
    generate_new_run_id()
    logger.info("Running inference on new transaction(s)...", extra={"event_type": "pipeline_inference_start"})

    kp = known_persons if known_persons is not None else config.KNOWN_PERSONS
    sa = self_accounts if self_accounts is not None else config.SELF_ACCOUNTS
    
    _validate_state_version(state, known_persons=kp, self_accounts=sa)

    # ── Phase 1: Preprocess ──
    debits, credits = preprocess(new_txn)
    
    # ── Phase 1.5: Known Persons ──
    debits = tag_known_persons(debits, known_persons=kp, self_accounts=sa)
    credits = tag_known_persons(credits, known_persons=kp, self_accounts=sa)
    history_df = tag_known_persons(history_df, known_persons=kp, self_accounts=sa)

    debits = _pre_initialize_ml_columns(debits)
    credits = _pre_initialize_ml_columns(credits)
    
    spend_mask = ~debits[Col.IS_KNOWN_PERSON]
    credit_spend_mask = ~credits[Col.IS_KNOWN_PERSON]
    history_spend_mask = ~history_df[Col.IS_KNOWN_PERSON]

    # ── Phase 2: Seed Labeling ────────────────────────────────────────────
    if spend_mask.any():
        spend_debits = debits.loc[spend_mask].copy()
        spend_debits = label_debits(spend_debits, label_col=Col.PSEUDO_LABEL)
        debits.loc[spend_mask, spend_debits.columns] = spend_debits
        
    if credit_spend_mask.any():
        spend_credits = credits.loc[credit_spend_mask].copy()
        spend_credits = label_credits(spend_credits, label_col=Col.PSEUDO_LABEL)
        credits.loc[credit_spend_mask, spend_credits.columns] = spend_credits
        
    debits = _enforce_known_person_schema(debits)
    credits = _enforce_known_person_schema(credits)

    # ── Phase 3: Feature Engineering (history-aware) ──
    if spend_mask.any():
        spend_debits = debits.loc[spend_mask].copy()
        spend_debits = engineer_features_inference(
            spend_debits,
            history_df=history_df.loc[history_spend_mask].copy(),
            global_mean=state.global_mean,
            global_std=state.global_std,
        )
        debits.loc[spend_mask, spend_debits.columns] = spend_debits
    debits = _enforce_known_person_schema(debits)

    # ── Phase 4: ML Prediction (pre-trained models) ──
    cat_pipeline, spend_pipeline, ranker_pipeline = (
        state.cat_pipeline, state.spend_pipeline, state.ranker_pipeline,
    )
    if spend_mask.any():
        spend_debits = debits.loc[spend_mask].copy()
        spend_debits = predict_categories(cat_pipeline, spend_debits)
        spend_debits = predict_expected_spend(spend_pipeline, spend_debits)
        debits.loc[spend_mask, spend_debits.columns] = spend_debits
    debits = _enforce_known_person_schema(debits)

    # ── Phase 5: Signal Detection ──
    if spend_mask.any():
        spend_debits = debits.loc[spend_mask].copy()
        spend_debits = detect_anomalies(
            spend_debits,
            zscore_threshold=zscore_threshold,
            pct_dev_threshold=pct_dev_threshold,
        )
        spend_debits = find_recurring_transactions(spend_debits, group_col=Col.CLEANED_REMARKS)
        debits.loc[spend_mask, spend_debits.columns] = spend_debits
    debits = _enforce_known_person_schema(debits)

    # ── Phase 5.5: ML Insight Ranking ──
    if spend_mask.any():
        spend_debits = debits.loc[spend_mask].copy()
        spend_debits = predict_insight_scores(ranker_pipeline, spend_debits)
        debits.loc[spend_mask, spend_debits.columns] = spend_debits
    debits = _enforce_known_person_schema(debits)

    # ── Phase 6: Finalize + Insight Generation ──
    debits = finalize_df(debits)
    credits = finalize_df(credits)
    
    debits = _optimize_memory_footprint(debits)
    credits = _optimize_memory_footprint(credits)
    
    insights = []
    if spend_mask.any():
        insights = generate_human_insights(debits.loc[spend_mask])

    return PipelineResult(
        debits=debits,
        credits=credits,
        insights=insights,
        cat_pipeline=cat_pipeline,
        spend_pipeline=spend_pipeline,
        ranker_pipeline=ranker_pipeline,
        global_mean=state.global_mean,
        global_std=state.global_std,
        stats_version=state.stats_version,
        kp_config_hash=state.kp_config_hash,
        personal_debits=debits.loc[spend_mask == False].copy(),
        personal_credits=credits.loc[credits[Col.IS_KNOWN_PERSON].fillna(False)].copy()
    )
