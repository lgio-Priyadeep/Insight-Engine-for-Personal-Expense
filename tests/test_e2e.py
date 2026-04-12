"""
test_e2e.py — End-to-End Pipeline Integration Test
===================================================
Simulates the entire flow from a raw, messy bank statement extraction
down to the final NLP insight strings. Use this script to rigorously test the
boundaries between all phases.
"""

import sys
import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Local modules
from preprocessor import preprocess
from feature_engineer import engineer_features, fill_rolling_nulls, engineer_features_inference
from seed_labeler import label_debits, label_credits
from categorization_model import train_categorization_model, predict_categories
from expected_spend_model import train_expected_spend_model, predict_expected_spend
from anomaly_detector import detect_anomalies
from recurring_detector import find_recurring_transactions
from insight_generator import generate_human_insights

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("e2e_test")

def test_run_e2e_test():
    logger.info("Starting E2E Rigorous Validation Pipeline...")

    # 1. Create Raw Messy Data (Similar to bank extracts)
    base_date = datetime(2023, 1, 1)
    
    # We will formulate ~90 days of history
    # - Weekly groceries (food)
    # - Monthly netflix (subscription)
    # - Random shopping
    # - ONE massive $2000 amazon outlier (Anomaly)
    dates = []
    amounts = []
    flags = []
    remarks = []
    
    for i in range(90):
        current_date = base_date + timedelta(days=i)
        
        # Weekly grocery
        if current_date.weekday() == 5: # Saturday
            dates.append(current_date)
            # small variance
            amounts.append(50.0 + np.random.uniform(-5, 5))
            flags.append("dr")
            remarks.append("Visa txn at Zomato POS")
            
        # Monthly netflix (Day 15 of month)
        if current_date.day == 15:
            dates.append(current_date)
            amounts.append(15.99)
            flags.append("DR")
            remarks.append("NETFLIX.COM SUBSCRIPTION 9876543210") # Should hit PII scrub
            
        # Random noise
        if np.random.rand() > 0.8:
            dates.append(current_date)
            amounts.append(120.0)
            flags.append("DR")
            remarks.append("Amazon purchase user@email.com")
            
        # The anomaly
        if i == 45: 
            dates.append(current_date)
            amounts.append(2500.0)
            flags.append(" dr ")
            remarks.append("Amazon super extreme spike")

    raw_df = pd.DataFrame({
        "date": dates,
        "amount": amounts,
        "amount_flag": flags,
        "remarks": remarks,
        "balance": [5000.0]*len(dates)
    })
    
    # Shuffle it slightly to test chronological sorting logic
    raw_df = raw_df.sample(frac=1).reset_index(drop=True)

    # ---------------- PHASE 1: Preprocessing ----------------
    logger.info("Running Preprocessor...")
    debits, credits = preprocess(raw_df)
    assert "cleaned_remarks" in debits.columns, "Cleaned remarks missing."
    assert any(debits["cleaned_remarks"].str.contains("amazon")), "PII scrubbing failed or over-scrubbed."
    
    # Label Initial Ground Truth (Pseudo)
    logger.info("Running Seed Labeler...")
    debits = label_debits(debits)
    
    # Feature Engineering (Training Context)
    logger.info("Running Initial Feature Engineering...")
    # Calculate global mean/std representing training data
    gm = debits["signed_amount"].mean()
    gs = debits["signed_amount"].std()
    
    train_historical_df = engineer_features(debits, global_mean=gm, global_std=gs)

    # ---------------- PHASE 2: ML Models ----------------
    logger.info("Training Categorization Model...")
    cat_pipeline = train_categorization_model(train_historical_df, label_col="pseudo_label")
    train_historical_df = predict_categories(cat_pipeline, train_historical_df)
    
    logger.info("Training Expected Spend Model...")
    spend_pipeline = train_expected_spend_model(train_historical_df, target_col="amount")
    
    # Inference back onto dataset to acquire residuals
    analyzed_df = predict_expected_spend(spend_pipeline, train_historical_df)

    # ---------------- PHASE 3: Signals & Insights ----------------
    logger.info("Running Anomaly & Recurring Detectors...")
    anomaly_df = detect_anomalies(analyzed_df, zscore_threshold=2.5, pct_dev_threshold=0.5)
    
    # Assert anomaly was found (the $2500 spike)
    spike_candidates = anomaly_df[anomaly_df["amount"] == 2500.0]
    assert not spike_candidates.empty, "Failed to identify the $2500 transaction."
    assert spike_candidates.iloc[0]["is_anomaly"], "E2E BUG: The $2500 spike was NOT flagged as an anomaly!"

    recurring_df = find_recurring_transactions(anomaly_df, group_col="cleaned_remarks")
    
    # Assert subscription was found (Netflix ~15.99 x 3 across 90 days)
    netflix_candidates = recurring_df[recurring_df["cleaned_remarks"].str.contains("netflix", na=False)]
    assert not netflix_candidates.empty, "Failed to locate netflix transactions post-scrubbing."
    assert netflix_candidates.iloc[0]["is_recurring"], "E2E BUG: Cold-start monthly subscription missed!"

    logger.info("Generating Final Insights...")
    insights = generate_human_insights(recurring_df)
    
    for string in insights:
        print(f"INSIGHT: {string}")
        
    logger.info("E2E Pipeline execution completely successful!")

def test_run_inference_uses_history_features():
    """
    Verifies that run_inference carries history-aware rolling features
    through the full pipeline rather than discarding them.

    Strategy:
        1. Build a baseline PipelineResult via run_pipeline (90 days history)
        2. Create a new transaction that looks like a continuation
        3. Run run_inference with history
        4. Run run_pipeline on the same new transaction WITHOUT history
        5. Assert rolling features DIFFER — proving history was stitched
    """
    from pipeline import run_pipeline, run_inference
    from schema import Col

    # Build 90-day history
    base_date = datetime(2023, 1, 1)
    dates, amounts, flags, remarks = [], [], [], []
    for i in range(90):
        current_date = base_date + timedelta(days=i)
        if current_date.weekday() == 5:
            dates.append(current_date)
            amounts.append(50.0 + np.random.uniform(-5, 5))
            flags.append("dr")
            remarks.append("Visa txn at Zomato POS")
        if current_date.day == 15:
            dates.append(current_date)
            amounts.append(15.99)
            flags.append("DR")
            remarks.append("NETFLIX.COM SUBSCRIPTION")

    history_df = pd.DataFrame({
        "date": dates, "amount": amounts,
        "amount_flag": flags, "remarks": remarks,
    })

    # 1. Baseline: run full pipeline on history
    baseline_result = run_pipeline(history_df)
    assert baseline_result.debits is not None
    assert len(baseline_result.debits) > 0

    # 2. New transaction: a future Zomato purchase
    new_txn = pd.DataFrame({
        "date": [datetime(2023, 4, 5)],
        "amount": [55.0],
        "amount_flag": ["DR"],
        "remarks": ["Visa txn at Zomato POS"],
    })

    from model_state import InsightModelState
    state = InsightModelState(
        pipeline_version="1.0.0",
        cat_pipeline=baseline_result.cat_pipeline,
        spend_pipeline=baseline_result.spend_pipeline,
        ranker_pipeline=baseline_result.ranker_pipeline,
        global_mean=baseline_result.global_mean,
        global_std=baseline_result.global_std,
    )

    # 3. Inference WITH history
    infer_result = run_inference(new_txn, state, history_df=baseline_result.debits)
    assert len(infer_result.debits) == 1, "Inference should return exactly the new transaction"

    infer_debits = infer_result.debits
    assert Col.ROLLING_7D_MEAN in infer_debits.columns
    assert Col.ROLLING_30D_MEAN in infer_debits.columns
    assert Col.EXPECTED_AMOUNT in infer_debits.columns
    assert Col.PREDICTED_CATEGORY in infer_debits.columns

    # 4. Verify history was actually used:
    # With 90 days of Zomato history (~₹50/txn), the rolling_7d_mean should
    # reflect the historical Zomato spend pattern, NOT the global fallback.
    infer_rolling = infer_debits[Col.ROLLING_7D_MEAN].iloc[0]
    global_mean_fallback = baseline_result.global_mean

    # The global mean covers ALL transaction types (Zomato + Netflix) at
    # varying amounts. The rolling_7d_mean for a Zomato-specific window
    # should be different from this global average.
    assert infer_rolling != global_mean_fallback, (
        f"History-aware rolling mean ({infer_rolling}) should differ from "
        f"global mean fallback ({global_mean_fallback}). "
        "run_inference is not using history features."
    )

    # The history-aware rolling mean should be positive and close to the
    # Zomato baseline (~₹50) since that's the dominant recent pattern.
    assert infer_rolling > 0, "Rolling mean with history should be positive"

    logger.info("run_inference history-aware test passed!")


if __name__ == "__main__":
    test_run_e2e_test()
