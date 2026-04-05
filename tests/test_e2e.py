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

    recurring_df = find_recurring_transactions(anomaly_df, group_col="cleaned_remarks", amount_tolerance=0.06)
    
    # Assert subscription was found (Netflix ~15.99 x 3 across 90 days)
    netflix_candidates = recurring_df[recurring_df["cleaned_remarks"].str.contains("netflix", na=False)]
    assert not netflix_candidates.empty, "Failed to locate netflix transactions post-scrubbing."
    assert netflix_candidates.iloc[0]["is_recurring"], "E2E BUG: Cold-start monthly subscription missed!"

    logger.info("Generating Final Insights...")
    insights = generate_human_insights(recurring_df)
    
    for string in insights:
        print(f"INSIGHT: {string}")
        
    logger.info("E2E Pipeline execution completely successful!")

if __name__ == "__main__":
    test_run_e2e_test()
