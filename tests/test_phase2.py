"""
tests/test_phase2.py — Phase 2 Test Suite
==========================================
Covers ML models creation and outputs.

Run with:
    pytest tests/test_phase2.py -v
"""

import sys
import os
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from categorization_model import train_categorization_model, predict_categories
from expected_spend_model import train_expected_spend_model, predict_expected_spend

def test_categorization_training_and_prediction():
    df = pd.DataFrame({
        "cleaned_remarks": ["swiggy order", "uber cab ride", "amazon shopping", "swiggy zomato food"],
        "amount_log": [2.3, 3.4, 4.5, 2.5],
        "pseudo_label": ["food", "transport", "shopping", "food"]
    })
    
    pipeline = train_categorization_model(df, label_col="pseudo_label")
    assert pipeline is not None
    
    preds_df = predict_categories(pipeline, df)
    
    assert "predicted_category" in preds_df.columns
    assert "category_confidence" in preds_df.columns
    assert len(preds_df) == 4

def test_spend_model_training_and_prediction():
    df = pd.DataFrame({
        "is_weekend": [0, 1, 0, 1],
        "month_sin": [0.5, -0.5, 0.5, -0.5],
        "month_cos": [0.5, -0.5, 0.5, -0.5],
        "dow_sin": [0.1, 0.2, 0.3, 0.4],
        "dow_cos": [0.1, 0.2, 0.3, 0.4],
        "week_of_month": [1, 2, 3, 4],
        "rolling_7d_mean": [100.0, 200.0, 150.0, 300.0],
        "rolling_30d_mean": [110.0, 190.0, 160.0, 290.0],
        "rolling_7d_std": [5.0, 10.0, 7.5, 15.0],
        "predicted_category": ["food", "transport", "shopping", "food"],
        "amount": [105.0, 210.0, 145.0, 315.0]
    })
    
    pipeline = train_expected_spend_model(df, target_col="amount")
    assert pipeline is not None
    
    preds_df = predict_expected_spend(pipeline, df)
    
    assert "expected_amount" in preds_df.columns
    assert "residual" in preds_df.columns
    assert "percent_deviation" in preds_df.columns
    assert len(preds_df) == 4

def test_expected_spend_model_extrapolates():
    """
    Tests if the RidgeCV is capable of extrapolating predictions significantly outside the training range.
    A RandomForest would fail this and cap out at the max training value.
    """
    df_train = pd.DataFrame({
        "is_weekend": [0, 1, 0, 1],
        "month_sin": [0.5]*4, "month_cos": [0.5]*4,
        "dow_sin": [0.1]*4, "dow_cos": [0.1]*4,
        "week_of_month": [1]*4,
        "rolling_7d_mean": [10.0, 20.0, 30.0, 40.0],
        "rolling_30d_mean": [10.0, 20.0, 30.0, 40.0],
        "rolling_7d_std": [1.0, 2.0, 1.0, 2.0],
        "predicted_category": ["food"]*4,
        "amount": [11.0, 22.0, 31.0, 42.0]
    })
    
    pipeline = train_expected_spend_model(df_train, target_col="amount")
    
    # Test on inputs completely out of bounds (1000+ vs 40 max)
    df_test = pd.DataFrame({
        "is_weekend": [0], "month_sin": [0.5], "month_cos": [0.5],
        "dow_sin": [0.1], "dow_cos": [0.1], "week_of_month": [1],
        "rolling_7d_mean": [1000.0], "rolling_30d_mean": [1000.0],
        "rolling_7d_std": [50.0], "predicted_category": ["food"],
        "amount": [1050.0]
    })
    
    preds_df = predict_expected_spend(pipeline, df_test)
    predicted_extrapolation = preds_df["expected_amount"].iloc[0]
    
    # A tree model would cap strictly <= 42. Ridge will confidently scale ~1000 linearly.
    assert predicted_extrapolation > 500.0


def test_percent_deviation_no_inf():
    """percent_deviation must never contain inf or -inf."""
    df = pd.DataFrame({
        "is_weekend": [0, 1, 0], "month_sin": [0.5]*3, "month_cos": [0.5]*3,
        "dow_sin": [0.1]*3, "dow_cos": [0.1]*3, "week_of_month": [1]*3,
        "rolling_7d_mean": [100.0, 0.01, 200.0],
        "rolling_30d_mean": [100.0, 0.01, 200.0],
        "rolling_7d_std": [5.0, 0.01, 10.0],
        "predicted_category": ["food"]*3,
        "amount": [105.0, 0.5, 210.0]
    })
    pipeline = train_expected_spend_model(df)
    result = predict_expected_spend(pipeline, df)
    import numpy as np
    assert not result["percent_deviation"].isin([np.inf, -np.inf]).any(), \
        "percent_deviation contains inf values"


def test_percent_deviation_negative_expected_amount():
    """When expected_amount < 0 (RidgeCV extrapolation), percent_deviation must still be finite."""
    df = pd.DataFrame({
        "is_weekend": [0]*4, "month_sin": [0.5]*4, "month_cos": [0.5]*4,
        "dow_sin": [0.1]*4, "dow_cos": [0.1]*4, "week_of_month": [1]*4,
        "rolling_7d_mean": [10.0, 20.0, 30.0, 40.0],
        "rolling_30d_mean": [10.0, 20.0, 30.0, 40.0],
        "rolling_7d_std": [1.0]*4,
        "predicted_category": ["food"]*4,
        "amount": [11.0, 22.0, 31.0, 42.0]
    })
    pipeline = train_expected_spend_model(df)

    # Force a scenario where expected_amount could go negative
    df_test = df.copy()
    df_test["rolling_7d_mean"] = [-100.0]*4
    df_test["rolling_30d_mean"] = [-100.0]*4
    result = predict_expected_spend(pipeline, df_test)
    import numpy as np
    assert not result["percent_deviation"].isin([np.inf, -np.inf]).any()
    assert not result["percent_deviation"].isna().any()


def test_percent_deviation_zero_expected_amount():
    """When expected_amount ≈ 0, percent_deviation must be finite (floor clamp)."""
    df = pd.DataFrame({
        "is_weekend": [0]*4, "month_sin": [0.5]*4, "month_cos": [0.5]*4,
        "dow_sin": [0.1]*4, "dow_cos": [0.1]*4, "week_of_month": [1]*4,
        "rolling_7d_mean": [0.0]*4, "rolling_30d_mean": [0.0]*4,
        "rolling_7d_std": [0.0]*4,
        "predicted_category": ["food"]*4,
        "amount": [0.0, 0.01, 0.5, 1.0]
    })
    pipeline = train_expected_spend_model(df)
    result = predict_expected_spend(pipeline, df)
    import numpy as np
    assert not result["percent_deviation"].isin([np.inf, -np.inf]).any()


def test_percent_deviation_normal_case():
    """Standard case: deviation should be (actual - expected) / |expected|."""
    df = pd.DataFrame({
        "is_weekend": [0]*4, "month_sin": [0.5]*4, "month_cos": [0.5]*4,
        "dow_sin": [0.1]*4, "dow_cos": [0.1]*4, "week_of_month": [1]*4,
        "rolling_7d_mean": [100.0]*4, "rolling_30d_mean": [100.0]*4,
        "rolling_7d_std": [5.0]*4,
        "predicted_category": ["food"]*4,
        "amount": [100.0, 200.0, 300.0, 400.0]
    })
    pipeline = train_expected_spend_model(df)
    result = predict_expected_spend(pipeline, df)
    # All percent_deviations must be finite floats
    assert result["percent_deviation"].dtype == float
    assert result["percent_deviation"].notna().all()
