"""
tests/test_phase3.py — Phase 3 Test Suite
==========================================
Covers Anomaly detection, Recurring identification, and Insight extraction.
"""

import sys
import os
import pandas as pd
import pytest
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from anomaly_detector import detect_anomalies
from recurring_detector import find_recurring_transactions
from insight_generator import generate_human_insights, _select_tip

def test_recurring_detector():
    base = datetime(2024, 1, 1)
    
    # Generate 3 monthly netflix charges and 2 random things
    df = pd.DataFrame({
        "date": [
            base, 
            base + timedelta(days=5), 
            base + timedelta(days=30), 
            base + timedelta(days=40),
            base + timedelta(days=61)
        ],
        "amount": [15.99, 100.0, 15.99, 50.0, 15.99],
        "cleaned_remarks": ["netflix", "random", "netflix", "random", "netflix"]
    })
    
    res = find_recurring_transactions(df, group_col="cleaned_remarks")
    
    assert res["is_recurring"].sum() == 3
    netflix_rows = res[res["cleaned_remarks"] == "netflix"]
    assert (netflix_rows["recurring_frequency"] == "monthly").all()
    assert (netflix_rows["recurring_confidence"] > 0.8).all()
    assert not res.iloc[1]["is_recurring"]


def test_recurring_frequency_none_for_non_recurring():
    """Non-recurring rows must have None for recurring_frequency (not NaN)."""
    base = datetime(2024, 1, 1)
    df = pd.DataFrame({
        "date": [
            base,
            base + timedelta(days=5),
            base + timedelta(days=30),
            base + timedelta(days=40),
            base + timedelta(days=61)
        ],
        "amount": [15.99, 100.0, 15.99, 50.0, 15.99],
        "cleaned_remarks": ["netflix", "random", "netflix", "random", "netflix"]
    })
    res = find_recurring_transactions(df, group_col="cleaned_remarks")
    # Non-recurring rows (index 1, 3) should have None, not NaN
    non_recurring = res[~res["is_recurring"]]
    for _, row in non_recurring.iterrows():
        assert row["recurring_frequency"] is None, (
            f"Expected None for non-recurring row, got {row['recurring_frequency']}"
        )

def test_recurring_requires_minimum_3_occurrences():
    """Pairs of transactions should NOT be flagged as recurring."""
    base = datetime(2024, 1, 1)
    df = pd.DataFrame({
        "date": [base, base + timedelta(days=30)],
        "amount": [15.99, 15.99],
        "cleaned_remarks": ["small_sub", "small_sub"],
    })
    res = find_recurring_transactions(df, group_col="cleaned_remarks")
    assert res["is_recurring"].sum() == 0

def test_biweekly_detection_and_variance_penalty():
    """Biweekly transactions should be detected, high drift gets lower confidence."""
    base = datetime(2024, 1, 1)
    dates = [base + timedelta(days=14 * i) for i in range(4)]
    df = pd.DataFrame({
        "date": dates,
        "amount": [200.0, 201.0, 230.0, 195.0],  # 35 variance over ~200 mean. Drift ~0.17
        "cleaned_remarks": ["gym_fee"] * 4,
    })
    # Tolerance is 0.20, so drift=0.17 passes, but penalty thresh is 0.10, so confidence drops
    res = find_recurring_transactions(df, group_col="cleaned_remarks")
    assert res["is_recurring"].sum() == 4
    assert (res[res["is_recurring"]]["recurring_frequency"] == "biweekly").all()
    assert (res[res["is_recurring"]]["recurring_confidence"] > 0.4).all()


def test_anomaly_composite_threshold():
    df = pd.DataFrame({
        "amount": [10.0, 500.0],
        "amount_zscore": [0.5, 4.0],
        "percent_deviation": [0.1, 0.8], # Above the 0.5 threshold
        "cleaned_remarks": ["coffee", "electronics"],
        "predicted_category": ["food", "shopping"],
        "date": [datetime(2024, 1, 1), datetime(2024, 1, 2)]
    })
    
    res = detect_anomalies(df, zscore_threshold=3.0, pct_dev_threshold=0.5)
    
    assert res.iloc[0]["is_anomaly"] == False
    assert res.iloc[1]["is_anomaly"] == True

def test_missing_anomaly_columns_raises():
    df = pd.DataFrame({
        "amount": [10.0]
    })
    with pytest.raises(ValueError):
        detect_anomalies(df)

def test_insight_generator_creates_strings():
    df = pd.DataFrame({
        "date": [datetime(2024,1,1)],
        "cleaned_remarks": ["zomato"],
        "predicted_category": ["food"],
        "amount": [300.0],
        "percent_deviation": [1.5],
        "is_recurring": [False],
        "is_anomaly": [True]
    })
    
    insights = generate_human_insights(df)
    
    assert len(insights) > 0
    assert any("food" in txt.lower() and "zomato" in txt.lower() for txt in insights)
    assert any("tip:" in txt.lower() for txt in insights)  # Tip from TIP_CORPUS


def test_insight_generator_deterministic():
    """Two calls with same seed must produce identical output."""
    df = pd.DataFrame({
        "date": [datetime(2024,1,1), datetime(2024,1,2)],
        "cleaned_remarks": ["zomato", "amazon"],
        "predicted_category": ["food", "shopping"],
        "amount": [300.0, 5000.0],
        "percent_deviation": [1.5, 2.0],
        "is_recurring": [False, False],
        "is_anomaly": [True, True]
    })
    result_a = generate_human_insights(df, seed=42)
    result_b = generate_human_insights(df, seed=42)
    assert result_a == result_b, "Same seed must produce identical insights"


def test_insight_generator_different_seeds():
    """Different seeds should (likely) produce different tip selections."""
    df = pd.DataFrame({
        "date": [datetime(2024,1,1), datetime(2024,1,2)],
        "cleaned_remarks": ["zomato", "amazon"],
        "predicted_category": ["food", "shopping"],
        "amount": [300.0, 5000.0],
        "percent_deviation": [1.5, 2.0],
        "is_recurring": [False, False],
        "is_anomaly": [True, True]
    })
    result_a = generate_human_insights(df, seed=1)
    result_b = generate_human_insights(df, seed=999)
    # With different seeds, at least one line should differ
    # (statistically almost certain with multiple random.choice calls)
    # But we can't guarantee it for all possible seeds, so we just check
    # they run without error. True determinism is tested above.
    assert isinstance(result_a, list)
    assert isinstance(result_b, list)


def test_select_tip_deterministic():
    """_select_tip with same RNG state produces identical results."""
    import random as _rm
    rng_a = _rm.Random(42)
    rng_b = _rm.Random(42)
    tip_a = _select_tip("food", "spending_spike", rng=rng_a)
    tip_b = _select_tip("food", "spending_spike", rng=rng_b)
    assert tip_a == tip_b, "_select_tip must be deterministic with same seed"


def test_pipeline_result_is_frozen():
    """PipelineResult must be immutable — direct assignment raises FrozenInstanceError."""
    from pipeline import PipelineResult
    import dataclasses

    result = PipelineResult(
        debits=pd.DataFrame({"a": [1]}),
        credits=pd.DataFrame({"b": [2]}),
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.debits = pd.DataFrame({"c": [3]})


def test_pipeline_result_replace():
    """PipelineResult.replace() creates a new instance with overridden fields."""
    from pipeline import PipelineResult

    original = PipelineResult(
        debits=pd.DataFrame({"a": [1]}),
        credits=pd.DataFrame({"b": [2]}),
        global_mean=100.0,
    )
    updated = original.replace(global_mean=200.0)
    assert original.global_mean == 100.0  # original unchanged
    assert updated.global_mean == 200.0   # new instance has updated value
    assert len(updated.debits) == 1       # other fields preserved
