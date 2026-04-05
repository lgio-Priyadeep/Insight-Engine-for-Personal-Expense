"""
test_benchmark.py — Test Suite for Synthetic Data Generator & Benchmark Sanity
================================================================================
Validates that:
  1. Synthetic data generator produces valid, correctly shaped data
  2. Class distributions match expected ratios
  3. No NaN/inf in feature matrix
  4. Feature-label consistency (anomalies have high z-scores, etc.)
  5. Edge cases are present and correctly labeled
  6. At least one model can train and predict without errors
"""

import sys
import os
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from training_data_generator import (
    generate_insight_dataset,
    _find_best_tip,
    _generate_base_features,
    ALL_CATEGORIES,
)
from config import INSIGHT_TYPES, TIP_CORPUS


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def dataset():
    """Generate dataset once for all tests in this module."""
    X_train, X_test, y_train, y_test = generate_insight_dataset(
        n_samples=1000, n_edge_cases=100, test_size=0.2, random_state=42,
    )
    return X_train, X_test, y_train, y_test


# ── Data Shape Tests ─────────────────────────────────────────────────────────

def test_data_generator_shape(dataset):
    """Verify output dimensions are consistent."""
    X_train, X_test, y_train, y_test = dataset

    assert len(X_train) + len(X_test) == 1100  # 1000 + 100 edge cases
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)

    # Feature columns
    expected_features = {
        "amount", "amount_zscore", "percent_deviation", "category_confidence",
        "is_anomaly", "is_recurring", "is_weekend",
        "rolling_7d_mean", "rolling_30d_mean", "rolling_7d_std",
        "month_sin", "month_cos", "amount_log", "predicted_category",
    }
    assert set(X_train.columns) == expected_features
    assert set(X_test.columns) == expected_features

    # Label columns
    assert set(y_train.columns) == {"insight_type", "tip_id"}
    assert set(y_test.columns) == {"insight_type", "tip_id"}


def test_no_nan_in_features(dataset):
    """No NaN or inf values in the feature matrix."""
    X_train, X_test, _, _ = dataset

    numeric_cols = X_train.select_dtypes(include=[np.number]).columns

    for df_name, df in [("X_train", X_train), ("X_test", X_test)]:
        assert not df[numeric_cols].isna().any().any(), \
            f"NaN found in {df_name}: {df[numeric_cols].isna().sum()}"
        assert not np.isinf(df[numeric_cols].values).any(), \
            f"Inf found in {df_name}"


def test_no_nan_in_labels(dataset):
    """No NaN values in labels."""
    _, _, y_train, y_test = dataset

    assert not y_train.isna().any().any()
    assert not y_test.isna().any().any()


# ── Distribution Tests ────────────────────────────────────────────────────────

def test_data_generator_distribution(dataset):
    """Verify insight_type distribution roughly matches target ratios."""
    _, _, y_train, y_test = dataset
    y_all = pd.concat([y_train, y_test])

    dist = y_all["insight_type"].value_counts(normalize=True)

    # no_action should be ~55-65%
    assert 0.40 < dist.get("no_action", 0) < 0.75, \
        f"no_action distribution out of range: {dist.get('no_action', 0):.2%}"

    # Each actionable type should be >3% (allowing for edge case dilution)
    for insight_type in ["spending_spike", "subscription", "trend_warning", "budget_risk"]:
        assert dist.get(insight_type, 0) > 0.03, \
            f"{insight_type} underrepresented: {dist.get(insight_type, 0):.2%}"


def test_all_insight_types_present(dataset):
    """Every insight type from config must appear in the dataset."""
    _, _, y_train, y_test = dataset
    y_all = pd.concat([y_train, y_test])

    present = set(y_all["insight_type"].unique())
    expected = set(INSIGHT_TYPES)

    assert expected.issubset(present), \
        f"Missing insight types: {expected - present}"


def test_categories_are_valid(dataset):
    """All predicted_category values must be from the known set."""
    X_train, X_test, _, _ = dataset

    valid_cats = set(ALL_CATEGORIES)
    train_cats = set(X_train["predicted_category"].unique())
    test_cats = set(X_test["predicted_category"].unique())

    assert train_cats.issubset(valid_cats), \
        f"Unknown categories in train: {train_cats - valid_cats}"
    assert test_cats.issubset(valid_cats), \
        f"Unknown categories in test: {test_cats - valid_cats}"


# ── Feature-Label Consistency Tests ───────────────────────────────────────────

def test_spikes_have_high_zscore(dataset):
    """spending_spike labels should have elevated z-scores on average."""
    X_train, _, y_train, _ = dataset

    spike_mask = y_train["insight_type"] == "spending_spike"
    normal_mask = y_train["insight_type"] == "no_action"

    if spike_mask.sum() > 0 and normal_mask.sum() > 0:
        spike_mean_z = X_train.loc[spike_mask.values, "amount_zscore"].mean()
        normal_mean_z = X_train.loc[normal_mask.values, "amount_zscore"].mean()

        assert spike_mean_z > normal_mean_z, \
            f"Spike z-score ({spike_mean_z:.2f}) should exceed normal ({normal_mean_z:.2f})"


def test_subscriptions_have_recurring_flag(dataset):
    """subscription labels should have is_recurring=1."""
    X_train, _, y_train, _ = dataset

    sub_mask = y_train["insight_type"] == "subscription"
    if sub_mask.sum() > 0:
        recurring_rate = X_train.loc[sub_mask.values, "is_recurring"].mean()
        assert recurring_rate > 0.9, \
            f"Subscriptions should mostly have is_recurring=1, got rate={recurring_rate:.2f}"


def test_no_action_has_benign_features(dataset):
    """no_action labels should have low z-scores and no anomaly flags."""
    X_train, _, y_train, _ = dataset

    normal_mask = y_train["insight_type"] == "no_action"
    if normal_mask.sum() > 0:
        anomaly_rate = X_train.loc[normal_mask.values, "is_anomaly"].mean()
        # Allow some from edge cases (case 2: high z-score tiny amount → no_action with is_anomaly=1)
        assert anomaly_rate < 0.15, \
            f"no_action should rarely have is_anomaly=1, got rate={anomaly_rate:.2f}"


# ── Tip Consistency Tests ─────────────────────────────────────────────────────

def test_no_action_has_no_tip(dataset):
    """no_action insight_type should always map to no_tip."""
    _, _, y_train, y_test = dataset
    y_all = pd.concat([y_train, y_test])

    no_action = y_all[y_all["insight_type"] == "no_action"]
    assert (no_action["tip_id"] == "no_tip").all(), \
        "Some no_action rows have non-null tip_ids"


def test_actionable_insights_have_tips(dataset):
    """Actionable insight types should have valid tip_ids (not no_tip)."""
    _, _, y_train, y_test = dataset
    y_all = pd.concat([y_train, y_test])

    actionable = y_all[y_all["insight_type"] != "no_action"]
    no_tip_rate = (actionable["tip_id"] == "no_tip").mean()

    assert no_tip_rate < 0.05, \
        f"Too many actionable insights without tips: {no_tip_rate:.2%}"


def test_tip_ids_are_valid(dataset):
    """All tip_ids must be from TIP_CORPUS or 'no_tip'."""
    _, _, y_train, y_test = dataset
    y_all = pd.concat([y_train, y_test])

    valid_tips = set(TIP_CORPUS.keys()) | {"no_tip"}
    actual_tips = set(y_all["tip_id"].unique())

    assert actual_tips.issubset(valid_tips), \
        f"Unknown tip_ids: {actual_tips - valid_tips}"


# ── Tip Lookup Tests ──────────────────────────────────────────────────────────

def test_find_best_tip_category_specific():
    """Category-specific tips should be preferred over generic."""
    tip = _find_best_tip("food", "spending_spike")
    assert tip.startswith("tip_food_"), f"Expected food tip, got {tip}"


def test_find_best_tip_generic_fallback():
    """Unknown category should fall back to generic tip."""
    tip = _find_best_tip("unknown_category", "spending_spike")
    assert tip.startswith("tip_generic_"), f"Expected generic tip, got {tip}"


def test_find_best_tip_no_match():
    """Completely unknown insight type should return no_tip."""
    tip = _find_best_tip("food", "nonexistent_type")
    assert tip == "no_tip"


# ── Amount Range Tests ────────────────────────────────────────────────────────

def test_amounts_are_positive(dataset):
    """All synthetic amounts should be positive."""
    X_train, X_test, _, _ = dataset

    assert (X_train["amount"] > 0).all()
    assert (X_test["amount"] > 0).all()


def test_rolling_std_no_zero(dataset):
    """rolling_7d_std should never be exactly zero (prevents div-by-zero)."""
    X_train, X_test, _, _ = dataset

    assert (X_train["rolling_7d_std"] > 0).all()
    assert (X_test["rolling_7d_std"] > 0).all()


# ── Quick Smoke Test ──────────────────────────────────────────────────────────

def test_model_can_train_and_predict(dataset):
    """Smoke test: at least LogisticRegression can train and predict."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder

    X_train, X_test, y_train, y_test = dataset

    numeric_features = [
        "amount", "amount_zscore", "percent_deviation", "category_confidence",
        "is_anomaly", "is_recurring", "is_weekend",
        "rolling_7d_mean", "rolling_30d_mean", "rolling_7d_std",
        "month_sin", "month_cos", "amount_log",
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["predicted_category"]),
        ],
        remainder="drop",
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000)),
    ])

    pipeline.fit(X_train, y_train["insight_type"])
    preds = pipeline.predict(X_test)

    assert len(preds) == len(X_test)
    assert set(preds).issubset(set(INSIGHT_TYPES))
