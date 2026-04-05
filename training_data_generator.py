"""
training_data_generator.py — Synthetic Labeled Dataset for Insight & Tip Models
================================================================================
Generates feature vectors with ground-truth insight_type and tip_id labels
by encoding the current rule-based logic plus edge-case augmentation.

The synthetic data mirrors the exact feature schema produced by the existing
pipeline (feature_engineer + categorization_model + expected_spend_model +
anomaly_detector + recurring_detector) so that models trained here can be
plugged directly into the live pipeline.

Security:
    - No real user data is used. All amounts, categories, and temporal
      patterns are synthetically generated with a deterministic seed.
    - z-scored amounts are used in features, not raw financials.
"""

import logging
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import (
    CATEGORY_PRIORITY,
    INSIGHT_TYPES,
    TIP_CORPUS,
)

logger = logging.getLogger(__name__)

# All categories the pipeline can produce (debit + fallback)
ALL_CATEGORIES: list[str] = CATEGORY_PRIORITY + ["uncategorized"]

# Actionable insight types (excludes no_action)
ACTIONABLE_INSIGHTS: list[str] = [t for t in INSIGHT_TYPES if t != "no_action"]


def _find_best_tip(category: str, insight_type: str) -> str:
    """
    Deterministically select the best tip_id for a (category, insight_type) pair.

    Priority:
      1. Category-specific tip matching the insight type
      2. Generic tip matching the insight type
      3. "no_tip" if nothing matches

    Returns:
        tip_id string.
    """
    # First pass: category-specific match
    for tip_id, tip in TIP_CORPUS.items():
        if category in tip["categories"] and insight_type in tip["insights"]:
            return tip_id

    # Second pass: generic match (empty categories list)
    for tip_id, tip in TIP_CORPUS.items():
        if len(tip["categories"]) == 0 and insight_type in tip["insights"]:
            return tip_id

    return "no_tip"


def _generate_base_features(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """
    Generates a DataFrame of synthetic feature vectors that mimic real
    pipeline output distributions. Calibrated against typical Indian
    bank statement patterns.

    Args:
        n:   Number of samples.
        rng: Seeded random generator for reproducibility.

    Returns:
        DataFrame with 14 feature columns (no labels yet).
    """
    # Amount: lognormal distribution centred around typical Indian txn values
    # Mean ~₹500, with some ₹50 coffees and some ₹50,000 purchases
    raw_amounts = rng.lognormal(mean=6.0, sigma=1.2, size=n).clip(10, 100_000)

    # Rolling stats: computed from a simulated history window
    rolling_7d_mean = raw_amounts * rng.uniform(0.7, 1.3, size=n)
    rolling_30d_mean = raw_amounts * rng.uniform(0.8, 1.2, size=n)
    rolling_7d_std = np.abs(rng.normal(loc=raw_amounts * 0.3, scale=raw_amounts * 0.1, size=n))
    # Prevent zero std
    rolling_7d_std = np.maximum(rolling_7d_std, 1.0)

    # Z-score: (amount - rolling_mean) / rolling_std
    amount_zscore = ((raw_amounts - rolling_7d_mean) / rolling_7d_std).clip(-5, 5)

    # Percent deviation: residual / expected (simulated)
    # Most transactions are close to expected; outliers diverge
    percent_deviation = rng.normal(loc=0.0, scale=0.3, size=n)

    # Category confidence: how confident the categorisation model was
    category_confidence = rng.beta(a=5, b=2, size=n)  # skewed toward high confidence

    # Binary flags (initially all False — labeling logic sets them)
    is_anomaly = np.zeros(n, dtype=int)
    is_recurring = np.zeros(n, dtype=int)

    # Time features
    is_weekend = rng.choice([0, 1], size=n, p=[5 / 7, 2 / 7])
    months = rng.integers(1, 13, size=n)
    month_sin = np.sin(2 * np.pi * months / 12)
    month_cos = np.cos(2 * np.pi * months / 12)

    # Amount log
    amount_log = np.log1p(raw_amounts)

    # Predicted category: weighted by typical Indian spending distribution
    category_weights = {
        "food": 0.22, "transport": 0.12, "shopping": 0.15,
        "utilities": 0.10, "health": 0.05, "finance": 0.10,
        "entertainment": 0.08, "atm": 0.05, "transfer": 0.08,
        "uncategorized": 0.05,
    }
    cats = list(category_weights.keys())
    probs = [category_weights[c] for c in cats]
    predicted_category = rng.choice(cats, size=n, p=probs)

    return pd.DataFrame({
        "amount": raw_amounts,
        "amount_zscore": amount_zscore,
        "percent_deviation": percent_deviation,
        "category_confidence": category_confidence,
        "is_anomaly": is_anomaly,
        "is_recurring": is_recurring,
        "is_weekend": is_weekend,
        "rolling_7d_mean": rolling_7d_mean,
        "rolling_30d_mean": rolling_30d_mean,
        "rolling_7d_std": rolling_7d_std,
        "month_sin": month_sin,
        "month_cos": month_cos,
        "amount_log": amount_log,
        "predicted_category": predicted_category,
    })


def _apply_labels(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """
    Applies insight_type and tip_id labels based on rule-based logic
    that mirrors the current hardcoded insight_generator.py, extended
    with trend_warning and budget_risk scenarios.

    Mutates is_anomaly and is_recurring flags to be consistent with labels.
    Also adjusts feature values so that anomalies actually LOOK anomalous
    in the feature space (prevents label/feature contradiction).
    """
    df = df.copy()
    n = len(df)

    # Determine insight type distribution targets
    # ~60% no_action, ~10% each for others
    target_counts = {
        "no_action": int(n * 0.60),
        "spending_spike": int(n * 0.10),
        "subscription": int(n * 0.10),
        "trend_warning": int(n * 0.10),
        "budget_risk": int(n * 0.10),
    }

    # Shuffle indices to assign labels randomly
    indices = rng.permutation(n)
    labels = np.full(n, "no_action", dtype=object)
    tip_ids = np.full(n, "no_tip", dtype=object)

    cursor = 0
    for insight_type in ACTIONABLE_INSIGHTS:
        count = target_counts.get(insight_type, 0)
        selected = indices[cursor: cursor + count]
        labels[selected] = insight_type
        cursor += count

    df["insight_type"] = labels

    # === Make features consistent with labels ===

    # Spending spikes: high z-score, high deviation, is_anomaly=1
    spike_mask = df["insight_type"] == "spending_spike"
    spike_count = spike_mask.sum()
    if spike_count > 0:
        df.loc[spike_mask, "amount_zscore"] = rng.uniform(3.0, 5.0, size=spike_count)
        df.loc[spike_mask, "percent_deviation"] = rng.uniform(0.5, 3.0, size=spike_count)
        df.loc[spike_mask, "is_anomaly"] = 1
        # Spikes tend to be higher amounts
        df.loc[spike_mask, "amount"] = rng.lognormal(mean=8.0, sigma=0.8, size=spike_count).clip(500, 100_000)
        df.loc[spike_mask, "amount_log"] = np.log1p(df.loc[spike_mask, "amount"])

    # Subscriptions: is_recurring=1, low variance
    sub_mask = df["insight_type"] == "subscription"
    sub_count = sub_mask.sum()
    if sub_count > 0:
        df.loc[sub_mask, "is_recurring"] = 1
        df.loc[sub_mask, "amount_zscore"] = rng.uniform(-1.0, 1.0, size=sub_count)
        df.loc[sub_mask, "percent_deviation"] = rng.uniform(-0.1, 0.1, size=sub_count)
        # Subscriptions are typically small, consistent amounts
        df.loc[sub_mask, "amount"] = rng.choice(
            [99, 149, 199, 299, 499, 799, 999, 1499],
            size=sub_count,
        ).astype(float)
        df.loc[sub_mask, "amount_log"] = np.log1p(df.loc[sub_mask, "amount"])

    # Trend warnings: rolling_7d_mean > rolling_30d_mean * 1.2
    trend_mask = df["insight_type"] == "trend_warning"
    trend_count = trend_mask.sum()
    if trend_count > 0:
        base = df.loc[trend_mask, "rolling_30d_mean"].values
        df.loc[trend_mask, "rolling_7d_mean"] = base * rng.uniform(1.2, 1.8, size=trend_count)
        df.loc[trend_mask, "amount_zscore"] = rng.uniform(1.0, 2.5, size=trend_count)
        df.loc[trend_mask, "percent_deviation"] = rng.uniform(0.1, 0.5, size=trend_count)

    # Budget risk: elevated z-score but below anomaly threshold
    budget_mask = df["insight_type"] == "budget_risk"
    budget_count = budget_mask.sum()
    if budget_count > 0:
        df.loc[budget_mask, "amount_zscore"] = rng.uniform(2.0, 3.0, size=budget_count)
        df.loc[budget_mask, "percent_deviation"] = rng.uniform(0.2, 0.5, size=budget_count)
        base = df.loc[budget_mask, "rolling_30d_mean"].values
        df.loc[budget_mask, "rolling_7d_mean"] = base * rng.uniform(1.05, 1.2, size=budget_count)

    # No-action: ensure features are benign
    normal_mask = df["insight_type"] == "no_action"
    normal_count = normal_mask.sum()
    if normal_count > 0:
        df.loc[normal_mask, "amount_zscore"] = rng.uniform(-1.5, 1.5, size=normal_count)
        df.loc[normal_mask, "percent_deviation"] = rng.uniform(-0.2, 0.2, size=normal_count)
        df.loc[normal_mask, "is_anomaly"] = 0
        df.loc[normal_mask, "is_recurring"] = 0

    # === Assign tip_ids based on (category, insight_type) ===
    for idx in df.index:
        insight = df.at[idx, "insight_type"]
        if insight == "no_action":
            df.at[idx, "tip_id"] = "no_tip"
        else:
            category = df.at[idx, "predicted_category"]
            df.at[idx, "tip_id"] = _find_best_tip(category, insight)

    # Backfill: if _find_best_tip returned "no_tip" for an actionable insight,
    # fall back to generic
    still_no_tip = (df["tip_id"] == "no_tip") & (df["insight_type"] != "no_action")
    for idx in df[still_no_tip].index:
        insight = df.at[idx, "insight_type"]
        df.at[idx, "tip_id"] = _find_best_tip("", insight)

    return df


def _add_edge_cases(
    df: pd.DataFrame, n_edge: int, rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Augments the dataset with deliberately ambiguous / boundary samples
    to stress-test model generalisation.

    Edge cases:
      1. Borderline z-scores (2.9–3.1) — tests the anomaly decision boundary
      2. High z-score but low amount — should be no_action, not spike
      3. Recurring-looking but too variable — should be no_action
      4. Weekend vs weekday pattern shifts
      5. Low-confidence categorisation with anomaly features
    """
    cases = []
    per_case = n_edge // 5

    # Case 1: Borderline z-scores → label as no_action (below threshold)
    for _ in range(per_case):
        cat = rng.choice(ALL_CATEGORIES)
        cases.append({
            "amount": float(rng.lognormal(6.5, 0.5)),
            "amount_zscore": float(rng.uniform(2.8, 3.1)),
            "percent_deviation": float(rng.uniform(0.3, 0.6)),
            "category_confidence": float(rng.uniform(0.6, 0.9)),
            "is_anomaly": 0,  # borderline → NOT flagged
            "is_recurring": 0,
            "is_weekend": int(rng.choice([0, 1])),
            "rolling_7d_mean": float(rng.lognormal(6.0, 0.5)),
            "rolling_30d_mean": float(rng.lognormal(6.0, 0.5)),
            "rolling_7d_std": float(rng.lognormal(4.0, 0.5)),
            "month_sin": float(np.sin(2 * np.pi * rng.integers(1, 13) / 12)),
            "month_cos": float(np.cos(2 * np.pi * rng.integers(1, 13) / 12)),
            "amount_log": float(np.log1p(rng.lognormal(6.5, 0.5))),
            "predicted_category": cat,
            "insight_type": "budget_risk",  # borderline → budget risk, not spike
            "tip_id": _find_best_tip(cat, "budget_risk"),
        })

    # Case 2: High z-score but tiny amount (₹10–50) → no_action
    for _ in range(per_case):
        tiny_amount = float(rng.uniform(10, 50))
        cases.append({
            "amount": tiny_amount,
            "amount_zscore": float(rng.uniform(3.5, 5.0)),
            "percent_deviation": float(rng.uniform(1.0, 3.0)),
            "category_confidence": float(rng.uniform(0.5, 0.9)),
            "is_anomaly": 1,  # statistically anomalous but trivial
            "is_recurring": 0,
            "is_weekend": int(rng.choice([0, 1])),
            "rolling_7d_mean": float(rng.uniform(5, 20)),
            "rolling_30d_mean": float(rng.uniform(5, 20)),
            "rolling_7d_std": float(rng.uniform(2, 8)),
            "month_sin": float(np.sin(2 * np.pi * rng.integers(1, 13) / 12)),
            "month_cos": float(np.cos(2 * np.pi * rng.integers(1, 13) / 12)),
            "amount_log": float(np.log1p(tiny_amount)),
            "predicted_category": rng.choice(ALL_CATEGORIES),
            "insight_type": "no_action",
            "tip_id": "no_tip",
        })

    # Case 3: Looks recurring but high amount variance → no_action
    for _ in range(per_case):
        cases.append({
            "amount": float(rng.lognormal(6.0, 1.0)),
            "amount_zscore": float(rng.uniform(-0.5, 0.5)),
            "percent_deviation": float(rng.uniform(-0.1, 0.1)),
            "category_confidence": float(rng.uniform(0.7, 0.95)),
            "is_anomaly": 0,
            "is_recurring": 0,  # looks recurring but isn't
            "is_weekend": int(rng.choice([0, 1])),
            "rolling_7d_mean": float(rng.lognormal(6.0, 0.5)),
            "rolling_30d_mean": float(rng.lognormal(6.0, 0.5)),
            "rolling_7d_std": float(rng.lognormal(5.0, 0.5)),
            "month_sin": float(np.sin(2 * np.pi * rng.integers(1, 13) / 12)),
            "month_cos": float(np.cos(2 * np.pi * rng.integers(1, 13) / 12)),
            "amount_log": float(np.log1p(rng.lognormal(6.0, 1.0))),
            "predicted_category": rng.choice(ALL_CATEGORIES),
            "insight_type": "no_action",
            "tip_id": "no_tip",
        })

    # Case 4: Weekend spending spikes (context-dependent)
    for _ in range(per_case):
        cat = rng.choice(["food", "entertainment", "shopping"])
        cases.append({
            "amount": float(rng.lognormal(7.5, 0.5)),
            "amount_zscore": float(rng.uniform(3.0, 4.5)),
            "percent_deviation": float(rng.uniform(0.5, 2.0)),
            "category_confidence": float(rng.uniform(0.7, 0.95)),
            "is_anomaly": 1,
            "is_recurring": 0,
            "is_weekend": 1,
            "rolling_7d_mean": float(rng.lognormal(6.5, 0.5)),
            "rolling_30d_mean": float(rng.lognormal(6.5, 0.5)),
            "rolling_7d_std": float(rng.lognormal(4.5, 0.5)),
            "month_sin": float(np.sin(2 * np.pi * rng.integers(1, 13) / 12)),
            "month_cos": float(np.cos(2 * np.pi * rng.integers(1, 13) / 12)),
            "amount_log": float(np.log1p(rng.lognormal(7.5, 0.5))),
            "predicted_category": cat,
            "insight_type": "spending_spike",
            "tip_id": _find_best_tip(cat, "spending_spike"),
        })

    # Case 5: Low-confidence categorisation with anomaly features
    for _ in range(per_case):
        cat = "uncategorized"
        cases.append({
            "amount": float(rng.lognormal(7.0, 0.8)),
            "amount_zscore": float(rng.uniform(3.0, 5.0)),
            "percent_deviation": float(rng.uniform(0.5, 2.5)),
            "category_confidence": float(rng.uniform(0.1, 0.4)),
            "is_anomaly": 1,
            "is_recurring": 0,
            "is_weekend": int(rng.choice([0, 1])),
            "rolling_7d_mean": float(rng.lognormal(6.0, 0.5)),
            "rolling_30d_mean": float(rng.lognormal(6.0, 0.5)),
            "rolling_7d_std": float(rng.lognormal(4.5, 0.5)),
            "month_sin": float(np.sin(2 * np.pi * rng.integers(1, 13) / 12)),
            "month_cos": float(np.cos(2 * np.pi * rng.integers(1, 13) / 12)),
            "amount_log": float(np.log1p(rng.lognormal(7.0, 0.8))),
            "predicted_category": cat,
            "insight_type": "spending_spike",
            "tip_id": _find_best_tip(cat, "spending_spike"),
        })

    edge_df = pd.DataFrame(cases)
    return pd.concat([df, edge_df], ignore_index=True)


# ── Public API ────────────────────────────────────────────────────────────────

def generate_insight_dataset(
    n_samples: int = 5000,
    n_edge_cases: int = 500,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Generates a complete labeled dataset for insight & tip model training.

    Returns:
        (X_train, X_test, y_train, y_test)

        X columns: 14 feature columns
        y columns: ['insight_type', 'tip_id']

    All data is synthetic. No real user data is used.
    """
    rng = np.random.default_rng(random_state)

    logger.info(f"Generating {n_samples} base samples + {n_edge_cases} edge cases...")

    # Generate base features
    df = _generate_base_features(n_samples, rng)

    # Apply rule-based labels
    df = _apply_labels(df, rng)

    # Add edge cases
    df = _add_edge_cases(df, n_edge_cases, rng)

    # Shuffle
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # Separate features and labels
    label_cols = ["insight_type", "tip_id"]
    feature_cols = [c for c in df.columns if c not in label_cols]

    X = df[feature_cols]
    y = df[label_cols]

    # Stratified split on insight_type (primary task)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y["insight_type"],
    )

    logger.info(
        f"Dataset ready. Train: {len(X_train)}, Test: {len(X_test)}. "
        f"Insight classes: {y_train['insight_type'].nunique()}, "
        f"Tip classes: {y_train['tip_id'].nunique()}"
    )

    # Log class distribution
    dist = y_train["insight_type"].value_counts(normalize=True)
    for cls, pct in dist.items():
        logger.info(f"  {cls}: {pct:.1%}")

    return X_train, X_test, y_train, y_test
