"""
model_benchmark.py — Multi-Model Benchmark for Insight & Tip Selection
========================================================================
Trains and evaluates 12 candidate ML models on synthetic data to identify
the best architecture for:
  1. Insight Ranking  (5-class: spending_spike, subscription, trend_warning,
                       budget_risk, no_action)
  2. Tip Selection    (~36-class: tip_id from TIP_CORPUS + no_tip)

Each model is wrapped in a sklearn Pipeline with ColumnTransformer for
consistent preprocessing. Evaluation uses 5-fold stratified CV on the
training set, then a final holdout test evaluation.

Two benchmark variants are run per task:
  - WITH is_anomaly feature (production mode)
  - WITHOUT is_anomaly feature (robustness / leakage test)

Usage:
    python model_benchmark.py

Dependencies:
    scikit-learn, xgboost, lightgbm, catboost (all already installed)
"""

import logging
import time
import warnings
import sys
import os
import io
import pickle
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from training_data_generator import generate_insight_dataset

# Suppress CatBoost verbose output and convergence warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Feature Definitions ──────────────────────────────────────────────────────

NUMERIC_FEATURES: list[str] = [
    "amount", "amount_zscore", "percent_deviation", "category_confidence",
    "is_anomaly", "is_recurring", "is_weekend",
    "rolling_7d_mean", "rolling_30d_mean", "rolling_7d_std",
    "month_sin", "month_cos", "amount_log",
]

NUMERIC_FEATURES_NO_ANOMALY: list[str] = [
    f for f in NUMERIC_FEATURES if f != "is_anomaly"
]

CATEGORICAL_FEATURES: list[str] = ["predicted_category"]


# ── Model Registry ───────────────────────────────────────────────────────────

def get_candidate_models() -> Dict[str, object]:
    """
    Returns a dict of {name: classifier_instance} for all 12 candidates.
    Each classifier is configured with sensible defaults for this problem.
    """
    return {
        "LogisticRegression": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            solver="lbfgs",
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "LinearSVC": CalibratedClassifierCV(
            LinearSVC(
                class_weight="balanced",
                max_iter=2000,
                dual="auto",
            ),
            cv=3,
        ),
        "KNeighbors": KNeighborsClassifier(
            n_neighbors=7,
            weights="distance",
            n_jobs=-1,
        ),
        "DecisionTree": DecisionTreeClassifier(
            max_depth=6,
            class_weight="balanced",
            random_state=42,
        ),
        "MLPClassifier": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42,
        ),
        "AdaBoost": AdaBoostClassifier(
            n_estimators=100,
            learning_rate=0.5,
            random_state=42,
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=100,
            max_depth=8,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            use_label_encoder=False,
            eval_metric="mlogloss",
            random_state=42,
            verbosity=0,
            n_jobs=-1,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            class_weight="balanced",
            random_state=42,
            verbose=-1,
            n_jobs=-1,
        ),
        "CatBoost": CatBoostClassifier(
            iterations=100,
            depth=4,
            learning_rate=0.1,
            auto_class_weights="Balanced",
            random_seed=42,
            verbose=0,
        ),
    }


# ── Pipeline Builder ─────────────────────────────────────────────────────────

def build_pipeline(
    classifier: object,
    include_anomaly_feature: bool = True,
    categorical_features: list[str] | None = None,
) -> Pipeline:
    """
    Wraps a classifier in a sklearn Pipeline with ColumnTransformer.

    Args:
        classifier:              Any sklearn-compatible classifier.
        include_anomaly_feature: If False, drops is_anomaly from features.
        categorical_features:    Override the default categorical feature list.
    """
    num_feats = NUMERIC_FEATURES if include_anomaly_feature else NUMERIC_FEATURES_NO_ANOMALY
    cat_feats = categorical_features if categorical_features is not None else CATEGORICAL_FEATURES

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_feats),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_feats),
        ],
        remainder="drop",
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])


# ── Evaluation Engine ────────────────────────────────────────────────────────

# Models that require integer-encoded labels instead of string labels
_NEEDS_LABEL_ENCODING: set[str] = {"XGBoost", "CatBoost", "MLPClassifier"}


def evaluate_model(
    name: str,
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    cv_folds: int = 5,
) -> Dict:
    """
    Evaluates a single model using cross-validation + holdout test.

    Returns a dict with all metrics.
    """
    logger.info(f"  Evaluating {name}...")

    # Some models (XGBoost, CatBoost, MLP) fail on string labels.
    # Encode to integers for training, decode predictions for metric computation.
    le = None
    y_train_enc = y_train
    y_test_enc = y_test
    if name in _NEEDS_LABEL_ENCODING:
        le = LabelEncoder()
        y_train_enc = le.fit_transform(y_train)
        y_test_enc = le.transform(y_test)

    # ── Cross-Validation ──
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    cv_results = cross_validate(
        pipeline, X_train, y_train_enc,
        cv=skf,
        scoring=["accuracy", "f1_macro", "f1_weighted"],
        return_train_score=True,
        n_jobs=1,  # some models don't parallelise well nested
    )

    cv_acc = cv_results["test_accuracy"].mean()
    cv_f1_macro = cv_results["test_f1_macro"].mean()
    cv_f1_weighted = cv_results["test_f1_weighted"].mean()
    train_acc = cv_results["train_accuracy"].mean()

    # ── Final Train + Test ──
    train_start = time.perf_counter()
    pipeline.fit(X_train, y_train_enc)
    train_time = time.perf_counter() - train_start

    # Inference timing (average over test set)
    infer_start = time.perf_counter()
    y_pred_raw = pipeline.predict(X_test)
    infer_time = time.perf_counter() - infer_start
    ms_per_sample = (infer_time / len(X_test)) * 1000

    # Decode predictions back to original string labels for consistent metrics
    if le is not None:
        y_pred = le.inverse_transform(y_pred_raw)
    else:
        y_pred = y_pred_raw

    # Test metrics (always computed on original string labels)
    test_acc = accuracy_score(y_test, y_pred)
    test_f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    test_f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    test_precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
    test_recall = recall_score(y_test, y_pred, average="macro", zero_division=0)

    # Model size (serialised bytes)
    model_bytes = len(pickle.dumps(pipeline))
    model_kb = model_bytes / 1024

    # Overfit indicator
    overfit_gap = train_acc - cv_acc

    return {
        "name": name,
        "cv_accuracy": cv_acc,
        "cv_f1_macro": cv_f1_macro,
        "cv_f1_weighted": cv_f1_weighted,
        "test_accuracy": test_acc,
        "test_f1_macro": test_f1_macro,
        "test_f1_weighted": test_f1_weighted,
        "test_precision": test_precision,
        "test_recall": test_recall,
        "train_time_s": train_time,
        "ms_per_sample": ms_per_sample,
        "model_size_kb": model_kb,
        "overfit_gap": overfit_gap,
        "y_pred": y_pred,
        "y_test": y_test,
    }


def run_benchmark(
    task_name: str,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    include_anomaly: bool = True,
    categorical_features: list[str] | None = None,
) -> List[Dict]:
    """
    Runs all 12 models on a given task and returns sorted results.
    """
    variant = "WITH is_anomaly" if include_anomaly else "WITHOUT is_anomaly"
    logger.info(f"\n{'='*70}")
    logger.info(f"  BENCHMARK: {task_name} ({variant})")
    logger.info(f"  Train: {len(X_train)} | Test: {len(X_test)} | "
                f"Classes: {len(np.unique(y_train))}")
    logger.info(f"{'='*70}")

    results = []
    models = get_candidate_models()

    for name, clf in models.items():
        try:
            pipeline = build_pipeline(
                clf,
                include_anomaly_feature=include_anomaly,
                categorical_features=categorical_features,
            )
            result = evaluate_model(
                name, pipeline,
                X_train, y_train,
                X_test, y_test,
            )
            results.append(result)
        except Exception as e:
            logger.error(f"  ✗ {name} FAILED: {e}")
            results.append({
                "name": name,
                "cv_accuracy": 0, "cv_f1_macro": 0, "cv_f1_weighted": 0,
                "test_accuracy": 0, "test_f1_macro": 0, "test_f1_weighted": 0,
                "test_precision": 0, "test_recall": 0,
                "train_time_s": 0, "ms_per_sample": 0, "model_size_kb": 0,
                "overfit_gap": 0, "y_pred": np.array([]), "y_test": y_test,
                "error": str(e),
            })

    # Sort by F1 macro (primary metric)
    results.sort(key=lambda r: r["test_f1_macro"], reverse=True)
    return results


# ── Report Generation ────────────────────────────────────────────────────────

def format_results_table(results: List[Dict], task_name: str, variant: str) -> str:
    """Formats benchmark results into a readable table string."""
    lines = []
    lines.append("")
    lines.append(f"  {task_name} — {variant}")
    lines.append("=" * 130)
    header = (
        f"{'Rank':<5} {'Model':<22} {'CV Acc':>7} {'CV F1m':>7} "
        f"{'Test Acc':>8} {'F1(ma)':>7} {'F1(wt)':>7} "
        f"{'Prec':>6} {'Rec':>6} {'Train(s)':>9} {'ms/samp':>8} "
        f"{'Size(KB)':>9} {'Overfit':>8}"
    )
    lines.append(header)
    lines.append("-" * 130)

    for rank, r in enumerate(results, 1):
        if "error" in r:
            lines.append(f"  {rank:<3} {r['name']:<22} {'FAILED':>7} — {r.get('error', '')[:60]}")
            continue

        line = (
            f"  {rank:<3} {r['name']:<22} "
            f"{r['cv_accuracy']:>6.4f} {r['cv_f1_macro']:>6.4f} "
            f"{r['test_accuracy']:>7.4f} {r['test_f1_macro']:>6.4f} "
            f"{r['test_f1_weighted']:>6.4f} "
            f"{r['test_precision']:>5.4f} {r['test_recall']:>5.4f} "
            f"{r['train_time_s']:>8.3f} {r['ms_per_sample']:>7.4f} "
            f"{r['model_size_kb']:>8.1f} {r['overfit_gap']:>+7.4f}"
        )
        lines.append(line)

    lines.append("-" * 130)
    return "\n".join(lines)


def print_confusion_matrix(y_test: np.ndarray, y_pred: np.ndarray, model_name: str):
    """Prints a confusion matrix for the given predictions."""
    labels = sorted(set(y_test) | set(y_pred))
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    print(f"\n  Confusion Matrix — {model_name}")
    print(f"  {'':>20}", end="")
    for l in labels:
        print(f" {l[:12]:>12}", end="")
    print()
    for i, row_label in enumerate(labels):
        print(f"  {row_label:>20}", end="")
        for val in cm[i]:
            print(f" {val:>12}", end="")
        print()


def print_classification_report_top(results: List[Dict], top_n: int = 3):
    """Prints detailed classification reports for the top N models."""
    for i, r in enumerate(results[:top_n]):
        if "error" in r or len(r["y_pred"]) == 0:
            continue
        print(f"\n  #{i+1} — {r['name']} — Detailed Classification Report")
        print("  " + "-" * 60)
        report = classification_report(
            r["y_test"], r["y_pred"],
            zero_division=0,
        )
        for line in report.split("\n"):
            print(f"  {line}")
        print_confusion_matrix(r["y_test"], r["y_pred"], r["name"])


def generate_feature_importance(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    include_anomaly: bool = True,
    categorical_features: list[str] | None = None,
) -> str:
    """
    Trains a RandomForest to extract feature importances.
    Returns a formatted string showing feature ranking.
    """
    cat_feats = categorical_features if categorical_features is not None else CATEGORICAL_FEATURES

    pipeline = build_pipeline(
        RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        include_anomaly_feature=include_anomaly,
        categorical_features=cat_feats,
    )
    pipeline.fit(X_train, y_train)

    # Extract feature names after transformation
    preprocessor = pipeline.named_steps["preprocessor"]
    num_feats = NUMERIC_FEATURES if include_anomaly else NUMERIC_FEATURES_NO_ANOMALY

    cat_encoder = preprocessor.named_transformers_["cat"]
    cat_feature_names = list(cat_encoder.get_feature_names_out(cat_feats))
    all_feature_names = list(num_feats) + cat_feature_names

    importances = pipeline.named_steps["classifier"].feature_importances_
    sorted_idx = np.argsort(importances)[::-1]

    lines = ["\n  Feature Importance (RandomForest)", "  " + "-" * 50]
    for i in sorted_idx[:15]:  # top 15
        name = all_feature_names[i] if i < len(all_feature_names) else f"feature_{i}"
        lines.append(f"  {name:<35} {importances[i]:.4f}")

    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    """Entry point: generates data, runs benchmarks, prints report."""

    print("\n" + "█" * 70)
    print("  INSIGHT ENGINE — MODEL SELECTION BENCHMARK")
    print("█" * 70)

    # ── Generate Data ──
    print("\n  Generating synthetic training data...")
    X_train, X_test, y_train, y_test = generate_insight_dataset(
        n_samples=5000, n_edge_cases=500, test_size=0.2, random_state=42,
    )

    # ══════════════════════════════════════════════════════════════════════
    # TASK 1: Insight Ranking (5-class classification)
    # ══════════════════════════════════════════════════════════════════════

    # Variant A: WITH is_anomaly
    insight_results_with = run_benchmark(
        "INSIGHT RANKER",
        X_train, y_train["insight_type"].values,
        X_test, y_test["insight_type"].values,
        include_anomaly=True,
    )

    table = format_results_table(insight_results_with, "INSIGHT RANKER", "WITH is_anomaly")
    print(table)
    print_classification_report_top(insight_results_with, top_n=3)

    # Feature importance
    fi = generate_feature_importance(
        X_train, y_train["insight_type"].values, include_anomaly=True,
    )
    print(fi)

    # Variant B: WITHOUT is_anomaly (leakage test)
    insight_results_without = run_benchmark(
        "INSIGHT RANKER",
        X_train, y_train["insight_type"].values,
        X_test, y_test["insight_type"].values,
        include_anomaly=False,
    )

    table = format_results_table(insight_results_without, "INSIGHT RANKER", "WITHOUT is_anomaly")
    print(table)
    print_classification_report_top(insight_results_without, top_n=2)

    # ══════════════════════════════════════════════════════════════════════
    # TASK 2: Tip Selection (~36-class classification)
    # ══════════════════════════════════════════════════════════════════════

    # Add insight_type as a feature for tip prediction
    X_train_tip = X_train.copy()
    X_train_tip["insight_type_feature"] = y_train["insight_type"].values

    X_test_tip = X_test.copy()
    X_test_tip["insight_type_feature"] = y_test["insight_type"].values

    # Extended categorical features for tip task (no global mutation)
    tip_cat_features = ["predicted_category", "insight_type_feature"]

    tip_results = run_benchmark(
        "TIP SELECTOR",
        X_train_tip, y_train["tip_id"].values,
        X_test_tip, y_test["tip_id"].values,
        include_anomaly=True,
        categorical_features=tip_cat_features,
    )

    table = format_results_table(tip_results, "TIP SELECTOR", "WITH insight_type as feature")
    print(table)
    print_classification_report_top(tip_results, top_n=3)

    # Feature importance for tip task
    fi_tip = generate_feature_importance(
        X_train_tip, y_train["tip_id"].values,
        include_anomaly=True,
        categorical_features=tip_cat_features,
    )
    print(fi_tip)

    # ══════════════════════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════════════════════

    print("\n" + "█" * 70)
    print("  BENCHMARK SUMMARY")
    print("█" * 70)

    best_insight = insight_results_with[0]
    best_insight_no_anom = insight_results_without[0]
    best_tip = tip_results[0]

    print(f"\n  Best Insight Ranker (with is_anomaly):    {best_insight['name']}")
    print(f"    F1(macro): {best_insight['test_f1_macro']:.4f}  |  "
          f"Accuracy: {best_insight['test_accuracy']:.4f}  |  "
          f"Size: {best_insight['model_size_kb']:.1f}KB  |  "
          f"Latency: {best_insight['ms_per_sample']:.4f}ms/sample")

    print(f"\n  Best Insight Ranker (without is_anomaly): {best_insight_no_anom['name']}")
    print(f"    F1(macro): {best_insight_no_anom['test_f1_macro']:.4f}  |  "
          f"Accuracy: {best_insight_no_anom['test_accuracy']:.4f}  |  "
          f"Size: {best_insight_no_anom['model_size_kb']:.1f}KB  |  "
          f"Latency: {best_insight_no_anom['ms_per_sample']:.4f}ms/sample")

    print(f"\n  Best Tip Selector:                        {best_tip['name']}")
    print(f"    F1(macro): {best_tip['test_f1_macro']:.4f}  |  "
          f"Accuracy: {best_tip['test_accuracy']:.4f}  |  "
          f"Size: {best_tip['model_size_kb']:.1f}KB  |  "
          f"Latency: {best_tip['ms_per_sample']:.4f}ms/sample")

    # Leakage check
    delta = best_insight['test_f1_macro'] - best_insight_no_anom['test_f1_macro']
    if delta > 0.15:
        print(f"\n  ⚠️  LEAKAGE WARNING: Removing is_anomaly drops F1 by {delta:.4f}.")
        print(f"      The model may be over-relying on the pre-computed anomaly flag.")
    else:
        print(f"\n  ✓  Leakage check passed. F1 delta: {delta:+.4f} (within tolerance).")

    print("\n" + "█" * 70 + "\n")


if __name__ == "__main__":
    main()
