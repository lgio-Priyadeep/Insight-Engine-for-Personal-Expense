# DEEP DIVE — Complex Architecture Components

> Detailed mechanical explanations of the non-obvious design decisions, mathematical invariants, and subtle engineering choices that make the Insight Engine reliable.  
> Each section answers: *What does this do? Why is it done this way? What would break if it were done differently?*

---

## Table of Contents

1. [Leak-Free Rolling Feature Engineering](#1-leak-free-rolling-feature-engineering)
2. [The Dual-Gate Anomaly Detection Design](#2-the-dual-gate-anomaly-detection-design)
3. [Priority-Based Keyword Matching (Seed Labeler)](#3-priority-based-keyword-matching-seed-labeler)
4. [Why RidgeCV Instead of Tree Models for Expected Spend](#4-why-ridgecv-instead-of-tree-models-for-expected-spend)
5. [The Schema Contract System](#5-the-schema-contract-system)
6. [Fallback Label Exclusion During Training](#6-fallback-label-exclusion-during-training)
7. [Insight Score Formula: 1 - P(no_action)](#7-insight-score-formula-1---pno_action)
8. [Diversity-Aware Insight Ranking](#8-diversity-aware-insight-ranking)
9. [Synthetic Data Generator: Feature-Label Consistency](#9-synthetic-data-generator-feature-label-consistency)
10. [Recurring Detection: Frequency Band Design](#10-recurring-detection-frequency-band-design)
11. [TF-IDF Sparsity Preservation](#11-tf-idf-sparsity-preservation)
12. [The Defensive Copy Pattern](#12-the-defensive-copy-pattern)
13. [NaN Handling Strategy Across the Pipeline](#13-nan-handling-strategy-across-the-pipeline)
14. [Cyclical Encoding for Time Features](#14-cyclical-encoding-for-time-features)
15. [The 12-Model Benchmark and Leakage Detection](#15-the-12-model-benchmark-and-leakage-detection)
16. [Graceful Degradation Architecture](#16-graceful-degradation-architecture)
17. [PII Stripping in Remark Cleaning](#17-pii-stripping-in-remark-cleaning)
18. [Merchant Alias Normalization](#18-merchant-alias-normalization)
19. [Tip Corpus Architecture](#19-tip-corpus-architecture)

---

## 1. Leak-Free Rolling Feature Engineering

**Location:** `feature_engineer.py`, functions `add_rolling_features` and `fill_rolling_nulls`.

### The Problem

When computing features like "average spending over the past 7 days", the most common mistake is including the current transaction's amount in the window. This creates **data leakage** — the model effectively gets to "see" the answer (current amount) embedded in its features (rolling mean that includes the current amount).

### The Solution

```python
shifted = df["amount"].shift(1)
df["rolling_7d_mean"]  = shifted.rolling(7, min_periods=1).mean()
df["rolling_30d_mean"] = shifted.rolling(30, min_periods=1).mean()
df["rolling_7d_std"]   = shifted.rolling(7, min_periods=2).std()
```

The `.shift(1)` operation displaces the entire amount column downward by one position:
- Row 0 sees: nothing (NaN)
- Row 1 sees: Row 0's amount only
- Row 2 sees: Row 0 and Row 1's amounts
- Row N sees: Rows 0 through N-1's amounts

The rolling window then operates on this shifted series. The current transaction's amount is **never** in its own feature window.

### What Would Break Without `.shift(1)`

Without the shift, `rolling_7d_mean` for a ₹50,000 transaction would include that ₹50,000 in its own average. The z-score would be suppressed (the mean is artificially pulled toward the outlier), and the anomaly detector would miss legitimate spending spikes. In benchmarking, removing the shift degraded anomaly detection F1 by ~30%.

### NaN Handling After Shift

Row 0's rolling features are always NaN (no prior data exists). These are filled using `fill_rolling_nulls`:
- `rolling_7d_mean` / `rolling_30d_mean` → filled with `global_mean` (mean of all `signed_amount` values in the training set)
- `rolling_7d_std` → filled with `max(global_std, 1.0)`

The `max(..., 1.0)` clause prevents a degenerate case: if all transactions happen to have the same amount, `global_std = 0`, and any subsequent z-score computation would produce infinity.

---

## 2. The Dual-Gate Anomaly Detection Design

**Location:** `anomaly_detector.py`, function `detect_anomalies`.

### The Design

A transaction is flagged as anomalous **only if both conditions pass simultaneously**:

```
Gate 1 (Statistical):  |amount_zscore| > 3.0
Gate 2 (Contextual):   |percent_deviation| > 0.5
```

### Why Two Gates?

**Gate 1 alone is insufficient.** Consider: A user buys ₹10 coffees most days but occasionally buys ₹15 coffee. The ₹15 coffee might have a z-score > 3 (statistically rare), but flagging a ₹5 extra spend as an anomaly produces noise, not insight.

**Gate 2 alone is insufficient.** Consider: A user's spending fluctuates naturally. The RidgeCV model might predict ₹500 for a grocery run, but the user spends ₹600. The 20% deviation is notable, but the absolute z-score might be below 1.0 (₹600 is well within the normal range historically).

**Both gates together filter for transactions that are BOTH:**
1. Historically unusual for this user's overall spending pattern (z-score gate)
2. Unusual relative to what the ML model expected for this specific transaction type (percent deviation gate)

### Threshold Calibration

- Z-score default `3.0`: Standard 3-sigma threshold for statistical significance.
- Percent deviation default `0.5`: A 50% deviation from expected means the transaction cost 1.5× what the model predicted. This filters out minor fluctuations while catching genuine surprises.
- Both thresholds are configurable parameters (not hardcoded constants), allowing tuning without code changes.

### Bidirectionality

Both gates use `.abs()`. This means the system captures:
- **Spending spikes:** User normally spends ₹200/month on food, suddenly spends ₹5,000.
- **Spending drops:** A ₹999/month subscription suddenly charges ₹99 (possible downgrade, trial period ending, or error).

---

## 3. Priority-Based Keyword Matching (Seed Labeler)

**Location:** `seed_labeler.py`, function `_match_remark`.

### The Problem

A transaction remark like `"emi hospital bill payment"` contains keywords that match multiple categories:
- `"emi"` → matches `finance`
- `"hospital"` → matches `health`
- `"bill"` → matches `utilities`

Without a deterministic tiebreaker, the category assigned would be arbitrary or depend on dictionary iteration order (which is insertion order in Python 3.7+, but semantically undefined for this purpose).

### The Solution

`CATEGORY_PRIORITY` in `config.py` defines an explicit ordered list:

```python
CATEGORY_PRIORITY = [
    "food", "transport", "shopping", "utilities", "health",
    "finance", "entertainment", "atm", "transfer", "education",
    "government", "insurance"
]
```

The matching algorithm iterates categories **in this exact order**. The first matching category wins. For the example `"emi hospital bill"`:
1. `food` → no match
2. `transport` → no match
3. `shopping` → no match
4. `utilities` → `"bill"` matches → candidate
5. `health` → `"hospital"` matches → candidate
6. `finance` → `"emi"` matches → candidate

All three are candidates, but the winner is determined by position in `CATEGORY_PRIORITY`. `utilities` (index 3) appears before `health` (index 4) and `finance` (index 5), so `utilities` wins.

### Multi-Word Keywords

Some keywords like `"cash withdrawal"` are multi-word. The matching logic handles this:

```python
keyword_tokens = keyword.split()
if len(keyword_tokens) > 1:
    # ALL tokens must be present in the remark
    match = all(token in remark_tokens for token in keyword_tokens)
else:
    match = keyword in remark_tokens
```

This prevents `"cash reward"` from matching the `atm` category's `"cash withdrawal"` keyword — both tokens must be present.

### Why Return All Matches?

`_match_remark` returns a tuple `(winner, all_matches_dict)`. The `all_matches_dict` is used in testing (`test_phase1.py::TestMatchRemark::test_priority_order_respected`) to verify that priority ordering is correctly resolving ambiguity, and could be used for future multi-label classification.

---

## 4. Why RidgeCV Instead of Tree Models for Expected Spend

**Location:** `expected_spend_model.py`, function `build_spend_pipeline`.

### The Problem

The Expected Spend model predicts "how much should this transaction normally cost?" If the model predicts ₹500 and the actual is ₹5,000, the ₹4,500 residual feeds into anomaly detection.

### Why Not Random Forest or Gradient Boosting?

Tree-based models (Random Forest, XGBoost, LightGBM) **cannot extrapolate**. Their predictions are bounded by the maximum value seen in training data.

**Concrete Scenario:** Training data contains transactions between ₹10 and ₹10,000. A new transaction comes in with `rolling_7d_mean = 50,000` (the user recently received a large salary). A Random Forest would predict at most ₹10,000 (the maximum training value), producing a huge negative residual. The anomaly detector would flag this normal high-income period as anomalous.

RidgeCV is a linear model. It extrapolates naturally:
```
predicted_amount ≈ w₁×rolling_7d_mean + w₂×rolling_30d_mean + ... + bias
```
If `rolling_7d_mean = 50,000`, the prediction scales proportionally, producing a reasonable expected amount.

### Proof in Tests

`test_phase2.py::test_expected_spend_model_extrapolates` validates this explicitly:

```python
# Training data: amounts 11–42, rolling means 10–40
# Test input: rolling_7d_mean = 1000, rolling_30d_mean = 1000
# Expected: RidgeCV predicts > 500 (linear extrapolation)
# A tree model would cap at ≤ 42
assert predicted_extrapolation > 500.0
```

### Why RidgeCV Specifically?

RidgeCV (Ridge with built-in cross-validation over alpha values) is chosen over plain Ridge because:
1. It automatically selects the optimal regularization strength from `alphas=[0.1, 1.0, 10.0, 100.0]`.
2. The cross-validation is efficient (Leave-One-Out for Ridge is analytically solvable, no actual CV loop).
3. It prevents overfitting on the limited feature set (9 numeric + one-hot encoded categories).

---

## 5. The Schema Contract System

**Location:** `schema.py`, class `Col` and function `require_columns`.

### Design

Every column name in the system is defined exactly once in `Col`:

```python
class Col:
    DATE = "date"
    AMOUNT = "amount"
    AMOUNT_ZSCORE = "amount_zscore"
    # ... 20+ more
```

Every module defines its input contract as a static method:

```python
@staticmethod
def anomaly_detector_input() -> list[str]:
    return [Col.AMOUNT_ZSCORE, Col.PERCENT_DEVIATION]
```

At the top of every processing function:

```python
require_columns(df, Col.anomaly_detector_input(), "anomaly_detector")
```

### Why This Matters

**Without this pattern:** A developer renames a column in `feature_engineer.py` from `"amount_zscore"` to `"zscore"`. The `anomaly_detector.py` still references `"amount_zscore"`. The code runs without error until the `df["amount_zscore"]` lookup fails at runtime, deep inside a pipeline that might have already consumed minutes of computation.

**With this pattern:** The rename in `feature_engineer.py` would use `Col.AMOUNT_ZSCORE`, which is a single constant. Changing it in `schema.py` would either fix all references simultaneously or, if someone hardcodes a string, the `require_columns` check would fail **immediately** at the top of the function with a clear error message:

```
ValueError: [anomaly_detector] Missing required columns: {'amount_zscore'}
```

### Failure Mode

`require_columns` raises `ValueError`, not a warning. This is intentional. A missing column means the function **cannot** produce correct output. Silent fallback (e.g., filling with zeros) would produce nonsensical results that propagate silently through the pipeline.

---

## 6. Fallback Label Exclusion During Training

**Location:** `categorization_model.py`, function `train_categorization_model`, lines 73–76.

### The Code

```python
fallbacks = {FALLBACK_DEBIT_LABEL, FALLBACK_CREDIT_LABEL}
valid_mask = ~train_df[label_col].isin(fallbacks)
train_df = train_df[valid_mask].copy()
```

### Why This Is Done

The seed labeler assigns `"uncategorized"` to transactions it cannot match (e.g., `"random xyz merchant"`). If the ML model trains on these rows, it learns a pattern:

> "Any remark I don't recognize should be labeled 'uncategorized'."

This defeats the entire purpose of the ML model. The model exists to **generalize** — to predict real categories for remarks the keyword dictionary doesn't cover.

By excluding fallback labels:
- The model only learns the features of real categories (food, transport, etc.).
- When presented with an unknown remark, it predicts the **most similar real category**, not a meaningless fallback.
- The `category_confidence` field provides a calibrated probability. Low confidence on an unknown remark is more informative than a confident "uncategorized" prediction.

### What Would Break

If fallback labels were included in training and the dataset had, say, 30% uncategorized transactions, the model would learn to predict "uncategorized" for anything vaguely unusual. The downstream anomaly detector and insight generator would lose categorization context for a third of all transactions.

---

## 7. Insight Score Formula: 1 - P(no_action)

**Location:** `insight_model.py`, function `predict_insight_scores`, lines 89–94.

### The Formula

```python
classes = list(pipeline.classes_)
if "no_action" in classes:
    idx = classes.index("no_action")
    scores = 1.0 - probs[:, idx]
else:
    scores = probs.max(axis=1)  # Fallback
```

### Why Not Just `max(predict_proba)` Across All Classes?

The LightGBM ranker is a 5-class classifier: `spending_spike`, `subscription`, `trend_warning`, `budget_risk`, `no_action`.

Using `max(predict_proba)` would produce high scores for transactions that are **confidently** no_action, because a confident "no_action" prediction might have P(no_action) = 0.95.

The formula `1 - P(no_action)` converts the problem into:

> "How likely is it that this transaction has **any** actionable insight?"

This is mathematically equivalent to `P(spending_spike) + P(subscription) + P(trend_warning) + P(budget_risk)` (since all probabilities sum to 1.0).

### Score Interpretation

| Score | Meaning |
|-------|---------|
| 0.95 | 95% chance this transaction has an actionable insight |
| 0.50 | Moderately likely to be interesting |
| 0.05 | Almost certainly normal/boring |

### Graceful Degradation

If the ranker model is not installed (`models/insight_ranker.pkl` does not exist), all scores default to `0.0`. The insight generator then falls back to pure deterministic ordering (whatever order anomalies and subscriptions appear in the DataFrame), which is a reasonable baseline.

---

## 8. Diversity-Aware Insight Ranking

**Location:** `insight_generator.py`, function `generate_human_insights`, lines 136–157.

### The Problem

Without diversity logic, a user with 8 anomalous transactions and 2 recurring subscriptions would see only anomaly insights in their top 10. The subscriptions — which might represent ₹2,000/month of wasted spend — would never surface.

### The Two-Pass Algorithm

```
Pass 1 (Diversity guarantee):
  Sort all candidates by insight_score descending.
  Iterate: for each candidate, if its TYPE has not been seen yet,
           add it to the top_candidates list and mark the type as seen.

Pass 2 (Score-based fill):
  Iterate again: for each candidate not already selected,
                 add it if top_candidates < top_n.

Final: Sort top_candidates by insight_score descending.
```

**Pass 1** ensures at least 1 "subscription" and 1 "spike" insight make it into the top N (if they exist). **Pass 2** fills the remaining slots purely by ML score. The final sort ensures the user sees the most important insights first.

### Concrete Example

Candidates (sorted by score):
```
[0.95, "spike",  "₹15,000 at Electronics Store"]
[0.90, "spike",  "₹8,000 at Amazon"]
[0.85, "spike",  "₹5,000 at Apple"]
[0.80, "spike",  "₹3,000 at Flipkart"]
[0.75, "subscription", "Netflix ₹499/month"]
[0.60, "spike",  "₹2,000 at Zomato"]
```

Without diversity (pure top-5 by score): All spikes, Netflix missed.

With diversity (Pass 1 picks 1 spike + 1 subscription, Pass 2 fills):
```
[0.95, "spike", "₹15,000 at Electronics Store"]  ← Pass 1 (first spike)
[0.90, "spike", "₹8,000 at Amazon"]              ← Pass 2
[0.85, "spike", "₹5,000 at Apple"]               ← Pass 2
[0.80, "spike", "₹3,000 at Flipkart"]            ← Pass 2
[0.75, "subscription", "Netflix ₹499/month"]      ← Pass 1 (first subscription)
```

Netflix now appears in the results even though 4 spikes outscored it.

---

## 9. Synthetic Data Generator: Feature-Label Consistency

**Location:** `training_data_generator.py`, function `_apply_labels`.

### The Problem

Naive synthetic data generation assigns labels randomly. But if a row labeled `"spending_spike"` has `amount_zscore = 0.5` and `is_anomaly = 0`, the ML model learns **contradictory** patterns. In production, real spending spikes have high z-scores and are flagged as anomalies. The model trained on inconsistent data would fail to generalize.

### The Solution

After random label assignment, `_apply_labels` **adjusts feature values** to be consistent with the assigned label:

```python
# Spending spikes: high z-score, high deviation, anomaly flag set
spike_mask = df["insight_type"] == "spending_spike"
if spike_count > 0:
    df.loc[spike_mask, "amount_zscore"] = rng.uniform(3.0, 5.0, ...)
    df.loc[spike_mask, "percent_deviation"] = rng.uniform(0.5, 3.0, ...)
    df.loc[spike_mask, "is_anomaly"] = 1
    df.loc[spike_mask, "amount"] = rng.lognormal(mean=8.0, ...).clip(500, 100_000)

# Subscriptions: recurring flag, low variance, consistent amounts
sub_mask = df["insight_type"] == "subscription"
if sub_count > 0:
    df.loc[sub_mask, "is_recurring"] = 1
    df.loc[sub_mask, "amount_zscore"] = rng.uniform(-1.0, 1.0, ...)
    df.loc[sub_mask, "amount"] = rng.choice([99, 149, 199, 299, 499, 799, ...])

# No-action: explicitly benign features
normal_mask = df["insight_type"] == "no_action"
if normal_count > 0:
    df.loc[normal_mask, "amount_zscore"] = rng.uniform(-1.5, 1.5, ...)
    df.loc[normal_mask, "is_anomaly"] = 0
    df.loc[normal_mask, "is_recurring"] = 0
```

### Edge Case Augmentation

`_add_edge_cases` adds 5 categories of deliberately ambiguous samples:

| Case | Features | Label | Purpose |
|------|----------|-------|---------|
| Borderline z-score (2.8–3.1) | Near anomaly threshold | `budget_risk` | Tests decision boundary |
| High z-score + tiny amount (₹10–50) | z > 3.5 but amount trivial | `no_action` | Prevents trivial anomaly noise |
| Looks recurring but high variance | Low z-score, is_recurring=0 | `no_action` | Prevents false subscriptions |
| Weekend spending spike | is_weekend=1, z > 3.0 | `spending_spike` | Context-dependent anomalies |
| Low-confidence + anomaly features | confidence 0.1–0.4, z > 3.0 | `spending_spike` | Tests robustness to uncertain categorization |

These edge cases comprise ~10% of the dataset and are critical for preventing the model from learning overly simplistic decision rules.

---

## 10. Recurring Detection: Frequency Band Design

**Location:** `recurring_detector.py`, function `find_recurring_transactions`.

### Band Definition

| Frequency | Gap Range (days) | Max Variance (days²) | Rationale |
|-----------|-----------------|---------------------|-----------|
| Weekly | 6–8 | < 3 | 7 ± 1 day. Variance < 3 means std < 1.7 days. |
| Biweekly | 13–16 | < 5 | 14 ± 2 days. Variance < 5 means std < 2.2 days. |
| Monthly | 27–33 | < 10 | 30 ± 3 days. Variance < 10 means std < 3.2 days. |
| Quarterly | 85–95 | < 20 | 90 ± 5 days. Variance < 20 means std < 4.5 days. |

### Why Minimum 3 Occurrences?

With only 2 transactions, there is exactly 1 time gap, and variance is undefined. The gap could be 30 days purely by coincidence. 3 occurrences provide 2 gaps, allowing meaningful variance computation and reducing false positive rates.

### Confidence Penalty

If amount drift exceeds `RECURRING_FLUCTUATION_PENALTY_THRESHOLD` (10%), confidence drops from 1.0 to 0.5. This handles variable-cost subscriptions (e.g., utility bills that fluctuate monthly). The insight generator still reports them, but the ML ranker may deprioritize them.

### Gaps Between Bands

Note the deliberate gaps: 9–12 days, 17–26 days, 34–84 days, 96+ days. Transactions falling in these ranges are **not** classified as recurring. This is by design — the system only identifies clean, well-defined frequencies. A transaction occurring every 20 days doesn't fit any standard billing cycle and is more likely irregular than recurring.

---

## 11. TF-IDF Sparsity Preservation

**Location:** `categorization_model.py`, function `build_categorization_pipeline`, lines 34–42.

### The Code

```python
preprocessor = ColumnTransformer(
    transformers=[
        ("text", TfidfVectorizer(ngram_range=(1, 2), max_features=2000), Col.CLEANED_REMARKS),
        ("num", StandardScaler(with_mean=False), ["amount_log"])
    ],
    remainder="drop",
    sparse_threshold=1.0  # Force output to remain sparse
)
```

### Why `with_mean=False`?

`StandardScaler` normally subtracts the mean from each feature, which converts a sparse matrix into a dense matrix (subtracting the mean from zero values creates non-zero values everywhere). Since TF-IDF produces a sparse matrix (most documents contain few of the 2000 possible features), calling `StandardScaler(with_mean=True)` would materialize the full dense matrix, potentially consuming gigabytes of memory for large datasets.

`with_mean=False` only divides by the standard deviation, preserving the sparsity structure.

### Why `sparse_threshold=1.0`?

The `ColumnTransformer` default behavior is to convert the output to dense if fewer than 50% of values are zero. Setting `sparse_threshold=1.0` forces it to always return a sparse matrix. This prevents a memory explosion when combining the 2000-dimensional TF-IDF matrix with the single `amount_log` column.

### Impact

For a dataset of 50,000 transactions:
- **Dense:** 50,000 × 2001 × 8 bytes = ~762 MB
- **Sparse (typical ~1% fill):** ~7.6 MB

This is the difference between the stress test passing in seconds vs. OOM-killing.

---

## 12. The Defensive Copy Pattern

Every processing function in the pipeline follows this pattern:

```python
def some_function(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()  # Line 1: defensive copy
    # ... transformations ...
    return df
```

### Why

Pandas DataFrames are mutable reference types. Without `.copy()`, modifications inside a function would silently mutate the caller's DataFrame. This creates subtle, hard-to-debug issues:

```python
# Without copy:
original = pd.DataFrame({"a": [1, 2, 3]})
result = some_function(original)
# original["a"] might now be [1, 2, 3, 4] if some_function appended a row
```

### Where It's Enforced

Validated in tests:
- `test_phase1.py::TestPreprocess::test_does_not_mutate_input`
- `test_phase1.py::TestEngineerFeaturesFull::test_does_not_mutate_input`
- `test_phase1.py::TestLabelDebits::test_input_not_mutated`

These tests record `list(df.columns)` before and after function calls and assert exact equality.

### Performance Note

`.copy()` is an O(n) operation that doubles memory usage temporarily. For the 50,000-row stress test, this adds ~2–5% overhead — acceptable for correctness guarantees.

---

## 13. NaN Handling Strategy Across the Pipeline

The pipeline has a **layered** NaN strategy, not a single global approach:

| Phase | Module | NaN Source | Strategy |
|-------|--------|------------|----------|
| 1 | preprocessor | Unparseable dates from `pd.to_datetime(errors='coerce')` | **Drop** (`.dropna(subset=["date"])`) |
| 2 | feature_engineer | Rolling window cold-start (first N rows) | **Fill** with training statistics (`global_mean`, `global_std`) |
| 2 | feature_engineer | Division-by-zero risk in z-score | **Epsilon** (`rolling_7d_std + 1e-9`) |
| 3 | categorization_model | Missing `cleaned_remarks` during prediction | **Fill** with `""` |
| 3 | categorization_model | Missing `amount_log` during prediction | **Fill** with `0.0` |
| 4 | expected_spend_model | Missing numeric features during prediction | **Fill** with `0.0` |
| 4 | expected_spend_model | Missing `predicted_category` during prediction | **Fill** with `"uncategorized"` |
| 4 | expected_spend_model | Division-by-zero in percent_deviation | **Epsilon** (`expected_amount + 1e-5`) |
| 6 | insight_model | Missing numeric features for ranker | **Fill** with `0.0` |
| 6 | insight_model | Missing categorical features for ranker | **Fill** with `"unknown"` |
| 6 | insight_generator | Missing `insight_score` column | **Fill** column with `0.0` |

### Design Principle

- **Phase 1 (input data):** NaN means the data is corrupt. Drop it aggressively.
- **Phase 2 (computed features):** NaN is a cold-start artifact. Fill with statistically reasonable defaults.
- **Phase 3–6 (ML inference):** NaN might reach inference defensively. Fill silently because prediction must not crash.

**No module ever allows NaN to propagate to a downstream module's required columns.**

---

## 14. Cyclical Encoding for Time Features

**Location:** `feature_engineer.py`, function `add_time_features`.

### The Problem

Months are cyclical: December (12) is adjacent to January (1), but their integer representations (12 and 1) are maximally distant. A linear model would learn that December and January are far apart, when financially they are adjacent (holiday spending carries into January).

### The Solution

```python
df["month_sin"] = np.sin(2 * np.pi * df["date"].dt.month / 12)
df["month_cos"] = np.cos(2 * np.pi * df["date"].dt.month / 12)
```

This maps each month to a point on the unit circle:

| Month | month_sin | month_cos | Circle Position |
|-------|-----------|-----------|-----------------|
| Jan (1) | 0.50 | 0.87 | 30° |
| Apr (4) | 0.87 | -0.50 | 120° |
| Jul (7) | -0.50 | -0.87 | 210° |
| Oct (10) | -0.87 | 0.50 | 300° |
| Dec (12) | 0.00 | 1.00 | 360° ≈ 0° |

December (month_sin=0, month_cos=1) and January (month_sin=0.5, month_cos=0.87) are now close in feature space. The same encoding is applied to `day_of_week` with a period of 7.

### Why Both Sin AND Cos?

A single trigonometric function creates ambiguity: `sin(30°) = sin(150°)`, meaning January and June would have the same sin value. Using both sin and cos provides a unique 2D representation for each month/day.

---

## 15. The 12-Model Benchmark and Leakage Detection

**Location:** `model_benchmark.py`.

### Benchmark Architecture

12 classifiers are wrapped in identical `StandardScaler + OneHotEncoder + classifier` pipelines:

| Model | Key Config | Notes |
|-------|-----------|-------|
| LogisticRegression | class_weight='balanced' | Linear baseline |
| GradientBoosting | n_estimators=100, depth=4 | Sklearn implementation |
| RandomForest | depth=8, balanced weights | Ensemble baseline |
| LinearSVC | calibrated, dual='auto' | Linear SVM with probability calibration |
| KNeighbors | k=7, distance-weighted | Instance-based |
| DecisionTree | depth=6, balanced | Single tree (interpretability) |
| MLPClassifier | (64,32), early stopping | Neural network baseline |
| AdaBoost | 100 estimators, lr=0.5 | Boosting baseline |
| ExtraTrees | depth=8, balanced | Randomized ensemble |
| XGBoost | depth=4, lr=0.1 | Gradient boosting (optimized) |
| LightGBM | depth=4, lr=0.1, balanced | Gradient boosting (production choice) |
| CatBoost | depth=4, lr=0.1, balanced | Gradient boosting (categorical-native) |

### Leakage Detection Protocol

The benchmark runs Insight Ranking **twice**:
1. **With `is_anomaly` as a feature** (production mode)
2. **Without `is_anomaly`** (leakage test)

The `is_anomaly` flag is a pre-computed summary of other features (z-score + percent deviation). If removing it causes F1 to drop by more than 0.15, the model is over-relying on a feature that "gives away" the answer.

```python
delta = best_insight['test_f1_macro'] - best_insight_no_anom['test_f1_macro']
if delta > 0.15:
    print("⚠️ LEAKAGE WARNING: Removing is_anomaly drops F1 by {delta}")
```

This check ensures the production model can identify insights from raw feature patterns, not just from the presence of a pre-flagged anomaly bit.

### Label Encoding Handling

XGBoost, CatBoost, and MLPClassifier cannot train on string labels. The benchmark automatically detects these models and applies `LabelEncoder` for training, then `inverse_transform` for metric computation. This ensures all models are evaluated on the same string-labeled test set.

---

## 16. Graceful Degradation Architecture

The pipeline is designed to produce **useful output even when components fail or are unavailable**:

| Failure Scenario | Fallback Behavior |
|-----------------|-------------------|
| `models/insight_ranker.pkl` not found | `load_insight_ranker` returns `None` → all insight scores default to 0.0 → insights ranked by appearance order |
| Categorization model trains on < 4 samples | sklearn may warn, but still fits a model. The `class_weight='balanced'` prevents single-class domination |
| No anomalies detected in dataset | `is_anomaly` column is all False → anomaly scan produces 0 candidates → only subscription insights appear |
| No recurring transactions found | `is_recurring` column is all False → subscription scan produces 0 candidates → only anomaly insights appear |
| No anomalies AND no subscriptions | `generate_human_insights` returns an empty list → `PipelineResult.insights = []` |
| `insight_score` column missing on DataFrame | `generate_human_insights` fills it with 0.0 and continues |
| Error during `predict_proba` in insight ranker | Caught by try/except → scores default to 0.0 |

### Design Principle

**Never crash. Degrade quality, not availability.** The absence of the pre-trained LightGBM ranker means insights are unranked but still generated. The absence of anomalies means fewer insights, not an error.

---

## 17. PII Stripping in Remark Cleaning

**Location:** `preprocessor.py`, function `clean_remark`.

### What Is Stripped

| Pattern | Regex | Example Input | Result After Stripping |
|---------|-------|---------------|----------------------|
| Email addresses | `\S+@\S+` | `"txn user@email.com"` | `"txn"` |
| Long numeric sequences | `\d{4,}` | `"UPI/9876543210/swiggy"` | `"UPI//swiggy"` |
| VPA identifiers | (caught by email regex) | `"merchant@upi"` | `""` |
| Phone numbers | (caught by 4+ digit regex) | `"call 9876543210"` | `"call"` |
| Card last-4 | (caught by 4+ digit regex) | `"card 1234"` | `"card"` |

### Why 4+ Digits?

3-digit numbers (e.g., `"order 123"`) are harmless and sometimes semantically useful (order numbers, city codes). 4+ digit numbers are almost always: phone numbers, card numbers, transaction references, or account identifiers — all PII.

### Ordering Matters

The stripping order is:
1. Lowercase first
2. Email regex
3. Digit regex
4. Special character removal
5. Noise token removal
6. Whitespace collapse

If step 3 (digit removal) happened before step 2 (email removal), the email `user@123456.com` would become `user@.com` → then fail to match the email regex → partial PII leakage.

---

## 18. Merchant Alias Normalization

**Location:** `preprocessor.py`, function `_normalize_merchant`. Configuration in `config.py`, constant `MERCHANT_ALIASES`.

### How It Works

`MERCHANT_ALIASES` is an ordered dictionary of regex patterns → canonical names:

```python
MERCHANT_ALIASES = {
    r"swiggy|swigy": "swiggy",
    r"zomato|zomto": "zomato",
    r"netflix|netflx": "netflix",
    r"amazon|amzn|a]mazon": "amazon",
    # ... 20+ more
}
```

For each cleaned remark, every regex pattern is tested via `re.sub`. The first matching pattern replaces all instances.

### Why Regex?

Bank statement remarks contain misspellings, abbreviations, and inconsistent formatting:
- `"SWIGGY"`, `"Swigy"`, `"swigy order"` → all become `"swiggy"`
- `"AMZN"`, `"Amazon"`, `"a]mazon"` (OCR artifact) → all become `"amazon"`

Simple string equality would miss these variants.

### Impact on Downstream

Normalization directly affects:
1. **Seed labeling accuracy:** `"swigy"` without normalization would miss the `"swiggy"` keyword → fallback to "uncategorized" → lost training signal.
2. **Recurring detection accuracy:** `"NETFLIX.COM"` and `"netflix subscription"` without normalization would be two separate groups → neither hits the 3-occurrence minimum → subscription missed.

---

## 19. Tip Corpus Architecture

**Location:** `config.py`, constant `TIP_CORPUS`. Consumed in `insight_generator.py` and `training_data_generator.py`.

### Structure

Each tip entry:
```python
"tip_food_01": {
    "text": "Consider meal prepping on weekends to reduce food delivery expenses by up to 40%.",
    "categories": ["food"],          # Which categories this tip applies to
    "insights": ["spending_spike"],  # Which insight types trigger this tip
}
```

### Selection Priority

`_select_tip(category, insight_type)` in `insight_generator.py`:

```
1. Category-specific match: tip.categories contains the category AND
                           tip.insights contains the insight_type
   → Return random.choice(candidates)

2. Generic match: tip.categories is empty AND
                  tip.insights contains the insight_type
   → Return random.choice(candidates)

3. No match → return ""
```

### Why `random.choice` Instead of Deterministic?

Variety. If the same user runs the pipeline repeatedly, they should see different tips. The randomness is at the tip level only — the underlying insights and their scores remain deterministic.

### Tip Corpus in Training

`training_data_generator.py` uses `_find_best_tip` (which is deterministic, unlike `_select_tip`) to assign `tip_id` labels to synthetic data. This allows training a Tip Selector model that predicts the best tip for a given (category, insight_type) pair, eventually replacing the rule-based selection with ML-based selection.

### Tip Coverage Guarantee

`tests/test_benchmark.py::test_actionable_insights_have_tips` verifies that < 5% of actionable insights have `"no_tip"` in the synthetic dataset. This ensures the tip corpus has sufficient coverage across all category × insight_type combinations.
