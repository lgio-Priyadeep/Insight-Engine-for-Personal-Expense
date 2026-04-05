<![CDATA[# API Reference — Insight Engine

> *Every public function in the pipeline, its contract, its arguments, and what it promises.*

---

## Table of Contents

- [preprocessor](#preprocessor)
- [feature_engineer](#feature_engineer)
- [seed_labeler](#seed_labeler)
- [categorization_model](#categorization_model)
- [expected_spend_model](#expected_spend_model)
- [anomaly_detector](#anomaly_detector)
- [recurring_detector](#recurring_detector)
- [insight_generator](#insight_generator)

---

## preprocessor

### `validate_schema(df: pd.DataFrame) → None`

Asserts that all required columns (`date`, `amount`, `amount_flag`, `remarks`) are present.

| Parameter | Type | Description |
|---|---|---|
| `df` | `pd.DataFrame` | Raw input DataFrame |

**Raises**: `ValueError` if any required column is missing (message lists the missing columns).

**Side effects**: Logs `"Schema validation passed."` on success.

---

### `clean_remark(remark) → str`

Cleans a single transaction remark string.

| Parameter | Type | Description |
|---|---|---|
| `remark` | `Any` | Raw remark value (handles non-string gracefully) |

**Returns**: Cleaned, lowercased string. Empty string `""` if nothing meaningful remains.

**Cleaning order**: lowercase → merchant alias matching → email removal → long-number removal → special char removal → whitespace collapse → noise token removal.

**Contract**: If a specific merchant alias matches (e.g., Swiggy), returns the alias immediately. If only generic routing tags match (UPI Transfer, Bank Transfer), strips them and continues standard cleaning. Unmapped merchants are **never** silently destroyed.

---

### `preprocess(df: pd.DataFrame) → Tuple[pd.DataFrame, pd.DataFrame]`

Full preprocessing pipeline.

| Parameter | Type | Description |
|---|---|---|
| `df` | `pd.DataFrame` | Raw transaction DataFrame (must contain `date`, `amount`, `amount_flag`, `remarks`) |

**Returns**: `(debit_df, credit_df)` — two independently indexed, cleaned DataFrames.

**Guarantees**:
- `date` is `datetime64`, sorted chronologically
- `amount_flag` ∈ `{"DR", "CR"}`
- `signed_amount` column exists (`-abs` for DR, `+abs` for CR)
- `cleaned_remarks` column exists
- Zero-amount rows removed
- Exact duplicates removed
- Input DataFrame is **not mutated**

**Raises**: `ValueError` on schema mismatch or unparseable dates.

---

## feature_engineer

### `add_time_features(df: pd.DataFrame) → pd.DataFrame`

Adds calendar-based time features.

| Parameter | Type | Description |
|---|---|---|
| `df` | `pd.DataFrame` | Must have `date` column of dtype `datetime64` |

**Adds columns**: `is_weekend`, `week_of_month`, `month_sin`, `month_cos`, `dow_sin`, `dow_cos`

**Raises**: `TypeError` if `date` is not `datetime64`.

---

### `add_rolling_features(df: pd.DataFrame, amount_col: str = "amount") → pd.DataFrame`

Adds leakage-safe rolling statistics via `shift(1)`.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `df` | `pd.DataFrame` | — | Must be sorted chronologically |
| `amount_col` | `str` | `"amount"` | Column to compute rolling stats on |

**Adds columns**: `rolling_7d_mean`, `rolling_30d_mean`, `rolling_7d_std`

**NaN behavior**: Row 0 will have NaN for all rolling columns (shift(1) creates an empty window). NaN values are intentionally **left for the caller** to fill with training-set statistics via `fill_rolling_nulls()`.

---

### `fill_rolling_nulls(df: pd.DataFrame, global_mean: float, global_std: float) → pd.DataFrame`

Fills NaN values in rolling columns using externally provided statistics.

| Parameter | Type | Description |
|---|---|---|
| `df` | `pd.DataFrame` | After `add_rolling_features()` |
| `global_mean` | `float` | Training-set mean (must not be computed from test data) |
| `global_std` | `float` | Training-set std (falls back to `1.0` if ≤ 0) |

**Guarantee**: No NaN values remain in rolling columns after this call.

---

### `add_amount_features(df: pd.DataFrame, amount_col: str = "amount") → pd.DataFrame`

Adds derived amount features. Requires `rolling_7d_mean` and `rolling_7d_std` to be present.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `df` | `pd.DataFrame` | — | Must have rolling columns filled |
| `amount_col` | `str` | `"amount"` | Column to derive features from |

**Adds columns**: `amount_log` (≥ 0), `amount_zscore` (clipped to [-5, 5])

**Safety**: Zero-std rows use `std=1.0` for z-score computation (prevents ±∞).

---

### `engineer_features(df, global_mean=None, global_std=None, amount_col="amount") → pd.DataFrame`

Full feature engineering pipeline. Calls all sub-functions in order.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `df` | `pd.DataFrame` | — | Cleaned DataFrame (output of `preprocess()`) |
| `global_mean` | `Optional[float]` | `None` | Training-set mean. If None, auto-computed from `df` (with warning) |
| `global_std` | `Optional[float]` | `None` | Training-set std. If None, auto-computed from `df` (with warning) |
| `amount_col` | `str` | `"amount"` | Column to engineer features from |

**Returns**: Feature-enriched DataFrame. All original columns preserved; 11 new columns added.

**Guarantee**: No NaN values in any feature column. Output is sorted chronologically.

**Warning**: If `global_mean`/`global_std` are `None`, a warning is logged. This is acceptable for inference but indicates potential leakage for training splits.

---

### `engineer_features_inference(new_txn, history_df, global_mean, global_std, amount_col="amount") → pd.DataFrame`

Live inference entry point. Computes features for new transactions using historical context.

| Parameter | Type | Description |
|---|---|---|
| `new_txn` | `pd.DataFrame` | New transaction(s) to process |
| `history_df` | `pd.DataFrame` | User's past transactions (≥30 recommended for rolling accuracy) |
| `global_mean` | `float` | Training-set mean |
| `global_std` | `float` | Training-set std |
| `amount_col` | `str` | Column name for amounts |

**Returns**: DataFrame with feature columns, containing **only** the rows from `new_txn` (not the history).

---

## seed_labeler

### `label_debits(df, remark_col="cleaned_remarks", label_col="pseudo_label", priority_order=None, keyword_map=None) → pd.DataFrame`

Assigns pseudo-labels to debit transactions.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `df` | `pd.DataFrame` | — | Debit DataFrame (output of `preprocess()[0]`) |
| `remark_col` | `str` | `"cleaned_remarks"` | Column with cleaned text |
| `label_col` | `str` | `"pseudo_label"` | Name of the new label column |
| `priority_order` | `Optional[list[str]]` | `CATEGORY_PRIORITY` | Category evaluation order |
| `keyword_map` | `Optional[dict[str, list[str]]]` | `CATEGORY_KEYWORDS` | Keyword dictionary |

**Adds columns**: `pseudo_label`, `all_matched_categories`

**Fallback**: Unmatched rows receive `"uncategorized"`.

---

### `label_credits(df, remark_col="cleaned_remarks", label_col="pseudo_label", priority_order=None, keyword_map=None) → pd.DataFrame`

Assigns pseudo-labels to credit transactions (separate keyword map, separate fallback).

Same parameters as `label_debits()` but defaults to `CREDIT_PRIORITY` and `CREDIT_KEYWORDS`.

**Fallback**: Unmatched rows receive `"other_credit"` (not `"uncategorized"`).

---

## categorization_model

### `build_categorization_pipeline() → Pipeline`

Constructs the scikit-learn pipeline (TF-IDF + Logistic Regression). Does not fit it.

**Returns**: Unfitted `sklearn.pipeline.Pipeline`.

---

### `train_categorization_model(df: pd.DataFrame, label_col: str = "pseudo_label") → Pipeline`

Trains the categorization model.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `df` | `pd.DataFrame` | — | Must have `cleaned_remarks`, `amount_log`, and `label_col` |
| `label_col` | `str` | `"pseudo_label"` | Column with training labels |

**Returns**: Fitted `Pipeline`.

**Exclusions**: Rows with `"uncategorized"` or `"other_credit"` labels are excluded from training.

---

### `predict_categories(pipeline: Pipeline, df: pd.DataFrame) → pd.DataFrame`

Predicts categories for a DataFrame.

| Parameter | Type | Description |
|---|---|---|
| `pipeline` | `Pipeline` | Fitted categorization pipeline |
| `df` | `pd.DataFrame` | Must have `cleaned_remarks` and `amount_log` |

**Adds columns**: `predicted_category`, `category_confidence`

**NaN safety**: Missing `cleaned_remarks` filled with `""`, missing `amount_log` filled with `0.0`.

---

## expected_spend_model

### `train_expected_spend_model(df: pd.DataFrame, target_col: str = "amount") → Pipeline`

Trains the expected spend regressor.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `df` | `pd.DataFrame` | — | Must have time features, rolling features, and `predicted_category` |
| `target_col` | `str` | `"amount"` | Regression target |

**Returns**: Fitted `Pipeline`.

---

### `predict_expected_spend(pipeline: Pipeline, df: pd.DataFrame) → pd.DataFrame`

Predicts expected amounts and computes residuals.

| Parameter | Type | Description |
|---|---|---|
| `pipeline` | `Pipeline` | Fitted expected spend pipeline |
| `df` | `pd.DataFrame` | Must have same features as training |

**Adds columns**: `expected_amount`, `residual`, `percent_deviation`

---

## anomaly_detector

### `detect_anomalies(df, zscore_threshold=3.0, pct_dev_threshold=0.5) → pd.DataFrame`

Flags anomalous transactions using composite dual-gate logic.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `df` | `pd.DataFrame` | — | Must have `amount_zscore`, `percent_deviation`, `amount` |
| `zscore_threshold` | `float` | `3.0` | Statistical outlier gate |
| `pct_dev_threshold` | `float` | `0.5` | ML deviation gate (50%) |

**Adds column**: `is_anomaly` (bool)

**Raises**: `ValueError` if required columns are missing.

---

## recurring_detector

### `find_recurring_transactions(df, group_col="cleaned_remarks", amount_tolerance=0.05) → pd.DataFrame`

Identifies recurring (subscription-like) transaction patterns.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `df` | `pd.DataFrame` | — | Must have `date`, `amount`, and `group_col` |
| `group_col` | `str` | `"cleaned_remarks"` | Column to group transactions by |
| `amount_tolerance` | `float` | `0.05` | Max allowed amount drift (5%) |

**Adds columns**: `is_recurring` (bool), `recurring_frequency` (`"monthly"`, `"weekly"`, or `None`)

---

## insight_generator

### `generate_human_insights(df: pd.DataFrame) → List[str]`

Generates human-readable spending insight strings from a fully analyzed DataFrame.

| Parameter | Type | Description |
|---|---|---|
| `df` | `pd.DataFrame` | Must have `date`, `cleaned_remarks`, `predicted_category`, `amount`, plus `is_recurring`/`is_anomaly` flags |

**Returns**: `List[str]` — Formatted insight strings with ₹ currency symbols.

**Categories of insights**: Subscription identification, anomaly alerts, category-specific optimization tips.

---

*Last updated: March 2026*
]]>
