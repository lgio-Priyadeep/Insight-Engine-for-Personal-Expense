<![CDATA[# Architecture — Insight Engine

> *This document is for developers who will modify, extend, or debug this pipeline.*
> *Read the [README](../README.md) first for a high-level overview.*

---

## Table of Contents

- [System Overview](#system-overview)
- [Data Flow Contract](#data-flow-contract)
- [Module-by-Module Deep Dive](#module-by-module-deep-dive)
  - [config.py — Domain Knowledge Registry](#configpy--domain-knowledge-registry)
  - [preprocessor.py — Data Integrity Gate](#preprocessorpy--data-integrity-gate)
  - [feature_engineer.py — Leakage-Safe Signal Extraction](#feature_engineerpy--leakage-safe-signal-extraction)
  - [seed_labeler.py — Deterministic Pseudo-Label Generator](#seed_labelerpy--deterministic-pseudo-label-generator)
  - [categorization_model.py — ML Categorization](#categorization_modelpy--ml-categorization)
  - [expected_spend_model.py — Spending Baseline Regressor](#expected_spend_modelpy--spending-baseline-regressor)
  - [anomaly_detector.py — Composite Anomaly Flagging](#anomaly_detectorpy--composite-anomaly-flagging)
  - [recurring_detector.py — Subscription Identification](#recurring_detectorpy--subscription-identification)
  - [insight_model.py — LightGBM Insight Ranker](#insight_modelpy--lightgbm-insight-ranker)
  - [insight_generator.py — Human-Readable Synthesis](#insight_generatorpy--human-readable-synthesis)
- [Column Lifecycle](#column-lifecycle)
- [Leakage Prevention Strategy](#leakage-prevention-strategy)
- [Security & Privacy Considerations](#security--privacy-considerations)
- [Error Handling Philosophy](#error-handling-philosophy)
- [Testing Architecture](#testing-architecture)
- [Extension Guide](#extension-guide)
- [Failure Modes & Recovery](#failure-modes--recovery)

---

## System Overview

The Insight Engine is a **synchronous, single-pass analytical pipeline** that transforms raw Indian bank statement CSV data into structured spending insights. It is not a web service, not a daemon, and not a streaming system. It processes a batch of transactions, learns from them, and outputs results.

### Design Principles

1. **Immutability**: Every function operates on `.copy()` of its input. No function mutates its arguments. This eliminates an entire class of debugging nightmares.

2. **Fail-Loud**: Schema violations, type mismatches, and missing columns raise `ValueError` or `TypeError` immediately. The pipeline does not silently impute or guess.

3. **Separation of Concerns**: Domain knowledge lives in `config.py`. Data cleaning lives in `preprocessor.py`. Feature math lives in `feature_engineer.py`. They never cross boundaries.

4. **Leakage-Awareness**: Every rolling computation uses `shift(1)` to ensure row *i* never sees its own value. NaN fills accept external statistics, never self-derived ones.

5. **Indian Market Specificity**: Merchant aliases, UPI routing patterns, ₹ formatting, and category taxonomies are all designed for Indian transaction data.

---

## Data Flow Contract

This is the pipeline's **invariant chain** — the guarantee each module makes to the next.

```
                    ┌─────────────────┐
                    │   Raw CSV/DF    │
                    │  Columns: date, │
                    │  amount,        │
                    │  amount_flag,   │
                    │  remarks        │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  preprocessor   │
                    │   .preprocess() │
                    │                 │
                    │ GUARANTEES:     │
                    │ • date is       │
                    │   datetime64    │
                    │ • amount_flag   │
                    │   ∈ {DR, CR}    │
                    │ • signed_amount │
                    │   exists        │
                    │ • cleaned_      │
                    │   remarks       │
                    │   exists        │
                    │ • no zero-      │
                    │   amount rows   │
                    │ • no exact      │
                    │   duplicates    │
                    │ • chronological │
                    │   sort          │
                    └───┬─────────┬───┘
                        │         │
                   debits_df  credits_df
                        │         │
            ┌───────────▼──┐  ┌───▼───────────┐
            │ seed_labeler │  │ seed_labeler  │
            │.label_debits │  │.label_credits │
            │              │  │               │
            │ ADDS:        │  │ ADDS:         │
            │ pseudo_label │  │ pseudo_label  │
            │ all_matched_ │  │ all_matched_  │
            │ categories   │  │ categories    │
            └──────┬───────┘  └───────────────┘
                   │
            ┌──────▼───────┐
            │ feature_     │
            │ engineer     │
            │ .engineer_   │
            │ features()   │
            │              │
            │ ADDS:        │
            │ is_weekend   │
            │ week_of_month│
            │ month_sin/cos│
            │ dow_sin/cos  │
            │ rolling_7d_  │
            │   mean       │
            │ rolling_30d_ │
            │   mean       │
            │ rolling_7d_  │
            │   std        │
            │ amount_log   │
            │ amount_zscore│
            │              │
            │ GUARANTEES:  │
            │ • No NaN in  │
            │   any feature│
            │ • Chron sort │
            └──────┬───────┘
                   │
            ┌──────▼───────┐
            │ categorize.  │
            │ train → pred │
            │              │
            │ ADDS:        │
            │ predicted_   │
            │  category    │
            │ category_    │
            │  confidence  │
            └──────┬───────┘
                   │
            ┌──────▼───────┐
            │ expected_    │
            │ spend.       │
            │ train → pred │
            │              │
            │ ADDS:        │
            │ expected_    │
            │  amount      │
            │ residual     │
            │ percent_     │
            │  deviation   │
            └──────┬───────┘
                   │
            ┌──────▼───────┐
            │ anomaly_     │
            │ detector     │
            │              │
            │ ADDS:        │
            │ is_anomaly   │
            └──────┬───────┘
                   │
            ┌──────▼───────┐
            │ recurring_   │
            │ detector     │
            │              │
            │ ADDS:        │
            │ is_recurring │
            │ recurring_   │
            │ frequency    │
            └──────┬───────┘
                   │
            ┌──────▼───────┐
            │ insight_model│
            │ .predict_    │
            │  scores()    │
            │              │
            │ ADDS:        │
            │ insight_score│
            └──────┬───────┘
                   │
            ┌──────▼───────┐
            │ insight_     │
            │ generator    │
            │              │
            │ RETURNS:     │
            │ List[str]    │
            │ (Top N)      │
            └──────────────┘
```

**Critical Rule**: Each module's output is a **superset** of its input. Columns are only ever added, never removed. This means a downstream module can always access any column created by an upstream module.

---

## Module-by-Module Deep Dive

### `config.py` — Domain Knowledge Registry

**Lines**: 320 | **Dependencies**: None | **Tested by**: Implicitly via all other test files

This is the **single source of truth** for all domain knowledge. If you need to change how the pipeline categorizes or recognizes merchants, this is the only file you touch.

#### `MERCHANT_ALIASES` (dict[str, str])

Maps regex patterns to normalized merchant names. **Order matters** — the preprocessor evaluates all patterns against a remark and uses priority logic to resolve conflicts.

```python
# Pattern → Normalized Name
r"swiggy"       → "Swiggy"
r"gpay|google\s?pay" → "Google Pay"
```

**Design decision**: Patterns are compiled at each call, not pre-compiled. This keeps the config serializable (no `re.compile` objects) at a negligible per-transaction cost.

**Gotcha**: The pattern `r"ola\b"` uses `\b` (word boundary) to prevent matching "Ola" inside "Polaroid" or "Scholarship". If you add new patterns, consider word boundaries carefully.

#### `CATEGORY_KEYWORDS` / `CREDIT_KEYWORDS`

Flat lists of keywords per category. Multi-word keywords (e.g., `"cash withdrawal"`) are matched by checking that **all tokens** appear in the remark's token set. This means ordering within the keyword doesn't matter — `"withdrawal cash"` in a remark would still match `"cash withdrawal"`.

#### `CATEGORY_PRIORITY` / `CREDIT_PRIORITY`

Ordered lists controlling tie-breaking. Position 0 has the highest priority.

**Why this matters**: A remark like `"emi hospital"` matches both `finance` (via `"emi"`) and `health` (via `"hospital"`). Without explicit priority, the result is non-deterministic. The priority list makes it deterministic and auditable.

---

### `preprocessor.py` — Data Integrity Gate

**Lines**: 264 | **Dependencies**: `config.py` (NOISE_TOKENS, MERCHANT_ALIASES) | **Tested by**: `test_phase1.py`

This module's job is to transform raw, messy bank data into a clean, trustworthy DataFrame. It is the most critical module in the pipeline — if it fails silently, everything downstream is corrupt.

#### Public API

```python
def preprocess(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (debit_df, credit_df) — both cleaned, sorted, deduped."""
```

#### Internal Steps (in order)

1. **`validate_schema(df)`** — Asserts `{date, amount, amount_flag, remarks}` exist. Raises `ValueError` with a clear message listing missing columns.

2. **`_parse_and_sort_dates(df)`** — Parses `date` column with `dayfirst=True` (Indian date format: DD/MM/YYYY). Sorts chronologically. Logs the date range.

3. **`_compute_signed_amount(df)`** — Normalizes `amount_flag` to `{"DR", "CR"}` via `_normalize_flag()`. Invalid flags are defaulted to `"DR"` with a warning (not dropped — we don't lose data). Computes `signed_amount = -abs(amount)` for DR, `+abs(amount)` for CR.

4. **`_drop_zero_amount(df)`** — Removes rows where `amount == 0` (pass-through entries, opening balance markers).

5. **`_deduplicate(df)`** — Drops exact duplicates on `(date, amount, remarks, amount_flag)`. Keeps first occurrence.

6. **`clean_remark()` applied** — Creates `cleaned_remarks` column. This is the most complex step.

7. **`_split_debit_credit(df)`** — Splits into two DataFrames, independently re-indexed.

#### `clean_remark()` — The Heart of Preprocessing

This function deserves its own section because it handles the ugliest part of Indian bank data: the `remarks` field.

**A real bank remark might look like:**
```
UPI/9812345678/swiggy@paytm/OrderID-3847593/Ref: TXN98765
```

**`clean_remark()` does this:**

1. **Guard**: Non-string or empty → return `""`
2. **Lowercase**: `"UPI/9812345678/swiggy@paytm"` → `"upi/9812345678/swiggy@paytm"`
3. **Merchant alias matching**: Check all `MERCHANT_ALIASES` patterns against the text
   - If a **specific** merchant is found (e.g., Swiggy), return it immediately
   - If only **generic** routing tags (e.g., "UPI Transfer", "Bank Transfer") match, **strip the generic pattern from the text** and continue cleaning. This is critical — without this, every UPI transaction would be flattened to "upi transfer", destroying the actual merchant identity.
4. **Strip emails**: Remove `user@email.com` patterns
5. **Strip long numbers**: Remove 4+ digit sequences (account numbers, reference IDs, UPI routing metadata)
6. **Strip special characters**: Keep only `[a-z0-9\s]`
7. **Collapse whitespace**: Multiple spaces → single space
8. **Remove noise tokens**: Strip semantically empty words (`"ref"`, `"no"`, `"by"`, `"payment"`, `"txn"`, etc.)
9. **Remove single-character tokens**: `"a"`, `"i"` — too noisy
10. **Return**: The cleaned string, or `""` if nothing meaningful remains

**The unmapped merchant survival rule**: If a UPI transaction involves a merchant not in `MERCHANT_ALIASES` (e.g., `"RajuTeaStall"`), the function MUST NOT discard it. It strips the generic UPI routing noise and preserves `"rajuteastall"`. This is tested explicitly in `test_phase1.py::TestCleanRemark::test_unmapped_merchant_survives_generic_routing`.

---

### `feature_engineer.py` — Leakage-Safe Signal Extraction

**Lines**: 282 | **Dependencies**: `numpy`, `pandas` | **Tested by**: `test_phase1.py`

#### Feature Groups

| Feature | Type | Computation | Purpose |
|---|---|---|---|
| `is_weekend` | int (0/1) | `dayofweek ∈ {5, 6}` | Weekend spending patterns differ |
| `week_of_month` | int (1–5) | `(day - 1) // 7 + 1` | Early-month vs. late-month behavior |
| `month_sin`, `month_cos` | float [-1, 1] | `sin/cos(2π × month / 12)` | Cyclical month encoding — prevents Jan↔Dec cliff |
| `dow_sin`, `dow_cos` | float [-1, 1] | `sin/cos(2π × dow / 7)` | Cyclical day-of-week encoding |
| `rolling_7d_mean` | float | `shift(1).rolling(7, min_periods=1).mean()` | Recent spending trend |
| `rolling_30d_mean` | float | `shift(1).rolling(30, min_periods=1).mean()` | Broader spending trend |
| `rolling_7d_std` | float | `shift(1).rolling(7, min_periods=2).std()` | Recent spending volatility |
| `amount_log` | float ≥ 0 | `log1p(abs(amount))` | Compressed scale, handles negatives |
| `amount_zscore` | float [-5, 5] | `(amount − rolling_mean) / rolling_std` | Statistical outlier detection, clipped |

#### Why Cyclical Encoding?

Linear month encoding (1, 2, ..., 12) creates an artificial cliff between December (12) and January (1). The model would learn that "December is as far from January as it is from July," which is wrong. Sine/cosine encoding maps months onto a circle where December and January are neighbors.

#### The `shift(1)` Guarantee

```python
# WRONG (leaks current row into its own rolling window):
df["rolling_7d_mean"] = df["amount"].rolling(7).mean()

# CORRECT (row i only sees rows 0..i-1):
shifted = df["amount"].shift(1)
df["rolling_7d_mean"] = shifted.rolling(7, min_periods=1).mean()
```

This is tested in `test_phase1.py::TestRollingFeatures::test_first_row_rolling_mean_excludes_itself` — row 0's rolling mean must be NaN (empty window after shift).

#### NaN Fill Strategy

Rolling windows create NaN values for early rows (not enough history). These are filled with `global_mean` and `global_std` parameters:

- **During training**: These must be computed from the training set *before* any train/test split
- **During inference**: The same training-set values must be passed in

If `global_std` is 0 (constant-value series), it falls back to `1.0` to prevent division-by-zero in downstream z-score computation.

#### `engineer_features_inference()` — Live Inference Entry Point

For real-time transaction analysis, single-row DataFrames produce degenerate rolling statistics (everything defaults to global mean). This function **prepends the user's recent transaction history** before computing features, then slices only the new transactions from the output:

```python
combined = pd.concat([history_df, new_txn])
engineered = engineer_features(combined, global_mean, global_std)
return engineered.tail(len(new_txn))  # Only the new ones
```

This gives the new transaction accurate rolling context without retraining the model.

---

### `seed_labeler.py` — Deterministic Pseudo-Label Generator

**Lines**: 224 | **Dependencies**: `config.py` | **Tested by**: `test_phase1.py`

Pseudo-labels are the **ground truth substitute** for this pipeline. We don't have hand-labeled training data, so we create labels by matching cleaned remarks against keyword dictionaries.

#### Matching Strategy

```
For each category in priority_order:
    For each keyword in that category:
        Tokenize the keyword: "cash withdrawal" → {"cash", "withdrawal"}
        If ALL keyword tokens ∈ remark tokens:
            → Match found
            → Record this category
            → If it's the first match by priority: mark as winner
```

**Why token-set matching instead of substring?** Because `"ola"` as a substring would match `"scholarship"`. Token-level matching requires `"ola"` to be a standalone word.

#### Coverage Metric

After labeling, the module computes and logs coverage:

```
[debits] Labeling coverage: 37/50 (74.0%)
```

If coverage drops below `MIN_COVERAGE_THRESHOLD` (default 40%), a warning fires. Low coverage means the keyword dictionary is inadequate for the dataset — the model will have too few training examples to generalize.

---

### `categorization_model.py` — ML Categorization

**Lines**: 122 | **Dependencies**: `scikit-learn`, `config.py` | **Tested by**: `test_phase2.py`

#### Pipeline Architecture

```
┌─────────────────────────────────────────────────────┐
│ ColumnTransformer                                   │
│ ├─ "text": TfidfVectorizer(ngram_range=(1,2),      │
│ │          max_features=2000)                       │
│ │          ↑ Applied to: cleaned_remarks            │
│ │                                                   │
│ └─ "num":  StandardScaler(with_mean=False)          │
│            ↑ Applied to: amount_log                 │
│            ↑ with_mean=False prevents sparse→dense  │
│              explosion (TF-IDF output is sparse)    │
│                                                     │
│ sparse_threshold=1.0 → Force output to stay sparse  │
│                        (prevents OOM on large data) │
└────────────────────┬────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────┐
│ LogisticRegression                                  │
│   class_weight='balanced'                           │
│   max_iter=1000                                     │
│                                                     │
│ Why balanced? Prevents "predict everything as food" │
│ syndrome. Minority classes (health, finance) get    │
│ appropriately boosted.                              │
└─────────────────────────────────────────────────────┘
```

#### Training Exclusions

Rows with fallback labels (`"uncategorized"`, `"other_credit"`) are **excluded from training**. Rationale: these labels carry no signal — they mean "we don't know." Training on them would bias the model toward predicting "uncategorized," which is the opposite of helpful.

#### Output Columns

- `predicted_category`: The model's best guess
- `category_confidence`: Maximum class probability (softmax output)

---

### `expected_spend_model.py` — Spending Baseline Regressor

**Lines**: 125 | **Dependencies**: `scikit-learn` | **Tested by**: `test_phase2.py`

This model answers: *"How much would this user normally spend on this type of transaction, at this time, given their recent history?"*

#### Pipeline Architecture

```
┌──────────────────────────────────────────┐
│ ColumnTransformer                        │
│ ├─ "num": StandardScaler()               │
│ │   Features: is_weekend, month_sin/cos, │
│ │   dow_sin/cos, week_of_month,          │
│ │   rolling_7d/30d_mean, rolling_7d_std  │
│ │                                        │
│ └─ "cat": OneHotEncoder(                 │
│       handle_unknown='ignore')           │
│   Feature: predicted_category            │
└────────────────┬─────────────────────────┘
                 │
┌────────────────▼─────────────────────────┐
│ RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0]) │
│                                          │
│ Why Ridge, not Random Forest?            │
│ → Trees CANNOT extrapolate.              │
│ → If max training amount = ₹500,         │
│   a tree will never predict > ₹500.      │
│ → Ridge extrapolates linearly.           │
│ → This is CRITICAL for anomaly           │
│   detection to work correctly.           │
└──────────────────────────────────────────┘
```

#### Output Columns

- `expected_amount`: The model's prediction of "normal" spend
- `residual`: `amount - expected_amount`
- `percent_deviation`: `residual / (expected_amount + ε)` where `ε = 1e-5` prevents divide-by-zero

---

### `anomaly_detector.py` — Composite Anomaly Flagging

**Lines**: 46 | **Dependencies**: None (beyond pandas) | **Tested by**: `test_phase3.py`

#### Dual-Gate Logic

```python
is_anomaly = (|amount_zscore| > zscore_threshold) AND (|percent_deviation| > pct_dev_threshold)
```

**Default thresholds**: `zscore_threshold=3.0`, `pct_dev_threshold=0.5` (50%)

**Why dual-gate?**

| Scenario | Z-Score | % Deviation | Flagged? | Why |
|---|---|---|---|---|
| ₹30 chai (usually ₹20) | 3.5 | 50% | No (borderline) | ₹10 over-spend is trivial |
| ₹2,500 Amazon (usually ₹120) | 8.2 | 340% | **Yes** | Massive, contextually significant |
| ₹50 Uber (usually ₹50) | 0.0 | 0% | No | Normal |

Both conditions must be true. This dramatically reduces false positives without missing genuine anomalies.

---

### `recurring_detector.py` — Subscription Identification

**Lines**: 82 | **Dependencies**: None (beyond pandas/numpy) | **Tested by**: `test_phase3.py`

#### Detection Criteria

A group of transactions (grouped by `cleaned_remarks`) is flagged as recurring if:

1. **Count ≥ 2**: Need at least two occurrences to establish a pattern
2. **Time gap consistency**:
   - Monthly: mean gap between 27–33 days, variance < 10
   - Weekly: mean gap between 6–8 days, variance < 3
3. **Amount stability**: `(max - min) / mean ≤ amount_tolerance` (default 5%)

#### Why These Bounds?

- **27–33 days (monthly)**: Accounts for months with 28–31 days, plus slight processing delays
- **6–8 days (weekly)**: Buffer for weekend processing delays
- **5% amount tolerance**: Covers tax adjustments, micro-rounding, promotional pricing changes

**Current limitation**: Only detects weekly and monthly frequencies. Biweekly, quarterly, and annual patterns are not covered.

---

### `insight_model.py` — LightGBM Insight Ranker

**Dependencies**: `lightgbm` | **Tested by**: `test_ml_integration.py`

This module addresses the "Alert Fatigue" problem. Without it, the pipeline would generate a text string for *every single* mathematical anomaly and subscription.

#### Modeling Architecture
It loads a pre-trained offline LightGBM model trained on 5,500 highly engineered synthetic scenarios. It uses all numerical and cyclical features + the predicted spend category to score each row's likelihood (`1.0 - P(no_action)`) of being an actionable insight.

---

### `insight_generator.py` — Human-Readable Synthesis

**Dependencies**: None (beyond pandas) | **Tested by**: `test_phase3.py`

Transforms boolean flags and numeric features into strings a human can read and act on.

#### Output Categories

**1. Subscription insights** (from `is_recurring`):
```
"Subscription identified: 'Netflix' usually charged monthly for roughly ₹199.00."
```

**2. Anomaly insights** (from `is_anomaly`):
```
"Unusual shopping expense detected at 'Amazon' (₹2,500.00). This is 340.2% above your normal expected baseline."
```

**3. Category-specific optimization tips**:
```
"> Tip: A single ₹500 substitution per week could offset this spike!"     # food anomalies
"> Tip: Consider imposing a soft daily cap when browsing retail sites."    # shopping anomalies
```

**Design note**: The ₹ symbol is hardcoded. Multi-currency support would require a currency-aware formatting layer.

---

## Column Lifecycle

This table tracks every column — when it's created, who creates it, and who consumes it.

| Column | Created By | Consumed By | Type | Can Be NaN? |
|---|---|---|---|---|
| `date` | Input CSV | preprocessor, feature_engineer, recurring_detector | datetime64 | No (after preprocess) |
| `amount` | Input CSV | feature_engineer, expected_spend, anomaly_detector | float | No (zeros dropped) |
| `amount_flag` | Input CSV | preprocessor (normalized to DR/CR) | str | No (invalid → DR) |
| `remarks` | Input CSV | preprocessor (cleaned) | str | Tolerated |
| `signed_amount` | preprocessor | feature_engineer | float | No |
| `cleaned_remarks` | preprocessor | seed_labeler, categorization_model, recurring_detector, insight_generator | str | Can be "" |
| `pseudo_label` | seed_labeler | categorization_model (training target) | str | No (fallback assigned) |
| `all_matched_categories` | seed_labeler | (Debug/analysis) | list[str] | No (can be []) |
| `is_weekend` | feature_engineer | expected_spend_model | int (0/1) | No |
| `week_of_month` | feature_engineer | expected_spend_model | int (1-5) | No |
| `month_sin`, `month_cos` | feature_engineer | expected_spend_model | float [-1, 1] | No |
| `dow_sin`, `dow_cos` | feature_engineer | expected_spend_model | float [-1, 1] | No |
| `rolling_7d_mean` | feature_engineer | feature_engineer (zscore), expected_spend | float | No (after fill) |
| `rolling_30d_mean` | feature_engineer | expected_spend_model | float | No (after fill) |
| `rolling_7d_std` | feature_engineer | feature_engineer (zscore), expected_spend | float | No (after fill, min 1.0) |
| `amount_log` | feature_engineer | categorization_model | float ≥ 0 | No |
| `amount_zscore` | feature_engineer | anomaly_detector | float [-5, 5] | No |
| `predicted_category` | categorization_model | expected_spend_model, insight_generator | str | No (model always predicts) |
| `category_confidence` | categorization_model | (Analysis/filtering) | float [0, 1] | No |
| `expected_amount` | expected_spend_model | anomaly_detector (via residual) | float | No |
| `residual` | expected_spend_model | (Debug/analysis) | float | No |
| `percent_deviation` | expected_spend_model | anomaly_detector | float | No |
| `is_anomaly` | anomaly_detector | insight_generator | bool | No |
| `is_recurring` | recurring_detector | insight_generator | bool | No |
| `recurring_frequency` | recurring_detector | insight_generator | str/None | Yes (None if not recurring) |
| `insight_score` | insight_model | insight_generator | float [0, 1] | No (after inference) |

---

## Leakage Prevention Strategy

Data leakage is the most insidious bug in ML pipelines. It inflates training metrics, creates false confidence, and causes production failures that are nearly impossible to diagnose.

### Where Leakage Could Occur

| Vector | How We Prevent It |
|---|---|
| Rolling features include current row | `shift(1)` applied before `.rolling()` |
| NaN fill uses test-set statistics | `fill_rolling_nulls()` accepts external `global_mean`/`global_std` — caller must compute from training set only |
| Model trained on fallback labels | `train_categorization_model()` excludes `uncategorized` and `other_credit` rows |
| Feature engineering recomputes on full dataset | `engineer_features()` warns when `global_mean`/`global_std` are None (auto-computed from input) — acceptable for inference, not for training |

### For Future Developers

If you add new features:
1. **Ask**: "Does this feature use information from the future (relative to row *i*)?"
2. **Ask**: "Does this feature use information from the test set?"
3. If either answer is yes, **fix it before merging**.

---

## Security & Privacy Considerations

### What Gets Stripped

| PII Element | Regex Pattern | Example |
|---|---|---|
| Email addresses | `\S+@\S+` | `user@bank.com` → removed |
| Account/reference numbers | `\d{4,}` | `9812345678` → removed |
| UPI routing metadata | (via merchant alias matching) | `UPI/9812345678/swiggy@paytm` → `"swiggy"` |

### What Gets Preserved

- **Merchant identity** (after aliasing): e.g., `"Swiggy"`, `"Netflix"`
- **Transaction amounts**: Required for all downstream analysis
- **Dates**: Required for temporal features
- **Cleaned remarks**: After PII stripping, these become the training signal

### Considerations for Production Deployment

> **⚠️ Warning**: This pipeline processes **financial transaction data**. Before deploying to production, consider:

1. **Data at rest**: Transaction CSVs and DataFrames should be encrypted at rest
2. **Data in transit**: If served via API, enforce TLS
3. **Access control**: Restrict access to the pipeline and its outputs to authorized users only
4. **Audit logging**: Log who ran the pipeline, on what data, and when
5. **Data retention**: Define and enforce how long cleaned transaction data is kept
6. **Regulatory compliance**: Indian financial data may be subject to RBI guidelines and the Digital Personal Data Protection Act, 2023

---

## Error Handling Philosophy

The pipeline follows a **fail-fast** approach for structural errors and a **warn-and-default** approach for recoverable oddities.

| Scenario | Behavior | Rationale |
|---|---|---|
| Missing required column | `ValueError` raised immediately | Can't proceed without schema compliance |
| Unparseable date column | `ValueError` raised immediately | Temporal features are essential |
| Invalid `amount_flag` (not DR/CR) | **Warning logged**, defaulted to DR | Don't lose data; debits are the safe assumption |
| Zero-amount rows | **Warning logged**, rows removed | Pass-through entries carry no signal |
| Exact duplicate rows | **Warning logged**, duplicates removed | Banks sometimes duplicate entries |
| Below-threshold labeling coverage | **Warning logged** | Keywords may need expansion, but pipeline continues |
| `global_std = 0` | Falls back to `1.0` | Prevents division-by-zero; constant series is valid input |
| Non-datetime `date` column to feature_engineer | `TypeError` raised | Must run preprocess() first |

---

## Testing Architecture

### Structure

```
test_phase1.py (606 lines)
├── TestValidateSchema (4 tests)
├── TestNormalizeFlag (9 parameterized tests)
├── TestComputeSignedAmount (6 tests)
├── TestCleanRemark (14 tests)
├── TestDropZeroAmount (3 tests)
├── TestDeduplicate (3 tests)
├── TestPreprocess (8 tests)
├── TestTimeFeatures (5 tests)
├── TestRollingFeatures (4 tests)
├── TestFillRollingNulls (3 tests)
├── TestAmountFeatures (5 tests)
├── TestEngineerFeaturesFull (2 tests)
├── TestMatchRemark (10 tests)
├── TestLabelDebits (5 tests)
└── TestLabelCredits (5 tests)

test_phase2.py (94 lines)
├── test_categorization_training_and_prediction
├── test_spend_model_training_and_prediction
└── test_expected_spend_model_extrapolates    ← Proves Ridge > Trees

test_phase3.py (81 lines)
├── test_recurring_detector
├── test_anomaly_composite_threshold
├── test_missing_anomaly_columns_raises
└── test_insight_generator_creates_strings

test_e2e.py (143 lines)
└── run_e2e_test()                            ← Full 60-day simulation
```

### Key Testing Principles

1. **Input immutability tests**: Multiple test classes verify that functions don't mutate their input DataFrames
2. **Edge case coverage**: Empty strings, None values, non-string types, all-zero amounts, single-row DataFrames
3. **Leakage verification**: Tests explicitly assert that row 0's rolling mean is NaN (proving shift(1) is applied)
4. **Model behavior tests**: The extrapolation test proves the architecture choice (Ridge vs. trees) is correct
5. **Contract tests**: E2E test asserts that a known ₹2,500 spike is flagged as anomalous and a known monthly charge is flagged as recurring

### Running Tests

```bash
# Quick: all unit tests
pytest -v

# Detailed: with logging output
pytest -v --log-cli-level=INFO

# Single module
pytest test_phase1.py::TestCleanRemark -v

# E2E (requires all modules functional)
python test_e2e.py
```

---

## Extension Guide

### Adding a New Merchant

Edit `config.py → MERCHANT_ALIASES`:

```python
r"newmerchant":  "New Merchant",
```

Then add the merchant's keyword to the relevant category in `CATEGORY_KEYWORDS`:

```python
"food": [
    "swiggy", "zomato", ..., "newmerchant",
],
```

**No code changes required.** The pipeline will pick it up on the next run.

### Adding a New Spending Category

1. Add the category to `CATEGORY_PRIORITY` (position determines tie-breaking rank)
2. Add a keyword list to `CATEGORY_KEYWORDS`
3. The ML model will automatically learn it from the new pseudo-labels

### Adding a New Feature

1. Add the computation function in `feature_engineer.py`
2. Call it from `engineer_features()`
3. Add it to the relevant model's feature list (`categorization_model.py` or `expected_spend_model.py`)
4. Add tests verifying the feature is NaN-free and correctly computed
5. **Verify leakage safety**: Does row *i*'s feature use any information from rows *i* or later?

### Adding a New Recurring Frequency

In `recurring_detector.py`, add a new condition block:

```python
elif 13 <= mean_gap <= 15 and var < 3:
    assigned_freq = "biweekly"
```

### Adding a New Insight Type

In `insight_generator.py`, add a new conditional block that reads the relevant DataFrame columns and appends formatted strings to the `insights` list.

---

## Failure Modes & Recovery

| Failure | Symptom | Cause | Recovery |
|---|---|---|---|
| All transactions labeled "uncategorized" | 0% labeling coverage warning | Keywords don't match the bank's remark format | Inspect `cleaned_remarks` output — add matching keywords to `config.py` |
| Anomaly detector flags nothing | No anomalies in output | Thresholds too high, or data has no outliers | Lower `zscore_threshold` / `pct_dev_threshold`, or accept that the data is genuinely stable |
| Anomaly detector flags everything | 80%+ anomaly rate | Training data too small/homogeneous — poor expected spend model | Ensure ≥30 transactions for meaningful rolling statistics |
| Recurring detector misses subscriptions | `is_recurring` all False | Amount varies >5%, or time gaps are irregular | Increase `amount_tolerance`, or widen the day-range bounds |
| Model predicts same category for all rows | Low training accuracy | All pseudo-labels are the same category (e.g., all "food") | Diversify test data, or expand keyword coverage |
| `TypeError: 'date' column must be datetime64` | Crash at feature engineering | `preprocess()` was not called first | Always call `preprocess()` before `engineer_features()` |
| `ValueError: Missing required columns` | Crash at preprocessing | CSV is missing required columns | Check CSV headers; rename columns if needed |

---

*This document is a living specification. If you change the pipeline's behavior, update this document first. Documentation that lies is worse than no documentation at all.*
]]>
