# PIPELINE FLOW — End-to-End Execution Trace

> A mechanical, step-by-step trace of every function call, data transformation, and column mutation that occurs when `run_pipeline(df)` is invoked.  
> This document follows the exact execution order. Every DataFrame column that is created, modified, or consumed is explicitly tracked.

---

## Table of Contents

1. [Entry Point](#1-entry-point)
2. [Phase 1: Preprocessing](#2-phase-1-preprocessing)
3. [Phase 2: Feature Engineering](#3-phase-2-feature-engineering)
4. [Phase 3: Seed Labeling + ML Categorization](#4-phase-3-seed-labeling--ml-categorization)
5. [Phase 4: Expected Spend Regression](#5-phase-4-expected-spend-regression)
6. [Phase 5: Signal Detection](#6-phase-5-signal-detection)
7. [Phase 6: Insight Scoring + NLP Generation](#7-phase-6-insight-scoring--nlp-generation)
8. [Return Path](#8-return-path)
9. [Inference Mode Divergence](#9-inference-mode-divergence)
10. [Complete Column Lineage](#10-complete-column-lineage)

---

## 1. Entry Point

**File:** `pipeline.py`, function `run_pipeline(df: pd.DataFrame) -> PipelineResult`

**Input Contract:**
The raw DataFrame `df` must contain at minimum these 4 columns (validated in Phase 1):

| Column | Type | Example |
|--------|------|---------|
| `date` | str or datetime | `"2023-10-01"`, `"01/10/2023"` |
| `amount` | float | `500.0` |
| `amount_flag` | str | `"DR"`, `"cr"`, `" CR "` |
| `remarks` | str | `"UPI/9876543210/Swiggy Order"` |

Additional columns (e.g., `balance`, `payment_mode_l`) are preserved but not required.

**Execution begins.** The pipeline immediately enters Phase 1.

---

## 2. Phase 1: Preprocessing

**Call:** `preprocessor.preprocess(df)` → returns `(debits_df, credits_df)`

### Step-by-step execution inside `preprocess()`:

```
Step 1: validate_schema(df)
  └─ Checks: date, amount, amount_flag, remarks ∈ df.columns
  └─ On failure: raises ValueError("Missing required columns: {missing}")

Step 2: df = df.copy()
  └─ Defensive copy. Original input is never mutated.

Step 3: df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
  └─ Parses dates. Invalid dates become NaT.
  └─ df = df.dropna(subset=["date"])
  └─ Removes rows with unparseable dates.

Step 4: df = _compute_signed_amount(df)
  └─ For each row:
     └─ _normalize_flag(amount_flag) → "DR" or "CR" or None
     └─ None → defaults to "DR" (logged as warning)
     └─ amount_flag column is overwritten with normalized value
  └─ NEW COLUMN: signed_amount = -amount (if DR) or +amount (if CR)

Step 5: df["cleaned_remarks"] = df["remarks"].apply(clean_remark)
  └─ For each remark string:
     (a) Type check: non-string → ""
     (b) .lower()
     (c) Strip emails: regex \S+@\S+ → ""
     (d) Strip 4+ digit numbers: regex \d{4,} → ""
     (e) Strip non-alphanumeric: regex [^a-z0-9\s] → ""
     (f) Tokenize, remove NOISE_TOKENS, rejoin
     (g) Collapse whitespace, .strip()
  └─ NEW COLUMN: cleaned_remarks

Step 6: df["cleaned_remarks"] = df["cleaned_remarks"].apply(_normalize_merchant)
  └─ For each cleaned remark:
     └─ Iterates MERCHANT_ALIASES (regex → canonical name)
     └─ First match wins (re.sub replaces all matches of that pattern)
  └─ MODIFIES: cleaned_remarks (in-place overwrite)

Step 7: df = _drop_zero_amount(df)
  └─ Removes rows where amount == 0

Step 8: df = _deduplicate(df)
  └─ Drops exact duplicates on (date, amount, amount_flag, remarks)

Step 9: df = df.sort_values(by="date").reset_index(drop=True)
  └─ Chronological sort. Index reset.

Step 10: Split
  └─ debits_df  = df[df["amount_flag"] == "DR"].copy()
  └─ credits_df = df[df["amount_flag"] == "CR"].copy()
```

### DataFrame state after Phase 1 (debits_df):

| Column | Status | Type |
|--------|--------|------|
| `date` | parsed | datetime64 |
| `amount` | original (absolute) | float64 |
| `amount_flag` | normalized | str ("DR") |
| `remarks` | original | str |
| `signed_amount` | **NEW** | float64 (negative) |
| `cleaned_remarks` | **NEW** | str |
| *(any extra columns)* | preserved | varies |

**Pipeline continues with `debits_df` only for the remaining phases.** Credits are returned as-is.

---

## 3. Phase 2: Feature Engineering

**Call:** `feature_engineer.engineer_features(debits_df, global_mean=gm, global_std=gs)`

Where:
```python
gm = debits_df["signed_amount"].mean()
gs = debits_df["signed_amount"].std()
```

### Step-by-step execution inside `engineer_features()`:

```
Step 1: df = df.copy()
  └─ Defensive copy.

Step 2: df = add_time_features(df)
  └─ Requires: df["date"].dtype == datetime64 (raises TypeError otherwise)
  └─ NEW COLUMNS:
     is_weekend    = 1 if dayofweek ∈ {5, 6} else 0       (int)
     week_of_month = (day - 1) // 7 + 1                    (int, range [1,5])
     month_sin     = sin(2π × month / 12)                  (float, range [-1,1])
     month_cos     = cos(2π × month / 12)                  (float, range [-1,1])
     dow_sin       = sin(2π × dayofweek / 7)               (float, range [-1,1])
     dow_cos       = cos(2π × dayofweek / 7)               (float, range [-1,1])

Step 3: df = add_rolling_features(df)
  └─ df = df.sort_values(by="date").reset_index(drop=True)
  └─ shifted = df["amount"].shift(1)   ← LEAK PREVENTION: current row excluded
  └─ NEW COLUMNS:
     rolling_7d_mean  = shifted.rolling(7, min_periods=1).mean()
     rolling_30d_mean = shifted.rolling(30, min_periods=1).mean()
     rolling_7d_std   = shifted.rolling(7, min_periods=2).std()
  └─ Row 0: all rolling columns are NaN (no prior data after shift)

Step 4: df = fill_rolling_nulls(df, global_mean=gm, global_std=gs)
  └─ rolling_7d_mean  NaN → gm
  └─ rolling_30d_mean NaN → gm
  └─ rolling_7d_std   NaN → max(gs, 1.0)   ← Prevents division-by-zero downstream
  └─ After this step: NO NaN values remain in rolling columns.

Step 5: df = add_amount_features(df)
  └─ NEW COLUMNS:
     amount_log   = log1p(abs(amount))                     (float, ≥ 0)
     amount_zscore = clip((amount - rolling_7d_mean) / (rolling_7d_std + ε), -5, 5)
       where ε = 1e-9
```

### DataFrame state after Phase 2:

| New Column | Type | Range / Constraint |
|------------|------|--------------------|
| `is_weekend` | int | {0, 1} |
| `week_of_month` | int | [1, 5] |
| `month_sin` | float | [-1, 1] |
| `month_cos` | float | [-1, 1] |
| `dow_sin` | float | [-1, 1] |
| `dow_cos` | float | [-1, 1] |
| `rolling_7d_mean` | float | no NaN |
| `rolling_30d_mean` | float | no NaN |
| `rolling_7d_std` | float | > 0 (no NaN, no zero) |
| `amount_log` | float | ≥ 0 |
| `amount_zscore` | float | [-5, 5] |

**Total new columns: 11. Zero NaN values.**

---

## 4. Phase 3: Seed Labeling + ML Categorization

### Step 3a: Seed Labeling

**Call:** `seed_labeler.label_debits(debits_df)` → returns debits_df with `pseudo_label`

```
For each row in debits_df:
  └─ _match_remark(cleaned_remarks, CATEGORY_KEYWORDS, CATEGORY_PRIORITY, FALLBACK_DEBIT_LABEL)
     └─ Tokenizes cleaned_remarks
     └─ For each category in CATEGORY_PRIORITY order:
        └─ For each keyword in CATEGORY_KEYWORDS[category]:
           └─ Single-word keyword: check if token ∈ remark tokens
           └─ Multi-word keyword: check if ALL tokens of keyword ∈ remark tokens
        └─ If match found: record (category, keyword, match_position)
     └─ Winner = first matched category in priority order
     └─ No match → FALLBACK_DEBIT_LABEL ("uncategorized")

NEW COLUMN: pseudo_label (str)

Coverage check:
  coverage = (pseudo_label != FALLBACK_DEBIT_LABEL).mean()
  if coverage < MIN_COVERAGE_THRESHOLD (0.60):
      logger.warning("Low labeling coverage")
```

### Step 3b: Train Categorization Model

**Call:** `categorization_model.train_categorization_model(debits_df, label_col="pseudo_label")`

```
1. require_columns(df, [cleaned_remarks, amount_log])
2. Drop rows where cleaned_remarks, amount_log, or pseudo_label is NaN
3. EXCLUDE rows where pseudo_label ∈ {"uncategorized", "other_credit"}
   └─ Prevents the model from learning a "garbage" category
4. X = df[["cleaned_remarks", "amount_log"]]
5. y = df["pseudo_label"]
6. Pipeline:
   └─ ColumnTransformer:
      ├─ "text": TfidfVectorizer(ngram_range=(1,2), max_features=2000) on cleaned_remarks
      └─ "num":  StandardScaler(with_mean=False) on [amount_log]
   └─ LogisticRegression(class_weight="balanced", max_iter=1000)
7. pipeline.fit(X, y)
8. Logs training accuracy
```

Returns: fitted `sklearn.Pipeline` object.

### Step 3c: Predict Categories

**Call:** `categorization_model.predict_categories(cat_pipeline, debits_df)`

```
1. df = df.copy()
2. Fill NaN: cleaned_remarks → "", amount_log → 0.0
3. X = df[["cleaned_remarks", "amount_log"]]
4. preds = pipeline.predict(X)
5. probs = pipeline.predict_proba(X)
6. confidences = probs.max(axis=1)

NEW COLUMNS:
  predicted_category     (str)    — ML model's predicted category
  category_confidence    (float)  — probability of the predicted class [0, 1]
```

### DataFrame state after Phase 3:

| New Column | Type | Source |
|------------|------|--------|
| `pseudo_label` | str | Keyword matching (deterministic) |
| `predicted_category` | str | LogisticRegression (probabilistic) |
| `category_confidence` | float [0,1] | max(predict_proba) |

---

## 5. Phase 4: Expected Spend Regression

### Step 4a: Train Spend Model

**Call:** `expected_spend_model.train_expected_spend_model(debits_df, target_col="amount")`

```
1. require_columns(df, [9 features + predicted_category])
2. Drop rows with NaN in: amount, predicted_category, rolling_7d_mean
3. y = df["amount"] (absolute value)
4. Pipeline:
   └─ ColumnTransformer:
      ├─ "num": StandardScaler() on [is_weekend, month_sin, month_cos, dow_sin, dow_cos,
      │                              week_of_month, rolling_7d_mean, rolling_30d_mean, rolling_7d_std]
      └─ "cat": OneHotEncoder(handle_unknown="ignore") on [predicted_category]
   └─ RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0])
5. pipeline.fit(df, y)
6. Logs train R² score
```

Returns: fitted `sklearn.Pipeline` object.

### Step 4b: Predict Expected Spend

**Call:** `expected_spend_model.predict_expected_spend(spend_pipeline, debits_df)`

```
1. df = df.copy()
2. Defensive NaN fill for all 9 numeric features → 0.0
3. Defensive NaN fill for predicted_category → "uncategorized"
4. expected_amount = pipeline.predict(df)

NEW COLUMNS:
  expected_amount     (float)  — what the model thinks this txn should cost
  residual            (float)  — amount - expected_amount
  percent_deviation   (float)  — residual / (expected_amount + 1e-5)
```

### DataFrame state after Phase 4:

| New Column | Type | Description |
|------------|------|-------------|
| `expected_amount` | float | RidgeCV prediction |
| `residual` | float | actual - predicted |
| `percent_deviation` | float | residual / (predicted + ε) |

---

## 6. Phase 5: Signal Detection

### Step 5a: Anomaly Detection

**Call:** `anomaly_detector.detect_anomalies(debits_df, zscore_threshold=3.0, pct_dev_threshold=0.5)`

```
1. df = df.copy()
2. require_columns(df, [amount_zscore, percent_deviation])
3. is_spike = (abs(amount_zscore) > 3.0) AND (abs(percent_deviation) > 0.5)

NEW COLUMN:
  is_anomaly   (bool)  — True only if BOTH gates pass
```

### Step 5b: Recurring Detection

**Call:** `recurring_detector.find_recurring_transactions(debits_df, group_col="cleaned_remarks")`

```
1. df = df.copy()
2. require_columns(df, [date, amount, cleaned_remarks])
3. Sort by date
4. Initialize: is_recurring=False, recurring_frequency=None, recurring_confidence=0.0

5. For each (cleaned_remarks) group:
   └─ Skip if group.size < 3
   └─ time_diffs = date.diff().dt.days (between consecutive rows)
   └─ amount_drift = (max - min) / mean
   └─ Skip if amount_drift > amount_tolerance (default 0.05)
   
   └─ mean_gap = time_diffs.mean()
   └─ variance = time_diffs.var()
   
   └─ Frequency classification:
      ├─ 27 ≤ mean_gap ≤ 33 AND var < 10  → "monthly"
      ├─ 13 ≤ mean_gap ≤ 16 AND var < 5   → "biweekly"
      ├─ 6  ≤ mean_gap ≤ 8  AND var < 3   → "weekly"
      ├─ 85 ≤ mean_gap ≤ 95 AND var < 20  → "quarterly"
      └─ else → skip (not recurring)
   
   └─ confidence = 1.0
   └─ if amount_drift > RECURRING_FLUCTUATION_PENALTY_THRESHOLD (0.10):
          confidence = 0.5
   
   └─ Mark all rows in group: is_recurring=True, frequency, confidence

NEW COLUMNS:
  is_recurring          (bool)
  recurring_frequency   (str or None)  — "weekly" | "biweekly" | "monthly" | "quarterly"
  recurring_confidence  (float)        — 1.0 or 0.5
```

### DataFrame state after Phase 5:

| New Column | Type | Description |
|------------|------|-------------|
| `is_anomaly` | bool | Dual-gate composite flag |
| `is_recurring` | bool | Heuristic subscription flag |
| `recurring_frequency` | str/None | Detected period |
| `recurring_confidence` | float | 1.0 or 0.5 |

---

## 7. Phase 6: Insight Scoring + NLP Generation

### Step 6a: Load Insight Ranker

**Call:** `insight_model.load_insight_ranker("models/insight_ranker.pkl")`

```
If file exists:
  └─ pickle.load → sklearn.Pipeline (LightGBM inside)

If file does NOT exist:
  └─ Returns None
  └─ logger.warning("Run train_and_save_models.py to generate it")
```

### Step 6b: Score Transactions

**Call:** `insight_model.predict_insight_scores(ranker_pipeline, debits_df)`

```
If pipeline is None:
  └─ insight_score = 0.0 for all rows (graceful fallback)

If pipeline exists:
  └─ X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES] (14 columns)
  └─ Fill NaN: numerics → 0.0, categoricals → "unknown"
  └─ probs = pipeline.predict_proba(X)
  └─ Find index of "no_action" class
  └─ insight_score = 1.0 - probs[:, no_action_index]

NEW COLUMN:
  insight_score   (float [0, 1])  — higher = more actionable
```

### Step 6c: Generate Human-Readable Insights

**Call:** `insight_generator.generate_human_insights(debits_df, top_n=10)`

```
1. require_columns(df, [date, cleaned_remarks, predicted_category, amount,
                        percent_deviation, is_recurring, is_anomaly])
2. If insight_score column missing → fill with 0.0
3. Sort by date

4. SUBSCRIPTION SCAN:
   └─ Filter rows where is_recurring == True
   └─ Group by cleaned_remarks
   └─ For each group:
      ├─ freq = last row's recurring_frequency
      ├─ amt = group amount mean
      ├─ score = max(insight_score) in group
      ├─ Select template from INSIGHT_TEMPLATES["subscription"]
      ├─ Format: template.format(merchant=name, amount=amt, frequency=freq)
      ├─ Select tip via _select_tip("", "subscription")
      └─ Append (score, "subscription", insight_text, tip) to candidates

5. ANOMALY SCAN:
   └─ Filter rows where is_anomaly == True
   └─ For each row:
      ├─ Extract: name, category, amount, percent_deviation, date, score
      ├─ Select template from INSIGHT_TEMPLATES["spending_spike"]
      ├─ Format: template.format(category, merchant, amount, pct, date)
      ├─ Select tip via _select_tip(category, "spending_spike")
      └─ Append (score, "spike", insight_text, tip) to candidates

6. DIVERSITY RANKING:
   └─ Sort all candidates by score (descending)
   └─ Pass 1: Pick the highest-scoring insight of EACH type first
              (guarantees at least 1 subscription + 1 spike if they exist)
   └─ Pass 2: Fill remaining top_n slots with highest-scoring candidates
   └─ Final sort: by score (descending)

7. Format output:
   └─ For each (score, type, text, tip) in top_candidates:
      ├─ insights.append(text)
      └─ if tip: insights.append(f"  > Tip: {tip}")

RETURNS: List[str]  — alternating insight text and tip strings
```

---

## 8. Return Path

```python
return PipelineResult(
    debits=debits_df,      # DataFrame with all 25+ computed columns
    credits=credits_df,    # DataFrame from Phase 1 (minimal processing)
    insights=insights      # List[str] from Phase 6
)
```

The caller accesses:
- `result.debits` — fully enriched debit DataFrame
- `result.credits` — cleaned credit DataFrame
- `result.insights` — ranked, human-readable insight strings

---

## 9. Inference Mode Divergence

**Function:** `run_inference(df, cat_pipeline, spend_pipeline, insight_pipeline)` in `pipeline.py`

The inference path differs from `run_pipeline` in these specific ways:

| Aspect | `run_pipeline` (Training) | `run_inference` (Inference) |
|--------|--------------------------|----------------------------|
| Feature engineering | `engineer_features(df, gm, gs)` | `engineer_features_inference(df, gm, gs)` |
| Categorization model | Trained inline | Pre-trained pipeline passed as argument |
| Expected spend model | Trained inline | Pre-trained pipeline passed as argument |
| Insight ranker | Loaded from disk | Pre-trained pipeline passed as argument |
| Global mean/std source | Computed from current batch debits | **Should be** from training set (passed as params or stored) |

**Key architectural concern:** In `run_inference`, `global_mean` and `global_std` are still computed from the incoming batch. For true production use, these should be stored from training and passed in, but the current implementation re-computes them from the inference batch. This is documented as acceptable for single-batch inference but would introduce leakage in streaming scenarios.

---

## 10. Complete Column Lineage

### Input → Output Column Map

```
RAW INPUT (4 mandatory)
  │
  ├─ date              → parsed to datetime64 [Phase 1]
  ├─ amount            → preserved as-is (absolute value)
  ├─ amount_flag       → normalized to "DR"/"CR"
  └─ remarks           → preserved as-is
  
PHASE 1 ADDITIONS
  ├─ signed_amount     ← -amount (DR) or +amount (CR)
  └─ cleaned_remarks   ← clean_remark(remarks) → _normalize_merchant()
  
PHASE 2 ADDITIONS
  ├─ is_weekend        ← datetime.dayofweek ∈ {5,6}
  ├─ week_of_month     ← (day-1)//7 + 1
  ├─ month_sin         ← sin(2π·month/12)
  ├─ month_cos         ← cos(2π·month/12)
  ├─ dow_sin           ← sin(2π·dayofweek/7)
  ├─ dow_cos           ← cos(2π·dayofweek/7)
  ├─ rolling_7d_mean   ← shifted rolling mean (leak-free)
  ├─ rolling_30d_mean  ← shifted rolling mean (leak-free)
  ├─ rolling_7d_std    ← shifted rolling std  (leak-free)
  ├─ amount_log        ← log1p(abs(amount))
  └─ amount_zscore     ← clipped z-score [-5, 5]
  
PHASE 3 ADDITIONS
  ├─ pseudo_label         ← keyword match or "uncategorized"
  ├─ predicted_category   ← LogisticRegression prediction
  └─ category_confidence  ← max(predict_proba)
  
PHASE 4 ADDITIONS
  ├─ expected_amount      ← RidgeCV prediction
  ├─ residual             ← amount - expected_amount
  └─ percent_deviation    ← residual / (expected_amount + ε)
  
PHASE 5 ADDITIONS
  ├─ is_anomaly           ← (|zscore| > 3) AND (|pct_dev| > 0.5)
  ├─ is_recurring         ← heuristic frequency detection
  ├─ recurring_frequency  ← "weekly"/"biweekly"/"monthly"/"quarterly"/None
  └─ recurring_confidence ← 1.0 or 0.5
  
PHASE 6 ADDITIONS
  └─ insight_score        ← 1.0 - P(no_action) or 0.0 fallback
```

### Total: 4 input columns → 25+ output columns on the debit DataFrame.

---

## Data Flow Diagram

```
                    ┌─────────────┐
                    │  Raw CSV/DF │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │  PHASE 1    │
                    │ preprocessor│
                    └──┬──────┬───┘
                       │      │
                  debits_df  credits_df → (returned as-is)
                       │
                ┌──────▼──────┐
                │  PHASE 2    │
                │ feature_eng │
                └──────┬──────┘
                       │
                ┌──────▼──────┐     ┌──────────────┐
                │  PHASE 3    │────►│ Train Cat     │
                │ seed_label  │◄────│ Model (LR)   │
                └──────┬──────┘     └──────────────┘
                       │
                ┌──────▼──────┐     ┌──────────────┐
                │  PHASE 4    │────►│ Train Spend   │
                │ exp. spend  │◄────│ Model (Ridge) │
                └──────┬──────┘     └──────────────┘
                       │
                ┌──────▼──────┐
                │  PHASE 5    │
                │ anomaly +   │
                │ recurring   │
                └──────┬──────┘
                       │
                ┌──────▼──────┐     ┌──────────────┐
                │  PHASE 6    │◄────│ LightGBM     │
                │ insight gen │     │ Ranker (pkl)  │
                └──────┬──────┘     └──────────────┘
                       │
                ┌──────▼──────┐
                │ PipelineResult │
                │ .debits        │
                │ .credits       │
                │ .insights      │
                └────────────────┘
```
