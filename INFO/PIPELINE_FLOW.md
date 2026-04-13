# Insight Engine — End-to-End Pipeline Execution Flow

> **Scope**: Traces every instruction from raw CSV input to final insight output.
> **Audience**: A developer who needs to understand what happens at every step.

---

## Visual Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         run_pipeline(raw_df)                                    │
│   pipeline.py:93                                                                │
│                                                                                 │
│   ┌─────────────┐   ┌──────────────┐   ┌───────────────────┐                    │
│   │  PHASE 1    │   │   PHASE 2    │   │     PHASE 3       │                    │
│   │ Preprocess  │──▶│ Seed Labeler │──▶│ Feature Engineer  │                    │
│   │             │   │              │   │                   │                    │
│   │ • validate  │   │ • keyword    │   │ • time features   │                    │
│   │ • coerce    │   │   matching   │   │ • rolling stats   │                    │
│   │ • dates     │   │ • priority   │   │ • z-scores        │                    │
│   │ • sign      │   │   tiebreak   │   │ • leakage-free    │                    │
│   │ • clean     │   │ • pseudo     │   │                   │                    │
│   │ • split     │   │   labels     │   │                   │                    │
│   └──────┬──────┘   └──────────────┘   └─────────┬─────────┘                    │
│          │                                        │                              │
│          │ debits, credits                         │ enriched debits              │
│          ▼                                        ▼                              │
│   ┌──────────────────────────────────────────────────────┐                       │
│   │                     PHASE 4                          │                       │
│   │               ML Model Training                     │                       │
│   │                                                      │                       │
│   │  ┌─────────────────────┐  ┌────────────────────────┐ │                       │
│   │  │ Categorization      │  │ Expected Spend         │ │                       │
│   │  │ TF-IDF + LogReg     │  │ RidgeCV Regression     │ │                       │
│   │  │                     │  │                        │ │                       │
│   │  │ → predicted_category│  │ → expected_amount      │ │                       │
│   │  │ → category_conf     │  │ → residual             │ │                       │
│   │  │                     │  │ → percent_deviation    │ │                       │
│   │  └─────────────────────┘  └────────────────────────┘ │                       │
│   └──────────────────────────┬───────────────────────────┘                       │
│                              │                                                   │
│                              ▼                                                   │
│   ┌──────────────────────────────────────────────────────┐                       │
│   │                     PHASE 5                          │                       │
│   │            Signal Detection + Scoring                │                       │
│   │                                                      │                       │
│   │  ┌────────────────┐  ┌────────────────┐              │                       │
│   │  │ Anomaly Det.   │  │ Recurring Det. │              │                       │
│   │  │ |z| > 3.0 AND  │  │ 3+ matches,    │              │                       │
│   │  │ |dev| > 0.5    │  │ time + amount  │              │                       │
│   │  │                │  │ consistency    │              │                       │
│   │  │ → is_anomaly   │  │ → is_recurring │              │                       │
│   │  └────────────────┘  └────────────────┘              │                       │
│   │                                                      │                       │
│   │  ┌────────────────────────────────────────┐          │                       │
│   │  │ Insight Ranker (LightGBM)              │          │                       │
│   │  │ score = 1.0 − P(no_action)             │          │                       │
│   │  │ → insight_score                        │          │                       │
│   │  └────────────────────────────────────────┘          │                       │
│   └──────────────────────────┬───────────────────────────┘                       │
│                              │                                                   │
│                              ▼                                                   │
│   ┌──────────────────────────────────────────────────────┐                       │
│   │                     PHASE 6                          │                       │
│   │            Insight Generation + Output               │                       │
│   │                                                      │                       │
│   │  • Finalize (fill NaN scores)                        │                       │
│   │  • Optimize memory (categoricals, bools)             │                       │
│   │  • Generate human insight strings                    │                       │
│   │  • Diversity ranking (type-first, then score)        │                       │
│   │  • Return frozen PipelineResult                      │                       │
│   └──────────────────────────────────────────────────────┘                       │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Detailed Phase-by-Phase Execution

### Phase 1: Preprocessing (`preprocessor.py`)

**Entry**: `pipeline.py:116` — `debits, credits = preprocess(raw_df)`

```
Input:  pd.DataFrame with columns [date, amount, amount_flag, remarks, balance]
Output: (debit_df, credit_df) — each cleaned, sorted, and deduplicated
```

#### Step 1.1: Schema Validation
```python
# preprocessor.py:274
validate_schema(df)
# → Calls require_columns(df, Col.raw_input())
# → Col.raw_input() = frozenset({"date", "amount", "amount_flag", "remarks", "balance"})
# → Raises ValueError if ANY column is missing
```

#### Step 1.2: Type Coercion
```python
# preprocessor.py:277
df = coerce_and_validate_types(df)
# → pd.to_numeric(df["amount"], errors="coerce")
# → Drops rows where amount couldn't be parsed
# → Logs how many rows were dropped
```

#### Step 1.3: Date Parsing & Sorting
```python
# preprocessor.py:280
df = _parse_and_sort_dates(df)
# → pd.to_datetime(df["date"], format=ISO8601)
# → Sorts ascending by date
# → Resets index
```

#### Step 1.4: Flag Normalization + Signed Amount
```python
# preprocessor.py:283
df = _compute_signed_amount(df)
# Step A: Normalize each flag via _normalize_flag()
#   "dr" → "DR", "  CR  " → "CR", "XX" → None → "DR" (default)
# Step B: Compute signed_amount
#   DR rows: signed_amount = -abs(amount)
#   CR rows: signed_amount = +abs(amount)
```

#### Step 1.5: Zero-Amount Removal
```python
# preprocessor.py:286
df = _drop_zero_amount(df)
# → Drops rows where amount == 0
```

#### Step 1.6: Deduplication
```python
# preprocessor.py:289
df = _deduplicate(df)
# → drop_duplicates on (date, amount, remarks, amount_flag), keep="first"
```

#### Step 1.7: Remark Cleaning
```python
# preprocessor.py:292
df["cleaned_remarks"] = df["remarks"].apply(clean_remark)
```

**`clean_remark()` internals** (the most complex single function):

```
Input:  "NETFLIX.COM SUBSCRIPTION 9876543210 user@email.com"

Step A: Lowercase
        → "netflix.com subscription 9876543210 user@email.com"

Step B: Merchant Alias Check
        Iterate _COMPILED_ALIASES (pre-compiled from MERCHANT_ALIASES):
          Pattern r"netflix" matches!
          Netflix is a SPECIFIC alias (not generic).
          → Return "netflix" immediately. DONE.

Input:  "UPI/9876543210/RajuTeaStall"

Step A: Lowercase
        → "upi/9876543210/rajuteastall"

Step B: Merchant Alias Check
        Pattern r"upi" matches, but UPI is a GENERIC router.
        → Do NOT return. Strip the "upi" prefix. Continue.

Step C: Remove emails (regex)
        → "upi/9876543210/rajuteastall" (no email here)

Step D: Remove 4+ digit sequences
        → "upi//rajuteastall"

Step E: Remove special characters (/, etc.)
        → "upi rajuteastall"

Step F: Split, filter noise tokens ("upi" is noise)
        → ["rajuteastall"]

Step G: Join and return
        → "rajuteastall"
```

#### Step 1.8: Split by Debit/Credit
```python
# preprocessor.py:295–296
debits = df[df["amount_flag"] == "DR"].reset_index(drop=True)
credits = df[df["amount_flag"] == "CR"].reset_index(drop=True)
return (debits, credits)
```

**Phase 1 Output Schema** (new columns added):
- `signed_amount` — negative for debits, positive for credits
- `cleaned_remarks` — merchant name or cleaned text

---

### Phase 2: Seed Labeling (`seed_labeler.py`)

**Entry**: `pipeline.py:120–121`
```python
debits = label_debits(debits)
credits = label_credits(credits)
```

#### Step 2.1: Keyword Compilation (happens once at module import)
```python
# seed_labeler.py:110–111
_DEFAULT_DEBIT_KWS = _compile_keywords(CATEGORY_KEYWORDS, is_credit=False)
# Each keyword becomes a CompiledKeyword:
#   text="zomato", norm="zomato", pattern=re.compile(r"\bzomato\b"),
#   category="food", tier_name="MEDIUM", priority=2, confidence=0.70
```

#### Step 2.2: Per-Row Matching
```python
# For each row in debits:
#   1. Re-normalize the cleaned_remark (boundary hardening)
#   2. Run ALL compiled keyword patterns against it
#   3. Collect matches
#   4. Find highest-priority tier among matches
#   5. Among same-tier: pick longest keyword (then alphabetical)
#   6. Return: (category, reason, keyword_text, keyword_norm, confidence)
```

**Example trace**:
```
cleaned_remark = "swiggy order food"
normalized     = "swiggy order food"

Matches found:
  ✓ "swiggy"  → food,    MEDIUM, priority=2, len=6
  ✓ "food"    → food,    MEDIUM, priority=2, len=4
  ✓ "order"   → (no match)

Highest priority tier: MEDIUM (priority=2)
Same-tier candidates: swiggy (len=6), food (len=4)
Tiebreak: longest → "swiggy"

Result: pseudo_label="food", label_keyword="swiggy", confidence=0.70
```

#### Step 2.3: Coverage Logging
```python
# seed_labeler.py:119–137 (_log_coverage)
# Counts % of rows with non-fallback label
# Warns if < 40%
```

**Phase 2 Output Schema** (new columns added):
- `pseudo_label` — "food", "transport", ..., "uncategorized"
- `label_reason` — "keyword_match" or "fallback"
- `label_keyword` — the keyword that matched
- `label_keyword_norm` — normalized form of keyword
- `label_confidence` — 0.70 (MEDIUM), 0.90 (HIGH), 0.50 (LOW)

---

### Phase 3: Feature Engineering (`feature_engineer.py`)

**Entry**: `pipeline.py:125–132`
```python
global_mean = debits["signed_amount"].mean()
global_std  = debits["signed_amount"].std()
debits = engineer_features(debits, global_mean=global_mean, global_std=global_std)
```

#### Step 3.1: Time Features
```python
# feature_engineer.py:46–69
# Computed from the date column:
is_weekend    = 1 if dayofweek ∈ {5,6} else 0
week_of_month = ((day - 1) // 7) + 1              # 1–5
month_sin     = sin(2π × month / 12)               # cyclical
month_cos     = cos(2π × month / 12)
dow_sin       = sin(2π × dayofweek / 7)
dow_cos       = cos(2π × dayofweek / 7)
```

#### Step 3.2: Rolling Features (Leak-Free)
```python
# feature_engineer.py:74–110
# CRITICAL: shift(1) BEFORE rolling
shifted = df["amount"].shift(1)   # row i sees only rows 0…i-1
rolling_7d_mean  = shifted.rolling(window=7,  min_periods=1).mean()
rolling_30d_mean = shifted.rolling(window=30, min_periods=1).mean()
rolling_7d_std   = shifted.rolling(window=7,  min_periods=2).std()
# Row 0: all NaN (no prior data)
# Row 1: rolling_7d_mean = row_0_amount (single prior value)
```

#### Step 3.3: NaN Filling
```python
# feature_engineer.py:113–144
# Rolling means → filled with global_mean (training-set statistic)
# Rolling std   → filled with max(global_std, 1.0)  # 1.0 floor prevents div-by-zero
```

#### Step 3.4: Amount Features
```python
# feature_engineer.py:149–184
amount_log   = log1p(|amount|)                     # handles negatives
amount_zscore = clip((amount − rolling_7d_mean) / max(rolling_7d_std, 1.0), -5, +5)
```

**Phase 3 Output Schema** (new columns added):
- `is_weekend`, `week_of_month`, `month_sin`, `month_cos`, `dow_sin`, `dow_cos`
- `rolling_7d_mean`, `rolling_30d_mean`, `rolling_7d_std`
- `amount_log`, `amount_zscore`

---

### Phase 4: ML Model Training & Prediction

**Entry**: `pipeline.py:135–155`

#### Step 4.1: Categorization Model
```python
# pipeline.py:136 → categorization_model.py:57
# Training:
#   1. Drop rows with pseudo_label == FALLBACK_DEBIT_LABEL (don't train on unknowns)
#   2. Build Pipeline:
#        ColumnTransformer:
#          "text" → TfidfVectorizer(cleaned_remarks, ngram_range=(1,2), max_features=2000)
#          "num"  → StandardScaler(amount_log, with_mean=False)
#        LogisticRegression(class_weight="balanced", max_iter=1000)
#   3. Fit on (cleaned_remarks + amount_log) → pseudo_label
#
# Prediction:
#   predicted_category  = pipeline.predict(X)
#   category_confidence = max(pipeline.predict_proba(X), axis=1)
```

#### Step 4.2: Expected Spend Model
```python
# pipeline.py:147 → expected_spend_model.py:59
# Training:
#   1. Build Pipeline:
#        ColumnTransformer:
#          "num" → StandardScaler(is_weekend, month_sin, month_cos, dow_sin, dow_cos,
#                                 week_of_month, rolling_7d_mean, rolling_30d_mean, rolling_7d_std)
#          "cat" → OneHotEncoder(predicted_category)
#        RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0])
#   2. Fit on features → amount
#
# Prediction:
#   expected_amount  = pipeline.predict(X)
#   residual         = actual_amount − expected_amount
#   percent_deviation = residual / max(|expected_amount|, 1.0)
```

**Phase 4 Output Schema** (new columns added):
- `predicted_category`, `category_confidence`
- `expected_amount`, `residual`, `percent_deviation`

---

### Phase 5: Signal Detection + ML Scoring

**Entry**: `pipeline.py:158–167`

#### Step 5.1: Anomaly Detection
```python
# anomaly_detector.py:16–48
# Dual-gate heuristic (BOTH must be true):
is_anomaly = (|amount_zscore| > 3.0) AND (|percent_deviation| > 0.5)
```

**Logic flow for a single transaction**:
```
amount = ₹2500, mean = ₹150, std = ₹40
z-score = (2500 − 150) / 40 = 58.75 → clipped to 5.0
|z-score| = 5.0 > 3.0 ✓ (Gate 1 passes)

expected_amount = ₹160 (from RidgeCV)
residual = 2500 − 160 = 2340
percent_deviation = 2340 / 160 = 14.625
|deviation| = 14.625 > 0.5 ✓ (Gate 2 passes)

→ is_anomaly = True
```

#### Step 5.2: Recurring Detection
```python
# recurring_detector.py:21–117
# For each merchant group with ≥3 transactions:
#   1. Compute time gaps between consecutive transactions
#   2. Amount Score (A): 1 − (max_deviation / tolerance)
#   3. Temporal Score (T): Match against frequency templates (monthly, weekly, biweekly, quarterly)
#   4. Volume Score (V): n_occurrences / 12
#   5. Final = 0.4A + 0.4T + 0.2V
#   6. Reject if A == 0 or T == 0
```

**Logic flow for Netflix (3 transactions, ₹15.99 each, ~30-day gaps)**:
```
Group: "netflix" — 3 transactions

Time gaps: [30 days, 31 days]
Mean gap: 30.5 days
Monthly template: [27, 33] → 30.5 is within range ✓

Amount drift: max(|15.99 − 15.99|) / 15.99 = 0.0
A = 1.0 − (0.0 / 0.20) = 1.0

Temporal variance: low, within 10% of template
T = 1.0 (very consistent)

Volume: 3/12 = 0.25
V = 0.25

Final = 0.4(1.0) + 0.4(1.0) + 0.2(0.25) = 0.85

assigned_freq = "monthly"
is_recurring = True ✓
recurring_confidence ≈ 0.85
```

#### Step 5.3: ML Insight Scoring
```python
# insight_model.py:157–207
# If LightGBM ranker exists:
#   proba = pipeline.predict_proba(X)   # shape: (n, 5)
#   no_action_idx = classes.index("no_action")
#   insight_score = 1.0 − proba[:, no_action_idx]
# If ranker is None:
#   insight_score = 0.0 for all rows
```

**Phase 5 Output Schema** (new columns added):
- `is_anomaly` — bool
- `is_recurring`, `recurring_frequency`, `recurring_confidence`, `recurring_score`
- `insight_score` — float [0.0, 1.0]

---

### Phase 6: Insight Generation + Output

**Entry**: `pipeline.py:170–179`

#### Step 6.1: Finalization
```python
# pipeline.py:57–62 (finalize_df)
# Fill NaN in recurring_score with 0.0
```

#### Step 6.2: Memory Optimization
```python
# pipeline.py:64–71 (_optimize_memory_footprint)
# predicted_category → pd.Categorical (reduces memory ~8x)
# is_weekend → bool dtype
```

#### Step 6.3: Human Insight Generation
```python
# insight_generator.py:57–181
```

**Logic flow**:

```
Phase A — Collect Subscription Insights:
  1. Filter rows where is_recurring == True
  2. Group by cleaned_remarks (merchant)
  3. For each group:
     a. Pick a template from INSIGHT_TEMPLATES["subscription"]
        Example: "💰 Recurring: {merchant} charges ₹{amount:.0f} {freq}."
     b. Format with merchant name, amount, frequency
     c. Pick a matching tip via _select_tip(category, "subscription")
     d. Score = mean(insight_score) of the group
     e. Append to candidates as ("subscription", formatted_string, score)

Phase B — Collect Anomaly Insights:
  1. Filter rows where is_anomaly == True
  2. For each row:
     a. Pick a template from INSIGHT_TEMPLATES["spending_spike"]
        Example: "🔴 Unusual spend on {category}: ₹{amount:.0f} at {merchant} ({deviation:.0%} above expected) on {date}."
     b. Format with category, amount, merchant, deviation, date
     c. Pick a matching tip via _select_tip(category, "spending_spike")
     d. Score = insight_score of this row
     e. Append to candidates as ("spending_spike", formatted_string, score)

Phase C — Diversity Ranking:
  1. Sort all candidates by score descending
  2. Pass 1: Take the TOP candidate of EACH type
     e.g., Best subscription + Best spending_spike = 2 guaranteed diverse insights
  3. Pass 2: Fill remaining up to top_n from remaining candidates by score
  4. Sort final selection by score descending

Phase D — Format Output:
  For each selected insight:
    "[insight_string]\n  💡 Tip: [tip_text]"

Return: List[str]
```

**Example output**:
```
🔴 Unusual spend on shopping: ₹2500 at amazon (1462.5% above expected) on 2023-02-15.
  💡 Tip: Use price tracking tools like Keepa or CamelCamelCamel for big purchases.

💰 Recurring: netflix charges ₹16 monthly.
  💡 Tip: Audit your subscriptions quarterly — cancel unused ones via apps like Trim.
```

---

## Inference Path (`run_inference`)

**Entry**: `pipeline.py:184`
```python
def run_inference(raw_df, model_state: InsightModelState, history_df=None)
```

**Differences from `run_pipeline`**:

| Step | `run_pipeline` | `run_inference` |
|------|---------------|-----------------|
| Preprocess | ✓ Full | ✓ Full |
| Seed Label | ✓ Full | ✓ Full |
| Feature Engineer | `engineer_features()` | `engineer_features_inference()` with history stitching |
| Training | Trains new models | Skips — uses `InsightModelState` |
| Categorization | Train + predict | Predict only (`model_state.cat_pipeline`) |
| Expected Spend | Train + predict | Predict only (`model_state.spend_pipeline`) |
| Anomaly/Recurring | ✓ Same | ✓ Same |
| Insight Scoring | ✓ Same | ✓ Same (`model_state.ranker_pipeline`) |
| Insight Generation | ✓ Same | ✓ Same |

**History Stitching Detail** (`engineer_features_inference`):
```python
# feature_engineer.py:246–303
# 1. Tag new transactions with __is_new_txn__ = True
# 2. Tag history with __is_new_txn__ = False
# 3. Concatenate [history, new_transactions]
# 4. Sort by date
# 5. Run full feature engineering on combined set
# 6. Filter to only __is_new_txn__ == True rows
# 7. Drop the tag column
# Result: New transactions have rolling features computed from real history
```

---

## Data Flow Summary Table

| Phase | Module | Input Columns | Output Columns Added |
|-------|--------|---------------|---------------------|
| 1 | preprocessor | date, amount, amount_flag, remarks, balance | signed_amount, cleaned_remarks |
| 2 | seed_labeler | cleaned_remarks | pseudo_label, label_reason, label_keyword, label_keyword_norm, label_confidence |
| 3 | feature_engineer | date, amount, signed_amount | is_weekend, week_of_month, month_sin, month_cos, dow_sin, dow_cos, rolling_7d_mean, rolling_30d_mean, rolling_7d_std, amount_log, amount_zscore |
| 4a | categorization_model | cleaned_remarks, amount_log, pseudo_label | predicted_category, category_confidence |
| 4b | expected_spend_model | time_features, rolling_features, predicted_category, amount | expected_amount, residual, percent_deviation |
| 5a | anomaly_detector | amount_zscore, percent_deviation | is_anomaly |
| 5b | recurring_detector | date, amount, cleaned_remarks | is_recurring, recurring_frequency, recurring_confidence, recurring_score |
| 5c | insight_model | 14 numeric+categorical features | insight_score |
| 6 | insight_generator | is_anomaly, is_recurring, predicted_category, cleaned_remarks, amount, percent_deviation, date, insight_score | (returns List[str], no columns added) |
