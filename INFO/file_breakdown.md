# Insight Engine — File-by-File Technical Breakdown

> **Scope**: Every production source file in the repository.  
**Audience**: A developer who is smart but has never seen this codebase.



---

## Table of Contents

| # | File | Role |
|---|------|------|
| 1 | `config.py` | Declarative configuration hub |
| 2 | `schema.py` | DataFrame column contract registry |
| 3 | `logger_factory.py` | Structured JSON logging + run ID |
| 4 | `preprocessor.py` | Data cleaning & normalization |
| 5 | `feature_engineer.py` | Leak-free feature engineering |
| 6 | `seed_labeler.py` | Keyword-based pseudo-label generation |
| 7 | `categorization_model.py` | ML transaction categorization |
| 8 | `expected_spend_model.py` | Expected spend regression |
| 9 | `anomaly_detector.py` | Composite statistical anomaly flagging |
| 10 | `recurring_detector.py` | Rule-based recurring transaction detection |
| 11 | `insight_model.py` | ML insight ranking + model security |
| 12 | `insight_generator.py` | Natural language insight translator |
| 13 | `pipeline.py` | Central orchestrator |
| 14 | `model_state.py` | Model persistence container |
| 15 | `training_data_generator.py` | Synthetic labeled dataset generator |
| 16 | `model_benchmark.py` | Multi-model benchmark harness |
| 17 | `train_and_save_models.py` | Model serialization script |
| 18 | `demo.py` | Minimal usage example |
| 19 | `tutorial_real_data.py` | Real bank statement ingestion tutorial |
| 20 | `requirements.txt` | Runtime dependencies |
| 21 | `requirements-dev.txt` | Dev/test dependencies |

---

## 1. `config.py` (639 lines)

### Purpose
Single declarative source of truth for all business rules, keyword dictionaries, merchant aliases, priority mappings, insight templates, tip corpus, and pipeline constants. Zero executable logic beyond one lookup function.

### Structure

#### Merchant Normalisation Dictionaries
- **`GENERIC_ROUTER_ALIASES`** (`dict[str, str]`, 14 entries): Maps regex patterns for generic payment routers (UPI, NEFT, RTGS, IMPS, Paytm, PhonePe, etc.) to canonical names. These are stripped from remarks during cleaning — they carry no merchant identity.
- **`SPECIFIC_MERCHANT_ALIASES`** (`dict[str, str]`, ~170 entries): Maps regex patterns for known Indian merchants (Swiggy, Zomato, Amazon, IRCTC, Netflix, etc.) to canonical display names. When matched, the entire remark is replaced with the canonical name. Organized into 18 sub-domains: Food/QSR, Grocery, E-commerce, Electronics, Fashion, Transport, Hotels, Fuel, Streaming, Telecom, Utilities/DTH, Healthcare, Finance/Insurance, Education, Gaming, Real Estate.

#### Category Priority System
- **`HIGH_PRIORITY`**: `["finance", "health", "utilities"]` — highest intent signal.
- **`MEDIUM_PRIORITY`**: `["food", "transport", "shopping", "entertainment"]`.
- **`LOW_PRIORITY`**: `["atm", "transfer"]` — least-actionable.
- **`CATEGORY_PRIORITY`**: Concatenation of all three. Used as the ordered list for seed labeling tiebreaking.
- **`TIER_MAPPING`** (`dict[str, dict]`): Auto-generated mapping from category → `{tier_name, priority (int), confidence (float)}`. High = priority 100+i, confidence 1.0. Medium = 200+i, 0.8. Low = 300+i, 0.6. This numeric priority drives deterministic label disambiguation in `seed_labeler.py`.

#### Keyword Dictionaries
- **`CATEGORY_KEYWORDS`** (`dict[str, list[str]]`, 9 categories): Debit-side keyword lists. Keys must exactly match `CATEGORY_PRIORITY` entries. All keywords are matched against normalized (lowercased, stripped) remarks.
- **`CREDIT_KEYWORDS`** (`dict[str, list[str]]`, 4 categories): Credit-side keyword lists. Categories: `salary`, `refund`, `interest`, `transfer_in`.
- **`CREDIT_PRIORITY`**: `["salary", "refund", "interest", "transfer_in"]`.

#### Noise Tokens
- **`NOISE_TOKENS`** (`set[str]`, 16 entries): Semantically empty tokens stripped during remark cleaning. Includes: `ref`, `no`, `by`, `being`, `towards`, `payment`, `txn`, `transaction`, `cr`, `dr`, `ac`, `a/c`, `the`, `and`, `or`, `to`, `from`.

#### Pipeline Constants
- **`FALLBACK_DEBIT_LABEL`**: `"uncategorized"` — assigned when no keyword matches.
- **`FALLBACK_CREDIT_LABEL`**: `"other_credit"` — same for credits.
- **`MIN_COVERAGE_THRESHOLD`**: `0.40` — labeling coverage below 40% triggers a warning.

#### Recurring Transaction Config
- **`RECURRING_CONFIG`** (`dict`): Defines 4 frequency buckets (`monthly`, `weekly`, `biweekly`, `quarterly`) each with `min_gap`, `max_gap`, and `var` thresholds in days. Also `global` config: `amount_tolerance=0.20`, `min_occurrences=3`, `fluctuation_penalty_threshold=0.10`.

#### Insight Engine ML Config
- **`INSIGHT_TYPES`** (`list[str]`, 5 entries): Classification targets for the Insight Ranker: `spending_spike`, `subscription`, `trend_warning`, `budget_risk`, `no_action`.
- **`TIP_CORPUS`** (`dict[str, dict]`, ~35 entries): Human-vetted financial tips, each tagged with applicable `categories` (list) and `insights` (list). Empty `categories` = universal applicability. Organized by domain: food, shopping, transport, utilities, entertainment, finance, health, ATM, transfer, generic.
- **`INSIGHT_TEMPLATES`** (`dict[str, list[str]]`): Multiple phrasing variants per insight type for variety. Each template contains Python format-string placeholders (`{category}`, `{merchant}`, `{amount}`, `{pct}`, `{date}`, `{frequency}`).

#### Function: `lookup_matching_tip_ids(category, insight_type) → list[str]`
Two-pass TIP_CORPUS lookup: category-specific first, then generic (empty categories list). Returns list of matching tip IDs. Used by both `insight_generator._select_tip` and `training_data_generator._find_best_tip` to centralize iteration logic.

### Dependencies
None (pure Python data declarations + one function).

### Consumed By
`preprocessor.py`, `seed_labeler.py`, `recurring_detector.py`, `insight_generator.py`, `training_data_generator.py`.

---

## 2. `schema.py` (194 lines)

### Purpose
Central contract registry for every DataFrame column name used across the pipeline. Prevents silent breakage from column renames by forcing all modules to reference constants from `Col`.

### Class: `Col`
A namespace class (not instantiated) holding string constants for every column name:

| Group | Constants |
|-------|-----------|
| Raw Input | `DATE`, `AMOUNT`, `AMOUNT_FLAG`, `REMARKS`, `BALANCE` |
| Preprocessor Output | `SIGNED_AMOUNT`, `CLEANED_REMARKS` |
| Feature Engineer Output | `DOW_SIN`, `DOW_COS`, `MONTH_SIN`, `MONTH_COS`, `IS_WEEKEND`, `WEEK_OF_MONTH`, `ROLLING_7D_MEAN`, `ROLLING_7D_STD`, `ROLLING_30D_MEAN`, `AMOUNT_LOG`, `AMOUNT_ZSCORE` |
| Seed Labeler Output | `PSEUDO_LABEL`, `LABEL_REASON`, `LABEL_KEYWORD`, `LABEL_KEYWORD_NORM`, `LABEL_CONFIDENCE` |
| Categorization Model Output | `PREDICTED_CATEGORY` |
| Expected Spend Output | `EXPECTED_AMOUNT`, `RESIDUAL`, `PERCENT_DEVIATION` |
| Anomaly Detector Output | `IS_ANOMALY` |
| Recurring Detector Output | `IS_RECURRING`, `RECURRING_FREQUENCY`, `RECURRING_CONFIDENCE`, `RECURRING_SCORE` |
| ML Insight Engine | `CATEGORY_CONFIDENCE`, `INSIGHT_TYPE`, `TIP_ID`, `INSIGHT_SCORE` |

#### Static Methods (Input Contracts)
Each returns a `FrozenSet[str]` defining the required columns for a specific module:

| Method | Required Columns |
|--------|-----------------|
| `raw_input()` | `date`, `amount`, `amount_flag`, `remarks` |
| `feature_engineer_input()` | `date`, `amount`, `signed_amount` |
| `seed_labeler_input()` | `cleaned_remarks` |
| `categorization_model_input()` | `cleaned_remarks`, `amount_log` |
| `expected_spend_input()` | `amount`, `predicted_category`, `rolling_7d_mean` |
| `anomaly_detector_input()` | `amount_zscore`, `percent_deviation`, `amount` |
| `recurring_detector_input()` | `cleaned_remarks`, `date`, `amount` |
| `insight_generator_input()` | `is_anomaly`, `is_recurring` |
| `insight_ranker_input()` | 14 columns (amount, zscore, deviation, confidence, anomaly, recurring, weekend, rolling stats, month encoding, amount_log, predicted_category) |

### Function: `require_columns(df, required, module_name) → None`
Validates a DataFrame contains all required columns. Raises `ValueError` with sorted list of missing columns and available columns.

### Function: `coerce_and_validate_types(df) → DataFrame`
Coerces the `amount` column to numeric (`pd.to_numeric` with `errors="coerce"`). Rows with unparseable garbage are dropped and logged. Uses DROP+LOG philosophy: corrupted data is removed, never silently converted.

### Dependencies
- `pandas` — DataFrame operations.
- `logger_factory` — structured logging.

### Design Rationale
By centralizing column names as constants, a column rename requires exactly one string change. Every consumer picks up the change via import. The `FrozenSet` return type prevents accidental mutation.

---

## 3. `logger_factory.py` (56 lines)

### Purpose
Provides structured JSON logging with automatic pipeline run ID injection via `contextvars`.

### Components

#### `pipeline_run_id_ctx` (`ContextVar[str]`)
Thread-safe context variable holding the current pipeline run ID. Default: `"UNKNOWN_RUN"`. Set by `generate_new_run_id()` at pipeline invocation start.

#### `generate_new_run_id() → str`
Creates a UUID-based run ID (`run_{hex8}`) and stores it in the context variable. Called at the top of `run_pipeline()` and `run_inference()`.

#### Class: `JSONFormatter(logging.Formatter)`
Custom formatter that emits structured JSON log lines with fields:
- `timestamp` — UTC ISO-8601.
- `level` — log level.
- `logger` — logger name (module path).
- `message` — formatted message.
- `pipeline_run_id` — injected from `contextvars`.
- `event_type` — optional, from `record.event_type`.
- `metrics` — optional dict, from `record.metrics`.
- `exception` — traceback string if present.

#### `get_logger(name) → Logger`
Factory function returning a logger configured with `JSONFormatter`. Guards against duplicate handler attachment. Sets `propagate=False` to prevent root logger pollution.

### Dependencies
- `json`, `logging`, `contextvars`, `uuid`, `datetime` (all stdlib).

### Consumed By
Every module in the codebase.

---

## 4. `preprocessor.py` (302 lines)

### Purpose
Entry point for data normalization. Performs schema validation, date parsing, flag normalization, signed amount computation, remark cleaning with merchant aliasing, zero-row removal, deduplication, and debit/credit splitting.

### Module-Level Compiled Patterns
- `_LONG_DIGIT_PATTERN` — matches 4+ digit runs (UPI refs, phone numbers).
- `_EMAIL_PATTERN` — matches email-like patterns.
- `_SPECIAL_CHAR_PATTERN` — matches non-alphanumeric, non-space characters.
- `_MULTI_SPACE_PATTERN` — matches consecutive whitespace.
- `_COMPILED_SPECIFIC` — pre-compiled `(regex, alias)` tuples from `SPECIFIC_MERCHANT_ALIASES`.
- `_COMPILED_GENERIC` — pre-compiled `(regex, alias)` tuples from `GENERIC_ROUTER_ALIASES`.

These are compiled at module load time (once), not per-transaction.

### Functions

#### `validate_schema(df) → None`
Delegates to `require_columns(df, Col.raw_input(), "preprocessor")`. Logs success.

#### `_parse_and_sort_dates(df) → DataFrame`
Parses the `date` column using strict ISO-8601 (`YYYY-MM-DD`). Raises `ValueError` on failure. Sorts chronologically. Logs min/max date range.

#### `_normalize_flag(flag) → Optional[str]`
Normalizes a single `amount_flag` to `"DR"` or `"CR"`. Strips whitespace, uppercases. Returns `None` for non-string or unrecognized values.

#### `_compute_signed_amount(df) → DataFrame`
Applies `_normalize_flag` to every row. Invalid flags default to `"DR"` (defensive). Computes `signed_amount`: DR → `-abs(amount)`, CR → `+abs(amount)`.

#### `normalize(text) → str`
Strips all special characters including `@` and `&`, lowercases, collapses whitespace. Returns pure alphanumeric + space. Used for safe regex `\b` boundary matching.

#### `clean_remark(remark) → str`
Core remark cleaning pipeline:
1. Guard: non-string/empty → `""`.
2. Lowercase.
3. **Specific merchant match**: iterates `_COMPILED_SPECIFIC`. On first match, returns the canonical alias (lowercased) immediately. This is a full replacement — the original remark is discarded.
4. **Generic router substitution**: iterates `_COMPILED_GENERIC`. On match, the router pattern is stripped from the text (not replaced with alias). This preserves the merchant identity embedded after routing information.
5. **Standard fallback**: strips emails, long digit runs, special characters, noise tokens, single-character tokens.

**Critical design choice**: Specific merchants short-circuit (return early). Generic routers only strip — they never overwrite unique merchant identity embedded in UPI narrations.

#### `_drop_zero_amount(df) → DataFrame`
Removes rows where `amount == 0`. Logs count.

#### `_deduplicate(df) → DataFrame`
Drops duplicates on `(date, amount, remarks, amount_flag)`, keeping first.

#### `_split_debit_credit(df) → Tuple[DataFrame, DataFrame]`
Splits by `amount_flag == "DR"` vs `"CR"`. Both independently reset-indexed.

#### `preprocess(df) → Tuple[DataFrame, DataFrame]`
Public API orchestrating the full pipeline: validate → coerce types → parse dates → compute signed amount → drop zeros → deduplicate → clean remarks → split.

### Dependencies
- `config.py` — `NOISE_TOKENS`, `SPECIFIC_MERCHANT_ALIASES`, `GENERIC_ROUTER_ALIASES`.
- `schema.py` — `Col`, `require_columns`, `coerce_and_validate_types`.
- `logger_factory.py` — `get_logger`.

### Output Contract
Two DataFrames (debits, credits) each containing: original columns + `signed_amount` + `cleaned_remarks`.

---

## 5. `feature_engineer.py` (304 lines)

### Purpose
Produces time, rolling, and amount features with strict leakage prevention. Rolling statistics use `shift(1)` to ensure row i's window only contains rows 0…i-1.

### Constants
- `ZSCORE_CLIP = 5.0` — clipping bound for z-score.

### Functions

#### `add_time_features(df) → DataFrame`
Adds from the `date` column:
- `is_weekend` — binary (Saturday/Sunday).
- `week_of_month` — 1-5 based on day of month.
- `month_sin`, `month_cos` — cyclical 12-period encoding.
- `dow_sin`, `dow_cos` — cyclical 7-period encoding.

Raises `TypeError` if `date` is not datetime64.

#### `add_rolling_features(df, amount_col) → DataFrame`
**Leakage-safe**: Shifts the amount column by 1 position FIRST, then applies `.rolling()`.
- `rolling_7d_mean` — mean(shifted, window=7, min_periods=1).
- `rolling_30d_mean` — mean(shifted, window=30, min_periods=1).
- `rolling_7d_std` — std(shifted, window=7, min_periods=2).

Short DataFrames (≤1 row) get NaN columns.

#### `fill_rolling_nulls(df, global_mean, global_std) → DataFrame`
Fills NaN from rolling window warmup with training-set statistics. **CRITICAL**: `global_mean`/`global_std` must be from the training partition only. Deriving from `df` would leak test set information.

#### `add_amount_features(df, amount_col) → DataFrame`
- `amount_log` — `log1p(|amount|)` (safe for negatives).
- `amount_zscore` — `(amount - rolling_7d_mean) / rolling_7d_std`, clipped to `[-5, 5]`. Zero-std rows use std=1.0.

#### `engineer_features(df, global_mean, global_std, amount_col) → DataFrame`
Public API for training: sorts chronologically, then chains time → rolling → fill → amount features. Warns if global stats are auto-derived (acceptable for inference, not training).

#### `engineer_features_inference(new_txn, history_df, global_mean, global_std, amount_col) → DataFrame`
Stateless live-inference entrypoint. Concatenates `new_txn` with `history_df`, uses a `__is_new_txn__` boolean tag column to track which rows are new. Runs full `engineer_features` on the combined set, then extracts only tagged rows. This ensures rolling windows reflect real spending history, not just the new transaction in isolation.

### Dependencies
- `schema.py` — `Col`, `require_columns`.
- `numpy`, `pandas`.

---

## 6. `seed_labeler.py` (215 lines)

### Purpose
Converts cleaned remarks into pseudo-labels using keyword dictionaries from `config.py`. These pseudo-labels become training targets for the ML categorization model.

### Dataclass: `CompiledKeyword`
Fields: `text`, `norm`, `pattern` (compiled regex), `category`, `tier_name`, `priority` (int), `confidence` (float).

### Functions

#### `_compile_keywords(keyword_map, is_credit) → list[CompiledKeyword]`
Compiles keyword strings into regex patterns using `\b` word boundaries on already-normalized text. For debit keywords, lookups `TIER_MAPPING` for priority/confidence metadata. Credit keywords get default `priority=999, confidence=0.5`.

#### `_match_remark(norm_remark, compiled_keywords, fallback) → tuple`
Matches a normalized remark against all compiled keywords. On multiple matches: selects the lowest-priority tier (highest priority), then within that tier selects the longest keyword (most specific), breaking ties lexicographically.

Returns: `(label, reason, keyword_triggered, keyword_norm, confidence)`.

#### `_log_coverage(df, label_col, fallback_labels, context, min_coverage_threshold) → float`
Computes labeling coverage (fraction of rows NOT assigned fallback labels). Warns if below `MIN_COVERAGE_THRESHOLD` (40%).

#### `label_debits(df, remark_col, label_col, keyword_map) → DataFrame`
Public API for debit labeling. Re-normalizes the remark column as a boundary hardening measure (safe even if called without preprocessor). Applies `_match_remark` to every row, producing 5 metadata columns: `pseudo_label`, `label_reason`, `label_keyword`, `label_keyword_norm`, `label_confidence`.

#### `label_credits(df, ...) → DataFrame`
Same structure as `label_debits`, using `CREDIT_KEYWORDS` and `FALLBACK_CREDIT_LABEL`.

### Module-Level Pre-Compilation
- `_DEFAULT_DEBIT_KWS` — compiled from `CATEGORY_KEYWORDS`.
- `_DEFAULT_CREDIT_KWS` — compiled from `CREDIT_KEYWORDS`.

Compiled once at module load. If custom `keyword_map` is passed to `label_debits`/`label_credits`, keywords are recompiled.

### Dependencies
- `config.py` — all keyword/priority/fallback constants.
- `preprocessor.py` — `normalize` function.
- `schema.py` — `Col`, `require_columns`.

---

## 7. `categorization_model.py` (127 lines)

### Purpose
Trains and applies an ML classifier to generalize pseudo-labels to unseen transaction remarks.

### Function: `build_categorization_pipeline() → Pipeline`
Constructs a sklearn Pipeline:
- **Preprocessor** (`ColumnTransformer`):
  - `text` — `TfidfVectorizer(ngram_range=(1,2), max_features=2000)` on `cleaned_remarks`.
  - `num` — `StandardScaler(with_mean=False)` on `amount_log`. `with_mean=False` prevents sparse→dense explosion.
- **Classifier**: `LogisticRegression(class_weight="balanced", max_iter=1000)`.
- `sparse_threshold=1.0` forces sparse output to prevent OOM on large datasets.

### Function: `train_categorization_model(df, label_col) → Pipeline`
Validates input contract. Drops rows with missing critical columns. **Excludes fallback labels** (`uncategorized`, `other_credit`) from training to prevent the model from learning the fallback as a class — the model should predict real categories from text patterns. Logs training accuracy.

### Function: `predict_categories(pipeline, df) → DataFrame`
Predicts `predicted_category` and `category_confidence` (max probability across all classes). Fills missing strings with `""` and missing amounts with `0.0` before prediction.

### Dependencies
- `config.py` — `FALLBACK_DEBIT_LABEL`, `FALLBACK_CREDIT_LABEL`.
- `schema.py` — `Col`, `require_columns`.
- `sklearn` — TF-IDF, StandardScaler, LogisticRegression, ColumnTransformer, Pipeline.

---

## 8. `expected_spend_model.py` (135 lines)

### Purpose
Trains a regression model estimating the normal expected spending amount for a transaction. The residual (actual - expected) feeds the anomaly detector.

### Function: `build_spend_pipeline() → Pipeline`
- **Numeric features** (9): `is_weekend`, `month_sin/cos`, `dow_sin/cos`, `week_of_month`, `rolling_7d_mean`, `rolling_30d_mean`, `rolling_7d_std`.
- **Categorical features** (1): `predicted_category` (OneHotEncoded, `handle_unknown="ignore"`).
- **Regressor**: `RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0])` — linear with automatic regularization tuning. Linear model prevents wild extrapolation.

### Function: `train_expected_spend_model(df, target_col) → Pipeline`
Drops rows with missing essential columns. Logs R² score.

### Function: `predict_expected_spend(pipeline, df) → DataFrame`
Adds 3 columns:
- `expected_amount` — model prediction.
- `residual` — `actual - expected`.
- `percent_deviation` — `residual / |expected|`.

**Division safety**: `|expected|` clipped to `lower=1.0` to prevent division by zero and sign inversion from negative RidgeCV extrapolation. Clipping events are logged.

### Dependencies
- `schema.py` — `Col`, `require_columns`.
- `sklearn` — StandardScaler, OneHotEncoder, RidgeCV, ColumnTransformer, Pipeline.

---

## 9. `anomaly_detector.py` (49 lines)

### Purpose
Flags unusual transactions using a dual-gate composite heuristic.

### Function: `detect_anomalies(df, zscore_threshold=3.0, pct_dev_threshold=0.5) → DataFrame`
A transaction is flagged `is_anomaly=True` only if BOTH conditions hold:
1. `|amount_zscore| > zscore_threshold` — historically unusual (statistical).
2. `|percent_deviation| > pct_dev_threshold` — ML expected spend failure (contextual).

This composite prevents low-value but statistically unusual expenses from triggering alarms. Both `.abs()` gates catch spikes in either direction.

### Dependencies
- `schema.py` — `Col`, `require_columns`.

---

## 10. `recurring_detector.py` (118 lines)

### Purpose
Groups transactions by merchant and identifies stable temporal frequencies indicating subscriptions, standing orders, or recurring bills.

### Function: `find_recurring_transactions(df, group_col) → DataFrame`
Groups by `cleaned_remarks`. For each group with ≥ `min_occurrences` (3):

1. **Amount Score (A)**: `1.0 - (amount_drift / amount_tolerance)`. Measures how stable the charge amount is. `amount_drift = (max - min) / mean`. Clamped to [0, 1].

2. **Temporal Score (T)**: Assigns a frequency bucket (weekly/biweekly/monthly/quarterly) if `mean_gap` falls within `[min_gap, max_gap]`. Score: `1.0 - (variance / expected_var)`. Clamped to [0, 1].

3. **Volume Score (V)**: `len(group) / 12.0` — normalized to 1.0 for a "perfect year" of occurrences. Clamped to [0, 1].

**Rejection gate**: If T=0 or A=0, the group is rejected (no frequency match or wildly varying amounts).

**Final score**: `0.4*A + 0.4*T + 0.2*V`.

Output columns: `is_recurring` (bool), `recurring_frequency` (str), `recurring_confidence` (float), `recurring_score` (float).

### Dependencies
- `config.py` — `RECURRING_CONFIG`.
- `schema.py` — `Col`, `require_columns`.

---

## 11. `insight_model.py` (222 lines)

### Purpose
Loads and applies the pre-trained LightGBM Insight Ranker, implementing security-hardened model loading.

### Constants
- `NUMERIC_FEATURES` — 13 features matching `Col.insight_ranker_input()`.
- `CATEGORICAL_FEATURES` — `[predicted_category]`.
- `_MODELS_DIR` — `models/` relative to `__file__`.

### Exception: `ModelSecurityError`
Raised on integrity verification failure.

### Security Functions

#### `_compute_checksum(file_path) → str`
SHA-256 hex digest, computed in 8KB chunks.

#### `_verify_checksum(model_path, checksum_path) → bool`
Reads expected checksum from `.sha256` companion file. Compares against actual. Raises `ModelSecurityError` on mismatch.

#### `_validate_model_path(model_path) → str`
Canonicalizes path via `os.path.realpath()`. Verifies it starts with the `_MODELS_DIR` prefix. Prevents path traversal and symlink attacks.

### Function: `load_insight_ranker(model_path) → Optional[Pipeline]`
Loading sequence:
1. Check file existence → return `None` if missing (graceful degradation).
2. Validate path (canonicalization + directory check).
3. Verify checksum (SHA-256). Refuse unsigned models (missing `.sha256` file).
4. `pickle.load()` on verified file.

Returns `None` on any non-security failure. **Raises** `ModelSecurityError` on checksum mismatch (potential tampering).

### Function: `predict_insight_scores(pipeline, df) → DataFrame`
If pipeline is `None`, defaults all scores to 0.0 (rule-based fallback). Otherwise:
1. Validates input contract.
2. Fills NaN defensively (0.0 for numeric, `"unknown"` for categorical).
3. Aligns column order to `pipeline.feature_names_in_` if exposed.
4. Calls `pipeline.predict_proba(X)`.
5. Computes `insight_score = 1.0 - P(no_action)`. The `no_action` class means zero insight value; 1−P(no_action) measures how "insightful" a transaction is.
6. Catches all exceptions, falling back to 0.0.

### Dependencies
- `schema.py` — `Col`, `require_columns`.
- `hashlib`, `os`, `pickle` — file I/O and security.
- `sklearn.pipeline.Pipeline` — type annotations.

---

## 12. `insight_generator.py` (182 lines)

### Purpose
Translates the enriched DataFrame into human-readable financial insight strings, ranked by the ML insight score.

### Function: `_select_tip(category, insight_type, rng) → str`
Uses `lookup_matching_tip_ids()` from config.py for a 2-pass lookup (specific → generic). Randomly selects from matches using a seeded `random.Random` instance. Returns empty string if no match.

### Function: `generate_human_insights(df, top_n=10, seed=42) → List[str]`
1. **Subscriptions**: Groups recurring transactions by merchant. For each group, formats a subscription insight using `INSIGHT_TEMPLATES["subscription"]` with merchant, amount, frequency.
2. **Anomalies**: Iterates anomaly-flagged rows. Formats spending spike insights using `INSIGHT_TEMPLATES["spending_spike"]` with category, merchant, amount, percent deviation, date.
3. **Diversity Ranking**: Two-pass algorithm guarantees at least one insight per type (subscription, spike) appears if available. Pass 1: grab highest-scoring insight of EACH type. Pass 2: fill remaining quota with absolute highest scores.
4. Each insight is optionally paired with a tip from TIP_CORPUS.
5. Final output sorted strictly by ML insight score.

### Reproducibility
All random selections (template choice, tip selection) use a seeded `random.Random(seed)` instance. Identical inputs + identical seed → identical outputs.

### Dependencies
- `config.py` — `TIP_CORPUS`, `INSIGHT_TEMPLATES`, `lookup_matching_tip_ids`.
- `schema.py` — `Col`, `require_columns`.

---

## 13. `pipeline.py` (265 lines)

### Purpose
Central orchestrator wiring all modules into a single `run_pipeline()` call. No module beyond this file needs to know execution order.

### Dataclass: `PipelineResult` (frozen)
Immutable container: `debits`, `credits` (DataFrames), `insights` (list[str]), `cat_pipeline`, `spend_pipeline`, `ranker_pipeline` (Optional sklearn Pipelines), `global_mean`, `global_std` (floats). Provides `.replace(**kwargs)` for functional updates.

### Function: `finalize_df(df) → DataFrame`
Fills `recurring_score` NaN with 0.0 to guarantee float consistency.

### Function: `_optimize_memory_footprint(df) → DataFrame`
Post-ML downcasting: `predicted_category` → pandas `category` dtype, `is_weekend` → `bool`.

### Function: `train_models(debits, label_col, target_col) → InsightModelState`
Trains all 3 ML models (categorization, expected spend, insight ranker) and packages them into `InsightModelState`.

### Function: `run_pipeline(raw_df, zscore_threshold, pct_dev_threshold, label_col, target_col, state) → PipelineResult`
Full pipeline in 6 phases:
1. **Preprocess** — `preprocess(raw_df)`.
2. **Seed Label** — `label_debits(debits)`, `label_credits(credits)`.
3. **Feature Engineer** — computes global stats, `engineer_features(debits)`.
4. **ML Models** — if `state is None`, trains all models; otherwise uses pre-trained. Then predicts categories + expected spend.
5. **Signal Detection** — `detect_anomalies()`, `find_recurring_transactions()`.
6. **Insight Generation** — `predict_insight_scores()`, finalizes, optimizes memory, `generate_human_insights()`.

If `state` is provided, skips training (inference-only mode).

### Function: `run_inference(new_txn, state, history_df, ...) → PipelineResult`
Inference path for new transactions with pre-trained models:
1. Preprocess + seed label.
2. Feature engineer using `engineer_features_inference` (history-aware).
3. Predict categories + expected spend.
4. Detect anomalies + recurring.
5. Score + generate insights.

Performance warning documented: recomputes historical rolling windows. Batch new_txn for efficiency.

### Dependencies
All other engine modules + `model_state.py`.

---

## 14. `model_state.py` (60 lines)

### Purpose
Immutable serialization container for ML pipeline components. Prevents accidental PII persistence.

### Dataclass: `InsightModelState`
Fields: `pipeline_version` (str), `cat_pipeline`, `spend_pipeline`, `ranker_pipeline` (Optional sklearn Pipelines), `global_mean`, `global_std` (floats).

### Function: `save_model_state(filepath, state) → None`
Type-asserts `global_mean` and `global_std` are floats (blocks accidental DataFrame serialization). Uses `joblib.dump`.

### Function: `load_model_state(filepath) → InsightModelState`
Loads via `joblib.load`. Validates `pipeline_version == "1.0.0"`. Raises `ValueError` on mismatch.

### Dependencies
- `joblib` — serialization.
- `numpy` — type checking.
- `sklearn.pipeline.Pipeline` — type annotations.

---

## 15. `training_data_generator.py` (430 lines)

### Purpose
Generates synthetic labeled datasets for training the Insight Ranker and Tip Selector models. No real user data is used.

### Constants
- `ALL_CATEGORIES` — `CATEGORY_PRIORITY + ["uncategorized"]`.
- `ACTIONABLE_INSIGHTS` — all `INSIGHT_TYPES` except `no_action`.

### Functions

#### `_find_best_tip(category, insight_type) → str`
Deterministic tip selection using `lookup_matching_tip_ids`. Returns first match or `"no_tip"`.

#### `_generate_base_features(n, rng) → DataFrame`
Generates 14-column synthetic feature vectors calibrated to Indian bank statement distributions:
- Amounts: lognormal centered around ~₹500 (range ₹10–₹100,000).
- Rolling stats: derived from amount with noise.
- Category confidence: Beta(5,2) skewed toward high confidence.
- Categories: weighted distribution matching typical Indian spending.

#### `_apply_labels(df, rng) → DataFrame`
Applies insight_type and tip_id labels. Distribution: ~60% no_action, ~10% each for spending_spike, subscription, trend_warning, budget_risk. **Crucially adjusts feature values** to be consistent with labels:
- Spikes get high z-scores (3–5), high deviation, `is_anomaly=1`, higher amounts.
- Subscriptions get `is_recurring=1`, low z-scores, low deviation, typical subscription amounts (₹99–₹1499).
- Trend warnings get `rolling_7d_mean > rolling_30d_mean * 1.2`.
- Budget risk gets elevated z-scores (2–3), moderate deviation.
- No-action gets benign feature values, all flags cleared.

#### `_add_edge_cases(df, n_edge, rng) → DataFrame`
Adds 5 categories of deliberately ambiguous samples:
1. Borderline z-scores (2.8–3.1) → budget_risk.
2. High z-score but tiny amount (₹10–50) → no_action.
3. Recurring-looking but high variance → no_action.
4. Weekend spending spikes → spending_spike.
5. Low-confidence categorization with anomaly features → spending_spike.

#### `generate_insight_dataset(n_samples, n_edge_cases, test_size, random_state) → Tuple`
Public API. Returns `(X_train, X_test, y_train, y_test)`. Stratified split on `insight_type`. Logs class distribution.

### Dependencies
- `config.py` — `CATEGORY_PRIORITY`, `INSIGHT_TYPES`, `TIP_CORPUS`, `lookup_matching_tip_ids`.
- `numpy`, `pandas`, `sklearn.model_selection`.

---

## 16. `model_benchmark.py` (604 lines)

### Purpose
Trains and evaluates 12 candidate ML models on synthetic data to identify the best architecture for Insight Ranking (5-class) and Tip Selection (~36-class).

### Model Registry (12 Models)
`LogisticRegression`, `GradientBoosting`, `RandomForest`, `LinearSVC` (calibrated), `KNeighbors`, `DecisionTree`, `MLPClassifier`, `AdaBoost`, `ExtraTrees`, `XGBoost`, `LightGBM`, `CatBoost`.

### Benchmark Structure
- 2 tasks: Insight Ranker + Tip Selector.
- 2 variants per task: with/without `is_anomaly` (leakage test).
- Metrics: CV accuracy, F1 macro/weighted, test accuracy, precision, recall, train time, inference latency (ms/sample), model size (KB), overfit gap.

### Leakage Detection
Compares F1 delta between with/without `is_anomaly`. Delta > 0.15 triggers a leakage warning.

### Dependencies
- `training_data_generator.py` — synthetic data.
- `sklearn`, `xgboost`, `lightgbm`, `catboost`.

---

## 17. `train_and_save_models.py` (82 lines)

### Purpose
Trains the production LightGBM Insight Ranker and serializes it to `models/insight_ranker.pkl` with SHA-256 checksum.

### Function: `train_and_save()`
1. Generates synthetic data (5000 samples + 500 edge cases).
2. Builds pipeline: StandardScaler + OneHotEncoder + LGBMClassifier.
3. Trains on `insight_type`.
4. Saves pickle to `models/insight_ranker.pkl`.
5. Writes SHA-256 checksum to `models/insight_ranker.pkl.sha256`.

### Dependencies
- `training_data_generator.py`, `insight_model.py` (`_compute_checksum`), `schema.py`.
- `lightgbm`, `sklearn`, `pickle`.

---

## 18. `demo.py` (22 lines)

### Purpose
Minimal usage example. Loads `test-data/scrubbed.csv`, maps bank columns to schema constants, runs `run_pipeline()`, prints insights.

---

## 19. `tutorial_real_data.py` (96 lines)

### Purpose
Demonstrates real bank statement ingestion. Supports CSV, JSON, Parquet. Shows correct column mapping pattern. Includes a self-contained mock data test in `__main__`.

---

## 20. `requirements.txt` (5 entries)

```
pandas==3.0.1
numpy==2.4.3
scikit-learn==1.8.0
lightgbm==4.6.0
scipy==1.17.1
```

Runtime-only. Minimal footprint.

---

## 21. `requirements-dev.txt` (11 entries)

Adds: `xgboost==3.2.0`, `catboost==1.2.10` (benchmarking), `pytest==9.0.2`, `psutil==7.2.2`, `joblib==1.5.3` (testing/profiling), `pyarrow`, `fastparquet`, `openpyxl` (file connectors).
