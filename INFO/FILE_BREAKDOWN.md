# Insight Engine — File-by-File Breakdown

> **Scope**: Every non-excluded source file in the repository.
> **Audience**: A developer who is smart but has never seen this codebase.

---

## Table of Contents

| # | File | Role |
|---|------|------|
| 1 | `config.py` | Central configuration & domain knowledge |
| 2 | `schema.py` | DataFrame column contract registry |
| 3 | `logger_factory.py` | Structured JSON logging infrastructure |
| 4 | `model_state.py` | ML model serialization container |
| 5 | `preprocessor.py` | Data cleaning & normalization |
| 6 | `feature_engineer.py` | Leak-free feature engineering |
| 7 | `seed_labeler.py` | Keyword-based pseudo-label generation |
| 8 | `categorization_model.py` | ML transaction categorization |
| 9 | `expected_spend_model.py` | Expected spend regression |
| 10 | `anomaly_detector.py` | Statistical anomaly flagging |
| 11 | `recurring_detector.py` | Recurring transaction identification |
| 12 | `insight_model.py` | ML insight ranking (LightGBM loader) |
| 13 | `insight_generator.py` | Natural language insight output |
| 14 | `pipeline.py` | Central orchestrator |
| 15 | `training_data_generator.py` | Synthetic labeled dataset generator |
| 16 | `train_and_save_models.py` | Model training & serialization script |
| 17 | `model_benchmark.py` | 12-model benchmark evaluation |
| 18 | `demo.py` | Minimal pipeline demo |
| 19 | `tutorial_real_data.py` | Real data ingestion tutorial |
| 20 | `fix_detector.py` | One-shot patch script (legacy) |
| 21 | `requirements.txt` | Production dependencies |
| 22 | `requirements-dev.txt` | Development/test dependencies |
| 23 | `.gitignore` | Repository exclusion rules |
| 24–34 | `tests/*.py` | Test suites (7 test files + 4 runners) |

---

## 1. `config.py` (638 lines)

**Purpose**: The central brain of domain knowledge. Contains every configurable constant: merchant aliases, category keywords, priority tiers, recurring detection thresholds, insight types, tip corpus, and insight templates.

**Role in system**: Imported by almost every other module. Acts as the single source of truth for all business rules. No module should hardcode category names, keywords, or thresholds — they come from here.

**Key Data Structures**:

### `MERCHANT_ALIASES: dict[str, str]` (lines 15–223)
- **What**: A dictionary mapping regex patterns to canonical merchant names.
- **How it works**: Each key is a regex pattern (e.g. `r"swiggy"`). Each value is a clean merchant name (e.g. `"Swiggy"`). The preprocessor iterates these patterns against raw remarks. If a specific (non-generic) pattern matches, the remark is replaced entirely with the canonical name.
- **Coverage**: ~150 Indian merchants across 18 categories (UPI/Payments, Food, Grocery, E-commerce, Electronics, Fashion, Transport, Hotels, Fuel, Streaming, Telecom, Utilities, Healthcare, Finance, Education, Gaming, Real Estate).

### `CATEGORY_PRIORITY` / `TIER_MAPPING` (lines 229–255)
- **What**: Defines the pecking order when a remark matches keywords from multiple categories.
- **How it works**: Categories are split into 3 tiers — HIGH (finance, health, utilities), MEDIUM (food, transport, shopping, entertainment), LOW (atm, transfer). Each tier has an integer priority and a confidence score. When multiple keywords match, the one from the highest tier wins. Within a tier, the longest keyword match wins.

### `CATEGORY_KEYWORDS: dict[str, list[str]]` (lines 261–307)
- **What**: Maps each debit category to a list of keywords. Used by `seed_labeler.py` to pseudo-label transactions.
- **Example**: `"food": ["swiggy", "zomato", "domino", ...]`

### `CREDIT_KEYWORDS` / `CREDIT_PRIORITY` (lines 312–324)
- **What**: Same concept as debit keywords, but for credit transactions. Categories: salary, refund, interest, transfer_in. Fallback: `other_credit`.

### `NOISE_TOKENS: set[str]` (lines 329–332)
- **What**: Words stripped from remarks during cleaning. Examples: "ref", "no", "by", "payment", "txn", "cr", "dr".

### Pipeline Constants (lines 335–372)
- `FALLBACK_DEBIT_LABEL = "uncategorized"` — default label when no keyword matches a debit.
- `FALLBACK_CREDIT_LABEL = "other_credit"` — default label when no keyword matches a credit.
- `MIN_COVERAGE_THRESHOLD = 0.40` — if less than 40% of transactions are labeled, a warning fires.
- `RECURRING_CONFIG` — thresholds for monthly (27–33 day gap), weekly (6–8), biweekly (13–16), quarterly (85–95), plus global settings (20% amount tolerance, min 3 occurrences).

### `INSIGHT_TYPES` (lines 379–385)
- 5 classification targets: `spending_spike`, `subscription`, `trend_warning`, `budget_risk`, `no_action`.

### `TIP_CORPUS: dict[str, dict]` (lines 390–584)
- **What**: ~30 human-vetted financial tips, each tagged with applicable categories and insight types.
- **How it works**: When the insight generator detects a spending spike in "food", it looks up tips tagged with `categories=["food"]` and `insights=["spending_spike"]`. If nothing category-specific matches, it falls back to generic tips (`categories=[]`).

### `INSIGHT_TEMPLATES` (lines 588–610)
- Multiple phrasing templates per insight type. The generator randomly selects one (seeded for determinism) to avoid repetitive output.

### `lookup_matching_tip_ids()` function (lines 613–637)
- **What**: 2-pass lookup — first category-specific tips, then generic. Used by both `insight_generator.py` and `training_data_generator.py`.

**Dependencies**: None (pure data module).
**Who uses it**: `preprocessor.py`, `seed_labeler.py`, `recurring_detector.py`, `insight_generator.py`, `training_data_generator.py`.

---

## 2. `schema.py` (194 lines)

**Purpose**: The DataFrame column contract registry. Every column name used anywhere in the pipeline is defined here as a constant.

**Role in system**: Prevents silent breakage from column renames. If a column name changes, you change it once here and every module picks it up.

### Class `Col` (lines 28–142)
- **Responsibility**: Holds all column name constants as class attributes plus static methods that return required column sets per module.
- **Column Groups**:
  - **Raw Input**: `date`, `amount`, `amount_flag`, `remarks`, `balance`
  - **Preprocessor Output**: `signed_amount`, `cleaned_remarks`
  - **Feature Engineer**: `dow_sin`, `dow_cos`, `month_sin`, `month_cos`, `is_weekend`, `week_of_month`, `rolling_7d_mean`, `rolling_7d_std`, `rolling_30d_mean`, `amount_log`, `amount_zscore`
  - **Seed Labeler**: `pseudo_label`, `label_reason`, `label_keyword`, `label_keyword_norm`, `label_confidence`
  - **Categorization Model**: `predicted_category`
  - **Expected Spend**: `expected_amount`, `residual`, `percent_deviation`
  - **Anomaly Detector**: `is_anomaly`
  - **Recurring Detector**: `is_recurring`, `recurring_frequency`, `recurring_confidence`, `recurring_score`
  - **ML Insight Engine**: `category_confidence`, `insight_type`, `tip_id`, `insight_score`
- **Static methods**: `raw_input()`, `feature_engineer_input()`, `seed_labeler_input()`, etc. — each returns a `FrozenSet[str]` of required columns for that module.

### `require_columns()` function (lines 145–166)
- **Input**: DataFrame, set of required column names, module name string.
- **Output**: Raises `ValueError` listing missing columns, or does nothing if all present.
- **Logic**: `missing = required - set(df.columns)`. If `missing` is non-empty, raise.

### `coerce_and_validate_types()` function (lines 168–193)
- **Input**: DataFrame.
- **Output**: DataFrame with `amount` column coerced to float. Rows with unparseable garbage are dropped and logged.
- **Logic**: Uses `pd.to_numeric(errors="coerce")` to turn invalid amount strings into NaN, then drops NaN rows.

**Dependencies**: `pandas`, `logger_factory`.
**Who uses it**: Every pipeline module.

---

## 3. `logger_factory.py` (56 lines)

**Purpose**: Provides structured JSON logging with pipeline run ID tracing.

**Role in system**: Every module calls `get_logger(__name__)` at import time. All log output is JSON-formatted with timestamp, level, logger name, message, pipeline_run_id, and optional event_type/metrics fields.

### `pipeline_run_id_ctx` (line 8)
- A `contextvars.ContextVar` holding the current run ID. Defaults to `"UNKNOWN_RUN"`. Set by `generate_new_run_id()` at pipeline start.

### `generate_new_run_id()` (lines 10–13)
- Generates `run_<8 hex chars>` and stores it in the context variable.

### Class `JSONFormatter` (lines 16–39)
- **Responsibility**: Formats log records as JSON objects.
- **Output fields**: `timestamp` (UTC ISO), `level`, `logger`, `message`, `pipeline_run_id`, optional `event_type`, optional `metrics`, optional `exception`.

### `get_logger()` (lines 41–55)
- Returns a logger with `JSONFormatter` attached. Uses `propagate=False` to prevent double-logging through root logger. Idempotent — won't attach duplicate handlers.

**Dependencies**: `json`, `logging`, `contextvars`, `uuid`, `datetime`.
**Who uses it**: Every module.

---

## 4. `model_state.py` (60 lines)

**Purpose**: Immutable container for serialized ML pipeline components. Ensures no raw DataFrames (containing PII) are accidentally persisted to disk.

### `InsightModelState` dataclass (lines 7–18)
- **Fields**: `pipeline_version` (str), `cat_pipeline` (sklearn Pipeline or None), `spend_pipeline`, `ranker_pipeline`, `global_mean` (float), `global_std` (float).
- **Used by**: `pipeline.py` to pass pre-trained models between `run_pipeline()` and `run_inference()`.

### `save_model_state()` (lines 20–41)
- **Input**: filepath, InsightModelState.
- **Logic**: Validates that `global_mean` and `global_std` are floats (not DataFrames). Constructs a dict payload and saves via `joblib.dump()`.

### `load_model_state()` (lines 43–59)
- **Input**: filepath.
- **Logic**: Loads via `joblib.load()`, verifies `pipeline_version == "1.0.0"`, reconstructs `InsightModelState`.

**Dependencies**: `joblib`, `numpy`, `dataclasses`, `sklearn`.
**Who uses it**: `pipeline.py`.

---

## 5. `preprocessor.py` (314 lines)

**Purpose**: Cleans raw bank statement data. The first stage of the pipeline.

**Role in system**: Takes messy CSV data and produces two clean DataFrames (debits, credits) with normalized flags, signed amounts, and cleaned remarks.

### Pre-compiled Regex Patterns (lines 32–43)
- `_LONG_DIGIT_PATTERN`: Matches 4+ digit runs (account numbers, phone numbers, UPI refs) for removal.
- `_EMAIL_PATTERN`: Matches email addresses for removal.
- `_SPECIAL_CHAR_PATTERN`: Matches non-alphanumeric non-space characters.
- `_MULTI_SPACE_PATTERN`: Collapses multiple spaces.
- `_COMPILED_ALIASES`: Pre-compiled `MERCHANT_ALIASES` regex patterns (compiled once at import time).

### `validate_schema()` (lines 48–54)
- Calls `require_columns(df, Col.raw_input(), "preprocessor")`. Raises if missing columns.

### `_parse_and_sort_dates()` (lines 59–81)
- Parses `date` column as ISO 8601 (`YYYY-MM-DD`), sorts chronologically. Raises `ValueError` on parse failure.

### `_normalize_flag()` (lines 86–100)
- Normalizes a single `amount_flag` value to `"DR"` or `"CR"`. Case-insensitive, strips whitespace. Returns `None` for unrecognized/non-string input.

### `_compute_signed_amount()` (lines 103–129)
- Applies `_normalize_flag` to every row. Invalid flags default to `"DR"` (not dropped). Derives `signed_amount`: DR → −|amount|, CR → +|amount|.

### `normalize()` (lines 134–142)
- Strips ALL special characters (including @ and &) from text. Used for safe regex `\b` boundary matching. Returns lowercase, space-separated alphanumeric tokens.

### `clean_remark()` (lines 144–204)
- **The most complex function in the preprocessor.** Step-by-step:
  1. Guard: non-string/empty → return `""`.
  2. Lowercase the text.
  3. **Merchant Alias Matching**: Iterate pre-compiled aliases. If a specific (non-generic) merchant matches, return its canonical lowercase name immediately (e.g., `"swiggy"`). Generic routers (UPI Transfer, Paytm, PhonePe, etc.) are NOT returned as the merchant name — instead, their matched patterns are stripped from the text, and processing continues.
  4. **Standard Cleanup Fallback**: Remove emails, long digit runs, special characters, collapse whitespace.
  5. Filter out noise tokens and single-character tokens.
  6. Return the cleaned string.
- **OBSERVATION**: The early return for specific merchants (line 180) relies on Python 3.7+ dict insertion order. If `MERCHANT_ALIASES` is ever loaded dynamically, order must be preserved.

### `_drop_zero_amount()` (lines 209–222)
- Removes rows where `amount == 0`.

### `_deduplicate()` (lines 225–243)
- Drops exact duplicates on `(date, amount, remarks, amount_flag)`, keeping first occurrence.

### `_split_debit_credit()` (lines 248–264)
- Splits DataFrame into debits (flag=="DR") and credits (flag=="CR"). Both are independently reset-indexed.

### `preprocess()` — Public API (lines 269–313)
- **Full pipeline**: validate_schema → coerce_and_validate_types → parse_and_sort_dates → compute_signed_amount → drop_zero_amount → deduplicate → clean_remark → split_debit_credit.
- **Returns**: `(debit_df, credit_df)`.

**Dependencies**: `re`, `pandas`, `numpy`, `config`, `schema`, `logger_factory`.
**Who uses it**: `pipeline.py`, `seed_labeler.py` (imports `normalize`).

---

## 6. `feature_engineer.py` (304 lines)

**Purpose**: Produces time, rolling, and amount features from preprocessed data. Explicitly prevents data leakage.

**Role in system**: Phase 3 of the pipeline. Enriches the DataFrame with 11 new columns that ML models consume.

### `add_time_features()` (lines 46–69)
- Computes: `is_weekend` (0/1), `week_of_month` (1–5), `month_sin`/`month_cos` (cyclical 12-period), `dow_sin`/`dow_cos` (cyclical 7-period).

### `add_rolling_features()` (lines 74–110)
- **Leakage Prevention**: Shifts the amount column by 1 position BEFORE computing rolling windows. Row i's window contains only rows 0…i−1.
- Computes: `rolling_7d_mean` (window=7, min_periods=1), `rolling_30d_mean` (window=30, min_periods=1), `rolling_7d_std` (window=7, min_periods=2).
- NaNs are left for the caller to fill.
- **GOOD DESIGN**: The `shift(1)` before `rolling()` is the core leakage prevention mechanism.

### `fill_rolling_nulls()` (lines 113–144)
- Fills NaN values in rolling columns with training-set global statistics. `global_mean` fills mean columns, `global_std` fills std column. If `global_std == 0`, uses 1.0 as fallback.

### `add_amount_features()` (lines 149–184)
- `amount_log`: `log1p(|amount|)` — handles negatives, compresses scale.
- `amount_zscore`: `(amount − rolling_mean) / rolling_std`, clipped to [−5, +5]. Zero-std rows use std=1.0.

### `engineer_features()` — Public API (lines 189–243)
- Full pipeline: add_time → add_rolling → fill_nulls → add_amount. Enforces chronological sort.

### `engineer_features_inference()` (lines 246–303)
- **For live inference**: Concatenates new transaction(s) with historical data, engineers features on the combined set, then extracts only the new rows using a tag column (`__is_new_txn__`).
- **GOOD DESIGN**: Uses tag-based row tracking instead of `tail()` to handle backdated transactions correctly.

**Dependencies**: `numpy`, `pandas`, `schema`.
**Who uses it**: `pipeline.py`.

---

## 7. `seed_labeler.py` (215 lines)

**Purpose**: Converts cleaned remarks into pseudo-labels using keyword matching. These pseudo-labels become the training target for the categorization ML model.

### `CompiledKeyword` dataclass (lines 36–44)
- Holds: `text` (original keyword), `norm` (normalized), `pattern` (compiled regex), `category`, `tier_name`, `priority` (int), `confidence` (float).

### `_compile_keywords()` (lines 49–72)
- Pre-compiles all keywords from a keyword map into `CompiledKeyword` objects with `\b` word-boundary regex patterns. Builds priority/confidence from `TIER_MAPPING`.

### `_match_remark()` (lines 75–106)
- **Matching logic**:
  1. Run all compiled keyword patterns against the normalized remark.
  2. Collect all matches.
  3. Find the HIGHEST priority tier among matches.
  4. Among same-tier matches, sort by longest keyword first (then alphabetically for determinism).
  5. Return the best match's category, reason, keyword text, keyword norm, and confidence.
- Returns fallback label if no match.

### Module-level precompilation (lines 110–111)
- `_DEFAULT_DEBIT_KWS` and `_DEFAULT_CREDIT_KWS` are compiled at import time from the default keyword maps.

### `_log_coverage()` (lines 116–140)
- Logs what percentage of rows received a non-fallback label. Warns if below `MIN_COVERAGE_THRESHOLD` (40%).

### `label_debits()` / `label_credits()` — Public API (lines 145–214)
- Apply `_match_remark` to every row in the DataFrame. Produce columns: `pseudo_label`, `label_reason`, `label_keyword`, `label_keyword_norm`, `label_confidence`.
- Re-normalizes remarks as a boundary hardening measure.

**Dependencies**: `re`, `pandas`, `config`, `preprocessor` (imports `normalize`), `schema`.
**Who uses it**: `pipeline.py`.

---

## 8. `categorization_model.py` (127 lines)

**Purpose**: ML-based transaction categorization using TF-IDF + Logistic Regression.

### `build_categorization_pipeline()` (lines 32–54)
- Builds a sklearn Pipeline:
  - **Text**: TF-IDF on `cleaned_remarks` (unigrams + bigrams, max 2000 features).
  - **Numeric**: StandardScaler on `amount_log` (with_mean=False to keep sparse matrix).
  - **Classifier**: LogisticRegression with `class_weight="balanced"`, max_iter=1000.
- `sparse_threshold=1.0` forces ColumnTransformer output to stay sparse (prevents OOM).

### `train_categorization_model()` (lines 57–94)
- Drops rows with missing values or fallback labels. Fits the pipeline. Logs training accuracy.

### `predict_categories()` (lines 97–127)
- Predicts `predicted_category` and `category_confidence` (max class probability).

**Dependencies**: `sklearn`, `pandas`, `config`, `schema`.
**Who uses it**: `pipeline.py`.

---

## 9. `expected_spend_model.py` (135 lines)

**Purpose**: Regression model that estimates the "normal" expected spending for a transaction.

### `build_spend_pipeline()` (lines 31–56)
- sklearn Pipeline:
  - **Numeric**: StandardScaler on 9 features (time + rolling stats).
  - **Categorical**: OneHotEncoder on `predicted_category`.
  - **Regressor**: RidgeCV (alphas: 0.1, 1.0, 10.0, 100.0).
- **GOOD DESIGN**: RidgeCV is chosen over tree-based models because it can **extrapolate** beyond training range — trees cap at the max training value.

### `train_expected_spend_model()` (lines 59–92)
- Drops rows with missing essential columns. Fits the pipeline. Logs R² score.

### `predict_expected_spend()` (lines 95–134)
- Predicts `expected_amount`. Derives `residual = actual − expected` and `percent_deviation = residual / |expected|`.
- **Safety**: Clips `|expected|` to a minimum of 1.0 to prevent division by zero or sign inversion from negative extrapolation.

**Dependencies**: `sklearn`, `pandas`, `numpy`, `schema`.
**Who uses it**: `pipeline.py`.

---

## 10. `anomaly_detector.py` (49 lines)

**Purpose**: Flags unusual transactions using a dual-gate composite heuristic.

### `detect_anomalies()` (lines 16–48)
- **Logic**: A transaction is anomalous if BOTH conditions hold:
  1. `|amount_zscore| > zscore_threshold` (default 3.0) — statistically unusual.
  2. `|percent_deviation| > pct_dev_threshold` (default 0.5) — ML expected spend model also surprised.
- **Design rationale**: The dual-gate prevents low-value transactions from triggering alarms just because they technically miss mathematical expectations.
- **Output**: Boolean column `is_anomaly`.

**Dependencies**: `pandas`, `schema`.
**Who uses it**: `pipeline.py`.

---

## 11. `recurring_detector.py` (118 lines)

**Purpose**: Identifies subscription-like recurring transactions using a deterministic scoring equation.

### `find_recurring_transactions()` (lines 21–117)
- **Step-by-step**:
  1. Group transactions by `cleaned_remarks`.
  2. Skip groups with fewer than `min_occurrences` (default 3).
  3. Compute time gaps between consecutive occurrences.
  4. **Amount Score (A)**: `1.0 − (amount_drift / amount_tolerance)`, clamped [0,1]. Measures how consistent the amounts are.
  5. **Temporal Score (T)**: Check if mean gap fits a known frequency (monthly, weekly, biweekly, quarterly). If yes, `T = 1.0 − (variance / allowed_variance)`, clamped [0,1].
  6. **Volume Score (V)**: `len(group) / 12`, clamped [0,1]. 12 occurrences = perfect year.
  7. **Final Score**: `0.4*A + 0.4*T + 0.2*V`.
  8. If either A or T is 0.0, the group is rejected.
  9. If `assigned_freq` is set and scores pass, mark all rows in that group as recurring.
- **Output columns**: `is_recurring` (bool), `recurring_frequency` (str or None), `recurring_confidence` (float), `recurring_score` (float).

**Dependencies**: `pandas`, `numpy`, `config`, `schema`.
**Who uses it**: `pipeline.py`.

---

## 12. `insight_model.py` (208 lines)

**Purpose**: Loads a pre-trained LightGBM model from disk and uses it to score transactions by "insightfulness".

### Security Architecture (lines 40–111)
- **`ModelSecurityError`**: Custom exception for integrity failures.
- **`_compute_checksum()`**: SHA-256 hex digest of a file.
- **`_verify_checksum()`**: Compares a model file against its companion `.sha256` file. Raises `ModelSecurityError` on mismatch.
- **`_validate_model_path()`**: Canonicalizes path and verifies it's within the `models/` directory. Prevents path traversal and symlink attacks.
- **GOOD DESIGN**: Model files are never deserialized without passing all three checks (existence, path validation, checksum verification).

### `load_insight_ranker()` (lines 114–154)
- Returns `None` if file not found (graceful degradation). Returns `None` if checksum file missing (unsigned model rejected). Raises `ModelSecurityError` if checksum fails. Loads via `pickle.load()` if all checks pass.

### `predict_insight_scores()` (lines 157–207)
- If pipeline is `None`: defaults all scores to 0.0 (falls back to rule-based prioritization).
- If pipeline exists: predicts class probabilities. Score = `1.0 − P(no_action)`. The higher the score, the more "insightful" the transaction.
- Catches all exceptions gracefully, defaulting to 0.0 on failure.

**Dependencies**: `hashlib`, `os`, `pickle`, `sklearn`, `pandas`, `schema`.
**Who uses it**: `pipeline.py`, `train_and_save_models.py`.

---

## 13. `insight_generator.py` (182 lines)

**Purpose**: Translates ML flags into human-readable financial insight strings.

### `_select_tip()` (lines 26–54)
- Uses `lookup_matching_tip_ids()` from config. Randomly selects from matching tips using a seeded RNG.

### `generate_human_insights()` (lines 57–181)
- **Step-by-step**:
  1. Sorts DataFrame chronologically.
  2. **Subscriptions**: Groups recurring transactions by merchant. For each group, selects a template from `INSIGHT_TEMPLATES["subscription"]`, formats it, and selects a matching tip.
  3. **Anomalies**: For each anomalous row, selects a template from `INSIGHT_TEMPLATES["spending_spike"]`, formats with category/merchant/amount/deviation/date.
  4. **Diversity Ranking**: 2-pass algorithm:
     - Pass 1: Take the highest-scoring insight of each TYPE (subscription, spike) to guarantee variety.
     - Pass 2: Fill remaining slots up to `top_n` with highest-scoring candidates regardless of type.
  5. Sorts final selection by ML score descending.
  6. Outputs insight + tip pairs as strings.
- **Determinism**: All random selections use a seeded `random.Random(seed)` instance.

**Dependencies**: `random`, `pandas`, `config`, `schema`.
**Who uses it**: `pipeline.py`.

---

## 14. `pipeline.py` (259 lines)

**Purpose**: The central orchestrator. Wires all modules into a single `run_pipeline()` call.

### `PipelineResult` dataclass (lines 38–54)
- **Frozen (immutable)**. Contains: `debits` (DataFrame), `credits` (DataFrame), `insights` (List[str]), `cat_pipeline`, `spend_pipeline`, `ranker_pipeline`, `global_mean`, `global_std`.
- Has a `replace()` helper using `dataclasses.replace()`.

### `finalize_df()` (lines 57–62)
- Fills NaN in `recurring_score` with 0.0.

### `_optimize_memory_footprint()` (lines 64–71)
- Downcasts `predicted_category` to pandas categorical type, `is_weekend` to bool. Called AFTER all ML predictions are complete.

### `train_models()` (lines 74–90)
- Helper: trains categorization model, predicts categories, trains expected spend model, loads ranker. Returns `InsightModelState`.

### `run_pipeline()` (lines 93–181) — **Primary Entry Point**
- **Phase 1**: `preprocess(raw_df)` → debits, credits.
- **Phase 2**: `label_debits(debits)`, `label_credits(credits)`.
- **Phase 3**: Compute `global_mean`/`global_std` on debits, then `engineer_features()`.
- **Phase 4**: If no pre-trained state, train all models. Then `predict_categories()` and `predict_expected_spend()`.
- **Phase 5**: `detect_anomalies()` and `find_recurring_transactions()`.
- **Phase 5.5**: `predict_insight_scores()` via the LightGBM ranker.
- **Phase 6**: `finalize_df()`, `_optimize_memory_footprint()`, `generate_human_insights()`.
- Returns `PipelineResult`.

### `run_inference()` (lines 184–258) — **Live Inference Entry Point**
- Designed for processing new transaction(s) against pre-trained models.
- Uses `engineer_features_inference()` to stitch new transactions against historical data for accurate rolling features.
- Same phases as `run_pipeline` but skips training — uses provided `InsightModelState`.

**Dependencies**: Every pipeline module, `logger_factory`, `model_state`, `dataclasses`, `sklearn`.
**Who uses it**: `demo.py`, `tutorial_real_data.py`, tests.

---

## 15. `training_data_generator.py` (430 lines)

**Purpose**: Generates synthetic labeled datasets for training the Insight Ranker. No real user data is used.

### `_generate_base_features()` (lines 52–125)
- Creates `n` synthetic feature vectors calibrated against typical Indian transaction patterns. Uses lognormal distribution for amounts (~₹500 center), beta distribution for category confidence (skewed high), etc.

### `_apply_labels()` (lines 128–235)
- Assigns `insight_type` labels with target distribution: ~60% no_action, ~10% each for spending_spike, subscription, trend_warning, budget_risk.
- **Crucially**: Adjusts feature values to be CONSISTENT with labels. Spending spikes get high z-scores. Subscriptions get `is_recurring=1`. No-action rows get benign features.
- Assigns `tip_id` using `_find_best_tip()`.

### `_add_edge_cases()` (lines 238–365)
- Adds 5 types of adversarial samples: borderline z-scores, high z-score + tiny amount, recurring-looking but variable, weekend spikes, low-confidence categorization.

### `generate_insight_dataset()` — Public API (lines 370–429)
- Returns `(X_train, X_test, y_train, y_test)` with stratified split on `insight_type`.

**Dependencies**: `numpy`, `pandas`, `sklearn`, `config`.
**Who uses it**: `train_and_save_models.py`, `model_benchmark.py`, tests.

---

## 16. `train_and_save_models.py` (82 lines)

**Purpose**: Standalone script to train the LightGBM Insight Ranker and serialize it to disk.

### `train_and_save()` (lines 32–78)
- Generates synthetic data via `generate_insight_dataset()`.
- Builds sklearn Pipeline: StandardScaler + OneHotEncoder → LGBMClassifier (100 estimators, depth 4, balanced weights).
- Trains on `insight_type` labels.
- Saves to `models/insight_ranker.pkl`.
- Writes SHA-256 checksum to `models/insight_ranker.pkl.sha256`.

**Execution**: `python train_and_save_models.py`

**Dependencies**: `lightgbm`, `sklearn`, `pickle`, `training_data_generator`, `insight_model`, `schema`.

---

## 17. `model_benchmark.py` (604 lines)

**Purpose**: Trains and evaluates 12 candidate ML models to identify the best architecture for insight ranking and tip selection.

### `get_candidate_models()` (lines 97–185)
- Returns 12 classifiers: LogisticRegression, GradientBoosting, RandomForest, LinearSVC, KNeighbors, DecisionTree, MLP, AdaBoost, ExtraTrees, XGBoost, LightGBM, CatBoost.

### `evaluate_model()` (lines 226–315)
- 5-fold stratified cross-validation + holdout test. Computes: accuracy, F1 (macro/weighted), precision, recall, training time, inference latency, model size, overfit gap.

### `run_benchmark()` (lines 318–367)
- Runs all 12 models on a given task. Two variants per task: with and without `is_anomaly` feature (leakage test).

### `main()` (lines 480–603)
- Runs benchmarks for: (1) Insight Ranking (5-class), (2) Tip Selection (~36-class). Prints leaderboards, confusion matrices, feature importances, and leakage check.

**Execution**: `python model_benchmark.py`

**Dependencies**: `sklearn`, `xgboost`, `lightgbm`, `catboost`, `training_data_generator`.

---

## 18. `demo.py` (22 lines)

**Purpose**: Minimal example of running the pipeline on a CSV file.
- Loads `test-data/scrubbed.csv`.
- Renames columns to match schema.
- Calls `run_pipeline()`.
- Prints insights.

---

## 19. `tutorial_real_data.py` (96 lines)

**Purpose**: Tutorial demonstrating how to ingest real bank statements.
- Supports CSV, JSON, Parquet.
- Shows column mapping from bank-specific headers to `Col.*` schema.
- When run as `__main__`, creates a mock CSV, processes it, and cleans up.

---

## 20. `fix_detector.py` (6 lines)

**Purpose**: One-shot patch script that replaced a threshold condition in `recurring_detector.py` from `if T < 0.3 or A < 0.2:` to `if T == 0.0 or A == 0.0:`.

**OBSERVATION**: This is a legacy artifact — a quick fix applied directly via file manipulation. Should ideally be deleted from the repository.

---

## 21–22. `requirements.txt` / `requirements-dev.txt`

**Production**: pandas 3.0.1, numpy 2.4.3, scikit-learn 1.8.0, lightgbm 4.6.0, scipy 1.17.1.
**Development**: Adds xgboost, catboost, pytest, psutil, joblib, pyarrow, fastparquet, openpyxl.

---

## 23. `.gitignore`

Excludes: `.code-review-graph/`, `venv/`, `__pycache__/`, `*.pyc`, `.pytest_cache/`, `catboost_info/`, `models/`, `test-data/`, and various IDE/agent config files.

---

## 24–34. Test Files

### `tests/test_phase1.py` (695 lines)
- **Coverage**: preprocessor (schema validation, flag normalization, signed amount, remark cleaning with 15+ edge cases, zero-drop, dedup, full preprocess), feature_engineer (time features, rolling leakage, z-score safety, full pipeline NaN freedom, inference with backdated/empty history), seed_labeler (keyword matching, priority order, adversarial tiebreaking, coverage, metadata columns).

### `tests/test_phase2.py` (168 lines)
- **Coverage**: categorization model training/prediction, expected spend model training/prediction, RidgeCV extrapolation verification, percent_deviation safety (no inf for zero/negative expected amounts).

### `tests/test_phase3.py` (206 lines)
- **Coverage**: recurring detector (monthly detection, frequency None for non-recurring, minimum 3 occurrences, biweekly detection), anomaly composite threshold, insight generator (creates strings, deterministic output, seeded tip selection), PipelineResult immutability.

### `tests/test_e2e.py` (236 lines)
- Full end-to-end integration: generates 90 days of messy synthetic data (weekly groceries, monthly Netflix, random shopping, one $2500 spike). Validates: PII scrubbing, anomaly detection of the spike, Netflix subscription detection, insight generation. Also tests `run_inference()` with history-aware features.

### `tests/test_ml_integration.py` (67 lines)
- Tests `load_insight_ranker()` and `predict_insight_scores()` with mock data. Validates graceful fallback when pipeline is None.

### `tests/test_benchmark.py` (282 lines)
- Validates synthetic data generator: shape, no NaN/inf, distribution ratios, feature-label consistency (spikes have high z-scores, subscriptions have is_recurring=1), tip validity, amount ranges.

### `tests/test_model_security.py` (188 lines)
- Tests SHA-256 checksum computation/verification, path traversal rejection, symlink resolution, unsigned model rejection, tampered model detection, valid signed model loading.

### `tests/run_smoke.py` (107 lines)
- Smoke test runner against `test-data/scrubbed.csv`. Prints structured summary of each phase's output.

### `tests/run_stress_heavy.py` (66 lines)
- Performance stress test: multiplies base data by 10x, 50x, 100x (up to 58,100 rows). Measures throughput (rows/second).

### `tests/run_stress_legacy.py` (92 lines)
- Legacy stress test on 50,000 synthetic rows. Measures execution time and memory delta using psutil.

### `tests/run_tests_legacy.py` (19 lines)
- Legacy manual test runner for Phase 1 and Phase 2 test subsets.
