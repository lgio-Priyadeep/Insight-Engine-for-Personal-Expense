# FILE BREAKDOWN — Insight Engine

> Every file in the repository, what it does, why it exists, and how it connects to everything else.  
> **Excluded:** All paths listed in `.gitignore` (venv/, models/, test-data/, __pycache__/, .agents/, .claude/, .vscode/, catboost_info/, .code-review-graph/, and editor config files).

---

## Table of Contents

| # | File | Role |
|---|------|------|
| 1 | `schema.py` | Column registry & contract validator |
| 2 | `config.py` | Central configuration store |
| 3 | `preprocessor.py` | Raw data cleaning pipeline |
| 4 | `feature_engineer.py` | Leak-free feature computation |
| 5 | `seed_labeler.py` | Rule-based pseudo-labeling |
| 6 | `categorization_model.py` | TF-IDF + LogisticRegression categorizer |
| 7 | `expected_spend_model.py` | RidgeCV expected-amount regressor |
| 8 | `anomaly_detector.py` | Composite statistical anomaly flagger |
| 9 | `recurring_detector.py` | Heuristic subscription detector |
| 10 | `insight_model.py` | LightGBM insight ranker (loader + scorer) |
| 11 | `insight_generator.py` | NLP insight string translator |
| 12 | `pipeline.py` | Central orchestrator |
| 13 | `training_data_generator.py` | Synthetic labeled dataset factory |
| 14 | `model_benchmark.py` | 12-model benchmark harness |
| 15 | `train_and_save_models.py` | Model serialization script |
| 16 | `demo.py` | Minimal usage example |
| 17 | `tutorial_real_data.py` | Real bank statement tutorial |
| 18 | `README.md` | Project documentation |
| 19 | `requirements.txt` | Production dependencies |
| 20 | `requirements-dev.txt` | Full dev/test dependencies |
| 21 | `.gitignore` | VCS exclusion rules |
| 22 | `tests/test_phase1.py` | Phase 1 unit tests (84 tests) |
| 23 | `tests/test_phase2.py` | Phase 2 ML model tests |
| 24 | `tests/test_phase3.py` | Phase 3 signal + insight tests |
| 25 | `tests/test_benchmark.py` | Synthetic data + benchmark sanity tests |
| 26 | `tests/test_e2e.py` | Full pipeline integration test |
| 27 | `tests/test_ml_integration.py` | Insight ranker integration tests |
| 28 | `tests/run_smoke.py` | Smoke test against real CSV |
| 29 | `tests/run_stress_legacy.py` | 50k-row synthetic stress test |
| 30 | `tests/run_stress_heavy.py` | Multi-tier (5.8k–58k) stress test |
| 31 | `tests/run_tests_legacy.py` | Manual test runner (pre-pytest) |

---

## 1. `schema.py` (163 lines)

**Purpose:** Defines the single source of truth for every DataFrame column name used across the entire pipeline, and provides a runtime contract validation utility.

**Key Exports:**

| Export | Type | Description |
|--------|------|-------------|
| `Col` | class | String-constant registry. Every column name is a class attribute (e.g., `Col.DATE = "date"`, `Col.AMOUNT_ZSCORE = "amount_zscore"`). |
| `Col.preprocessor_input()` | staticmethod → `list[str]` | Returns the 4 mandatory input columns: `date`, `amount`, `amount_flag`, `remarks`. |
| `Col.categorization_model_input()` | staticmethod → `list[str]` | Returns `cleaned_remarks`, `amount_log`. |
| `Col.expected_spend_input()` | staticmethod → `list[str]` | Returns 10 columns needed by the spend regressor. |
| `Col.anomaly_detector_input()` | staticmethod → `list[str]` | Returns `amount_zscore`, `percent_deviation`. |
| `Col.recurring_detector_input()` | staticmethod → `list[str]` | Returns `date`, `amount`, `cleaned_remarks`. |
| `Col.insight_generator_input()` | staticmethod → `list[str]` | Returns the 7 columns needed for NLP translation. |
| `Col.insight_ranker_input()` | staticmethod → `list[str]` | Returns the 14 features consumed by the LightGBM ranker. |
| `require_columns(df, cols, caller)` | function | Validates that a DataFrame contains all specified columns. Raises `ValueError` with the caller's name and missing column list on failure. |

**Dependencies:** None (stdlib only).

**Consumed by:** Every module in the pipeline. `require_columns` is called at the entry point of every processing function to enforce schema contracts.

**Design Rationale:** Centralizing column names prevents typo-induced silent failures. The `staticmethod` grouping per-module makes each module's input contract explicit and auditable. There is no fallback — missing columns produce immediate, loud failures.

---

## 2. `config.py` (566 lines)

**Purpose:** The global configuration store for all business logic parameters, keyword dictionaries, merchant normalization aliases, tip corpus definitions, and insight template strings.

**Key Exports:**

| Export | Type | Description |
|--------|------|-------------|
| `NOISE_TOKENS` | `set[str]` | 30+ tokens stripped during remark cleaning (`"dr"`, `"cr"`, `"txn"`, `"payment"`, `"upi"`, etc.). |
| `MERCHANT_ALIASES` | `dict[str, str]` | Regex → canonical name mapping (e.g., `r"swiggy\|swigy"` → `"swiggy"`). 20+ entries. |
| `CATEGORY_PRIORITY` | `list[str]` | Ordered list of debit categories. Position determines winner when a remark matches multiple categories. Order: `food > transport > shopping > utilities > health > finance > entertainment > atm > transfer > education > government > insurance`. |
| `CATEGORY_KEYWORDS` | `dict[str, list[str]]` | Category → keyword list mapping. Each keyword can be multi-word (e.g., `"cash withdrawal"` for `atm`). |
| `CREDIT_PRIORITY` | `list[str]` | Priority order for credit labels: `salary > refund > cashback > interest > dividend > reversal > transfer_in`. |
| `CREDIT_KEYWORDS` | `dict[str, list[str]]` | Same structure as `CATEGORY_KEYWORDS` but for credits. |
| `FALLBACK_DEBIT_LABEL` | `str` | `"uncategorized"` — assigned when no keyword matches a debit. |
| `FALLBACK_CREDIT_LABEL` | `str` | `"other_credit"` — assigned when no keyword matches a credit. |
| `MIN_COVERAGE_THRESHOLD` | `float` | `0.60` — warning is emitted if seed labeler coverage drops below this. |
| `RECURRING_AMOUNT_TOLERANCE` | `float` | `0.05` — max cost drift (5%) for recurring detection. |
| `RECURRING_FLUCTUATION_PENALTY_THRESHOLD` | `float` | `0.10` — drift above this penalizes confidence to 0.5. |
| `TIP_CORPUS` | `dict[str, dict]` | 36 curated financial tips. Each tip has: `text` (the tip string), `categories` (list of applicable categories, empty = generic), `insights` (list of insight types it applies to). |
| `INSIGHT_TEMPLATES` | `dict[str, list[str]]` | Template strings for `subscription` and `spending_spike` insight types with `{merchant}`, `{amount}`, `{frequency}`, `{category}`, `{pct}`, `{date}` placeholders. |
| `INSIGHT_TYPES` | `list[str]` | `["spending_spike", "subscription", "trend_warning", "budget_risk", "no_action"]`. |

**Dependencies:** None (pure Python constants).

**Consumed by:** `preprocessor.py` (NOISE_TOKENS, MERCHANT_ALIASES), `seed_labeler.py` (all keyword/priority constants), `recurring_detector.py` (tolerance thresholds), `insight_generator.py` (TIP_CORPUS, INSIGHT_TEMPLATES), `training_data_generator.py` (CATEGORY_PRIORITY, INSIGHT_TYPES, TIP_CORPUS), `categorization_model.py` (fallback labels).

**Design Rationale:** All business rules live here. Adding a new merchant alias, keyword, or tip requires editing only this file. No processing logic is embedded here — it is purely declarative.

---

## 3. `preprocessor.py` (267 lines)

**Purpose:** Transforms raw, messy bank statement data into a clean, schema-validated, chronologically sorted pair of DataFrames (debits and credits).

**Key Exports:**

| Export | Type | Description |
|--------|------|-------------|
| `validate_schema(df)` | function | Checks for the 4 mandatory columns. Raises `ValueError` if any are missing. |
| `_normalize_flag(val)` | function | Converts any DR/CR variant (case/whitespace insensitive) to uppercase `"DR"` or `"CR"`. Returns `None` for invalid inputs. |
| `_compute_signed_amount(df)` | function | Normalizes flags and computes `signed_amount` (negative for DR, positive for CR). Invalid flags default to DR. |
| `clean_remark(text)` | function | 7-step text cleaning: (1) type check, (2) lowercase, (3) strip emails via regex, (4) strip 4+ digit numbers, (5) remove non-alphanumeric characters, (6) remove noise tokens, (7) collapse whitespace. |
| `_normalize_merchant(text)` | function | Applies MERCHANT_ALIASES regex patterns to canonicalize merchant names. |
| `_drop_zero_amount(df)` | function | Removes rows where `amount == 0`. |
| `_deduplicate(df)` | function | Drops exact duplicate rows based on `date`, `amount`, `amount_flag`, `remarks`. |
| `preprocess(df)` | function | **Main entry point.** Executes the full 8-step pipeline: validate → copy → parse dates → compute signed amounts → clean remarks → normalize merchants → drop zeros → deduplicate → sort chronologically → split into debits/credits. Returns `(debits_df, credits_df)`. |

**Dependencies:** `pandas`, `numpy`, `re`, `logging`, `config.py` (NOISE_TOKENS, MERCHANT_ALIASES), `schema.py` (Col).

**Consumed by:** `pipeline.py` (Phase 1), `tests/test_phase1.py`, `tests/test_e2e.py`.

**Critical Implementation Details:**
- The input DataFrame is `.copy()`ed immediately — no input mutation.
- Date parsing uses `pd.to_datetime` with `dayfirst=True` and `errors='coerce'` — NaT rows are dropped post-parse.
- `clean_remark` returns `""` for `None`, non-string, or whitespace-only inputs.
- The function strips PII: emails (regex `\S+@\S+`), long numeric sequences (4+ digits), and VPA-style identifiers.

---

## 4. `feature_engineer.py` (291 lines)

**Purpose:** Computes time-series, rolling statistical, and amount-based features for downstream ML models. Designed to be leak-free: rolling windows use `shift(1)` to exclude the current row, and NaN filling uses pre-computed global statistics.

**Key Exports:**

| Export | Type | Description |
|--------|------|-------------|
| `add_time_features(df)` | function | Adds: `is_weekend`, `week_of_month`, `month_sin`, `month_cos`, `dow_sin`, `dow_cos`. Requires `date` to be `datetime64`. Raises `TypeError` otherwise. |
| `add_rolling_features(df)` | function | Sorts by date, applies `.shift(1)`, then computes `rolling_7d_mean`, `rolling_30d_mean`, `rolling_7d_std` on the shifted amount column. The shift guarantees the current transaction never leaks into its own rolling window. |
| `fill_rolling_nulls(df, global_mean, global_std)` | function | Fills NaN values in rolling columns using the provided training-set statistics. If `global_std == 0`, it is forced to `1.0` to prevent downstream division by zero. |
| `add_amount_features(df)` | function | Computes `amount_log = log1p(abs(amount))` and `amount_zscore = clip((amount - rolling_7d_mean) / rolling_7d_std, -5, 5)`. The clip range `[-5, 5]` prevents extreme outliers from dominating. Uses epsilon `1e-9` to prevent division by zero when `rolling_7d_std` is near zero. |
| `engineer_features(df, global_mean, global_std)` | function | **Training entry point.** Runs all four sub-functions in sequence. Returns a copy with all features computed. |
| `engineer_features_inference(df, global_mean, global_std)` | function | **Inference entry point.** Identical to `engineer_features` but uses the pre-computed training-set `global_mean`/`global_std` for NaN filling, ensuring no data leakage during live prediction. |

**Dependencies:** `pandas`, `numpy`, `logging`, `schema.py` (Col).

**Consumed by:** `pipeline.py` (Phase 1), `tests/test_phase1.py`, `tests/test_e2e.py`.

**Critical Implementation Details:**
- Sin/cos encoding for `month` and `day_of_week` creates cyclical features (Jan is close to Dec in feature space).
- `week_of_month` is computed as `(day - 1) // 7 + 1`, capped at range [1, 5].
- The `.copy()` at entry prevents input mutation.

---

## 5. `seed_labeler.py` (229 lines)

**Purpose:** Assigns pseudo-labels to transactions using keyword matching against `config.py` dictionaries. These labels serve as training targets for the downstream ML categorization model.

**Key Exports:**

| Export | Type | Description |
|--------|------|-------------|
| `_match_remark(text, keywords, priority, fallback)` | function | Core matching logic. For each category in priority order, checks if any keyword (single or multi-word) appears in the cleaned remark text. Returns `(winning_category, dict_of_all_matches)`. Multi-word keywords require ALL tokens to be present. |
| `label_debits(df, label_col)` | function | Applies `_match_remark` against `CATEGORY_KEYWORDS` / `CATEGORY_PRIORITY`. Adds `pseudo_label` (or custom `label_col`) column. Emits a coverage warning if the fraction of non-fallback labels is below `MIN_COVERAGE_THRESHOLD`. |
| `label_credits(df, label_col)` | function | Same as `label_debits` but uses `CREDIT_KEYWORDS` / `CREDIT_PRIORITY` / `FALLBACK_CREDIT_LABEL`. |

**Dependencies:** `pandas`, `logging`, `config.py` (all keyword/priority/fallback constants, MIN_COVERAGE_THRESHOLD), `schema.py` (Col, require_columns).

**Consumed by:** `pipeline.py` (Phase 2), `tests/test_phase1.py`, `tests/test_e2e.py`.

**Critical Implementation Details:**
- Priority order is deterministic. If a remark matches both `"finance"` and `"health"`, the one appearing first in `CATEGORY_PRIORITY` wins.
- The function operates on `cleaned_remarks`, not raw remarks. All PII has already been stripped.
- The `.copy()` at entry prevents input mutation.

---

## 6. `categorization_model.py` (127 lines)

**Purpose:** Trains a Logistic Regression classifier on pseudo-labeled data to generalize categorization to unseen/ambiguous transaction remarks.

**Key Exports:**

| Export | Type | Description |
|--------|------|-------------|
| `build_categorization_pipeline()` | function | Constructs a `sklearn.Pipeline` with: TF-IDF vectorizer on `cleaned_remarks` (unigrams + bigrams, max 2000 features), StandardScaler (with_mean=False to preserve sparsity) on `amount_log`, and Logistic Regression (class_weight='balanced', max_iter=1000). |
| `train_categorization_model(df, label_col)` | function | Drops rows with NaN in features or target. **Explicitly excludes fallback labels** (`FALLBACK_DEBIT_LABEL`, `FALLBACK_CREDIT_LABEL`) from training to prevent the model from learning a "garbage" class. Logs training accuracy. Returns the fitted pipeline. |
| `predict_categories(pipeline, df)` | function | Predicts on a copy of the input. Fills missing `cleaned_remarks` with `""` and missing `amount_log` with `0.0` defensively. Adds `predicted_category` and `category_confidence` (max class probability) columns. |

**Dependencies:** `sklearn` (Pipeline, ColumnTransformer, TfidfVectorizer, StandardScaler, LogisticRegression), `pandas`, `config.py` (fallback labels), `schema.py` (Col, require_columns).

**Consumed by:** `pipeline.py` (Phase 3), `tests/test_phase2.py`, `tests/test_e2e.py`.

**Critical Implementation Details:**
- `sparse_threshold=1.0` in the ColumnTransformer forces output to remain sparse, preventing memory explosion on large datasets (the TF-IDF matrix is sparse).
- Fallback label exclusion means the model never trains on "uncategorized" / "other_credit" samples. This is intentional: the model should predict real categories for previously unmatched remarks.

---

## 7. `expected_spend_model.py` (129 lines)

**Purpose:** Trains a RidgeCV linear regressor to predict the "expected" spending amount for a transaction given its temporal context and category. The residual (actual - expected) forms the basis for anomaly detection.

**Key Exports:**

| Export | Type | Description |
|--------|------|-------------|
| `build_spend_pipeline()` | function | Constructs a `sklearn.Pipeline` with: StandardScaler on 9 numeric features (time + rolling), OneHotEncoder (handle_unknown='ignore') on `predicted_category`, and RidgeCV with alphas `[0.1, 1.0, 10.0, 100.0]`. |
| `train_expected_spend_model(df, target_col)` | function | Drops rows with NaN in target, predicted_category, or rolling features. Trains and logs train R² score. Returns fitted pipeline. |
| `predict_expected_spend(pipeline, df)` | function | Predicts `expected_amount`, then computes `residual = amount - expected_amount` and `percent_deviation = residual / (expected_amount + epsilon)`. Epsilon is `1e-5`. |

**Dependencies:** `sklearn` (Pipeline, ColumnTransformer, StandardScaler, OneHotEncoder, RidgeCV), `pandas`, `numpy`, `schema.py` (Col, require_columns).

**Consumed by:** `pipeline.py` (Phase 4), `tests/test_phase2.py`, `tests/test_e2e.py`.

**Critical Implementation Details:**
- RidgeCV was chosen over tree-based models specifically because it can **extrapolate** beyond the training range. A RandomForest would cap predictions at the maximum training value, causing all large future transactions to have zero residual. This is validated by `test_expected_spend_model_extrapolates()`.
- Defensive NaN filling for all 9 numeric features happens before prediction (fills with 0.0 for numerics, `"uncategorized"` for category).

---

## 8. `anomaly_detector.py` (49 lines)

**Purpose:** Flags transactions as anomalous using a dual-gate composite heuristic.

**Key Exports:**

| Export | Type | Description |
|--------|------|-------------|
| `detect_anomalies(df, zscore_threshold, pct_dev_threshold)` | function | A transaction is flagged as anomalous **only if BOTH** conditions are met: `abs(amount_zscore) > zscore_threshold` (default 3.0) AND `abs(percent_deviation) > pct_dev_threshold` (default 0.5). Adds `is_anomaly` boolean column. |

**Dependencies:** `pandas`, `logging`, `schema.py` (Col, require_columns).

**Consumed by:** `pipeline.py` (Phase 5), `tests/test_phase3.py`, `tests/test_e2e.py`.

**Critical Implementation Details:**
- The dual-gate design prevents false positives: a small ₹10 coffee that happens to have a high z-score (because the user rarely buys coffee) will not trigger an anomaly unless it also deviates significantly from the ML model's expected amount.
- Both gates use `.abs()`, so unusually **low** spending (e.g., a subscription dropping from ₹999 to ₹99) can also be flagged.

---

## 9. `recurring_detector.py` (99 lines)

**Purpose:** Identifies recurring transactions (subscriptions, standing orders, regular bills) using time-gap analysis and amount consistency heuristics.

**Key Exports:**

| Export | Type | Description |
|--------|------|-------------|
| `find_recurring_transactions(df, group_col, amount_tolerance)` | function | Groups by `cleaned_remarks`. For each group with ≥3 occurrences, checks: (1) Amount drift (max - min) / mean ≤ tolerance (default 5%). (2) Mean time gap falls within a frequency band. Adds: `is_recurring` (bool), `recurring_frequency` (weekly/biweekly/monthly/quarterly or None), `recurring_confidence` (1.0 or 0.5 if drift exceeds penalty threshold). |

**Frequency Bands:**

| Frequency | Mean Gap (days) | Max Variance |
|-----------|----------------|--------------|
| Weekly | 6–8 | < 3 |
| Biweekly | 13–16 | < 5 |
| Monthly | 27–33 | < 10 |
| Quarterly | 85–95 | < 20 |

**Dependencies:** `pandas`, `numpy`, `logging`, `config.py` (RECURRING_AMOUNT_TOLERANCE, RECURRING_FLUCTUATION_PENALTY_THRESHOLD), `schema.py` (Col, require_columns).

**Consumed by:** `pipeline.py` (Phase 5), `tests/test_phase3.py`, `tests/test_e2e.py`.

**Critical Implementation Details:**
- Minimum 3 occurrences required (not 2). This prevents spurious matches.
- The data is sorted chronologically before gap computation.
- Confidence is penalized from 1.0 to 0.5 when amount drift exceeds `RECURRING_FLUCTUATION_PENALTY_THRESHOLD` (10%), providing a signal to downstream rankers that the subscription might be variable (e.g., utility bills).

---

## 10. `insight_model.py` (104 lines)

**Purpose:** Loads a pre-trained LightGBM classifier from disk and uses it to score transactions by their likelihood of being a valuable insight.

**Key Exports:**

| Export | Type | Description |
|--------|------|-------------|
| `NUMERIC_FEATURES` | `list[str]` | 13 numeric feature columns consumed by the ranker. |
| `CATEGORICAL_FEATURES` | `list[str]` | `["predicted_category"]`. |
| `load_insight_ranker(model_path)` | function | Loads `models/insight_ranker.pkl` via `pickle.load`. Returns `None` if file not found (graceful degradation). |
| `predict_insight_scores(pipeline, df)` | function | If pipeline is None, sets all scores to 0.0 (falls back to rule-based ordering in insight_generator). Otherwise, calls `predict_proba`, finds the `no_action` class probability, and computes `insight_score = 1.0 - P(no_action)`. |

**Dependencies:** `sklearn.pipeline.Pipeline`, `pandas`, `pickle`, `os`, `logging`, `schema.py` (Col, require_columns).

**Consumed by:** `pipeline.py` (Phase 6), `tests/test_ml_integration.py`.

**Critical Implementation Details:**
- The scoring formula `1.0 - P(no_action)` means high scores indicate actionable insights. This is monotonically related to the probability of any actionable class.
- NaN values are filled defensively: numerics → 0.0, categoricals → `"unknown"`.
- `UserWarning` about feature names is explicitly suppressed to prevent log noise.
- Errors during prediction are caught and scores default to 0.0 (never crashes the pipeline).

---

## 11. `insight_generator.py` (168 lines)

**Purpose:** Translates the enriched, flagged DataFrame into human-readable English insight strings, ranked by the ML insight score, with diversity guarantees.

**Key Exports:**

| Export | Type | Description |
|--------|------|-------------|
| `_select_tip(category, insight_type)` | function | Selects a tip from `TIP_CORPUS`. Priority: (1) category-specific match, (2) generic match (empty categories list), (3) empty string. Uses `random.choice` for variety within matching candidates. |
| `generate_human_insights(df, top_n)` | function | **Main entry point.** Scans for recurring subscriptions and anomalies. For each, selects a template from `INSIGHT_TEMPLATES`, formats it with transaction data, and attaches a tip. Uses a diversity-aware ranking: Pass 1 grabs the highest-scoring insight of each TYPE first (guaranteeing at least 1 subscription + 1 anomaly if they exist), then Pass 2 fills the remaining `top_n` slots by pure score. Returns a flat list of strings. |

**Dependencies:** `pandas`, `random`, `logging`, `config.py` (TIP_CORPUS, INSIGHT_TEMPLATES), `schema.py` (Col, require_columns).

**Consumed by:** `pipeline.py` (Phase 6), `tests/test_phase3.py`, `tests/test_e2e.py`.

**Critical Implementation Details:**
- If `INSIGHT_SCORE` column is absent, it defaults to 0.0 (graceful degradation if the ranker model is not available).
- Subscription insights aggregate by `cleaned_remarks` group, reporting frequency and average amount.
- Anomaly insights include the date, amount, category, and percent deviation from baseline.
- The diversity ranking prevents a scenario where 10 anomalies dominate and the user never sees their ₹999/month subscription.

---

## 12. `pipeline.py` (214 lines)

**Purpose:** The central orchestrator that wires all modules into a 6-phase execution pipeline.

**Key Exports:**

| Export | Type | Description |
|--------|------|-------------|
| `PipelineResult` | `dataclass` | Container with fields: `debits` (DataFrame), `credits` (DataFrame), `insights` (list[str]). |
| `run_pipeline(df)` | function | **Batch/training mode.** Executes all 6 phases end-to-end. Trains ML models inline. Returns `PipelineResult`. |
| `run_inference(df, cat_pipeline, spend_pipeline, insight_pipeline)` | function | **Inference mode.** Uses pre-trained model pipelines passed as arguments. No training occurs. Calls `engineer_features_inference` instead of `engineer_features`. Returns `PipelineResult`. |

**Phase Breakdown (inside `run_pipeline`):**

| Phase | Module(s) Called | Output |
|-------|-----------------|--------|
| 1. Preprocess | `preprocessor.preprocess` | debits_df, credits_df |
| 2. Feature Engineering | `feature_engineer.engineer_features` | debits_df with 11 new columns |
| 3. Seed Labeling + ML Categorization | `seed_labeler.label_debits`, `categorization_model.train_categorization_model`, `categorization_model.predict_categories` | `pseudo_label`, `predicted_category`, `category_confidence` |
| 4. Expected Spend Regression | `expected_spend_model.train_expected_spend_model`, `expected_spend_model.predict_expected_spend` | `expected_amount`, `residual`, `percent_deviation` |
| 5. Signal Detection | `anomaly_detector.detect_anomalies`, `recurring_detector.find_recurring_transactions` | `is_anomaly`, `is_recurring`, `recurring_frequency`, `recurring_confidence` |
| 6. Insight Ranking + NLP | `insight_model.load_insight_ranker`, `insight_model.predict_insight_scores`, `insight_generator.generate_human_insights` | `insight_score`, list of insight strings |

**Dependencies:** All core modules, `pandas`, `logging`, `dataclasses`.

**Consumed by:** `demo.py`, `tutorial_real_data.py`, `tests/run_smoke.py`, `tests/run_stress_*.py`, `tests/test_e2e.py`.

---

## 13. `training_data_generator.py` (443 lines)

**Purpose:** Generates fully synthetic, labeled datasets for training the Insight Ranker and Tip Selector models. No real user data is used.

**Key Exports:**

| Export | Type | Description |
|--------|------|-------------|
| `ALL_CATEGORIES` | `list[str]` | `CATEGORY_PRIORITY + ["uncategorized"]`. |
| `ACTIONABLE_INSIGHTS` | `list[str]` | All insight types except `"no_action"`. |
| `_find_best_tip(category, insight_type)` | function | Deterministic tip selection (unlike `_select_tip` in insight_generator which uses `random.choice`). Returns a `tip_id` string. |
| `_generate_base_features(n, rng)` | function | Creates a DataFrame of `n` synthetic feature vectors using lognormal amounts, beta-distributed confidences, and weighted category sampling calibrated to Indian spending patterns. |
| `_apply_labels(df, rng)` | function | Assigns `insight_type` labels (~60% no_action, ~10% each for spending_spike, subscription, trend_warning, budget_risk) and then **adjusts feature values to be consistent** with the label. E.g., spending_spike rows get z-scores in [3.0, 5.0] and is_anomaly=1. |
| `_add_edge_cases(df, n_edge, rng)` | function | Adds 5 categories of deliberate boundary cases: borderline z-scores, high z-score but tiny amount, recurring-looking but too variable, weekend spending spikes, low-confidence categorization with anomaly features. |
| `generate_insight_dataset(n_samples, n_edge_cases, test_size, random_state)` | function | **Public API.** Generates features, applies labels, adds edge cases, shuffles, and performs stratified train/test split. Returns `(X_train, X_test, y_train, y_test)`. |

**Dependencies:** `numpy`, `pandas`, `sklearn.model_selection.train_test_split`, `config.py` (CATEGORY_PRIORITY, INSIGHT_TYPES, TIP_CORPUS).

**Consumed by:** `model_benchmark.py`, `train_and_save_models.py`, `tests/test_benchmark.py`.

---

## 14. `model_benchmark.py` (602 lines)

**Purpose:** Trains and evaluates 12 candidate ML models on synthetic data for two tasks: Insight Ranking (5-class) and Tip Selection (~36-class). Outputs formatted comparison tables, confusion matrices, and feature importances.

**Key Exports:**

| Export | Type | Description |
|--------|------|-------------|
| `NUMERIC_FEATURES` | `list[str]` | 13 numeric feature names. |
| `NUMERIC_FEATURES_NO_ANOMALY` | `list[str]` | Same list minus `is_anomaly` (for leakage testing). |
| `get_candidate_models()` | function | Returns 12 configured classifiers: LogisticRegression, GradientBoosting, RandomForest, LinearSVC (calibrated), KNeighbors, DecisionTree, MLP, AdaBoost, ExtraTrees, XGBoost, LightGBM, CatBoost. |
| `build_pipeline(classifier, include_anomaly, categorical_features)` | function | Wraps any classifier in a StandardScaler + OneHotEncoder + classifier Pipeline. |
| `evaluate_model(name, pipeline, X_train, y_train, X_test, y_test, cv_folds)` | function | Runs 5-fold stratified CV, fits on full training set, evaluates on holdout: accuracy, F1 (macro/weighted), precision, recall, train time, inference latency (ms/sample), model size (KB), overfit gap. |
| `run_benchmark(task_name, X_train, y_train, X_test, y_test, include_anomaly, categorical_features)` | function | Evaluates all 12 models, sorts by F1 macro. |
| `main()` | function | Runs 4 benchmark variants: insight ranking (with/without is_anomaly), tip selection, and prints a summary with leakage check. |

**Dependencies:** `sklearn`, `xgboost`, `lightgbm`, `catboost`, `numpy`, `pandas`, `pickle`, `time`, `training_data_generator.py`.

**Consumed by:** Direct execution (`python model_benchmark.py`).

**Critical Implementation Details:**
- Models requiring integer-encoded labels (XGBoost, CatBoost, MLP) are automatically handled via `LabelEncoder`.
- The leakage check compares F1 with and without `is_anomaly`: a delta >0.15 triggers a warning that the model may be over-relying on the pre-computed anomaly flag.

---

## 15. `train_and_save_models.py` (71 lines)

**Purpose:** Trains the production LightGBM Insight Ranker on synthetic data and serializes it to `models/insight_ranker.pkl`.

**Key Exports:**

| Export | Type | Description |
|--------|------|-------------|
| `train_and_save()` | function | Generates 5000 + 500 edge case samples, builds a StandardScaler + OneHotEncoder + LGBMClassifier pipeline, trains on `insight_type`, and writes the pickle file. |

**Dependencies:** `lightgbm`, `sklearn`, `pickle`, `os`, `training_data_generator.py`, `schema.py`.

**Consumed by:** Direct execution (`python train_and_save_models.py`).

---

## 16. `demo.py` (21 lines)

**Purpose:** Minimal 4-step usage example showing how to load a CSV, map columns to the schema, run the pipeline, and print insights.

---

## 17. `tutorial_real_data.py` (95 lines)

**Purpose:** Step-by-step tutorial for ingesting real bank statements. Demonstrates schema mapping, privacy guarantees (PII stripping happens automatically), and supports CSV/JSON/Parquet input. Includes a self-contained mock dataset for demonstration when run directly.

---

## 18. `README.md` (98 lines)

**Purpose:** Project documentation covering installation (venv + pip), testing (pytest), smoke testing (run_smoke.py), real data usage (tutorial_real_data.py), and a high-level 6-phase architecture overview.

---

## 19. `requirements.txt` (5 lines)

**Contents:** `pandas>=2.0.0`, `numpy>=1.24.0`, `scikit-learn>=1.3.0`, `lightgbm>=4.0.0`, `scipy>=1.14.0`.

---

## 20. `requirements-dev.txt` (20 lines)

**Contents:** Pinned versions of all production deps plus: `xgboost==3.2.0`, `catboost==1.2.10`, `pytest==9.0.2`, `psutil==7.2.2`, `joblib==1.5.3`, `pyarrow`, `fastparquet`, `openpyxl`.

---

## 21. `.gitignore` (27 lines)

**Excludes:** `.code-review-graph/`, `venv/`, `__pycache__/`, `*.pyc`, `.pytest_cache/`, `catboost_info/`, `models/`, `test-data/`, `.agents/`, `.claude/`, `.vscode/`, `.mcp.json`, `.opencode.json`, `.windsurfrules`, `AGENTS.md`, `CLAUDE.md`, `GEMINI.md`, `.cursorrules`.

---

## 22–31. Test Files

### `tests/test_phase1.py` (616 lines)
84 tests across 11 test classes covering: `TestValidateSchema` (4 tests), `TestNormalizeFlag` (3 parametrized tests × multiple inputs), `TestComputeSignedAmount` (6), `TestCleanRemark` (13), `TestDropZeroAmount` (3), `TestDeduplicate` (3), `TestPreprocess` (7), `TestTimeFeatures` (5), `TestRollingFeatures` (4), `TestFillRollingNulls` (3), `TestAmountFeatures` (5), `TestEngineerFeaturesFull` (2), `TestMatchRemark` (7), `TestLabelDebits` (7), `TestLabelCredits` (5).

### `tests/test_phase2.py` (94 lines)
3 tests: categorization model train+predict, spend model train+predict, RidgeCV extrapolation validation.

### `tests/test_phase3.py` (108 lines)
5 tests: monthly recurring detection, minimum 3-occurrence requirement, biweekly detection with variance penalty, composite anomaly threshold, insight generator string output.

### `tests/test_benchmark.py` (282 lines)
16 tests validating synthetic data shape, NaN/inf absence, distribution ratios, insight type coverage, category validity, feature-label consistency (spikes have high z-scores, subscriptions have is_recurring=1, no_action is benign), tip consistency, amount ranges, and a LogisticRegression smoke test.

### `tests/test_e2e.py` (145 lines)
1 large integration test that generates 90 days of synthetic bank data (weekly groceries, monthly Netflix, random shopping, one $2500 spike), runs the entire pipeline manually phase-by-phase, and asserts: (1) Netflix is detected as recurring, (2) the $2500 spike is flagged as anomalous, (3) insights are generated.

### `tests/test_ml_integration.py` (67 lines)
3 tests: `load_insight_ranker` handles missing file gracefully, `predict_insight_scores` works with a real or None pipeline, graceful fallback to 0.0 scores.

### `tests/run_smoke.py` (107 lines)
Loads `test-data/scrubbed.csv`, runs `run_pipeline`, prints: raw data summary, pseudo-label distribution, predicted category distribution, anomaly count, recurring transaction count, generated insights, and final column inventory.

### `tests/run_stress_legacy.py` (92 lines)
Generates 50,000 synthetic rows (lognormal amounts, 6 merchant types, 80/20 DR/CR split), benchmarks `run_pipeline` wall-clock time and memory delta. Warns if execution exceeds 20 seconds.

### `tests/run_stress_heavy.py` (66 lines)
Loads `test-data/scrubbed.csv`, duplicates it at 10x/50x/100x with time and amount noise, runs `run_pipeline` at each tier, reports throughput (rows/second).

### `tests/run_tests_legacy.py` (19 lines)
Pre-pytest manual test runner. Imports and calls specific test functions from `test_phase1.py` and `test_phase2.py` directly.
