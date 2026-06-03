*(Part 1 of 4 - split due to length)*

# Files by Role

## CONFIG

### `banned_content.py`

#### Dependencies
- **Direct internal**:
  - none
- **Transitive internal**:
  - none
- **Direct external**:
  - none
- **Runtime / injected deps**: none

#### Identity
- **Role**: CONFIG
- **Purpose**: Stores configuration values, thresholds, and static lists.

#### Exports
- `BANNED_PATTERN` (constant)
- `_CONFUSABLES` (constant)
- `_normalize_display_text` (function)
- `_COMPACT_SAFE_TERMS` (constant)
- `_SEP` (constant)
- `_obfuscated_word_pattern` (function)
- `_OBFUSCATED_SAFE_PATTERN` (constant)
- `contains_banned_content` (function)
- `__all__` (constant)

#### Side Effects
none

---

### `config.py`

#### Dependencies
- **Direct internal**:
  - none
- **Transitive internal**:
  - none
- **Direct external**:
  - none
- **Runtime / injected deps**: none

#### Identity
- **Role**: CONFIG
- **Purpose**: Stores configuration values, thresholds, and static lists.

#### Exports
- `LOG_LEVEL` (constant)
- `VALID_LOG_LEVELS` (constant)
- `ENABLE_CRASH_DUMPS` (constant)
- `CRASH_DUMP_DIR` (constant)
- `ENABLE_PII_DEBUG_LOGS` (constant)
- `HIGH_PRIORITY` (constant)
- `MEDIUM_PRIORITY` (constant)
- `LOW_PRIORITY` (constant)
- `CATEGORY_PRIORITY` (constant)
- `TIER_MAPPING` (constant)
- `CREDIT_PRIORITY` (constant)
- `FALLBACK_DEBIT_LABEL` (constant)
- `FALLBACK_CREDIT_LABEL` (constant)
- `MIN_COVERAGE_THRESHOLD` (constant)
- `RECURRING_CONFIG` (constant)
- `lookup_matching_tip_ids` (function)
- `KNOWN_PERSON_MATCH_THRESHOLD` (constant)
- `CONCAT_MIN_LENGTH` (constant)
- `CONCAT_PARTIAL_MIN_LENGTH` (constant)
- `MIN_SPEND_TRANSACTIONS_FOR_ML` (constant)

#### Side Effects
none

---

### `config_passion.py`

#### Dependencies
- **Direct internal**:
  - `config.py` → [SPECIFIC_MERCHANT_ALIASES, CATEGORY_KEYWORDS, CATEGORY_PRIORITY]
- **Transitive internal**:
  - none
- **Direct external**:
  - none
- **Runtime / injected deps**: none

#### Identity
- **Role**: CONFIG
- **Purpose**: Stores configuration values, thresholds, and static lists.

#### Exports
- `_CORE_ALIASES` (constant)
- `_PASSION_EXTRAS` (constant)
- `_NORMALIZED_EXTRAS` (constant)
- `_ALIAS_CONFLICTS` (constant)
- `validate_merchant_aliases` (function)
- `validate_config` (function)
- `__all__` (constant)

#### Side Effects
none

---

## CORE_LOGIC

### `anomaly_detector.py`

#### Dependencies
- **Direct internal**:
  - `schema.py` → [Col, require_columns]
- **Transitive internal**:
  - `config.py` (via `schema.py`) → [module/symbols used indirectly]
  - `logger_factory.py` (via `schema.py`) → [module/symbols used indirectly]
- **Direct external**:
  - `pandas` → [module]
- **Runtime / injected deps**: none

#### Identity
- **Role**: CORE_LOGIC
- **Purpose**: Composite statistical anomaly flagging.

#### Exports
- `detect_anomalies` (function)

#### Side Effects
file I/O

---

### `categorization_model.py`

#### Dependencies
- **Direct internal**:
  - `config.py` → [FALLBACK_DEBIT_LABEL, FALLBACK_CREDIT_LABEL]
  - `schema.py` → [Col, require_columns]
- **Transitive internal**:
  - `logger_factory.py` (via `schema.py`) → [module/symbols used indirectly]
- **Direct external**:
  - `pandas` → [module]
  - `sklearn` → [Pipeline, StandardScaler, LogisticRegression, ColumnTransformer, TfidfVectorizer]
- **Runtime / injected deps**: none

#### Identity
- **Role**: CORE_LOGIC
- **Purpose**: Contains core business logic, statistical methods, or ML inference.

#### Exports
- `build_categorization_pipeline` (function)
- `train_categorization_model` (function)
- `predict_categories` (function)

#### Side Effects
file I/O

---

### `expected_spend_model.py`

#### Dependencies
- **Direct internal**:
  - `schema.py` → [Col, require_columns]
- **Transitive internal**:
  - `config.py` (via `schema.py`) → [module/symbols used indirectly]
  - `logger_factory.py` (via `schema.py`) → [module/symbols used indirectly]
- **Direct external**:
  - `numpy` → [module]
  - `pandas` → [module]
  - `sklearn` → [RidgeCV, Pipeline, StandardScaler, OneHotEncoder, ColumnTransformer]
- **Runtime / injected deps**: none

#### Identity
- **Role**: CORE_LOGIC
- **Purpose**: Contains core business logic, statistical methods, or ML inference.

#### Exports
- `build_spend_pipeline` (function)
- `train_expected_spend_model` (function)
- `predict_expected_spend` (function)

#### Side Effects
file I/O

---

### `feature_engineer.py`

#### Dependencies
- **Direct internal**:
  - `schema.py` → [Col, require_columns]
- **Transitive internal**:
  - `config.py` (via `schema.py`) → [module/symbols used indirectly]
  - `logger_factory.py` (via `schema.py`) → [module/symbols used indirectly]
- **Direct external**:
  - `numpy` → [module]
  - `pandas` → [module]
- **Runtime / injected deps**: none

#### Identity
- **Role**: CORE_LOGIC
- **Purpose**: Generates machine learning features.

#### Exports
- `ZSCORE_CLIP` (constant)
- `add_time_features` (function)
- `add_rolling_features` (function)
- `fill_rolling_nulls` (function)
- `add_amount_features` (function)
- `engineer_features` (function)
- `engineer_features_inference` (function)

#### Side Effects
file I/O

---

### `insight_generator.py`

#### Dependencies
- **Direct internal**:
  - `contracts.py` → [lookup_matching_tip_ids, INSIGHT_TEMPLATES, TIP_CORPUS]
  - `schema.py` → [Col, require_columns]
- **Transitive internal**:
  - `config.py` (via `contracts.py`) → [module/symbols used indirectly]
  - `logger_factory.py` (via `schema.py`) → [module/symbols used indirectly]
- **Direct external**:
  - `pandas` → [module]
- **Runtime / injected deps**: none

#### Identity
- **Role**: CORE_LOGIC
- **Purpose**: Generates insights from categorized transactions.

#### Exports
- `_select_tip` (function)
- `generate_human_insights` (function)

#### Side Effects
network calls, file I/O

---

### `insight_model.py`

#### Dependencies
- **Direct internal**:
  - `schema.py` → [Col, require_columns]
- **Transitive internal**:
  - `config.py` (via `schema.py`) → [module/symbols used indirectly]
  - `logger_factory.py` (via `schema.py`) → [module/symbols used indirectly]
- **Direct external**:
  - `pandas` → [module]
  - `sklearn` → [Pipeline]
- **Runtime / injected deps**: none

#### Identity
- **Role**: CORE_LOGIC
- **Purpose**: Generates insights from categorized transactions.

#### Exports
- `NUMERIC_FEATURES` (constant)
- `CATEGORICAL_FEATURES` (constant)
- `_MODELS_DIR` (constant)
- `ModelSecurityError` (class)
- `_compute_checksum` (function)
- `_verify_checksum` (function)
- `_validate_model_path` (function)
- `load_insight_ranker` (function)
- `predict_insight_scores` (function)

#### Side Effects
file I/O (open/print), file I/O

---

### `model_benchmark.py`

#### Dependencies
- **Direct internal**:
  - `training_data_generator.py` → [generate_insight_dataset]
- **Transitive internal**:
  - `config.py` (via `training_data_generator.py`) → [module/symbols used indirectly]
  - `contracts.py` (via `training_data_generator.py`) → [module/symbols used indirectly]
- **Direct external**:
  - `catboost` → [CatBoostClassifier]
  - `lightgbm` → [LGBMClassifier]
  - `numpy` → [module]
  - `pandas` → [module]
  - `sklearn` → [recall_score, StandardScaler, classification_report, precision_score, accuracy_score, AdaBoostClassifier, OneHotEncoder, LogisticRegression, ColumnTransformer, f1_score, KNeighborsClassifier, DecisionTreeClassifier, CalibratedClassifierCV, RandomForestClassifier, ExtraTreesClassifier, StratifiedKFold, MLPClassifier, LabelEncoder, LinearSVC, GradientBoostingClassifier, Pipeline, cross_validate, confusion_matrix]
  - `xgboost` → [XGBClassifier]
- **Runtime / injected deps**: none

#### Identity
- **Role**: CORE_LOGIC
- **Purpose**: Contains core business logic, statistical methods, or ML inference.

#### Exports
- `get_candidate_models` (function)
- `build_pipeline` (function)
- `evaluate_model` (function)
- `run_benchmark` (function)
- `format_results_table` (function)
- `print_confusion_matrix` (function)
- `print_classification_report_top` (function)
- `generate_feature_importance` (function)
- `main` (function)

#### Side Effects
file I/O (open/print), network calls, file I/O

---

### `passion_detector.py`

#### Dependencies
- **Direct internal**:
  - `config_passion.py` → [PASSION_ANOMALY_SUPPRESSION_THRESHOLD, DISTRESS_FEES_THRESHOLD, PASSION_SPEND_SHARE_THRESHOLD, PASSION_MIN_MONTHS, PASSION_MERCHANT_COUNT_MIN]
  - `logger_factory.py` → [get_logger]
  - `marketplace_subcategory.py` → [resolve_merchant_vectorized]
  - `passion_models.py` → [PassionSignal]
  - `passion_utils.py` → [assert_columns_exist, sanitize_mask, safe_numeric, _safe_isna, coerce_bool_column]
  - `schema.py` → [Col]
- **Transitive internal**:
  - `config.py` (via `schema.py`) → [module/symbols used indirectly]
- **Direct external**:
  - `numpy` → [module]
  - `pandas` → [module]
- **Runtime / injected deps**: none

#### Identity
- **Role**: CORE_LOGIC
- **Purpose**: Detects passion or specialized categories.

#### Exports
- `__all__` (constant)
- `_FEE_KEYWORDS` (constant)
- `_FEE_PATTERN` (constant)
- `_safe_coerce_anomaly` (function)
- `_check_distress_gate` (function)
- `_check_anomaly_suppression` (function)
- `_parse_dates_safe` (function)
- `_is_non_declining` (function)
- `detect_passions` (function)

#### Side Effects
network calls, file I/O

---

### `passion_insight_generator.py`

#### Dependencies
- **Direct internal**:
  - `banned_content.py` → [contains_banned_content]
  - `config_passion.py` → [PASSION_INSIGHT_TEMPLATES]
  - `contracts.py` → [lookup_matching_tip_ids, INSIGHT_TEMPLATES, TIP_CORPUS]
  - `logger_factory.py` → [get_logger]
  - `passion_models.py` → [PassionSignal]
  - `passion_utils.py` → [validate_template_values]
  - `schema.py` → [Col]
- **Transitive internal**:
  - `config.py` (via `schema.py`) → [module/symbols used indirectly]
- **Direct external**:
  - none
- **Runtime / injected deps**: none

#### Identity
- **Role**: CORE_LOGIC
- **Purpose**: Generates insights from categorized transactions.

#### Exports
- `__all__` (constant)
- `_select_tip` (function)
- `_render_candidate` (function)
- `generate_passion_insights` (function)

#### Side Effects
network calls

---

### `passion_models.py`

#### Dependencies
- **Direct internal**:
  - none
- **Transitive internal**:
  - none
- **Direct external**:
  - `numpy` → [module]
- **Runtime / injected deps**: none

#### Identity
- **Role**: CORE_LOGIC
- **Purpose**: Contains core business logic, statistical methods, or ML inference.

#### Exports
- `__all__` (constant)
- `_EPS` (constant)
- `PassionSignal` (class)

#### Side Effects
none

---

### `preprocessor.py`

#### Dependencies
- **Direct internal**:
  - `config.py` → [SPECIFIC_MERCHANT_ALIASES, NOISE_TOKENS, GENERIC_ROUTER_ALIASES]
  - `logger_factory.py` → [get_logger]
  - `schema.py` → [Col, coerce_and_validate_types, require_columns]
- **Transitive internal**:
  - none
- **Direct external**:
  - `numpy` → [module]
  - `pandas` → [module]
- **Runtime / injected deps**: none

#### Identity
- **Role**: CORE_LOGIC
- **Purpose**: Contains core business logic, statistical methods, or ML inference.

#### Exports
- `_LONG_DIGIT_PATTERN` (constant)
- `_EMAIL_PATTERN` (constant)
- `_SPECIAL_CHAR_PATTERN` (constant)
- `_MULTI_SPACE_PATTERN` (constant)
- `validate_schema` (function)
- `_parse_and_sort_dates` (function)
- `_normalize_flag` (function)
- `_compute_signed_amount` (function)
- `normalize` (function)
- `clean_remark` (function)
- `_drop_zero_amount` (function)
- `_deduplicate` (function)
- `_split_debit_credit` (function)
- `preprocess` (function)

#### Side Effects
file I/O

---

### `recurring_detector.py`

#### Dependencies
- **Direct internal**:
  - `config.py` → [module, RECURRING_CONFIG]
  - `log_utils.py` → [log_safe_merchant]
  - `logger_factory.py` → [get_logger]
  - `schema.py` → [Col, require_columns]
- **Transitive internal**:
  - none
- **Direct external**:
  - `numpy` → [module]
  - `pandas` → [module]
- **Runtime / injected deps**: none

#### Identity
- **Role**: CORE_LOGIC
- **Purpose**: Rule-based recurring transaction identifier.

#### Exports
- `find_recurring_transactions` (function)

#### Side Effects
file I/O

---

### `training_data_generator.py`

#### Dependencies
- **Direct internal**:
  - `config.py` → [lookup_matching_tip_ids, INSIGHT_TYPES, CATEGORY_PRIORITY]
  - `contracts.py` → [TIP_CORPUS]
- **Transitive internal**:
  - none
- **Direct external**:
  - `numpy` → [module]
  - `pandas` → [module]
  - `sklearn` → [train_test_split]
- **Runtime / injected deps**: none

#### Identity
- **Role**: CORE_LOGIC
- **Purpose**: Contains core business logic, statistical methods, or ML inference.

#### Exports
- `_find_best_tip` (function)
- `_generate_base_features` (function)
- `_apply_labels` (function)
- `_add_edge_cases` (function)
- `generate_insight_dataset` (function)

#### Side Effects
network calls, file I/O

---

## DATA_LAYER

### `known_persons.py`

#### Dependencies
- **Direct internal**:
  - `config.py` → [module]
  - `logger_factory.py` → [get_logger]
  - `schema.py` → [Col]
- **Transitive internal**:
  - none
- **Direct external**:
  - `pandas` → [module]
- **Runtime / injected deps**: none

#### Identity
- **Role**: DATA_LAYER
- **Purpose**: Manages data retrieval, labeling, or enrichment.

#### Exports
- `_NAME_NOISE_TOKENS` (constant)
- `_MERCHANT_INDICATOR_TOKENS` (constant)
- `_MERCHANT_SUFFIXES` (constant)
- `_TRANSFER_CONTEXT_TOKENS` (constant)
- `_SEPARATOR_PATTERN` (constant)
- `SignalBundle` (class)
- `_extract_signals` (function)
- `_is_contiguous_subsequence` (function)
- `_find_concat_partial_match` (function)
- `_find_account_fragment_match` (function)
- `_compile_matchers` (function)
- `_score_remark` (function)
- `tag_known_persons` (function)
- `_enforce_known_person_schema` (function)
- `_suggestion_key` (function)
- `log_unmatched_recurring_transfers` (function)
- `_analyze_person_group` (function)
- `detect_personal_patterns` (function)

#### Side Effects
network calls, file I/O

---

### `marketplace_subcategory.py`

#### Dependencies
- **Direct internal**:
  - `config_passion.py` → [MARKETPLACE_HIGH_CONFIDENCE, PASSION_MERCHANT_ALIASES, ELECTRONICS_ALLOWED_CATEGORIES, MARKETPLACE_HIGH_AMOUNT_THRESHOLD, MARKETPLACE_LOW_CONFIDENCE, GENERALIST_CANONICALS]
  - `logger_factory.py` → [get_logger]
  - `passion_utils.py` → [safe_numeric, coerce_bool_column, assert_columns_exist]
  - `schema.py` → [Col]
- **Transitive internal**:
  - `config.py` (via `schema.py`) → [module/symbols used indirectly]
- **Direct external**:
  - `numpy` → [module]
  - `pandas` → [module]
- **Runtime / injected deps**: none

#### Identity
- **Role**: DATA_LAYER
- **Purpose**: Manages data retrieval, labeling, or enrichment.

#### Exports
- `__all__` (constant)
- `_ALIAS_PATTERN` (constant)
- `resolve_merchant_vectorized` (function)
- `enrich_subcategories` (function)

#### Side Effects
network calls, file I/O

---

### `seed_labeler.py`

#### Dependencies
- **Direct internal**:
  - `config.py` → [TIER_MAPPING, CREDIT_PRIORITY, CREDIT_KEYWORDS, CATEGORY_KEYWORDS, CATEGORY_PRIORITY, FALLBACK_DEBIT_LABEL, MIN_COVERAGE_THRESHOLD, FALLBACK_CREDIT_LABEL]
  - `preprocessor.py` → [normalize]
  - `schema.py` → [Col, require_columns]
- **Transitive internal**:
  - `logger_factory.py` (via `preprocessor.py`) → [module/symbols used indirectly]
- **Direct external**:
  - `pandas` → [module]
- **Runtime / injected deps**: none

#### Identity
- **Role**: DATA_LAYER
- **Purpose**: Manages data retrieval, labeling, or enrichment.

#### Exports
- `CompiledKeyword` (class)
- `_compile_keywords` (function)
- `_match_remark` (function)
- `_DEFAULT_DEBIT_KWS` (constant)
- `_DEFAULT_CREDIT_KWS` (constant)
- `_log_coverage` (function)
- `label_debits` (function)
- `label_credits` (function)

#### Side Effects
file I/O

---


*(Part 2 of 4 - split due to length)*


### `tests/test_known_persons.py`

#### Dependencies
- **Direct internal**:
  - `known_persons.py` → [_suggestion_key, _enforce_known_person_schema, tag_known_persons, _extract_signals]
  - `model_state.py` → [InsightModelState]
  - `pipeline.py` → [_compute_config_hash, run_pipeline, PipelineResult, run_inference]
  - `schema.py` → [Col]
- **Transitive internal**:
  - `anomaly_detector.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `banned_content.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `bootstrap.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `candidate.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `categorization_model.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `config.py` (via `known_persons.py`) → [module/symbols used indirectly]
  - `config_passion.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `contracts.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `expected_spend_model.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `feature_engineer.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `insight_generator.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `insight_model.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `log_utils.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `logger_factory.py` (via `known_persons.py`) → [module/symbols used indirectly]
  - `marketplace_subcategory.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `passion_detector.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `passion_insight_generator.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `passion_models.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `passion_pipeline.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `passion_utils.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `pipeline_result.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `preprocessor.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `recurring_detector.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `seed_labeler.py` (via `pipeline.py`) → [module/symbols used indirectly]
- **Direct external**:
  - `numpy` → [module]
  - `pandas` → [module]
  - `pytest` → [module]
- **Runtime / injected deps**: none

#### Identity
- **Role**: DATA_LAYER
- **Purpose**: Manages data retrieval, labeling, or enrichment.

#### Exports
- `test_known_persons_cfg` (function)
- `test_self_accounts_cfg` (function)
- `_create_minimal_test_df` (function)
- `test_extract_signals_upi_digits` (function)
- `test_tagging_exact_upi` (function)
- `test_tagging_self_account_fragment` (function)
- `test_merchant_suppression` (function)
- `test_concat_partial_with_bounds` (function)
- `test_enforce_schema_schema` (function)
- `test_suggestion_key_min_len` (function)
- `test_state_version_uses_di_params_not_globals` (function)

#### Side Effects
file I/O

---

## ENTRYPOINT

### `bootstrap.py`

#### Dependencies
- **Direct internal**:
  - `config_passion.py` → [validate_config, PASSION_INSIGHT_TEMPLATES, validate_merchant_aliases]
  - `contracts.py` → [INSIGHT_TEMPLATES, TIP_CORPUS]
  - `log_utils.py` → [_get_secret]
  - `logger_factory.py` → [get_logger]
  - `schema.py` → [Col]
- **Transitive internal**:
  - `config.py` (via `contracts.py`) → [module/symbols used indirectly]
- **Direct external**:
  - none
- **Runtime / injected deps**: none

#### Identity
- **Role**: ENTRYPOINT
- **Purpose**: Initializes and bootstraps application environments or models.

#### Exports
- `__all__` (constant)
- `ALLOWED_FORMAT_SPECS` (constant)
- `validate_template_fields` (function)
- `_validate_python_version` (function)
- `_validate_schema_columns` (function)
- `_validate_insight_templates` (function)
- `_validate_passion_templates` (function)
- `_validate_tip_corpus` (function)
- `_validate_secret` (function)
- `_dry_render_templates` (function)
- `run_startup_checks` (function)

#### Side Effects
network calls

---

### `demo.py`

#### Dependencies
- **Direct internal**:
  - `pipeline.py` → [run_pipeline]
  - `schema.py` → [Col]
  - `summary_utils.py` → [print_summary]
- **Transitive internal**:
  - `anomaly_detector.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `banned_content.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `bootstrap.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `candidate.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `categorization_model.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `config.py` (via `schema.py`) → [module/symbols used indirectly]
  - `config_passion.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `contracts.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `expected_spend_model.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `feature_engineer.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `insight_generator.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `insight_model.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `known_persons.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `log_utils.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `logger_factory.py` (via `schema.py`) → [module/symbols used indirectly]
  - `marketplace_subcategory.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `model_state.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `passion_detector.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `passion_insight_generator.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `passion_models.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `passion_pipeline.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `passion_utils.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `pipeline_result.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `preprocessor.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `recurring_detector.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `seed_labeler.py` (via `pipeline.py`) → [module/symbols used indirectly]
- **Direct external**:
  - `pandas` → [module]
- **Runtime / injected deps**: none

#### Identity
- **Role**: ENTRYPOINT
- **Purpose**: Runs a demo of the pipeline.

#### Exports
- none


#### Side Effects
file I/O (open/print), network calls, file I/O

---

### `passion_pipeline.py`

#### Dependencies
- **Direct internal**:
  - `banned_content.py` → [contains_banned_content]
  - `bootstrap.py` → [run_startup_checks]
  - `candidate.py` → [Candidate]
  - `config_passion.py` → [PIPELINE_BUDGET_MS, PASSION_MIN_DEBIT_ROWS, PIPELINE_TOP_N, MAX_SPIKE_CANDIDATES, PIPELINE_HARD_TIMEOUT_MS]
  - `logger_factory.py` → [get_logger]
  - `marketplace_subcategory.py` → [enrich_subcategories, resolve_merchant_vectorized]
  - `passion_detector.py` → [detect_passions]
  - `passion_insight_generator.py` → [generate_passion_insights]
  - `passion_utils.py` → [_safe_isna, coerce_bool_column, assert_columns_exist, safe_numeric]
  - `pipeline_result.py` → [PassionResult]
  - `schema.py` → [Col]
- **Transitive internal**:
  - `config.py` (via `schema.py`) → [module/symbols used indirectly]
  - `contracts.py` (via `bootstrap.py`) → [module/symbols used indirectly]
  - `log_utils.py` (via `bootstrap.py`) → [module/symbols used indirectly]
  - `passion_models.py` (via `candidate.py`) → [module/symbols used indirectly]
- **Direct external**:
  - `numpy` → [module]
  - `pandas` → [module]
- **Runtime / injected deps**: none

#### Identity
- **Role**: ENTRYPOINT
- **Purpose**: Orchestrator for the passion insights feature.

#### Exports
- `__all__` (constant)
- `_PERMANENT_STARTUP_ERRORS` (constant)
- `_neutral_passion_result` (function)
- `_ensure_initialized` (function)
- `_FATAL_EXCEPTIONS` (constant)
- `_should_reraise` (function)
- `PASSION_OWNED_OUTPUT_COLUMNS` (constant)
- `safe_assign_new_columns` (function)
- `_looks_like_compact_yyyymmdd` (function)
- `_normalize_ts` (function)
- `_CURRENCY_RE_LOCAL` (constant)
- `_INR_RE_LOCAL` (constant)
- `_is_unparseable_amount` (function)
- `_StepBudgetGuard` (class)
- `process_pipeline` (function)

#### Side Effects
network calls, file I/O

---

### `pipeline.py`

#### Dependencies
- **Direct internal**:
  - `anomaly_detector.py` → [detect_anomalies]
  - `categorization_model.py` → [predict_categories, train_categorization_model]
  - `config.py` → [module]
  - `expected_spend_model.py` → [predict_expected_spend, train_expected_spend_model]
  - `feature_engineer.py` → [engineer_features_inference, engineer_features]
  - `insight_generator.py` → [generate_human_insights]
  - `insight_model.py` → [predict_insight_scores, load_insight_ranker]
  - `known_persons.py` → [log_unmatched_recurring_transfers, detect_personal_patterns, _enforce_known_person_schema, tag_known_persons]
  - `log_utils.py` → [log_safe_text, log_safe_merchant]
  - `logger_factory.py` → [generate_new_run_id, get_logger, pipeline_run_id_ctx]
  - `model_state.py` → [InsightModelState]
  - `passion_pipeline.py` → [process_pipeline]
  - `preprocessor.py` → [preprocess]
  - `recurring_detector.py` → [find_recurring_transactions]
  - `schema.py` → [Col]
  - `seed_labeler.py` → [label_credits, label_debits]
- **Transitive internal**:
  - `banned_content.py` (via `passion_pipeline.py`) → [module/symbols used indirectly]
  - `bootstrap.py` (via `passion_pipeline.py`) → [module/symbols used indirectly]
  - `candidate.py` (via `passion_pipeline.py`) → [module/symbols used indirectly]
  - `config_passion.py` (via `passion_pipeline.py`) → [module/symbols used indirectly]
  - `contracts.py` (via `insight_generator.py`) → [module/symbols used indirectly]
  - `marketplace_subcategory.py` (via `passion_pipeline.py`) → [module/symbols used indirectly]
  - `passion_detector.py` (via `passion_pipeline.py`) → [module/symbols used indirectly]
  - `passion_insight_generator.py` (via `passion_pipeline.py`) → [module/symbols used indirectly]
  - `passion_models.py` (via `passion_pipeline.py`) → [module/symbols used indirectly]
  - `passion_utils.py` (via `passion_pipeline.py`) → [module/symbols used indirectly]
  - `pipeline_result.py` (via `passion_pipeline.py`) → [module/symbols used indirectly]
- **Direct external**:
  - `numpy` → [module]
  - `pandas` → [module]
  - `sklearn` → [Pipeline]
- **Runtime / injected deps**: none

#### Identity
- **Role**: ENTRYPOINT
- **Purpose**: Central Orchestrator, defines complete Insight Engine pipeline.

#### Exports
- `_compute_config_hash` (function)
- `_validate_state_version` (function)
- `PipelineResult` (class)
- `finalize_df` (function)
- `_optimize_memory_footprint` (function)
- `train_models` (function)
- `_pre_initialize_ml_columns` (function)
- `_write_crash_dumps` (function)
- `_resolve_passion_crash_fields` (function)
- `_attach_passion_results` (function)
- `run_pipeline` (function)
- `run_inference` (function)

#### Side Effects
file I/O (open/print), network calls, file I/O

---

### `train_and_save_models.py`

#### Dependencies
- **Direct internal**:
  - `insight_model.py` → [_compute_checksum]
  - `schema.py` → [Col]
  - `training_data_generator.py` → [generate_insight_dataset]
- **Transitive internal**:
  - `config.py` (via `training_data_generator.py`) → [module/symbols used indirectly]
  - `contracts.py` (via `training_data_generator.py`) → [module/symbols used indirectly]
  - `logger_factory.py` (via `insight_model.py`) → [module/symbols used indirectly]
- **Direct external**:
  - `lightgbm` → [LGBMClassifier]
  - `sklearn` → [ColumnTransformer, Pipeline, StandardScaler, OneHotEncoder]
- **Runtime / injected deps**: none

#### Identity
- **Role**: ENTRYPOINT
- **Purpose**: Script to train and save machine learning models.

#### Exports
- `NUMERIC_FEATURES` (constant)
- `CATEGORICAL_FEATURES` (constant)
- `train_and_save` (function)

#### Side Effects
file I/O (open/print), file I/O

---

### `tutorial_real_data.py`

#### Dependencies
- **Direct internal**:
  - `logger_factory.py` → [get_logger]
  - `pipeline.py` → [run_pipeline]
  - `schema.py` → [Col]
- **Transitive internal**:
  - `anomaly_detector.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `banned_content.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `bootstrap.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `candidate.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `categorization_model.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `config.py` (via `logger_factory.py`) → [module/symbols used indirectly]
  - `config_passion.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `contracts.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `expected_spend_model.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `feature_engineer.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `insight_generator.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `insight_model.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `known_persons.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `log_utils.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `marketplace_subcategory.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `model_state.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `passion_detector.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `passion_insight_generator.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `passion_models.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `passion_pipeline.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `passion_utils.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `pipeline_result.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `preprocessor.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `recurring_detector.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `seed_labeler.py` (via `pipeline.py`) → [module/symbols used indirectly]
- **Direct external**:
  - `pandas` → [module]
- **Runtime / injected deps**: none

#### Identity
- **Role**: ENTRYPOINT
- **Purpose**: Demonstrates ingesting real bank statements.

#### Exports
- `run_real_data_tutorial` (function)

#### Side Effects
file I/O (open/print), file I/O

---

## SCHEMA

### `candidate.py`

#### Dependencies
- **Direct internal**:
  - `passion_models.py` → [PassionSignal]
- **Transitive internal**:
  - none
- **Direct external**:
  - `numpy` → [module]
- **Runtime / injected deps**: none

#### Identity
- **Role**: SCHEMA
- **Purpose**: Defines data schemas, types, classes, or domain models.

#### Exports
- `__all__` (constant)
- `_coerce_finite_float` (function)
- `Candidate` (class)

#### Side Effects
network calls

---

### `contracts.py`

#### Dependencies
- **Direct internal**:
  - `config.py` → [INSIGHT_TEMPLATES, TIP_CORPUS]
- **Transitive internal**:
  - none
- **Direct external**:
  - none
- **Runtime / injected deps**: none

#### Identity
- **Role**: SCHEMA
- **Purpose**: Defines data schemas, types, classes, or domain models.

#### Exports
- `_GENERIC_TIP_PREFIX` (constant)
- `_is_generic_tip_id` (function)
- `_freeze_insight_templates` (function)
- `_freeze_tip_corpus` (function)
- `INSIGHT_TEMPLATES` (constant)
- `TIP_CORPUS` (constant)
- `lookup_matching_tip_ids` (function)
- `__all__` (constant)

#### Side Effects
network calls

---

### `model_state.py`

#### Dependencies
- **Direct internal**:
  - none
- **Transitive internal**:
  - none
- **Direct external**:
  - `joblib` → [module]
  - `numpy` → [module]
  - `sklearn` → [Pipeline]
- **Runtime / injected deps**: none

#### Identity
- **Role**: SCHEMA
- **Purpose**: Defines data schemas, types, classes, or domain models.

#### Exports
- `InsightModelState` (class)
- `save_model_state` (function)
- `load_model_state` (function)

#### Side Effects
network calls

---

### `pipeline_result.py`

#### Dependencies
- **Direct internal**:
  - `candidate.py` → [Candidate]
  - `passion_models.py` → [PassionSignal]
- **Transitive internal**:
  - none
- **Direct external**:
  - `pandas` → [module]
- **Runtime / injected deps**: none

#### Identity
- **Role**: SCHEMA
- **Purpose**: Defines data schemas, types, classes, or domain models.

#### Exports
- `__all__` (constant)
- `PassionResult` (class)

#### Side Effects
file I/O

---

### `schema.py`

#### Dependencies
- **Direct internal**:
  - `logger_factory.py` → [get_logger]
- **Transitive internal**:
  - `config.py` (via `logger_factory.py`) → [module/symbols used indirectly]
- **Direct external**:
  - `pandas` → [module]
- **Runtime / injected deps**: none

#### Identity
- **Role**: SCHEMA
- **Purpose**: Defines data schemas, types, classes, or domain models.

#### Exports
- `Col` (class)
- `require_columns` (function)
- `coerce_and_validate_types` (function)

#### Side Effects
file I/O

---

## TEST

### `tests/conftest.py`

#### Dependencies
- **Direct internal**:
  - `log_utils.py` → [_reset_secret_cache]
  - `passion_pipeline.py` → [module]
- **Transitive internal**:
  - `banned_content.py` (via `passion_pipeline.py`) → [module/symbols used indirectly]
  - `bootstrap.py` (via `passion_pipeline.py`) → [module/symbols used indirectly]
  - `candidate.py` (via `passion_pipeline.py`) → [module/symbols used indirectly]
  - `config.py` (via `passion_pipeline.py`) → [module/symbols used indirectly]
  - `config_passion.py` (via `passion_pipeline.py`) → [module/symbols used indirectly]
  - `contracts.py` (via `passion_pipeline.py`) → [module/symbols used indirectly]
  - `logger_factory.py` (via `passion_pipeline.py`) → [module/symbols used indirectly]
  - `marketplace_subcategory.py` (via `passion_pipeline.py`) → [module/symbols used indirectly]
  - `passion_detector.py` (via `passion_pipeline.py`) → [module/symbols used indirectly]
  - `passion_insight_generator.py` (via `passion_pipeline.py`) → [module/symbols used indirectly]
  - `passion_models.py` (via `passion_pipeline.py`) → [module/symbols used indirectly]
  - `passion_utils.py` (via `passion_pipeline.py`) → [module/symbols used indirectly]
  - `pipeline_result.py` (via `passion_pipeline.py`) → [module/symbols used indirectly]
  - `schema.py` (via `passion_pipeline.py`) → [module/symbols used indirectly]
- **Direct external**:
  - `pytest` → [module]
- **Runtime / injected deps**: none

#### Identity
- **Role**: TEST
- **Purpose**: Test suite or testing script.

#### Exports
- `_set_test_env` (function)
- `_reset_pipeline_initialized` (function)
- `_reset_dev_secret` (function)
- `real_startup_env` (function)

#### Side Effects
network calls

---

### `tests/run_smoke.py`

#### Dependencies
- **Direct internal**:
  - `pipeline.py` → [run_pipeline, PipelineResult]
- **Transitive internal**:
  - `anomaly_detector.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `banned_content.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `bootstrap.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `candidate.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `categorization_model.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `config.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `config_passion.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `contracts.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `expected_spend_model.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `feature_engineer.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `insight_generator.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `insight_model.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `known_persons.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `log_utils.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `logger_factory.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `marketplace_subcategory.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `model_state.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `passion_detector.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `passion_insight_generator.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `passion_models.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `passion_pipeline.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `passion_utils.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `pipeline_result.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `preprocessor.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `recurring_detector.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `schema.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `seed_labeler.py` (via `pipeline.py`) → [module/symbols used indirectly]
- **Direct external**:
  - `pandas` → [module]
- **Runtime / injected deps**: none

#### Identity
- **Role**: TEST
- **Purpose**: Test suite or testing script.

#### Exports
- `CSV_PATH` (constant)

#### Side Effects
file I/O (open/print), file I/O

---

### `tests/run_stress_heavy.py`

#### Dependencies
- **Direct internal**:
  - `pipeline.py` → [run_pipeline]
- **Transitive internal**:
  - `anomaly_detector.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `banned_content.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `bootstrap.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `candidate.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `categorization_model.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `config.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `config_passion.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `contracts.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `expected_spend_model.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `feature_engineer.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `insight_generator.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `insight_model.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `known_persons.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `log_utils.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `logger_factory.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `marketplace_subcategory.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `model_state.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `passion_detector.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `passion_insight_generator.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `passion_models.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `passion_pipeline.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `passion_utils.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `pipeline_result.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `preprocessor.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `recurring_detector.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `schema.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `seed_labeler.py` (via `pipeline.py`) → [module/symbols used indirectly]
- **Direct external**:
  - `numpy` → [module]
  - `pandas` → [module]
- **Runtime / injected deps**: none

#### Identity
- **Role**: TEST
- **Purpose**: Test suite or testing script.

#### Exports
- `generate_stress_data` (function)

#### Side Effects
file I/O (open/print), file I/O

---
*(Part 3 of 4 - split due to length)*


### `tests/run_stress_legacy.py`

#### Dependencies
- **Direct internal**:
  - `pipeline.py` → [run_pipeline]
  - `schema.py` → [Col]
- **Transitive internal**:
  - `anomaly_detector.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `banned_content.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `bootstrap.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `candidate.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `categorization_model.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `config.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `config_passion.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `contracts.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `expected_spend_model.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `feature_engineer.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `insight_generator.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `insight_model.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `known_persons.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `log_utils.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `logger_factory.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `marketplace_subcategory.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `model_state.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `passion_detector.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `passion_insight_generator.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `passion_models.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `passion_pipeline.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `passion_utils.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `pipeline_result.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `preprocessor.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `recurring_detector.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `seed_labeler.py` (via `pipeline.py`) → [module/symbols used indirectly]
- **Direct external**:
  - `numpy` → [module]
  - `pandas` → [module]
  - `psutil` → [module]
- **Runtime / injected deps**: none

#### Identity
- **Role**: TEST
- **Purpose**: Test suite or testing script.

#### Exports
- `generate_large_dataset` (function)
- `run_stress_test` (function)

#### Side Effects
file I/O (open/print), file I/O

---

### `tests/run_tests_legacy.py`

#### Dependencies
- **Direct internal**:
  - `tests/test_phase1.py` → [*]
  - `tests/test_phase2.py` → [test_spend_model_training_and_prediction, test_categorization_training_and_prediction]
- **Transitive internal**:
  - `categorization_model.py` (via `tests/test_phase2.py`) → [module/symbols used indirectly]
  - `config.py` (via `tests/test_phase2.py`) → [module/symbols used indirectly]
  - `expected_spend_model.py` (via `tests/test_phase2.py`) → [module/symbols used indirectly]
  - `feature_engineer.py` (via `tests/test_phase1.py`) → [module/symbols used indirectly]
  - `logger_factory.py` (via `tests/test_phase2.py`) → [module/symbols used indirectly]
  - `preprocessor.py` (via `tests/test_phase1.py`) → [module/symbols used indirectly]
  - `schema.py` (via `tests/test_phase2.py`) → [module/symbols used indirectly]
  - `seed_labeler.py` (via `tests/test_phase1.py`) → [module/symbols used indirectly]
- **Direct external**:
  - none
- **Runtime / injected deps**: none

#### Identity
- **Role**: TEST
- **Purpose**: Test suite or testing script.

#### Exports
- none


#### Side Effects
file I/O (open/print)

---

### `tests/test_benchmark.py`

#### Dependencies
- **Direct internal**:
  - `config.py` → [INSIGHT_TYPES, TIP_CORPUS]
  - `training_data_generator.py` → [_find_best_tip, _generate_base_features, generate_insight_dataset, ALL_CATEGORIES]
- **Transitive internal**:
  - `contracts.py` (via `training_data_generator.py`) → [module/symbols used indirectly]
- **Direct external**:
  - `numpy` → [module]
  - `pandas` → [module]
  - `pytest` → [module]
  - `sklearn` → [Pipeline, StandardScaler, OneHotEncoder, LogisticRegression, ColumnTransformer]
- **Runtime / injected deps**: none

#### Identity
- **Role**: TEST
- **Purpose**: Test suite or testing script.

#### Exports
- `dataset` (function)
- `test_data_generator_shape` (function)
- `test_no_nan_in_features` (function)
- `test_no_nan_in_labels` (function)
- `test_data_generator_distribution` (function)
- `test_all_insight_types_present` (function)
- `test_categories_are_valid` (function)
- `test_spikes_have_high_zscore` (function)
- `test_subscriptions_have_recurring_flag` (function)
- `test_no_action_has_benign_features` (function)
- `test_no_action_has_no_tip` (function)
- `test_actionable_insights_have_tips` (function)
- `test_tip_ids_are_valid` (function)
- `test_find_best_tip_category_specific` (function)
- `test_find_best_tip_generic_fallback` (function)
- `test_find_best_tip_no_match` (function)
- `test_amounts_are_positive` (function)
- `test_rolling_std_no_zero` (function)
- `test_model_can_train_and_predict` (function)

#### Side Effects
network calls, file I/O

---

### `tests/test_e2e.py`

#### Dependencies
- **Direct internal**:
  - `anomaly_detector.py` → [detect_anomalies]
  - `categorization_model.py` → [predict_categories, train_categorization_model]
  - `expected_spend_model.py` → [predict_expected_spend, train_expected_spend_model]
  - `feature_engineer.py` → [engineer_features_inference, fill_rolling_nulls, engineer_features]
  - `insight_generator.py` → [generate_human_insights]
  - `model_state.py` → [InsightModelState]
  - `pipeline.py` → [run_pipeline, run_inference]
  - `preprocessor.py` → [preprocess]
  - `recurring_detector.py` → [find_recurring_transactions]
  - `schema.py` → [Col]
  - `seed_labeler.py` → [label_credits, label_debits]
- **Transitive internal**:
  - `banned_content.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `bootstrap.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `candidate.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `config.py` (via `preprocessor.py`) → [module/symbols used indirectly]
  - `config_passion.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `contracts.py` (via `insight_generator.py`) → [module/symbols used indirectly]
  - `insight_model.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `known_persons.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `log_utils.py` (via `recurring_detector.py`) → [module/symbols used indirectly]
  - `logger_factory.py` (via `preprocessor.py`) → [module/symbols used indirectly]
  - `marketplace_subcategory.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `passion_detector.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `passion_insight_generator.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `passion_models.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `passion_pipeline.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `passion_utils.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `pipeline_result.py` (via `pipeline.py`) → [module/symbols used indirectly]
- **Direct external**:
  - `numpy` → [module]
  - `pandas` → [module]
- **Runtime / injected deps**: none

#### Identity
- **Role**: TEST
- **Purpose**: Test suite or testing script.

#### Exports
- `test_run_e2e_test` (function)
- `test_run_inference_uses_history_features` (function)

#### Side Effects
file I/O (open/print), file I/O

---

### `tests/test_logging_safety.py`

#### Dependencies
- **Direct internal**:
  - `config.py` → [module]
  - `log_utils.py` → [log_safe_merchant]
  - `logger_factory.py` → [module]
  - `pipeline.py` → [run_pipeline, generate_new_run_id]
  - `recurring_detector.py` → [find_recurring_transactions]
  - `schema.py` → [Col]
- **Transitive internal**:
  - `anomaly_detector.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `banned_content.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `bootstrap.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `candidate.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `categorization_model.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `config_passion.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `contracts.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `expected_spend_model.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `feature_engineer.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `insight_generator.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `insight_model.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `known_persons.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `marketplace_subcategory.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `model_state.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `passion_detector.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `passion_insight_generator.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `passion_models.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `passion_pipeline.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `passion_utils.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `pipeline_result.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `preprocessor.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `seed_labeler.py` (via `pipeline.py`) → [module/symbols used indirectly]
- **Direct external**:
  - `pandas` → [module]
  - `pytest` → [module]
- **Runtime / injected deps**: config

#### Identity
- **Role**: TEST
- **Purpose**: Test suite or testing script.

#### Exports
- `valid_debits_df` (function)
- `valid_credits_df` (function)
- `raw_df` (function)
- `test_crash_dump_created` (function)
- `test_crash_exception_identity_matching` (function)
- `test_crash_dump_failure_safety` (function)
- `test_invalid_log_level` (function)
- `test_pii_redaction_coverage` (function)

#### Side Effects
file I/O (open/print), network calls, file I/O

---

### `tests/test_ml_integration.py`

#### Dependencies
- **Direct internal**:
  - `insight_model.py` → [predict_insight_scores, load_insight_ranker]
  - `schema.py` → [Col]
- **Transitive internal**:
  - `config.py` (via `insight_model.py`) → [module/symbols used indirectly]
  - `logger_factory.py` (via `insight_model.py`) → [module/symbols used indirectly]
- **Direct external**:
  - `pandas` → [module]
  - `pytest` → [module]
  - `sklearn` → [Pipeline]
- **Runtime / injected deps**: none

#### Identity
- **Role**: TEST
- **Purpose**: Test suite or testing script.

#### Exports
- `mock_df` (function)
- `test_load_insight_ranker` (function)
- `test_predict_insight_scores` (function)
- `test_predict_insight_scores_graceful_fallback` (function)

#### Side Effects
file I/O

---

### `tests/test_model_security.py`

#### Dependencies
- **Direct internal**:
  - `insight_model.py` → [load_insight_ranker, module, _MODELS_DIR, ModelSecurityError, _verify_checksum, _validate_model_path, _compute_checksum]
- **Transitive internal**:
  - `config.py` (via `insight_model.py`) → [module/symbols used indirectly]
  - `logger_factory.py` (via `insight_model.py`) → [module/symbols used indirectly]
  - `schema.py` (via `insight_model.py`) → [module/symbols used indirectly]
- **Direct external**:
  - `pytest` → [module]
- **Runtime / injected deps**: none

#### Identity
- **Role**: TEST
- **Purpose**: Test suite or testing script.

#### Exports
- `tmp_models_dir` (function)
- `TestChecksumComputation` (class)
- `TestChecksumVerification` (class)
- `TestPathValidation` (class)
- `TestLoadInsightRanker` (class)

#### Side Effects
file I/O (open/print)

---

### `tests/test_passion_engine.py`

#### Dependencies
- **Direct internal**:
  - `banned_content.py` → [module, contains_banned_content]
  - `bootstrap.py` → [validate_template_fields, _validate_schema_columns, _dry_render_templates, module, _validate_tip_corpus]
  - `candidate.py` → [module, Candidate]
  - `config.py` → [module]
  - `config_passion.py` → [GENERALIST_CANONICALS, module, PASSION_MERCHANT_ALIASES, validate_merchant_aliases]
  - `contracts.py` → [module, _freeze_tip_corpus, lookup_matching_tip_ids]
  - `hash_utils.py` → [stable_hash]
  - `log_utils.py` → [log_safe_text, verify_merchant_token, _reset_secret_cache, log_safe_merchant, module, _get_secret]
  - `marketplace_subcategory.py` → [module, enrich_subcategories, resolve_merchant_vectorized]
  - `passion_detector.py` → [_is_non_declining, module, detect_passions, _check_distress_gate, _check_anomaly_suppression]
  - `passion_insight_generator.py` → [module, generate_passion_insights]
  - `passion_models.py` → [module, PassionSignal]
  - `passion_pipeline.py` → [_neutral_passion_result, safe_assign_new_columns, process_pipeline, _normalize_ts, _StepBudgetGuard, module]
  - `passion_utils.py` → [validate_template_values, safe_last_nonnull, sanitize_mask, safe_numeric, _safe_isna, module, coerce_bool_column, to_bool_strict]
  - `pipeline.py` → [_write_crash_dumps, module, PipelineResult]
  - `pipeline_result.py` → [module, PassionResult]
  - `schema.py` → [module, Col]
- **Transitive internal**:
  - `anomaly_detector.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `categorization_model.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `expected_spend_model.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `feature_engineer.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `insight_generator.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `insight_model.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `known_persons.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `logger_factory.py` (via `passion_pipeline.py`) → [module/symbols used indirectly]
  - `model_state.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `preprocessor.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `recurring_detector.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `seed_labeler.py` (via `pipeline.py`) → [module/symbols used indirectly]
- **Direct external**:
  - `numpy` → [module]
  - `pandas` → [module]
  - `pytest` → [module]
- **Runtime / injected deps**: none

#### Identity
- **Role**: TEST
- **Purpose**: Test suite or testing script.

#### Exports
- `make_pipeline_result` (function)
- `TestBoolCoercion` (class)
- `TestSanitizeMask` (class)
- `TestNormalizeTs` (class)
- `TestStepBudgetGuard` (class)
- `TestConfigPassion` (class)
- `TestLogUtils` (class)
- `TestBannedContent` (class)
- `TestDistressGate` (class)
- `TestMerchantVectorizer` (class)
- `TestMarketplaceSubcategory` (class)
- `TestIsNonDeclining` (class)
- `TestEnrichSubcategories` (class)
- `TestAnomalySuppression` (class)
- `TestDetectPassions` (class)
- `TestSuppressedSignals` (class)
- `TestNarrowIndexException` (class)
- `TestPipeline` (class)
- `TestBootstrap` (class)
- `TestUtils` (class)
- `TestPassionSignalValidation` (class)
- `TestCandidateValidation` (class)
- `TestPassionResultRejection` (class)
- `TestThreadSafeInit` (class)
- `TestModuleImports` (class)
- `TestA2TipCorpusImportIsolation` (class)
- `TestA3StableHashProductionBan` (class)
- `TestNeutralPassionResult` (class)
- `TestE3PassionEngineE2EIntegration` (class)
- `TestE4DefensiveCopyIntegrity` (class)
- `TestF1CowSafeResolve` (class)
- `TestF2AliasBoundaryHardening` (class)
- `TestF3SecretCachingBehavior` (class)
- `TestF4MemoryPerformanceBenchmark` (class)
- `TestF5CrashDumpWithPassion` (class)
- `TestF6UnexpectedColumnsStrictMode` (class)
- `TestConfigPassionCanonicals` (class)

#### Side Effects
file I/O (open/print), network calls, file I/O

---

### `tests/test_phase1.py`

#### Dependencies
- **Direct internal**:
  - `config.py` → [CREDIT_PRIORITY, CREDIT_KEYWORDS, CATEGORY_KEYWORDS, CATEGORY_PRIORITY, FALLBACK_DEBIT_LABEL, FALLBACK_CREDIT_LABEL]
  - `feature_engineer.py` → [add_rolling_features, engineer_features, fill_rolling_nulls, add_time_features, engineer_features_inference, add_amount_features]
  - `preprocessor.py` → [clean_remark, _compute_signed_amount, preprocess, _normalize_flag, _drop_zero_amount, _deduplicate, validate_schema]
  - `schema.py` → [Col]
  - `seed_labeler.py` → [label_credits, _match_remark, _compile_keywords, label_debits]
- **Transitive internal**:
  - `logger_factory.py` (via `schema.py`) → [module/symbols used indirectly]
- **Direct external**:
  - `numpy` → [module]
  - `pandas` → [module]
  - `pytest` → [module]
- **Runtime / injected deps**: none

#### Identity
- **Role**: TEST
- **Purpose**: Test suite or testing script.

#### Exports
- `_make_df` (function)
- `_make_debit_df_with_remarks` (function)
- `TestValidateSchema` (class)
- `TestNormalizeFlag` (class)
- `TestComputeSignedAmount` (class)
- `TestCleanRemark` (class)
- `TestDropZeroAmount` (class)
- `TestDeduplicate` (class)
- `TestPreprocess` (class)
- `_base_fe_df` (function)
- `TestTimeFeatures` (class)
- `TestRollingFeatures` (class)
- `TestFillRollingNulls` (class)
- `TestAmountFeatures` (class)
- `TestEngineerFeaturesFull` (class)
- `TestMatchRemark` (class)
- `TestLabelDebits` (class)
- `TestLabelCredits` (class)
- `TestInferenceFeatures` (class)

#### Side Effects
file I/O

---

### `tests/test_phase2.py`

#### Dependencies
- **Direct internal**:
  - `categorization_model.py` → [predict_categories, train_categorization_model]
  - `expected_spend_model.py` → [predict_expected_spend, train_expected_spend_model]
- **Transitive internal**:
  - `config.py` (via `categorization_model.py`) → [module/symbols used indirectly]
  - `logger_factory.py` (via `categorization_model.py`) → [module/symbols used indirectly]
  - `schema.py` (via `categorization_model.py`) → [module/symbols used indirectly]
- **Direct external**:
  - `numpy` → [module]
  - `pandas` → [module]
  - `pytest` → [module]
- **Runtime / injected deps**: none

#### Identity
- **Role**: TEST
- **Purpose**: Test suite or testing script.

#### Exports
- `test_categorization_training_and_prediction` (function)
- `test_spend_model_training_and_prediction` (function)
- `test_expected_spend_model_extrapolates` (function)
- `test_percent_deviation_no_inf` (function)
- `test_percent_deviation_negative_expected_amount` (function)
- `test_percent_deviation_zero_expected_amount` (function)
- `test_percent_deviation_normal_case` (function)

#### Side Effects
file I/O

---

### `tests/test_phase3.py`

#### Dependencies
- **Direct internal**:
  - `anomaly_detector.py` → [detect_anomalies]
  - `insight_generator.py` → [_select_tip, generate_human_insights]
  - `pipeline.py` → [PipelineResult]
  - `recurring_detector.py` → [find_recurring_transactions]
- **Transitive internal**:
  - `banned_content.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `bootstrap.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `candidate.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `categorization_model.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `config.py` (via `anomaly_detector.py`) → [module/symbols used indirectly]
  - `config_passion.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `contracts.py` (via `insight_generator.py`) → [module/symbols used indirectly]
  - `expected_spend_model.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `feature_engineer.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `insight_model.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `known_persons.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `log_utils.py` (via `recurring_detector.py`) → [module/symbols used indirectly]
  - `logger_factory.py` (via `anomaly_detector.py`) → [module/symbols used indirectly]
  - `marketplace_subcategory.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `model_state.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `passion_detector.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `passion_insight_generator.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `passion_models.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `passion_pipeline.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `passion_utils.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `pipeline_result.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `preprocessor.py` (via `pipeline.py`) → [module/symbols used indirectly]
  - `schema.py` (via `anomaly_detector.py`) → [module/symbols used indirectly]
  - `seed_labeler.py` (via `pipeline.py`) → [module/symbols used indirectly]
- **Direct external**:
  - `pandas` → [module]
  - `pytest` → [module]
- **Runtime / injected deps**: none

#### Identity
- **Role**: TEST
- **Purpose**: Test suite or testing script.

#### Exports
- `test_recurring_detector` (function)
- `test_recurring_frequency_none_for_non_recurring` (function)
- `test_recurring_requires_minimum_3_occurrences` (function)
- `test_biweekly_detection_and_variance_penalty` (function)
- `test_anomaly_composite_threshold` (function)
- `test_missing_anomaly_columns_raises` (function)
- `test_insight_generator_creates_strings` (function)
- `test_insight_generator_deterministic` (function)
- `test_insight_generator_different_seeds` (function)
- `test_select_tip_deterministic` (function)
- `test_pipeline_result_is_frozen` (function)
- `test_pipeline_result_replace` (function)

#### Side Effects
file I/O

---

## UTILITY

### `hash_utils.py`

#### Dependencies
- **Direct internal**:
  - none
- **Transitive internal**:
  - none
- **Direct external**:
  - none
- **Runtime / injected deps**: none

#### Identity
- **Role**: UTILITY
- **Purpose**: Provides reusable helper functions or utilities.

#### Exports
- `__all__` (constant)
- `stable_hash` (function)

#### Side Effects
none

---

### `log_utils.py`

#### Dependencies
- **Direct internal**:
  - `logger_factory.py` → [get_logger]
- **Transitive internal**:
  - `config.py` (via `logger_factory.py`) → [module/symbols used indirectly]
- **Direct external**:
  - `numpy` → [module]
  - `pandas` → [module]
- **Runtime / injected deps**: none

#### Identity
- **Role**: UTILITY
- **Purpose**: Provides reusable helper functions or utilities.

#### Exports
- `__all__` (constant)
- `_DEV_SECRET_FALLBACK` (constant)
- `_reset_secret_cache` (function)
- `_get_secret` (function)
- `_hmac_hex` (function)
- `_is_safe_scalar` (function)
- `log_safe_merchant` (function)
- `log_safe_text` (function)
- `verify_merchant_token` (function)

#### Side Effects
network calls, file I/O

---

### `logger_factory.py`

#### Dependencies
- **Direct internal**:
  - `config.py` → [module]
- **Transitive internal**:
  - none
- **Direct external**:
  - none
- **Runtime / injected deps**: none

#### Identity
- **Role**: UTILITY
- **Purpose**: Provides reusable helper functions or utilities.

#### Exports
- `generate_new_run_id` (function)
- `JSONFormatter` (class)
- `get_logger` (function)

#### Side Effects
network calls

---
*(Part 4 of 4 - split due to length)*


### `passion_utils.py`

#### Dependencies
- **Direct internal**:
  - `logger_factory.py` → [get_logger]
- **Transitive internal**:
  - `config.py` (via `logger_factory.py`) → [module/symbols used indirectly]
- **Direct external**:
  - `numpy` → [module]
  - `pandas` → [module]
- **Runtime / injected deps**: none

#### Identity
- **Role**: UTILITY
- **Purpose**: Provides reusable helper functions or utilities.

#### Exports
- `__all__` (constant)
- `assert_columns_exist` (function)
- `_safe_isna` (function)
- `to_bool_strict` (function)
- `coerce_bool_column` (function)
- `sanitize_mask` (function)
- `safe_last_nonnull` (function)
- `_ALLOWED_TEMPLATE_SCALAR_TYPES` (constant)
- `validate_template_values` (function)
- `_CURRENCY_RE` (constant)
- `_INR_RE` (constant)
- `safe_numeric` (function)

#### Side Effects
file I/O

---

### `refactor_pipeline.py`

#### Dependencies
- **Direct internal**:
  - none
- **Transitive internal**:
  - none
- **Direct external**:
  - none
- **Runtime / injected deps**: none

#### Identity
- **Role**: UTILITY
- **Purpose**: Provides reusable helper functions or utilities.

#### Exports
- `indent_code` (function)

#### Side Effects
file I/O (open/print), file I/O

---

### `summary_utils.py`

#### Dependencies
- **Direct internal**:
  - `schema.py` → [Col]
- **Transitive internal**:
  - `config.py` (via `schema.py`) → [module/symbols used indirectly]
  - `logger_factory.py` (via `schema.py`) → [module/symbols used indirectly]
- **Direct external**:
  - `pandas` → [module]
- **Runtime / injected deps**: none

#### Identity
- **Role**: UTILITY
- **Purpose**: Provides reusable helper functions or utilities.

#### Exports
- `print_summary` (function)

#### Side Effects
file I/O (open/print), file I/O

---
