*(Part 1 of 3 - split due to length)*

# Internal Dependency Graph

## Full Direct Import Map
`anomaly_detector.py` → `schema.py` [Col, require_columns]
`bootstrap.py` → `config_passion.py` [validate_config, PASSION_INSIGHT_TEMPLATES, validate_merchant_aliases]
`bootstrap.py` → `contracts.py` [INSIGHT_TEMPLATES, TIP_CORPUS]
`bootstrap.py` → `log_utils.py` [_get_secret]
`bootstrap.py` → `logger_factory.py` [get_logger]
`bootstrap.py` → `schema.py` [Col]
`candidate.py` → `passion_models.py` [PassionSignal]
`categorization_model.py` → `config.py` [FALLBACK_DEBIT_LABEL, FALLBACK_CREDIT_LABEL]
`categorization_model.py` → `schema.py` [Col, require_columns]
`config_passion.py` → `config.py` [SPECIFIC_MERCHANT_ALIASES, CATEGORY_KEYWORDS, CATEGORY_PRIORITY]
`contracts.py` → `config.py` [INSIGHT_TEMPLATES, TIP_CORPUS]
`demo.py` → `pipeline.py` [run_pipeline]
`demo.py` → `schema.py` [Col]
`demo.py` → `summary_utils.py` [print_summary]
`expected_spend_model.py` → `schema.py` [Col, require_columns]
`feature_engineer.py` → `schema.py` [Col, require_columns]
`insight_generator.py` → `contracts.py` [lookup_matching_tip_ids, INSIGHT_TEMPLATES, TIP_CORPUS]
`insight_generator.py` → `schema.py` [Col, require_columns]
`insight_model.py` → `schema.py` [Col, require_columns]
`known_persons.py` → `config.py` [module]
`known_persons.py` → `logger_factory.py` [get_logger]
`known_persons.py` → `schema.py` [Col]
`log_utils.py` → `logger_factory.py` [get_logger]
`logger_factory.py` → `config.py` [module]
`marketplace_subcategory.py` → `config_passion.py` [MARKETPLACE_HIGH_CONFIDENCE, PASSION_MERCHANT_ALIASES, ELECTRONICS_ALLOWED_CATEGORIES, MARKETPLACE_HIGH_AMOUNT_THRESHOLD, MARKETPLACE_LOW_CONFIDENCE, GENERALIST_CANONICALS]
`marketplace_subcategory.py` → `logger_factory.py` [get_logger]
`marketplace_subcategory.py` → `passion_utils.py` [safe_numeric, coerce_bool_column, assert_columns_exist]
`marketplace_subcategory.py` → `schema.py` [Col]
`model_benchmark.py` → `training_data_generator.py` [generate_insight_dataset]
`passion_detector.py` → `config_passion.py` [PASSION_ANOMALY_SUPPRESSION_THRESHOLD, DISTRESS_FEES_THRESHOLD, PASSION_SPEND_SHARE_THRESHOLD, PASSION_MIN_MONTHS, PASSION_MERCHANT_COUNT_MIN]
`passion_detector.py` → `logger_factory.py` [get_logger]
`passion_detector.py` → `marketplace_subcategory.py` [resolve_merchant_vectorized]
`passion_detector.py` → `passion_models.py` [PassionSignal]
`passion_detector.py` → `passion_utils.py` [assert_columns_exist, sanitize_mask, safe_numeric, _safe_isna, coerce_bool_column]
`passion_detector.py` → `schema.py` [Col]
`passion_insight_generator.py` → `banned_content.py` [contains_banned_content]
`passion_insight_generator.py` → `config_passion.py` [PASSION_INSIGHT_TEMPLATES]
`passion_insight_generator.py` → `contracts.py` [lookup_matching_tip_ids, INSIGHT_TEMPLATES, TIP_CORPUS]
`passion_insight_generator.py` → `logger_factory.py` [get_logger]
`passion_insight_generator.py` → `passion_models.py` [PassionSignal]
`passion_insight_generator.py` → `passion_utils.py` [validate_template_values]
`passion_insight_generator.py` → `schema.py` [Col]
`passion_pipeline.py` → `banned_content.py` [contains_banned_content]
`passion_pipeline.py` → `bootstrap.py` [run_startup_checks]
`passion_pipeline.py` → `candidate.py` [Candidate]
`passion_pipeline.py` → `config_passion.py` [PIPELINE_BUDGET_MS, PASSION_MIN_DEBIT_ROWS, PIPELINE_TOP_N, MAX_SPIKE_CANDIDATES, PIPELINE_HARD_TIMEOUT_MS]
`passion_pipeline.py` → `logger_factory.py` [get_logger]
`passion_pipeline.py` → `marketplace_subcategory.py` [enrich_subcategories, resolve_merchant_vectorized]
`passion_pipeline.py` → `passion_detector.py` [detect_passions]
`passion_pipeline.py` → `passion_insight_generator.py` [generate_passion_insights]
`passion_pipeline.py` → `passion_utils.py` [_safe_isna, coerce_bool_column, assert_columns_exist, safe_numeric]
`passion_pipeline.py` → `pipeline_result.py` [PassionResult]
`passion_pipeline.py` → `schema.py` [Col]
`passion_utils.py` → `logger_factory.py` [get_logger]
`pipeline.py` → `anomaly_detector.py` [detect_anomalies]
`pipeline.py` → `categorization_model.py` [predict_categories, train_categorization_model]
`pipeline.py` → `config.py` [module]
`pipeline.py` → `expected_spend_model.py` [predict_expected_spend, train_expected_spend_model]
`pipeline.py` → `feature_engineer.py` [engineer_features_inference, engineer_features]
`pipeline.py` → `insight_generator.py` [generate_human_insights]
`pipeline.py` → `insight_model.py` [predict_insight_scores, load_insight_ranker]
`pipeline.py` → `known_persons.py` [log_unmatched_recurring_transfers, detect_personal_patterns, _enforce_known_person_schema, tag_known_persons]
`pipeline.py` → `log_utils.py` [log_safe_text, log_safe_merchant]
`pipeline.py` → `logger_factory.py` [generate_new_run_id, get_logger, pipeline_run_id_ctx]
`pipeline.py` → `model_state.py` [InsightModelState]
`pipeline.py` → `passion_pipeline.py` [process_pipeline]
`pipeline.py` → `preprocessor.py` [preprocess]
`pipeline.py` → `recurring_detector.py` [find_recurring_transactions]
`pipeline.py` → `schema.py` [Col]
`pipeline.py` → `seed_labeler.py` [label_credits, label_debits]
`pipeline_result.py` → `candidate.py` [Candidate]
`pipeline_result.py` → `passion_models.py` [PassionSignal]
`preprocessor.py` → `config.py` [SPECIFIC_MERCHANT_ALIASES, NOISE_TOKENS, GENERIC_ROUTER_ALIASES]
`preprocessor.py` → `logger_factory.py` [get_logger]
`preprocessor.py` → `schema.py` [Col, coerce_and_validate_types, require_columns]
`recurring_detector.py` → `config.py` [module, RECURRING_CONFIG]
`recurring_detector.py` → `log_utils.py` [log_safe_merchant]
`recurring_detector.py` → `logger_factory.py` [get_logger]
`recurring_detector.py` → `schema.py` [Col, require_columns]
`schema.py` → `logger_factory.py` [get_logger]
`seed_labeler.py` → `config.py` [TIER_MAPPING, CREDIT_PRIORITY, CREDIT_KEYWORDS, CATEGORY_KEYWORDS, CATEGORY_PRIORITY, FALLBACK_DEBIT_LABEL, MIN_COVERAGE_THRESHOLD, FALLBACK_CREDIT_LABEL]
`seed_labeler.py` → `preprocessor.py` [normalize]
`seed_labeler.py` → `schema.py` [Col, require_columns]
`summary_utils.py` → `schema.py` [Col]
`tests/conftest.py` → `log_utils.py` [_reset_secret_cache]
`tests/conftest.py` → `passion_pipeline.py` [module]
`tests/run_smoke.py` → `pipeline.py` [run_pipeline, PipelineResult]
`tests/run_stress_heavy.py` → `pipeline.py` [run_pipeline]
`tests/run_stress_legacy.py` → `pipeline.py` [run_pipeline]
`tests/run_stress_legacy.py` → `schema.py` [Col]
`tests/run_tests_legacy.py` → `tests/test_phase1.py` [*]
`tests/run_tests_legacy.py` → `tests/test_phase2.py` [test_spend_model_training_and_prediction, test_categorization_training_and_prediction]
`tests/test_benchmark.py` → `config.py` [INSIGHT_TYPES, TIP_CORPUS]
`tests/test_benchmark.py` → `training_data_generator.py` [_find_best_tip, _generate_base_features, generate_insight_dataset, ALL_CATEGORIES]
`tests/test_e2e.py` → `anomaly_detector.py` [detect_anomalies]
`tests/test_e2e.py` → `categorization_model.py` [predict_categories, train_categorization_model]
`tests/test_e2e.py` → `expected_spend_model.py` [predict_expected_spend, train_expected_spend_model]
`tests/test_e2e.py` → `feature_engineer.py` [engineer_features_inference, fill_rolling_nulls, engineer_features]
`tests/test_e2e.py` → `insight_generator.py` [generate_human_insights]
`tests/test_e2e.py` → `model_state.py` [InsightModelState]
`tests/test_e2e.py` → `pipeline.py` [run_pipeline, run_inference]
`tests/test_e2e.py` → `preprocessor.py` [preprocess]
`tests/test_e2e.py` → `recurring_detector.py` [find_recurring_transactions]
`tests/test_e2e.py` → `schema.py` [Col]
`tests/test_e2e.py` → `seed_labeler.py` [label_credits, label_debits]
`tests/test_known_persons.py` → `known_persons.py` [_suggestion_key, _enforce_known_person_schema, tag_known_persons, _extract_signals]
`tests/test_known_persons.py` → `model_state.py` [InsightModelState]
`tests/test_known_persons.py` → `pipeline.py` [_compute_config_hash, run_pipeline, PipelineResult, run_inference]
`tests/test_known_persons.py` → `schema.py` [Col]
`tests/test_logging_safety.py` → `config.py` [module]
`tests/test_logging_safety.py` → `log_utils.py` [log_safe_merchant]
`tests/test_logging_safety.py` → `logger_factory.py` [module]
`tests/test_logging_safety.py` → `pipeline.py` [run_pipeline, generate_new_run_id]
`tests/test_logging_safety.py` → `recurring_detector.py` [find_recurring_transactions]
`tests/test_logging_safety.py` → `schema.py` [Col]
`tests/test_ml_integration.py` → `insight_model.py` [predict_insight_scores, load_insight_ranker]
`tests/test_ml_integration.py` → `schema.py` [Col]
`tests/test_model_security.py` → `insight_model.py` [load_insight_ranker, module, _MODELS_DIR, ModelSecurityError, _verify_checksum, _validate_model_path, _compute_checksum]
`tests/test_passion_engine.py` → `banned_content.py` [module, contains_banned_content]
`tests/test_passion_engine.py` → `bootstrap.py` [validate_template_fields, _validate_schema_columns, _dry_render_templates, module, _validate_tip_corpus]
`tests/test_passion_engine.py` → `candidate.py` [module, Candidate]
`tests/test_passion_engine.py` → `config.py` [module]
`tests/test_passion_engine.py` → `config_passion.py` [GENERALIST_CANONICALS, module, PASSION_MERCHANT_ALIASES, validate_merchant_aliases]
`tests/test_passion_engine.py` → `contracts.py` [module, _freeze_tip_corpus, lookup_matching_tip_ids]
`tests/test_passion_engine.py` → `hash_utils.py` [stable_hash]
`tests/test_passion_engine.py` → `log_utils.py` [log_safe_text, verify_merchant_token, _reset_secret_cache, log_safe_merchant, module, _get_secret]
`tests/test_passion_engine.py` → `marketplace_subcategory.py` [module, enrich_subcategories, resolve_merchant_vectorized]
`tests/test_passion_engine.py` → `passion_detector.py` [_is_non_declining, module, detect_passions, _check_distress_gate, _check_anomaly_suppression]
`tests/test_passion_engine.py` → `passion_insight_generator.py` [module, generate_passion_insights]
`tests/test_passion_engine.py` → `passion_models.py` [module, PassionSignal]
`tests/test_passion_engine.py` → `passion_pipeline.py` [_neutral_passion_result, safe_assign_new_columns, process_pipeline, _normalize_ts, _StepBudgetGuard, module]
`tests/test_passion_engine.py` → `passion_utils.py` [validate_template_values, safe_last_nonnull, sanitize_mask, safe_numeric, _safe_isna, module, coerce_bool_column, to_bool_strict]
`tests/test_passion_engine.py` → `pipeline.py` [_write_crash_dumps, module, PipelineResult]
`tests/test_passion_engine.py` → `pipeline_result.py` [module, PassionResult]
`tests/test_passion_engine.py` → `schema.py` [module, Col]
`tests/test_phase1.py` → `config.py` [CREDIT_PRIORITY, CREDIT_KEYWORDS, CATEGORY_KEYWORDS, CATEGORY_PRIORITY, FALLBACK_DEBIT_LABEL, FALLBACK_CREDIT_LABEL]
`tests/test_phase1.py` → `feature_engineer.py` [add_rolling_features, engineer_features, fill_rolling_nulls, add_time_features, engineer_features_inference, add_amount_features]
`tests/test_phase1.py` → `preprocessor.py` [clean_remark, _compute_signed_amount, preprocess, _normalize_flag, _drop_zero_amount, _deduplicate, validate_schema]
`tests/test_phase1.py` → `schema.py` [Col]
`tests/test_phase1.py` → `seed_labeler.py` [label_credits, _match_remark, _compile_keywords, label_debits]
`tests/test_phase2.py` → `categorization_model.py` [predict_categories, train_categorization_model]
`tests/test_phase2.py` → `expected_spend_model.py` [predict_expected_spend, train_expected_spend_model]
`tests/test_phase3.py` → `anomaly_detector.py` [detect_anomalies]
`tests/test_phase3.py` → `insight_generator.py` [_select_tip, generate_human_insights]
`tests/test_phase3.py` → `pipeline.py` [PipelineResult]
`tests/test_phase3.py` → `recurring_detector.py` [find_recurring_transactions]
`train_and_save_models.py` → `insight_model.py` [_compute_checksum]
`train_and_save_models.py` → `schema.py` [Col]
`train_and_save_models.py` → `training_data_generator.py` [generate_insight_dataset]
`training_data_generator.py` → `config.py` [lookup_matching_tip_ids, INSIGHT_TYPES, CATEGORY_PRIORITY]
`training_data_generator.py` → `contracts.py` [TIP_CORPUS]
`tutorial_real_data.py` → `logger_factory.py` [get_logger]
`tutorial_real_data.py` → `pipeline.py` [run_pipeline]
`tutorial_real_data.py` → `schema.py` [Col]

## Transitive Dependency Chains
**[HIGH RISK]** `anomaly_detector.py` → `schema.py` → `logger_factory.py` → `config.py`
`bootstrap.py` → `contracts.py` → `config.py`
`bootstrap.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `bootstrap.py` → `schema.py` → `logger_factory.py` → `config.py`
`bootstrap.py` → `config_passion.py` → `config.py`
**[HIGH RISK]** `bootstrap.py` → `log_utils.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `categorization_model.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `demo.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `demo.py` → `pipeline.py` → `logger_factory.py` → `config.py`
`demo.py` → `pipeline.py` → `model_state.py`
`demo.py` → `pipeline.py` → `config.py`
**[HIGH RISK]** `demo.py` → `pipeline.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `demo.py` → `pipeline.py` → `preprocessor.py` → `config.py`
**[HIGH RISK]** `demo.py` → `pipeline.py` → `preprocessor.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `demo.py` → `pipeline.py` → `preprocessor.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `demo.py` → `pipeline.py` → `feature_engineer.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `demo.py` → `pipeline.py` → `seed_labeler.py` → `config.py`
**[HIGH RISK]** `demo.py` → `pipeline.py` → `seed_labeler.py` → `preprocessor.py` → `config.py`
**[HIGH RISK]** `demo.py` → `pipeline.py` → `seed_labeler.py` → `preprocessor.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `demo.py` → `pipeline.py` → `seed_labeler.py` → `preprocessor.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `demo.py` → `pipeline.py` → `seed_labeler.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `demo.py` → `pipeline.py` → `categorization_model.py` → `config.py`
**[HIGH RISK]** `demo.py` → `pipeline.py` → `categorization_model.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `demo.py` → `pipeline.py` → `expected_spend_model.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `demo.py` → `pipeline.py` → `anomaly_detector.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `demo.py` → `pipeline.py` → `recurring_detector.py` → `config.py`
**[HIGH RISK]** `demo.py` → `pipeline.py` → `recurring_detector.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `demo.py` → `pipeline.py` → `recurring_detector.py` → `log_utils.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `demo.py` → `pipeline.py` → `recurring_detector.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `demo.py` → `pipeline.py` → `insight_model.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `demo.py` → `pipeline.py` → `insight_generator.py` → `contracts.py` → `config.py`
**[HIGH RISK]** `demo.py` → `pipeline.py` → `insight_generator.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `demo.py` → `pipeline.py` → `known_persons.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `demo.py` → `pipeline.py` → `known_persons.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `demo.py` → `pipeline.py` → `known_persons.py` → `config.py`
**[HIGH RISK]** `demo.py` → `pipeline.py` → `log_utils.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `demo.py` → `pipeline.py` → `passion_pipeline.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `demo.py` → `pipeline.py` → `passion_pipeline.py` → `config_passion.py` → `config.py`
**[HIGH RISK]** `demo.py` → `pipeline.py` → `passion_pipeline.py` → `passion_utils.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `demo.py` → `pipeline.py` → `passion_pipeline.py` → `candidate.py` → `passion_models.py`
**[HIGH RISK]** `demo.py` → `pipeline.py` → `passion_pipeline.py` → `banned_content.py`
**[HIGH RISK]** `demo.py` → `pipeline.py` → `passion_pipeline.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `demo.py` → `pipeline.py` → `passion_pipeline.py` → `pipeline_result.py` → `candidate.py` → `passion_models.py`
**[HIGH RISK]** `demo.py` → `pipeline.py` → `passion_pipeline.py` → `pipeline_result.py` → `passion_models.py`
**[HIGH RISK]** `demo.py` → `pipeline.py` → `passion_pipeline.py` → `bootstrap.py` → `contracts.py` → `config.py`
**[HIGH RISK]** `demo.py` → `pipeline.py` → `passion_pipeline.py` → `bootstrap.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `demo.py` → `pipeline.py` → `passion_pipeline.py` → `bootstrap.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `demo.py` → `pipeline.py` → `passion_pipeline.py` → `bootstrap.py` → `config_passion.py` → `config.py`
**[HIGH RISK]** `demo.py` → `pipeline.py` → `passion_pipeline.py` → `bootstrap.py` → `log_utils.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `demo.py` → `pipeline.py` → `passion_pipeline.py` → `marketplace_subcategory.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `demo.py` → `pipeline.py` → `passion_pipeline.py` → `marketplace_subcategory.py` → `config_passion.py` → `config.py`
**[HIGH RISK]** `demo.py` → `pipeline.py` → `passion_pipeline.py` → `marketplace_subcategory.py` → `passion_utils.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `demo.py` → `pipeline.py` → `passion_pipeline.py` → `marketplace_subcategory.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `demo.py` → `pipeline.py` → `passion_pipeline.py` → `passion_detector.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `demo.py` → `pipeline.py` → `passion_pipeline.py` → `passion_detector.py` → `config_passion.py` → `config.py`
**[HIGH RISK]** `demo.py` → `pipeline.py` → `passion_pipeline.py` → `passion_detector.py` → `passion_utils.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `demo.py` → `pipeline.py` → `passion_pipeline.py` → `passion_detector.py` → `marketplace_subcategory.py` → `config_passion.py` → `config.py`
**[HIGH RISK]** `demo.py` → `pipeline.py` → `passion_pipeline.py` → `passion_detector.py` → `marketplace_subcategory.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `demo.py` → `pipeline.py` → `passion_pipeline.py` → `passion_detector.py` → `passion_models.py`
**[HIGH RISK]** `demo.py` → `pipeline.py` → `passion_pipeline.py` → `passion_detector.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `demo.py` → `pipeline.py` → `passion_pipeline.py` → `passion_insight_generator.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `demo.py` → `pipeline.py` → `passion_pipeline.py` → `passion_insight_generator.py` → `contracts.py` → `config.py`
**[HIGH RISK]** `demo.py` → `pipeline.py` → `passion_pipeline.py` → `passion_insight_generator.py` → `config_passion.py` → `config.py`
**[HIGH RISK]** `demo.py` → `pipeline.py` → `passion_pipeline.py` → `passion_insight_generator.py` → `passion_models.py`
**[HIGH RISK]** `demo.py` → `pipeline.py` → `passion_pipeline.py` → `passion_insight_generator.py` → `passion_utils.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `demo.py` → `pipeline.py` → `passion_pipeline.py` → `passion_insight_generator.py` → `banned_content.py`
**[HIGH RISK]** `demo.py` → `pipeline.py` → `passion_pipeline.py` → `passion_insight_generator.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `demo.py` → `summary_utils.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `expected_spend_model.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `feature_engineer.py` → `schema.py` → `logger_factory.py` → `config.py`
`insight_generator.py` → `contracts.py` → `config.py`
**[HIGH RISK]** `insight_generator.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `insight_model.py` → `schema.py` → `logger_factory.py` → `config.py`
`known_persons.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `known_persons.py` → `schema.py` → `logger_factory.py` → `config.py`
`log_utils.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `marketplace_subcategory.py` → `schema.py` → `logger_factory.py` → `config.py`
`marketplace_subcategory.py` → `config_passion.py` → `config.py`
**[HIGH RISK]** `marketplace_subcategory.py` → `passion_utils.py` → `logger_factory.py` → `config.py`
`marketplace_subcategory.py` → `logger_factory.py` → `config.py`
`model_benchmark.py` → `training_data_generator.py` → `config.py`
**[HIGH RISK]** `model_benchmark.py` → `training_data_generator.py` → `contracts.py` → `config.py`
**[HIGH RISK]** `passion_detector.py` → `schema.py` → `logger_factory.py` → `config.py`
`passion_detector.py` → `config_passion.py` → `config.py`
**[HIGH RISK]** `passion_detector.py` → `passion_utils.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `passion_detector.py` → `marketplace_subcategory.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `passion_detector.py` → `marketplace_subcategory.py` → `config_passion.py` → `config.py`
**[HIGH RISK]** `passion_detector.py` → `marketplace_subcategory.py` → `passion_utils.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `passion_detector.py` → `marketplace_subcategory.py` → `logger_factory.py` → `config.py`
`passion_detector.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `passion_insight_generator.py` → `schema.py` → `logger_factory.py` → `config.py`
`passion_insight_generator.py` → `contracts.py` → `config.py`
`passion_insight_generator.py` → `config_passion.py` → `config.py`
**[HIGH RISK]** `passion_insight_generator.py` → `passion_utils.py` → `logger_factory.py` → `config.py`
`passion_insight_generator.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `passion_pipeline.py` → `schema.py` → `logger_factory.py` → `config.py`
`passion_pipeline.py` → `config_passion.py` → `config.py`
**[HIGH RISK]** `passion_pipeline.py` → `passion_utils.py` → `logger_factory.py` → `config.py`
`passion_pipeline.py` → `candidate.py` → `passion_models.py`
`passion_pipeline.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `passion_pipeline.py` → `pipeline_result.py` → `candidate.py` → `passion_models.py`
`passion_pipeline.py` → `pipeline_result.py` → `passion_models.py`
**[HIGH RISK]** `passion_pipeline.py` → `bootstrap.py` → `contracts.py` → `config.py`
**[HIGH RISK]** `passion_pipeline.py` → `bootstrap.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `passion_pipeline.py` → `bootstrap.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `passion_pipeline.py` → `bootstrap.py` → `config_passion.py` → `config.py`
**[HIGH RISK]** `passion_pipeline.py` → `bootstrap.py` → `log_utils.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `passion_pipeline.py` → `marketplace_subcategory.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `passion_pipeline.py` → `marketplace_subcategory.py` → `config_passion.py` → `config.py`
**[HIGH RISK]** `passion_pipeline.py` → `marketplace_subcategory.py` → `passion_utils.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `passion_pipeline.py` → `marketplace_subcategory.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `passion_pipeline.py` → `passion_detector.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `passion_pipeline.py` → `passion_detector.py` → `config_passion.py` → `config.py`
**[HIGH RISK]** `passion_pipeline.py` → `passion_detector.py` → `passion_utils.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `passion_pipeline.py` → `passion_detector.py` → `marketplace_subcategory.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `passion_pipeline.py` → `passion_detector.py` → `marketplace_subcategory.py` → `config_passion.py` → `config.py`
**[HIGH RISK]** `passion_pipeline.py` → `passion_detector.py` → `marketplace_subcategory.py` → `passion_utils.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `passion_pipeline.py` → `passion_detector.py` → `marketplace_subcategory.py` → `logger_factory.py` → `config.py`
`passion_pipeline.py` → `passion_detector.py` → `passion_models.py`
**[HIGH RISK]** `passion_pipeline.py` → `passion_detector.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `passion_pipeline.py` → `passion_insight_generator.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `passion_pipeline.py` → `passion_insight_generator.py` → `contracts.py` → `config.py`
**[HIGH RISK]** `passion_pipeline.py` → `passion_insight_generator.py` → `config_passion.py` → `config.py`
`passion_pipeline.py` → `passion_insight_generator.py` → `passion_models.py`
**[HIGH RISK]** `passion_pipeline.py` → `passion_insight_generator.py` → `passion_utils.py` → `logger_factory.py` → `config.py`
`passion_pipeline.py` → `passion_insight_generator.py` → `banned_content.py`
**[HIGH RISK]** `passion_pipeline.py` → `passion_insight_generator.py` → `logger_factory.py` → `config.py`
`passion_utils.py` → `logger_factory.py` → `config.py`
`pipeline.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `pipeline.py` → `schema.py` → `logger_factory.py` → `config.py`
`pipeline.py` → `preprocessor.py` → `config.py`
**[HIGH RISK]** `pipeline.py` → `preprocessor.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `pipeline.py` → `preprocessor.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `pipeline.py` → `feature_engineer.py` → `schema.py` → `logger_factory.py` → `config.py`
`pipeline.py` → `seed_labeler.py` → `config.py`
**[HIGH RISK]** `pipeline.py` → `seed_labeler.py` → `preprocessor.py` → `config.py`
**[HIGH RISK]** `pipeline.py` → `seed_labeler.py` → `preprocessor.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `pipeline.py` → `seed_labeler.py` → `preprocessor.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `pipeline.py` → `seed_labeler.py` → `schema.py` → `logger_factory.py` → `config.py`
`pipeline.py` → `categorization_model.py` → `config.py`
**[HIGH RISK]** `pipeline.py` → `categorization_model.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `pipeline.py` → `expected_spend_model.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `pipeline.py` → `anomaly_detector.py` → `schema.py` → `logger_factory.py` → `config.py`
`pipeline.py` → `recurring_detector.py` → `config.py`
**[HIGH RISK]** `pipeline.py` → `recurring_detector.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `pipeline.py` → `recurring_detector.py` → `log_utils.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `pipeline.py` → `recurring_detector.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `pipeline.py` → `insight_model.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `pipeline.py` → `insight_generator.py` → `contracts.py` → `config.py`
**[HIGH RISK]** `pipeline.py` → `insight_generator.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `pipeline.py` → `known_persons.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `pipeline.py` → `known_persons.py` → `schema.py` → `logger_factory.py` → `config.py`
`pipeline.py` → `known_persons.py` → `config.py`
**[HIGH RISK]** `pipeline.py` → `log_utils.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `pipeline.py` → `passion_pipeline.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `pipeline.py` → `passion_pipeline.py` → `config_passion.py` → `config.py`
**[HIGH RISK]** `pipeline.py` → `passion_pipeline.py` → `passion_utils.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `pipeline.py` → `passion_pipeline.py` → `candidate.py` → `passion_models.py`
`pipeline.py` → `passion_pipeline.py` → `banned_content.py`
**[HIGH RISK]** `pipeline.py` → `passion_pipeline.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `pipeline.py` → `passion_pipeline.py` → `pipeline_result.py` → `candidate.py` → `passion_models.py`
**[HIGH RISK]** `pipeline.py` → `passion_pipeline.py` → `pipeline_result.py` → `passion_models.py`
**[HIGH RISK]** `pipeline.py` → `passion_pipeline.py` → `bootstrap.py` → `contracts.py` → `config.py`
**[HIGH RISK]** `pipeline.py` → `passion_pipeline.py` → `bootstrap.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `pipeline.py` → `passion_pipeline.py` → `bootstrap.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `pipeline.py` → `passion_pipeline.py` → `bootstrap.py` → `config_passion.py` → `config.py`
**[HIGH RISK]** `pipeline.py` → `passion_pipeline.py` → `bootstrap.py` → `log_utils.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `pipeline.py` → `passion_pipeline.py` → `marketplace_subcategory.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `pipeline.py` → `passion_pipeline.py` → `marketplace_subcategory.py` → `config_passion.py` → `config.py`
**[HIGH RISK]** `pipeline.py` → `passion_pipeline.py` → `marketplace_subcategory.py` → `passion_utils.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `pipeline.py` → `passion_pipeline.py` → `marketplace_subcategory.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `pipeline.py` → `passion_pipeline.py` → `passion_detector.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `pipeline.py` → `passion_pipeline.py` → `passion_detector.py` → `config_passion.py` → `config.py`
**[HIGH RISK]** `pipeline.py` → `passion_pipeline.py` → `passion_detector.py` → `passion_utils.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `pipeline.py` → `passion_pipeline.py` → `passion_detector.py` → `marketplace_subcategory.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `pipeline.py` → `passion_pipeline.py` → `passion_detector.py` → `marketplace_subcategory.py` → `config_passion.py` → `config.py`
**[HIGH RISK]** `pipeline.py` → `passion_pipeline.py` → `passion_detector.py` → `marketplace_subcategory.py` → `passion_utils.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `pipeline.py` → `passion_pipeline.py` → `passion_detector.py` → `marketplace_subcategory.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `pipeline.py` → `passion_pipeline.py` → `passion_detector.py` → `passion_models.py`
**[HIGH RISK]** `pipeline.py` → `passion_pipeline.py` → `passion_detector.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `pipeline.py` → `passion_pipeline.py` → `passion_insight_generator.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `pipeline.py` → `passion_pipeline.py` → `passion_insight_generator.py` → `contracts.py` → `config.py`
**[HIGH RISK]** `pipeline.py` → `passion_pipeline.py` → `passion_insight_generator.py` → `config_passion.py` → `config.py`
**[HIGH RISK]** `pipeline.py` → `passion_pipeline.py` → `passion_insight_generator.py` → `passion_models.py`
**[HIGH RISK]** `pipeline.py` → `passion_pipeline.py` → `passion_insight_generator.py` → `passion_utils.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `pipeline.py` → `passion_pipeline.py` → `passion_insight_generator.py` → `banned_content.py`
**[HIGH RISK]** `pipeline.py` → `passion_pipeline.py` → `passion_insight_generator.py` → `logger_factory.py` → `config.py`
`pipeline_result.py` → `candidate.py` → `passion_models.py`
**[HIGH RISK]** `preprocessor.py` → `schema.py` → `logger_factory.py` → `config.py`
`preprocessor.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `recurring_detector.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `recurring_detector.py` → `log_utils.py` → `logger_factory.py` → `config.py`
`recurring_detector.py` → `logger_factory.py` → `config.py`
`schema.py` → `logger_factory.py` → `config.py`
`seed_labeler.py` → `preprocessor.py` → `config.py`
**[HIGH RISK]** `seed_labeler.py` → `preprocessor.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `seed_labeler.py` → `preprocessor.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `seed_labeler.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `summary_utils.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/conftest.py` → `passion_pipeline.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/conftest.py` → `passion_pipeline.py` → `config_passion.py` → `config.py`
**[HIGH RISK]** `tests/conftest.py` → `passion_pipeline.py` → `passion_utils.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/conftest.py` → `passion_pipeline.py` → `candidate.py` → `passion_models.py`
`tests/conftest.py` → `passion_pipeline.py` → `banned_content.py`
**[HIGH RISK]** `tests/conftest.py` → `passion_pipeline.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/conftest.py` → `passion_pipeline.py` → `pipeline_result.py` → `candidate.py` → `passion_models.py`
**[HIGH RISK]** `tests/conftest.py` → `passion_pipeline.py` → `pipeline_result.py` → `passion_models.py`
**[HIGH RISK]** `tests/conftest.py` → `passion_pipeline.py` → `bootstrap.py` → `contracts.py` → `config.py`
**[HIGH RISK]** `tests/conftest.py` → `passion_pipeline.py` → `bootstrap.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/conftest.py` → `passion_pipeline.py` → `bootstrap.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/conftest.py` → `passion_pipeline.py` → `bootstrap.py` → `config_passion.py` → `config.py`
**[HIGH RISK]** `tests/conftest.py` → `passion_pipeline.py` → `bootstrap.py` → `log_utils.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/conftest.py` → `passion_pipeline.py` → `marketplace_subcategory.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/conftest.py` → `passion_pipeline.py` → `marketplace_subcategory.py` → `config_passion.py` → `config.py`
**[HIGH RISK]** `tests/conftest.py` → `passion_pipeline.py` → `marketplace_subcategory.py` → `passion_utils.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/conftest.py` → `passion_pipeline.py` → `marketplace_subcategory.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/conftest.py` → `passion_pipeline.py` → `passion_detector.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/conftest.py` → `passion_pipeline.py` → `passion_detector.py` → `config_passion.py` → `config.py`
**[HIGH RISK]** `tests/conftest.py` → `passion_pipeline.py` → `passion_detector.py` → `passion_utils.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/conftest.py` → `passion_pipeline.py` → `passion_detector.py` → `marketplace_subcategory.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/conftest.py` → `passion_pipeline.py` → `passion_detector.py` → `marketplace_subcategory.py` → `config_passion.py` → `config.py`
**[HIGH RISK]** `tests/conftest.py` → `passion_pipeline.py` → `passion_detector.py` → `marketplace_subcategory.py` → `passion_utils.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/conftest.py` → `passion_pipeline.py` → `passion_detector.py` → `marketplace_subcategory.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/conftest.py` → `passion_pipeline.py` → `passion_detector.py` → `passion_models.py`
**[HIGH RISK]** `tests/conftest.py` → `passion_pipeline.py` → `passion_detector.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/conftest.py` → `passion_pipeline.py` → `passion_insight_generator.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/conftest.py` → `passion_pipeline.py` → `passion_insight_generator.py` → `contracts.py` → `config.py`
**[HIGH RISK]** `tests/conftest.py` → `passion_pipeline.py` → `passion_insight_generator.py` → `config_passion.py` → `config.py`
**[HIGH RISK]** `tests/conftest.py` → `passion_pipeline.py` → `passion_insight_generator.py` → `passion_models.py`
**[HIGH RISK]** `tests/conftest.py` → `passion_pipeline.py` → `passion_insight_generator.py` → `passion_utils.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/conftest.py` → `passion_pipeline.py` → `passion_insight_generator.py` → `banned_content.py`
**[HIGH RISK]** `tests/conftest.py` → `passion_pipeline.py` → `passion_insight_generator.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/conftest.py` → `log_utils.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/run_smoke.py` → `pipeline.py` → `logger_factory.py` → `config.py`
`tests/run_smoke.py` → `pipeline.py` → `model_state.py`
`tests/run_smoke.py` → `pipeline.py` → `config.py`
**[HIGH RISK]** `tests/run_smoke.py` → `pipeline.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/run_smoke.py` → `pipeline.py` → `preprocessor.py` → `config.py`
**[HIGH RISK]** `tests/run_smoke.py` → `pipeline.py` → `preprocessor.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/run_smoke.py` → `pipeline.py` → `preprocessor.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/run_smoke.py` → `pipeline.py` → `feature_engineer.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/run_smoke.py` → `pipeline.py` → `seed_labeler.py` → `config.py`
**[HIGH RISK]** `tests/run_smoke.py` → `pipeline.py` → `seed_labeler.py` → `preprocessor.py` → `config.py`
**[HIGH RISK]** `tests/run_smoke.py` → `pipeline.py` → `seed_labeler.py` → `preprocessor.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/run_smoke.py` → `pipeline.py` → `seed_labeler.py` → `preprocessor.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/run_smoke.py` → `pipeline.py` → `seed_labeler.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/run_smoke.py` → `pipeline.py` → `categorization_model.py` → `config.py`
**[HIGH RISK]** `tests/run_smoke.py` → `pipeline.py` → `categorization_model.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/run_smoke.py` → `pipeline.py` → `expected_spend_model.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/run_smoke.py` → `pipeline.py` → `anomaly_detector.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/run_smoke.py` → `pipeline.py` → `recurring_detector.py` → `config.py`
**[HIGH RISK]** `tests/run_smoke.py` → `pipeline.py` → `recurring_detector.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/run_smoke.py` → `pipeline.py` → `recurring_detector.py` → `log_utils.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/run_smoke.py` → `pipeline.py` → `recurring_detector.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/run_smoke.py` → `pipeline.py` → `insight_model.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/run_smoke.py` → `pipeline.py` → `insight_generator.py` → `contracts.py` → `config.py`
**[HIGH RISK]** `tests/run_smoke.py` → `pipeline.py` → `insight_generator.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/run_smoke.py` → `pipeline.py` → `known_persons.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/run_smoke.py` → `pipeline.py` → `known_persons.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/run_smoke.py` → `pipeline.py` → `known_persons.py` → `config.py`
**[HIGH RISK]** `tests/run_smoke.py` → `pipeline.py` → `log_utils.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/run_smoke.py` → `pipeline.py` → `passion_pipeline.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/run_smoke.py` → `pipeline.py` → `passion_pipeline.py` → `config_passion.py` → `config.py`
**[HIGH RISK]** `tests/run_smoke.py` → `pipeline.py` → `passion_pipeline.py` → `passion_utils.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/run_smoke.py` → `pipeline.py` → `passion_pipeline.py` → `candidate.py` → `passion_models.py`
**[HIGH RISK]** `tests/run_smoke.py` → `pipeline.py` → `passion_pipeline.py` → `banned_content.py`
**[HIGH RISK]** `tests/run_smoke.py` → `pipeline.py` → `passion_pipeline.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/run_smoke.py` → `pipeline.py` → `passion_pipeline.py` → `pipeline_result.py` → `candidate.py` → `passion_models.py`
**[HIGH RISK]** `tests/run_smoke.py` → `pipeline.py` → `passion_pipeline.py` → `pipeline_result.py` → `passion_models.py`
**[HIGH RISK]** `tests/run_smoke.py` → `pipeline.py` → `passion_pipeline.py` → `bootstrap.py` → `contracts.py` → `config.py`
**[HIGH RISK]** `tests/run_smoke.py` → `pipeline.py` → `passion_pipeline.py` → `bootstrap.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/run_smoke.py` → `pipeline.py` → `passion_pipeline.py` → `bootstrap.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/run_smoke.py` → `pipeline.py` → `passion_pipeline.py` → `bootstrap.py` → `config_passion.py` → `config.py`
**[HIGH RISK]** `tests/run_smoke.py` → `pipeline.py` → `passion_pipeline.py` → `bootstrap.py` → `log_utils.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/run_smoke.py` → `pipeline.py` → `passion_pipeline.py` → `marketplace_subcategory.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/run_smoke.py` → `pipeline.py` → `passion_pipeline.py` → `marketplace_subcategory.py` → `config_passion.py` → `config.py`
**[HIGH RISK]** `tests/run_smoke.py` → `pipeline.py` → `passion_pipeline.py` → `marketplace_subcategory.py` → `passion_utils.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/run_smoke.py` → `pipeline.py` → `passion_pipeline.py` → `marketplace_subcategory.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/run_smoke.py` → `pipeline.py` → `passion_pipeline.py` → `passion_detector.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/run_smoke.py` → `pipeline.py` → `passion_pipeline.py` → `passion_detector.py` → `config_passion.py` → `config.py`
**[HIGH RISK]** `tests/run_smoke.py` → `pipeline.py` → `passion_pipeline.py` → `passion_detector.py` → `passion_utils.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/run_smoke.py` → `pipeline.py` → `passion_pipeline.py` → `passion_detector.py` → `marketplace_subcategory.py` → `config_passion.py` → `config.py`
**[HIGH RISK]** `tests/run_smoke.py` → `pipeline.py` → `passion_pipeline.py` → `passion_detector.py` → `marketplace_subcategory.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/run_smoke.py` → `pipeline.py` → `passion_pipeline.py` → `passion_detector.py` → `passion_models.py`
**[HIGH RISK]** `tests/run_smoke.py` → `pipeline.py` → `passion_pipeline.py` → `passion_detector.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/run_smoke.py` → `pipeline.py` → `passion_pipeline.py` → `passion_insight_generator.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/run_smoke.py` → `pipeline.py` → `passion_pipeline.py` → `passion_insight_generator.py` → `contracts.py` → `config.py`
**[HIGH RISK]** `tests/run_smoke.py` → `pipeline.py` → `passion_pipeline.py` → `passion_insight_generator.py` → `config_passion.py` → `config.py`
**[HIGH RISK]** `tests/run_smoke.py` → `pipeline.py` → `passion_pipeline.py` → `passion_insight_generator.py` → `passion_models.py`
**[HIGH RISK]** `tests/run_smoke.py` → `pipeline.py` → `passion_pipeline.py` → `passion_insight_generator.py` → `passion_utils.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/run_smoke.py` → `pipeline.py` → `passion_pipeline.py` → `passion_insight_generator.py` → `banned_content.py`
**[HIGH RISK]** `tests/run_smoke.py` → `pipeline.py` → `passion_pipeline.py` → `passion_insight_generator.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/run_stress_heavy.py` → `pipeline.py` → `logger_factory.py` → `config.py`
`tests/run_stress_heavy.py` → `pipeline.py` → `model_state.py`
`tests/run_stress_heavy.py` → `pipeline.py` → `config.py`
**[HIGH RISK]** `tests/run_stress_heavy.py` → `pipeline.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/run_stress_heavy.py` → `pipeline.py` → `preprocessor.py` → `config.py`
**[HIGH RISK]** `tests/run_stress_heavy.py` → `pipeline.py` → `preprocessor.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/run_stress_heavy.py` → `pipeline.py` → `preprocessor.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/run_stress_heavy.py` → `pipeline.py` → `feature_engineer.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/run_stress_heavy.py` → `pipeline.py` → `seed_labeler.py` → `config.py`
**[HIGH RISK]** `tests/run_stress_heavy.py` → `pipeline.py` → `seed_labeler.py` → `preprocessor.py` → `config.py`
**[HIGH RISK]** `tests/run_stress_heavy.py` → `pipeline.py` → `seed_labeler.py` → `preprocessor.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/run_stress_heavy.py` → `pipeline.py` → `seed_labeler.py` → `preprocessor.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/run_stress_heavy.py` → `pipeline.py` → `seed_labeler.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/run_stress_heavy.py` → `pipeline.py` → `categorization_model.py` → `config.py`
**[HIGH RISK]** `tests/run_stress_heavy.py` → `pipeline.py` → `categorization_model.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/run_stress_heavy.py` → `pipeline.py` → `expected_spend_model.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/run_stress_heavy.py` → `pipeline.py` → `anomaly_detector.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/run_stress_heavy.py` → `pipeline.py` → `recurring_detector.py` → `config.py`
**[HIGH RISK]** `tests/run_stress_heavy.py` → `pipeline.py` → `recurring_detector.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/run_stress_heavy.py` → `pipeline.py` → `recurring_detector.py` → `log_utils.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/run_stress_heavy.py` → `pipeline.py` → `recurring_detector.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/run_stress_heavy.py` → `pipeline.py` → `insight_model.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/run_stress_heavy.py` → `pipeline.py` → `insight_generator.py` → `contracts.py` → `config.py`
**[HIGH RISK]** `tests/run_stress_heavy.py` → `pipeline.py` → `insight_generator.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/run_stress_heavy.py` → `pipeline.py` → `known_persons.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/run_stress_heavy.py` → `pipeline.py` → `known_persons.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/run_stress_heavy.py` → `pipeline.py` → `known_persons.py` → `config.py`
**[HIGH RISK]** `tests/run_stress_heavy.py` → `pipeline.py` → `log_utils.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/run_stress_heavy.py` → `pipeline.py` → `passion_pipeline.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/run_stress_heavy.py` → `pipeline.py` → `passion_pipeline.py` → `config_passion.py` → `config.py`
**[HIGH RISK]** `tests/run_stress_heavy.py` → `pipeline.py` → `passion_pipeline.py` → `passion_utils.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/run_stress_heavy.py` → `pipeline.py` → `passion_pipeline.py` → `candidate.py` → `passion_models.py`
**[HIGH RISK]** `tests/run_stress_heavy.py` → `pipeline.py` → `passion_pipeline.py` → `banned_content.py`
**[HIGH RISK]** `tests/run_stress_heavy.py` → `pipeline.py` → `passion_pipeline.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/run_stress_heavy.py` → `pipeline.py` → `passion_pipeline.py` → `pipeline_result.py` → `candidate.py` → `passion_models.py`
**[HIGH RISK]** `tests/run_stress_heavy.py` → `pipeline.py` → `passion_pipeline.py` → `pipeline_result.py` → `passion_models.py`
**[HIGH RISK]** `tests/run_stress_heavy.py` → `pipeline.py` → `passion_pipeline.py` → `bootstrap.py` → `contracts.py` → `config.py`
**[HIGH RISK]** `tests/run_stress_heavy.py` → `pipeline.py` → `passion_pipeline.py` → `bootstrap.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/run_stress_heavy.py` → `pipeline.py` → `passion_pipeline.py` → `bootstrap.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/run_stress_heavy.py` → `pipeline.py` → `passion_pipeline.py` → `bootstrap.py` → `config_passion.py` → `config.py`
**[HIGH RISK]** `tests/run_stress_heavy.py` → `pipeline.py` → `passion_pipeline.py` → `bootstrap.py` → `log_utils.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/run_stress_heavy.py` → `pipeline.py` → `passion_pipeline.py` → `marketplace_subcategory.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/run_stress_heavy.py` → `pipeline.py` → `passion_pipeline.py` → `marketplace_subcategory.py` → `config_passion.py` → `config.py`
**[HIGH RISK]** `tests/run_stress_heavy.py` → `pipeline.py` → `passion_pipeline.py` → `marketplace_subcategory.py` → `passion_utils.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/run_stress_heavy.py` → `pipeline.py` → `passion_pipeline.py` → `marketplace_subcategory.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/run_stress_heavy.py` → `pipeline.py` → `passion_pipeline.py` → `passion_detector.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/run_stress_heavy.py` → `pipeline.py` → `passion_pipeline.py` → `passion_detector.py` → `config_passion.py` → `config.py`
**[HIGH RISK]** `tests/run_stress_heavy.py` → `pipeline.py` → `passion_pipeline.py` → `passion_detector.py` → `passion_utils.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/run_stress_heavy.py` → `pipeline.py` → `passion_pipeline.py` → `passion_detector.py` → `marketplace_subcategory.py` → `config_passion.py` → `config.py`
**[HIGH RISK]** `tests/run_stress_heavy.py` → `pipeline.py` → `passion_pipeline.py` → `passion_detector.py` → `marketplace_subcategory.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/run_stress_heavy.py` → `pipeline.py` → `passion_pipeline.py` → `passion_detector.py` → `passion_models.py`
**[HIGH RISK]** `tests/run_stress_heavy.py` → `pipeline.py` → `passion_pipeline.py` → `passion_detector.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/run_stress_heavy.py` → `pipeline.py` → `passion_pipeline.py` → `passion_insight_generator.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/run_stress_heavy.py` → `pipeline.py` → `passion_pipeline.py` → `passion_insight_generator.py` → `contracts.py` → `config.py`
**[HIGH RISK]** `tests/run_stress_heavy.py` → `pipeline.py` → `passion_pipeline.py` → `passion_insight_generator.py` → `config_passion.py` → `config.py`
**[HIGH RISK]** `tests/run_stress_heavy.py` → `pipeline.py` → `passion_pipeline.py` → `passion_insight_generator.py` → `passion_models.py`
**[HIGH RISK]** `tests/run_stress_heavy.py` → `pipeline.py` → `passion_pipeline.py` → `passion_insight_generator.py` → `passion_utils.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/run_stress_heavy.py` → `pipeline.py` → `passion_pipeline.py` → `passion_insight_generator.py` → `banned_content.py`
**[HIGH RISK]** `tests/run_stress_heavy.py` → `pipeline.py` → `passion_pipeline.py` → `passion_insight_generator.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/run_stress_legacy.py` → `pipeline.py` → `logger_factory.py` → `config.py`
`tests/run_stress_legacy.py` → `pipeline.py` → `model_state.py`
`tests/run_stress_legacy.py` → `pipeline.py` → `config.py`
**[HIGH RISK]** `tests/run_stress_legacy.py` → `pipeline.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/run_stress_legacy.py` → `pipeline.py` → `preprocessor.py` → `config.py`
**[HIGH RISK]** `tests/run_stress_legacy.py` → `pipeline.py` → `preprocessor.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/run_stress_legacy.py` → `pipeline.py` → `preprocessor.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/run_stress_legacy.py` → `pipeline.py` → `feature_engineer.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/run_stress_legacy.py` → `pipeline.py` → `seed_labeler.py` → `config.py`
**[HIGH RISK]** `tests/run_stress_legacy.py` → `pipeline.py` → `seed_labeler.py` → `preprocessor.py` → `config.py`
**[HIGH RISK]** `tests/run_stress_legacy.py` → `pipeline.py` → `seed_labeler.py` → `preprocessor.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/run_stress_legacy.py` → `pipeline.py` → `seed_labeler.py` → `preprocessor.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/run_stress_legacy.py` → `pipeline.py` → `seed_labeler.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/run_stress_legacy.py` → `pipeline.py` → `categorization_model.py` → `config.py`
**[HIGH RISK]** `tests/run_stress_legacy.py` → `pipeline.py` → `categorization_model.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/run_stress_legacy.py` → `pipeline.py` → `expected_spend_model.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/run_stress_legacy.py` → `pipeline.py` → `anomaly_detector.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/run_stress_legacy.py` → `pipeline.py` → `recurring_detector.py` → `config.py`
**[HIGH RISK]** `tests/run_stress_legacy.py` → `pipeline.py` → `recurring_detector.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/run_stress_legacy.py` → `pipeline.py` → `recurring_detector.py` → `log_utils.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/run_stress_legacy.py` → `pipeline.py` → `recurring_detector.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/run_stress_legacy.py` → `pipeline.py` → `insight_model.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/run_stress_legacy.py` → `pipeline.py` → `insight_generator.py` → `contracts.py` → `config.py`
**[HIGH RISK]** `tests/run_stress_legacy.py` → `pipeline.py` → `insight_generator.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/run_stress_legacy.py` → `pipeline.py` → `known_persons.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/run_stress_legacy.py` → `pipeline.py` → `known_persons.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/run_stress_legacy.py` → `pipeline.py` → `known_persons.py` → `config.py`
**[HIGH RISK]** `tests/run_stress_legacy.py` → `pipeline.py` → `log_utils.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/run_stress_legacy.py` → `pipeline.py` → `passion_pipeline.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/run_stress_legacy.py` → `pipeline.py` → `passion_pipeline.py` → `config_passion.py` → `config.py`
**[HIGH RISK]** `tests/run_stress_legacy.py` → `pipeline.py` → `passion_pipeline.py` → `passion_utils.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/run_stress_legacy.py` → `pipeline.py` → `passion_pipeline.py` → `candidate.py` → `passion_models.py`
**[HIGH RISK]** `tests/run_stress_legacy.py` → `pipeline.py` → `passion_pipeline.py` → `banned_content.py`
**[HIGH RISK]** `tests/run_stress_legacy.py` → `pipeline.py` → `passion_pipeline.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/run_stress_legacy.py` → `pipeline.py` → `passion_pipeline.py` → `pipeline_result.py` → `candidate.py` → `passion_models.py`
**[HIGH RISK]** `tests/run_stress_legacy.py` → `pipeline.py` → `passion_pipeline.py` → `pipeline_result.py` → `passion_models.py`
**[HIGH RISK]** `tests/run_stress_legacy.py` → `pipeline.py` → `passion_pipeline.py` → `bootstrap.py` → `contracts.py` → `config.py`
**[HIGH RISK]** `tests/run_stress_legacy.py` → `pipeline.py` → `passion_pipeline.py` → `bootstrap.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/run_stress_legacy.py` → `pipeline.py` → `passion_pipeline.py` → `bootstrap.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/run_stress_legacy.py` → `pipeline.py` → `passion_pipeline.py` → `bootstrap.py` → `config_passion.py` → `config.py`
**[HIGH RISK]** `tests/run_stress_legacy.py` → `pipeline.py` → `passion_pipeline.py` → `bootstrap.py` → `log_utils.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/run_stress_legacy.py` → `pipeline.py` → `passion_pipeline.py` → `marketplace_subcategory.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/run_stress_legacy.py` → `pipeline.py` → `passion_pipeline.py` → `marketplace_subcategory.py` → `config_passion.py` → `config.py`
**[HIGH RISK]** `tests/run_stress_legacy.py` → `pipeline.py` → `passion_pipeline.py` → `marketplace_subcategory.py` → `passion_utils.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/run_stress_legacy.py` → `pipeline.py` → `passion_pipeline.py` → `marketplace_subcategory.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/run_stress_legacy.py` → `pipeline.py` → `passion_pipeline.py` → `passion_detector.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/run_stress_legacy.py` → `pipeline.py` → `passion_pipeline.py` → `passion_detector.py` → `config_passion.py` → `config.py`
**[HIGH RISK]** `tests/run_stress_legacy.py` → `pipeline.py` → `passion_pipeline.py` → `passion_detector.py` → `passion_utils.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/run_stress_legacy.py` → `pipeline.py` → `passion_pipeline.py` → `passion_detector.py` → `marketplace_subcategory.py` → `config_passion.py` → `config.py`
**[HIGH RISK]** `tests/run_stress_legacy.py` → `pipeline.py` → `passion_pipeline.py` → `passion_detector.py` → `marketplace_subcategory.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/run_stress_legacy.py` → `pipeline.py` → `passion_pipeline.py` → `passion_detector.py` → `passion_models.py`
**[HIGH RISK]** `tests/run_stress_legacy.py` → `pipeline.py` → `passion_pipeline.py` → `passion_detector.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/run_stress_legacy.py` → `pipeline.py` → `passion_pipeline.py` → `passion_insight_generator.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/run_stress_legacy.py` → `pipeline.py` → `passion_pipeline.py` → `passion_insight_generator.py` → `contracts.py` → `config.py`
**[HIGH RISK]** `tests/run_stress_legacy.py` → `pipeline.py` → `passion_pipeline.py` → `passion_insight_generator.py` → `config_passion.py` → `config.py`
**[HIGH RISK]** `tests/run_stress_legacy.py` → `pipeline.py` → `passion_pipeline.py` → `passion_insight_generator.py` → `passion_models.py`
**[HIGH RISK]** `tests/run_stress_legacy.py` → `pipeline.py` → `passion_pipeline.py` → `passion_insight_generator.py` → `passion_utils.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/run_stress_legacy.py` → `pipeline.py` → `passion_pipeline.py` → `passion_insight_generator.py` → `banned_content.py`
**[HIGH RISK]** `tests/run_stress_legacy.py` → `pipeline.py` → `passion_pipeline.py` → `passion_insight_generator.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/run_stress_legacy.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/run_tests_legacy.py` → `tests/test_phase2.py` → `categorization_model.py` → `config.py`
**[HIGH RISK]** `tests/run_tests_legacy.py` → `tests/test_phase2.py` → `categorization_model.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/run_tests_legacy.py` → `tests/test_phase2.py` → `expected_spend_model.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/run_tests_legacy.py` → `tests/test_phase1.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/run_tests_legacy.py` → `tests/test_phase1.py` → `preprocessor.py` → `config.py`
**[HIGH RISK]** `tests/run_tests_legacy.py` → `tests/test_phase1.py` → `preprocessor.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/run_tests_legacy.py` → `tests/test_phase1.py` → `preprocessor.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/run_tests_legacy.py` → `tests/test_phase1.py` → `feature_engineer.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/run_tests_legacy.py` → `tests/test_phase1.py` → `seed_labeler.py` → `config.py`
**[HIGH RISK]** `tests/run_tests_legacy.py` → `tests/test_phase1.py` → `seed_labeler.py` → `preprocessor.py` → `config.py`
**[HIGH RISK]** `tests/run_tests_legacy.py` → `tests/test_phase1.py` → `seed_labeler.py` → `preprocessor.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/run_tests_legacy.py` → `tests/test_phase1.py` → `seed_labeler.py` → `preprocessor.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/run_tests_legacy.py` → `tests/test_phase1.py` → `seed_labeler.py` → `schema.py` → `logger_factory.py` → `config.py`
`tests/run_tests_legacy.py` → `tests/test_phase1.py` → `config.py`
`tests/test_benchmark.py` → `training_data_generator.py` → `config.py`
**[HIGH RISK]** `tests/test_benchmark.py` → `training_data_generator.py` → `contracts.py` → `config.py`
`tests/test_e2e.py` → `preprocessor.py` → `config.py`
**[HIGH RISK]** `tests/test_e2e.py` → `preprocessor.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_e2e.py` → `preprocessor.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_e2e.py` → `feature_engineer.py` → `schema.py` → `logger_factory.py` → `config.py`
`tests/test_e2e.py` → `seed_labeler.py` → `config.py`
**[HIGH RISK]** `tests/test_e2e.py` → `seed_labeler.py` → `preprocessor.py` → `config.py`
**[HIGH RISK]** `tests/test_e2e.py` → `seed_labeler.py` → `preprocessor.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_e2e.py` → `seed_labeler.py` → `preprocessor.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_e2e.py` → `seed_labeler.py` → `schema.py` → `logger_factory.py` → `config.py`
`tests/test_e2e.py` → `categorization_model.py` → `config.py`
**[HIGH RISK]** `tests/test_e2e.py` → `categorization_model.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_e2e.py` → `expected_spend_model.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_e2e.py` → `anomaly_detector.py` → `schema.py` → `logger_factory.py` → `config.py`
`tests/test_e2e.py` → `recurring_detector.py` → `config.py`


*(Part 2 of 3 - split due to length)*

**[HIGH RISK]** `tests/test_e2e.py` → `recurring_detector.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_e2e.py` → `recurring_detector.py` → `log_utils.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_e2e.py` → `recurring_detector.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_e2e.py` → `insight_generator.py` → `contracts.py` → `config.py`
**[HIGH RISK]** `tests/test_e2e.py` → `insight_generator.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_e2e.py` → `pipeline.py` → `logger_factory.py` → `config.py`
`tests/test_e2e.py` → `pipeline.py` → `model_state.py`
`tests/test_e2e.py` → `pipeline.py` → `config.py`
**[HIGH RISK]** `tests/test_e2e.py` → `pipeline.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_e2e.py` → `pipeline.py` → `preprocessor.py` → `config.py`
**[HIGH RISK]** `tests/test_e2e.py` → `pipeline.py` → `preprocessor.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_e2e.py` → `pipeline.py` → `preprocessor.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_e2e.py` → `pipeline.py` → `feature_engineer.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_e2e.py` → `pipeline.py` → `seed_labeler.py` → `config.py`
**[HIGH RISK]** `tests/test_e2e.py` → `pipeline.py` → `seed_labeler.py` → `preprocessor.py` → `config.py`
**[HIGH RISK]** `tests/test_e2e.py` → `pipeline.py` → `seed_labeler.py` → `preprocessor.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_e2e.py` → `pipeline.py` → `seed_labeler.py` → `preprocessor.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_e2e.py` → `pipeline.py` → `seed_labeler.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_e2e.py` → `pipeline.py` → `categorization_model.py` → `config.py`
**[HIGH RISK]** `tests/test_e2e.py` → `pipeline.py` → `categorization_model.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_e2e.py` → `pipeline.py` → `expected_spend_model.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_e2e.py` → `pipeline.py` → `anomaly_detector.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_e2e.py` → `pipeline.py` → `recurring_detector.py` → `config.py`
**[HIGH RISK]** `tests/test_e2e.py` → `pipeline.py` → `recurring_detector.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_e2e.py` → `pipeline.py` → `recurring_detector.py` → `log_utils.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_e2e.py` → `pipeline.py` → `recurring_detector.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_e2e.py` → `pipeline.py` → `insight_model.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_e2e.py` → `pipeline.py` → `insight_generator.py` → `contracts.py` → `config.py`
**[HIGH RISK]** `tests/test_e2e.py` → `pipeline.py` → `insight_generator.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_e2e.py` → `pipeline.py` → `known_persons.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_e2e.py` → `pipeline.py` → `known_persons.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_e2e.py` → `pipeline.py` → `known_persons.py` → `config.py`
**[HIGH RISK]** `tests/test_e2e.py` → `pipeline.py` → `log_utils.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_e2e.py` → `pipeline.py` → `passion_pipeline.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_e2e.py` → `pipeline.py` → `passion_pipeline.py` → `config_passion.py` → `config.py`
**[HIGH RISK]** `tests/test_e2e.py` → `pipeline.py` → `passion_pipeline.py` → `passion_utils.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_e2e.py` → `pipeline.py` → `passion_pipeline.py` → `candidate.py` → `passion_models.py`
**[HIGH RISK]** `tests/test_e2e.py` → `pipeline.py` → `passion_pipeline.py` → `banned_content.py`
**[HIGH RISK]** `tests/test_e2e.py` → `pipeline.py` → `passion_pipeline.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_e2e.py` → `pipeline.py` → `passion_pipeline.py` → `pipeline_result.py` → `candidate.py` → `passion_models.py`
**[HIGH RISK]** `tests/test_e2e.py` → `pipeline.py` → `passion_pipeline.py` → `pipeline_result.py` → `passion_models.py`
**[HIGH RISK]** `tests/test_e2e.py` → `pipeline.py` → `passion_pipeline.py` → `bootstrap.py` → `contracts.py` → `config.py`
**[HIGH RISK]** `tests/test_e2e.py` → `pipeline.py` → `passion_pipeline.py` → `bootstrap.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_e2e.py` → `pipeline.py` → `passion_pipeline.py` → `bootstrap.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_e2e.py` → `pipeline.py` → `passion_pipeline.py` → `bootstrap.py` → `config_passion.py` → `config.py`
**[HIGH RISK]** `tests/test_e2e.py` → `pipeline.py` → `passion_pipeline.py` → `bootstrap.py` → `log_utils.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_e2e.py` → `pipeline.py` → `passion_pipeline.py` → `marketplace_subcategory.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_e2e.py` → `pipeline.py` → `passion_pipeline.py` → `marketplace_subcategory.py` → `config_passion.py` → `config.py`
**[HIGH RISK]** `tests/test_e2e.py` → `pipeline.py` → `passion_pipeline.py` → `marketplace_subcategory.py` → `passion_utils.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_e2e.py` → `pipeline.py` → `passion_pipeline.py` → `marketplace_subcategory.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_e2e.py` → `pipeline.py` → `passion_pipeline.py` → `passion_detector.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_e2e.py` → `pipeline.py` → `passion_pipeline.py` → `passion_detector.py` → `config_passion.py` → `config.py`
**[HIGH RISK]** `tests/test_e2e.py` → `pipeline.py` → `passion_pipeline.py` → `passion_detector.py` → `passion_utils.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_e2e.py` → `pipeline.py` → `passion_pipeline.py` → `passion_detector.py` → `marketplace_subcategory.py` → `config_passion.py` → `config.py`
**[HIGH RISK]** `tests/test_e2e.py` → `pipeline.py` → `passion_pipeline.py` → `passion_detector.py` → `marketplace_subcategory.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_e2e.py` → `pipeline.py` → `passion_pipeline.py` → `passion_detector.py` → `passion_models.py`
**[HIGH RISK]** `tests/test_e2e.py` → `pipeline.py` → `passion_pipeline.py` → `passion_detector.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_e2e.py` → `pipeline.py` → `passion_pipeline.py` → `passion_insight_generator.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_e2e.py` → `pipeline.py` → `passion_pipeline.py` → `passion_insight_generator.py` → `contracts.py` → `config.py`
**[HIGH RISK]** `tests/test_e2e.py` → `pipeline.py` → `passion_pipeline.py` → `passion_insight_generator.py` → `config_passion.py` → `config.py`
**[HIGH RISK]** `tests/test_e2e.py` → `pipeline.py` → `passion_pipeline.py` → `passion_insight_generator.py` → `passion_models.py`
**[HIGH RISK]** `tests/test_e2e.py` → `pipeline.py` → `passion_pipeline.py` → `passion_insight_generator.py` → `passion_utils.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_e2e.py` → `pipeline.py` → `passion_pipeline.py` → `passion_insight_generator.py` → `banned_content.py`
**[HIGH RISK]** `tests/test_e2e.py` → `pipeline.py` → `passion_pipeline.py` → `passion_insight_generator.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_e2e.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_known_persons.py` → `known_persons.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_known_persons.py` → `known_persons.py` → `schema.py` → `logger_factory.py` → `config.py`
`tests/test_known_persons.py` → `known_persons.py` → `config.py`
**[HIGH RISK]** `tests/test_known_persons.py` → `pipeline.py` → `logger_factory.py` → `config.py`
`tests/test_known_persons.py` → `pipeline.py` → `model_state.py`
`tests/test_known_persons.py` → `pipeline.py` → `config.py`
**[HIGH RISK]** `tests/test_known_persons.py` → `pipeline.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_known_persons.py` → `pipeline.py` → `preprocessor.py` → `config.py`
**[HIGH RISK]** `tests/test_known_persons.py` → `pipeline.py` → `preprocessor.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_known_persons.py` → `pipeline.py` → `preprocessor.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_known_persons.py` → `pipeline.py` → `feature_engineer.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_known_persons.py` → `pipeline.py` → `seed_labeler.py` → `config.py`
**[HIGH RISK]** `tests/test_known_persons.py` → `pipeline.py` → `seed_labeler.py` → `preprocessor.py` → `config.py`
**[HIGH RISK]** `tests/test_known_persons.py` → `pipeline.py` → `seed_labeler.py` → `preprocessor.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_known_persons.py` → `pipeline.py` → `seed_labeler.py` → `preprocessor.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_known_persons.py` → `pipeline.py` → `seed_labeler.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_known_persons.py` → `pipeline.py` → `categorization_model.py` → `config.py`
**[HIGH RISK]** `tests/test_known_persons.py` → `pipeline.py` → `categorization_model.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_known_persons.py` → `pipeline.py` → `expected_spend_model.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_known_persons.py` → `pipeline.py` → `anomaly_detector.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_known_persons.py` → `pipeline.py` → `recurring_detector.py` → `config.py`
**[HIGH RISK]** `tests/test_known_persons.py` → `pipeline.py` → `recurring_detector.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_known_persons.py` → `pipeline.py` → `recurring_detector.py` → `log_utils.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_known_persons.py` → `pipeline.py` → `recurring_detector.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_known_persons.py` → `pipeline.py` → `insight_model.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_known_persons.py` → `pipeline.py` → `insight_generator.py` → `contracts.py` → `config.py`
**[HIGH RISK]** `tests/test_known_persons.py` → `pipeline.py` → `insight_generator.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_known_persons.py` → `pipeline.py` → `known_persons.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_known_persons.py` → `pipeline.py` → `known_persons.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_known_persons.py` → `pipeline.py` → `known_persons.py` → `config.py`
**[HIGH RISK]** `tests/test_known_persons.py` → `pipeline.py` → `log_utils.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_known_persons.py` → `pipeline.py` → `passion_pipeline.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_known_persons.py` → `pipeline.py` → `passion_pipeline.py` → `config_passion.py` → `config.py`
**[HIGH RISK]** `tests/test_known_persons.py` → `pipeline.py` → `passion_pipeline.py` → `passion_utils.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_known_persons.py` → `pipeline.py` → `passion_pipeline.py` → `candidate.py` → `passion_models.py`
**[HIGH RISK]** `tests/test_known_persons.py` → `pipeline.py` → `passion_pipeline.py` → `banned_content.py`
**[HIGH RISK]** `tests/test_known_persons.py` → `pipeline.py` → `passion_pipeline.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_known_persons.py` → `pipeline.py` → `passion_pipeline.py` → `pipeline_result.py` → `candidate.py` → `passion_models.py`
**[HIGH RISK]** `tests/test_known_persons.py` → `pipeline.py` → `passion_pipeline.py` → `pipeline_result.py` → `passion_models.py`
**[HIGH RISK]** `tests/test_known_persons.py` → `pipeline.py` → `passion_pipeline.py` → `bootstrap.py` → `contracts.py` → `config.py`
**[HIGH RISK]** `tests/test_known_persons.py` → `pipeline.py` → `passion_pipeline.py` → `bootstrap.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_known_persons.py` → `pipeline.py` → `passion_pipeline.py` → `bootstrap.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_known_persons.py` → `pipeline.py` → `passion_pipeline.py` → `bootstrap.py` → `config_passion.py` → `config.py`
**[HIGH RISK]** `tests/test_known_persons.py` → `pipeline.py` → `passion_pipeline.py` → `bootstrap.py` → `log_utils.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_known_persons.py` → `pipeline.py` → `passion_pipeline.py` → `marketplace_subcategory.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_known_persons.py` → `pipeline.py` → `passion_pipeline.py` → `marketplace_subcategory.py` → `config_passion.py` → `config.py`
**[HIGH RISK]** `tests/test_known_persons.py` → `pipeline.py` → `passion_pipeline.py` → `marketplace_subcategory.py` → `passion_utils.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_known_persons.py` → `pipeline.py` → `passion_pipeline.py` → `marketplace_subcategory.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_known_persons.py` → `pipeline.py` → `passion_pipeline.py` → `passion_detector.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_known_persons.py` → `pipeline.py` → `passion_pipeline.py` → `passion_detector.py` → `config_passion.py` → `config.py`
**[HIGH RISK]** `tests/test_known_persons.py` → `pipeline.py` → `passion_pipeline.py` → `passion_detector.py` → `passion_utils.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_known_persons.py` → `pipeline.py` → `passion_pipeline.py` → `passion_detector.py` → `marketplace_subcategory.py` → `config_passion.py` → `config.py`
**[HIGH RISK]** `tests/test_known_persons.py` → `pipeline.py` → `passion_pipeline.py` → `passion_detector.py` → `marketplace_subcategory.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_known_persons.py` → `pipeline.py` → `passion_pipeline.py` → `passion_detector.py` → `passion_models.py`
**[HIGH RISK]** `tests/test_known_persons.py` → `pipeline.py` → `passion_pipeline.py` → `passion_detector.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_known_persons.py` → `pipeline.py` → `passion_pipeline.py` → `passion_insight_generator.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_known_persons.py` → `pipeline.py` → `passion_pipeline.py` → `passion_insight_generator.py` → `contracts.py` → `config.py`
**[HIGH RISK]** `tests/test_known_persons.py` → `pipeline.py` → `passion_pipeline.py` → `passion_insight_generator.py` → `config_passion.py` → `config.py`
**[HIGH RISK]** `tests/test_known_persons.py` → `pipeline.py` → `passion_pipeline.py` → `passion_insight_generator.py` → `passion_models.py`
**[HIGH RISK]** `tests/test_known_persons.py` → `pipeline.py` → `passion_pipeline.py` → `passion_insight_generator.py` → `passion_utils.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_known_persons.py` → `pipeline.py` → `passion_pipeline.py` → `passion_insight_generator.py` → `banned_content.py`
**[HIGH RISK]** `tests/test_known_persons.py` → `pipeline.py` → `passion_pipeline.py` → `passion_insight_generator.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_known_persons.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_logging_safety.py` → `pipeline.py` → `logger_factory.py` → `config.py`
`tests/test_logging_safety.py` → `pipeline.py` → `model_state.py`
`tests/test_logging_safety.py` → `pipeline.py` → `config.py`
**[HIGH RISK]** `tests/test_logging_safety.py` → `pipeline.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_logging_safety.py` → `pipeline.py` → `preprocessor.py` → `config.py`
**[HIGH RISK]** `tests/test_logging_safety.py` → `pipeline.py` → `preprocessor.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_logging_safety.py` → `pipeline.py` → `preprocessor.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_logging_safety.py` → `pipeline.py` → `feature_engineer.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_logging_safety.py` → `pipeline.py` → `seed_labeler.py` → `config.py`
**[HIGH RISK]** `tests/test_logging_safety.py` → `pipeline.py` → `seed_labeler.py` → `preprocessor.py` → `config.py`
**[HIGH RISK]** `tests/test_logging_safety.py` → `pipeline.py` → `seed_labeler.py` → `preprocessor.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_logging_safety.py` → `pipeline.py` → `seed_labeler.py` → `preprocessor.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_logging_safety.py` → `pipeline.py` → `seed_labeler.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_logging_safety.py` → `pipeline.py` → `categorization_model.py` → `config.py`
**[HIGH RISK]** `tests/test_logging_safety.py` → `pipeline.py` → `categorization_model.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_logging_safety.py` → `pipeline.py` → `expected_spend_model.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_logging_safety.py` → `pipeline.py` → `anomaly_detector.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_logging_safety.py` → `pipeline.py` → `recurring_detector.py` → `config.py`
**[HIGH RISK]** `tests/test_logging_safety.py` → `pipeline.py` → `recurring_detector.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_logging_safety.py` → `pipeline.py` → `recurring_detector.py` → `log_utils.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_logging_safety.py` → `pipeline.py` → `recurring_detector.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_logging_safety.py` → `pipeline.py` → `insight_model.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_logging_safety.py` → `pipeline.py` → `insight_generator.py` → `contracts.py` → `config.py`
**[HIGH RISK]** `tests/test_logging_safety.py` → `pipeline.py` → `insight_generator.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_logging_safety.py` → `pipeline.py` → `known_persons.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_logging_safety.py` → `pipeline.py` → `known_persons.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_logging_safety.py` → `pipeline.py` → `known_persons.py` → `config.py`
**[HIGH RISK]** `tests/test_logging_safety.py` → `pipeline.py` → `log_utils.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_logging_safety.py` → `pipeline.py` → `passion_pipeline.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_logging_safety.py` → `pipeline.py` → `passion_pipeline.py` → `config_passion.py` → `config.py`
**[HIGH RISK]** `tests/test_logging_safety.py` → `pipeline.py` → `passion_pipeline.py` → `passion_utils.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_logging_safety.py` → `pipeline.py` → `passion_pipeline.py` → `candidate.py` → `passion_models.py`
**[HIGH RISK]** `tests/test_logging_safety.py` → `pipeline.py` → `passion_pipeline.py` → `banned_content.py`
**[HIGH RISK]** `tests/test_logging_safety.py` → `pipeline.py` → `passion_pipeline.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_logging_safety.py` → `pipeline.py` → `passion_pipeline.py` → `pipeline_result.py` → `candidate.py` → `passion_models.py`
**[HIGH RISK]** `tests/test_logging_safety.py` → `pipeline.py` → `passion_pipeline.py` → `pipeline_result.py` → `passion_models.py`
**[HIGH RISK]** `tests/test_logging_safety.py` → `pipeline.py` → `passion_pipeline.py` → `bootstrap.py` → `contracts.py` → `config.py`
**[HIGH RISK]** `tests/test_logging_safety.py` → `pipeline.py` → `passion_pipeline.py` → `bootstrap.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_logging_safety.py` → `pipeline.py` → `passion_pipeline.py` → `bootstrap.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_logging_safety.py` → `pipeline.py` → `passion_pipeline.py` → `bootstrap.py` → `config_passion.py` → `config.py`
**[HIGH RISK]** `tests/test_logging_safety.py` → `pipeline.py` → `passion_pipeline.py` → `bootstrap.py` → `log_utils.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_logging_safety.py` → `pipeline.py` → `passion_pipeline.py` → `marketplace_subcategory.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_logging_safety.py` → `pipeline.py` → `passion_pipeline.py` → `marketplace_subcategory.py` → `config_passion.py` → `config.py`
**[HIGH RISK]** `tests/test_logging_safety.py` → `pipeline.py` → `passion_pipeline.py` → `marketplace_subcategory.py` → `passion_utils.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_logging_safety.py` → `pipeline.py` → `passion_pipeline.py` → `marketplace_subcategory.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_logging_safety.py` → `pipeline.py` → `passion_pipeline.py` → `passion_detector.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_logging_safety.py` → `pipeline.py` → `passion_pipeline.py` → `passion_detector.py` → `config_passion.py` → `config.py`
**[HIGH RISK]** `tests/test_logging_safety.py` → `pipeline.py` → `passion_pipeline.py` → `passion_detector.py` → `passion_utils.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_logging_safety.py` → `pipeline.py` → `passion_pipeline.py` → `passion_detector.py` → `marketplace_subcategory.py` → `config_passion.py` → `config.py`
**[HIGH RISK]** `tests/test_logging_safety.py` → `pipeline.py` → `passion_pipeline.py` → `passion_detector.py` → `marketplace_subcategory.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_logging_safety.py` → `pipeline.py` → `passion_pipeline.py` → `passion_detector.py` → `passion_models.py`
**[HIGH RISK]** `tests/test_logging_safety.py` → `pipeline.py` → `passion_pipeline.py` → `passion_detector.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_logging_safety.py` → `pipeline.py` → `passion_pipeline.py` → `passion_insight_generator.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_logging_safety.py` → `pipeline.py` → `passion_pipeline.py` → `passion_insight_generator.py` → `contracts.py` → `config.py`
**[HIGH RISK]** `tests/test_logging_safety.py` → `pipeline.py` → `passion_pipeline.py` → `passion_insight_generator.py` → `config_passion.py` → `config.py`
**[HIGH RISK]** `tests/test_logging_safety.py` → `pipeline.py` → `passion_pipeline.py` → `passion_insight_generator.py` → `passion_models.py`
**[HIGH RISK]** `tests/test_logging_safety.py` → `pipeline.py` → `passion_pipeline.py` → `passion_insight_generator.py` → `passion_utils.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_logging_safety.py` → `pipeline.py` → `passion_pipeline.py` → `passion_insight_generator.py` → `banned_content.py`
**[HIGH RISK]** `tests/test_logging_safety.py` → `pipeline.py` → `passion_pipeline.py` → `passion_insight_generator.py` → `logger_factory.py` → `config.py`
`tests/test_logging_safety.py` → `recurring_detector.py` → `config.py`
**[HIGH RISK]** `tests/test_logging_safety.py` → `recurring_detector.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_logging_safety.py` → `recurring_detector.py` → `log_utils.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_logging_safety.py` → `recurring_detector.py` → `logger_factory.py` → `config.py`
`tests/test_logging_safety.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_logging_safety.py` → `log_utils.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_logging_safety.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_ml_integration.py` → `insight_model.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_ml_integration.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_model_security.py` → `insight_model.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_passion_engine.py` → `passion_pipeline.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_passion_engine.py` → `passion_pipeline.py` → `config_passion.py` → `config.py`
**[HIGH RISK]** `tests/test_passion_engine.py` → `passion_pipeline.py` → `passion_utils.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_passion_engine.py` → `passion_pipeline.py` → `candidate.py` → `passion_models.py`
`tests/test_passion_engine.py` → `passion_pipeline.py` → `banned_content.py`
**[HIGH RISK]** `tests/test_passion_engine.py` → `passion_pipeline.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_passion_engine.py` → `passion_pipeline.py` → `pipeline_result.py` → `candidate.py` → `passion_models.py`
**[HIGH RISK]** `tests/test_passion_engine.py` → `passion_pipeline.py` → `pipeline_result.py` → `passion_models.py`
**[HIGH RISK]** `tests/test_passion_engine.py` → `passion_pipeline.py` → `bootstrap.py` → `contracts.py` → `config.py`
**[HIGH RISK]** `tests/test_passion_engine.py` → `passion_pipeline.py` → `bootstrap.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_passion_engine.py` → `passion_pipeline.py` → `bootstrap.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_passion_engine.py` → `passion_pipeline.py` → `bootstrap.py` → `config_passion.py` → `config.py`
**[HIGH RISK]** `tests/test_passion_engine.py` → `passion_pipeline.py` → `bootstrap.py` → `log_utils.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_passion_engine.py` → `passion_pipeline.py` → `marketplace_subcategory.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_passion_engine.py` → `passion_pipeline.py` → `marketplace_subcategory.py` → `config_passion.py` → `config.py`
**[HIGH RISK]** `tests/test_passion_engine.py` → `passion_pipeline.py` → `marketplace_subcategory.py` → `passion_utils.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_passion_engine.py` → `passion_pipeline.py` → `marketplace_subcategory.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_passion_engine.py` → `passion_pipeline.py` → `passion_detector.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_passion_engine.py` → `passion_pipeline.py` → `passion_detector.py` → `config_passion.py` → `config.py`
**[HIGH RISK]** `tests/test_passion_engine.py` → `passion_pipeline.py` → `passion_detector.py` → `passion_utils.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_passion_engine.py` → `passion_pipeline.py` → `passion_detector.py` → `marketplace_subcategory.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_passion_engine.py` → `passion_pipeline.py` → `passion_detector.py` → `marketplace_subcategory.py` → `config_passion.py` → `config.py`
**[HIGH RISK]** `tests/test_passion_engine.py` → `passion_pipeline.py` → `passion_detector.py` → `marketplace_subcategory.py` → `passion_utils.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_passion_engine.py` → `passion_pipeline.py` → `passion_detector.py` → `marketplace_subcategory.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_passion_engine.py` → `passion_pipeline.py` → `passion_detector.py` → `passion_models.py`
**[HIGH RISK]** `tests/test_passion_engine.py` → `passion_pipeline.py` → `passion_detector.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_passion_engine.py` → `passion_pipeline.py` → `passion_insight_generator.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_passion_engine.py` → `passion_pipeline.py` → `passion_insight_generator.py` → `contracts.py` → `config.py`
**[HIGH RISK]** `tests/test_passion_engine.py` → `passion_pipeline.py` → `passion_insight_generator.py` → `config_passion.py` → `config.py`
**[HIGH RISK]** `tests/test_passion_engine.py` → `passion_pipeline.py` → `passion_insight_generator.py` → `passion_models.py`
**[HIGH RISK]** `tests/test_passion_engine.py` → `passion_pipeline.py` → `passion_insight_generator.py` → `passion_utils.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_passion_engine.py` → `passion_pipeline.py` → `passion_insight_generator.py` → `banned_content.py`
**[HIGH RISK]** `tests/test_passion_engine.py` → `passion_pipeline.py` → `passion_insight_generator.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_passion_engine.py` → `passion_utils.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_passion_engine.py` → `pipeline_result.py` → `candidate.py` → `passion_models.py`
`tests/test_passion_engine.py` → `pipeline_result.py` → `passion_models.py`
**[HIGH RISK]** `tests/test_passion_engine.py` → `log_utils.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_passion_engine.py` → `passion_detector.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_passion_engine.py` → `passion_detector.py` → `config_passion.py` → `config.py`
**[HIGH RISK]** `tests/test_passion_engine.py` → `passion_detector.py` → `passion_utils.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_passion_engine.py` → `passion_detector.py` → `marketplace_subcategory.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_passion_engine.py` → `passion_detector.py` → `marketplace_subcategory.py` → `config_passion.py` → `config.py`
**[HIGH RISK]** `tests/test_passion_engine.py` → `passion_detector.py` → `marketplace_subcategory.py` → `passion_utils.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_passion_engine.py` → `passion_detector.py` → `marketplace_subcategory.py` → `logger_factory.py` → `config.py`
`tests/test_passion_engine.py` → `passion_detector.py` → `passion_models.py`
**[HIGH RISK]** `tests/test_passion_engine.py` → `passion_detector.py` → `logger_factory.py` → `config.py`
`tests/test_passion_engine.py` → `config_passion.py` → `config.py`
**[HIGH RISK]** `tests/test_passion_engine.py` → `passion_insight_generator.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_passion_engine.py` → `passion_insight_generator.py` → `contracts.py` → `config.py`
**[HIGH RISK]** `tests/test_passion_engine.py` → `passion_insight_generator.py` → `config_passion.py` → `config.py`
`tests/test_passion_engine.py` → `passion_insight_generator.py` → `passion_models.py`
**[HIGH RISK]** `tests/test_passion_engine.py` → `passion_insight_generator.py` → `passion_utils.py` → `logger_factory.py` → `config.py`
`tests/test_passion_engine.py` → `passion_insight_generator.py` → `banned_content.py`
**[HIGH RISK]** `tests/test_passion_engine.py` → `passion_insight_generator.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_passion_engine.py` → `marketplace_subcategory.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_passion_engine.py` → `marketplace_subcategory.py` → `config_passion.py` → `config.py`
**[HIGH RISK]** `tests/test_passion_engine.py` → `marketplace_subcategory.py` → `passion_utils.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_passion_engine.py` → `marketplace_subcategory.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_passion_engine.py` → `pipeline.py` → `logger_factory.py` → `config.py`
`tests/test_passion_engine.py` → `pipeline.py` → `model_state.py`
`tests/test_passion_engine.py` → `pipeline.py` → `config.py`
**[HIGH RISK]** `tests/test_passion_engine.py` → `pipeline.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_passion_engine.py` → `pipeline.py` → `preprocessor.py` → `config.py`
**[HIGH RISK]** `tests/test_passion_engine.py` → `pipeline.py` → `preprocessor.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_passion_engine.py` → `pipeline.py` → `preprocessor.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_passion_engine.py` → `pipeline.py` → `feature_engineer.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_passion_engine.py` → `pipeline.py` → `seed_labeler.py` → `config.py`
**[HIGH RISK]** `tests/test_passion_engine.py` → `pipeline.py` → `seed_labeler.py` → `preprocessor.py` → `config.py`
**[HIGH RISK]** `tests/test_passion_engine.py` → `pipeline.py` → `seed_labeler.py` → `preprocessor.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_passion_engine.py` → `pipeline.py` → `seed_labeler.py` → `preprocessor.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_passion_engine.py` → `pipeline.py` → `seed_labeler.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_passion_engine.py` → `pipeline.py` → `categorization_model.py` → `config.py`
**[HIGH RISK]** `tests/test_passion_engine.py` → `pipeline.py` → `categorization_model.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_passion_engine.py` → `pipeline.py` → `expected_spend_model.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_passion_engine.py` → `pipeline.py` → `anomaly_detector.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_passion_engine.py` → `pipeline.py` → `recurring_detector.py` → `config.py`
**[HIGH RISK]** `tests/test_passion_engine.py` → `pipeline.py` → `recurring_detector.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_passion_engine.py` → `pipeline.py` → `recurring_detector.py` → `log_utils.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_passion_engine.py` → `pipeline.py` → `recurring_detector.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_passion_engine.py` → `pipeline.py` → `insight_model.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_passion_engine.py` → `pipeline.py` → `insight_generator.py` → `contracts.py` → `config.py`
**[HIGH RISK]** `tests/test_passion_engine.py` → `pipeline.py` → `insight_generator.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_passion_engine.py` → `pipeline.py` → `known_persons.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_passion_engine.py` → `pipeline.py` → `known_persons.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_passion_engine.py` → `pipeline.py` → `known_persons.py` → `config.py`
**[HIGH RISK]** `tests/test_passion_engine.py` → `pipeline.py` → `log_utils.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_passion_engine.py` → `pipeline.py` → `passion_pipeline.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_passion_engine.py` → `pipeline.py` → `passion_pipeline.py` → `config_passion.py` → `config.py`
**[HIGH RISK]** `tests/test_passion_engine.py` → `pipeline.py` → `passion_pipeline.py` → `passion_utils.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_passion_engine.py` → `pipeline.py` → `passion_pipeline.py` → `candidate.py` → `passion_models.py`
**[HIGH RISK]** `tests/test_passion_engine.py` → `pipeline.py` → `passion_pipeline.py` → `banned_content.py`
**[HIGH RISK]** `tests/test_passion_engine.py` → `pipeline.py` → `passion_pipeline.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_passion_engine.py` → `pipeline.py` → `passion_pipeline.py` → `pipeline_result.py` → `candidate.py` → `passion_models.py`
**[HIGH RISK]** `tests/test_passion_engine.py` → `pipeline.py` → `passion_pipeline.py` → `pipeline_result.py` → `passion_models.py`
**[HIGH RISK]** `tests/test_passion_engine.py` → `pipeline.py` → `passion_pipeline.py` → `bootstrap.py` → `contracts.py` → `config.py`
**[HIGH RISK]** `tests/test_passion_engine.py` → `pipeline.py` → `passion_pipeline.py` → `bootstrap.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_passion_engine.py` → `pipeline.py` → `passion_pipeline.py` → `bootstrap.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_passion_engine.py` → `pipeline.py` → `passion_pipeline.py` → `bootstrap.py` → `config_passion.py` → `config.py`
**[HIGH RISK]** `tests/test_passion_engine.py` → `pipeline.py` → `passion_pipeline.py` → `bootstrap.py` → `log_utils.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_passion_engine.py` → `pipeline.py` → `passion_pipeline.py` → `marketplace_subcategory.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_passion_engine.py` → `pipeline.py` → `passion_pipeline.py` → `marketplace_subcategory.py` → `config_passion.py` → `config.py`
**[HIGH RISK]** `tests/test_passion_engine.py` → `pipeline.py` → `passion_pipeline.py` → `marketplace_subcategory.py` → `passion_utils.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_passion_engine.py` → `pipeline.py` → `passion_pipeline.py` → `marketplace_subcategory.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_passion_engine.py` → `pipeline.py` → `passion_pipeline.py` → `passion_detector.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_passion_engine.py` → `pipeline.py` → `passion_pipeline.py` → `passion_detector.py` → `config_passion.py` → `config.py`
**[HIGH RISK]** `tests/test_passion_engine.py` → `pipeline.py` → `passion_pipeline.py` → `passion_detector.py` → `passion_utils.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_passion_engine.py` → `pipeline.py` → `passion_pipeline.py` → `passion_detector.py` → `marketplace_subcategory.py` → `config_passion.py` → `config.py`
**[HIGH RISK]** `tests/test_passion_engine.py` → `pipeline.py` → `passion_pipeline.py` → `passion_detector.py` → `marketplace_subcategory.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_passion_engine.py` → `pipeline.py` → `passion_pipeline.py` → `passion_detector.py` → `passion_models.py`
**[HIGH RISK]** `tests/test_passion_engine.py` → `pipeline.py` → `passion_pipeline.py` → `passion_detector.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_passion_engine.py` → `pipeline.py` → `passion_pipeline.py` → `passion_insight_generator.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_passion_engine.py` → `pipeline.py` → `passion_pipeline.py` → `passion_insight_generator.py` → `contracts.py` → `config.py`
**[HIGH RISK]** `tests/test_passion_engine.py` → `pipeline.py` → `passion_pipeline.py` → `passion_insight_generator.py` → `config_passion.py` → `config.py`
**[HIGH RISK]** `tests/test_passion_engine.py` → `pipeline.py` → `passion_pipeline.py` → `passion_insight_generator.py` → `passion_models.py`
**[HIGH RISK]** `tests/test_passion_engine.py` → `pipeline.py` → `passion_pipeline.py` → `passion_insight_generator.py` → `passion_utils.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_passion_engine.py` → `pipeline.py` → `passion_pipeline.py` → `passion_insight_generator.py` → `banned_content.py`
**[HIGH RISK]** `tests/test_passion_engine.py` → `pipeline.py` → `passion_pipeline.py` → `passion_insight_generator.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_passion_engine.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_passion_engine.py` → `bootstrap.py` → `contracts.py` → `config.py`
**[HIGH RISK]** `tests/test_passion_engine.py` → `bootstrap.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_passion_engine.py` → `bootstrap.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_passion_engine.py` → `bootstrap.py` → `config_passion.py` → `config.py`
**[HIGH RISK]** `tests/test_passion_engine.py` → `bootstrap.py` → `log_utils.py` → `logger_factory.py` → `config.py`
`tests/test_passion_engine.py` → `contracts.py` → `config.py`
`tests/test_passion_engine.py` → `candidate.py` → `passion_models.py`
**[HIGH RISK]** `tests/test_phase1.py` → `schema.py` → `logger_factory.py` → `config.py`
`tests/test_phase1.py` → `preprocessor.py` → `config.py`
**[HIGH RISK]** `tests/test_phase1.py` → `preprocessor.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_phase1.py` → `preprocessor.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_phase1.py` → `feature_engineer.py` → `schema.py` → `logger_factory.py` → `config.py`
`tests/test_phase1.py` → `seed_labeler.py` → `config.py`
**[HIGH RISK]** `tests/test_phase1.py` → `seed_labeler.py` → `preprocessor.py` → `config.py`
**[HIGH RISK]** `tests/test_phase1.py` → `seed_labeler.py` → `preprocessor.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_phase1.py` → `seed_labeler.py` → `preprocessor.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_phase1.py` → `seed_labeler.py` → `schema.py` → `logger_factory.py` → `config.py`
`tests/test_phase2.py` → `categorization_model.py` → `config.py`
**[HIGH RISK]** `tests/test_phase2.py` → `categorization_model.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_phase2.py` → `expected_spend_model.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_phase3.py` → `anomaly_detector.py` → `schema.py` → `logger_factory.py` → `config.py`
`tests/test_phase3.py` → `recurring_detector.py` → `config.py`
**[HIGH RISK]** `tests/test_phase3.py` → `recurring_detector.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_phase3.py` → `recurring_detector.py` → `log_utils.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_phase3.py` → `recurring_detector.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_phase3.py` → `insight_generator.py` → `contracts.py` → `config.py`
**[HIGH RISK]** `tests/test_phase3.py` → `insight_generator.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_phase3.py` → `pipeline.py` → `logger_factory.py` → `config.py`
`tests/test_phase3.py` → `pipeline.py` → `model_state.py`
`tests/test_phase3.py` → `pipeline.py` → `config.py`
**[HIGH RISK]** `tests/test_phase3.py` → `pipeline.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_phase3.py` → `pipeline.py` → `preprocessor.py` → `config.py`
**[HIGH RISK]** `tests/test_phase3.py` → `pipeline.py` → `preprocessor.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_phase3.py` → `pipeline.py` → `preprocessor.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_phase3.py` → `pipeline.py` → `feature_engineer.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_phase3.py` → `pipeline.py` → `seed_labeler.py` → `config.py`
**[HIGH RISK]** `tests/test_phase3.py` → `pipeline.py` → `seed_labeler.py` → `preprocessor.py` → `config.py`
**[HIGH RISK]** `tests/test_phase3.py` → `pipeline.py` → `seed_labeler.py` → `preprocessor.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_phase3.py` → `pipeline.py` → `seed_labeler.py` → `preprocessor.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_phase3.py` → `pipeline.py` → `seed_labeler.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_phase3.py` → `pipeline.py` → `categorization_model.py` → `config.py`
**[HIGH RISK]** `tests/test_phase3.py` → `pipeline.py` → `categorization_model.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_phase3.py` → `pipeline.py` → `expected_spend_model.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_phase3.py` → `pipeline.py` → `anomaly_detector.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_phase3.py` → `pipeline.py` → `recurring_detector.py` → `config.py`
**[HIGH RISK]** `tests/test_phase3.py` → `pipeline.py` → `recurring_detector.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_phase3.py` → `pipeline.py` → `recurring_detector.py` → `log_utils.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_phase3.py` → `pipeline.py` → `recurring_detector.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_phase3.py` → `pipeline.py` → `insight_model.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_phase3.py` → `pipeline.py` → `insight_generator.py` → `contracts.py` → `config.py`
**[HIGH RISK]** `tests/test_phase3.py` → `pipeline.py` → `insight_generator.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_phase3.py` → `pipeline.py` → `known_persons.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_phase3.py` → `pipeline.py` → `known_persons.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_phase3.py` → `pipeline.py` → `known_persons.py` → `config.py`
**[HIGH RISK]** `tests/test_phase3.py` → `pipeline.py` → `log_utils.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_phase3.py` → `pipeline.py` → `passion_pipeline.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_phase3.py` → `pipeline.py` → `passion_pipeline.py` → `config_passion.py` → `config.py`
**[HIGH RISK]** `tests/test_phase3.py` → `pipeline.py` → `passion_pipeline.py` → `passion_utils.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_phase3.py` → `pipeline.py` → `passion_pipeline.py` → `candidate.py` → `passion_models.py`
**[HIGH RISK]** `tests/test_phase3.py` → `pipeline.py` → `passion_pipeline.py` → `banned_content.py`
**[HIGH RISK]** `tests/test_phase3.py` → `pipeline.py` → `passion_pipeline.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_phase3.py` → `pipeline.py` → `passion_pipeline.py` → `pipeline_result.py` → `candidate.py` → `passion_models.py`
**[HIGH RISK]** `tests/test_phase3.py` → `pipeline.py` → `passion_pipeline.py` → `pipeline_result.py` → `passion_models.py`
**[HIGH RISK]** `tests/test_phase3.py` → `pipeline.py` → `passion_pipeline.py` → `bootstrap.py` → `contracts.py` → `config.py`
**[HIGH RISK]** `tests/test_phase3.py` → `pipeline.py` → `passion_pipeline.py` → `bootstrap.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_phase3.py` → `pipeline.py` → `passion_pipeline.py` → `bootstrap.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_phase3.py` → `pipeline.py` → `passion_pipeline.py` → `bootstrap.py` → `config_passion.py` → `config.py`
**[HIGH RISK]** `tests/test_phase3.py` → `pipeline.py` → `passion_pipeline.py` → `bootstrap.py` → `log_utils.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_phase3.py` → `pipeline.py` → `passion_pipeline.py` → `marketplace_subcategory.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_phase3.py` → `pipeline.py` → `passion_pipeline.py` → `marketplace_subcategory.py` → `config_passion.py` → `config.py`
**[HIGH RISK]** `tests/test_phase3.py` → `pipeline.py` → `passion_pipeline.py` → `marketplace_subcategory.py` → `passion_utils.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_phase3.py` → `pipeline.py` → `passion_pipeline.py` → `marketplace_subcategory.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_phase3.py` → `pipeline.py` → `passion_pipeline.py` → `passion_detector.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_phase3.py` → `pipeline.py` → `passion_pipeline.py` → `passion_detector.py` → `config_passion.py` → `config.py`
**[HIGH RISK]** `tests/test_phase3.py` → `pipeline.py` → `passion_pipeline.py` → `passion_detector.py` → `passion_utils.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_phase3.py` → `pipeline.py` → `passion_pipeline.py` → `passion_detector.py` → `marketplace_subcategory.py` → `config_passion.py` → `config.py`
**[HIGH RISK]** `tests/test_phase3.py` → `pipeline.py` → `passion_pipeline.py` → `passion_detector.py` → `marketplace_subcategory.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_phase3.py` → `pipeline.py` → `passion_pipeline.py` → `passion_detector.py` → `passion_models.py`
**[HIGH RISK]** `tests/test_phase3.py` → `pipeline.py` → `passion_pipeline.py` → `passion_detector.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_phase3.py` → `pipeline.py` → `passion_pipeline.py` → `passion_insight_generator.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_phase3.py` → `pipeline.py` → `passion_pipeline.py` → `passion_insight_generator.py` → `contracts.py` → `config.py`
**[HIGH RISK]** `tests/test_phase3.py` → `pipeline.py` → `passion_pipeline.py` → `passion_insight_generator.py` → `config_passion.py` → `config.py`
**[HIGH RISK]** `tests/test_phase3.py` → `pipeline.py` → `passion_pipeline.py` → `passion_insight_generator.py` → `passion_models.py`
**[HIGH RISK]** `tests/test_phase3.py` → `pipeline.py` → `passion_pipeline.py` → `passion_insight_generator.py` → `passion_utils.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tests/test_phase3.py` → `pipeline.py` → `passion_pipeline.py` → `passion_insight_generator.py` → `banned_content.py`
**[HIGH RISK]** `tests/test_phase3.py` → `pipeline.py` → `passion_pipeline.py` → `passion_insight_generator.py` → `logger_factory.py` → `config.py`
`train_and_save_models.py` → `training_data_generator.py` → `config.py`
**[HIGH RISK]** `train_and_save_models.py` → `training_data_generator.py` → `contracts.py` → `config.py`
**[HIGH RISK]** `train_and_save_models.py` → `insight_model.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `train_and_save_models.py` → `schema.py` → `logger_factory.py` → `config.py`
`training_data_generator.py` → `contracts.py` → `config.py`
`tutorial_real_data.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tutorial_real_data.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tutorial_real_data.py` → `pipeline.py` → `logger_factory.py` → `config.py`
`tutorial_real_data.py` → `pipeline.py` → `model_state.py`
`tutorial_real_data.py` → `pipeline.py` → `config.py`
**[HIGH RISK]** `tutorial_real_data.py` → `pipeline.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tutorial_real_data.py` → `pipeline.py` → `preprocessor.py` → `config.py`
**[HIGH RISK]** `tutorial_real_data.py` → `pipeline.py` → `preprocessor.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tutorial_real_data.py` → `pipeline.py` → `preprocessor.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tutorial_real_data.py` → `pipeline.py` → `feature_engineer.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tutorial_real_data.py` → `pipeline.py` → `seed_labeler.py` → `config.py`
**[HIGH RISK]** `tutorial_real_data.py` → `pipeline.py` → `seed_labeler.py` → `preprocessor.py` → `config.py`
**[HIGH RISK]** `tutorial_real_data.py` → `pipeline.py` → `seed_labeler.py` → `preprocessor.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tutorial_real_data.py` → `pipeline.py` → `seed_labeler.py` → `preprocessor.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tutorial_real_data.py` → `pipeline.py` → `seed_labeler.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tutorial_real_data.py` → `pipeline.py` → `categorization_model.py` → `config.py`
**[HIGH RISK]** `tutorial_real_data.py` → `pipeline.py` → `categorization_model.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tutorial_real_data.py` → `pipeline.py` → `expected_spend_model.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tutorial_real_data.py` → `pipeline.py` → `anomaly_detector.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tutorial_real_data.py` → `pipeline.py` → `recurring_detector.py` → `config.py`
**[HIGH RISK]** `tutorial_real_data.py` → `pipeline.py` → `recurring_detector.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tutorial_real_data.py` → `pipeline.py` → `recurring_detector.py` → `log_utils.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tutorial_real_data.py` → `pipeline.py` → `recurring_detector.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tutorial_real_data.py` → `pipeline.py` → `insight_model.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tutorial_real_data.py` → `pipeline.py` → `insight_generator.py` → `contracts.py` → `config.py`
**[HIGH RISK]** `tutorial_real_data.py` → `pipeline.py` → `insight_generator.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tutorial_real_data.py` → `pipeline.py` → `known_persons.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tutorial_real_data.py` → `pipeline.py` → `known_persons.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tutorial_real_data.py` → `pipeline.py` → `known_persons.py` → `config.py`
**[HIGH RISK]** `tutorial_real_data.py` → `pipeline.py` → `log_utils.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tutorial_real_data.py` → `pipeline.py` → `passion_pipeline.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tutorial_real_data.py` → `pipeline.py` → `passion_pipeline.py` → `config_passion.py` → `config.py`
**[HIGH RISK]** `tutorial_real_data.py` → `pipeline.py` → `passion_pipeline.py` → `passion_utils.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tutorial_real_data.py` → `pipeline.py` → `passion_pipeline.py` → `candidate.py` → `passion_models.py`
**[HIGH RISK]** `tutorial_real_data.py` → `pipeline.py` → `passion_pipeline.py` → `banned_content.py`
**[HIGH RISK]** `tutorial_real_data.py` → `pipeline.py` → `passion_pipeline.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tutorial_real_data.py` → `pipeline.py` → `passion_pipeline.py` → `pipeline_result.py` → `candidate.py` → `passion_models.py`
**[HIGH RISK]** `tutorial_real_data.py` → `pipeline.py` → `passion_pipeline.py` → `pipeline_result.py` → `passion_models.py`
**[HIGH RISK]** `tutorial_real_data.py` → `pipeline.py` → `passion_pipeline.py` → `bootstrap.py` → `contracts.py` → `config.py`
**[HIGH RISK]** `tutorial_real_data.py` → `pipeline.py` → `passion_pipeline.py` → `bootstrap.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tutorial_real_data.py` → `pipeline.py` → `passion_pipeline.py` → `bootstrap.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tutorial_real_data.py` → `pipeline.py` → `passion_pipeline.py` → `bootstrap.py` → `config_passion.py` → `config.py`
**[HIGH RISK]** `tutorial_real_data.py` → `pipeline.py` → `passion_pipeline.py` → `bootstrap.py` → `log_utils.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tutorial_real_data.py` → `pipeline.py` → `passion_pipeline.py` → `marketplace_subcategory.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tutorial_real_data.py` → `pipeline.py` → `passion_pipeline.py` → `marketplace_subcategory.py` → `config_passion.py` → `config.py`
**[HIGH RISK]** `tutorial_real_data.py` → `pipeline.py` → `passion_pipeline.py` → `marketplace_subcategory.py` → `passion_utils.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tutorial_real_data.py` → `pipeline.py` → `passion_pipeline.py` → `marketplace_subcategory.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tutorial_real_data.py` → `pipeline.py` → `passion_pipeline.py` → `passion_detector.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tutorial_real_data.py` → `pipeline.py` → `passion_pipeline.py` → `passion_detector.py` → `config_passion.py` → `config.py`
**[HIGH RISK]** `tutorial_real_data.py` → `pipeline.py` → `passion_pipeline.py` → `passion_detector.py` → `passion_utils.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tutorial_real_data.py` → `pipeline.py` → `passion_pipeline.py` → `passion_detector.py` → `marketplace_subcategory.py` → `config_passion.py` → `config.py`
**[HIGH RISK]** `tutorial_real_data.py` → `pipeline.py` → `passion_pipeline.py` → `passion_detector.py` → `marketplace_subcategory.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tutorial_real_data.py` → `pipeline.py` → `passion_pipeline.py` → `passion_detector.py` → `passion_models.py`
**[HIGH RISK]** `tutorial_real_data.py` → `pipeline.py` → `passion_pipeline.py` → `passion_detector.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tutorial_real_data.py` → `pipeline.py` → `passion_pipeline.py` → `passion_insight_generator.py` → `schema.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tutorial_real_data.py` → `pipeline.py` → `passion_pipeline.py` → `passion_insight_generator.py` → `contracts.py` → `config.py`
**[HIGH RISK]** `tutorial_real_data.py` → `pipeline.py` → `passion_pipeline.py` → `passion_insight_generator.py` → `config_passion.py` → `config.py`
**[HIGH RISK]** `tutorial_real_data.py` → `pipeline.py` → `passion_pipeline.py` → `passion_insight_generator.py` → `passion_models.py`
**[HIGH RISK]** `tutorial_real_data.py` → `pipeline.py` → `passion_pipeline.py` → `passion_insight_generator.py` → `passion_utils.py` → `logger_factory.py` → `config.py`
**[HIGH RISK]** `tutorial_real_data.py` → `pipeline.py` → `passion_pipeline.py` → `passion_insight_generator.py` → `banned_content.py`
**[HIGH RISK]** `tutorial_real_data.py` → `pipeline.py` → `passion_pipeline.py` → `passion_insight_generator.py` → `logger_factory.py` → `config.py`

## Longest Dependency Chain
`tutorial_real_data.py` → `pipeline.py` → `passion_pipeline.py` → `passion_detector.py` → `marketplace_subcategory.py` → `schema.py` → `logger_factory.py` → `config.py`

*This indicates the deepest coupling path.*

## Circular Dependencies
None detected.

## Hub Files
| File | Imported-By Count | Who Imports It |
|------|------------------|---------------|
| `schema.py` | 27 | marketplace_subcategory.py, tutorial_real_data.py, pipeline.py, recurring_detector.py, anomaly_detector.py, known_persons.py, summary_utils.py, demo.py, bootstrap.py, seed_labeler.py, train_and_save_models.py, insight_generator.py, categorization_model.py, insight_model.py, passion_insight_generator.py, preprocessor.py, feature_engineer.py, expected_spend_model.py, passion_pipeline.py, passion_detector.py, tests/test_ml_integration.py, tests/test_known_persons.py, tests/test_passion_engine.py, tests/test_logging_safety.py, tests/test_phase1.py, tests/test_e2e.py, tests/run_stress_legacy.py |
| `logger_factory.py` | 14 | marketplace_subcategory.py, tutorial_real_data.py, pipeline.py, recurring_detector.py, known_persons.py, bootstrap.py, schema.py, passion_insight_generator.py, preprocessor.py, log_utils.py, passion_utils.py, passion_pipeline.py, passion_detector.py, tests/test_logging_safety.py |
| `config.py` | 14 | pipeline.py, recurring_detector.py, known_persons.py, seed_labeler.py, contracts.py, logger_factory.py, categorization_model.py, training_data_generator.py, preprocessor.py, config_passion.py, tests/test_passion_engine.py, tests/test_benchmark.py, tests/test_logging_safety.py, tests/test_phase1.py |
| `pipeline.py` | 10 | tutorial_real_data.py, demo.py, tests/run_stress_heavy.py, tests/test_phase3.py, tests/test_known_persons.py, tests/test_passion_engine.py, tests/test_logging_safety.py, tests/test_e2e.py, tests/run_smoke.py, tests/run_stress_legacy.py |
| `config_passion.py` | 6 | marketplace_subcategory.py, bootstrap.py, passion_insight_generator.py, passion_pipeline.py, passion_detector.py, tests/test_passion_engine.py |
| `log_utils.py` | 6 | pipeline.py, recurring_detector.py, bootstrap.py, tests/conftest.py, tests/test_passion_engine.py, tests/test_logging_safety.py |
| `passion_utils.py` | 5 | marketplace_subcategory.py, passion_insight_generator.py, passion_pipeline.py, passion_detector.py, tests/test_passion_engine.py |
| `contracts.py` | 5 | bootstrap.py, insight_generator.py, training_data_generator.py, passion_insight_generator.py, tests/test_passion_engine.py |
| `passion_models.py` | 5 | passion_insight_generator.py, candidate.py, pipeline_result.py, passion_detector.py, tests/test_passion_engine.py |

## Leaf Files
- `banned_content.py`
- `config.py`
- `hash_utils.py`
- `model_state.py`
- `passion_models.py`
- `refactor_pipeline.py`

## Mermaid Diagram
```mermaid
graph TD
  subgraph CONFIG
    banned_content_py["banned_content.py"]
    config_py["config.py (Imported by: 14)"]
    config_passion_py["config_passion.py (Imported by: 6)"]
  end
  subgraph CORE_LOGIC
    recurring_detector_py["recurring_detector.py"]
    anomaly_detector_py["anomaly_detector.py"]
    model_benchmark_py["model_benchmark.py"]
    insight_generator_py["insight_generator.py"]
    categorization_model_py["categorization_model.py"]
    training_data_generator_py["training_data_generator.py"]
    insight_model_py["insight_model.py"]
    passion_insight_generator_py["passion_insight_generator.py"]
    preprocessor_py["preprocessor.py"]
    feature_engineer_py["feature_engineer.py"]
    passion_models_py["passion_models.py (Imported by: 5)"]
    expected_spend_model_py["expected_spend_model.py"]
    passion_detector_py["passion_detector.py"]
  end
  subgraph DATA_LAYER
    marketplace_subcategory_py["marketplace_subcategory.py"]
    known_persons_py["known_persons.py"]
    seed_labeler_py["seed_labeler.py"]
    tests_test_known_persons_py["tests/test_known_persons.py"]
  end
  subgraph ENTRYPOINT
    tutorial_real_data_py["tutorial_real_data.py"]
    pipeline_py["pipeline.py (Imported by: 10)"]
    demo_py["demo.py"]
    bootstrap_py["bootstrap.py"]
    train_and_save_models_py["train_and_save_models.py"]
    passion_pipeline_py["passion_pipeline.py"]
  end
  subgraph SCHEMA
    schema_py["schema.py (Imported by: 27)"]
    contracts_py["contracts.py (Imported by: 5)"]
    model_state_py["model_state.py"]
    candidate_py["candidate.py"]
    pipeline_result_py["pipeline_result.py"]
  end
  subgraph TEST
    tests_run_stress_heavy_py["tests/run_stress_heavy.py"]
    tests_test_phase3_py["tests/test_phase3.py"]
    tests_run_tests_legacy_py["tests/run_tests_legacy.py"]
    tests_test_ml_integration_py["tests/test_ml_integration.py"]
    tests_conftest_py["tests/conftest.py"]
    tests_test_phase2_py["tests/test_phase2.py"]
    tests_test_passion_engine_py["tests/test_passion_engine.py"]
    tests_test_benchmark_py["tests/test_benchmark.py"]
    tests_test_model_security_py["tests/test_model_security.py"]
    tests_test_logging_safety_py["tests/test_logging_safety.py"]
    tests_test_phase1_py["tests/test_phase1.py"]
    tests_test_e2e_py["tests/test_e2e.py"]
    tests_run_smoke_py["tests/run_smoke.py"]
    tests_run_stress_legacy_py["tests/run_stress_legacy.py"]
  end
  subgraph UTILITY
    summary_utils_py["summary_utils.py"]
    logger_factory_py["logger_factory.py (Imported by: 14)"]
    hash_utils_py["hash_utils.py"]
    refactor_pipeline_py["refactor_pipeline.py"]
    log_utils_py["log_utils.py (Imported by: 6)"]
    passion_utils_py["passion_utils.py (Imported by: 5)"]
  end
  marketplace_subcategory_py --> schema_py
  marketplace_subcategory_py --> config_passion_py
  marketplace_subcategory_py --> passion_utils_py
  marketplace_subcategory_py --> logger_factory_py
  marketplace_subcategory_py -.-> config_py
  tutorial_real_data_py --> logger_factory_py
  tutorial_real_data_py --> schema_py
  tutorial_real_data_py --> pipeline_py
  tutorial_real_data_py -.-> config_py
  tutorial_real_data_py -.-> model_state_py
  tutorial_real_data_py -.-> preprocessor_py
  tutorial_real_data_py -.-> feature_engineer_py
  tutorial_real_data_py -.-> seed_labeler_py
  tutorial_real_data_py -.-> categorization_model_py
  tutorial_real_data_py -.-> expected_spend_model_py
  tutorial_real_data_py -.-> anomaly_detector_py
  tutorial_real_data_py -.-> recurring_detector_py
  tutorial_real_data_py -.-> log_utils_py
  tutorial_real_data_py -.-> insight_model_py
  tutorial_real_data_py -.-> insight_generator_py
  tutorial_real_data_py -.-> contracts_py
  tutorial_real_data_py -.-> known_persons_py
  tutorial_real_data_py -.-> passion_pipeline_py
  tutorial_real_data_py -.-> config_passion_py
  tutorial_real_data_py -.-> passion_utils_py
  tutorial_real_data_py -.-> candidate_py
  tutorial_real_data_py -.-> passion_models_py
  tutorial_real_data_py -.-> banned_content_py
  tutorial_real_data_py -.-> pipeline_result_py
  tutorial_real_data_py -.-> bootstrap_py
  tutorial_real_data_py -.-> marketplace_subcategory_py
  tutorial_real_data_py -.-> passion_detector_py
  tutorial_real_data_py -.-> passion_insight_generator_py
  pipeline_py --> logger_factory_py
  pipeline_py --> model_state_py
  pipeline_py --> config_py
  pipeline_py --> schema_py
  pipeline_py --> preprocessor_py
  pipeline_py --> feature_engineer_py
  pipeline_py --> seed_labeler_py
  pipeline_py --> categorization_model_py
  pipeline_py --> expected_spend_model_py
  pipeline_py --> anomaly_detector_py
  pipeline_py --> recurring_detector_py
  pipeline_py --> insight_model_py
  pipeline_py --> insight_generator_py
  pipeline_py --> known_persons_py
  pipeline_py --> log_utils_py
  pipeline_py --> passion_pipeline_py
  pipeline_py -.-> contracts_py
  pipeline_py -.-> config_passion_py
  pipeline_py -.-> passion_utils_py
  pipeline_py -.-> candidate_py
  pipeline_py -.-> passion_models_py
  pipeline_py -.-> banned_content_py
  pipeline_py -.-> pipeline_result_py
  pipeline_py -.-> bootstrap_py
  pipeline_py -.-> marketplace_subcategory_py
  pipeline_py -.-> passion_detector_py
  pipeline_py -.-> passion_insight_generator_py
  recurring_detector_py --> config_py
  recurring_detector_py --> schema_py
  recurring_detector_py --> log_utils_py
  recurring_detector_py --> logger_factory_py
  anomaly_detector_py --> schema_py
  anomaly_detector_py -.-> logger_factory_py
  anomaly_detector_py -.-> config_py
  known_persons_py --> logger_factory_py
  known_persons_py --> schema_py
  known_persons_py --> config_py
  summary_utils_py --> schema_py
  summary_utils_py -.-> logger_factory_py
  summary_utils_py -.-> config_py
  demo_py --> schema_py
  demo_py --> pipeline_py
  demo_py --> summary_utils_py
  demo_py -.-> logger_factory_py
  demo_py -.-> config_py
  demo_py -.-> model_state_py
  demo_py -.-> preprocessor_py
  demo_py -.-> feature_engineer_py
  demo_py -.-> seed_labeler_py
  demo_py -.-> categorization_model_py
  demo_py -.-> expected_spend_model_py
  demo_py -.-> anomaly_detector_py
  demo_py -.-> recurring_detector_py
  demo_py -.-> log_utils_py
  demo_py -.-> insight_model_py
  demo_py -.-> insight_generator_py
  demo_py -.-> contracts_py
  demo_py -.-> known_persons_py
  demo_py -.-> passion_pipeline_py
  demo_py -.-> config_passion_py
  demo_py -.-> passion_utils_py
  demo_py -.-> candidate_py
  demo_py -.-> passion_models_py
  demo_py -.-> banned_content_py
  demo_py -.-> pipeline_result_py
  demo_py -.-> bootstrap_py
  demo_py -.-> marketplace_subcategory_py
  demo_py -.-> passion_detector_py
  demo_py -.-> passion_insight_generator_py
  bootstrap_py --> contracts_py
  bootstrap_py --> logger_factory_py
  bootstrap_py --> schema_py
  bootstrap_py --> config_passion_py
  bootstrap_py --> log_utils_py
  bootstrap_py -.-> config_py
  schema_py --> logger_factory_py
  schema_py -.-> config_py
  seed_labeler_py --> config_py
  seed_labeler_py --> preprocessor_py
  seed_labeler_py --> schema_py
  seed_labeler_py -.-> logger_factory_py
  contracts_py --> config_py
  model_benchmark_py --> training_data_generator_py
  model_benchmark_py -.-> config_py
  model_benchmark_py -.-> contracts_py
  train_and_save_models_py --> training_data_generator_py
  train_and_save_models_py --> insight_model_py
  train_and_save_models_py --> schema_py
  train_and_save_models_py -.-> config_py
  train_and_save_models_py -.-> contracts_py
  train_and_save_models_py -.-> logger_factory_py
  logger_factory_py --> config_py
  insight_generator_py --> contracts_py
  insight_generator_py --> schema_py
  insight_generator_py -.-> config_py
  insight_generator_py -.-> logger_factory_py
  categorization_model_py --> config_py
  categorization_model_py --> schema_py
  categorization_model_py -.-> logger_factory_py
  training_data_generator_py --> config_py
  training_data_generator_py --> contracts_py
  insight_model_py --> schema_py
  insight_model_py -.-> logger_factory_py
  insight_model_py -.-> config_py
  passion_insight_generator_py --> schema_py
  passion_insight_generator_py --> contracts_py
  passion_insight_generator_py --> config_passion_py
  passion_insight_generator_py --> passion_models_py
  passion_insight_generator_py --> passion_utils_py
  passion_insight_generator_py --> banned_content_py
  passion_insight_generator_py --> logger_factory_py
  passion_insight_generator_py -.-> config_py
  preprocessor_py --> config_py
  preprocessor_py --> schema_py
  preprocessor_py --> logger_factory_py
  config_passion_py --> config_py
  log_utils_py --> logger_factory_py
  log_utils_py -.-> config_py
  feature_engineer_py --> schema_py
  feature_engineer_py -.-> logger_factory_py
  feature_engineer_py -.-> config_py
  passion_utils_py --> logger_factory_py
  passion_utils_py -.-> config_py
  candidate_py --> passion_models_py
  pipeline_result_py --> candidate_py
  pipeline_result_py --> passion_models_py
  expected_spend_model_py --> schema_py
  expected_spend_model_py -.-> logger_factory_py
  expected_spend_model_py -.-> config_py
  passion_pipeline_py --> schema_py
  passion_pipeline_py --> config_passion_py
  passion_pipeline_py --> passion_utils_py
  passion_pipeline_py --> candidate_py
  passion_pipeline_py --> banned_content_py
  passion_pipeline_py --> logger_factory_py
  passion_pipeline_py --> pipeline_result_py
  passion_pipeline_py --> bootstrap_py
  passion_pipeline_py --> marketplace_subcategory_py
  passion_pipeline_py --> passion_detector_py
  passion_pipeline_py --> passion_insight_generator_py
  passion_pipeline_py -.-> config_py
  passion_pipeline_py -.-> passion_models_py
  passion_pipeline_py -.-> contracts_py
  passion_pipeline_py -.-> log_utils_py
  passion_detector_py --> schema_py
  passion_detector_py --> config_passion_py
  passion_detector_py --> passion_utils_py
  passion_detector_py --> marketplace_subcategory_py
  passion_detector_py --> passion_models_py
  passion_detector_py --> logger_factory_py
  passion_detector_py -.-> config_py
  tests_run_stress_heavy_py --> pipeline_py
  tests_run_stress_heavy_py -.-> logger_factory_py
  tests_run_stress_heavy_py -.-> config_py
  tests_run_stress_heavy_py -.-> model_state_py
  tests_run_stress_heavy_py -.-> schema_py
  tests_run_stress_heavy_py -.-> preprocessor_py
  tests_run_stress_heavy_py -.-> feature_engineer_py
  tests_run_stress_heavy_py -.-> seed_labeler_py
  tests_run_stress_heavy_py -.-> categorization_model_py
  tests_run_stress_heavy_py -.-> expected_spend_model_py
  tests_run_stress_heavy_py -.-> anomaly_detector_py
  tests_run_stress_heavy_py -.-> recurring_detector_py
  tests_run_stress_heavy_py -.-> log_utils_py
  tests_run_stress_heavy_py -.-> insight_model_py
  tests_run_stress_heavy_py -.-> insight_generator_py
  tests_run_stress_heavy_py -.-> contracts_py
  tests_run_stress_heavy_py -.-> known_persons_py
  tests_run_stress_heavy_py -.-> passion_pipeline_py
  tests_run_stress_heavy_py -.-> config_passion_py
  tests_run_stress_heavy_py -.-> passion_utils_py
  tests_run_stress_heavy_py -.-> candidate_py
  tests_run_stress_heavy_py -.-> passion_models_py
  tests_run_stress_heavy_py -.-> banned_content_py
  tests_run_stress_heavy_py -.-> pipeline_result_py
  tests_run_stress_heavy_py -.-> bootstrap_py
  tests_run_stress_heavy_py -.-> marketplace_subcategory_py
  tests_run_stress_heavy_py -.-> passion_detector_py
  tests_run_stress_heavy_py -.-> passion_insight_generator_py
  tests_test_phase3_py --> anomaly_detector_py
  tests_test_phase3_py --> recurring_detector_py
  tests_test_phase3_py --> insight_generator_py
  tests_test_phase3_py --> pipeline_py
  tests_test_phase3_py -.-> schema_py
  tests_test_phase3_py -.-> logger_factory_py
  tests_test_phase3_py -.-> config_py
  tests_test_phase3_py -.-> log_utils_py
  tests_test_phase3_py -.-> contracts_py
  tests_test_phase3_py -.-> model_state_py
  tests_test_phase3_py -.-> preprocessor_py
  tests_test_phase3_py -.-> feature_engineer_py
  tests_test_phase3_py -.-> seed_labeler_py
  tests_test_phase3_py -.-> categorization_model_py
  tests_test_phase3_py -.-> expected_spend_model_py
  tests_test_phase3_py -.-> insight_model_py
  tests_test_phase3_py -.-> known_persons_py
  tests_test_phase3_py -.-> passion_pipeline_py
  tests_test_phase3_py -.-> config_passion_py
  tests_test_phase3_py -.-> passion_utils_py
  tests_test_phase3_py -.-> candidate_py
  tests_test_phase3_py -.-> passion_models_py
  tests_test_phase3_py -.-> banned_content_py
  tests_test_phase3_py -.-> pipeline_result_py
  tests_test_phase3_py -.-> bootstrap_py
  tests_test_phase3_py -.-> marketplace_subcategory_py
  tests_test_phase3_py -.-> passion_detector_py
  tests_test_phase3_py -.-> passion_insight_generator_py
  tests_run_tests_legacy_py --> tests_test_phase2_py
  tests_run_tests_legacy_py --> tests_test_phase1_py
  tests_run_tests_legacy_py -.-> categorization_model_py
  tests_run_tests_legacy_py -.-> config_py
  tests_run_tests_legacy_py -.-> schema_py
  tests_run_tests_legacy_py -.-> logger_factory_py
  tests_run_tests_legacy_py -.-> expected_spend_model_py
  tests_run_tests_legacy_py -.-> preprocessor_py
  tests_run_tests_legacy_py -.-> feature_engineer_py
  tests_run_tests_legacy_py -.-> seed_labeler_py
  tests_test_ml_integration_py --> insight_model_py
  tests_test_ml_integration_py --> schema_py
  tests_test_ml_integration_py -.-> logger_factory_py
  tests_test_ml_integration_py -.-> config_py
  tests_test_known_persons_py --> known_persons_py
  tests_test_known_persons_py --> pipeline_py
  tests_test_known_persons_py --> model_state_py
  tests_test_known_persons_py --> schema_py
  tests_test_known_persons_py -.-> logger_factory_py
  tests_test_known_persons_py -.-> config_py
  tests_test_known_persons_py -.-> preprocessor_py
  tests_test_known_persons_py -.-> feature_engineer_py
  tests_test_known_persons_py -.-> seed_labeler_py
  tests_test_known_persons_py -.-> categorization_model_py
  tests_test_known_persons_py -.-> expected_spend_model_py
  tests_test_known_persons_py -.-> anomaly_detector_py
  tests_test_known_persons_py -.-> recurring_detector_py
  tests_test_known_persons_py -.-> log_utils_py
  tests_test_known_persons_py -.-> insight_model_py
  tests_test_known_persons_py -.-> insight_generator_py
  tests_test_known_persons_py -.-> contracts_py
  tests_test_known_persons_py -.-> passion_pipeline_py
  tests_test_known_persons_py -.-> config_passion_py
  tests_test_known_persons_py -.-> passion_utils_py
  tests_test_known_persons_py -.-> candidate_py
  tests_test_known_persons_py -.-> passion_models_py
  tests_test_known_persons_py -.-> banned_content_py
  tests_test_known_persons_py -.-> pipeline_result_py
  tests_test_known_persons_py -.-> bootstrap_py
  tests_test_known_persons_py -.-> marketplace_subcategory_py
  tests_test_known_persons_py -.-> passion_detector_py
  tests_test_known_persons_py -.-> passion_insight_generator_py
  tests_conftest_py --> passion_pipeline_py
  tests_conftest_py --> log_utils_py
  tests_conftest_py -.-> schema_py
  tests_conftest_py -.-> logger_factory_py
  tests_conftest_py -.-> config_py
  tests_conftest_py -.-> config_passion_py
  tests_conftest_py -.-> passion_utils_py
  tests_conftest_py -.-> candidate_py
  tests_conftest_py -.-> passion_models_py
  tests_conftest_py -.-> banned_content_py
  tests_conftest_py -.-> pipeline_result_py
  tests_conftest_py -.-> bootstrap_py
  tests_conftest_py -.-> contracts_py
  tests_conftest_py -.-> marketplace_subcategory_py
  tests_conftest_py -.-> passion_detector_py
  tests_conftest_py -.-> passion_insight_generator_py
  tests_test_phase2_py --> categorization_model_py
  tests_test_phase2_py --> expected_spend_model_py
  tests_test_phase2_py -.-> config_py
  tests_test_phase2_py -.-> schema_py
  tests_test_phase2_py -.-> logger_factory_py
  tests_test_passion_engine_py --> passion_pipeline_py
  tests_test_passion_engine_py --> passion_utils_py
  tests_test_passion_engine_py --> pipeline_result_py
  tests_test_passion_engine_py --> log_utils_py
  tests_test_passion_engine_py --> passion_detector_py
  tests_test_passion_engine_py --> config_passion_py
  tests_test_passion_engine_py --> passion_insight_generator_py
  tests_test_passion_engine_py --> marketplace_subcategory_py
  tests_test_passion_engine_py --> banned_content_py
  tests_test_passion_engine_py --> pipeline_py
  tests_test_passion_engine_py --> schema_py
  tests_test_passion_engine_py --> passion_models_py
  tests_test_passion_engine_py --> bootstrap_py
  tests_test_passion_engine_py --> contracts_py
  tests_test_passion_engine_py --> candidate_py
  tests_test_passion_engine_py --> config_py
  tests_test_passion_engine_py --> hash_utils_py
  tests_test_passion_engine_py -.-> logger_factory_py
  tests_test_passion_engine_py -.-> model_state_py
  tests_test_passion_engine_py -.-> preprocessor_py
  tests_test_passion_engine_py -.-> feature_engineer_py
  tests_test_passion_engine_py -.-> seed_labeler_py
  tests_test_passion_engine_py -.-> categorization_model_py
  tests_test_passion_engine_py -.-> expected_spend_model_py
  tests_test_passion_engine_py -.-> anomaly_detector_py
  tests_test_passion_engine_py -.-> recurring_detector_py
  tests_test_passion_engine_py -.-> insight_model_py
  tests_test_passion_engine_py -.-> insight_generator_py
  tests_test_passion_engine_py -.-> known_persons_py
  tests_test_benchmark_py --> training_data_generator_py
  tests_test_benchmark_py --> config_py
  tests_test_benchmark_py -.-> contracts_py
  tests_test_model_security_py --> insight_model_py
  tests_test_model_security_py -.-> schema_py
  tests_test_model_security_py -.-> logger_factory_py
  tests_test_model_security_py -.-> config_py
  tests_test_logging_safety_py --> pipeline_py
  tests_test_logging_safety_py --> recurring_detector_py
  tests_test_logging_safety_py --> config_py
  tests_test_logging_safety_py --> logger_factory_py
  tests_test_logging_safety_py --> log_utils_py
  tests_test_logging_safety_py --> schema_py
  tests_test_logging_safety_py -.-> model_state_py
  tests_test_logging_safety_py -.-> preprocessor_py
  tests_test_logging_safety_py -.-> feature_engineer_py
  tests_test_logging_safety_py -.-> seed_labeler_py
  tests_test_logging_safety_py -.-> categorization_model_py
  tests_test_logging_safety_py -.-> expected_spend_model_py
  tests_test_logging_safety_py -.-> anomaly_detector_py
  tests_test_logging_safety_py -.-> insight_model_py
  tests_test_logging_safety_py -.-> insight_generator_py
  tests_test_logging_safety_py -.-> contracts_py
  tests_test_logging_safety_py -.-> known_persons_py
  tests_test_logging_safety_py -.-> passion_pipeline_py
  tests_test_logging_safety_py -.-> config_passion_py
  tests_test_logging_safety_py -.-> passion_utils_py
  tests_test_logging_safety_py -.-> candidate_py
  tests_test_logging_safety_py -.-> passion_models_py
  tests_test_logging_safety_py -.-> banned_content_py
  tests_test_logging_safety_py -.-> pipeline_result_py
  tests_test_logging_safety_py -.-> bootstrap_py
  tests_test_logging_safety_py -.-> marketplace_subcategory_py
  tests_test_logging_safety_py -.-> passion_detector_py
  tests_test_logging_safety_py -.-> passion_insight_generator_py
  tests_test_phase1_py --> schema_py
  tests_test_phase1_py --> preprocessor_py
  tests_test_phase1_py --> feature_engineer_py
  tests_test_phase1_py --> seed_labeler_py
  tests_test_phase1_py --> config_py
  tests_test_phase1_py -.-> logger_factory_py
  tests_test_e2e_py --> preprocessor_py
  tests_test_e2e_py --> feature_engineer_py
  tests_test_e2e_py --> seed_labeler_py
  tests_test_e2e_py --> categorization_model_py
  tests_test_e2e_py --> expected_spend_model_py
  tests_test_e2e_py --> anomaly_detector_py
  tests_test_e2e_py --> recurring_detector_py
  tests_test_e2e_py --> insight_generator_py
  tests_test_e2e_py --> pipeline_py
  tests_test_e2e_py --> schema_py
  tests_test_e2e_py --> model_state_py
  tests_test_e2e_py -.-> config_py
  tests_test_e2e_py -.-> logger_factory_py
  tests_test_e2e_py -.-> log_utils_py
  tests_test_e2e_py -.-> contracts_py
  tests_test_e2e_py -.-> insight_model_py
  tests_test_e2e_py -.-> known_persons_py
  tests_test_e2e_py -.-> passion_pipeline_py
  tests_test_e2e_py -.-> config_passion_py
  tests_test_e2e_py -.-> passion_utils_py
  tests_test_e2e_py -.-> candidate_py
  tests_test_e2e_py -.-> passion_models_py
  tests_test_e2e_py -.-> banned_content_py
  tests_test_e2e_py -.-> pipeline_result_py
  tests_test_e2e_py -.-> bootstrap_py
  tests_test_e2e_py -.-> marketplace_subcategory_py
  tests_test_e2e_py -.-> passion_detector_py
  tests_test_e2e_py -.-> passion_insight_generator_py
  tests_run_smoke_py --> pipeline_py
  tests_run_smoke_py -.-> logger_factory_py
  tests_run_smoke_py -.-> config_py
  tests_run_smoke_py -.-> model_state_py
  tests_run_smoke_py -.-> schema_py
  tests_run_smoke_py -.-> preprocessor_py
  tests_run_smoke_py -.-> feature_engineer_py
  tests_run_smoke_py -.-> seed_labeler_py
  tests_run_smoke_py -.-> categorization_model_py
  tests_run_smoke_py -.-> expected_spend_model_py
  tests_run_smoke_py -.-> anomaly_detector_py
  tests_run_smoke_py -.-> recurring_detector_py
  tests_run_smoke_py -.-> log_utils_py
  tests_run_smoke_py -.-> insight_model_py
  tests_run_smoke_py -.-> insight_generator_py
  tests_run_smoke_py -.-> contracts_py
  tests_run_smoke_py -.-> known_persons_py
  tests_run_smoke_py -.-> passion_pipeline_py
  tests_run_smoke_py -.-> config_passion_py
  tests_run_smoke_py -.-> passion_utils_py
  tests_run_smoke_py -.-> candidate_py
  tests_run_smoke_py -.-> passion_models_py
  tests_run_smoke_py -.-> banned_content_py
  tests_run_smoke_py -.-> pipeline_result_py
  tests_run_smoke_py -.-> bootstrap_py
  tests_run_smoke_py -.-> marketplace_subcategory_py
  tests_run_smoke_py -.-> passion_detector_py
  tests_run_smoke_py -.-> passion_insight_generator_py
  tests_run_stress_legacy_py --> pipeline_py
  tests_run_stress_legacy_py --> schema_py
  tests_run_stress_legacy_py -.-> logger_factory_py
  tests_run_stress_legacy_py -.-> config_py
  tests_run_stress_legacy_py -.-> model_state_py
  tests_run_stress_legacy_py -.-> preprocessor_py
  tests_run_stress_legacy_py -.-> feature_engineer_py
  tests_run_stress_legacy_py -.-> seed_labeler_py
  tests_run_stress_legacy_py -.-> categorization_model_py
  tests_run_stress_legacy_py -.-> expected_spend_model_py
  tests_run_stress_legacy_py -.-> anomaly_detector_py
  tests_run_stress_legacy_py -.-> recurring_detector_py
  tests_run_stress_legacy_py -.-> log_utils_py
  tests_run_stress_legacy_py -.-> insight_model_py
  tests_run_stress_legacy_py -.-> insight_generator_py
  tests_run_stress_legacy_py -.-> contracts_py
  tests_run_stress_legacy_py -.-> known_persons_py
  tests_run_stress_legacy_py -.-> passion_pipeline_py
  tests_run_stress_legacy_py -.-> config_passion_py
  tests_run_stress_legacy_py -.-> passion_utils_py
  tests_run_stress_legacy_py -.-> candidate_py
  tests_run_stress_legacy_py -.-> passion_models_py
  tests_run_stress_legacy_py -.-> banned_content_py
  tests_run_stress_legacy_py -.-> pipeline_result_py
  tests_run_stress_legacy_py -.-> bootstrap_py
  tests_run_stress_legacy_py -.-> marketplace_subcategory_py
  tests_run_stress_legacy_py -.-> passion_detector_py
  tests_run_stress_legacy_py -.-> passion_insight_generator_py
```
