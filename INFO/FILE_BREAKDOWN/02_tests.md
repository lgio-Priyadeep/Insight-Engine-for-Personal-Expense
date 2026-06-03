# Test Coverage Map

## Test → Source Mapping

### `tests/conftest.py`

#### Dependencies
- **Source files under test**: passion_pipeline.py, log_utils.py
- **Mocked dependencies**: none
- **External packages used**: pytest

#### Coverage
- **Tested behaviors**: none detected
- **Gaps observed**: Verify if all edge cases are asserted.


### `tests/run_smoke.py`

#### Dependencies
- **Source files under test**: pipeline.py
- **Mocked dependencies**: none
- **External packages used**: pandas

#### Coverage
- **Tested behaviors**: none detected
- **Gaps observed**: Verify if all edge cases are asserted.


### `tests/run_stress_heavy.py`

#### Dependencies
- **Source files under test**: pipeline.py
- **Mocked dependencies**: none
- **External packages used**: pandas, numpy

#### Coverage
- **Tested behaviors**: none detected
- **Gaps observed**: Verify if all edge cases are asserted.


### `tests/run_stress_legacy.py`

#### Dependencies
- **Source files under test**: pipeline.py, schema.py
- **Mocked dependencies**: none
- **External packages used**: psutil, numpy, pandas

#### Coverage
- **Tested behaviors**: none detected
- **Gaps observed**: Verify if all edge cases are asserted.


### `tests/run_tests_legacy.py`

#### Dependencies
- **Source files under test**: none explicitly imported
- **Mocked dependencies**: none
- **External packages used**: none

#### Coverage
- **Tested behaviors**: none detected
- **Gaps observed**: Verify if all edge cases are asserted.


### `tests/test_benchmark.py`

#### Dependencies
- **Source files under test**: training_data_generator.py, config.py
- **Mocked dependencies**: none
- **External packages used**: numpy, pandas, pytest, sklearn

#### Coverage
- **Tested behaviors**: test_data_generator_shape, test_no_nan_in_features, test_no_nan_in_labels, test_data_generator_distribution, test_all_insight_types_present, test_categories_are_valid, test_spikes_have_high_zscore, test_subscriptions_have_recurring_flag, test_no_action_has_benign_features, test_no_action_has_no_tip, test_actionable_insights_have_tips, test_tip_ids_are_valid, test_find_best_tip_category_specific, test_find_best_tip_generic_fallback, test_find_best_tip_no_match, test_amounts_are_positive, test_rolling_std_no_zero, test_model_can_train_and_predict
- **Gaps observed**: Verify if all edge cases are asserted.


### `tests/test_e2e.py`

#### Dependencies
- **Source files under test**: preprocessor.py, feature_engineer.py, seed_labeler.py, categorization_model.py, expected_spend_model.py, anomaly_detector.py, recurring_detector.py, insight_generator.py, pipeline.py, schema.py, model_state.py
- **Mocked dependencies**: none
- **External packages used**: pandas, numpy

#### Coverage
- **Tested behaviors**: test_run_e2e_test, test_run_inference_uses_history_features
- **Gaps observed**: Verify if all edge cases are asserted.


### `tests/test_known_persons.py`

#### Dependencies
- **Source files under test**: known_persons.py, pipeline.py, model_state.py, schema.py
- **Mocked dependencies**: none
- **External packages used**: pytest, pandas, numpy

#### Coverage
- **Tested behaviors**: test_known_persons_cfg, test_self_accounts_cfg, test_extract_signals_upi_digits, test_tagging_exact_upi, test_tagging_self_account_fragment, test_merchant_suppression, test_concat_partial_with_bounds, test_enforce_schema_schema, test_suggestion_key_min_len, test_state_version_uses_di_params_not_globals
- **Gaps observed**: Verify if all edge cases are asserted.


### `tests/test_logging_safety.py`

#### Dependencies
- **Source files under test**: pipeline.py, recurring_detector.py, config.py, logger_factory.py, log_utils.py, schema.py
- **Mocked dependencies**: none
- **External packages used**: pytest, pandas

#### Coverage
- **Tested behaviors**: test_crash_dump_created, test_crash_exception_identity_matching, test_crash_dump_failure_safety, test_invalid_log_level, test_pii_redaction_coverage
- **Gaps observed**: Verify if all edge cases are asserted.


### `tests/test_ml_integration.py`

#### Dependencies
- **Source files under test**: insight_model.py, schema.py
- **Mocked dependencies**: none
- **External packages used**: pandas, pytest, sklearn

#### Coverage
- **Tested behaviors**: test_load_insight_ranker, test_predict_insight_scores, test_predict_insight_scores_graceful_fallback
- **Gaps observed**: Verify if all edge cases are asserted.


### `tests/test_model_security.py`

#### Dependencies
- **Source files under test**: insight_model.py
- **Mocked dependencies**: none
- **External packages used**: pytest

#### Coverage
- **Tested behaviors**: test_checksum_is_hex_string, test_checksum_deterministic, test_different_content_different_checksum, test_valid_checksum_passes, test_tampered_checksum_raises, test_valid_model_path_accepted, test_path_traversal_rejected, test_symlink_resolved, test_load_rejects_missing_checksum, test_load_rejects_tampered_model, test_load_accepts_valid_signed_model, test_load_missing_file_returns_none
- **Gaps observed**: Verify if all edge cases are asserted.


### `tests/test_passion_engine.py`

#### Dependencies
- **Source files under test**: passion_pipeline.py, passion_utils.py, pipeline_result.py, log_utils.py, passion_detector.py, config_passion.py, passion_insight_generator.py, marketplace_subcategory.py, banned_content.py, pipeline.py, schema.py, passion_models.py, bootstrap.py, contracts.py, candidate.py, config.py, hash_utils.py
- **Mocked dependencies**: MagicMock, patch
- **External packages used**: pytest, pandas, numpy

#### Coverage
- **Tested behaviors**: test_basic_values, test_none_and_na, test_string_raises, test_np_bool_true, test_np_bool_false, test_np_int64_one, test_np_int64_zero, test_np_float64_one, test_np_float64_zero, test_np_nan_returns_false, test_pd_na_returns_false, test_coerce_bool_column_mixed, test_coerce_bool_column_boolean_dtype, test_coerce_bool_column_int64_nullable, test_coerce_bool_column_object_mixed_without_strings, test_coerce_bool_column_object_string_raises, test_length_mismatch_raises, test_valid_list_mask, test_nonunique_target_index_raises, test_categorical_bool_mask, test_int_handled_as_seconds, test_nat_returns_nan, test_invalid_returns_nan, test_normalize_ts_memory_error_propagates_from_timestamp, test_bool_returns_nan, test_inf_returns_nan, test_negative_millisecond_truncation, test_negative_millisecond_large, test_normalize_ts_exact_boundaries, test_milliseconds, test_nanosecond_timestamp, test_yyyymmdd_optin_false_returns_nan, test_yyyymmdd_optin_true_converts, test_invalid_yyyymmdd_returns_nan, test_naive_datetime_string_is_utc_anchored, test_np_int64_handled, test_np_float64_handled, test_datetime_time_returns_nan, test_within_budget, test_budget_exceeded, test_hard_deadline_exceeded, test_validate_aliases_uppercase_key_raises, test_deterministic, test_prefix_and_length, test_collision_resistance, test_empty_returns_empty, test_nan_returns_empty, test_list_returns_empty, test_bytes_returns_empty, test_verify_merchant_token, test_detects_in_sentence, test_clean_text_passes, test_banned_in_middle_of_word_does_not_match, test_plural_scams_detected, test_cyrillic_homoglyph_blocked, test_compact_does_not_match_inside_words, test_obfuscated_safe_terms_detected, test_greek_homoglyph_not_covered_by_design, test_narrowed_separators_boundary, test_coffee_offered_feedback_do_not_trigger, test_actual_fee_keywords_trigger, test_distress_gate_boundaries, test_basic_alias, test_compound_alias_ordering, test_enrich_subcategories_accepts_currency_amount_strings, test_single_point_returns_false, test_flat_trend_returns_true, test_strict_decline_returns_false, test_all_zero_returns_false, test_all_negative_returns_false, test_integer_date_column_declining, test_integer_epoch_seconds_non_declining, test_integer_epoch_seconds_same_month_buckets_correctly, test_gap_month_non_declining, test_no_generalist, test_schema_col_lowercase, test_high_amount_gets_electronics_subcategory, test_anomaly_suppression_majority, test_anomaly_suppression_boundaries, test_merchant_list_contains_python_str, test_zero_spend_category_skipped, test_detect_passions_accepts_currency_amount_strings, test_detect_passions_tz_aware_dates_handles_active_months, test_detect_passions_resolved_merchants_nan_contract, test_merchant_count_boundaries, test_suppressed_signal_absent_from_passion_result, test_suppressed_signal_absent_from_pipeline_result, test_all_suppressed_yields_empty_passion_signals, test_suppressed_signals_counted_in_log, test_broken_index_lookup_preserves_active_months_and_latest_ts, test_empty_dataframe_missing_required_column_raises, test_process_pipeline_empty_df_has_passion_columns, test_process_pipeline_rejects_compact_yyyymmdd_without_opt_in, test_process_pipeline_accepts_compact_yyyymmdd_with_opt_in, test_pipeline_result_debits_is_dataframe, test_duplicate_index_raises, test_inf_amounts_excluded, test_strict_mode_enrich_failure_reraises, test_process_pipeline_strict_mode_false_enrich_subcategories_fails_safely, test_pipeline_result_insights_string_raises, test_memory_error_propagates, test_non_dataframe_input_raises, test_init_failure_prevents_retry, test_attach_passion_results_real_pipeline_result_enabled, test_attach_passion_results_disabled_does_not_call_process, test_attach_passion_results_exceeds_max_rows_skips, test_attach_passion_results_timeout_error_returns_original, test_invalid_compact_yyyymmdd_original_date_preserved, test_attach_passion_results_enabled_failure_logs_and_returns_original, test_attach_passion_results_strict_attach_propagates, test_attach_passion_results_memory_error_propagates, test_attach_passion_results_empty_debits_skips, test_memory_behavior_defensive_copies, test_pipeline_with_known_person_and_valid_spend, test_pipeline_with_invalid_amount_and_valid_spend, test_pipeline_all_known_persons, test_pipeline_all_invalid_amounts, test_pipeline_unsorted_mixed_index, test_resolve_merchant_vectorized_raises_strict_mode, test_resolve_merchant_vectorized_raises_soft_mode, test_run_pipeline_has_passion_hook_before_return, test_run_inference_has_passion_hook_before_return, test_startup_checks_run_successfully, test_tip_corpus_missing_text_key_raises, test_tip_corpus_missing_categories_key_raises, test_tip_corpus_missing_insights_key_raises, test_dotted_field_name_rejected, test_duplicate_col_values_detected, test_template_conversion_rejected, test_template_nested_format_spec_rejected, test_tip_corpus_non_string_element_rejected, test_tip_corpus_wildcard_policy_and_checks, test_dry_render_catches_field_typo, test_safe_last_nonnull_casts_to_str, test_validate_template_values_rejects_objects, test_validate_template_values_accepts_decimal, test_enrich_subcategories_bad_amounts, test_enrich_subcategories_missing_predicted_category_raises, test_generate_insights_deduplicates, test_generate_insights_fallback_banned, test_lookup_matching_tip_ids_callable, test_safe_assign_new_columns_whitelist, test_safe_numeric_currency_prefixes, test_merchant_count_mismatch_raises, test_negative_total_spend_raises, test_spend_share_above_one_raises, test_invalid_trend_direction_raises, test_suppressed_empty_reason_raises, test_not_suppressed_with_reason_raises, test_valid_signal_passes, test_passion_signal_merchant_list_list_converts_to_tuple, test_spend_share_float_precision_accepted, test_string_merchant_list_raises, test_nan_score_raises, test_inf_amount_raises, test_empty_category_raises, test_empty_merchant_raises, test_candidate_passion_valid_signal_does_not_raise, test_candidate_imports_successfully, test_set_field_raises, test_non_dataframe_debits_raises, test_concurrent_initialization, test_module_imports, test_insight_generator_does_not_import_tip_corpus_from_config, test_no_production_module_imports_stable_hash, test_stable_hash_emits_deprecation_warning_and_token_format, test_below_min_rows_returns_same_row_count, test_substage_failure_soft_mode_preserves_row_count, test_no_partial_enriched_values_after_failure, test_single_row_input_enriches_subcategory_but_no_passion_signals, test_attach_passion_results_e2e_enabled_populates_passion_fields, test_attach_passion_results_e2e_disabled_leaves_fields_empty, test_input_df_not_mutated_by_process_pipeline, test_result_debits_mutation_does_not_propagate_to_input, test_result_debits_mutation_does_not_propagate_to_passion_debits, test_resolve_merchant_vectorized_is_idempotent, test_repeated_alias_replacement, test_special_char_adjacent_alias, test_compound_alias_returns_canonical_not_phrase, test_unmatched_merchant_passthrough, test_null_empty_returns_empty_string, test_canonical_only_as_alias_key_fails_validation, test_secret_cached_after_first_call, test_reset_secret_cache_allows_env_var_change, test_100k_row_process_pipeline_completes_under_30s, test_crash_dump_contains_passion_fields_pii_masked, test_safe_assign_unexpected_columns_raises_in_strict_mode, test_config_passion_aliases_contain_required_generalist_canonicals
- **Gaps observed**: Verify if all edge cases are asserted.


### `tests/test_phase1.py`

#### Dependencies
- **Source files under test**: schema.py, preprocessor.py, feature_engineer.py, seed_labeler.py, config.py
- **Mocked dependencies**: none
- **External packages used**: numpy, pandas, pytest

#### Coverage
- **Tested behaviors**: test_valid_df_passes, test_single_missing_column_raises, test_multiple_missing_columns_raises, test_extra_columns_are_allowed, test_valid_flags, test_invalid_strings_return_none, test_non_string_returns_none, test_all_dr_are_negative, test_all_cr_are_positive, test_absolute_value_preserved, test_mixed_case_flags_handled, test_amount_flag_column_normalized_in_place, test_defaults_invalid_flags_to_dr, test_removes_4digit_pii_ref, test_retains_shorter_numbers, test_removes_email, test_removes_special_characters, test_strips_noise_tokens, test_lowercases_output, test_collapses_whitespace, test_empty_string_returns_empty, test_none_returns_empty, test_whitespace_only_returns_empty, test_only_noise_tokens_returns_empty, test_non_string_returns_empty, test_mixed_real_and_noise, test_unmapped_merchant_survives_generic_routing, test_drops_zero_rows, test_no_change_when_no_zeros, test_all_zeros_removed, test_removes_exact_duplicates, test_no_change_when_no_duplicates, test_different_date_same_amount_not_deduped, test_returns_two_dataframes, test_pure_debit_input, test_pure_credit_input, test_cleaned_remarks_column_present, test_signed_amount_column_present, test_output_is_sorted_chronologically, test_missing_column_raises, test_does_not_mutate_input, test_is_weekend_correct_for_known_dates, test_month_sin_cos_in_range, test_dow_sin_cos_in_range, test_week_of_month_range, test_non_datetime_raises, test_first_row_rolling_mean_excludes_itself, test_rolling_mean_does_not_include_current_row, test_rolling_std_nan_when_fewer_than_2_prior_rows, test_output_sorted_chronologically, test_no_nans_after_fill, test_fill_uses_provided_mean, test_zero_global_std_falls_back_to_one, test_zscore_clipped_upper, test_zscore_clipped_lower, test_zero_std_does_not_produce_inf, test_amount_log_is_non_negative, test_amount_log_uses_absolute_value, test_full_pipeline_no_nans, test_does_not_mutate_input, test_food_keyword_match, test_transport_keyword_match, test_shopping_keyword_match, test_empty_remark_returns_fallback, test_whitespace_only_returns_fallback, test_no_match_returns_fallback, test_multi_word_keyword_match, test_multi_word_keyword_requires_exact_boundary, test_priority_order_respected, test_adversarial_tiebreaking_logic, test_known_remarks_correctly_labeled, test_unknown_remark_gets_fallback, test_label_col_name_respected, test_input_not_mutated, test_all_rows_labeled, test_metadata_columns_always_present, test_spencers_labeled_as_food, test_acct_token_labeled_as_transfer, test_salary_match, test_refund_match, test_interest_match, test_unknown_credit_falls_back_to_other_credit, test_fallback_is_not_uncategorized, test_all_rows_receive_a_label, test_inference_with_backdated_transaction, test_inference_returns_correct_row_count, test_inference_with_empty_history
- **Gaps observed**: Verify if all edge cases are asserted.


### `tests/test_phase2.py`

#### Dependencies
- **Source files under test**: categorization_model.py, expected_spend_model.py
- **Mocked dependencies**: none
- **External packages used**: pandas, pytest, numpy

#### Coverage
- **Tested behaviors**: test_categorization_training_and_prediction, test_spend_model_training_and_prediction, test_expected_spend_model_extrapolates, test_percent_deviation_no_inf, test_percent_deviation_negative_expected_amount, test_percent_deviation_zero_expected_amount, test_percent_deviation_normal_case
- **Gaps observed**: Verify if all edge cases are asserted.


### `tests/test_phase3.py`

#### Dependencies
- **Source files under test**: anomaly_detector.py, recurring_detector.py, insight_generator.py, pipeline.py
- **Mocked dependencies**: none
- **External packages used**: pandas, pytest

#### Coverage
- **Tested behaviors**: test_recurring_detector, test_recurring_frequency_none_for_non_recurring, test_recurring_requires_minimum_3_occurrences, test_biweekly_detection_and_variance_penalty, test_anomaly_composite_threshold, test_missing_anomaly_columns_raises, test_insight_generator_creates_strings, test_insight_generator_deterministic, test_insight_generator_different_seeds, test_select_tip_deterministic, test_pipeline_result_is_frozen, test_pipeline_result_replace
- **Gaps observed**: Verify if all edge cases are asserted.

## Coverage Gaps Summary

| Untested Source Files |
|-----------------------|
| `demo.py` |
| `model_benchmark.py` |
| `refactor_pipeline.py` |
| `summary_utils.py` |
| `train_and_save_models.py` |
| `tutorial_real_data.py` |

## Test Quality Flags
- `tests/conftest.py` has no standard `test_` functions.
- `tests/run_smoke.py` has no standard `test_` functions.
- `tests/run_stress_heavy.py` has no standard `test_` functions.
- `tests/run_stress_legacy.py` has no standard `test_` functions.
- `tests/run_tests_legacy.py` has no standard `test_` functions.
- `tests/test_passion_engine.py` uses mocks: MagicMock, patch (could hide real behavior)
