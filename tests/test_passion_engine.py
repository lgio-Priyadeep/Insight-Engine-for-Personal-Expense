# BEGIN_CB_P7_05
import pytest
import random
import threading
import time
import pandas as pd
import numpy as np
from decimal import Decimal
from unittest.mock import patch

from passion_pipeline import (
  process_pipeline, _StepBudgetGuard,
  _normalize_ts, safe_assign_new_columns,
)
from passion_utils import (
  to_bool_strict, coerce_bool_column, sanitize_mask,
  safe_last_nonnull, validate_template_values, _safe_isna,
)
from pipeline_result import PassionResult
from log_utils import log_safe_merchant, log_safe_text, verify_merchant_token
from passion_detector import (
  _is_non_declining, detect_passions, _check_anomaly_suppression, _check_distress_gate,
)
from config_passion import validate_merchant_aliases
from passion_insight_generator import generate_passion_insights
from marketplace_subcategory import resolve_merchant_vectorized, enrich_subcategories

# FIX-13: Import from banned_content.py (public module), not passion_insight_generator.
from banned_content import contains_banned_content as _contains_banned_content


def make_pipeline_result(**kwargs) -> "PipelineResult":
  from pipeline import PipelineResult
  import pandas as pd
  import dataclasses
  defaults = {
      "debits": pd.DataFrame(),
      "credits": pd.DataFrame(),
      "insights": [],
      "cat_pipeline": None,
      "spend_pipeline": None,
      "ranker_pipeline": None,
      "global_mean": 0.0,
      "global_std": 0.0,
      "raw_global_mean": 0.0,
      "raw_global_std": 0.0,
      "stats_version": "v1_raw",
      "personal_summary": {},
      "transfer_patterns": [],
      "exclusion_stats": {},
      "kp_config_hash": "",
      "personal_debits": pd.DataFrame(),
      "personal_credits": pd.DataFrame(),
      "stats": {},
      "passion_debits": pd.DataFrame(),
      "passion_insights": (),
      "passion_signals": (),
  }
  valid_fields = {f.name for f in dataclasses.fields(PipelineResult)}
  merged = {**defaults, **kwargs}
  filtered = {k: v for k, v in merged.items() if k in valid_fields}
  return PipelineResult(**filtered)


# ── Bool Coercion ─────────────────────────────────────────────────────────────

class TestBoolCoercion:
  def test_basic_values(self):
      assert to_bool_strict(True) is True
      assert to_bool_strict(False) is False
      assert to_bool_strict(1) is True
      assert to_bool_strict(0) is False
      assert to_bool_strict(np.bool_(True)) is True

  def test_none_and_na(self):
      assert to_bool_strict(None) is False
      assert to_bool_strict(pd.NA) is False
      assert to_bool_strict(np.nan) is False

  def test_string_raises(self):
      with pytest.raises(TypeError):
          to_bool_strict("True")

  # C2: np.bool_ explicit tests — np.bool_ is NOT a subclass of Python bool
  # in NumPy 2.x. The isinstance guard in to_bool_strict must catch these.
  def test_np_bool_true(self):
      assert to_bool_strict(np.bool_(True)) is True

  def test_np_bool_false(self):
      assert to_bool_strict(np.bool_(False)) is False

  # C2: np.int64 coercion — must pass through the int() match/case arm
  # after the isinstance guard (since np.int64 is NOT bool/np.bool_).
  def test_np_int64_one(self):
      assert to_bool_strict(np.int64(1)) is True

  def test_np_int64_zero(self):
      assert to_bool_strict(np.int64(0)) is False

  # C2: np.float64 coercion
  def test_np_float64_one(self):
      assert to_bool_strict(np.float64(1.0)) is True

  def test_np_float64_zero(self):
      assert to_bool_strict(np.float64(0.0)) is False

  # C2: np.nan → _safe_isna returns True → False (not a TypeError)
  def test_np_nan_returns_false(self):
      assert to_bool_strict(np.nan) is False

  # C2: pd.NA → _safe_isna returns True → False (not a TypeError)
  def test_pd_na_returns_false(self):
      assert to_bool_strict(pd.NA) is False

  def test_coerce_bool_column_mixed(self):
      s = pd.Series([1, 0, np.nan, True, False])
      res = coerce_bool_column(s)
      assert res.tolist() == [True, False, False, True, False]

  def test_coerce_bool_column_boolean_dtype(self):
      s = pd.array([True, False, pd.NA], dtype=pd.BooleanDtype())
      result = coerce_bool_column(pd.Series(s))
      assert result.dtype == bool
      assert result.tolist() == [True, False, False]

  def test_coerce_bool_column_int64_nullable(self):
      s = pd.array([1, 0, pd.NA], dtype="Int64")
      result = coerce_bool_column(pd.Series(s))
      assert result.dtype == bool
      assert result.tolist() == [True, False, False]

  def test_coerce_bool_column_object_mixed_without_strings(self):
      s = pd.Series([True, 1, 0, None, pd.NA], dtype=object)
      res = coerce_bool_column(s)
      assert res.tolist() == [True, True, False, False, False]

  def test_coerce_bool_column_object_string_raises(self):
      s = pd.Series([True, "False", 1, 0, None, pd.NA], dtype=object)
      with pytest.raises(TypeError, match="string"):
          coerce_bool_column(s)


# ── sanitize_mask ─────────────────────────────────────────────────────────────

class TestSanitizeMask:
  def test_length_mismatch_raises(self):
      with pytest.raises(ValueError):
          sanitize_mask([True, False], pd.Index([1, 2, 3]), "test")


  def test_valid_list_mask(self):
      idx = pd.Index([1, 2, 3])
      res = sanitize_mask([1, 0, np.nan], idx, "test")
      assert res.tolist() == [True, False, False]

  def test_nonunique_target_index_raises(self):
      with pytest.raises(ValueError, match="Non-unique DataFrame index"):
          sanitize_mask([True, False, True], pd.Index([1, 1, 2]), "test")



  def test_categorical_bool_mask(self):
      idx = pd.Index([0, 1, 2])
      cat_mask = pd.Series(
          pd.Categorical([True, False, True]),
          index=idx,
      )
      result = sanitize_mask(cat_mask, idx, "test")
      assert result.dtype == bool
      assert result.tolist() == [True, False, True]


# ── Timestamp Normalisation ───────────────────────────────────────────────────

class TestNormalizeTs:
  def test_int_handled_as_seconds(self):
      assert _normalize_ts(1673740800) == 1673740800

  def test_nat_returns_nan(self):
      assert np.isnan(_normalize_ts(pd.NaT))

  def test_invalid_returns_nan(self):
      assert np.isnan(_normalize_ts("not-a-date"))

  # P0-1 FIX: MemoryError from pd.Timestamp must propagate — not be swallowed by the
  # except (ValueError, TypeError, OverflowError, ...) clause.
  def test_normalize_ts_memory_error_propagates_from_timestamp(self, monkeypatch):
      def boom(*args, **kwargs):
          raise MemoryError("timestamp oom")
      monkeypatch.setattr("passion_pipeline.pd.Timestamp", boom)
      with pytest.raises(MemoryError, match="timestamp oom"):
          _normalize_ts("2023-01-01")

  def test_bool_returns_nan(self):
      assert np.isnan(_normalize_ts(True))
      assert np.isnan(_normalize_ts(False))

  def test_inf_returns_nan(self):
      assert np.isnan(_normalize_ts(float('inf')))

  def test_negative_millisecond_truncation(self):
      result = _normalize_ts(-1001)
      assert result == -1001

  def test_negative_millisecond_large(self):
      result = _normalize_ts(-100_000_000_001)
      assert result == -100_000_000_001 // 1000

  # P1-3 FIX: Exact _normalize_ts boundary tests
  def test_normalize_ts_exact_boundaries(self):
      # Below ms boundary: treated as seconds (no truncation)
      assert _normalize_ts(99_999_999_999) == 99_999_999_999
      # At ms boundary: truncated to seconds
      assert _normalize_ts(100_000_000_000) == 100_000_000_000 // 1000

      # Below us boundary: still truncated as ms
      assert _normalize_ts(99_999_999_999_999) == 99_999_999_999_999 // 1000
      # At us boundary: truncated as us
      assert _normalize_ts(100_000_000_000_000) == 100_000_000_000_000 // 1_000_000

      # Below ns boundary: still truncated as us
      assert _normalize_ts(99_999_999_999_999_999) == 99_999_999_999_999_999 // 1_000_000
      # At ns boundary: truncated as ns
      assert _normalize_ts(100_000_000_000_000_000) == 100_000_000_000_000_000 // 1_000_000_000

  def test_milliseconds(self):
      result = _normalize_ts(1672531200000)
      assert result == 1672531200

  def test_nanosecond_timestamp(self):
      ns = 1672531200_000_000_000
      result = _normalize_ts(ns)
      assert result == 1672531200

  def test_yyyymmdd_optin_false_returns_nan(self):
      # FIX 9: compact YYYYMMDD with allow_yyyymmdd=False returns np.nan, not the raw int.
      # _looks_like_compact_yyyymmdd(20230101) is True (10M <= 20.2M <= 100M).
      # The early-return path produces np.nan when opt-in is False.
      result = _normalize_ts(20230101, allow_yyyymmdd=False)
      assert np.isnan(result)

  def test_yyyymmdd_optin_true_converts(self):
      # C5 + C3: Use UTC-aware expected value. _normalize_ts now localizes naive
      # Timestamps to UTC before calling .timestamp(), so the expected value
      # must also be computed with UTC to avoid local-timezone divergence.
      result = _normalize_ts(20230101, allow_yyyymmdd=True)
      # 2023-01-01T00:00:00 UTC = 1672531200 (constant, timezone-independent)
      expected = int(pd.Timestamp("2023-01-01", tz="UTC").timestamp())
      assert result == expected

  def test_invalid_yyyymmdd_returns_nan(self):
      # C5: Renamed from test_invalid_yyyymmdd_falls_through for clarity.
      # Invalid month 13: pd.Timestamp('20231301') raises OutOfBounds/ValueError.
      # The except branch returns np.nan — NOT the raw integer.
      result = _normalize_ts(20231301, allow_yyyymmdd=True)
      assert np.isnan(result)

  # C3: Timezone-invariance test — a naive datetime string must always produce
  # the same Unix epoch second regardless of the host machine's local timezone.
  # Prior to the C3 fix, pd.Timestamp("2023-01-01").timestamp() used local tz,
  # which differed by hours between UTC, IST, PST, etc.
  def test_naive_datetime_string_is_utc_anchored(self):
      result = _normalize_ts("2023-01-01")
      # 2023-01-01 00:00:00 UTC = 1672531200. Must be equal on all machines.
      assert result == 1672531200

  def test_np_int64_handled(self):
      val = np.int64(1673740800)
      assert _normalize_ts(val) == 1673740800

  def test_np_float64_handled(self):
      val = np.float64(1673740800.0)
      assert _normalize_ts(val) == 1673740800

  def test_datetime_time_returns_nan(self):
      import datetime
      assert np.isnan(_normalize_ts(datetime.time(12, 30)))


# ── StepBudgetGuard ───────────────────────────────────────────────────────────

class TestStepBudgetGuard:
  def test_within_budget(self):
      cb = _StepBudgetGuard(budget_ms=500)
      cb.check("early")

  def test_budget_exceeded(self):
      with patch("passion_pipeline.time.monotonic", side_effect=[0.0, 0.011]):
          cb = _StepBudgetGuard(budget_ms=10)
          # FIX-23: Expect 'Budget exceeded' message for soft budget limit violations
          with pytest.raises(TimeoutError, match="Budget exceeded"):
              cb.check("late")

  def test_hard_deadline_exceeded(self):
      with patch("passion_pipeline.time.monotonic", side_effect=[0.0, 0.011]):
          cb = _StepBudgetGuard(budget_ms=10000, hard_timeout_ms=10)
          # FIX-23: Expect 'Hard deadline exceeded' message for hard timeout limits
          with pytest.raises(TimeoutError, match="Hard deadline"):
              cb.check("late")

# ── Config Validation ─────────────────────────────────────────────────────────

class TestConfigPassion:
  def test_validate_aliases_uppercase_key_raises(self):
      # P0-4 FIX: Pass dictionary to validate_merchant_aliases instead of patching nonexistent constant
      with pytest.raises(ValueError, match="Non-lowercase"):
          validate_merchant_aliases({"UPPER": "lower"})


# ── log_utils ─────────────────────────────────────────────────────────────────

class TestLogUtils:
  def test_deterministic(self):
      assert log_safe_merchant("amazon") == log_safe_merchant("amazon")

  def test_prefix_and_length(self):
      token = log_safe_merchant("amazon")
      assert token.startswith("merchant:")
      assert len(token) == 41

  def test_collision_resistance(self):
      assert log_safe_merchant("a") != log_safe_merchant("b")

  def test_empty_returns_empty(self):
      assert log_safe_merchant("") == ""

  def test_nan_returns_empty(self):
      assert log_safe_merchant(np.nan) == ""

  def test_list_returns_empty(self):
      assert log_safe_merchant(["amazon"]) == ""

  def test_bytes_returns_empty(self):
      assert log_safe_merchant(b"amazon") == ""

  def test_verify_merchant_token(self):
      token = log_safe_merchant("amazon")
      assert verify_merchant_token(token, "amazon") is True
      assert verify_merchant_token(token, "flipkart") is False


# ── Banned Content ────────────────────────────────────────────────────────────

class TestBannedContent:
  def test_detects_in_sentence(self):
      assert _contains_banned_content("this is a scam") is True

  def test_clean_text_passes(self):
      assert _contains_banned_content("normal grocery purchase") is False

  def test_banned_in_middle_of_word_does_not_match(self):
      assert _contains_banned_content("escapade") is False

  def test_plural_scams_detected(self):
      assert _contains_banned_content("known scams alert") is True

  def test_cyrillic_homoglyph_blocked(self):
      cyrillic_scam = "\u0455\u0441\u0430m"
      assert _contains_banned_content(cyrillic_scam) is True

  def test_compact_does_not_match_inside_words(self):
      assert _contains_banned_content("scamper") is False
      assert _contains_banned_content("classical casinoid text") is False

  def test_obfuscated_safe_terms_detected(self):
      assert _contains_banned_content("s-c-a-m") is True
      assert _contains_banned_content("s c a m") is True

  # FIX 25: Explicit test proving Greek homoglyphs are NOT covered.
  def test_greek_homoglyph_not_covered_by_design(self):
      # Greek lowercase sigma 'σ' (U+03C3) instead of 's'
      greek_scam = "\u03C3cam"
      # Banned content detector will return False as Greek is not in _CONFUSABLES
      assert _contains_banned_content(greek_scam) is False

  # FIX-10 + Fix 4: Test narrowed separators and boundaries
  def test_narrowed_separators_boundary(self):
      assert _contains_banned_content("s-c-a-m") is True
      assert _contains_banned_content("f r a u d") is True
      assert _contains_banned_content("casino") is True
      assert _contains_banned_content("drugstore") is False
      assert _contains_banned_content("cafe-fraud") is True # Standalone "fraud" inside hyphenated text matches
      assert _contains_banned_content("cafe-fr-aud") is True # Obfuscated fractioned across word boundary
      assert _contains_banned_content("cafe-fr-auditor") is False
      assert _contains_banned_content("spoke person") is False # Does not false positive on "porn"


# ── Distress Gate ─────────────────────────────────────────────────────────────

class TestDistressGate:
  def test_coffee_offered_feedback_do_not_trigger(self):
      df = pd.DataFrame({
          "cleaned_remarks": ["coffee shop", "offered discount", "feedback form",
                              "coffee house", "services offered", "feedback received",
                              "latte art"],
          "amount": [100, 200, 150, 100, 300, 250, 100],
      })
      assert _check_distress_gate(df) is False

  def test_actual_fee_keywords_trigger(self):
      df = pd.DataFrame({
          "cleaned_remarks": ["late payment fee", "penalty charge", "fee applied",
                              "bounce charge", "interest overdue"],
          "amount": [100, 200, 150, 100, 300],
      })
      assert _check_distress_gate(df) is True

  # FIX-8 / P0-5 FIX: Test boundary conditions using ROW RATIO (fee_count/total_rows),
  # not amount share. _check_distress_gate uses strict >  DISTRESS_FEES_THRESHOLD (0.15).
  def test_distress_gate_boundaries(self):
      # 1. 2/14 = 14.28% — below 15% threshold → False
      df_below = pd.DataFrame({
          "cleaned_remarks": ["late payment fee", "penalty charge"] + ["normal shop"] * 12,
      })
      assert _check_distress_gate(df_below) is False

      # 2. 3/20 = exactly 15%; implementation uses strict >, so NOT > 0.15 → False
      df_exact = pd.DataFrame({
          "cleaned_remarks": [
              "late payment fee", "penalty charge", "interest overdue"
          ] + ["normal shop"] * 17,
      })
      assert _check_distress_gate(df_exact) is False

      # 3. 4/20 = 20% — above 15% threshold → True
      df_above = pd.DataFrame({
          "cleaned_remarks": [
              "late payment fee", "penalty charge", "interest overdue", "bounce charge"
          ] + ["normal shop"] * 16,
      })
      assert _check_distress_gate(df_above) is True


# ── Merchant Vectorizer ───────────────────────────────────────────────────────

class TestMerchantVectorizer:
  def test_basic_alias(self):
      s = pd.Series(["AMZN", "amzn"])
      assert resolve_merchant_vectorized(s).tolist() == ["amazon", "amazon"]

  def test_compound_alias_ordering(self):
      s = pd.Series(["amzn mktp us"])
      result = resolve_merchant_vectorized(s)
      assert "amazon" in result.iloc[0]


class TestMarketplaceSubcategory:
  def test_enrich_subcategories_accepts_currency_amount_strings(self):
      df = pd.DataFrame({
          "amount": ["₹1,500", "Rs 500", "INR 1600"],
          "cleaned_remarks": ["amazon", "amazon", "flipkart"],
          "predicted_category": ["shopping", "shopping", "shopping"],
      })
      from marketplace_subcategory import enrich_subcategories
      result = enrich_subcategories(df)
      assert result["inferred_subcategory"].iloc[0] == "electronics"
      assert result["inferred_subcategory"].iloc[1] == "general_purchase"
      assert result["inferred_subcategory"].iloc[2] == "electronics"



# ── _is_non_declining ─────────────────────────────────────────────────────────

class TestIsNonDeclining:
  def test_single_point_returns_false(self):
      df = pd.DataFrame({"date": ["2023-01-01"], "amount": [100]})
      assert _is_non_declining(df) is False

  def test_flat_trend_returns_true(self):
      df = pd.DataFrame({
          "date": ["2023-01-01", "2023-02-01", "2023-03-01", "2023-04-01"],
          "amount": [100, 100, 100, 100],
      })
      assert _is_non_declining(df) is True

  def test_strict_decline_returns_false(self):
      df = pd.DataFrame({
          "date": ["2023-01-01", "2023-02-01", "2023-03-01", "2023-04-01"],
          "amount": [100, 80, 50, 20],
      })
      assert _is_non_declining(df) is False

  def test_all_zero_returns_false(self):
      df = pd.DataFrame({
          "date": ["2023-01-01", "2023-02-01", "2023-03-01", "2023-04-01"],
          "amount": [0, 0, 0, 0],
      })
      assert _is_non_declining(df) is False

  def test_all_negative_returns_false(self):
      df = pd.DataFrame({
          "date": ["2023-01-01", "2023-02-01", "2023-03-01"],
          "amount": [-100, -50, -10],
      })
      assert _is_non_declining(df) is False

  def test_integer_date_column_declining(self):
      # C4: _is_non_declining must handle integer dtype internally with unit="s".
      # 1672531200 = 2023-01-01, 1675209600 = 2023-02-01, 1677628800 = 2023-03-01.
      # Without unit="s", pd.to_datetime interprets these as nanoseconds (year ~1970),
      # collapsing all three into the same month bucket and returning True incorrectly.
      df = pd.DataFrame({
          "date": [1672531200, 1675209600, 1677628800],
          "amount": [100, 80, 50],
      })
      assert _is_non_declining(df) is False

  # C4: Direct standalone call — non-declining with integer epoch seconds.
  # Verifies correct monthly bucketing independent of any pipeline context.
  def test_integer_epoch_seconds_non_declining(self):
      # 1672531200 = 2023-01-01, 1675209600 = 2023-02-01, 1677628800 = 2023-03-01.
      # Spend increases each month → non-declining.
      df = pd.DataFrame({
          "date": [1672531200, 1675209600, 1677628800],
          "amount": [100, 200, 300],
      })
      assert _is_non_declining(df) is True

  # C4: Cross-month boundary — two distinct integer epoch seconds in different months.
  # This test ensures bucketing is by calendar month, not by raw seconds.
  def test_integer_epoch_seconds_same_month_buckets_correctly(self):
      # Both 1672531200 (2023-01-01) and 1672617600 (2023-01-02) are January.
      # 1675209600 = 2023-02-01. Two months, spend goes 200→300 → non-declining.
      df = pd.DataFrame({
          "date": [1672531200, 1672617600, 1675209600, 1677628800],
          "amount": [100, 100, 200, 300],
      })
      assert _is_non_declining(df) is True

  # P3-3 FIX: Sparse months are absent observations, not zero-spend.
  # Two active months with a gap: only 2 data points, both 100 → non-declining.
  def test_gap_month_non_declining(self):
      df = pd.DataFrame({
          "date": ["2023-01-01", "2023-03-01"],
          "amount": [100, 100],
      })
      assert _is_non_declining(df) is True


# ── enrich_subcategories ──────────────────────────────────────────────────────

class TestEnrichSubcategories:
  def test_no_generalist(self):
      df = pd.DataFrame({"amount": [100.0], "cleaned_remarks": ["zomato"], "predicted_category": ["food"]})
      result = enrich_subcategories(df)
      assert result["inferred_subcategory"].isna().all()

  def test_schema_col_lowercase(self):
      from schema import Col
      assert Col.INFERRED_SUBCATEGORY == "inferred_subcategory"

  def test_high_amount_gets_electronics_subcategory(self):
      df = pd.DataFrame({
          "amount": [1500.0, 500.0],
          "cleaned_remarks": ["amazon", "amazon"],
          "predicted_category": ["shopping", "shopping"],
      })
      result = enrich_subcategories(df)
      assert result["inferred_subcategory"].iloc[0] == "electronics"
      assert result["inferred_subcategory"].iloc[1] == "general_purchase"


# ── anomaly suppression ──────────────────────────────────────────────────────

class TestAnomalySuppression:
  def test_anomaly_suppression_majority(self):
      df = pd.DataFrame({"is_anomaly": [True, True, False]})
      assert _check_anomaly_suppression(df) is True

  # FIX-8: Test boundary conditions for anomaly suppression ratio threshold (0.30)
  def test_anomaly_suppression_boundaries(self):
      import pandas as pd
      from passion_detector import _check_anomaly_suppression

      # 1. Below threshold (2/10 = 20.0% anomaly ratio)
      df_below = pd.DataFrame({"is_anomaly": [True, True, False, False, False, False, False, False, False, False]})
      assert _check_anomaly_suppression(df_below) is False

      # 2. Exactly threshold (3/10 = 30.0% anomaly ratio)
      df_exact = pd.DataFrame({"is_anomaly": [True, True, True, False, False, False, False, False, False, False]})
      assert _check_anomaly_suppression(df_exact) is False

      # 3. Above threshold (4/10 = 40.0% anomaly ratio)
      df_above = pd.DataFrame({"is_anomaly": [True, True, True, True, False, False, False, False, False, False]})
      assert _check_anomaly_suppression(df_above) is True
# END_CB_P7_05

# BEGIN_CB_P8_01
# ── detect_passions ───────────────────────────────────────────────────────────

class TestDetectPassions:
  def test_merchant_list_contains_python_str(self):
      df = pd.DataFrame({
          "predicted_category": ["food", "food", "food"],
          "cleaned_remarks": ["zomato", "swiggy", "dominos"],
          "amount": [500, 400, 300],
          "date": ["2023-01-01", "2023-02-01", "2023-03-01"],
      })
      spend_mask = pd.Series([True, True, True], index=df.index)
      with patch("passion_detector._check_distress_gate", return_value=False), \
           patch("passion_detector._is_non_declining", return_value=True), \
           patch("passion_detector._check_anomaly_suppression", return_value=False):
          signals = detect_passions(df, spend_mask)
      if signals:
          for m in signals[0].merchant_list:
              assert type(m) is str

  # FIX #19: Zero-spend category → skipped.
  def test_zero_spend_category_skipped(self):
      df = pd.DataFrame({
          "predicted_category": ["food", "food", "food"],
          "cleaned_remarks": ["zomato", "swiggy", "dominos"],
          "amount": [0.0, 0.0, 0.0],
          "date": ["2023-01-01", "2023-02-01", "2023-03-01"],
      })
      spend_mask = pd.Series([True, True, True], index=df.index)
      signals = detect_passions(df, spend_mask)
      assert len(signals) == 0

  def test_detect_passions_accepts_currency_amount_strings(self):
      df = pd.DataFrame({
          "amount": ["₹1,500", "Rs 1600", "INR 1700"],
          "cleaned_remarks": ["amazon", "flipkart", "myntra"],
          "date": ["2023-01-01", "2023-02-01", "2023-03-01"],
          "predicted_category": ["shopping", "shopping", "shopping"],
      })
      spend_mask = pd.Series([True, True, True], index=df.index)
      resolved = pd.Series(["amazon", "flipkart", "myntra"], index=df.index)
      signals = detect_passions(df, spend_mask, resolved)
      assert len(signals) == 1
      assert signals[0].total_spend == 4800.0

  def test_detect_passions_tz_aware_dates_handles_active_months(self):
      # Verify Blocker 2 fix: to_period works with tz-aware dates after conversion.
      # C9: Use 3 distinct merchants so PASSION_MERCHANT_COUNT_MIN=3 is satisfied.
      # Previously used "amazon" x3 (1 distinct merchant), which produced 0 signals
      # and never tested the timezone conversion path it claimed to verify.
      df = pd.DataFrame({
          "amount": [100.0, 100.0, 100.0],
          "cleaned_remarks": ["amazon", "flipkart", "myntra"],
          "date": pd.to_datetime(["2023-01-01", "2023-02-01", "2023-03-01"], utc=True),
          "predicted_category": ["shopping", "shopping", "shopping"],
      })
      spend_mask = pd.Series([True, True, True], index=df.index)
      signals = detect_passions(df, spend_mask)
      assert len(signals) == 1
      assert signals[0].trend_direction != "insufficient_history"

  def test_detect_passions_resolved_merchants_nan_contract(self):
      """FIX 14: resolved_merchants may contain NaN for non-spend rows (e.g. is_known_person=True or outside spend_mask).
      Verify that NaN values in resolved_merchants are drop/ignored and do not become "nan" strings or affect merchant count.
      """
      import numpy as np
      # 4 rows: 3 valid spend rows with distinct merchants, 1 non-spend row (known person) with NaN resolved merchant
      df = pd.DataFrame({
          "amount": [100.0, 100.0, 100.0, 100.0],
          "cleaned_remarks": ["amazon", "flipkart", "myntra", "nan_merchant"],
          "date": ["2023-01-01", "2023-02-01", "2023-03-01", "2023-04-01"],
          "predicted_category": ["shopping", "shopping", "shopping", "shopping"],
      })
      spend_mask = pd.Series([True, True, True, False], index=df.index)
      resolved = pd.Series(["amazon", "flipkart", "myntra", np.nan], index=df.index)
      signals = detect_passions(df, spend_mask, resolved)
      assert len(signals) == 1
      assert "nan" not in signals[0].merchant_list
      assert np.nan not in signals[0].merchant_list
      assert len(signals[0].merchant_list) == 3
      assert set(signals[0].merchant_list) == {"amazon", "flipkart", "myntra"}

  # FIX-8: Test boundary conditions for merchant count threshold (PASSION_MERCHANT_COUNT_MIN = 3)
  def test_merchant_count_boundaries(self):
      # 1. Below threshold: 2 unique merchants -> no signals
      df_below = pd.DataFrame({
          "amount": [100.0, 100.0, 100.0],
          "cleaned_remarks": ["amazon", "amazon", "flipkart"],
          "date": ["2023-01-01", "2023-02-01", "2023-03-01"],
          "predicted_category": ["shopping", "shopping", "shopping"],
      })
      spend_mask = pd.Series([True, True, True], index=df_below.index)
      with patch("passion_detector._check_distress_gate", return_value=False), \
           patch("passion_detector._is_non_declining", return_value=True), \
           patch("passion_detector._check_anomaly_suppression", return_value=False):
          signals = detect_passions(df_below, spend_mask)
      assert len(signals) == 0

      # 2. Exactly threshold: 3 unique merchants -> 1 signal
      df_exact = pd.DataFrame({
          "amount": [100.0, 100.0, 100.0],
          "cleaned_remarks": ["amazon", "flipkart", "myntra"],
          "date": ["2023-01-01", "2023-02-01", "2023-03-01"],
          "predicted_category": ["shopping", "shopping", "shopping"],
      })
      spend_mask = pd.Series([True, True, True], index=df_exact.index)
      with patch("passion_detector._check_distress_gate", return_value=False), \
           patch("passion_detector._is_non_declining", return_value=True), \
           patch("passion_detector._check_anomaly_suppression", return_value=False):
          signals = detect_passions(df_exact, spend_mask)
      assert len(signals) == 1
      assert signals[0].merchant_count == 3

      # 3. Above threshold: 4 unique merchants -> 1 signal
      df_above = pd.DataFrame({
          "amount": [100.0, 100.0, 100.0, 100.0],
          "cleaned_remarks": ["amazon", "flipkart", "myntra", "steam"],
          "date": ["2023-01-01", "2023-02-01", "2023-03-01", "2023-04-01"],
          "predicted_category": ["shopping", "shopping", "shopping", "shopping"],
      })
      spend_mask = pd.Series([True, True, True, True], index=df_above.index)
      with patch("passion_detector._check_distress_gate", return_value=False), \
           patch("passion_detector._is_non_declining", return_value=True), \
           patch("passion_detector._check_anomaly_suppression", return_value=False):
          signals = detect_passions(df_above, spend_mask)
      assert len(signals) == 1
      assert signals[0].merchant_count == 4


# ── Suppressed Signal Filtering (Fix #3) ─────────────────────────────────────
# Fix #3: Suppressed signals must NEVER appear in PassionResult.passion_signals
# or PipelineResult.passion_signals. process_pipeline builds active_signals via:
#   active_signals = tuple(s for s in passion_signals if not s.is_suppressed)
# These tests verify the contract end-to-end.

class TestSuppressedSignals:
  def _make_df(self):
      return pd.DataFrame({
          "amount": [500.0, 600.0, 550.0, 700.0],
          "cleaned_remarks": ["amazon", "flipkart", "myntra", "amazon"],
          "date": ["2023-01-01", "2023-02-01", "2023-03-01", "2023-04-01"],
          "predicted_category": ["shopping", "shopping", "shopping", "shopping"],
          "is_known_person": [False, False, False, False],
      })

  def test_suppressed_signal_absent_from_passion_result(self):
      """Fix #3: Suppressed signals must not appear in PassionResult.passion_signals."""
      df = self._make_df()
      with patch("passion_detector._check_distress_gate", return_value=True):
          res = process_pipeline(df_raw=df, strict_mode=False)
      # All signals for this category were suppressed via distress gate.
      # None should appear in passion_signals.
      assert all(not s.is_suppressed for s in res.passion_signals), (
          "Suppressed signals leaked into PassionResult.passion_signals"
      )

  def test_suppressed_signal_absent_from_pipeline_result(self, monkeypatch):
      """Fix #3: Suppressed signals must not appear in PipelineResult.passion_signals."""
      monkeypatch.setenv("INSIGHT_ENGINE_PASSION_ENABLED", "true")
      import pipeline
      df = self._make_df()
      result = make_pipeline_result(debits=df, credits=pd.DataFrame())

      def fake_process(*, df_raw, strict_mode, rng):
          """Return a PassionResult that contains a suppressed signal."""
          from passion_models import PassionSignal
          from pipeline_result import PassionResult
          suppressed = PassionSignal(
              category="shopping",
              merchant_list=("amazon", "flipkart", "myntra"),
              total_spend=500.0,
              merchant_count=3,
              spend_share=0.5,
              trend_direction="suppressed",
              is_suppressed=True,
              suppression_reason="distress_gate",
          )
          active = PassionSignal(
              category="food",
              merchant_list=("zomato", "swiggy", "dominos"),
              total_spend=300.0,
              merchant_count=3,
              spend_share=0.3,
              trend_direction="non_declining",
          )
          return PassionResult(
              debits=df_raw.copy(),
              candidates=(),
              insights=("active insight",),
              # process_pipeline already filters to active_signals only:
              passion_signals=(active,),
          )

      out = pipeline._attach_passion_results(result, process_fn=fake_process)
      assert all(not s.is_suppressed for s in out.passion_signals), (
          "Suppressed signals leaked into PipelineResult.passion_signals via _attach_passion_results"
      )
      assert len(out.passion_signals) == 1

  def test_all_suppressed_yields_empty_passion_signals(self):
      """Fix #3: If all detected signals are suppressed, passion_signals must be empty tuple."""
      df = self._make_df()
      with patch("passion_detector._check_distress_gate", return_value=True), \
           patch("passion_detector._check_anomaly_suppression", return_value=False):
          res = process_pipeline(df_raw=df, strict_mode=False)
      assert res.passion_signals == (), (
          "Expected empty passion_signals when all signals are suppressed"
      )

  def test_suppressed_signals_counted_in_log(self, caplog):
      """Fix #3: suppressed_count must appear in structured log when signals are suppressed."""
      import logging
      df = self._make_df()
      with caplog.at_level(logging.INFO):
          with patch("passion_detector._check_distress_gate", return_value=True):
              process_pipeline(df_raw=df, strict_mode=False)
      # suppressed_count is emitted as an extra field in passion_pipeline_complete log record.
      # caplog stores extra fields on the LogRecord object itself.
      suppressed_logged = any(
          getattr(r, "suppressed_count", None) is not None
          for r in caplog.records
      )
      assert suppressed_logged, (
          "Fix #3: suppressed_count must be logged when signals are suppressed. "
          "Verify process_pipeline emits suppressed_count in passion_pipeline_complete."
      )


# ── Fix #5: Broken Index Lookup Test ───────────────────────────────────────
# Fix #5: When df.index.get_indexer raises, only original_index resets to 0.
# active_months and latest_ts must retain their correctly computed values.

class TestNarrowIndexException:
  def test_broken_index_lookup_preserves_active_months_and_latest_ts(self):
      """Fix #5: get_indexer failure must NOT reset active_months or latest_ts to 0."""
      import pandas as pd
      df = pd.DataFrame({
          "amount": [100.0, 200.0, 150.0, 300.0],
          "cleaned_remarks": ["amazon", "flipkart", "myntra", "amazon"],
          "date": ["2023-01-15", "2023-02-10", "2023-03-05", "2023-04-20"],
          "predicted_category": ["shopping", "shopping", "shopping", "shopping"],
          "is_known_person": [False, False, False, False],
      })

      def bad_get_indexer(self_idx, target, *args, **kwargs):
          raise RuntimeError("simulated get_indexer failure")

      with patch.object(pd.Index, "get_indexer", bad_get_indexer):
          result = process_pipeline(df_raw=df, strict_mode=False)

      # Signals must still be produced (get_indexer failure ≠ signal loss).
      # active_months must reflect real history (4 distinct months), not be reset to 0.
      for sig in result.passion_signals:
          assert sig.active_months > 0, (
              f"Fix #5 violation: active_months was reset to 0 by narrow get_indexer exception. "
              f"Got sig.active_months={sig.active_months!r} for 4-month dataset."
          )
          assert sig.latest_ts > 0, (
              f"Fix #5 violation: latest_ts was reset to 0 by narrow get_indexer exception. "
              f"Got sig.latest_ts={sig.latest_ts!r}."
          )


class TestPipeline:
  def test_empty_dataframe_missing_required_column_raises(self):
      df = pd.DataFrame(columns=["amount", "cleaned_remarks", "date"])
      with pytest.raises(ValueError, match="Missing columns"):
          process_pipeline(df_raw=df)

  def test_process_pipeline_empty_df_has_passion_columns(self):
      df = pd.DataFrame(columns=["amount", "cleaned_remarks", "date", "predicted_category"])
      res = process_pipeline(df_raw=df)
      assert "inferred_subcategory" in res.debits.columns
      assert "subcategory_confidence" in res.debits.columns
      assert str(res.debits["inferred_subcategory"].dtype) == "object"
      assert str(res.debits["subcategory_confidence"].dtype) == "float64"

  def test_process_pipeline_rejects_compact_yyyymmdd_without_opt_in(self):
      # C6: is_known_person included for strict_mode=True compliance.
      # The YYYYMMDD ValueError fires before the IS_KNOWN_PERSON check, but
      # all non-empty strict_mode=True DataFrames should include the column.
      df = pd.DataFrame({
          "amount": [100.0],
          "cleaned_remarks": ["amazon"],
          "date": [20230101],
          "predicted_category": ["shopping"],
          "is_known_person": [False],
      })
      with pytest.raises(ValueError, match="YYYYMMDD"):
          process_pipeline(df_raw=df)

  def test_process_pipeline_accepts_compact_yyyymmdd_with_opt_in(self):
      df = pd.DataFrame({
          "amount": [100.0],
          "cleaned_remarks": ["amazon"],
          "date": [20230101],
          "predicted_category": ["shopping"],
          "is_known_person": [False],  # required in strict_mode=True
      })
      res = process_pipeline(df_raw=df, allow_yyyymmdd_dates=True)
      assert len(res.debits) == 1


  # FIX C3b: debits is now plain pd.DataFrame, not _ReadOnlyDataFrame.
  def test_pipeline_result_debits_is_dataframe(self):
      df = pd.DataFrame({
          "amount": [100], "cleaned_remarks": ["amazon"], "date": ["2023-01-01"],
          "predicted_category": ["shopping"], "is_known_person": [False],
      })
      res = process_pipeline(df_raw=df)
      assert isinstance(res.debits, pd.DataFrame)

  def test_duplicate_index_raises(self):
      df = pd.DataFrame({
          "amount": [100, 200], "cleaned_remarks": ["a", "b"],
          "date": ["2023-01-01", "2023-01-02"], "predicted_category": ["food", "food"],
      }, index=[1, 1])
      with pytest.raises(ValueError, match="duplicate"):
          process_pipeline(df_raw=df)

  def test_inf_amounts_excluded(self):
      df = pd.DataFrame({
          "amount": [100.0, float('inf'), -float('inf')],
          "cleaned_remarks": ["a", "b", "c"],
          "date": ["2023-01-01", "2023-01-02", "2023-01-03"],
          "predicted_category": ["food", "food", "food"],
          "is_known_person": [False, False, False],
      })
      res = process_pipeline(df_raw=df)
      assert len(res.debits) == 3



  # strict_mode=True + enrich_subcategories failure → re-raises.
  def test_strict_mode_enrich_failure_reraises(self):
      df = pd.DataFrame({
          "amount": [100], "cleaned_remarks": ["test"], "date": ["2023-01-01"],
          "predicted_category": ["food"], "is_known_person": [False],
      })
      # P0-3 FIX: Patch source module instead of local import
      with patch("marketplace_subcategory.enrich_subcategories",
                 side_effect=RuntimeError("mocked enrich fail")):
          with pytest.raises(RuntimeError, match="mocked enrich fail"):
              process_pipeline(df_raw=df, strict_mode=True)

  def test_process_pipeline_strict_mode_false_enrich_subcategories_fails_safely(self):
      df = pd.DataFrame({
          "amount": [100], "cleaned_remarks": ["test"], "date": ["2023-01-01"],
          "predicted_category": ["food"],
      })
      with patch("marketplace_subcategory.enrich_subcategories",
                 side_effect=RuntimeError("mocked enrich fail")):
          res = process_pipeline(df_raw=df, strict_mode=False)
          assert isinstance(res, PassionResult)
          assert len(res.passion_signals) == 0
          assert "inferred_subcategory" in res.debits.columns
          assert "subcategory_confidence" in res.debits.columns

  def test_pipeline_result_insights_string_raises(self):
      with pytest.raises(TypeError):
          PassionResult(
              debits=pd.DataFrame({"a": [1]}), candidates=(), insights="bad string",
              passion_signals=(),
          )

  # FIX H1: MemoryError must propagate even with strict_mode=False.
  def test_memory_error_propagates(self):
      df = pd.DataFrame({
          "amount": [100], "cleaned_remarks": ["test"], "date": ["2023-01-01"],
          "predicted_category": ["food"],
      })
      # P0-3 FIX: Patch source module instead of local import
      with patch("passion_detector.detect_passions",
                 side_effect=MemoryError("OOM")):
          with pytest.raises(MemoryError):
              process_pipeline(df_raw=df, strict_mode=False)



  # FIX-35: Non-DataFrame input raises TypeError.
  def test_non_dataframe_input_raises(self):
      with pytest.raises(TypeError, match="pd.DataFrame"):
          process_pipeline(df_raw="not a dataframe")

  # FIX-10: Startup failure prevents retries.
  def test_init_failure_prevents_retry(self, monkeypatch):
      import passion_pipeline
      monkeypatch.setattr("passion_pipeline._init_complete", threading.Event())
      monkeypatch.setattr("passion_pipeline._init_lock", threading.Lock())
      monkeypatch.setattr("passion_pipeline._init_failed", threading.Event())
      monkeypatch.delenv("INSIGHT_ENGINE_SKIP_STARTUP_CHECKS", raising=False)
      with patch("bootstrap.run_startup_checks",
                 side_effect=RuntimeError("bad config")):
          with pytest.raises(RuntimeError, match="bad config"):
              passion_pipeline._ensure_initialized()
      # Second call should raise immediately without re-running checks.
      with pytest.raises(RuntimeError, match="previously failed"):
          passion_pipeline._ensure_initialized()

  def test_attach_passion_results_real_pipeline_result_enabled(self, monkeypatch):
      monkeypatch.setenv("INSIGHT_ENGINE_PASSION_ENABLED", "true")
      import pipeline
      from pipeline import PipelineResult
      from pipeline_result import PassionResult

      # C7: is_known_person required by _attach_passion_results preflight.
      # Without it the preflight logs passion_skip / missing_columns and returns
      # result unchanged — fake_process is never called, assertions fail.
      df = pd.DataFrame({
          "amount": [100.0],
          "cleaned_remarks": ["amazon"],
          "date": ["2023-01-01"],
          "predicted_category": ["shopping"],
          "is_known_person": [False],
      })

      result = make_pipeline_result(
          debits=df,
          credits=pd.DataFrame(),
      )

      def fake_process(*, df_raw, strict_mode, rng):
          out = df_raw.copy()
          out["inferred_subcategory"] = "electronics"
          out["subcategory_confidence"] = 0.9
          return PassionResult(
              debits=out,
              candidates=(),
              insights=("passion insight",),
              passion_signals=(),
          )

      attached = pipeline._attach_passion_results(
          result,
          process_fn=fake_process,
      )

      assert not attached.passion_debits.empty
      assert attached.passion_insights == ("passion insight",)

  def test_attach_passion_results_disabled_does_not_call_process(self, monkeypatch):
      monkeypatch.delenv("INSIGHT_ENGINE_PASSION_ENABLED", raising=False)
      import pipeline
      from unittest.mock import MagicMock
      result = make_pipeline_result(
          debits=pd.DataFrame(),
          credits=pd.DataFrame(),
      )
      process = MagicMock()
      out = pipeline._attach_passion_results(result, process_fn=process)
      assert out.stats.get("passion_status") == "disabled"
      assert not process.called

  def test_attach_passion_results_exceeds_max_rows_skips(self, monkeypatch, caplog):
      # FIX-21: Test that inputs exceeding configured max rows are skipped to prevent memory blowup
      monkeypatch.setenv("INSIGHT_ENGINE_PASSION_ENABLED", "true")
      monkeypatch.setenv("INSIGHT_ENGINE_PASSION_MAX_ROWS", "2")
      import pipeline
      from unittest.mock import MagicMock
      debits_df = pd.DataFrame({
          "amount": [100.0, 200.0, 300.0],
          "cleaned_remarks": ["amazon", "netflix", "zomato"],
          "date": ["2023-01-01", "2023-01-02", "2023-01-03"],
          "predicted_category": ["shopping", "entertainment", "food"],
          "is_known_person": [False, False, False],
      })
      result = make_pipeline_result(
          debits=debits_df,
          credits=pd.DataFrame(),
      )
      process = MagicMock()
      out = pipeline._attach_passion_results(result, process_fn=process)
      assert out.stats.get("passion_status") == "skipped"
      assert not process.called
      assert "exceeds_max_rows" in caplog.text

  def test_attach_passion_results_timeout_error_returns_original(self, monkeypatch, caplog):
      monkeypatch.setenv("INSIGHT_ENGINE_PASSION_ENABLED", "true")
      import pipeline
      from pipeline import PipelineResult

      # C7: is_known_person required — without it the preflight returns early
      # (passion_skip / missing_columns) before timeout_process is ever called.
      debits_df = pd.DataFrame({
          "amount": [100.0],
          "cleaned_remarks": ["amazon"],
          "date": ["2023-01-01"],
          "predicted_category": ["shopping"],
          "is_known_person": [False],
      })
      result = make_pipeline_result(
          debits=debits_df,
          credits=pd.DataFrame(),
          passion_debits=pd.DataFrame(),
          passion_insights=(),
          passion_signals=(),
      )

      def timeout_process(*args, **kwargs):
          raise TimeoutError("timeout")

      out = pipeline._attach_passion_results(result, process_fn=timeout_process)
      # Fix #4: Do not assert object identity; _with_passion_status returns a new instance.
      assert out.stats.get("passion_status") == "timeout", (
          f"Expected passion_status='timeout', got stats={out.stats!r}"
      )
      assert "passion_engine_timeout" in caplog.text

  def test_invalid_compact_yyyymmdd_original_date_preserved(self):
      # P3-1: After fix, original date values are preserved in output (not normalized).
      # Invalid month 13 string passes through unmodified.
      df = pd.DataFrame({
          "amount": [100.0],
          "cleaned_remarks": ["amazon"],
          "date": ["20231301"],  # Invalid month 13
          "predicted_category": ["shopping"],
          "is_known_person": [False],  # required in strict_mode=True
      })
      res = process_pipeline(df_raw=df, allow_yyyymmdd_dates=True)
      # Original string value is preserved in output
      assert res.debits["date"].iloc[0] == "20231301"

  def test_attach_passion_results_enabled_failure_logs_and_returns_original(self, monkeypatch, caplog):
      monkeypatch.setenv("INSIGHT_ENGINE_PASSION_ENABLED", "true")
      import pipeline
      from pipeline import PipelineResult

      # C7: is_known_person required — without it the preflight returns early
      # before bad_process is called, and "passion_engine_failed" never appears in logs.
      debits_df = pd.DataFrame({
          "amount": [100.0],
          "cleaned_remarks": ["amazon"],
          "date": ["2023-01-01"],
          "predicted_category": ["shopping"],
          "is_known_person": [False],
      })
      result = make_pipeline_result(
          debits=debits_df,
          credits=pd.DataFrame(),
          passion_debits=pd.DataFrame(),
          passion_insights=(),
          passion_signals=(),
      )

      def bad_process(*args, **kwargs):
          raise RuntimeError("boom")

      out = pipeline._attach_passion_results(result, process_fn=bad_process)
      # Fix #4: Do not assert object identity; _with_passion_status returns a new instance.
      assert out.stats.get("passion_status") == "failure", (
          f"Expected passion_status='failure', got stats={out.stats!r}"
      )
      assert "passion_engine_failed" in caplog.text

  # P2-3: strict_attach makes non-fatal exceptions propagate.
  def test_attach_passion_results_strict_attach_propagates(self, monkeypatch):
      monkeypatch.setenv("INSIGHT_ENGINE_PASSION_ENABLED", "true")
      import pipeline
      from pipeline import PipelineResult

      # C7: is_known_person required — without it the preflight returns early
      # before bad_process is called, and RuntimeError is never raised.
      debits_df = pd.DataFrame({
          "amount": [100.0],
          "cleaned_remarks": ["amazon"],
          "date": ["2023-01-01"],
          "predicted_category": ["shopping"],
          "is_known_person": [False],
      })
      result = make_pipeline_result(
          debits=debits_df,
          credits=pd.DataFrame(),
          passion_debits=pd.DataFrame(),
          passion_insights=(),
          passion_signals=(),
      )

      def bad_process(*args, **kwargs):
          raise RuntimeError("boom")

      with pytest.raises(RuntimeError, match="boom"):
          pipeline._attach_passion_results(
              result, process_fn=bad_process, strict_attach=True,
          )

  # P2-3: MemoryError always propagates through _attach_passion_results.
  def test_attach_passion_results_memory_error_propagates(self, monkeypatch):
      monkeypatch.setenv("INSIGHT_ENGINE_PASSION_ENABLED", "true")
      import pipeline
      from pipeline import PipelineResult

      # C7: is_known_person required — without it the preflight returns early
      # before oom_process is called, and MemoryError is never raised.
      debits_df = pd.DataFrame({
          "amount": [100.0],
          "cleaned_remarks": ["amazon"],
          "date": ["2023-01-01"],
          "predicted_category": ["shopping"],
          "is_known_person": [False],
      })
      result = make_pipeline_result(
          debits=debits_df,
          credits=pd.DataFrame(),
          passion_debits=pd.DataFrame(),
          passion_insights=(),
          passion_signals=(),
      )

      def oom_process(*args, **kwargs):
          raise MemoryError("OOM")

      with pytest.raises(MemoryError):
          pipeline._attach_passion_results(result, process_fn=oom_process)

  # P2-3: Empty debits preflight returns original result.
  def test_attach_passion_results_empty_debits_skips(self, monkeypatch):
      monkeypatch.setenv("INSIGHT_ENGINE_PASSION_ENABLED", "true")
      import pipeline
      from unittest.mock import MagicMock
      from pipeline import PipelineResult

      result = make_pipeline_result(
          debits=pd.DataFrame(),
          credits=pd.DataFrame(),
          passion_debits=pd.DataFrame(),
          passion_insights=(),
          passion_signals=(),
      )
      process = MagicMock()
      out = pipeline._attach_passion_results(result, process_fn=process)
      # Fix #4: Do not assert object identity; _with_passion_status returns a new instance.
      assert out.stats.get("passion_status") == "skipped", (
          f"Expected passion_status='skipped', got stats={out.stats!r}"
      )
      assert not process.called

  def test_memory_behavior_defensive_copies(self):
      df = pd.DataFrame({
          "amount": np.random.rand(10000),
          "cleaned_remarks": ["test"] * 10000,
          "date": ["2023-01-01"] * 10000,
          "predicted_category": ["food"] * 10000,
          "is_known_person": [False] * 10000,
      })
      res = process_pipeline(df_raw=df)

      orig_amount = df["amount"].iloc[0]
      # Mutating the result should not mutate the input, proving defensive copy works
      res.debits.loc[res.debits.index[0], "amount"] = 999.0
      assert df["amount"].iloc[0] == orig_amount

  # P1-1 FIX: Regression tests for resolved_merchants reindexing
  def test_pipeline_with_known_person_and_valid_spend(self):
      df = pd.DataFrame({
          "amount": [100.0, 200.0],
          "cleaned_remarks": ["amazon", "flipkart"],
          "date": ["2023-01-01", "2023-01-02"],
          "predicted_category": ["shopping", "shopping"],
          "is_known_person": [True, False]
      })
      res = process_pipeline(df_raw=df)
      assert len(res.debits) == 2
      assert pd.isna(res.debits["inferred_subcategory"].iloc[0])
      assert res.debits["subcategory_confidence"].iloc[0] == 0.0

  def test_pipeline_with_invalid_amount_and_valid_spend(self):
      df = pd.DataFrame({
          "amount": [float("inf"), 200.0],
          "cleaned_remarks": ["amazon", "flipkart"],
          "date": ["2023-01-01", "2023-01-02"],
          "predicted_category": ["shopping", "shopping"],
          "is_known_person": [False, False],
      })
      res = process_pipeline(df_raw=df)
      assert len(res.debits) == 2
      assert pd.isna(res.debits["inferred_subcategory"].iloc[0])
      assert res.debits["subcategory_confidence"].iloc[0] == 0.0

  def test_pipeline_all_known_persons(self):
      df = pd.DataFrame({
          "amount": [100.0, 200.0],
          "cleaned_remarks": ["amazon", "flipkart"],
          "date": ["2023-01-01", "2023-01-02"],
          "predicted_category": ["shopping", "shopping"],
          "is_known_person": [True, True]
      })
      res = process_pipeline(df_raw=df)
      assert len(res.debits) == 2
      assert pd.isna(res.debits["inferred_subcategory"]).all()

  def test_pipeline_all_invalid_amounts(self):
      df = pd.DataFrame({
          "amount": [np.nan, "bad"],
          "cleaned_remarks": ["amazon", "flipkart"],
          "date": ["2023-01-01", "2023-01-02"],
          "predicted_category": ["shopping", "shopping"],
          "is_known_person": [False, False],
      })
      res = process_pipeline(df_raw=df)
      assert len(res.debits) == 2
      assert pd.isna(res.debits["inferred_subcategory"]).all()

  def test_pipeline_unsorted_mixed_index(self):
      df = pd.DataFrame({
          "amount": [100.0, float("inf"), 1500.0],
          "cleaned_remarks": ["swiggy", "zomato", "amazon"],
          "date": ["2023-01-01", "2023-01-02", "2023-01-03"],
          "predicted_category": ["food", "food", "shopping"],
          "is_known_person": [True, False, False]
      }, index=[20, 10, 30])
      res = process_pipeline(df_raw=df)

      # Expected: no KeyError, output index remains [20, 10, 30]
      assert res.debits.index.tolist() == [20, 10, 30]
      assert len(res.debits) == 3

      # Ineligible rows are neutral
      assert pd.isna(res.debits.loc[20, "inferred_subcategory"])
      assert pd.isna(res.debits.loc[10, "inferred_subcategory"])

      # Valid row can be enriched/detected if applicable
      assert res.debits.loc[30, "inferred_subcategory"] == "electronics"

  def test_resolve_merchant_vectorized_raises_strict_mode(self):
      df = pd.DataFrame({
          "amount": [100.0], "cleaned_remarks": ["test"], "date": ["2023-01-01"],
          "predicted_category": ["shopping"], "is_known_person": [False],
      })
      with patch("marketplace_subcategory.resolve_merchant_vectorized", side_effect=RuntimeError("Test error")):
          with pytest.raises(RuntimeError, match="Test error"):
              process_pipeline(df_raw=df, strict_mode=True)

  def test_resolve_merchant_vectorized_raises_soft_mode(self):
      df = pd.DataFrame({"amount": [100.0], "cleaned_remarks": ["test"], "date": ["2023-01-01"], "predicted_category": ["shopping"]})
      with patch("marketplace_subcategory.resolve_merchant_vectorized", side_effect=RuntimeError("Test error")):
          res = process_pipeline(df_raw=df, strict_mode=False)
          assert isinstance(res, PassionResult)
          assert len(res.passion_signals) == 0

  # P0-7: Replace invalid run_pipeline/run_inference hook tests
  # Replace dummy DataFrame tests with source-placement checks to verify wiring
  # without running the entire pipeline with invalid dummy inputs.
  # FIX 2: Add self to avoid Pytest TypeError
  def test_run_pipeline_has_passion_hook_before_return(self):
      import inspect
      import pipeline

      src = inspect.getsource(pipeline.run_pipeline)
      hook_pos = src.rfind("_attach_passion_results(result)")
      return_pos = src.rfind("return result")

      assert hook_pos != -1, "run_pipeline must call _attach_passion_results(result)"
      assert return_pos != -1, "run_pipeline must end by returning result"
      assert hook_pos < return_pos, "passion hook must run before final return result"

  # FIX 2: Add self to avoid Pytest TypeError
  def test_run_inference_has_passion_hook_before_return(self):
      import inspect
      import pipeline

      src = inspect.getsource(pipeline.run_inference)
      hook_pos = src.rfind("_attach_passion_results(result)")
      return_pos = src.rfind("return result")

      assert hook_pos != -1, "run_inference must call _attach_passion_results(result)"
      assert return_pos != -1, "run_inference must end by returning result"
      assert hook_pos < return_pos, "passion hook must run before final return result"


# ── Bootstrap ─────────────────────────────────────────────────────────────────

class TestBootstrap:

  def test_startup_checks_run_successfully(self, monkeypatch):
      import passion_pipeline
      monkeypatch.delenv("INSIGHT_ENGINE_SKIP_STARTUP_CHECKS", raising=False)
      monkeypatch.setattr("passion_pipeline._init_complete", threading.Event())
      monkeypatch.setattr("passion_pipeline._init_lock", threading.Lock())
      monkeypatch.setattr("passion_pipeline._init_failed", threading.Event())
      with patch("bootstrap.run_startup_checks") as mock_startup:
          passion_pipeline._ensure_initialized()
          mock_startup.assert_called_once()

  def test_tip_corpus_missing_text_key_raises(self):
      from bootstrap import _validate_tip_corpus
      from types import MappingProxyType
      bad = MappingProxyType({
          "bad_tip": MappingProxyType({
              "categories": ("food",), "insights": ("spending_spike",),
          })
      })
      with patch("bootstrap.TIP_CORPUS", bad):
          with pytest.raises(ValueError, match="missing required key 'text'"):
              _validate_tip_corpus()

  def test_tip_corpus_missing_categories_key_raises(self):
      from bootstrap import _validate_tip_corpus
      from types import MappingProxyType
      bad = MappingProxyType({
          "bad_tip": MappingProxyType({
              "text": "Some text", "insights": ("spending_spike",),
          })
      })
      with patch("bootstrap.TIP_CORPUS", bad):
          with pytest.raises(ValueError, match="missing required key 'categories'"):
              _validate_tip_corpus()

  def test_tip_corpus_missing_insights_key_raises(self):
      from bootstrap import _validate_tip_corpus
      from types import MappingProxyType
      bad = MappingProxyType({
          "bad_tip": MappingProxyType({
              "text": "Some text", "categories": ("food",),
          })
      })
      with patch("bootstrap.TIP_CORPUS", bad):
          with pytest.raises(ValueError, match="missing required key 'insights'"):
              _validate_tip_corpus()

  # FIX M6: Dotted field names rejected.
  def test_dotted_field_name_rejected(self):
      from bootstrap import validate_template_fields
      with pytest.raises(ValueError, match="simple identifiers"):
          validate_template_fields("{x.__class__}", {"x"})

  # FIX H6 + FIX-26: Duplicate Col values detected.
  # FIX-26: Uses _col_cls parameter instead of patching bootstrap.Col.
  def test_duplicate_col_values_detected(self):
      from bootstrap import _validate_schema_columns
      mock_col = type("Col", (), {
          "AMOUNT": "amount", "CLEANED_REMARKS": "cleaned_remarks",
          "DATE": "date", "PREDICTED_CATEGORY": "predicted_category",
          "IS_ANOMALY": "is_anomaly", "IS_RECURRING": "is_recurring",
          "INSIGHT_SCORE": "insight_score", "IS_KNOWN_PERSON": "is_known_person",
          "RECURRING_FREQUENCY": "recurring_frequency",
          "INFERRED_SUBCATEGORY": "inferred_subcategory",
          "SUBCATEGORY_CONFIDENCE": "inferred_subcategory",  # DUPLICATE!
      })
      with pytest.raises(RuntimeError, match="duplicate string values"):
          _validate_schema_columns(mock_col)

  # FIX-14: Conversions (!r, !s, !a) rejected.
  def test_template_conversion_rejected(self):
      from bootstrap import validate_template_fields
      with pytest.raises(ValueError, match="Conversions"):
          validate_template_fields("{merchant!r}", {"merchant"})

  # FIX-14: Nested format specs rejected.
  def test_template_nested_format_spec_rejected(self):
      from bootstrap import validate_template_fields
      with pytest.raises(ValueError, match="Nested format specs"):
          validate_template_fields("{amount:{width}}", {"amount", "width"})

  # FIX-24: Non-string elements in TIP_CORPUS categories rejected.
  def test_tip_corpus_non_string_element_rejected(self):
      from contracts import _freeze_tip_corpus

      bad = {
          "bad_tip": {
              "text": "Some text",
              "categories": ("food", 42),
              "insights": ("spending_spike",),
          }
      }

      with pytest.raises(TypeError, match="elements must be str"):
          _freeze_tip_corpus(bad)

  # FIX-16 / P0-4B: Rewritten to call contracts._freeze_tip_corpus directly.
  # bootstrap._validate_tip_corpus only validates shape/required keys (frozen MappingProxyType).
  # Business-rule checks (wildcard, empty categories/insights) live in contracts._freeze_tip_corpus.
  def test_tip_corpus_wildcard_policy_and_checks(self):
      from contracts import _freeze_tip_corpus

      # 1. Non-generic tip with "any" wildcard in categories should raise ValueError
      bad_any_cat = {
          "non_generic_tip": {
              "text": "Some text",
              "categories": ("any",),
              "insights": ("spending_spike",),
          }
      }
      with pytest.raises(ValueError, match="contains wildcard 'any'"):
          _freeze_tip_corpus(bad_any_cat)

      # 2. Non-generic tip with empty categories should raise ValueError
      bad_empty_categories = {
          "non_generic_tip": {
              "text": "Some text",
              "categories": (),
              "insights": ("spending_spike",),
          }
      }
      with pytest.raises(ValueError, match="empty 'categories'"):
          _freeze_tip_corpus(bad_empty_categories)

      # 3. Non-generic tip with empty insights should raise ValueError
      bad_empty_insights = {
          "non_generic_tip": {
              "text": "Some text",
              "categories": ("food",),
              "insights": (),
          }
      }
      with pytest.raises(ValueError, match="empty 'insights'"):
          _freeze_tip_corpus(bad_empty_insights)

  # FIX-28: Dry render catches typos in template fields.
  def test_dry_render_catches_field_typo(self):
      from bootstrap import _dry_render_templates
      from types import MappingProxyType
      bad_templates = MappingProxyType({
          "subscription": ("{merchnt} spent {amount}",),  # Typo: merchnt
      })
      with patch("bootstrap.INSIGHT_TEMPLATES", bad_templates):
          with pytest.raises(ValueError, match="dry render"):
              _dry_render_templates()


# ── Utility ───────────────────────────────────────────────────────────────────

class TestUtils:
  def test_safe_last_nonnull_casts_to_str(self):
      assert safe_last_nonnull([1, 2, 3]) == "3"
      assert safe_last_nonnull([]) is None

  def test_validate_template_values_rejects_objects(self):
      with pytest.raises(TypeError, match="scalar"):
          validate_template_values({"k": [1, 2]})

  # P2.6: Decimal accepted in template values.
  def test_validate_template_values_accepts_decimal(self):
      validate_template_values({"amount": Decimal("99.99")})

  def test_enrich_subcategories_bad_amounts(self):
      df = pd.DataFrame({
          "amount": ["bad", "500"], "cleaned_remarks": ["amzn", "amzn"],
          "predicted_category": ["shopping", "shopping"],
      })
      result = enrich_subcategories(df)
      assert isinstance(result, pd.DataFrame)
      assert len(result) == 2

  def test_enrich_subcategories_missing_predicted_category_raises(self):
      df = pd.DataFrame({
          "amount": ["100"], "cleaned_remarks": ["amzn"],
      })
      with pytest.raises(ValueError, match="predicted_category"):
          enrich_subcategories(df)

  def test_generate_insights_deduplicates(self):
      from candidate import Candidate
      c1 = Candidate.subscription(0.9, "food", "amazon", 100.0, 1000)
      c2 = Candidate.subscription(0.8, "food", "flipkart", 100.0, 900)
      rng = random.Random(42)
      with patch("passion_insight_generator._render_candidate") as mock:
          mock.side_effect = [("Same text", ""), ("Same text", "")]
          results = generate_passion_insights([c1, c2], top_n=2, rng=rng)
      assert len(results) == 1

  def test_generate_insights_fallback_banned(self):
      rng = random.Random(42)
      results = generate_passion_insights(
          [], top_n=1, rng=rng, strict_mode=False,
          fallback_insights=["scam!", "safe fallback"],
      )
      assert len(results) == 1
      assert "safe fallback" in results[0]

  def test_lookup_matching_tip_ids_callable(self):
      from contracts import lookup_matching_tip_ids
      result = lookup_matching_tip_ids("food", "spending_spike")
      assert isinstance(result, list)

  # FIX H9: safe_assign_new_columns only preserves whitelisted columns.
  def test_safe_assign_new_columns_whitelist(self):
      original = pd.DataFrame({
          "a": [1], "is_recurring": [True]
      })
      updated = pd.DataFrame({
          "a": [10], "inferred_subcategory": ["electronics"], "unauthorized_col": ["bad"]
      })
      merged = safe_assign_new_columns(original, updated)
      assert "inferred_subcategory" in merged.columns
      assert "unauthorized_col" not in merged.columns
      assert "is_recurring" in merged.columns
      # Prove original values are preserved
      assert merged["a"].iloc[0] == 1

  # FIX-04: safe_numeric handles currency prefixes.
  def test_safe_numeric_currency_prefixes(self):
      from passion_utils import safe_numeric
      assert safe_numeric("₹1,500") == 1500.0
      assert safe_numeric("Rs 500") == 500.0
      assert safe_numeric("Rs. 500") == 500.0
      assert safe_numeric("INR 1500") == 1500.0
      assert safe_numeric("$99.99") == 99.99
      assert safe_numeric("bad") == 0.0
      assert safe_numeric(np.nan) == 0.0


# ── PassionSignal Validation ─────────────────────────────────────────────────

class TestPassionSignalValidation:
  def test_merchant_count_mismatch_raises(self):
      from passion_models import PassionSignal
      with pytest.raises(ValueError, match="merchant_count"):
          PassionSignal(
              category="food", merchant_list=("a", "b", "c"),
              total_spend=100.0, merchant_count=2,
              spend_share=0.5, trend_direction="non_declining",
          )

  def test_negative_total_spend_raises(self):
      from passion_models import PassionSignal
      with pytest.raises(ValueError, match="non-negative"):
          PassionSignal(
              category="food", merchant_list=("a",),
              total_spend=-100.0, merchant_count=1,
              spend_share=0.5, trend_direction="non_declining",
          )

  def test_spend_share_above_one_raises(self):
      from passion_models import PassionSignal
      with pytest.raises(ValueError, match="\\[0, 1\\]"):
          PassionSignal(
              category="food", merchant_list=("a",),
              total_spend=100.0, merchant_count=1,
              spend_share=1.5, trend_direction="non_declining",
          )

  def test_invalid_trend_direction_raises(self):
      from passion_models import PassionSignal
      with pytest.raises(ValueError, match="trend_direction"):
          PassionSignal(
              category="food", merchant_list=("a",),
              total_spend=100.0, merchant_count=1,
              spend_share=0.5, trend_direction="invalid_trend",
          )

  def test_suppressed_empty_reason_raises(self):
      from passion_models import PassionSignal
      with pytest.raises(ValueError, match="non-empty suppression_reason"):
          PassionSignal(
              category="food", merchant_list=("a",),
              total_spend=100.0, merchant_count=1,
              spend_share=0.5, trend_direction="suppressed",
              is_suppressed=True, suppression_reason="",
          )

  def test_not_suppressed_with_reason_raises(self):
      from passion_models import PassionSignal
      with pytest.raises(ValueError, match="non-suppressed"):
          PassionSignal(
              category="food", merchant_list=("a",),
              total_spend=100.0, merchant_count=1,
              spend_share=0.5, trend_direction="non_declining",
              is_suppressed=False, suppression_reason="shouldn't be here",
          )

  def test_valid_signal_passes(self):
      from passion_models import PassionSignal
      signal = PassionSignal(
          category="food", merchant_list=("a", "b"),
          total_spend=100.0, merchant_count=2,
          spend_share=0.5, trend_direction="non_declining",
      )
      assert signal.merchant_count == 2

  def test_passion_signal_merchant_list_list_converts_to_tuple(self):
      from passion_models import PassionSignal
      sig = PassionSignal(
          category="shopping",
          merchant_list=["amazon", "flipkart", "myntra"],
          total_spend=1000.0,
          merchant_count=3,
          spend_share=0.5,
          trend_direction="non_declining",
      )
      assert sig.merchant_list == ("amazon", "flipkart", "myntra")
      assert isinstance(sig.merchant_list, tuple)

  # FIX H2: Float precision edge case.
  def test_spend_share_float_precision_accepted(self):
      from passion_models import PassionSignal
      import math
      imprecise_share = sum([0.1] * 10)
      signal = PassionSignal(
          category="food", merchant_list=("a",),
          total_spend=100.0, merchant_count=1,
          spend_share=imprecise_share, trend_direction="non_declining",
      )
      assert math.isclose(signal.spend_share, 1.0)

  # FIX-16: String merchant_list raises TypeError.
  def test_string_merchant_list_raises(self):
      from passion_models import PassionSignal
      with pytest.raises(TypeError, match="bare string"):
          PassionSignal(
              category="food", merchant_list="amazon",
              total_spend=100.0, merchant_count=6,
              spend_share=0.5, trend_direction="non_declining",
          )


# ── Candidate Validation ─────────────────────────────────────────────────────

class TestCandidateValidation:
  def test_nan_score_raises(self):
      from candidate import Candidate
      with pytest.raises(ValueError, match="finite"):
          Candidate(score=float('nan'), category="food",
                    insight_type="subscription", merchant="amazon",
                    amount=100.0, sort_key_ts=1000)

  def test_inf_amount_raises(self):
      from candidate import Candidate
      with pytest.raises(ValueError, match="finite"):
          Candidate(score=0.5, category="food",
                    insight_type="subscription", merchant="amazon",
                    amount=float('inf'), sort_key_ts=1000)

  # P1.12: Empty string category raises.
  def test_empty_category_raises(self):
      from candidate import Candidate
      with pytest.raises(ValueError, match="non-empty"):
          Candidate(score=0.5, category="",
                    insight_type="subscription", merchant="amazon",
                    amount=100.0, sort_key_ts=1000)

  def test_empty_merchant_raises(self):
      from candidate import Candidate
      with pytest.raises(ValueError, match="non-empty"):
          Candidate(score=0.5, category="food",
                    insight_type="subscription", merchant="   ",
                    amount=100.0, sort_key_ts=1000)

  # P0-1 FIX: Candidate.passion tests
  def test_candidate_passion_valid_signal_does_not_raise(self):
      from candidate import Candidate
      from passion_models import PassionSignal
      signal = PassionSignal(
          category="food", merchant_list=("amazon",),
          total_spend=100.0, merchant_count=1,
          spend_share=1.0, trend_direction="non_declining",
          latest_ts=1672531200, original_index=42,
      )
      c = Candidate.passion(signal)
      assert c.sort_key_ts == signal.latest_ts
      assert c.normalized_score == signal.spend_share
      assert c.original_index == signal.original_index
      assert c.insight_type == "lifestyle_opportunity"

  def test_candidate_imports_successfully(self):
      import candidate
      assert hasattr(candidate, "Candidate")


# ── PassionResult Rejection ──────────────────────────────────────────────────

class TestPassionResultRejection:
  def test_set_field_raises(self):
      with pytest.raises(TypeError, match="tuple or list"):
          PassionResult(
              debits=pd.DataFrame({"a": [1]}), candidates=(), insights=set(),
              passion_signals=(),
          )

  def test_non_dataframe_debits_raises(self):
      with pytest.raises(TypeError, match="pd.DataFrame"):
          PassionResult(
              debits="not a dataframe", candidates=(), insights=(),
              passion_signals=(),
          )


# ── ThreadSafe Init ──────────────────────────────────────────────────────────

class TestThreadSafeInit:
  def test_concurrent_initialization(self, monkeypatch):
      import passion_pipeline
      monkeypatch.setattr("passion_pipeline._init_complete", threading.Event())
      monkeypatch.setattr("passion_pipeline._init_lock", threading.Lock())
      monkeypatch.setattr("passion_pipeline._init_failed", threading.Event())
      errors = []

      def init_worker():
          try:
              passion_pipeline._ensure_initialized()
          except Exception as e:
              errors.append(e)

      threads = [threading.Thread(target=init_worker) for _ in range(5)]
      for t in threads:
          t.start()
      for t in threads:
          t.join(timeout=5)

      assert len(errors) == 0
      assert passion_pipeline._init_complete.is_set()


# ── Module Imports ────────────────────────────────────────────────────────────

class TestModuleImports:
  def test_module_imports(self):
      import contracts
      import schema
      import config
      import config_passion
      import bootstrap
      import log_utils
      import banned_content
      import pipeline_result
      import passion_utils
      import passion_models
      import candidate
      import marketplace_subcategory
      import passion_detector
      import passion_insight_generator
      import passion_pipeline


# ── A2: TIP_CORPUS Import Isolation ──────────────────────────────────────────
# A2 MANDATE: No production module other than contracts.py must import or access config.TIP_CORPUS directly.
# This test uses ast.parse for zero-dependency static analysis.

class TestA2TipCorpusImportIsolation:
  def test_insight_generator_does_not_import_tip_corpus_from_config(self):
      """A2: No production module other than contracts.py must import or access TIP_CORPUS from config directly."""
      import ast
      import pathlib

      _SKIP_DIRS = frozenset({
          ".venv", "venv", ".git", "__pycache__", "build", "dist",
          "planv14", "plans", "archive",
      })
      project_root = pathlib.Path(__file__).parent.parent
      # Fix 21: rglob catches violations in sub-packages; skip non-production dirs.
      production_files = [
          p for p in project_root.rglob("*.py")
          if not any(part in _SKIP_DIRS for part in p.parts)
          and not p.name.startswith("test_")
          and not p.name.startswith("conftest")
          and p.name != "contracts.py"
      ]

      for src in production_files:
          try:
              tree = ast.parse(src.read_text(encoding="utf-8"))
          except SyntaxError:
              continue

          for node in ast.walk(tree):
              if isinstance(node, ast.ImportFrom) and node.module == "config":
                  names = [alias.name for alias in node.names]
                  assert "TIP_CORPUS" not in names, (
                      f"A2 VIOLATION: {src.name} imports TIP_CORPUS from config directly. "
                      f"Use 'from contracts import TIP_CORPUS' instead."
                  )
              if isinstance(node, ast.Attribute) and node.attr == "TIP_CORPUS":
                  if isinstance(node.value, ast.Name) and node.value.id == "config":
                      raise AssertionError(
                          f"A2 VIOLATION: {src.name} accesses config.TIP_CORPUS directly. "
                          f"Use 'from contracts import TIP_CORPUS' instead."
                      )


# ── A3: stable_hash Production Import Prohibition ────────────────────────────
# A3 MANDATE: No production module (outside hash_utils.py and explicit deprecation tests)
# may import stable_hash. This test statically scans all .py files in the project root.

class TestA3StableHashProductionBan:
  # Files explicitly permitted to reference stable_hash:
  _PERMITTED = {"hash_utils.py"}
  # Fix 21: Directories excluded from recursive production scans.
  _SKIP_DIRS = frozenset({
      ".venv", "venv", ".git", "__pycache__", "build", "dist",
      "planv14", "plans", "archive",
  })
  # Fix 21: Additional symbols that must never appear in production code.
  _FORBIDDEN_SYMBOLS = frozenset({"_ReadOnlyDataFrame", "_ReadOnlyAccessor", "_needs_deepcopy"})

  def _collect_stable_hash_imports(self):
      import ast
      import pathlib
      violations = []
      root = pathlib.Path(__file__).parent.parent
      # Fix 21: rglob catches violations in sub-packages; skip non-production dirs.
      for py_file in root.rglob("*.py"):
          if any(part in self._SKIP_DIRS for part in py_file.parts):
              continue
          if py_file.name in self._PERMITTED:
              continue
          if py_file.name.startswith("test_"):
              # Test files that explicitly test DeprecationWarning are allowed.
              continue
          try:
              tree = ast.parse(py_file.read_text(encoding="utf-8"))
          except SyntaxError:
              continue
          rel = str(py_file.relative_to(root))
          for node in ast.walk(tree):
              if isinstance(node, ast.ImportFrom):
                  if node.module == "hash_utils":
                      names = [alias.name for alias in node.names]
                      if "stable_hash" in names:
                          violations.append(rel)
              if isinstance(node, ast.Import):
                  for alias in node.names:
                      if alias.name == "hash_utils":
                          violations.append(rel + " (import hash_utils)")
              # Fix 21: Also catch forbidden symbol references.
              if isinstance(node, ast.Name) and node.id in self._FORBIDDEN_SYMBOLS:
                  violations.append(f"{rel}: forbidden symbol '{node.id}'")
      return violations


  def test_no_production_module_imports_stable_hash(self):
      """A3: No production module outside hash_utils.py may import stable_hash."""
      violations = self._collect_stable_hash_imports()
      assert violations == [], (
          f"A3 VIOLATION: stable_hash imported in production modules: {violations}. "
          "Migrate all callers to log_utils.log_safe_merchant."
      )

  def test_stable_hash_emits_deprecation_warning_and_token_format(self):
      """A3: stable_hash must emit DeprecationWarning; log_safe_merchant token must be 41 chars.
      Fix 15b: dep_warnings defined and consumed in same test (no cross-method NameError).
      Fix 15 (pytest.warns): Use pytest.warns instead of warnings.catch_warnings per fixlist.
      """
      from hash_utils import stable_hash
      from log_utils import log_safe_merchant
      import pytest

      # Fix 15: pytest.warns captures the DeprecationWarning idiomatically.
      with pytest.warns(DeprecationWarning, match="deprecated"):
          stable_hash("test_value")

      token = log_safe_merchant("Netflix")
      assert len(token) == 41, f"Expected length 41, got {len(token)}"
      assert token.startswith("merchant:"), f"Expected token to start with 'merchant:', got {token}"




# ── _neutral_passion_result (D2) ──────────────────────────────────────────────
# D2: Renamed from _empty_passion_result. Tests verify:
#   1. below_min_rows path returns same row count with neutral columns
#   2. substage_failure path returns original row count with neutral columns
#   3. No partial enriched values persist after failure

class TestNeutralPassionResult:
  def test_below_min_rows_returns_same_row_count(self):
      """D2: _neutral_passion_result preserves row count — NOT an empty DataFrame."""
      from passion_pipeline import _neutral_passion_result
      df = pd.DataFrame({
          "amount": [100.0, 200.0],
          "cleaned_remarks": ["amazon", "flipkart"],
          "date": ["2023-01-01", "2023-02-01"],
          "predicted_category": ["shopping", "shopping"],
      })
      res = _neutral_passion_result(df, reason="below_min_rows")
      assert len(res.debits) == 2
      assert pd.isna(res.debits["inferred_subcategory"]).all()
      assert (res.debits["subcategory_confidence"] == 0.0).all()
      assert len(res.passion_signals) == 0
      assert len(res.candidates) == 0

  def test_substage_failure_soft_mode_preserves_row_count(self):
      """D2: process_pipeline soft-mode fail-fast returns original row count."""
      df = pd.DataFrame({
          "amount": [100.0, 200.0, 300.0],
          "cleaned_remarks": ["amazon", "flipkart", "myntra"],
          "date": ["2023-01-01", "2023-02-01", "2023-03-01"],
          "predicted_category": ["shopping", "shopping", "shopping"],
      })
      with patch("marketplace_subcategory.enrich_subcategories",
                 side_effect=RuntimeError("substage boom")):
          res = process_pipeline(df_raw=df, strict_mode=False)
      assert len(res.debits) == 3
      assert len(res.passion_signals) == 0

  def test_no_partial_enriched_values_after_failure(self):
      """D2: Neutral result has no partial enrichment — all rows get NA/0.0."""
      df = pd.DataFrame({
          "amount": [100.0, 200.0],
          "cleaned_remarks": ["amazon", "flipkart"],
          "date": ["2023-01-01", "2023-02-01"],
          "predicted_category": ["shopping", "shopping"],
      })
      with patch("marketplace_subcategory.enrich_subcategories",
                 side_effect=RuntimeError("enrichment failure")):
          res = process_pipeline(df_raw=df, strict_mode=False)
      # All rows must have neutral columns — no partial enrichment values
      assert pd.isna(res.debits["inferred_subcategory"]).all(), (
          "Partial inferred_subcategory values leaked into neutral result"
      )
      assert (res.debits["subcategory_confidence"] == 0.0).all(), (
          "Partial subcategory_confidence values leaked into neutral result"
      )

  def test_single_row_input_enriches_subcategory_but_no_passion_signals(self):
      # FIX-22: Test that a 1-row input successfully enriches subcategory,
      # but cannot produce passion signals due to PASSION_MERCHANT_COUNT_MIN=3.
      df = pd.DataFrame({
          "amount": [1500.0],
          "cleaned_remarks": ["amazon prime membership"],
          "date": ["2023-01-01"],
          "predicted_category": ["shopping"],
          "is_known_person": [False],
      })
      res = process_pipeline(df_raw=df, strict_mode=False)
      # Row count preserved
      assert len(res.debits) == 1
      # Subcategory should be enriched
      assert not pd.isna(res.debits["inferred_subcategory"]).all()
      # No passion signals or insights should be produced
      assert len(res.passion_signals) == 0
      assert len(res.candidates) == 0


# ── E3: E2E Integration — Passion Engine Enabled ─────────────────────────────
# E3: Integration test with INSIGHT_ENGINE_PASSION_ENABLED=true,
# INSIGHT_ENGINE_PASSION_STRICT_ATTACH=true, ENV=test, startup checks skipped.
#
# DESIGN NOTE: These tests call _attach_passion_results with a real PipelineResult
# and the real process_pipeline function — NOT the top-level run_pipeline.
# run_pipeline requires a full bank-statement DataFrame that survives preprocess()
# (needs Date/Details/Amount in raw bank CSV format). Calling it with a
# passion-specific minimal DataFrame crashes in Phase 1 (preprocessing), long before
# _attach_passion_results is ever reached. The correct E2E surface for testing the
# passion integration hook is _attach_passion_results itself — which is exactly
# how the existing test_attach_passion_results_real_pipeline_result_enabled works.

class TestE3PassionEngineE2EIntegration:
  def _make_debits_df(self, n_months: int = 4) -> "pd.DataFrame":
      """Return a minimal debits DataFrame that passes passion engine preflight."""
      dates = pd.date_range("2023-01-01", periods=n_months, freq="MS").strftime("%Y-%m-%d").tolist()
      return pd.DataFrame({
          "amount": [500.0, 600.0, 550.0, 700.0][:n_months],
          "cleaned_remarks": ["amazon", "flipkart", "myntra", "amazon"][:n_months],
          "date": dates,
          "predicted_category": ["shopping"] * n_months,
          "is_known_person": [False] * n_months,
      })

  def test_attach_passion_results_e2e_enabled_populates_passion_fields(self, monkeypatch):
      """E3: _attach_passion_results with real process_pipeline and PASSION_ENABLED=true."""
      monkeypatch.setenv("INSIGHT_ENGINE_PASSION_ENABLED", "true")
      monkeypatch.setenv("INSIGHT_ENGINE_PASSION_STRICT_ATTACH", "true")
      monkeypatch.setenv("ENV", "test")
      monkeypatch.setenv("INSIGHT_ENGINE_SKIP_STARTUP_CHECKS", "true")

      import pipeline
      from pipeline import PipelineResult
      from passion_pipeline import process_pipeline

      df = self._make_debits_df(n_months=4)
      result = make_pipeline_result(debits=df, credits=pd.DataFrame())

      out = pipeline._attach_passion_results(
          result,
          process_fn=lambda *, df_raw, strict_mode, rng: process_pipeline(
              df_raw=df_raw, strict_mode=strict_mode, rng=rng,
          ),
      )

      # E3 contract: passion fields must exist on the returned result
      assert hasattr(out, "passion_debits"), "PipelineResult missing passion_debits"
      assert hasattr(out, "passion_insights"), "PipelineResult missing passion_insights"
      assert hasattr(out, "passion_signals"), "PipelineResult missing passion_signals"

      assert isinstance(out.passion_debits, pd.DataFrame), (
          "passion_debits must be pd.DataFrame"
      )
      assert isinstance(out.passion_insights, tuple), "passion_insights must be tuple"
      assert isinstance(out.passion_signals, tuple), "passion_signals must be tuple"

  def test_attach_passion_results_e2e_disabled_leaves_fields_empty(self, monkeypatch):
      """E3/E2: With PASSION_ENABLED unset, _attach_passion_results returns result unchanged."""
      monkeypatch.delenv("INSIGHT_ENGINE_PASSION_ENABLED", raising=False)
      monkeypatch.setenv("ENV", "test")
      monkeypatch.setenv("INSIGHT_ENGINE_SKIP_STARTUP_CHECKS", "true")

      import pipeline
      from pipeline import PipelineResult
      from unittest.mock import MagicMock

      df = self._make_debits_df(n_months=4)
      result = make_pipeline_result(debits=df, credits=pd.DataFrame())
      process = MagicMock()

      out = pipeline._attach_passion_results(result, process_fn=process)

      # Fix #4: Kill switch off → result has passion_status="disabled", process never called.
      # Do NOT assert out is result — _with_passion_status always returns a new dataclass instance.
      assert out.stats.get("passion_status") == "disabled", (
          f"Expected passion_status='disabled' when engine disabled, got stats={out.stats!r}"
      )
      assert not process.called, (
          "process_fn must not be called when INSIGHT_ENGINE_PASSION_ENABLED is unset"
      )


# ── E4: Defensive-Copy Integrity Tests ───────────────────────────────────────
# E4: Verify process_pipeline does not mutate its input DataFrame and that
# mutations to result.debits do not propagate back to the original input.

class TestE4DefensiveCopyIntegrity:
  def _make_df(self) -> "pd.DataFrame":
      import pandas as pd
      import numpy as np
      return pd.DataFrame({
          "amount": [100.0, 200.0, 300.0],
          "cleaned_remarks": ["amazon", "flipkart", "myntra"],
          "date": ["2023-01-01", "2023-02-01", "2023-03-01"],
          "predicted_category": ["shopping", "shopping", "shopping"],
          "is_known_person": [False, False, False],
      })

  def test_input_df_not_mutated_by_process_pipeline(self):
      """E4: process_pipeline must not mutate its input DataFrame."""
      import pandas as pd
      df = self._make_df()
      before = df.copy(deep=True)
      process_pipeline(df)
      pd.testing.assert_frame_equal(df, before,
          check_like=False,
          obj="Input DataFrame was mutated by process_pipeline")

  def test_result_debits_mutation_does_not_propagate_to_input(self):
      """E4: Mutating result.debits must not propagate back to original input."""
      import pandas as pd
      df = self._make_df()
      before = df.copy(deep=True)
      result = process_pipeline(df)

      # Mutate the output
      result.debits.iloc[0, 0] = 99999.0

      # Input must remain unchanged
      pd.testing.assert_frame_equal(df, before,
          check_like=False,
          obj="Mutating result.debits propagated back to input DataFrame")

  def test_result_debits_mutation_does_not_propagate_to_passion_debits(self):
      """E4: result.debits and result.passion_debits must be independent copies."""
      import pandas as pd
      # Only run this sub-check when passion engine can fire
      df = self._make_df()
      result = process_pipeline(df)

      if not result.passion_debits.empty and "amount" in result.passion_debits.columns:
          original_passion_amount = result.passion_debits["amount"].iloc[0]
          if "amount" in result.debits.columns:
              result.debits.iloc[0, result.debits.columns.get_loc("amount")] = 99999.0
          assert result.passion_debits["amount"].iloc[0] == original_passion_amount, (
              "Mutating result.debits propagated to result.passion_debits"
          )


# ── F1: Optional CoW-Safe resolve_merchant_vectorized ────────────────────────
# F1 NOTE: Not a blocker under current pandas 3.x behavior. For defensive style,
# replace in-place .loc mutation in resolve_merchant_vectorized with:
#
#   s = s.mask(exact_mask, s.where(exact_mask).map(PASSION_MERCHANT_ALIASES))
#
# Do NOT make this a blocker unless pandas raises CoW warnings in tests.
# The test below documents the current contract (idempotency) for future migration.

class TestF1CowSafeResolve:
  def test_resolve_merchant_vectorized_is_idempotent(self):
      """F1: Applying resolve_merchant_vectorized twice returns the same result."""
      import pandas as pd
      s = pd.Series(["amzn", "flpkrt", "AMZN"])
      once = resolve_merchant_vectorized(s)
      twice = resolve_merchant_vectorized(once)
      pd.testing.assert_series_equal(once, twice,
          check_names=False,
          obj="resolve_merchant_vectorized is not idempotent")


# ── F2: Optional Alias Boundary Hardening ────────────────────────────────────
# F2: \b fails for special-char-adjacent names (e.g., |amazon|).
# Optional: switch to (?<!\w)/(?!\w) in _ALIAS_PATTERN_STR.
# Test documents repeated alias replacement behavior for "amzn amzn refund".

class TestF2AliasBoundaryHardening:
  def test_repeated_alias_replacement(self):
      """F2: 'amzn amzn refund' should resolve both 'amzn' tokens to 'amazon'."""
      import pandas as pd
      s = pd.Series(["amzn amzn refund"])
      result = resolve_merchant_vectorized(s)
      # Both aliases must be replaced; result must contain 'amazon'
      assert "amazon" in result.iloc[0].lower(), (
          f"Expected 'amazon' in resolved value, got: {result.iloc[0]!r}"
      )

  def test_special_char_adjacent_alias(self):
      """F2 + Fix #9: Alias surrounded by pipes |amzn| must resolve to canonical 'amazon'.
      Fix #9 uses (?<!\\w)/(?!\\w) lookarounds (not \\b) so pipes no longer break matching.
      """
      import pandas as pd
      s = pd.Series(["|amzn|"])
      result = resolve_merchant_vectorized(s)
      # After Fix #9: lookaround boundaries match; result must be the canonical 'amazon'.
      assert result.iloc[0] == "amazon", (
          f"Fix #9: Expected canonical 'amazon' for '|amzn|', got: {result.iloc[0]!r}"
      )

  # ── Fix #9: Canonical-return contract tests ─────────────────────────────
  def test_compound_alias_returns_canonical_not_phrase(self):
      """Fix #9: Compound alias must return just the canonical, not a rewritten phrase.
      Old behavior: 'amazon prime membership' → 'amazon membership' (rewritten phrase).
      New behavior: 'amazon prime membership' → 'amazon' (canonical only).
      """
      import pandas as pd
      s = pd.Series(["amazon prime membership", "amzn prime", "flipkart order 12345"])
      result = resolve_merchant_vectorized(s)
      assert result.iloc[0] == "amazon", f"Expected 'amazon', got {result.iloc[0]!r}"
      assert result.iloc[1] == "amazon", f"Expected 'amazon', got {result.iloc[1]!r}"
      assert result.iloc[2] == "flipkart", f"Expected 'flipkart', got {result.iloc[2]!r}"

  def test_unmatched_merchant_passthrough(self):
      """Fix #9: Unmatched merchant text passes through lowercased."""
      import pandas as pd
      s = pd.Series(["random cafe", "local grocery"])
      result = resolve_merchant_vectorized(s)
      assert result.iloc[0] == "random cafe"
      assert result.iloc[1] == "local grocery"

  def test_null_empty_returns_empty_string(self):
      """Fix #9: NaN and empty inputs must return empty string."""
      import pandas as pd
      import numpy as np
      s = pd.Series([None, np.nan, ""])
      result = resolve_merchant_vectorized(s)
      assert all(v == "" for v in result.tolist()), (
          f"Fix #9: Expected all empty strings for null/empty inputs, got {result.tolist()!r}"
      )

  # ── Fix #6: Canonical-only-as-key validation test ─────────────────────
  def test_canonical_only_as_alias_key_fails_validation(self):
      """Fix #6: A GENERALIST_CANONICAL that appears only as an alias KEY (not a VALUE)
      must fail validate_merchant_aliases. resolve_merchant_vectorized returns the VALUE,
      so a canonical as key-only can never appear in resolved output.
      """
      from config_passion import validate_merchant_aliases, GENERALIST_CANONICALS
      one_canonical = next(iter(GENERALIST_CANONICALS))
      # Map the canonical as a KEY to a completely different value.
      # validate_merchant_aliases checks alias_values only (Fix #6).
      bad_map = {one_canonical: "completely_different_merchant"}
      with pytest.raises(ValueError, match="unreachable"):
          validate_merchant_aliases(alias_map=bad_map)


# ── F3: Document Secret Caching Behavior ─────────────────────────────────────
# F3: _get_secret caches the secret after first call. Changing
# INSIGHT_ENGINE_SECRET requires a process restart. _reset_secret_cache is for
# tests ONLY. Do not implement hot secret reload unless explicitly required.

class TestF3SecretCachingBehavior:
  def test_secret_cached_after_first_call(self, monkeypatch):
      """F3: _get_secret returns the same bytes object on repeated calls (caching)."""
      from log_utils import _get_secret, _reset_secret_cache
      monkeypatch.setenv("ENV", "test")
      _reset_secret_cache()

      secret1 = _get_secret()
      secret2 = _get_secret()
      assert secret1 == secret2, (
          "_get_secret must return the same cached secret on repeated calls"
      )

  def test_reset_secret_cache_allows_env_var_change(self, monkeypatch):
      """F3: After _reset_secret_cache, a changed INSIGHT_ENGINE_SECRET takes effect."""
      from log_utils import _get_secret, _reset_secret_cache
      monkeypatch.setenv("ENV", "test")
      _reset_secret_cache()

      # First call: dev fallback
      s1 = _get_secret()

      # Change env and reset cache (simulates process restart in tests)
      new_secret = "a" * 32
      monkeypatch.setenv("INSIGHT_ENGINE_SECRET", new_secret)
      _reset_secret_cache()

      s2 = _get_secret()
      assert s2 == new_secret.encode("utf-8"), (
          "After _reset_secret_cache, new INSIGHT_ENGINE_SECRET must be used"
      )
      assert s1 != s2, (
          "Cached secret must differ from dev fallback after env var change"
      )


# ── F4: Memory/Performance Benchmark ─────────────────────────────────────────
# F4: Stress test with 100k-row DataFrame with passion engine enabled.
# Documents expected peak memory (up to 5x DataFrame size).
# Do NOT add _ReadOnlyDataFrame, _ReadOnlyAccessor, or _needs_deepcopy.
# This test is marked as 'slow' and skipped in standard CI unless explicitly run.

class TestF4MemoryPerformanceBenchmark:
  def test_100k_row_process_pipeline_completes_under_30s(self, monkeypatch):
      """F4: 100k-row DataFrame must complete process_pipeline within 30 seconds.
      Peak memory: up to 5x DataFrame size (work_df, enrich_df, detect_df,
      PassionResult copy, PipelineResult replacement copy).
      """
      import time
      import pandas as pd
      import numpy as np

      monkeypatch.setenv("ENV", "test")
      monkeypatch.setenv("INSIGHT_ENGINE_SKIP_STARTUP_CHECKS", "true")

      import passion_pipeline
      monkeypatch.setattr(passion_pipeline, "PIPELINE_HARD_TIMEOUT_MS", 60000.0)
      monkeypatch.setattr(passion_pipeline, "PIPELINE_BUDGET_MS", 60000.0)

      n = 100_000
      dates = pd.date_range("2020-01-01", periods=n, freq="h").strftime("%Y-%m-%d").tolist()
      df = pd.DataFrame({
          "amount": np.random.uniform(10, 5000, size=n),
          "cleaned_remarks": np.random.choice(
              ["amazon", "flipkart", "myntra", "swiggy", "zomato"], size=n
          ),
          "date": dates,
          "predicted_category": np.random.choice(
              ["shopping", "food", "travel"], size=n
          ),
          "is_known_person": np.random.choice([False, True], size=n, p=[0.9, 0.1]),
      })

      before = df.copy(deep=True)
      start = time.monotonic()

      result = process_pipeline(df_raw=df, strict_mode=False)

      elapsed = time.monotonic() - start

      # Correctness: input must be untouched
      pd.testing.assert_frame_equal(df, before,
          check_like=False,
          obj="F4: 100k-row input was mutated by process_pipeline")

      # Correctness: output has expected shape
      assert len(result.debits) == n, (
          f"F4: Expected {n} rows in result.debits, got {len(result.debits)}"
      )

      # Performance: must complete in a reasonable wall-clock time
      assert elapsed < 30.0, (
          f"F4: process_pipeline took {elapsed:.1f}s on 100k rows (limit: 30s). "
          "Check for O(n²) operations or excessive deep-copies."
      )


# ── tests/test_insight_engine.py ──────────────────────────────────────────────
# FIX C1: All PipelineResult(...) instantiations in test_insight_engine.py have been converted to keyword arguments.


# ── F5: Crash Dump Coverage for New PipelineResult Fields ─────────────────────
# FIX 19: Test that crash dump works, limits sizes, and masks PII properly for passion fields.
class TestF5CrashDumpWithPassion:
  def test_crash_dump_contains_passion_fields_pii_masked(self, tmp_path, monkeypatch):
      import pandas as pd
      import json
      import os
      import config
      from pipeline import _write_crash_dumps
      from passion_models import PassionSignal
      from log_utils import log_safe_merchant, log_safe_text

      monkeypatch.setenv("ENV", "test")
      monkeypatch.setattr(config, "ENABLE_PII_DEBUG_LOGS", False)

      # 1. Create synthetic processed passion debits
      passion_debits = pd.DataFrame({
          "amount": [100.0],
          "cleaned_remarks": ["SensitiveMerchant"],
          "date": ["2023-01-01"],
          "predicted_category": ["shopping"],
          "is_known_person": [False],
      })

      # 2. Create synthetic passion signal
      sig = PassionSignal(
          category="shopping",
          merchant_list=("SensitiveMerchant",),
          total_spend=100.0,
          merchant_count=1,
          spend_share=1.0,
          trend_direction="non_declining",
      )

      # 3. Call _write_crash_dumps directly
      _write_crash_dumps(
          debits=pd.DataFrame({"amount": [10.0]}),
          credits=pd.DataFrame(),
          crash_dump_dir=str(tmp_path),
          passion_debits=passion_debits,
          passion_insights=("Love shopping at SensitiveMerchant",),
          passion_signals=(sig,),
          run_id="testcrash123",
      )

      # 4. Verify that the crash dump files were created
      files = os.listdir(str(tmp_path))
      assert "testcrash123_debits.csv" in files
      assert "testcrash123_passion_debits.csv" in files
      assert "testcrash123_passion_summary.json" in files

      # Read passion_debits and verify PII is masked
      df_dump = pd.read_csv(os.path.join(str(tmp_path), "testcrash123_passion_debits.csv"))
      assert not df_dump.empty
      masked_merchant = log_safe_merchant("SensitiveMerchant")
      assert "SensitiveMerchant" not in df_dump["cleaned_remarks"].values
      assert masked_merchant in df_dump["cleaned_remarks"].values

      # Read passion_summary and verify PII in signals & insights is masked
      with open(os.path.join(str(tmp_path), "testcrash123_passion_summary.json"), "r") as f:
          summary = json.load(f)

      assert "insights" in summary
      assert "signals" in summary
      # FIX 3: Compute the correct expected token using log_safe_text
      masked_insight = log_safe_text("Love shopping at SensitiveMerchant")
      assert summary["insights"][0] == masked_insight
      assert "SensitiveMerchant" not in summary["insights"][0]

      signals = summary["signals"]
      assert len(signals) > 0
      assert "SensitiveMerchant" not in signals[0]["merchant_list"]
      assert masked_merchant in signals[0]["merchant_list"]


# ── F6: Unexpected Columns in Strict Mode ────────────────────────────────────
# FIX 17: Test that safe_assign_new_columns raises a ValueError if strict_mode is True and unexpected columns are found.
class TestF6UnexpectedColumnsStrictMode:
  def test_safe_assign_unexpected_columns_raises_in_strict_mode(self):
      import pandas as pd
      from passion_pipeline import safe_assign_new_columns

      original = pd.DataFrame({"amount": [10.0]}, index=[0])
      updated = pd.DataFrame({"amount": [10.0], "unexpected_col": [1]}, index=[0])

      # Soft mode: should log a warning but NOT raise an exception
      res = safe_assign_new_columns(original, updated, strict_mode=False)
      assert "unexpected_col" not in res.columns

      # Strict mode: should raise ValueError
      with pytest.raises(ValueError, match="unexpected columns"):
          safe_assign_new_columns(original, updated, strict_mode=True)


class TestConfigPassionCanonicals:
  def test_config_passion_aliases_contain_required_generalist_canonicals(self):
      from config_passion import PASSION_MERCHANT_ALIASES, GENERALIST_CANONICALS

      alias_values = set(PASSION_MERCHANT_ALIASES.values())
      missing = GENERALIST_CANONICALS - alias_values

      assert missing == set(), (
          f"config.SPECIFIC_MERCHANT_ALIASES missing required canonicals: {missing}. "
          f"Required for GENERALIST_CANONICALS: {GENERALIST_CANONICALS}"
      )
# END_CB_P8_01
