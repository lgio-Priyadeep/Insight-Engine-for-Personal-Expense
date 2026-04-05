"""
tests/test_phase1.py — Phase 1 Test Suite
==========================================
Covers:
  - preprocessor   : schema, flag normalization, signed amount, remarks cleaning,
                     zero-drop, dedup, debit/credit split
  - feature_engineer: time features, rolling leakage, z-score safety
  - seed_labeler   : matching, priority, multi-word keywords, coverage

Run with:
    pytest tests/test_phase1.py -v
"""

import sys
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

# Allow imports from parent directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from preprocessor import (
    validate_schema,
    _normalize_flag,
    _compute_signed_amount,
    clean_remark,
    _drop_zero_amount,
    _deduplicate,
    preprocess,
)
from feature_engineer import (
    add_time_features,
    add_rolling_features,
    fill_rolling_nulls,
    add_amount_features,
    engineer_features,
)
from seed_labeler import (
    _match_remark,
    label_debits,
    label_credits,
)
from config import (
    CATEGORY_PRIORITY,
    CATEGORY_KEYWORDS,
    CREDIT_PRIORITY,
    CREDIT_KEYWORDS,
    FALLBACK_DEBIT_LABEL,
    FALLBACK_CREDIT_LABEL,
)


# ─────────────────────────────── Fixtures ────────────────────────────────────

def _make_df(n: int = 10, flag: str = "DR") -> pd.DataFrame:
    """Minimal valid DataFrame for testing with messy realism."""
    base = datetime(2024, 1, 1)
    flags = [flag.lower() if i % 2 == 0 else f" {flag} " for i in range(n)]
    return pd.DataFrame({
        "date":         [base + timedelta(days=i) for i in range(n)],
        "amount":       [float(100 + i * 10) for i in range(n)],
        "amount_flag":  flags,
        "remarks":      [f"UPI/9876543210/swiggy order {i}" for i in range(n)],
        "balance":      [float(5000 - i * 50) for i in range(n)],
        "payment_mode_l": ["UPI"]   * n,
        "bank_code":      ["HDFC"]  * n,
        "counterparty":   ["Test"]  * n,
        "day_of_week":  [(base + timedelta(days=i)).weekday() for i in range(n)],
        "month":        [(base + timedelta(days=i)).month for i in range(n)],
        "amount_log":   [np.log1p(100 + i * 10) for i in range(n)],
        "pattern_used": ["UPI"] * n,
    })


def _make_debit_df_with_remarks(remarks: list[str]) -> pd.DataFrame:
    """Convenience fixture with custom cleaned_remarks."""
    df = _make_df(n=len(remarks))
    df["cleaned_remarks"] = remarks
    df["signed_amount"] = -df["amount"]
    return df


# ═══════════════════════════════ preprocessor ════════════════════════════════

class TestValidateSchema:

    def test_valid_df_passes(self):
        validate_schema(_make_df())  # must not raise

    def test_single_missing_column_raises(self):
        df = _make_df().drop(columns=["date"])
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_schema(df)

    def test_multiple_missing_columns_raises(self):
        df = _make_df().drop(columns=["date", "amount"])
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_schema(df)

    def test_extra_columns_are_allowed(self):
        df = _make_df()
        df["extra_col"] = 1
        validate_schema(df)  # must not raise


class TestNormalizeFlag:

    @pytest.mark.parametrize("raw, expected", [
        ("DR", "DR"), ("CR", "CR"),
        ("dr", "DR"), ("cr", "CR"),
        ("Dr", "DR"), ("Cr", "CR"),
        (" DR ",  "DR"), ("  cr  ", "CR"),
        ("\tDR\n", "DR"),
    ])
    def test_valid_flags(self, raw, expected):
        assert _normalize_flag(raw) == expected

    @pytest.mark.parametrize("bad", [
        "XX", "DEBIT", "CREDIT", "", "   ", "D R", "D", "C"
    ])
    def test_invalid_strings_return_none(self, bad):
        assert _normalize_flag(bad) is None

    @pytest.mark.parametrize("non_str", [1, 1.0, None, True, [], {}])
    def test_non_string_returns_none(self, non_str):
        assert _normalize_flag(non_str) is None


class TestComputeSignedAmount:

    def test_all_dr_are_negative(self):
        df = _compute_signed_amount(_make_df(n=5, flag="DR"))
        assert (df["signed_amount"] < 0).all()

    def test_all_cr_are_positive(self):
        df = _compute_signed_amount(_make_df(n=5, flag="CR"))
        assert (df["signed_amount"] > 0).all()

    def test_absolute_value_preserved(self):
        df = _make_df(n=3, flag="DR")
        df["amount"] = [100.0, 250.0, 50.0]
        result = _compute_signed_amount(df)
        assert result["signed_amount"].tolist() == [-100.0, -250.0, -50.0]

    def test_mixed_case_flags_handled(self):
        df = _make_df(n=4)
        df["amount_flag"] = ["DR", "CR", "dr", " CR "]
        result = _compute_signed_amount(df)
        signs = (result["signed_amount"] > 0).tolist()
        assert signs == [False, True, False, True]

    def test_amount_flag_column_normalized_in_place(self):
        df = _make_df(n=2)
        df["amount_flag"] = ["dr", " CR "]
        result = _compute_signed_amount(df)
        assert result["amount_flag"].tolist() == ["DR", "CR"]

    def test_defaults_invalid_flags_to_dr(self):
        df = _make_df(n=3)
        df["amount_flag"] = ["DR", "XX", "CR"]
        result = _compute_signed_amount(df)
        assert len(result) == 3
        # The 'XX' flag (at index 1) should become 'DR'
        assert result.loc[1, "amount_flag"] == "DR"
        assert result.loc[1, "signed_amount"] < 0


class TestCleanRemark:

    def test_removes_4digit_pii_ref(self):
        assert "9876" not in clean_remark("UPI/9876543210/swiggy")
        assert "1234" not in clean_remark("shop 1234 bought item")

    def test_retains_shorter_numbers(self):
        result = clean_remark("shop 123 bought item")
        assert "123" in result  # less than 4 digits → kept

    def test_removes_email(self):
        result = clean_remark("NEFT trf user@email.com")
        assert "user@email.com" not in result

    def test_removes_special_characters(self):
        result = clean_remark("swiggy@order#456!")
        assert "@" not in result
        assert "#" not in result
        assert "!" not in result

    def test_strips_noise_tokens(self):
        result = clean_remark("dr payment towards swiggy txn")
        assert "dr" not in result
        assert "swiggy" in result

    def test_lowercases_output(self):
        result = clean_remark("ZOMATO ORDER FOOD")
        assert result == result.lower()

    def test_collapses_whitespace(self):
        result = clean_remark("swiggy    food   order")
        assert "  " not in result

    def test_empty_string_returns_empty(self):
        assert clean_remark("") == ""

    def test_none_returns_empty(self):
        assert clean_remark(None) == ""

    def test_whitespace_only_returns_empty(self):
        assert clean_remark("   ") == ""

    def test_only_noise_tokens_returns_empty(self):
        result = clean_remark("ac txn payment ref cr")
        assert result == ""

    def test_non_string_returns_empty(self):
        assert clean_remark(12345) == ""

    def test_mixed_real_and_noise(self):
        result = clean_remark("towards amazon payment")
        assert "amazon" in result
        assert "payment" not in result

    def test_unmapped_merchant_survives_generic_routing(self):
        # Even though "UPI" maps to a generic pattern, the remaining string MUST NOT be discarded.
        result = clean_remark("UPI/12093/RajuTeaStall")
        assert "rajuteastall" in result
        assert "upi" not in result


class TestDropZeroAmount:

    def test_drops_zero_rows(self):
        df = _make_df(n=5)
        df.loc[2, "amount"] = 0
        result = _drop_zero_amount(df)
        assert len(result) == 4
        assert 0 not in result["amount"].values

    def test_no_change_when_no_zeros(self):
        df = _make_df(n=5)
        result = _drop_zero_amount(df)
        assert len(result) == 5

    def test_all_zeros_removed(self):
        df = _make_df(n=3)
        df["amount"] = 0
        result = _drop_zero_amount(df)
        assert len(result) == 0


class TestDeduplicate:

    def test_removes_exact_duplicates(self):
        df = _make_df(n=3)
        df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
        result = _deduplicate(df)
        assert len(result) == 3

    def test_no_change_when_no_duplicates(self):
        df = _make_df(n=5)
        result = _deduplicate(df)
        assert len(result) == 5

    def test_different_date_same_amount_not_deduped(self):
        df = _make_df(n=2)
        df["amount"] = [100.0, 100.0]
        df["remarks"] = ["same", "same"]
        # dates differ → should NOT be deduped
        result = _deduplicate(df)
        assert len(result) == 2


class TestPreprocess:

    def test_returns_two_dataframes(self):
        debits, credits = preprocess(_make_df())
        assert isinstance(debits, pd.DataFrame)
        assert isinstance(credits, pd.DataFrame)

    def test_pure_debit_input(self):
        debits, credits = preprocess(_make_df(n=5, flag="DR"))
        assert len(debits) == 5
        assert len(credits) == 0

    def test_pure_credit_input(self):
        debits, credits = preprocess(_make_df(n=5, flag="CR"))
        assert len(debits) == 0
        assert len(credits) == 5

    def test_cleaned_remarks_column_present(self):
        debits, _ = preprocess(_make_df(n=3))
        assert "cleaned_remarks" in debits.columns

    def test_signed_amount_column_present(self):
        debits, _ = preprocess(_make_df(n=3, flag="DR"))
        assert "signed_amount" in debits.columns

    def test_output_is_sorted_chronologically(self):
        df = _make_df(n=5)
        df = df.iloc[::-1].reset_index(drop=True)  # reverse order
        debits, _ = preprocess(df)
        assert debits["date"].is_monotonic_increasing

    def test_missing_column_raises(self):
        df = _make_df().drop(columns=["amount"])
        with pytest.raises(ValueError):
            preprocess(df)

    def test_does_not_mutate_input(self):
        df = _make_df(n=5)
        original_cols = list(df.columns)
        preprocess(df)
        assert list(df.columns) == original_cols


# ═══════════════════════════════ feature_engineer ════════════════════════════

def _base_fe_df(n: int = 15) -> pd.DataFrame:
    """DataFrame prepped for feature engineering (has date + signed_amount)."""
    df = _make_df(n=n, flag="DR")
    df["date"] = pd.to_datetime(df["date"])
    df["signed_amount"] = -df["amount"]
    return df


class TestTimeFeatures:

    def test_is_weekend_correct_for_known_dates(self):
        df = _base_fe_df(n=7)
        result = add_time_features(df)
        for _, row in result.iterrows():
            expected = 1 if row["date"].dayofweek in (5, 6) else 0
            assert row["is_weekend"] == expected

    def test_month_sin_cos_in_range(self):
        df = _base_fe_df(n=20)
        result = add_time_features(df)
        assert result["month_sin"].between(-1.0, 1.0).all()
        assert result["month_cos"].between(-1.0, 1.0).all()

    def test_dow_sin_cos_in_range(self):
        df = _base_fe_df(n=10)
        result = add_time_features(df)
        assert result["dow_sin"].between(-1.0, 1.0).all()
        assert result["dow_cos"].between(-1.0, 1.0).all()

    def test_week_of_month_range(self):
        df = _base_fe_df(n=31)
        result = add_time_features(df)
        assert result["week_of_month"].between(1, 5).all()

    def test_non_datetime_raises(self):
        df = _base_fe_df(n=3)
        df["date"] = "2024-01-01"   # string, not datetime
        with pytest.raises(TypeError):
            add_time_features(df)


class TestRollingFeatures:

    def test_first_row_rolling_mean_excludes_itself(self):
        """
        After shift(1), row 0's window is empty → rolling_7d_mean should be NaN.
        This proves shift(1) is correctly applied before rolling.
        """
        df = _base_fe_df(n=10)
        df["amount"] = [float(i * 100) for i in range(1, 11)]
        result = add_rolling_features(df)
        assert pd.isna(result.iloc[0]["rolling_7d_mean"])

    def test_rolling_mean_does_not_include_current_row(self):
        """
        Row 1's rolling mean should equal row 0's value (only prior row).
        """
        df = _base_fe_df(n=5)
        df["amount"] = [10.0, 20.0, 30.0, 40.0, 50.0]
        result = add_rolling_features(df)
        # After shift(1), row 1 sees only [10.0] → mean = 10.0
        assert result.iloc[1]["rolling_7d_mean"] == pytest.approx(10.0)

    def test_rolling_std_nan_when_fewer_than_2_prior_rows(self):
        df = _base_fe_df(n=5)
        result = add_rolling_features(df)
        # Row 0 shifted: window is empty → std is NaN
        assert pd.isna(result.iloc[0]["rolling_7d_std"])

    def test_output_sorted_chronologically(self):
        df = _base_fe_df(n=10)
        df = df.iloc[::-1].reset_index(drop=True)   # reverse order
        result = add_rolling_features(df)
        assert result["date"].is_monotonic_increasing


class TestFillRollingNulls:

    def test_no_nans_after_fill(self):
        df = _base_fe_df(n=5)
        df = add_rolling_features(df)
        df = fill_rolling_nulls(df, global_mean=-150.0, global_std=50.0)
        assert not df["rolling_7d_mean"].isna().any()
        assert not df["rolling_30d_mean"].isna().any()
        assert not df["rolling_7d_std"].isna().any()

    def test_fill_uses_provided_mean(self):
        df = _base_fe_df(n=2)
        df = add_rolling_features(df)
        df = fill_rolling_nulls(df, global_mean=-999.0, global_std=1.0)
        assert df.iloc[0]["rolling_7d_mean"] == pytest.approx(-999.0)

    def test_zero_global_std_falls_back_to_one(self):
        df = _base_fe_df(n=2)
        df = add_rolling_features(df)
        df = fill_rolling_nulls(df, global_mean=0.0, global_std=0.0)
        # Should not be 0 (that would cause div-by-zero downstream)
        filled_stds = df["rolling_7d_std"][df["rolling_7d_std"].notna()]
        assert (filled_stds != 0).all()


class TestAmountFeatures:

    def _prepped(self, amounts, means, stds) -> pd.DataFrame:
        df = _base_fe_df(n=len(amounts))
        df["amount"]   = amounts
        df["rolling_7d_mean"] = means
        df["rolling_7d_std"]  = stds
        return df

    def test_zscore_clipped_upper(self):
        df = self._prepped([1_000_000.0], [0.0], [1.0])
        result = add_amount_features(df)
        assert result["amount_zscore"].iloc[0] == pytest.approx(5.0)

    def test_zscore_clipped_lower(self):
        df = self._prepped([-1_000_000.0], [0.0], [1.0])
        result = add_amount_features(df)
        assert result["amount_zscore"].iloc[0] == pytest.approx(-5.0)

    def test_zero_std_does_not_produce_inf(self):
        df = self._prepped([-100.0, -200.0], [0.0, 0.0], [0.0, 0.0])
        result = add_amount_features(df)
        assert not result["amount_zscore"].isin([np.inf, -np.inf]).any()

    def test_amount_log_is_non_negative(self):
        df = self._prepped([-200.0, -1.0, -0.01], [0.0]*3, [1.0]*3)
        result = add_amount_features(df)
        assert (result["amount_log"] >= 0).all()

    def test_amount_log_uses_absolute_value(self):
        """log1p(-x) would give NaN; we use log1p(abs(x)) instead."""
        df = self._prepped([-500.0], [0.0], [1.0])
        result = add_amount_features(df)
        assert not result["amount_log"].isna().any()
        assert result["amount_log"].iloc[0] == pytest.approx(np.log1p(500.0))


class TestEngineerFeaturesFull:

    def test_full_pipeline_no_nans(self):
        df = _base_fe_df(n=20)
        result = engineer_features(df, global_mean=-150.0, global_std=50.0)
        feature_cols = [
            "is_weekend", "week_of_month", "month_sin", "month_cos",
            "dow_sin", "dow_cos", "rolling_7d_mean", "rolling_30d_mean",
            "rolling_7d_std", "amount_log", "amount_zscore",
        ]
        for col in feature_cols:
            assert col in result.columns, f"Missing: {col}"
            assert not result[col].isna().any(), f"NaN found in: {col}"

    def test_does_not_mutate_input(self):
        df = _base_fe_df(n=5)
        cols_before = list(df.columns)
        engineer_features(df, global_mean=-100.0, global_std=30.0)
        assert list(df.columns) == cols_before


# ═══════════════════════════════ seed_labeler ════════════════════════════════

class TestMatchRemark:

    def test_food_keyword_match(self):
        result, _ = _match_remark("zomato order food", CATEGORY_KEYWORDS, CATEGORY_PRIORITY, FALLBACK_DEBIT_LABEL)
        assert result == "food"

    def test_transport_keyword_match(self):
        result, _ = _match_remark("uber ride cab", CATEGORY_KEYWORDS, CATEGORY_PRIORITY, FALLBACK_DEBIT_LABEL)
        assert result == "transport"

    def test_shopping_keyword_match(self):
        result, _ = _match_remark("amazon purchase delivered", CATEGORY_KEYWORDS, CATEGORY_PRIORITY, FALLBACK_DEBIT_LABEL)
        assert result == "shopping"

    def test_empty_remark_returns_fallback(self):
        result, _ = _match_remark("", CATEGORY_KEYWORDS, CATEGORY_PRIORITY, FALLBACK_DEBIT_LABEL)
        assert result == FALLBACK_DEBIT_LABEL

    def test_whitespace_only_returns_fallback(self):
        result, _ = _match_remark("   ", CATEGORY_KEYWORDS, CATEGORY_PRIORITY, FALLBACK_DEBIT_LABEL)
        assert result == FALLBACK_DEBIT_LABEL

    def test_no_match_returns_fallback(self):
        result, _ = _match_remark("xyzcompany abc unknown", CATEGORY_KEYWORDS, CATEGORY_PRIORITY, FALLBACK_DEBIT_LABEL)
        assert result == FALLBACK_DEBIT_LABEL

    def test_multi_word_keyword_match(self):
        # "cash withdrawal" is a 2-word keyword in the atm category
        result, _ = _match_remark("cash withdrawal nearby", CATEGORY_KEYWORDS, CATEGORY_PRIORITY, FALLBACK_DEBIT_LABEL)
        assert result == "atm"

    def test_multi_word_keyword_requires_all_tokens(self):
        # Only "cash" present, not "withdrawal" → should NOT match atm
        result, _ = _match_remark("cash payment done", CATEGORY_KEYWORDS, CATEGORY_PRIORITY, FALLBACK_DEBIT_LABEL)
        assert result != "atm"

    def test_priority_order_respected(self):
        """
        'emi hospital' matches both finance (emi) and health (hospital).
        finance comes before health in CATEGORY_PRIORITY → finance wins.
        """
        finance_idx = CATEGORY_PRIORITY.index("finance")
        health_idx  = CATEGORY_PRIORITY.index("health")
        result, matches = _match_remark("emi hospital bill", CATEGORY_KEYWORDS, CATEGORY_PRIORITY, FALLBACK_DEBIT_LABEL)

        expected = "finance" if finance_idx < health_idx else "health"
        assert result == expected
        assert "health" in matches and "finance" in matches

    def test_custom_priority_overrides_default(self):
        """Reversing priority should flip the winner for ambiguous remarks."""
        reversed_priority = list(reversed(CATEGORY_PRIORITY))
        default_result, _  = _match_remark("emi hospital", CATEGORY_KEYWORDS, CATEGORY_PRIORITY, FALLBACK_DEBIT_LABEL)
        reversed_result, _ = _match_remark("emi hospital", CATEGORY_KEYWORDS, reversed_priority, FALLBACK_DEBIT_LABEL)
        # They should differ (both match something, but different categories win)
        assert default_result != reversed_result


class TestLabelDebits:

    def test_known_remarks_correctly_labeled(self):
        df = _make_debit_df_with_remarks(["zomato", "uber", "amazon"])
        result = label_debits(df)
        assert result.iloc[0]["pseudo_label"] == "food"
        assert result.iloc[1]["pseudo_label"] == "transport"
        assert result.iloc[2]["pseudo_label"] == "shopping"

    def test_unknown_remark_gets_fallback(self):
        df = _make_debit_df_with_remarks(["completelyrandommerchant"])
        result = label_debits(df)
        assert result.iloc[0]["pseudo_label"] == FALLBACK_DEBIT_LABEL

    def test_label_col_name_respected(self):
        df = _make_debit_df_with_remarks(["zomato"])
        result = label_debits(df, label_col="category")
        assert "category" in result.columns

    def test_input_not_mutated(self):
        df = _make_debit_df_with_remarks(["zomato", "uber"])
        cols_before = list(df.columns)
        label_debits(df)
        assert list(df.columns) == cols_before

    def test_all_rows_labeled(self):
        remarks = ["zomato", "uber", "xyz", "amazon", "salary"]
        df = _make_debit_df_with_remarks(remarks)
        result = label_debits(df)
        assert result["pseudo_label"].notna().all()
        assert len(result) == len(remarks)

    def test_spencers_labeled_as_food(self):
        df = _make_debit_df_with_remarks(["spencers retail purchase"])
        result = label_debits(df)
        assert result.iloc[0]["pseudo_label"] == "food"

    def test_acct_token_labeled_as_transfer(self):
        df = _make_debit_df_with_remarks(["random acct string"])
        result = label_debits(df)
        assert result.iloc[0]["pseudo_label"] == "transfer"


class TestLabelCredits:

    def test_salary_match(self):
        df = _make_debit_df_with_remarks(["salary credited"])
        result = label_credits(df)
        assert result.iloc[0]["pseudo_label"] == "salary"

    def test_refund_match(self):
        df = _make_debit_df_with_remarks(["amazon refund processed"])
        result = label_credits(df)
        assert result.iloc[0]["pseudo_label"] == "refund"

    def test_interest_match(self):
        df = _make_debit_df_with_remarks(["fd interest quarterly"])
        result = label_credits(df)
        assert result.iloc[0]["pseudo_label"] == "interest"

    def test_unknown_credit_falls_back_to_other_credit(self):
        df = _make_debit_df_with_remarks(["completely unknown source credit"])
        result = label_credits(df)
        assert result.iloc[0]["pseudo_label"] == FALLBACK_CREDIT_LABEL

    def test_fallback_is_not_uncategorized(self):
        """Credits must NOT fall back to FALLBACK_DEBIT_LABEL."""
        df = _make_debit_df_with_remarks(["random xyz"])
        result = label_credits(df)
        assert result.iloc[0]["pseudo_label"] != FALLBACK_DEBIT_LABEL
        assert result.iloc[0]["pseudo_label"] == FALLBACK_CREDIT_LABEL

    def test_all_rows_receive_a_label(self):
        remarks = ["salary", "refund amazon", "fd interest", "mystery payment", ""]
        df = _make_debit_df_with_remarks(remarks)
        result = label_credits(df)
        assert result["pseudo_label"].notna().all()
