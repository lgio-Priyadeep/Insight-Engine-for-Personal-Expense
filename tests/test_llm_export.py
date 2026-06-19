"""
tests/test_llm_export.py — LLM Export Unit & Integration Tests
===============================================================
26 test functions covering:
  - build_llm_context() schema, determinism, type safety
  - export_to_json() file creation, atomicity, run_id injection
  - Per-section correctness (period, spend_profile, anomalies, recurring,
    personal_transfers, budget_health, passion_insights)
  - PII masking gate
  - numpy/NaN/Inf coercion
  - Backward compat: generate_human_insights() still returns List[str]
  - Edge cases: empty debits, missing columns, inference result (no personal_summary)
"""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from typing import List
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline import PipelineResult, run_pipeline
from llm_export import build_llm_context, export_to_json
from llm_serializer import serialize_to_json, _json_safe
from insight_generator import generate_human_insights, _generate_insight_records
from schema import Col
import config


# ──────────────────────────────────────────────────────────────────────────────
# Shared fixtures
# ──────────────────────────────────────────────────────────────────────────────

def _make_minimal_raw_df(n_rows: int = 90, seed: int = 0) -> pd.DataFrame:
    """Return a minimal but valid raw bank statement DataFrame for run_pipeline()."""
    rng = np.random.default_rng(seed)
    base = datetime(2024, 1, 1)
    dates, amounts, flags, remarks = [], [], [], []

    merchants_dr = ["zomato", "swiggy", "netflix", "amazon", "flipkart", "uber"]
    merchants_cr = ["salary credit", "upi transfer received"]

    for i in range(n_rows):
        d = base + timedelta(days=i)
        # debit
        dates.append(d)
        amounts.append(float(rng.uniform(50, 500)))
        flags.append("DR")
        remarks.append(rng.choice(merchants_dr))

        # occasional credit
        if i % 30 == 0:
            dates.append(d)
            amounts.append(float(rng.uniform(20000, 50000)))
            flags.append("CR")
            remarks.append(rng.choice(merchants_cr))

    return pd.DataFrame({
        "date": dates,
        "amount": amounts,
        "amount_flag": flags,
        "remarks": remarks,
    })


@pytest.fixture(scope="module")
def pipeline_result():
    """Run the full pipeline once per module — expensive but realistic."""
    raw = _make_minimal_raw_df(n_rows=90)
    return run_pipeline(raw)


@pytest.fixture(scope="module")
def llm_context(pipeline_result):
    """Build LLM context once per module."""
    return build_llm_context(pipeline_result)


# ──────────────────────────────────────────────────────────────────────────────
# 1. test_build_llm_context_returns_dict
# ──────────────────────────────────────────────────────────────────────────────

def test_build_llm_context_returns_dict(llm_context):
    """Return type is dict; run_id key is ABSENT (injected only by export_to_json)."""
    assert isinstance(llm_context, dict), "build_llm_context must return dict"
    assert "run_id" not in llm_context, (
        "run_id must NOT be in the dict returned by build_llm_context; "
        "it is injected by export_to_json"
    )


# ──────────────────────────────────────────────────────────────────────────────
# 2. test_schema_version_field
# ──────────────────────────────────────────────────────────────────────────────

def test_schema_version_field(llm_context):
    """schema_version must equal '1.0'."""
    assert llm_context.get("schema_version") == "1.0"


# ──────────────────────────────────────────────────────────────────────────────
# 3. test_period_section_completeness
# ──────────────────────────────────────────────────────────────────────────────

def test_period_section_completeness(llm_context):
    """All period keys present; total_spend >= 0."""
    period = llm_context.get("period", {})
    required_keys = {
        "start", "end", "total_transactions",
        "total_spend_inr", "total_inflow_inr", "net_cashflow_inr",
        "data_completeness_pct",
    }
    assert required_keys.issubset(set(period.keys())), (
        f"Missing period keys: {required_keys - set(period.keys())}"
    )
    assert period["total_spend_inr"] >= 0


# ──────────────────────────────────────────────────────────────────────────────
# 4. test_category_breakdown_capped
# ──────────────────────────────────────────────────────────────────────────────

def test_category_breakdown_capped(llm_context):
    """spend_profile never exceeds LLM_EXPORT_MAX_CATEGORIES."""
    profile = llm_context.get("spend_profile", [])
    assert isinstance(profile, list)
    assert len(profile) <= config.LLM_EXPORT_MAX_CATEGORIES


# ──────────────────────────────────────────────────────────────────────────────
# 5. test_anomaly_section_capped
# ──────────────────────────────────────────────────────────────────────────────

def test_anomaly_section_capped(llm_context):
    """top_anomalies never exceeds LLM_EXPORT_MAX_ANOMALIES."""
    anomaly = llm_context.get("anomaly_summary", {})
    top = anomaly.get("top_anomalies", [])
    assert isinstance(top, list)
    assert len(top) <= config.LLM_EXPORT_MAX_ANOMALIES


# ──────────────────────────────────────────────────────────────────────────────
# 6. test_recurring_section_capped
# ──────────────────────────────────────────────────────────────────────────────

def test_recurring_section_capped(llm_context):
    """recurring_subscriptions never exceeds LLM_EXPORT_MAX_SUBSCRIPTIONS."""
    recurring = llm_context.get("recurring_subscriptions", [])
    assert isinstance(recurring, list)
    assert len(recurring) <= config.LLM_EXPORT_MAX_SUBSCRIPTIONS


# ──────────────────────────────────────────────────────────────────────────────
# 7. test_passion_disabled_graceful
# ──────────────────────────────────────────────────────────────────────────────

def test_passion_disabled_graceful(pipeline_result, monkeypatch):
    """When passion feature is not enabled (default), passion_insights.enabled=False and no crash."""
    # INSIGHT_ENGINE_PASSION_ENABLED is an os.environ gate (CP-01 C1b fix).
    monkeypatch.setenv("INSIGHT_ENGINE_PASSION_ENABLED", "false")
    ctx = build_llm_context(pipeline_result)
    pi = ctx.get("passion_insights", {})
    assert pi.get("enabled") is False
    assert pi.get("signals") == []


# ──────────────────────────────────────────────────────────────────────────────
# 8. test_passion_suppressed_signals_excluded
# ──────────────────────────────────────────────────────────────────────────────

def test_passion_suppressed_signals_excluded(monkeypatch):
    """Suppressed signals are excluded from the export (gate activated via env-var)."""
    from llm_export_passion import build_passion_context

    suppressed = MagicMock()
    suppressed.is_suppressed = True
    suppressed.total_spend = 1000.0
    suppressed.category = "food"
    suppressed.subcategory = ""
    suppressed.merchant_count = 2
    suppressed.active_months = 3
    suppressed.trend_direction = "non_declining"
    suppressed.spend_share = 0.1
    suppressed.suppression_reason = "low_spend"

    mock_result = MagicMock()
    mock_result.passion_signals = (suppressed,)

    # Gate is now os.environ (CP-01 C1b). Use monkeypatch — not config patch.
    monkeypatch.setenv("INSIGHT_ENGINE_PASSION_ENABLED", "true")
    ctx = build_passion_context(mock_result, pii_safe=True)

    # enabled=True (gate open), but signal is suppressed → excluded
    assert ctx["enabled"] is True
    assert ctx["signal_count"] == 0
    assert ctx["signals"] == []


# ──────────────────────────────────────────────────────────────────────────────
# 9. test_pii_masking_applied
# ──────────────────────────────────────────────────────────────────────────────

def test_pii_masking_applied(pipeline_result):
    """Raw merchant names must not appear in anomaly/recurring sections when pii_safe=True."""
    with patch.object(config, "ENABLE_PII_DEBUG_LOGS", False):
        ctx = build_llm_context(pipeline_result)

    raw_merchants = set()
    if Col.CLEANED_REMARKS in pipeline_result.debits.columns:
        raw_merchants = set(
            pipeline_result.debits[Col.CLEANED_REMARKS].dropna().unique()
        )

    if not raw_merchants:
        pytest.skip("No cleaned_remarks in debits to check PII masking.")

    # Check anomaly merchant names
    for a in ctx.get("anomaly_summary", {}).get("top_anomalies", []):
        assert a.get("merchant") not in raw_merchants, (
            f"Raw merchant '{a['merchant']}' leaked into anomaly export"
        )


# ──────────────────────────────────────────────────────────────────────────────
# 10. test_numpy_types_not_in_output
# ──────────────────────────────────────────────────────────────────────────────

def test_numpy_types_not_in_output(llm_context):
    """json.dumps(ctx) must not raise TypeError — all numpy types coerced."""
    try:
        json.dumps(llm_context, default=_json_safe)
    except TypeError as e:
        pytest.fail(f"json.dumps raised TypeError — numpy type not coerced: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# 11. test_nan_inf_coerced_to_null
# ──────────────────────────────────────────────────────────────────────────────

def test_nan_inf_coerced_to_null():
    """NaN and Inf values are coerced to JSON null by _json_safe."""
    assert _json_safe(float("nan")) is None
    assert _json_safe(float("inf")) is None
    assert _json_safe(float("-inf")) is None
    assert _json_safe(np.nan) is None
    assert _json_safe(np.float64("nan")) is None
    assert _json_safe(pd.NA) is None
    assert _json_safe(pd.NaT) is None


# ──────────────────────────────────────────────────────────────────────────────
# 12. test_export_to_json_creates_file
# ──────────────────────────────────────────────────────────────────────────────

def test_export_to_json_creates_file(pipeline_result, tmp_path):
    """export_to_json writes a real file to disk."""
    out_path = str(tmp_path / "test_output.json")
    result_path = export_to_json(pipeline_result, output_path=out_path)
    assert os.path.isfile(result_path), f"File not created at {result_path}"
    assert os.path.getsize(result_path) > 0


# ──────────────────────────────────────────────────────────────────────────────
# 13. test_export_to_json_atomic
# ──────────────────────────────────────────────────────────────────────────────

def test_export_to_json_atomic(pipeline_result, tmp_path):
    """No .tmp file should remain after a successful export."""
    out_path = str(tmp_path / "atomic_test.json")
    export_to_json(pipeline_result, output_path=out_path)
    tmp_file = out_path + ".tmp"
    assert not os.path.exists(tmp_file), f".tmp file was not cleaned up: {tmp_file}"


# ──────────────────────────────────────────────────────────────────────────────
# 14. test_export_to_json_has_run_id
# ──────────────────────────────────────────────────────────────────────────────

def test_export_to_json_has_run_id(pipeline_result, tmp_path):
    """Written JSON file contains run_id; in-memory dict from build_llm_context does NOT."""
    out_path = str(tmp_path / "run_id_test.json")
    export_to_json(pipeline_result, output_path=out_path)

    with open(out_path, "r", encoding="utf-8") as f:
        on_disk = json.load(f)

    assert "run_id" in on_disk, "run_id must be present in the written JSON file"

    in_memory = build_llm_context(pipeline_result)
    assert "run_id" not in in_memory, "run_id must NOT be in the dict from build_llm_context"


# ──────────────────────────────────────────────────────────────────────────────
# 15. test_export_to_json_idempotent
# ──────────────────────────────────────────────────────────────────────────────

def test_export_to_json_idempotent(pipeline_result, tmp_path):
    """Two calls with same result + mocked uuid4 + frozen datetime produce byte-identical JSON."""
    fixed_hex = "aabbccdd1122"
    fixed_now = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

    mock_uuid = MagicMock()
    mock_uuid.hex = fixed_hex + "extra"  # [:12] slices to fixed_hex

    mock_dt = MagicMock()
    mock_dt.now.return_value = fixed_now
    mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)  # allow direct construction

    path1 = str(tmp_path / "idempotent_1.json")
    path2 = str(tmp_path / "idempotent_2.json")

    with patch("llm_export.datetime", mock_dt), \
         patch("llm_export.uuid4", return_value=mock_uuid):
        export_to_json(pipeline_result, output_path=path1)
        export_to_json(pipeline_result, output_path=path2)

    with open(path1, "rb") as f1, open(path2, "rb") as f2:
        content1 = f1.read()
        content2 = f2.read()

    assert content1 == content2, "Two export calls with fixed uuid4+datetime must produce byte-identical files"


# ──────────────────────────────────────────────────────────────────────────────
# 16. test_empty_debits_raises
# ──────────────────────────────────────────────────────────────────────────────

def test_empty_debits_raises():
    """ValueError raised when result.debits is empty."""
    empty_result = MagicMock()
    empty_result.debits = pd.DataFrame()
    empty_result.credits = pd.DataFrame()

    with pytest.raises(ValueError, match="non-empty DataFrame"):
        build_llm_context(empty_result)


# ──────────────────────────────────────────────────────────────────────────────
# 17. test_insight_records_structured_shape
# ──────────────────────────────────────────────────────────────────────────────

def test_insight_records_structured_shape(pipeline_result):
    """_generate_insight_records returns List[dict] with type/text/tip/score keys."""
    spend_debits = pipeline_result.debits
    if Col.IS_KNOWN_PERSON in spend_debits.columns:
        spend_debits = spend_debits[~spend_debits[Col.IS_KNOWN_PERSON].fillna(False)]

    if spend_debits.empty:
        pytest.skip("No spend debits available for this test.")

    records = _generate_insight_records(spend_debits)
    assert isinstance(records, list)
    for rec in records:
        assert isinstance(rec, dict), f"Expected dict, got {type(rec)}"
        assert "type" in rec, "Record missing 'type' key"
        assert "text" in rec, "Record missing 'text' key"
        assert "tip" in rec, "Record missing 'tip' key"
        assert "score" in rec, "Record missing 'score' key"
        assert isinstance(rec["score"], float)


# ──────────────────────────────────────────────────────────────────────────────
# 18. test_generate_human_insights_backward_compat
# ──────────────────────────────────────────────────────────────────────────────

def test_generate_human_insights_backward_compat(pipeline_result):
    """generate_human_insights still returns List[str] — no signature change."""
    spend_debits = pipeline_result.debits
    if Col.IS_KNOWN_PERSON in spend_debits.columns:
        spend_debits = spend_debits[~spend_debits[Col.IS_KNOWN_PERSON].fillna(False)]

    if spend_debits.empty:
        pytest.skip("No spend debits available for this test.")

    result = generate_human_insights(spend_debits)
    assert isinstance(result, list), "generate_human_insights must return list"
    for item in result:
        assert isinstance(item, str), f"Expected str element, got {type(item)}"


# ──────────────────────────────────────────────────────────────────────────────
# 19. test_budget_health_savings_rate
# ──────────────────────────────────────────────────────────────────────────────

def test_budget_health_savings_rate():
    """Savings rate formula: (inflow - spend) / inflow * 100."""
    from llm_export_aggregators import build_budget_health

    debits = pd.DataFrame({
        Col.AMOUNT: [1000.0, 2000.0],
        Col.PREDICTED_CATEGORY: ["food", "shopping"],
        Col.IS_KNOWN_PERSON: [False, False],
    })
    credits = pd.DataFrame({
        Col.AMOUNT: [10000.0],
    })

    health = build_budget_health(debits, credits)
    expected_rate = (10000.0 - 3000.0) / 10000.0 * 100  # 70.0
    assert health["savings_rate_pct"] == pytest.approx(expected_rate, rel=1e-3)


# ──────────────────────────────────────────────────────────────────────────────
# 20. test_month_over_month_is_null
# ──────────────────────────────────────────────────────────────────────────────

def test_month_over_month_is_null(llm_context):
    """month_over_month_change_pct is None in v1 for all spend_profile entries."""
    for entry in llm_context.get("spend_profile", []):
        assert entry.get("month_over_month_change_pct") is None, (
            f"Expected None for month_over_month_change_pct in category "
            f"'{entry.get('category')}', got {entry.get('month_over_month_change_pct')}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# 21. test_personal_transfers_by_alias_present
# ──────────────────────────────────────────────────────────────────────────────

def test_personal_transfers_by_alias_present(llm_context):
    """personal_transfers.by_alias is a list (may be empty if no known persons configured)."""
    pt = llm_context.get("personal_transfers", {})
    assert "by_alias" in pt, "personal_transfers must have 'by_alias' key"
    assert isinstance(pt["by_alias"], list)


# ──────────────────────────────────────────────────────────────────────────────
# 22. test_personal_transfers_self_alias_pattern_null
# ──────────────────────────────────────────────────────────────────────────────

def test_personal_transfers_self_alias_pattern_null():
    """Self:* aliases always have pattern=null regardless of personal_summary content."""
    from llm_export_aggregators import build_personal_transfers

    personal_debits = pd.DataFrame({
        Col.AMOUNT: [3000.0, 1000.0],
        Col.DATE: [datetime(2024, 1, 5), datetime(2024, 2, 5)],
        Col.KNOWN_PERSON_ALIAS: ["Self:HDFC_Savings", "Self:HDFC_Savings"],
        Col.TRANSFER_CLASS: ["transfer_self", "transfer_self"],
    })

    mock_result = MagicMock()
    mock_result.personal_debits = personal_debits
    mock_result.personal_credits = pd.DataFrame()
    # personal_summary has data for Self:HDFC_Savings (should be ignored)
    mock_result.personal_summary = {
        "Self:HDFC_Savings": {"pattern": "monthly", "avg_amount": 2000.0, "frequency_days": 30.0}
    }
    mock_result.transfer_patterns = []
    mock_result.exclusion_stats = {"exclusion_rate": 0.1}
    mock_result.debits = personal_debits

    pt = build_personal_transfers(mock_result)
    for alias_entry in pt["by_alias"]:
        if alias_entry["alias"].startswith("Self:"):
            assert alias_entry["pattern"] is None, (
                f"Self:* alias '{alias_entry['alias']}' must have pattern=null"
            )
            assert alias_entry["avg_amount"] is None
            assert alias_entry["frequency_days"] is None


# ──────────────────────────────────────────────────────────────────────────────
# 23. test_personal_transfers_inference_result_no_crash
# ──────────────────────────────────────────────────────────────────────────────

def test_personal_transfers_inference_result_no_crash():
    """Empty personal_summary (inference result) → graceful nulls, no crash."""
    from llm_export_aggregators import build_personal_transfers

    personal_debits = pd.DataFrame({
        Col.AMOUNT: [2000.0],
        Col.DATE: [datetime(2024, 3, 1)],
        Col.KNOWN_PERSON_ALIAS: ["Mom"],
        Col.TRANSFER_CLASS: ["transfer_known"],
    })

    mock_result = MagicMock()
    mock_result.personal_debits = personal_debits
    mock_result.personal_credits = pd.DataFrame()
    mock_result.personal_summary = {}   # empty — inference result (Conflict 1)
    mock_result.transfer_patterns = []
    mock_result.exclusion_stats = {}
    mock_result.debits = personal_debits

    try:
        pt = build_personal_transfers(mock_result)
    except Exception as e:
        pytest.fail(f"build_personal_transfers raised on empty personal_summary: {e}")

    # All pattern fields must be null when personal_summary is empty
    for alias_entry in pt["by_alias"]:
        assert alias_entry["pattern"] is None
        assert alias_entry["avg_amount"] is None
        assert alias_entry["frequency_days"] is None


# ──────────────────────────────────────────────────────────────────────────────
# 24. test_transfer_class_summary_all_three_classes
# ──────────────────────────────────────────────────────────────────────────────

def test_transfer_class_summary_all_three_classes():
    """transfer_class_summary always has all 3 classes (including transfer_external=0)."""
    from llm_export_aggregators import build_personal_transfers

    personal_debits = pd.DataFrame({
        Col.AMOUNT: [5000.0, 1000.0],
        Col.DATE: [datetime(2024, 1, 10), datetime(2024, 2, 10)],
        Col.KNOWN_PERSON_ALIAS: ["Mom", "Self:HDFC"],
        Col.TRANSFER_CLASS: ["transfer_known", "transfer_self"],
    })

    mock_result = MagicMock()
    mock_result.personal_debits = personal_debits
    mock_result.personal_credits = pd.DataFrame()
    mock_result.personal_summary = {}
    mock_result.transfer_patterns = []
    mock_result.exclusion_stats = {"exclusion_rate": 0.05}
    mock_result.debits = personal_debits

    pt = build_personal_transfers(mock_result)
    summary = pt["transfer_class_summary"]

    assert summary is not None, "transfer_class_summary must not be None when TRANSFER_CLASS col exists"
    assert "transfer_self" in summary
    assert "transfer_known" in summary
    assert "transfer_external" in summary, "transfer_external must always be present (zero-floor)"
    assert summary["transfer_external"]["count"] == 0
    assert summary["transfer_external"]["total_inr"] == 0.0


# ──────────────────────────────────────────────────────────────────────────────
# 25. test_transfer_class_missing_column_graceful
# ──────────────────────────────────────────────────────────────────────────────

def test_transfer_class_missing_column_graceful():
    """Missing TRANSFER_CLASS column → transfer_class_summary=null, warning logged, no crash."""
    from llm_export_aggregators import build_personal_transfers

    # personal_debits WITHOUT Col.TRANSFER_CLASS
    personal_debits = pd.DataFrame({
        Col.AMOUNT: [2000.0],
        Col.DATE: [datetime(2024, 1, 1)],
        Col.KNOWN_PERSON_ALIAS: ["Mom"],
        # Col.TRANSFER_CLASS intentionally absent
    })

    mock_result = MagicMock()
    mock_result.personal_debits = personal_debits
    mock_result.personal_credits = pd.DataFrame()
    mock_result.personal_summary = {}
    mock_result.transfer_patterns = []
    mock_result.exclusion_stats = {}
    mock_result.debits = personal_debits

    with patch("llm_export_aggregators.logger") as mock_logger:
        pt = build_personal_transfers(mock_result)
        # Verify warning was emitted (Conflict 3)
        mock_logger.warning.assert_called()

    assert pt["transfer_class_summary"] is None, (
        "transfer_class_summary must be null when TRANSFER_CLASS column is absent"
    )


# ──────────────────────────────────────────────────────────────────────────────
# 26. test_exclusion_rate_fallback_formula
# ──────────────────────────────────────────────────────────────────────────────

def test_exclusion_rate_fallback_formula():
    """exclusion_stats={} → exclusion_rate_pct computed from DataFrame lengths."""
    from llm_export_aggregators import build_personal_transfers

    personal_debits = pd.DataFrame({
        Col.AMOUNT: [1000.0, 2000.0],
        Col.DATE: [datetime(2024, 1, 1), datetime(2024, 2, 1)],
        Col.KNOWN_PERSON_ALIAS: ["Mom", "Mom"],
        Col.TRANSFER_CLASS: ["transfer_known", "transfer_known"],
    })
    # 10 total debits, 2 personal → fallback rate = 2/10 * 100 = 20.0
    all_debits = pd.DataFrame({
        Col.AMOUNT: [100.0] * 10,
        Col.DATE: [datetime(2024, 1, i + 1) for i in range(10)],
    })

    mock_result = MagicMock()
    mock_result.personal_debits = personal_debits
    mock_result.personal_credits = pd.DataFrame()
    mock_result.personal_summary = {}
    mock_result.transfer_patterns = []
    mock_result.exclusion_stats = {}   # empty — triggers fallback (Conflict 4)
    mock_result.debits = all_debits

    pt = build_personal_transfers(mock_result)
    expected_rate = round(2 / 10 * 100, 2)  # 20.0
    assert pt["exclusion_rate_pct"] == pytest.approx(expected_rate, rel=1e-3), (
        f"Fallback exclusion_rate_pct: expected {expected_rate}, got {pt['exclusion_rate_pct']}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# 27. test_passion_context_reads_os_environ_enabled
# ──────────────────────────────────────────────────────────────────────────────

def test_passion_context_reads_os_environ_enabled(monkeypatch):
    """build_passion_context returns enabled=True when env-var is 'true' (C1b)."""
    from llm_export_passion import build_passion_context
    monkeypatch.setenv("INSIGHT_ENGINE_PASSION_ENABLED", "true")
    mock_result = MagicMock()
    mock_result.passion_signals = ()
    ctx = build_passion_context(mock_result, pii_safe=True)
    assert ctx["enabled"] is True


# ──────────────────────────────────────────────────────────────────────────────
# 28. test_passion_context_reads_os_environ_disabled
# ──────────────────────────────────────────────────────────────────────────────

def test_passion_context_reads_os_environ_disabled(monkeypatch):
    """build_passion_context returns enabled=False when env-var is 'false' (C1b)."""
    from llm_export_passion import build_passion_context
    monkeypatch.setenv("INSIGHT_ENGINE_PASSION_ENABLED", "false")
    mock_result = MagicMock()
    mock_result.passion_signals = ()
    ctx = build_passion_context(mock_result, pii_safe=True)
    assert ctx["enabled"] is False
    assert ctx["signals"] == []


# ──────────────────────────────────────────────────────────────────────────────
# 29. test_build_spend_insights_paired_objects
# ──────────────────────────────────────────────────────────────────────────────

def test_build_spend_insights_paired_objects():
    """build_spend_insights returns paired objects with tip always present (C2/C6)."""
    from llm_export_aggregators import build_spend_insights
    records = [
        {"type": "spending_spike", "text": "Anomaly at X", "tip": "Tip A", "score": 0.9},
        {"type": "subscription",   "text": "Sub at Y",    "tip": "",      "score": 0.7},
    ]
    stats_in = {"total_transactions": 100, "excluded_transactions": 5, "exclusion_rate": 0.05}
    out = build_spend_insights(records, stats_in)
    assert len(out["insights"]) == 2
    assert out["insights"][0]["tip"] == "Tip A"
    assert out["insights"][1]["tip"] == ""
    assert out["insights"][0]["score"] == pytest.approx(0.9)
    assert out["stats"]["total_transactions"] == 100
    assert out["stats"]["exclusion_rate"] == pytest.approx(0.05)


# ──────────────────────────────────────────────────────────────────────────────
# 30. test_build_spend_insights_empty_records
# ──────────────────────────────────────────────────────────────────────────────

def test_build_spend_insights_empty_records():
    """Empty insight_records → insights=[], stats all zeros."""
    from llm_export_aggregators import build_spend_insights
    out = build_spend_insights([], {})
    assert out["insights"] == []
    assert out["stats"]["total_transactions"] == 0
    assert out["stats"]["exclusion_rate"] == 0.0


# ──────────────────────────────────────────────────────────────────────────────
# 31. test_build_spend_insights_skips_empty_text
# ──────────────────────────────────────────────────────────────────────────────

def test_build_spend_insights_skips_empty_text():
    """Records with empty text are filtered — malformed record guard."""
    from llm_export_aggregators import build_spend_insights
    records = [
        {"type": "spending_spike", "text": "",      "tip": "t", "score": 0.5},
        {"type": "subscription",   "text": "Valid", "tip": "",  "score": 0.3},
    ]
    out = build_spend_insights(records, {})
    assert len(out["insights"]) == 1
    assert out["insights"][0]["text"] == "Valid"
