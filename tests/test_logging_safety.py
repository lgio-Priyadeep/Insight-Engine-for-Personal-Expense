import os
import sys
import importlib
import logging
import pytest
import pandas as pd

from pipeline import run_pipeline, generate_new_run_id
from recurring_detector import find_recurring_transactions
import config
from hash_utils import stable_hash

from schema import Col

@pytest.fixture
def valid_debits_df():
    return pd.DataFrame({
        Col.DATE: pd.date_range("2023-01-01", periods=10),
        Col.AMOUNT: [10.0] * 10,
        Col.SIGNED_AMOUNT: [-10.0] * 10,
        Col.CLEANED_REMARKS: ["Test"] * 10,
        Col.IS_KNOWN_PERSON: [False] * 10,
        "IS_DEBIT": [True] * 10,
    })

@pytest.fixture
def valid_credits_df():
    return pd.DataFrame({
        Col.DATE: pd.date_range("2023-01-01", periods=10),
        Col.AMOUNT: [10.0] * 10,
        Col.SIGNED_AMOUNT: [10.0] * 10,
        Col.CLEANED_REMARKS: ["Test"] * 10,
        Col.IS_KNOWN_PERSON: [False] * 10,
        "IS_DEBIT": [False] * 10,
    })

@pytest.fixture
def raw_df():
    # Only needs minimal structure for preprocess to return our mocked data
    return pd.DataFrame()

def test_crash_dump_created(monkeypatch, tmp_path, raw_df, valid_debits_df, valid_credits_df):
    """Test 1: Normal crash dump is correctly generated upon catastrophic pipeline failure."""
    monkeypatch.setattr(config, "ENABLE_CRASH_DUMPS", True)
    monkeypatch.setattr(config, "CRASH_DUMP_DIR", str(tmp_path))

    # Force preprocessing to return deterministic frames
    monkeypatch.setattr("pipeline.preprocess", lambda df: (valid_debits_df, valid_credits_df))
    
    # Force a known run_id so we can assert the final path
    monkeypatch.setattr("pipeline.generate_new_run_id", lambda: "test-run-001")
    run_id = "test-run-001"
    
    # Introduce deterministic crash AFTER preprocessing starts
    def _crash(*args, **kwargs):
        raise ValueError("Simulated Error")
    monkeypatch.setattr("pipeline.tag_known_persons", _crash)

    with pytest.raises(ValueError, match="Simulated Error"):
        run_pipeline(raw_df)

    # Validate state was correctly dumped instead of being swallowed
    df_dump = pd.read_csv(f"{str(tmp_path)}/{run_id}_debits.csv")
    assert list(df_dump.columns) == list(valid_debits_df.columns)
    assert len(df_dump) <= 1000

def test_crash_exception_identity_matching(monkeypatch, tmp_path, raw_df, valid_debits_df, valid_credits_df):
    """Test 2: Exception Identity Matching. The original error is evaluated verbatim."""
    monkeypatch.setattr(config, "ENABLE_CRASH_DUMPS", True)
    monkeypatch.setattr(config, "CRASH_DUMP_DIR", str(tmp_path))

    monkeypatch.setattr("pipeline.preprocess", lambda df: (valid_debits_df, valid_credits_df))
    monkeypatch.setattr("pipeline.generate_new_run_id", lambda: "test-run-001")
    
    # Exception source
    def _crash(*args, **kwargs):
        raise ValueError("Explicit Signal Bubble")
    monkeypatch.setattr("pipeline.tag_known_persons", _crash)

    with pytest.raises(ValueError, match="Explicit Signal Bubble"):
        run_pipeline(raw_df)

def test_crash_dump_failure_safety(monkeypatch, tmp_path, raw_df, valid_debits_df, valid_credits_df, caplog):
    """Test 3: If the crash dump physically fails, we log it and STILL propagate the original fault smoothly."""
    monkeypatch.setattr(config, "ENABLE_CRASH_DUMPS", True)
    
    # Create a file instead of a directory, then tell the pipeline to treat it as a directory.
    # This guarantees a native OS error (NotADirectoryError) regardless of container root privileges.
    file_collision = tmp_path / "not_a_dir.txt"
    file_collision.touch()
    monkeypatch.setattr(config, "CRASH_DUMP_DIR", str(file_collision / "crashes"))

    monkeypatch.setattr("pipeline.preprocess", lambda df: (valid_debits_df, valid_credits_df))
    monkeypatch.setattr("pipeline.generate_new_run_id", lambda: "test-run-001")
    
    def _crash(*args, **kwargs):
        raise ValueError("Simulated Error")
    monkeypatch.setattr("pipeline.tag_known_persons", _crash)

    monkeypatch.setattr(logging.getLogger("pipeline"), "propagate", True)
    
    with caplog.at_level(logging.INFO):
        with pytest.raises(ValueError, match="Simulated Error"):
            run_pipeline(raw_df)

    records = [getattr(record, "event_type", "NONE") for record in caplog.records]
    print(f"DEBUG: Records: {records}")
    print(f"DEBUG: caplog.text:\n{caplog.text}")
    # Check if we caught the backup
    assert "crash_dump_failed" in records

def test_invalid_log_level(monkeypatch):
    """Test 4: Evaluating the strict configuration constraint block actively enforcing boot behavior."""
    monkeypatch.setenv("INSIGHT_LOG_LEVEL", "GARBAGE")
    
    # Unload instance from sys mapping intentionally
    # This evaluates native OS startup behavior flawlessly simulating config start
    sys.modules.pop("config", None)
    
    with pytest.raises(ValueError, match="Invalid LOG_LEVEL: GARBAGE"):
        importlib.import_module("config")

def test_pii_redaction_coverage(monkeypatch, caplog):
    """Test 5: Validating deterministic logging paths apply proper data transforms preventing state leakage."""
    # Ensure PII logs are disabled naturally as they are in production bounds
    monkeypatch.setattr(config, "ENABLE_PII_DEBUG_LOGS", False)
    
    # Generate bounded synthetic data representing sensitive information payload
    df = pd.DataFrame({
        Col.CLEANED_REMARKS: ["Netflix", "Netflix", "Netflix"], 
        Col.DATE: pd.date_range("2023-01-01", periods=3), 
        Col.AMOUNT: [10.0, 10.0, 10.0]
    })
    
    # Observe output scope securely
    with caplog.at_level(logging.DEBUG, logger="recurring_detector"):
        find_recurring_transactions(df, group_col=Col.CLEANED_REMARKS)
        
    assert not any("Netflix" in msg for msg in caplog.messages)
    assert any(stable_hash("Netflix") in msg for msg in caplog.messages)
