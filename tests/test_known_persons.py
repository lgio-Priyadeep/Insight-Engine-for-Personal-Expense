import pytest
import pandas as pd
import numpy as np

from known_persons import (
    tag_known_persons,
    _enforce_known_person_schema,
    _suggestion_key,
    _extract_signals
)
from pipeline import run_pipeline, run_inference, PipelineResult, _compute_config_hash
from model_state import InsightModelState
from schema import Col

# Basic fixtures for testing
@pytest.fixture
def test_known_persons_cfg():
    return {
        "Mom": {
            "names": ["sujata", "sujata devi"],
            "upi_ids": ["sujata@ybl"]
        },
        "Landlord": {
            "names": ["amit sharma", "amit"],
            "upi_ids": ["amit@okicici"]
        }
    }

@pytest.fixture
def test_self_accounts_cfg():
    return {
        "HDFC_Savings": {
            "names": ["rahul kumar"],
            "account_fragments": ["50100"],
            "upi_ids": ["rahul@hdfcbank"]
        }
    }

def _create_minimal_test_df(docs: list[str]) -> pd.DataFrame:
    df = pd.DataFrame({"remarks": docs})
    df[Col.REMARKS] = df["remarks"]
    df[Col.CLEANED_REMARKS] = df["remarks"].str.lower()
    df[Col.DATE] = "2023-01-01"
    df[Col.AMOUNT] = 1000.0
    df[Col.AMOUNT_FLAG] = "Dr"
    df[Col.BALANCE] = 5000.0
    return df

def test_extract_signals_upi_digits():
    bundle = _extract_signals("sent to rahul@oksbi123")
    assert "rahul@oksbi123" in bundle.upi_ids
    
def test_tagging_exact_upi(test_known_persons_cfg):
    df = _create_minimal_test_df(["paid sujata@ybl for rent"])
    df = tag_known_persons(df, known_persons=test_known_persons_cfg, self_accounts={})
    assert bool(df.loc[0, Col.IS_KNOWN_PERSON]) is True
    assert df.loc[0, Col.KNOWN_PERSON_ALIAS] == "Mom"
    assert df.loc[0, Col.TRANSFER_CLASS] == "transfer_known"
    
def test_tagging_self_account_fragment(test_self_accounts_cfg):
    df = _create_minimal_test_df(["transfer to a/c 50100"])
    df = tag_known_persons(df, known_persons={}, self_accounts=test_self_accounts_cfg)
    assert bool(df.loc[0, Col.IS_KNOWN_PERSON]) is True
    assert df.loc[0, Col.KNOWN_PERSON_ALIAS] == "Self:HDFC_Savings"
    assert df.loc[0, Col.TRANSFER_CLASS] == "transfer_self"

def test_merchant_suppression(test_known_persons_cfg):
    df = _create_minimal_test_df(["paid amit electronics store"])
    df = tag_known_persons(df, known_persons=test_known_persons_cfg, self_accounts={})
    assert bool(df.loc[0, Col.IS_KNOWN_PERSON]) is False

def test_concat_partial_with_bounds(test_known_persons_cfg):
    df = _create_minimal_test_df(["sujataglobal", "sujataservices", "sujatadevi"])
    df = tag_known_persons(df, known_persons=test_known_persons_cfg, self_accounts={})
    df_ctx = _create_minimal_test_df(["upi sujataglobal", "upi sujataservices", "upi sujatadevi"])
    df_ctx = tag_known_persons(df_ctx, known_persons=test_known_persons_cfg, self_accounts={})
    
    assert bool(df_ctx.loc[0, Col.IS_KNOWN_PERSON]) is False
    assert bool(df_ctx.loc[1, Col.IS_KNOWN_PERSON]) is False
    assert bool(df_ctx.loc[2, Col.IS_KNOWN_PERSON]) is True

def test_enforce_schema_schema():
    df = _create_minimal_test_df(["a", "b"])
    df[Col.IS_KNOWN_PERSON] = [True, False]
    df[Col.EXPECTED_AMOUNT] = [500.0, 500.0]
    
    df_enforced = _enforce_known_person_schema(df)
    assert pd.isna(df_enforced.loc[0, Col.EXPECTED_AMOUNT])
    assert pd.notna(df_enforced.loc[1, Col.EXPECTED_AMOUNT])

def test_suggestion_key_min_len():
    bundle = _extract_signals("rahul electronics store")
    key = _suggestion_key(bundle)
    assert key == "rahul"
    
def test_state_version_uses_di_params_not_globals():
    from known_persons import tag_known_persons
    
    # Preprocessor deduplicates based on date+amount, so we need unique dates/amounts
    # We also need valid seed keywords (e.g. swiggy, uber) so the model doesn't drop unmatched rows.
    remarks = ["swiggy"] * 15 + ["uber"] * 14
    
    raw_df = pd.DataFrame({
        Col.DATE: [f"2023-01-{i:02d}" for i in range(1, 30)],
        Col.AMOUNT: [100.0 + i for i in range(29)],
        Col.AMOUNT_FLAG: ["Dr"] * 29,
        Col.REMARKS: remarks,
        Col.BALANCE: [1000.0] * 29
    })
    
    kp1 = {"Mom": {"names": [], "upi_ids": ["sujata@ybl"]}}
    res1 = run_pipeline(raw_df, known_persons=kp1, self_accounts={})
    
    state1 = InsightModelState(
        pipeline_version="1.0.0",
        cat_pipeline=res1.cat_pipeline,
        spend_pipeline=res1.spend_pipeline,
        ranker_pipeline=res1.ranker_pipeline,
        global_mean=res1.global_mean,
        global_std=res1.global_std,
        stats_version=res1.stats_version,
        kp_config_hash=res1.kp_config_hash
    )
    
    # Should work fine with same DI
    run_inference(raw_df.head(2).copy(), state1, res1.debits, known_persons=kp1, self_accounts={})
    
    # Should fail if DI changed
    kp2 = {"Mom2": {"names": [], "upi_ids": ["other@ybl"]}}
    with pytest.raises(ValueError, match="config has changed since"):
         run_inference(raw_df.head(2).copy(), state1, res1.debits, known_persons=kp2, self_accounts={})
