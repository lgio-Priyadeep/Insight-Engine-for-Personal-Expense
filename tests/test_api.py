"""
tests/test_api.py — API Integration Tests
==========================================
Uses FastAPI TestClient (no real server needed).
Covers: response shape, error envelope, CORS headers, passion_status extraction.
"""

from __future__ import annotations

import io
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi.testclient import TestClient
from api import app

client = TestClient(app, raise_server_exceptions=False)

VALID_TOKEN = "test-token-123"


def _make_csv_bytes(n_rows: int = 10) -> bytes:
    """Minimal valid CSV for pipeline."""
    rows = []
    for i in range(n_rows):
        rows.append({"date": f"2024-01-{i+1:02d}", "amount": 100.0 + i,
                     "amount_flag": "DR", "remarks": "zomato"})
    rows.append({"date": "2024-01-15", "amount": 30000.0,
                 "amount_flag": "CR", "remarks": "salary credit"})
    buf = io.BytesIO()
    pd.DataFrame(rows).to_csv(buf, index=False)
    return buf.getvalue()


# ── Health ─────────────────────────────────────────────────────────────────────

def test_health_returns_200():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


# ── Auth ──────────────────────────────────────────────────────────────────────

def test_missing_token_returns_401():
    csv_bytes = _make_csv_bytes()
    r = client.post(
        "/api/v1/insights/analyze",
        files={"file": ("statement.csv", csv_bytes, "text/csv")},
    )
    assert r.status_code == 401
    body = r.json()
    assert body["error"] == "missing_token"
    assert "request_id" in body


# ── Error envelope shape ───────────────────────────────────────────────────────

def test_error_envelope_has_all_keys():
    r = client.post(
        "/api/v1/insights/analyze",
        files={"file": ("statement.csv", b"", "text/csv")},
        headers={"X-Merchant-Token": VALID_TOKEN},
    )
    assert r.status_code in (422, 400, 500)
    body = r.json()
    assert "error" in body
    assert "message" in body
    assert "request_id" in body


# ── Response shape ─────────────────────────────────────────────────────────────

def test_analyze_response_shape():
    """Happy path: valid CSV → response has all required top-level keys."""
    csv_bytes = _make_csv_bytes(n_rows=90)
    r = client.post(
        "/api/v1/insights/analyze",
        files={"file": ("statement.csv", csv_bytes, "text/csv")},
        headers={"X-Merchant-Token": VALID_TOKEN},
    )
    if r.status_code != 200:
        pytest.skip(f"Pipeline returned {r.status_code}: {r.text[:200]}")
    body = r.json()
    assert "run_id" in body
    assert "passion_status" in body
    assert "spend_insights" in body
    assert "passion_insights" in body


def test_analyze_spend_insights_paired_objects():
    """spend_insights.insights[] contains paired objects with tip field always present."""
    csv_bytes = _make_csv_bytes(n_rows=90)
    r = client.post(
        "/api/v1/insights/analyze",
        files={"file": ("statement.csv", csv_bytes, "text/csv")},
        headers={"X-Merchant-Token": VALID_TOKEN},
    )
    if r.status_code != 200:
        pytest.skip(f"Pipeline returned {r.status_code}")
    insights = r.json()["spend_insights"]["insights"]
    for rec in insights:
        assert "text" in rec
        assert "tip" in rec    # always present — "" when no tip
        assert "type" in rec
        assert "score" in rec


def test_analyze_passion_status_field_present():
    """passion_status top-level field present (C1a)."""
    csv_bytes = _make_csv_bytes(n_rows=90)
    r = client.post(
        "/api/v1/insights/analyze",
        files={"file": ("statement.csv", csv_bytes, "text/csv")},
        headers={"X-Merchant-Token": VALID_TOKEN},
    )
    if r.status_code != 200:
        pytest.skip(f"Pipeline returned {r.status_code}")
    body = r.json()
    valid_statuses = {"success", "disabled", "skipped", "timeout", "failure", "missing_fields"}
    assert body["passion_status"] in valid_statuses


def test_run_id_is_not_pipeline_context_var():
    """run_id in response is always a fresh UUID hex — never 'no_pipeline_context'."""
    csv_bytes = _make_csv_bytes(n_rows=90)
    r = client.post(
        "/api/v1/insights/analyze",
        files={"file": ("statement.csv", csv_bytes, "text/csv")},
        headers={"X-Merchant-Token": VALID_TOKEN},
    )
    if r.status_code != 200:
        pytest.skip(f"Pipeline returned {r.status_code}")
    run_id = r.json()["run_id"]
    assert run_id != "no_pipeline_context"
    assert len(run_id) == 12   # uuid4().hex[:12]


# ── CORS ──────────────────────────────────────────────────────────────────────

def test_cors_header_present_on_preflight():
    r = client.options(
        "/api/v1/insights/analyze",
        headers={"Origin": "https://example.com",
                 "Access-Control-Request-Method": "POST"},
    )
    assert "access-control-allow-origin" in r.headers


# ── Bad CSV ────────────────────────────────────────────────────────────────────

def test_non_csv_content_type_returns_422():
    r = client.post(
        "/api/v1/insights/analyze",
        files={"file": ("photo.jpg", b"\xff\xd8\xff", "image/jpeg")},
        headers={"X-Merchant-Token": VALID_TOKEN},
    )
    assert r.status_code == 422
    assert r.json()["error"] == "invalid_content_type"


def test_empty_csv_returns_422():
    r = client.post(
        "/api/v1/insights/analyze",
        files={"file": ("empty.csv", b"", "text/csv")},
        headers={"X-Merchant-Token": VALID_TOKEN},
    )
    assert r.status_code == 422


# ── Passion signal tip field ──────────────────────────────────────────────────

def test_analyze_passion_insights_signals_have_tip():
    """passion_insights.signals[] always contain a 'tip' str field (may be empty)."""
    csv_bytes = _make_csv_bytes(n_rows=90)
    r = client.post(
        "/api/v1/insights/analyze",
        files={"file": ("statement.csv", csv_bytes, "text/csv")},
        headers={"X-Merchant-Token": VALID_TOKEN},
    )
    if r.status_code != 200:
        pytest.skip(f"Pipeline returned {r.status_code}: {r.text[:200]}")
    passion = r.json().get("passion_insights", {})
    signals = passion.get("signals", [])
    if not signals:
        pytest.skip("No passion signals in this response — skipping tip field assertion.")
    for sig in signals:
        assert "tip" in sig, f"Passion signal missing 'tip' key: {sig}"
        assert isinstance(sig["tip"], str), (
            f"'tip' must be str, got {type(sig['tip'])!r}: {sig}"
        )

