"""
tests/test_chatbot_api.py — FastAPI TestClient integration tests for chat endpoints.

Model is mocked via monkeypatch — no real inference runs.
SessionStore singleton is reset between tests so sessions don't bleed.
"""
from __future__ import annotations

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_session_store():
    """Reset SessionStore singleton before every test."""
    import chatbot_session_store
    if chatbot_session_store.SessionStore._instance is not None:
        chatbot_session_store.SessionStore._instance._stop_event.set()
    chatbot_session_store.SessionStore._instance = None
    yield
    if chatbot_session_store.SessionStore._instance is not None:
        chatbot_session_store.SessionStore._instance._stop_event.set()
    chatbot_session_store.SessionStore._instance = None


@pytest.fixture()
def mock_generate(monkeypatch):
    """Patch generate_response to return a canned answer instantly."""
    import chat_router
    monkeypatch.setattr(chat_router, "generate_response", lambda msgs, **kw: "Mocked answer.")
    import chatbot_engine
    monkeypatch.setattr(chatbot_engine, "_llm_instance", MagicMock())


@pytest.fixture()
def mock_model_ready(monkeypatch):
    """Patch _get_llm_instance so model-ready check passes."""
    import chat_router
    monkeypatch.setattr(chat_router, "_get_llm_instance", lambda: MagicMock())


@pytest.fixture()
def context_file(tmp_path):
    """Write a valid minimal LLM context file. Returns (path, run_id, user_hash)."""
    from chatbot_context_loader import hash_token
    token    = "test-token-abc"
    user_hash = hash_token(token)
    run_id   = "abcdef123456"
    user_dir = tmp_path / user_hash
    user_dir.mkdir()
    ctx = {
        "schema_version": "1.0",
        "run_id": run_id,
        "period": {"start": "2024-01-01", "end": "2024-03-31"},
        "spend_profile": [],
        "expense_insights": [],
        "recurring_subscriptions": [],
        "passion_insights": {},
        "budget_health": {},
        "anomaly_summary": {},
        "personal_transfers": {},
    }
    (user_dir / f"{run_id}_llm_context.json").write_text(json.dumps(ctx))
    return {"path": str(tmp_path), "run_id": run_id, "user_hash": user_hash, "token": token}


@pytest.fixture()
def client(context_file, mock_generate, mock_model_ready, monkeypatch):
    """TestClient with context loader pointing at tmp_path."""
    import chatbot_context_loader
    monkeypatch.setattr(chatbot_context_loader, "SAFE_BASE_DIR", context_file["path"])
    from api import app
    return TestClient(app, raise_server_exceptions=False)


TOKEN   = "test-token-abc"
BAD_TOK = "wrong-user-xyz"
HEADERS = {"X-Merchant-Token": TOKEN}
BAD_HDR = {"X-Merchant-Token": BAD_TOK}

# ── /chat/start tests ─────────────────────────────────────────────────────────

class TestChatStart:
    def test_missing_token_401(self, client):
        r = client.post("/api/v1/chat/start", json={})
        assert r.status_code == 401
        assert r.json()["error"] == "missing_token"

    def test_invalid_run_id_format_400(self, client):
        r = client.post("/api/v1/chat/start",
                        json={"run_id": "not-hex!"},
                        headers=HEADERS)
        assert r.status_code == 400
        assert r.json()["error"] == "invalid_run_id_format"

    def test_run_id_not_found_404(self, client):
        r = client.post("/api/v1/chat/start",
                        json={"run_id": "000000000000"},
                        headers=HEADERS)
        assert r.status_code == 404
        assert r.json()["error"] == "context_not_found"

    def test_auto_latest_no_files_424(self, client, context_file, monkeypatch):
        """Empty user dir → 424 no_context_available."""
        import chatbot_context_loader
        empty = tempfile.mkdtemp()
        from chatbot_context_loader import hash_token
        user_dir = os.path.join(empty, hash_token(TOKEN))
        os.makedirs(user_dir)
        monkeypatch.setattr(chatbot_context_loader, "SAFE_BASE_DIR", empty)
        r = client.post("/api/v1/chat/start", json={}, headers=HEADERS)
        assert r.status_code == 424
        assert r.json()["error"] == "no_context_available"

    def test_success_200(self, client, context_file):
        r = client.post("/api/v1/chat/start",
                        json={"run_id": context_file["run_id"]},
                        headers=HEADERS)
        assert r.status_code == 200
        body = r.json()
        assert "session_id" in body
        assert body["run_id"] == context_file["run_id"]
        assert body["turns_remaining"] == 10
        assert len(body["session_id"]) == 32


# ── /chat/message tests ───────────────────────────────────────────────────────

def _start_session(client, context_file) -> str:
    r = client.post("/api/v1/chat/start",
                    json={"run_id": context_file["run_id"]},
                    headers=HEADERS)
    assert r.status_code == 200
    return r.json()["session_id"]


class TestChatMessage:
    def test_missing_token_401(self, client, context_file):
        sid = _start_session(client, context_file)
        r = client.post("/api/v1/chat/message",
                        json={"session_id": sid, "message": "hi"})
        assert r.status_code == 401

    def test_invalid_session_400(self, client):
        r = client.post("/api/v1/chat/message",
                        json={"session_id": "a" * 32, "message": "hi"},
                        headers=HEADERS)
        assert r.status_code == 400
        assert r.json()["error"] == "invalid_session"

    def test_message_too_long_422(self, client, context_file):
        sid = _start_session(client, context_file)
        r = client.post("/api/v1/chat/message",
                        json={"session_id": sid, "message": "x" * 513},
                        headers=HEADERS)
        assert r.status_code == 422
        assert r.json()["error"] == "message_too_long"

    def test_success_200(self, client, context_file):
        sid = _start_session(client, context_file)
        r = client.post("/api/v1/chat/message",
                        json={"session_id": sid, "message": "What is my top category?"},
                        headers=HEADERS)
        assert r.status_code == 200
        body = r.json()
        assert body["answer"] == "Mocked answer."
        assert body["turn_number"] == 1
        assert body["turns_remaining"] == 9

    def test_wrong_owner_403(self, client, context_file):
        """IDOR guard: different token cannot message another user's session."""
        sid = _start_session(client, context_file)
        r = client.post("/api/v1/chat/message",
                        json={"session_id": sid, "message": "hi"},
                        headers=BAD_HDR)
        assert r.status_code == 403
        assert r.json()["error"] == "forbidden"

    def test_session_exhausted_400(self, client, context_file):
        """After MAX_TURNS messages, next message returns session_exhausted."""
        from chatbot_session_store import MAX_TURNS
        sid = _start_session(client, context_file)
        for _ in range(MAX_TURNS):
            r = client.post("/api/v1/chat/message",
                            json={"session_id": sid, "message": "q"},
                            headers=HEADERS)
            assert r.status_code == 200
        r = client.post("/api/v1/chat/message",
                        json={"session_id": sid, "message": "one more"},
                        headers=HEADERS)
        assert r.status_code == 400
        assert r.json()["error"] == "session_exhausted"


# ── /chat/session DELETE tests ────────────────────────────────────────────────

class TestChatDelete:
    def test_missing_token_401(self, client, context_file):
        sid = _start_session(client, context_file)
        r = client.delete(f"/api/v1/chat/session/{sid}")
        assert r.status_code == 401

    def test_wrong_owner_403(self, client, context_file):
        sid = _start_session(client, context_file)
        r = client.delete(f"/api/v1/chat/session/{sid}", headers=BAD_HDR)
        assert r.status_code == 403
        assert r.json()["error"] == "forbidden"

    def test_delete_success_200(self, client, context_file):
        sid = _start_session(client, context_file)
        r = client.delete(f"/api/v1/chat/session/{sid}", headers=HEADERS)
        assert r.status_code == 200
        assert r.json()["status"] == "deleted"

    def test_deleted_session_returns_400_on_message(self, client, context_file):
        sid = _start_session(client, context_file)
        client.delete(f"/api/v1/chat/session/{sid}", headers=HEADERS)
        r = client.post("/api/v1/chat/message",
                        json={"session_id": sid, "message": "hi"},
                        headers=HEADERS)
        assert r.status_code == 400
        assert r.json()["error"] == "invalid_session"

    def test_delete_nonexistent_session_400(self, client):
        r = client.delete(f"/api/v1/chat/session/{'z' * 32}", headers=HEADERS)
        assert r.status_code == 400
        assert r.json()["error"] == "invalid_session"


# ── Error envelope shape ──────────────────────────────────────────────────────

class TestErrorEnvelopeShape:
    def test_all_error_fields_present(self, client):
        """Every 4xx error must contain error, message, request_id fields."""
        r = client.post("/api/v1/chat/start", json={})   # missing token → 401
        body = r.json()
        assert "error"      in body, "missing 'error' field"
        assert "message"    in body, "missing 'message' field"
        assert "request_id" in body, "missing 'request_id' field"

    def test_request_id_not_unknown(self, client):
        """request_id must not fall back to 'unknown' — handler sets it at entry."""
        r = client.post("/api/v1/chat/start", json={})
        assert r.json()["request_id"] != "unknown"
