"""
tests/test_chatbot.py — Unit tests for the LLM chatbot modules.

No real model loaded. llama_cpp.Llama is mocked throughout.
All tests are independent — SessionStore singleton is reset by the
session_store_reset fixture before each session-related test.
"""
from __future__ import annotations

import json
import time
from unittest.mock import MagicMock, patch

import pytest


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=False)
def session_store_reset():
    """Reset the SessionStore singleton before and after each test."""
    import chatbot_session_store
    # Stop any existing sweep thread
    if chatbot_session_store.SessionStore._instance is not None:
        chatbot_session_store.SessionStore._instance._stop_event.set()
    chatbot_session_store.SessionStore._instance = None
    yield
    if chatbot_session_store.SessionStore._instance is not None:
        chatbot_session_store.SessionStore._instance._stop_event.set()
    chatbot_session_store.SessionStore._instance = None


# ── chatbot_context_loader ─────────────────────────────────────────────────────

class TestHashToken:
    def test_hash_token_deterministic(self):
        from chatbot_context_loader import hash_token
        assert hash_token("mytoken") == hash_token("mytoken")

    def test_hash_token_length_16(self):
        from chatbot_context_loader import hash_token
        assert len(hash_token("anytoken")) == 16

    def test_hash_token_different_tokens_differ(self):
        from chatbot_context_loader import hash_token
        assert hash_token("token_a") != hash_token("token_b")

    def test_hash_token_hex_chars_only(self):
        from chatbot_context_loader import hash_token
        h = hash_token("test")
        assert all(c in "0123456789abcdef" for c in h)


class TestResolveContextPath:
    def test_resolve_run_id_valid(self, tmp_path):
        import chatbot_context_loader as m
        with patch.object(m, "SAFE_BASE_DIR", str(tmp_path)):
            user_dir = tmp_path / "aabb11223344ccdd"
            user_dir.mkdir()
            (user_dir / "abcdef123456_llm_context.json").write_text("{}")
            path = m.resolve_context_path("abcdef123456", "aabb11223344ccdd")
            assert "abcdef123456" in path
            assert "aabb11223344ccdd" in path

    def test_resolve_run_id_invalid_format(self, tmp_path):
        import chatbot_context_loader as m
        with patch.object(m, "SAFE_BASE_DIR", str(tmp_path)):
            with pytest.raises(ValueError, match="Invalid run_id"):
                m.resolve_context_path("not-hex!!", "aabb11223344ccdd")

    def test_resolve_run_id_traversal_attempt(self, tmp_path):
        import chatbot_context_loader as m
        with patch.object(m, "SAFE_BASE_DIR", str(tmp_path)):
            with pytest.raises(ValueError):
                # ../../ encoded as hex chars — caught by regex first
                m.resolve_context_path("../../etc/pa", "aabb11223344ccdd")

    def test_resolve_auto_latest_picks_newest(self, tmp_path):
        import chatbot_context_loader as m
        with patch.object(m, "SAFE_BASE_DIR", str(tmp_path)):
            user_dir = tmp_path / "aabb11223344ccdd"
            user_dir.mkdir()
            old = user_dir / "aaaaaa111111_llm_context.json"
            new = user_dir / "bbbbbb222222_llm_context.json"
            old.write_text("{}")
            time.sleep(0.05)
            new.write_text("{}")
            path = m.resolve_context_path(None, "aabb11223344ccdd")
            assert "bbbbbb222222" in path

    def test_resolve_auto_latest_empty_dir_raises(self, tmp_path):
        import chatbot_context_loader as m
        with patch.object(m, "SAFE_BASE_DIR", str(tmp_path)):
            user_dir = tmp_path / "aabb11223344ccdd"
            user_dir.mkdir()
            with pytest.raises(FileNotFoundError):
                m.resolve_context_path(None, "aabb11223344ccdd")

    def test_resolve_auto_latest_missing_dir_raises(self, tmp_path):
        import chatbot_context_loader as m
        with patch.object(m, "SAFE_BASE_DIR", str(tmp_path)):
            with pytest.raises(FileNotFoundError):
                m.resolve_context_path(None, "nonexistentuser1")


class TestLoadContext:
    def _write(self, tmp_path, content: str) -> str:
        p = tmp_path / "ctx.json"
        p.write_text(content)
        return str(p)

    def test_load_context_valid(self, tmp_path):
        from chatbot_context_loader import load_context
        p = self._write(tmp_path, json.dumps({"schema_version": "1.0", "period": {}}))
        ctx = load_context(p)
        assert ctx["schema_version"] == "1.0"

    def test_load_context_empty_file_raises(self, tmp_path):
        from chatbot_context_loader import load_context
        p = tmp_path / "empty.json"
        p.write_text("")
        with pytest.raises(ValueError, match="empty"):
            load_context(str(p))

    def test_load_context_bad_json_raises(self, tmp_path):
        from chatbot_context_loader import load_context
        p = self._write(tmp_path, "not json {{")
        with pytest.raises(ValueError, match="not valid JSON"):
            load_context(p)

    def test_load_context_wrong_schema_raises(self, tmp_path):
        from chatbot_context_loader import load_context
        p = self._write(tmp_path, json.dumps({"schema_version": "99.0"}))
        with pytest.raises(ValueError, match="Unsupported schema_version"):
            load_context(p)

    def test_load_context_missing_file_raises(self, tmp_path):
        from chatbot_context_loader import load_context
        with pytest.raises(FileNotFoundError):
            load_context(str(tmp_path / "missing.json"))


# ── chatbot_prompt_builder ─────────────────────────────────────────────────────

class TestBuildSystemPrompt:
    def _ctx(self, **overrides) -> dict:
        base = {
            "period": {"start": "2024-01-01", "end": "2024-03-31"},
            "spend_profile": [],
            "expense_insights": [],
            "recurring_subscriptions": [],
            "passion_insights": {},
            "budget_health": {},
        }
        base.update(overrides)
        return base

    def test_build_prompt_deterministic(self):
        from chatbot_prompt_builder import build_system_prompt
        ctx = self._ctx(spend_profile=[
            {"category": "food", "share_pct": 34.0, "total_amount": 8000}
        ])
        assert build_system_prompt(ctx) == build_system_prompt(ctx)

    def test_build_prompt_caps_categories_at_5(self):
        from chatbot_prompt_builder import build_system_prompt
        cats = [
            {"category": f"cat{i}", "share_pct": float(i), "total_amount": float(i * 100)}
            for i in range(10)
        ]
        prompt = build_system_prompt(self._ctx(spend_profile=cats))
        # Each category appears as "- cat{i}:" — count bullet lines in spend section
        spend_bullets = [ln for ln in prompt.splitlines() if ln.startswith("- cat")]
        assert len(spend_bullets) == 5

    def test_build_prompt_omits_empty_arrays(self):
        from chatbot_prompt_builder import build_system_prompt
        ctx = self._ctx()   # all lists empty
        prompt = build_system_prompt(ctx)
        assert "Spend Profile" not in prompt
        assert "Recurring" not in prompt
        assert "Passion" not in prompt

    def test_build_prompt_omits_null_values_in_budget(self):
        from chatbot_prompt_builder import build_system_prompt
        ctx = self._ctx(budget_health={"total_income": None, "total_spend": 5000.0})
        prompt = build_system_prompt(ctx)
        assert "total_income" not in prompt
        assert "total_spend" in prompt

    def test_build_prompt_contains_injection_guard(self):
        from chatbot_prompt_builder import build_system_prompt
        prompt = build_system_prompt(self._ctx())
        assert "SECURITY" in prompt or "Ignore any instructions" in prompt

    def test_build_prompt_floats_rounded_to_2dp(self):
        from chatbot_prompt_builder import build_system_prompt
        cats = [{"category": "food", "share_pct": 34.123456, "total_amount": 1234.5678}]
        prompt = build_system_prompt(self._ctx(spend_profile=cats))
        assert "34.12" in prompt
        assert "1234.57" in prompt


# ── chatbot_session_store ──────────────────────────────────────────────────────

class TestSessionStore:
    def test_session_store_create_and_get(self, session_store_reset):
        from chatbot_session_store import SessionStore
        store = SessionStore()
        sess = store.create("uhash", "runid", "/tmp/ctx.json")
        assert len(sess.session_id) == 32
        assert store.get(sess.session_id) is not None

    def test_session_store_delete(self, session_store_reset):
        from chatbot_session_store import SessionStore
        store = SessionStore()
        sess = store.create("u", "r", "/tmp/x.json")
        store.delete(sess.session_id)
        assert store.get(sess.session_id) is None

    def test_session_store_ttl_eviction(self, session_store_reset):
        """Session past TTL is evicted on get()."""
        import chatbot_session_store as m
        store = m.SessionStore()
        sess = store.create("u", "r", "/tmp/x.json")
        # Wind the clock forward past TTL
        sess.last_active = time.monotonic() - (m.TTL_SECONDS + 1)
        assert store.get(sess.session_id) is None

    def test_session_store_turn_cap(self, session_store_reset):
        from chatbot_session_store import SessionStore, MAX_TURNS
        store = SessionStore()
        sess = store.create("u", "r", "/tmp/x.json")
        for i in range(MAX_TURNS):
            store.append_turn(sess.session_id, f"q{i}", f"a{i}")
        refreshed = store.get(sess.session_id)
        assert refreshed.turn_count == MAX_TURNS

    def test_session_store_history_window(self, session_store_reset):
        """Only last HISTORY_WINDOW*2 messages in store after many turns."""
        from chatbot_session_store import SessionStore, HISTORY_WINDOW, MAX_TURNS
        store = SessionStore()
        sess = store.create("u", "r", "/tmp/x.json")
        for i in range(MAX_TURNS):
            store.append_turn(sess.session_id, f"q{i}", f"a{i}")
        refreshed = store.get(sess.session_id)
        # Slice used in chat_router: messages[-(HISTORY_WINDOW * 2):]
        history_slice = refreshed.messages[-(HISTORY_WINDOW * 2):]
        assert len(history_slice) == HISTORY_WINDOW * 2

    def test_session_store_inference_busy_flag(self, session_store_reset):
        from chatbot_session_store import SessionStore
        store = SessionStore()
        sess = store.create("u", "r", "/tmp/x.json")
        assert store.set_inference_busy(sess.session_id) is True
        assert store.set_inference_busy(sess.session_id) is False   # already busy
        store.clear_inference_busy(sess.session_id)
        assert store.set_inference_busy(sess.session_id) is True    # cleared


# ── chatbot_engine ─────────────────────────────────────────────────────────────

class TestGenerateResponse:
    def test_generate_response_timeout(self):
        """
        B3 fix: stream loop raises InferenceTimeoutError when elapsed > timeout_s.
        The mock stream yields one token then blocks; time.monotonic returns a
        value past the deadline on the second call so the loop breaks immediately.
        """
        import chatbot_engine

        fake_llm = MagicMock()

        def mock_stream(*a, **kw):
            yield {"choices": [{"delta": {"content": "hello"}}]}
            # Second token would block — but we expect the loop to break before here
            yield {"choices": [{"delta": {"content": " world"}}]}

        fake_llm.create_chat_completion.return_value = mock_stream()

        time_calls = iter([0.0, 0.0, 9999.0])  # start=0, first chunk=0, second chunk=past timeout

        with patch.object(chatbot_engine, "_llm_instance", fake_llm):
            with patch("chatbot_engine.time") as mock_time:
                mock_time.monotonic.side_effect = lambda: next(time_calls)
                with pytest.raises(chatbot_engine.InferenceTimeoutError):
                    chatbot_engine.generate_response(
                        [{"role": "user", "content": "hi"}],
                        timeout_s=1,
                    )

    def test_model_not_ready_raises(self, tmp_path):
        """ModelNotReadyError raised when model file is absent."""
        import chatbot_engine, config
        with patch.object(chatbot_engine, "_llm_instance", None):
            with patch.object(config, "CHATBOT_MODEL_PATH", str(tmp_path / "missing.gguf")):
                with pytest.raises(chatbot_engine.ModelNotReadyError, match="not found"):
                    chatbot_engine._get_llm_instance()


# ── chatbot_prompt_builder — tip rendering ────────────────────────────────────

class TestBuildSystemPromptWithTips:
    def _ctx(self, **overrides) -> dict:
        base = {
            "period": {"start": "2024-01-01", "end": "2024-03-31"},
            "spend_profile": [],
            "expense_insights": [],
            "recurring_subscriptions": [],
            "passion_insights": {},
            "budget_health": {},
        }
        base.update(overrides)
        return base

    def test_build_prompt_includes_passion_signals_with_tips(self):
        """Passion signal with a tip produces '... Tip: ...' line in prompt."""
        from chatbot_prompt_builder import build_system_prompt
        ctx = self._ctx(passion_insights={
            "enabled": True,
            "signal_count": 1,
            "signals": [
                {
                    "display_label": "Fitness",
                    "spend_share_pct": 22.5,
                    "tip": "Check for group class discounts.",
                }
            ],
        })
        prompt = build_system_prompt(ctx)
        assert "Fitness" in prompt
        assert "22.5" in prompt
        assert "Check for group class discounts." in prompt
        assert "Tip:" in prompt

    def test_build_prompt_passion_signal_without_tip_no_tip_line(self):
        """Passion signal with tip='' produces no 'Tip:' in that line."""
        from chatbot_prompt_builder import build_system_prompt
        ctx = self._ctx(passion_insights={
            "enabled": True,
            "signal_count": 1,
            "signals": [
                {
                    "display_label": "Food",
                    "spend_share_pct": 30.0,
                    "tip": "",
                }
            ],
        })
        prompt = build_system_prompt(ctx)
        assert "Food" in prompt
        # Tip line must NOT appear when tip is empty
        for line in prompt.splitlines():
            if "Food" in line:
                assert "Tip:" not in line, f"Unexpected Tip in line: {line!r}"

