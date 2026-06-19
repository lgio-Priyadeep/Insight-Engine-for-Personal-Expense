"""
chatbot_session_store.py — Thread-safe in-memory session store for the LLM chatbot.

B4 FIX: Background sweep performs expiration check AND deletion inside a single
with self._lock: block — atomic. Taking a snapshot outside the lock and deleting
later creates a TOCTOU window where a refreshed session is incorrectly evicted.

Known Constraint KC-06: Sessions are in-process only. Lost on server restart.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from uuid import uuid4

import config

MAX_TURNS: int      = getattr(config, "CHATBOT_MAX_TURNS", 10)
HISTORY_WINDOW: int = getattr(config, "CHATBOT_HISTORY_WINDOW", 6)
TTL_SECONDS: float  = float(getattr(config, "CHATBOT_SESSION_TTL_S", 1800))

logger = logging.getLogger(__name__)


@dataclass
class ChatSession:
    session_id:           str
    user_hash:            str
    run_id:               str
    context_path:         str
    messages:             list = field(default_factory=list)
    turn_count:           int  = 0
    last_active:          float = field(default_factory=time.monotonic)
    inference_in_progress: bool = False


class SessionStore:
    """
    Singleton in-memory session store.

    All public methods acquire self._lock for every mutation and read.
    The singleton is created once via __new__ with a class-level lock to
    prevent double-initialisation in multi-threaded startup scenarios.
    """

    _instance: "SessionStore | None" = None
    _class_lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "SessionStore":
        with cls._class_lock:
            if cls._instance is None:
                inst = super().__new__(cls)
                inst._store: dict[str, ChatSession] = {}
                inst._lock = threading.Lock()
                inst._stop_event = threading.Event()
                inst._sweep_thread = threading.Thread(
                    target=inst._sweep, daemon=True, name="chatbot-session-sweep"
                )
                inst._sweep_thread.start()
                cls._instance = inst
        return cls._instance

    # ── Public API ────────────────────────────────────────────────────────────

    def create(self, user_hash: str, run_id: str, context_path: str) -> ChatSession:
        """Create and store a new session. Returns the new ChatSession."""
        session_id = uuid4().hex          # full 32-char; never truncate
        session = ChatSession(
            session_id=session_id,
            user_hash=user_hash,
            run_id=run_id,
            context_path=context_path,
        )
        with self._lock:
            self._store[session_id] = session
        logger.debug("session_created", extra={"session_id": session_id})
        return session

    def get(self, session_id: str) -> ChatSession | None:
        """
        Return session if present and not TTL-expired; else None.
        Expired sessions are deleted on access (lazy eviction).
        """
        with self._lock:
            session = self._store.get(session_id)
            if session is None:
                return None
            if time.monotonic() - session.last_active > TTL_SECONDS:
                del self._store[session_id]
                logger.debug("session_expired_on_get", extra={"session_id": session_id})
                return None
            return session

    def append_turn(
        self, session_id: str, user_msg: str, assistant_msg: str
    ) -> None:
        """Append one user+assistant turn. All mutations inside lock."""
        with self._lock:
            session = self._store.get(session_id)
            if session is None:
                return
            session.messages.append({"role": "user",      "content": user_msg})
            session.messages.append({"role": "assistant", "content": assistant_msg})
            session.turn_count  += 1
            session.last_active  = time.monotonic()

    def set_inference_busy(self, session_id: str) -> bool:
        """
        Atomic check-and-set of inference_in_progress.
        Returns True if caller may proceed; False if already in progress (→ 429).
        """
        with self._lock:
            session = self._store.get(session_id)
            if session is None:
                return False
            if session.inference_in_progress:
                return False
            session.inference_in_progress = True
            return True

    def clear_inference_busy(self, session_id: str) -> None:
        """Clear inference flag. MUST be called in a finally block in chat_message."""
        with self._lock:
            session = self._store.get(session_id)
            if session is not None:
                session.inference_in_progress = False

    def delete(self, session_id: str) -> None:
        """Explicitly delete a session (user-initiated logout)."""
        with self._lock:
            self._store.pop(session_id, None)

    # ── Background sweep ──────────────────────────────────────────────────────

    def _sweep(self) -> None:
        """
        TTL sweep — runs every 5 minutes until _stop_event is set.

        B4 FIX: expiration check AND deletion are both inside a single
        `with self._lock:` block. This is the only correct pattern.
        DO NOT move the deletion outside the lock — that creates a TOCTOU
        window where a session refreshed by a concurrent request is evicted.
        """
        while not self._stop_event.wait(timeout=300):
            now = time.monotonic()
            with self._lock:
                expired = [
                    sid for sid, s in self._store.items()
                    if now - s.last_active > TTL_SECONDS
                ]
                for sid in expired:
                    del self._store[sid]
                    logger.debug("session_swept", extra={"session_id": sid})


__all__ = ["ChatSession", "SessionStore", "MAX_TURNS", "HISTORY_WINDOW", "TTL_SECONDS"]
