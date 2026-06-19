"""
chat_router.py — FastAPI APIRouter for all /api/v1/chat/* endpoints.

Kept separate from api.py to honour single-responsibility and prevent api.py
from growing unbounded. Registered in api.py via app.include_router(chat_router).

Security contracts (all three handlers):
  - X-Merchant-Token header required; missing → 401 missing_token.
  - request_id = uuid4().hex[:12] set at handler entry; written to
    request.state.request_id so the global exception handler can embed it
    in flat error envelopes (Mandatory API Constraint #5).
  - IDOR guard: hash_token(token) == session.user_hash checked on every
    handler that touches a session. Prevents one user reading or exhausting
    another user's session.
"""
from __future__ import annotations

import asyncio
import logging
from uuid import uuid4

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from api_schemas import (
    ChatStartRequest,
    ChatStartResponse,
    ChatMessageRequest,
    ChatMessageResponse,
    ChatDeleteResponse,
    ErrorResponse,
)
from chatbot_context_loader import hash_token, resolve_context_path, load_context
from chatbot_engine import (
    generate_response,
    ModelNotReadyError,
    InferenceTimeoutError,
    _get_llm_instance,
)
from chatbot_prompt_builder import build_system_prompt
from chatbot_session_store import SessionStore, MAX_TURNS, HISTORY_WINDOW

logger = logging.getLogger(__name__)

chat_router = APIRouter(prefix="/api/v1/chat", tags=["chat"])

_store = SessionStore()   # singleton — same object across all requests


# ── Helpers ───────────────────────────────────────────────────────────────────

def _err(status: int, error: str, message: str, request_id: str) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content=ErrorResponse(
            error=error, message=message, request_id=request_id
        ).model_dump(),
    )


def _get_token(request: Request) -> str:
    """Extract X-Merchant-Token header value (stripped). Empty string if absent."""
    return request.headers.get("X-Merchant-Token", "").strip()


# ── POST /api/v1/chat/start ───────────────────────────────────────────────────

@chat_router.post("/start", response_model=ChatStartResponse)
async def chat_start(body: ChatStartRequest, request: Request):
    request_id = uuid4().hex[:12]
    request.state.request_id = request_id

    token = _get_token(request)
    if not token:
        return _err(401, "missing_token", "X-Merchant-Token header is required.", request_id)

    user_hash = hash_token(token)

    # Resolve context file path
    try:
        path = resolve_context_path(body.run_id, user_hash)
    except ValueError:
        return _err(
            400, "invalid_run_id_format",
            "run_id must be exactly 12 lowercase hex characters.", request_id
        )
    except FileNotFoundError:
        if body.run_id is not None:
            return _err(404, "context_not_found",
                        "No context file found for this run_id.", request_id)
        return _err(424, "no_context_available",
                    "No exported analysis found for this user. Run analysis first.", request_id)

    # Load and validate context file
    try:
        context = load_context(path)
    except (ValueError, FileNotFoundError) as exc:
        return _err(422, "context_corrupted", str(exc), request_id)

    # Verify model is available before creating the session
    try:
        _get_llm_instance()
    except ModelNotReadyError as exc:
        return _err(503, "model_not_ready", str(exc), request_id)

    run_id_echo = body.run_id if body.run_id is not None else context.get("run_id", "")
    session = _store.create(user_hash=user_hash, run_id=run_id_echo, context_path=path)

    return ChatStartResponse(
        session_id=session.session_id,
        run_id=run_id_echo,
        period=context.get("period") or {},
        turns_remaining=MAX_TURNS,
    )


# ── POST /api/v1/chat/message ─────────────────────────────────────────────────

@chat_router.post("/message", response_model=ChatMessageResponse)
async def chat_message(body: ChatMessageRequest, request: Request):
    request_id = uuid4().hex[:12]
    request.state.request_id = request_id

    token = _get_token(request)
    if not token:
        return _err(401, "missing_token", "X-Merchant-Token header is required.", request_id)

    session = _store.get(body.session_id)
    if session is None:
        return _err(400, "invalid_session", "Session not found or expired.", request_id)

    # IDOR guard
    if hash_token(token) != session.user_hash:
        return _err(403, "forbidden", "Token does not own this session.", request_id)

    if session.turn_count >= MAX_TURNS:
        return _err(400, "session_exhausted", f"Turn cap ({MAX_TURNS}) reached.", request_id)

    if len(body.message) > 512:
        return _err(422, "message_too_long",
                    "Message must be 512 characters or fewer.", request_id)

    if not _store.set_inference_busy(body.session_id):
        return _err(429, "inference_busy",
                    "A prior message is still being processed.", request_id)

    try:
        context = load_context(session.context_path)
        system_prompt = build_system_prompt(context)

        # Last HISTORY_WINDOW turns (each turn = 2 messages: user + assistant)
        history = session.messages[-(HISTORY_WINDOW * 2):]
        messages = (
            [{"role": "system", "content": system_prompt}]
            + history
            + [{"role": "user", "content": body.message}]
        )

        answer: str = await asyncio.get_running_loop().run_in_executor(
            None, generate_response, messages
        )

        _store.append_turn(body.session_id, body.message, answer)

        refreshed = _store.get(body.session_id)
        turn_number = refreshed.turn_count if refreshed else MAX_TURNS
        turns_remaining = max(0, MAX_TURNS - turn_number)

        return ChatMessageResponse(
            session_id=body.session_id,
            answer=answer,
            turn_number=turn_number,
            turns_remaining=turns_remaining,
        )

    except (ValueError, FileNotFoundError) as exc:
        return _err(422, "context_corrupted", str(exc), request_id)
    except ModelNotReadyError as exc:
        return _err(503, "model_not_ready", str(exc), request_id)
    except InferenceTimeoutError:
        return _err(504, "inference_timeout",
                    "LLM did not respond in time. Try again later.", request_id)
    finally:
        # Always clear the inference flag — even on timeout or exception
        _store.clear_inference_busy(body.session_id)


# ── DELETE /api/v1/chat/session/{session_id} ──────────────────────────────────

@chat_router.delete("/session/{session_id}", response_model=ChatDeleteResponse)
async def chat_delete(session_id: str, request: Request):
    request_id = uuid4().hex[:12]
    request.state.request_id = request_id

    token = _get_token(request)
    if not token:
        return _err(401, "missing_token", "X-Merchant-Token header is required.", request_id)

    session = _store.get(session_id)
    if session is None:
        return _err(400, "invalid_session",
                    "Session not found or already expired.", request_id)

    # IDOR guard
    if hash_token(token) != session.user_hash:
        return _err(403, "forbidden", "Token does not own this session.", request_id)

    _store.delete(session_id)
    return ChatDeleteResponse(status="deleted")


__all__ = ["chat_router"]
