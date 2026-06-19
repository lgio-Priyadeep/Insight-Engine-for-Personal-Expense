"""
api.py — Insight Engine REST API
=================================
Stack-agnostic FastAPI application.
Exposes the full pipeline as a single POST endpoint.

Mandatory API Constraints (feature_split_design.md § Stack-Agnostic Contract):
  1. CORS — wildcard in dev, env-configured whitelist in production.
  2. Content-Type — accepts multipart/form-data (CSV) and application/json.
  3. Stateless — no session state, no cookies.
  4. Auth — X-Merchant-Token header (validated in endpoint, not middleware).
  5. Flat error envelope — { error, message, request_id } for all 4xx/5xx.
  6. No framework wrappers — plain JSON only.
  7. Versioned path — /api/v1/...
  8. No passion toggle in request — controlled by env-var only.
"""

from __future__ import annotations

import io
import logging
import os
from uuid import uuid4

import pandas as pd
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api_schemas import AnalyzeResponse, ErrorResponse

logger = logging.getLogger(__name__)

# ── App ───────────────────────────────────────────────────────────────────────

from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):
    """B1 fix: stop session sweep thread cleanly on server shutdown."""
    yield
    from chatbot_session_store import SessionStore
    SessionStore()._stop_event.set()


app = FastAPI(
    title="Insight Engine API",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url=None,
    openapi_url="/api/openapi.json",
    lifespan=lifespan,
)

# ── CORS middleware ───────────────────────────────────────────────────────────
# Origin list read from env at startup.
# Production: set API_CORS_ORIGINS="https://yoursite.com,https://app.yoursite.com"
# Dev / unset: defaults to wildcard ["*"]

_raw_origins = os.environ.get("API_CORS_ORIGINS", "*")
_cors_origins: list[str] = (
    [o.strip() for o in _raw_origins.split(",") if o.strip()]
    if _raw_origins != "*"
    else ["*"]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=False,
    allow_methods=["POST", "GET", "DELETE", "OPTIONS"],
    allow_headers=["X-Merchant-Token", "Content-Type"],
    expose_headers=["X-Request-Id"],
)


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["ops"], summary="Liveness probe")
def health():
    """Returns 200 OK when the server is accepting requests."""
    return {"status": "ok"}


# ── Exception handlers (uniform error envelope) ───────────────────────────────
# Full handlers added in CP-07. Stub here so skeleton imports cleanly.

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    request_id = getattr(request.state, "request_id", "unknown")
    logger.error(
        "unhandled_exception",
        extra={"error_type": type(exc).__name__, "request_id": request_id},
        exc_info=True,
    )
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="internal_error",
            message="An unexpected error occurred.",
            request_id=request_id,
        ).model_dump(),
    )


from fastapi import HTTPException as _HTTPException
from fastapi.exceptions import RequestValidationError


@app.exception_handler(_HTTPException)
async def http_exception_handler(request: Request, exc: _HTTPException):
    """Pass through structured HTTPException detail as-is when it is already
    an ErrorResponse dict; otherwise wrap it."""
    request_id = getattr(request.state, "request_id", "unknown")
    detail = exc.detail
    if isinstance(detail, dict) and "error" in detail:
        # Already an ErrorResponse payload from the endpoint handler
        return JSONResponse(status_code=exc.status_code, content=detail)
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error="http_error",
            message=str(detail),
            request_id=request_id,
        ).model_dump(),
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    request_id = getattr(request.state, "request_id", "unknown")
    return JSONResponse(
        status_code=422,
        content=ErrorResponse(
            error="validation_error",
            message=str(exc.errors()),
            request_id=request_id,
        ).model_dump(),
    )

# ── Chat router ───────────────────────────────────────────────────────────────
from chat_router import chat_router
app.include_router(chat_router)

# ── Endpoint placeholder ──────────────────────────────────────────────────────
# Full implementation added in CP-06.

@app.post(
    "/api/v1/insights/analyze",
    response_model=AnalyzeResponse,
    tags=["insights"],
    summary="Analyze a bank statement and return spend + passion insights",
)
async def analyze(
    request: Request,
    file: UploadFile = File(..., description="CSV bank statement"),
):
    """
    Accepts a CSV bank statement upload and returns unified spend + passion insights.

    Auth: X-Merchant-Token header checked (non-empty). No token validation logic
    in v1 — presence enforced to establish the header contract for future use.

    run_id is generated at handler entry (uuid4().hex[:12]).
    It is NOT read from pipeline_run_id_ctx — that ContextVar is reset
    at pipeline.py:814 before this handler receives the PipelineResult.
    """
    # ── 1. Generate request_id at entry ─────────────────────────────────────
    request_id = uuid4().hex[:12]
    request.state.request_id = request_id   # available to exception handlers

    # ── 2. Auth check (header presence) ─────────────────────────────────────
    token = request.headers.get("X-Merchant-Token", "").strip()
    if not token:
        raise HTTPException(
            status_code=401,
            detail=ErrorResponse(
                error="missing_token",
                message="X-Merchant-Token header is required.",
                request_id=request_id,
            ).model_dump(),
        )

    # ── 3. Parse CSV upload ──────────────────────────────────────────────────
    if file.content_type not in ("text/csv", "application/octet-stream", "text/plain"):
        raise HTTPException(
            status_code=422,
            detail=ErrorResponse(
                error="invalid_content_type",
                message=f"Expected CSV file, got '{file.content_type}'.",
                request_id=request_id,
            ).model_dump(),
        )
    raw_bytes = await file.read()
    try:
        raw_df = pd.read_csv(io.BytesIO(raw_bytes))
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=ErrorResponse(
                error="csv_parse_error",
                message=f"Could not parse CSV: {exc}",
                request_id=request_id,
            ).model_dump(),
        )

    if raw_df.empty:
        raise HTTPException(
            status_code=422,
            detail=ErrorResponse(
                error="empty_csv",
                message="Uploaded CSV contains no rows.",
                request_id=request_id,
            ).model_dump(),
        )

    # ── 4. Run pipeline ──────────────────────────────────────────────────────
    from pipeline import run_pipeline
    try:
        result = run_pipeline(raw_df)
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=ErrorResponse(
                error="pipeline_validation_error",
                message=str(exc),
                request_id=request_id,
            ).model_dump(),
        )

    # ── 4b. Export LLM context per-user (chatbot auto-latest) ──────────────
    from chatbot_context_loader import hash_token
    from llm_export import export_to_json
    _user_hash = hash_token(token)
    try:
        export_to_json(result, user_id=_user_hash, run_id=request_id)
        logger.debug("llm_context_exported",
                     extra={"user_hash": _user_hash, "run_id": request_id})
    except Exception as _exc:
        logger.warning("llm_context_export_failed", extra={"error": str(_exc)})
    # Non-fatal — AnalyzeResponse is still returned on export failure.
    # ────────────────────────────────────────────────────────────────────────

    # ── 5. Extract passion_status (C1a) ──────────────────────────────────────
    # passion_status lives in result.stats — not exposed by build_llm_context.
    passion_status: str = (getattr(result, "stats", {}) or {}).get(
        "passion_status", "disabled"
    )

    # ── 6. Build spend_insights (C2/C6) ─────────────────────────────────────
    # Source: _generate_insight_records, NOT result.insights.
    # result.insights merges personal-transfer strings (pipeline.py:716).
    from schema import Col
    from insight_generator import _generate_insight_records
    from llm_export_aggregators import build_spend_insights

    spend_debits = result.debits
    if Col.IS_KNOWN_PERSON in spend_debits.columns:
        spend_debits = spend_debits[~spend_debits[Col.IS_KNOWN_PERSON].fillna(False)]

    insight_records = _generate_insight_records(spend_debits) if not spend_debits.empty else []
    spend_section = build_spend_insights(
        insight_records=insight_records,
        exclusion_stats=getattr(result, "exclusion_stats", {}) or {},
    )

    # ── 7. Build passion_insights (C1b) ─────────────────────────────────────
    pii_safe = True   # API responses are always PII-safe
    from llm_export_passion import build_passion_context
    passion_section = build_passion_context(result, pii_safe=pii_safe)

    # ── 8. Assemble and return ───────────────────────────────────────────────
    return AnalyzeResponse(
        run_id=request_id,
        passion_status=passion_status,
        spend_insights=spend_section,
        passion_insights=passion_section,
    )
