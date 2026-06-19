"""
api_schemas.py — Pydantic models for Insight Engine API
========================================================
Defines the exact wire shapes for:
  - POST /api/v1/insights/analyze  (request + response)
  - Uniform error envelope (all 4xx / 5xx)

Kept separate from api.py to allow schema import without
importing the FastAPI app (useful in tests and CLI tooling).
"""

from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel, Field


# ── Spend Insights ────────────────────────────────────────────────────────────

class InsightRecord(BaseModel):
    """One paired insight: observation text + optional tip."""
    text:  str
    tip:   str          # "" when no tip — always present, never omitted
    type:  str          # "spending_spike" | "subscription"
    score: float


class SpendInsightsStats(BaseModel):
    total_transactions:   int
    excluded_transactions: int
    exclusion_rate:       float


class SpendInsights(BaseModel):
    insights: List[InsightRecord]
    stats:    SpendInsightsStats


# ── Passion Insights ──────────────────────────────────────────────────────────

class PassionSignalOut(BaseModel):
    category:          str
    subcategory:       str
    display_label:     str
    total_spend:       float
    spend_share_pct:   float
    merchant_count:    int
    active_months:     int
    trend:             str
    is_suppressed:     bool
    suppression_reason: str
    narrative:         str


class PassionInsights(BaseModel):
    enabled:      bool
    signal_count: int
    signals:      List[PassionSignalOut]


# ── Top-level response ────────────────────────────────────────────────────────

class AnalyzeResponse(BaseModel):
    """
    Unified response for POST /api/v1/insights/analyze.

    run_id:         Request-scoped UUID (uuid4().hex[:12]) — NOT pipeline_run_id_ctx.
                    pipeline_run_id_ctx is reset at pipeline.py:814 before the
                    handler receives the result. run_id is generated at handler
                    entry and serves as both response run_id and error request_id.
    passion_status: Extracted from result.stats.get("passion_status", "disabled").
                    Values: success | disabled | skipped | timeout | failure | missing_fields
    """
    run_id:          str
    passion_status:  str
    spend_insights:  SpendInsights
    passion_insights: PassionInsights


# ── Chatbot Schemas ───────────────────────────────────────────────────────────

class ChatStartRequest(BaseModel):
    run_id: str | None = None


class ChatStartResponse(BaseModel):
    session_id:      str
    run_id:          str
    period:          dict
    turns_remaining: int


class ChatMessageRequest(BaseModel):
    session_id: str
    message:    str = Field(...)  # 512-char limit enforced in handler (→ message_too_long)


class ChatMessageResponse(BaseModel):
    session_id:      str
    answer:          str
    turn_number:     int
    turns_remaining: int


class ChatDeleteResponse(BaseModel):
    status: Literal["deleted"]


# ── Error envelope ────────────────────────────────────────────────────────────

class ErrorResponse(BaseModel):
    """
    Uniform error shape for all 4xx and 5xx responses.
    Mandatory API Constraint #5 from feature_split_design.md.
    """
    error:      str   # machine-readable error code
    message:    str   # human-readable detail
    request_id: str   # same value as run_id for the in-flight request
