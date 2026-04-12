"""
seed_labeler.py — Keyword-Based Pseudo-Label Generation
=========================================================
Converts cleaned remarks into pseudo-labels using the keyword
dictionaries defined in config.py.

Design decisions:
  - Multi-word keywords are matched as exact phrases via precompiled \b regex boundaries.
  - Priority order relies on semantic tier mappings.
  - Generates detailed metadata rows: confidence, reasons, exact matching strings.
"""

import logging
import re
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from config import (
    CATEGORY_PRIORITY,
    CATEGORY_KEYWORDS,
    CREDIT_PRIORITY,
    CREDIT_KEYWORDS,
    FALLBACK_DEBIT_LABEL,
    FALLBACK_CREDIT_LABEL,
    MIN_COVERAGE_THRESHOLD,
    TIER_MAPPING,
)
from preprocessor import normalize
from schema import Col, require_columns

logger = logging.getLogger(__name__)


@dataclass
class CompiledKeyword:
    text: str
    norm: str
    pattern: re.Pattern
    category: str
    tier_name: str
    priority: int
    confidence: float


# ── Core Matching Logic ───────────────────────────────────────────────────────

def _compile_keywords(keyword_map: dict[str, list[str]], is_credit: bool = False) -> list[CompiledKeyword]:
    compiled = []
    for category, keywords in keyword_map.items():
        if not is_credit and category in TIER_MAPPING:
            meta = TIER_MAPPING[category]
        else:
            meta = {"tier_name": "unknown", "priority": 999, "confidence": 0.5}

        for kw in keywords:
            norm_kw = normalize(kw)
            if not norm_kw:
                continue
            # Note: We stripped special characters entirely during normalize (no @, &) so \b is safe.
            pattern = re.compile(rf"\b{re.escape(norm_kw)}\b")
            compiled.append(CompiledKeyword(
                text=kw,
                norm=norm_kw,
                pattern=pattern,
                category=category,
                tier_name=meta["tier_name"],
                priority=meta["priority"],
                confidence=meta["confidence"]
            ))
    return compiled


def _match_remark(
    norm_remark: str,
    compiled_keywords: list[CompiledKeyword],
    fallback: str,
) -> tuple[str, str, str, str, float]:
    """
    Match against the precompiled keywords.
    Returns: (label, reason, keyword_triggered, keyword_norm, confidence)
    """
    if not norm_remark or not norm_remark.strip():
        return fallback, f"fallback_{fallback}", "", "", 0.0

    matches = []
    for kw in compiled_keywords:
        if kw.pattern.search(norm_remark):
            matches.append(kw)

    if not matches:
        return fallback, f"fallback_{fallback}", "", "", 0.0

    best_tier = min(m.priority for m in matches)
    tier_matches = [m for m in matches if m.priority == best_tier]
    tier_matches.sort(key=lambda x: (-len(x.norm), x.norm))
    best_match = tier_matches[0]

    return (
        best_match.category,
        f"keyword_match_{best_match.tier_name}",
        best_match.text,
        best_match.norm,
        best_match.confidence
    )


# ── Module-Level Precompiled Keywords (default maps only) ─────────────────────
_DEFAULT_DEBIT_KWS = _compile_keywords(CATEGORY_KEYWORDS, is_credit=False)
_DEFAULT_CREDIT_KWS = _compile_keywords(CREDIT_KEYWORDS, is_credit=True)


# ── Coverage Logging ──────────────────────────────────────────────────────────

def _log_coverage(
    df: pd.DataFrame,
    label_col: str,
    fallback_labels: Optional[set[str]] = None,
    context: str = "",
    min_coverage_threshold: float = MIN_COVERAGE_THRESHOLD,
) -> float:
    if fallback_labels is None:
        fallback_labels = {FALLBACK_DEBIT_LABEL}

    total   = len(df)
    labeled = df[~df[label_col].isin(fallback_labels)].shape[0]
    coverage = labeled / total if total > 0 else 0.0

    tag = f"[{context}] " if context else ""
    logger.info(
        f"{tag}Labeling coverage: {labeled}/{total} ({coverage * 100:.1f}%)"
    )
    if coverage < min_coverage_threshold:
        logger.warning(
            f"{tag}⚠️  Coverage is {coverage * 100:.1f}% — below the "
            f"{min_coverage_threshold * 100:.0f}% threshold. "
            "Extend keyword dictionaries in config.py to improve model quality."
        )
    return coverage


# ── Public API ────────────────────────────────────────────────────────────────

def label_debits(
    df: pd.DataFrame,
    remark_col: str = Col.CLEANED_REMARKS,
    label_col: str = Col.PSEUDO_LABEL,
    keyword_map: Optional[dict[str, list[str]]] = None,
) -> pd.DataFrame:
    if keyword_map is None:
        keyword_map = CATEGORY_KEYWORDS

    require_columns(df, Col.seed_labeler_input(), "seed_labeler.label_debits")

    compiled_kws = _DEFAULT_DEBIT_KWS if keyword_map is CATEGORY_KEYWORDS else _compile_keywords(keyword_map, is_credit=False)
    
    df = df.copy()
    # Intentional re-normalization: cleaned_remarks was already cleaned by
    # preprocessor.clean_remark(), but we normalize again here as a boundary
    # hardening measure. If this function is ever called without the
    # preprocessor (e.g., direct API use), this guarantees safe matching.
    norm_series = df[remark_col].apply(normalize)
    
    matches = norm_series.apply(
        lambda r: _match_remark(r, compiled_kws, FALLBACK_DEBIT_LABEL)
    )
    
    df[label_col] = matches.apply(lambda x: x[0])
    df[Col.LABEL_REASON] = matches.apply(lambda x: x[1])
    df[Col.LABEL_KEYWORD] = matches.apply(lambda x: x[2])
    df[Col.LABEL_KEYWORD_NORM] = matches.apply(lambda x: x[3])
    df[Col.LABEL_CONFIDENCE] = matches.apply(lambda x: x[4])

    _log_coverage(
        df, label_col,
        fallback_labels={FALLBACK_DEBIT_LABEL},
        context="debits",
    )
    return df


def label_credits(
    df: pd.DataFrame,
    remark_col: str = Col.CLEANED_REMARKS,
    label_col: str = Col.PSEUDO_LABEL,
    keyword_map: Optional[dict[str, list[str]]] = None,
) -> pd.DataFrame:
    if keyword_map is None:
        keyword_map = CREDIT_KEYWORDS

    require_columns(df, Col.seed_labeler_input(), "seed_labeler.label_credits")

    compiled_kws = _DEFAULT_CREDIT_KWS if keyword_map is CREDIT_KEYWORDS else _compile_keywords(keyword_map, is_credit=True)
    
    df = df.copy()
    norm_series = df[remark_col].apply(normalize)
    
    matches = norm_series.apply(
        lambda r: _match_remark(r, compiled_kws, FALLBACK_CREDIT_LABEL)
    )
    
    df[label_col] = matches.apply(lambda x: x[0])
    df[Col.LABEL_REASON] = matches.apply(lambda x: x[1])
    df[Col.LABEL_KEYWORD] = matches.apply(lambda x: x[2])
    df[Col.LABEL_KEYWORD_NORM] = matches.apply(lambda x: x[3])
    df[Col.LABEL_CONFIDENCE] = matches.apply(lambda x: x[4])

    _log_coverage(
        df, label_col,
        fallback_labels={FALLBACK_CREDIT_LABEL, FALLBACK_DEBIT_LABEL},
        context="credits",
    )
    return df
