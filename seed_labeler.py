"""
seed_labeler.py — Keyword-Based Pseudo-Label Generation
=========================================================
Converts cleaned remarks into pseudo-labels using the keyword
dictionaries defined in config.py.

Design decisions:
  - Priority order is fully user-controlled via config.CATEGORY_PRIORITY.
  - Multi-word keywords (e.g. "cash withdrawal") are matched as phrases.
  - Debits and credits use separate keyword maps and fallback labels.
  - Coverage metric is logged; a warning fires if < MIN_COVERAGE_THRESHOLD.
  - The input DataFrame is never mutated (defensive .copy() throughout).

These pseudo-labels feed the categorization model in Phase 2.
Low-confidence rows (uncategorized / other_credit) are handled by
the model's 'balanced' class weighting — they are NOT dropped.
"""

import logging
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
)
from schema import Col, require_columns

logger = logging.getLogger(__name__)


# ── Core Matching Logic ───────────────────────────────────────────────────────

def _match_remark(
    cleaned_remark: str,
    keyword_map: dict[str, list[str]],
    priority_order: list[str],
    fallback: str,
) -> tuple[str, list[str]]:
    """
    Match a single cleaned remark against the keyword map,
    respecting the caller-supplied priority order.

    Matching strategy:
        - Remark is tokenized into a set of words.
        - Each keyword (possibly multi-word) is checked by verifying
          ALL of its tokens appear in the remark token set.
        - First category in priority_order whose any keyword matches → wins.
        - No match → fallback label.

    Args:
        cleaned_remark: Pre-processed remark string.
        keyword_map:    Dict of {category: [keywords]}.
        priority_order: Ordered list of categories to evaluate.
        fallback:       Label to assign if no keyword matches.

    Returns:
        (Matched category string or fallback, List of all matched categories)
    """
    if not cleaned_remark or not cleaned_remark.strip():
        return fallback, []

    remark_tokens = set(cleaned_remark.split())

    matches = []
    top_match = None

    for category in priority_order:
        keywords = keyword_map.get(category, [])
        for kw in keywords:
            kw_tokens = kw.split()          # support multi-word keywords
            if all(t in remark_tokens for t in kw_tokens):
                matches.append(category)
                if top_match is None:
                    top_match = category
                break

    if top_match is None:
        return fallback, []

    return top_match, matches


# ── Coverage Logging ──────────────────────────────────────────────────────────

def _log_coverage(
    df: pd.DataFrame,
    label_col: str,
    fallback_labels: Optional[set[str]] = None,
    context: str = "",
    min_coverage_threshold: float = MIN_COVERAGE_THRESHOLD,
) -> float:
    """
    Log labeling coverage and emit a warning if below threshold.

    Args:
        df:              Labeled DataFrame.
        label_col:       Column containing pseudo-labels.
        fallback_labels: Set of labels considered "unlabeled".
        context:         String tag for log messages (e.g. "debits").
        min_coverage_threshold: Dynamic coverage threshold to trigger warnings.

    Returns:
        Coverage ratio (0.0 – 1.0).
    """
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
    priority_order: Optional[list[str]] = None,
    keyword_map: Optional[dict[str, list[str]]] = None,
) -> pd.DataFrame:
    """
    Assign pseudo-labels to debit transactions using keyword rules.

    Args:
        df:             Debit DataFrame (output of preprocess()[0]).
        remark_col:     Column containing cleaned remark text.
        label_col:      Name of the new label column to create.
        priority_order: Override config.CATEGORY_PRIORITY if needed.
        keyword_map:    Override config.CATEGORY_KEYWORDS if needed.

    Returns:
        Copy of df with 'pseudo_label' column added.
        Unmatched rows receive FALLBACK_DEBIT_LABEL ('uncategorized').
    """
    if priority_order is None:
        priority_order = CATEGORY_PRIORITY
    if keyword_map is None:
        keyword_map = CATEGORY_KEYWORDS

    # Validate no unknown categories in priority list
    unknown = set(priority_order) - set(keyword_map.keys())
    if unknown:
        logger.warning(
            f"Priority list contains categories with no keywords: {unknown}. "
            "They will never match."
        )

    require_columns(df, Col.seed_labeler_input(), "seed_labeler.label_debits")

    df = df.copy()
    matches = df[remark_col].apply(
        lambda r: _match_remark(r, keyword_map, priority_order, FALLBACK_DEBIT_LABEL)
    )
    df[label_col] = matches.apply(lambda x: x[0])
    df["all_matched_categories"] = matches.apply(lambda x: x[1])

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
    priority_order: Optional[list[str]] = None,
    keyword_map: Optional[dict[str, list[str]]] = None,
) -> pd.DataFrame:
    """
    Assign pseudo-labels to credit transactions using credit-specific rules.

    Credits use a separate keyword map (CREDIT_KEYWORDS) and fall back to
    'other_credit' rather than 'uncategorized'.

    Args:
        df:             Credit DataFrame (output of preprocess()[1]).
        remark_col:     Column containing cleaned remark text.
        label_col:      Name of the new label column to create.
        priority_order: Override config.CREDIT_PRIORITY if needed.
        keyword_map:    Override config.CREDIT_KEYWORDS if needed.

    Returns:
        Copy of df with 'pseudo_label' column.
        Unmatched rows receive FALLBACK_CREDIT_LABEL ('other_credit').
    """
    if priority_order is None:
        priority_order = CREDIT_PRIORITY
    if keyword_map is None:
        keyword_map = CREDIT_KEYWORDS

    require_columns(df, Col.seed_labeler_input(), "seed_labeler.label_credits")

    df = df.copy()
    matches = df[remark_col].apply(
        lambda r: _match_remark(r, keyword_map, priority_order, FALLBACK_CREDIT_LABEL)
    )
    df[label_col] = matches.apply(lambda x: x[0])
    df["all_matched_categories"] = matches.apply(lambda x: x[1])

    _log_coverage(
        df, label_col,
        fallback_labels={FALLBACK_CREDIT_LABEL, FALLBACK_DEBIT_LABEL},
        context="credits",
    )
    return df
