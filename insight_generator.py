"""
insight_generator.py — Natural Language Translator
==================================================
Reads flags generated across the ML Insight Engine and packages them
into human-understandable spending optimization strings.

Tips are sourced from TIP_CORPUS in config.py — no hardcoded tip text.

Reproducibility:
    All random selections use a seeded random.Random() instance.
    Identical inputs + identical seed → identical outputs.
"""

import logging
import random as _random_module
from typing import List, Optional

import pandas as pd

from config import TIP_CORPUS, INSIGHT_TEMPLATES, lookup_matching_tip_ids
from schema import Col, require_columns

logger = logging.getLogger(__name__)


def _select_tip(
    category: str,
    insight_type: str,
    rng: Optional[_random_module.Random] = None,
) -> str:
    """
    Select a tip text for a (category, insight_type) pair from TIP_CORPUS.

    Uses the shared ``lookup_matching_tip_ids`` helper for the 2-pass
    category-specific → generic lookup, then randomly selects from
    matching tip texts.

    Args:
        category:     Transaction category (e.g. "food").
        insight_type: Insight class (e.g. "spending_spike").
        rng:          Seeded random.Random instance for deterministic selection.
                      If None, creates one with default seed 42.

    Returns:
        Tip text string, or empty string if no match.
    """
    if rng is None:
        rng = _random_module.Random(42)

    tip_ids = lookup_matching_tip_ids(category, insight_type)
    if not tip_ids:
        return ""
    texts = [TIP_CORPUS[tid]["text"] for tid in tip_ids]
    return rng.choice(texts)


def generate_human_insights(
    df: pd.DataFrame,
    top_n: int = 10,
    seed: int = 42,
) -> List[str]:
    """
    Parses a fully contextualized DataFrame to yield simple string
    summaries representing anomalous and recurring transaction signals,
    ranked by the LightGBM Insight Ranker.

    Tips are selected from the curated TIP_CORPUS in config.py.

    Args:
        df:    Fully enriched DataFrame from the pipeline.
        top_n: Maximum number of insight strings to return.
        seed:  Random seed for deterministic tip/template selection.
               Same seed + same data → identical output.
    """
    logger.info("Generating NLP insights...")
    rng = _random_module.Random(seed)

    require_columns(df, Col.insight_generator_input(), "insight_generator")
    
    if Col.INSIGHT_SCORE not in df.columns:
        logger.warning(f"{Col.INSIGHT_SCORE} missing. Defaulting to 0.0.")
        df = df.copy()
        df[Col.INSIGHT_SCORE] = 0.0

    # Needs chronological sort for rolling string contexts
    df = df.sort_values(by=Col.DATE)
    
    candidates = []

    # 1. Discover Subscriptions & Subtotals
    if Col.IS_RECURRING in df.columns:
        recurring_subset = df[df[Col.IS_RECURRING]]
        unique_groups = recurring_subset.groupby(Col.CLEANED_REMARKS)

        for name, group in unique_groups:
            freq = group[Col.RECURRING_FREQUENCY].iloc[-1]
            amt = group[Col.AMOUNT].mean()
            score = float(group[Col.INSIGHT_SCORE].max())

            # Select a template from INSIGHT_TEMPLATES
            templates = INSIGHT_TEMPLATES.get("subscription", [])
            if templates:
                template = rng.choice(templates)
                insight_text = template.format(
                    merchant=name.title(),
                    amount=amt,
                    frequency=freq,
                )
            else:
                insight_text = (
                    f"Subscription identified: '{name.title()}' usually "
                    f"charged {freq} for roughly ₹{amt:.2f}."
                )

            tip = _select_tip("", "subscription", rng=rng)
            candidates.append((score, "subscription", insight_text, tip))

    # 2. Extract Anomalies
    if Col.IS_ANOMALY in df.columns:
        anomalies = df[df[Col.IS_ANOMALY]]
        for _, row in anomalies.iterrows():
            name = row.get(Col.CLEANED_REMARKS, "Unknown Merchant").title()
            cat = row.get(Col.PREDICTED_CATEGORY, "uncategorized")
            amt = row.get(Col.AMOUNT, 0.0)
            pct = row.get(Col.PERCENT_DEVIATION, 0.0) * 100
            score = float(row.get(Col.INSIGHT_SCORE, 0.0))

            date_str = row[Col.DATE].strftime("%d %b %Y") if Col.DATE in row and pd.notna(row[Col.DATE]) else "an unknown date"
            
            # Select a template from INSIGHT_TEMPLATES
            templates = INSIGHT_TEMPLATES.get("spending_spike", [])
            if templates:
                template = rng.choice(templates)
                insight_text = template.format(
                    category=cat,
                    merchant=name,
                    amount=amt,
                    pct=abs(pct),
                    date=date_str
                )
            else:
                insight_text = (
                    f"Unusual {cat} expense detected at '{name}' (₹{amt:.2f}) on {date_str}. "
                    f"This is {pct:.1f}% above your normal expected baseline."
                )

            tip = _select_tip(cat, "spending_spike", rng=rng)
            candidates.append((score, "spike", insight_text, tip))

    # 3. Diversity Ranking logic
    # We want to ensure at least 1 Subscription and 1 Anomaly make it to the top if they exist.
    candidates.sort(key=lambda x: x[0], reverse=True)
    
    top_candidates = []
    seen_types = set()
    
    # Pass 1: Grab the highest scoring insight of EACH type first (to guarantee variety)
    for cand in candidates:
        if cand[1] not in seen_types:
            top_candidates.append(cand)
            seen_types.add(cand[1])
            
    # Pass 2: Fill the rest of the top_n quota with the absolute highest scoring remaining 
    for cand in candidates:
        if len(top_candidates) >= top_n:
            break
        if cand not in top_candidates:
            top_candidates.append(cand)

    # Sort final selection strictly by ML score again
    top_candidates.sort(key=lambda x: x[0], reverse=True)
    
    insights = []
    for score, _type, text, tip in top_candidates:
        insights.append(text)
        if tip:
            insights.append(f"  > Tip: {tip}")

    logger.debug(f"Translated DataFrame into {len(insights)} text insights (from {len(candidates)} candidates).")
    return insights

