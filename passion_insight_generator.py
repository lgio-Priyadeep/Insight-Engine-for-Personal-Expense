import random as _random_module
from schema import Col
from contracts import TIP_CORPUS, INSIGHT_TEMPLATES, lookup_matching_tip_ids
# K1: Removed OPPORTUNITY_CORPUS import
from config_passion import PASSION_INSIGHT_TEMPLATES
from passion_models import PassionSignal
from passion_utils import validate_template_values

# FIX-13: Import from banned_content.py (public API) instead of defining locally.
from banned_content import contains_banned_content as _contains_banned_content

# FIX H11: Use structured logger from logger_factory instead of plain logging.
from logger_factory import get_logger

__all__ = ["generate_passion_insights"]

logger = get_logger(__name__)

# FIX-9: RNG seed 0 is intentional. Output must be deterministic per input.
# Template rotation is not part of this phase.

def _select_tip(category: str, insight_type: str, rng: _random_module.Random,
                subcategory: str = "") -> str:
    # K3: Render tips using lookup_matching_tip_ids.
    # Step 5: When subcategory has its own passion tips, prefer them over parent category tips.
    # This handles cases where category="shopping" but subcategory="education" (e.g. Coursera).
    tip_ids: list[str] = []
    if subcategory:
        try:
            subcat_tip_ids = lookup_matching_tip_ids(subcategory, insight_type)
            if subcat_tip_ids:
                tip_ids = subcat_tip_ids
        except (KeyError, TypeError, IndexError, ValueError):
            pass  # Fall through to category tips

    if not tip_ids:
        try:
            tip_ids = lookup_matching_tip_ids(category, insight_type)
        except (KeyError, TypeError, IndexError, ValueError) as e:
            logger.warning(
                "tip_lookup_failed",
                extra={
                    "category": category,
                    "insight_type": insight_type,
                    "error_type": type(e).__name__,
                },
            )
            return ""
    if not tip_ids:
        return ""
    tip_id = rng.choice(tip_ids)
    tip_data = TIP_CORPUS.get(tip_id, {})
    return tip_data.get("text", "")



def _render_candidate(candidate, rng: _random_module.Random) -> tuple[str, str]:
    # K2: Render Candidate.passion through PASSION_INSIGHT_TEMPLATES["lifestyle_opportunity"]
    if candidate.insight_type == "lifestyle_opportunity":
        merchant_count = getattr(candidate, "merchant_count", 1)
        values = {
            "category": candidate.category,
            "merchant_count": merchant_count,
            # Pre-pluralized: "1 merchant" not "1 merchants".
            "merchant_label": f"{merchant_count} merchant" if merchant_count == 1 else f"{merchant_count} merchants",
            "spend_share": getattr(candidate, "spend_share", 0.0),
            "total_spend": getattr(candidate, "total_spend", 0.0),
            "trend_direction": getattr(candidate, "trend_direction", ""),
            "subcategory": getattr(candidate, "subcategory", ""),  # Step 5c
        }
        templates = PASSION_INSIGHT_TEMPLATES.get("lifestyle_opportunity", ())
        # Step 5e: When subcategory is not resolved, exclude templates that reference
        # {subcategory} to avoid blank renders like "Strong  interest detected under...".
        # Filter is purely cosmetic — RNG sequence is not consumed by filtered templates.
        subcategory_val = getattr(candidate, "subcategory", "")
        if not subcategory_val:
            templates = tuple(t for t in templates if "{subcategory}" not in t) or templates
        fallback_insight = f"High engagement observed in {candidate.category}."

    else:
        values = {
            "merchant": candidate.merchant,
            "amount": candidate.amount,
            "category": candidate.category,
        }
        templates = INSIGHT_TEMPLATES.get(candidate.insight_type, ())
        # Fix #10: Do NOT expose candidate.merchant in fallback — raw merchant strings
        # can leak PII into crash dumps and logs before HMAC masking is applied upstream.
        fallback_insight = f"{candidate.insight_type}: merchant unavailable (Rs.{candidate.amount:.0f})"

    validate_template_values(values)
    try:
        tip_template = _select_tip(
            candidate.category, candidate.insight_type, rng,
            subcategory=getattr(candidate, "subcategory", ""),
        )
        tip = tip_template.format(**values) if tip_template else ""
    except (KeyError, ValueError, TypeError, IndexError) as e:
        logger.warning(
            "tip_render_failed",
            extra={
                "insight_type": getattr(candidate, 'insight_type', 'unknown'),
                "error_type": type(e).__name__,
            },
        )
        tip = ""

    if not templates:
        insight = fallback_insight
    else:
        try:
            insight = rng.choice(templates).format(**values)
        except (KeyError, ValueError, TypeError, IndexError) as e:
            # K6: Log template key only
            logger.warning(
                "template_render_failed",
                extra={
                    "insight_type": candidate.insight_type,
                    "error_type": type(e).__name__,
                },
            )
            insight = fallback_insight

    return insight, tip


def generate_passion_insights(
    candidates: list,
    top_n: int = 3,
    rng: _random_module.Random | None = None,
    strict_mode: bool = True,
    fallback_insights: list[str] | None = None,
) -> list[str]:
    # K7: Require caller to pass rng
    if rng is None:
        raise ValueError("rng must be provided for deterministic insights")

    results: list[str] = []
    seen_texts: set[str] = set()

    for c in candidates:
        if len(results) >= top_n:
            break
        # P1.4: Narrowed exception types for _render_candidate.
        try:
            insight, tip = _render_candidate(c, rng)
        except (KeyError, TypeError, ValueError, IndexError) as e:
            if strict_mode:
                raise
            logger.warning("render_candidate_failed", extra={"error": str(e)})
            continue

        combined = f"{insight} {tip}".strip() if tip else insight

        if _contains_banned_content(combined):
            logger.warning(
                "banned_content_filtered",
                extra={"insight_type": getattr(c, 'insight_type', 'unknown')},
            )
            continue

        normalized = combined.strip().lower()
        if normalized in seen_texts:
            continue
        seen_texts.add(normalized)
        results.append(combined)

    # FIX #11: Fallback insights deduped and filtered.
    if not results and fallback_insights:
        for fb in fallback_insights:
            if len(results) >= top_n:
                break
            norm = fb.strip().lower()
            if not _contains_banned_content(fb) and norm not in seen_texts:
                seen_texts.add(norm)
                results.append(fb)

    return results
