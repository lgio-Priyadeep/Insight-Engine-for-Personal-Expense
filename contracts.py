"""
contracts.py -- Facade for external dependencies.
Wraps config.py objects into immutable MappingProxyType at import time.

FIX: Always normalize regardless of source shape. If config.TIP_CORPUS is
already a MappingProxyType with plain dict inner values, the bypass would
skip re-freezing, and _validate_tip_corpus raises confusingly.
"""

from types import MappingProxyType

_GENERIC_TIP_PREFIX = "tip_generic_"

def _is_generic_tip_id(tip_id: str) -> bool:
    return str(tip_id).startswith(_GENERIC_TIP_PREFIX)

# B4: Col is NOT re-exported. All callers must import Col directly from schema.
from config import INSIGHT_TEMPLATES as _IT, TIP_CORPUS as _TC

# C3: Freezer rules
def _freeze_insight_templates(raw: dict | MappingProxyType) -> MappingProxyType:
    if not isinstance(raw, (dict, MappingProxyType)):
        raise TypeError("INSIGHT_TEMPLATES must be dict or MappingProxyType")
    result = {}
    for k, v in raw.items():
        if not isinstance(v, (list, tuple)):
            raise TypeError(f"INSIGHT_TEMPLATES[{k}] must be a list or tuple of strings")
        result[k] = tuple(v)
    return MappingProxyType(result)

def _freeze_tip_corpus(raw: dict | MappingProxyType) -> MappingProxyType:
    # Fix 13: _freeze_tip_corpus now owns ALL business rule validation.
    # Error order: required keys → type checks → content rules → freeze.
    _ALLOWED_INSIGHTS: frozenset[str] = frozenset({
        "subscription", "spending_spike", "lifestyle_opportunity",
        "trend_warning", "budget_risk", "no_action", "any",
    })
    if not isinstance(raw, (dict, MappingProxyType)):
        raise TypeError(f"TIP_CORPUS must be dict/MappingProxyType, got {type(raw)}")
    result = {}
    for k, v in raw.items():
        if not isinstance(v, (dict, MappingProxyType)):
            raise TypeError(
                f"Config Migration Error: TIP_CORPUS[{k!r}] must be a dict containing "
                f"'text', 'categories', and 'insights'. Got {type(v)}. "
                "Please update config.TIP_CORPUS to the new schema."
            )
        # Fix 13: Required key check BEFORE empty/content checks.
        for req in ("text", "categories", "insights"):
            if req not in v:
                raise ValueError(
                    f"Config Migration Error: TIP_CORPUS[{k!r}] is missing required key {req!r}. "
                    "Ensure config.TIP_CORPUS matches the new schema."
                )
        # Fix 13: text must be str.
        if not isinstance(v["text"], str):
            raise TypeError(f"TIP_CORPUS[{k!r}]['text'] must be str, got {type(v['text'])}")
        # Fix 13: categories and insights must be list/tuple of str.
        for seq_key in ("categories", "insights"):
            seq = v[seq_key]
            if not isinstance(seq, (list, tuple)):
                raise TypeError(
                    f"TIP_CORPUS[{k!r}][{seq_key!r}] must be list or tuple, got {type(seq)}"
                )
            bad = [x for x in seq if not isinstance(x, str)]
            if bad:
                raise TypeError(
                    f"TIP_CORPUS[{k!r}][{seq_key!r}] elements must be str, got {[type(x).__name__ for x in bad]}"
                )
        is_generic = _is_generic_tip_id(k)
        cats = tuple(c.strip().lower() for c in v["categories"])
        insights_tup = tuple(i.strip().lower() for i in v["insights"])
        # Fix 13: Validate lowercase-stripped values.
        for cat in cats:
            if not cat:
                raise ValueError(f"TIP_CORPUS[{k!r}] 'categories' contains blank string after strip")
        for ins in insights_tup:
            if not ins:
                raise ValueError(f"TIP_CORPUS[{k!r}] 'insights' contains blank string after strip")
            if ins not in _ALLOWED_INSIGHTS:
                raise ValueError(
                    f"TIP_CORPUS[{k!r}] insight {ins!r} not in allowed set {sorted(_ALLOWED_INSIGHTS)}"
                )
        # Fix 13: Non-generic tips cannot have empty categories or insights.
        # B5: Empty-tuple wildcard is ONLY permitted for 'tip_generic_*' tips.
        if not is_generic:
            if not cats:
                raise ValueError(
                    f"TIP_CORPUS[{k!r}] has empty 'categories' but its tip_id does not start "
                    f"with 'tip_generic_'. Only 'tip_generic_*' tips may use empty-tuple wildcard behavior."
                )
            if not insights_tup:
                raise ValueError(
                    f"TIP_CORPUS[{k!r}] has empty 'insights' but its tip_id does not start "
                    f"with 'tip_generic_'. Only 'tip_generic_*' tips may use empty-tuple wildcard behavior."
                )
            # Fix 13: Non-generic tips cannot contain "any" wildcard.
            if "any" in cats:
                raise ValueError(
                    f"TIP_CORPUS[{k!r}] is a non-generic tip but contains wildcard 'any' in 'categories'."
                )
            if "any" in insights_tup:
                raise ValueError(
                    f"TIP_CORPUS[{k!r}] is a non-generic tip but contains wildcard 'any' in 'insights'."
                )
        frozen_inner: dict = {"text": v["text"], "categories": cats, "insights": insights_tup}
        # Preserve any extra scalar fields (future-proof).
        for kk, vv in v.items():
            if kk in frozen_inner:
                continue
            if isinstance(vv, (str, int, float, bool)):
                frozen_inner[kk] = vv
            else:
                raise TypeError(f"Invalid type in TIP_CORPUS[{k!r}][{kk!r}]: {type(vv)}")
        result[k] = MappingProxyType(frozen_inner)
    return MappingProxyType(result)

INSIGHT_TEMPLATES = _freeze_insight_templates(_IT)
TIP_CORPUS = _freeze_tip_corpus(_TC)

# C2: Define lookup_matching_tip_ids
def lookup_matching_tip_ids(category: str, insight_type: str) -> list[str]:
    cat_norm = str(category).strip().lower()
    type_norm = str(insight_type).strip().lower()
    matches = []
    for tip_id, tip_data in TIP_CORPUS.items():
        cats = tip_data.get("categories", ())
        types = tip_data.get("insights", ())

        # B5: Empty-tuple wildcard is ONLY honoured for "tip_generic_*" tips.
        # Non-generic tips must have explicit category/insight lists (enforced
        # at startup by _freeze_tip_corpus) so the wildcard path here is
        # purely a runtime safety valve for generic tips that did pass validation.
        is_generic = _is_generic_tip_id(tip_id)
        cat_match = (cat_norm in cats) or ("any" in cats) or (is_generic and len(cats) == 0)
        type_match = (type_norm in types) or ("any" in types) or (is_generic and len(types) == 0)

        if cat_match and type_match:
            matches.append(tip_id)
    return matches


# B4: Col removed from __all__. Import Col from schema directly.
__all__ = ["INSIGHT_TEMPLATES", "TIP_CORPUS", "lookup_matching_tip_ids"]
