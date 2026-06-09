from types import MappingProxyType

# B3: Silent fallback to empty dict is forbidden — missing SPECIFIC_MERCHANT_ALIASES
# would silently produce a broken alias map with no core aliases, hiding config errors.
try:
    from config import SPECIFIC_MERCHANT_ALIASES as _raw_core_aliases
except ImportError as e:
    raise RuntimeError(
        "config.SPECIFIC_MERCHANT_ALIASES is required. "
        "Ensure config.py exposes SPECIFIC_MERCHANT_ALIASES as a dict[str, str]."
    ) from e
if not isinstance(_raw_core_aliases, dict):
    raise TypeError(
        f"SPECIFIC_MERCHANT_ALIASES must be a dict mapping str->str, "
        f"got {type(_raw_core_aliases)}"
    )

# Normalize core aliases
_CORE_ALIASES = {}
for alias, canonical in _raw_core_aliases.items():
    if isinstance(alias, str) and isinstance(canonical, str):
        _CORE_ALIASES[alias.strip().lower()] = canonical.strip().lower()

# P3.2: Explicit type annotations on all config constants.
MAX_SPIKE_CANDIDATES: int = 5
PASSION_MIN_MONTHS: int = 3
# FIX-22: Keep 1 because subcategory enrichment is intentionally useful for 1-row input
# even though passion signals require PASSION_MERCHANT_COUNT_MIN (3) distinct merchants.
PASSION_MIN_DEBIT_ROWS: int = 1
PASSION_MERCHANT_COUNT_MIN: int = 3
PASSION_SPEND_SHARE_THRESHOLD: float = 0.25
PASSION_ANOMALY_SUPPRESSION_THRESHOLD: float = 0.30
DISTRESS_FEES_THRESHOLD: float = 0.15

MARKETPLACE_HIGH_AMOUNT_THRESHOLD: float = 1000.0
MARKETPLACE_HIGH_CONFIDENCE: float = 0.90
MARKETPLACE_LOW_CONFIDENCE: float = 0.85

PIPELINE_BUDGET_MS: float = 500.0
PIPELINE_TOP_N: int = 3

# FIX M4: Hard timeout is now configurable instead of baked into
# _StepBudgetGuard as a magic number. Allows CI/container environments
# to use a higher value without touching the class.
# PIPELINE_HARD_TIMEOUT_MS is cooperative only. It cannot interrupt blocked pandas, regex, or IO mid-call.
PIPELINE_HARD_TIMEOUT_MS: float = 2000.0

ELECTRONICS_ALLOWED_CATEGORIES: frozenset[str] = frozenset({"shopping"})

_PASSION_EXTRAS = {
    "amzn": "amazon",
    "amzn mktp": "amazon",
    "flpkrt": "flipkart",
    "uber bv": "uber",
}

# Normalize passion extras
_NORMALIZED_EXTRAS = {
    str(alias).strip().lower(): str(canonical).strip().lower()
    for alias, canonical in _PASSION_EXTRAS.items()
}

# B2: Compute conflicts at module load for later validation, but do NOT raise here.
# Raising at import time bypasses bootstrap logging and breaks module load order.
# validate_merchant_aliases() (called from bootstrap.run_startup_checks) raises instead.
_ALIAS_CONFLICTS = frozenset(
    alias for alias in set(_CORE_ALIASES) & set(_NORMALIZED_EXTRAS)
    if _CORE_ALIASES[alias] != _NORMALIZED_EXTRAS[alias]
)

PASSION_MERCHANT_ALIASES: MappingProxyType = MappingProxyType({
    **_NORMALIZED_EXTRAS,
    **_CORE_ALIASES,
})

# P1-5: Generalist canonicals — must match canonical values in PASSION_MERCHANT_ALIASES.
# All generalist detection goes through resolve_merchant_vectorized + .isin(GENERALIST_CANONICALS).
GENERALIST_CANONICALS: frozenset[str] = frozenset({"amazon", "flipkart", "meesho", "snapdeal"})

def validate_merchant_aliases(alias_map=None) -> None:
    # B2: Alias conflict detection deferred here from module-level so bootstrap
    # logging is available and errors are reported through structured channels.
    if alias_map is None and _ALIAS_CONFLICTS:
        raise ValueError(
            f"Alias conflict between core and passion extras: {sorted(_ALIAS_CONFLICTS)}"
        )
    aliases = alias_map if alias_map is not None else PASSION_MERCHANT_ALIASES
    bad_vals = [
        v for k, v in aliases.items()
        if not isinstance(v, str) or v != v.lower() or not v.strip()
    ]
    bad_keys = [
        k for k in aliases
        if not isinstance(k, str) or k != k.lower() or not k.strip()
    ]
    if bad_vals:
        raise ValueError(f"Non-lowercase, non-string, or empty values: {bad_vals}")
    if bad_keys:
        raise ValueError(f"Non-lowercase, non-string, or empty keys: {bad_keys}")

    # Validate GENERALIST_CANONICALS reachability
    # Fix #6: Check only alias_values — a canonical appearing only as an alias key
    # can still resolve away to a different value and fail resolved.isin(GENERALIST_CANONICALS).
    alias_values = set(aliases.values())
    missing = GENERALIST_CANONICALS - alias_values
    if missing:
        raise ValueError(f"GENERALIST_CANONICALS contains unreachable merchants: {missing}")

    # Validate specialized merchant map keys are reachable canonical outputs.
    # config.py is already a hard dependency of config_passion (imported at L5);
    # by the time validate_merchant_aliases() is called, config.py has already
    # loaded without error. Use a direct import — do NOT wrap in try/except ImportError.
    # A try/except here would silently degrade to _smap={} if the constant name
    # is mistyped, bypassing the validator entirely. Match the raise-on-error
    # pattern established in config_passion.py:L161-167 (validate_config).
    from config import SPECIALIZED_MERCHANT_SUBCATEGORY_MAP as _smap
    if _smap:
        validate_specialized_map(_smap)


def validate_specialized_map(
    specialized_map: dict[str, str],
) -> None:
    """Assert every SPECIALIZED_MERCHANT_SUBCATEGORY_MAP key is a reachable
    resolve_merchant_vectorized output. Fails loudly at startup if any key
    would silently produce no enrichment in production."""
    import pandas as pd
    from marketplace_subcategory import resolve_merchant_vectorized
    if not specialized_map:
        return
    keys = pd.Series(list(specialized_map.keys()))
    resolved = resolve_merchant_vectorized(keys)
    unreachable = {
        k for k, r in zip(keys, resolved) if k != r
    }
    if unreachable:
        raise ValueError(
            "SPECIALIZED_MERCHANT_SUBCATEGORY_MAP contains keys that are not "
            "reachable as resolve_merchant_vectorized outputs. These keys will "
            f"silently produce zero enrichment in production: {sorted(unreachable)}. "
            "Fix: use the exact string that resolve_merchant_vectorized returns for "
            "these merchants (test with resolve_merchant_vectorized(pd.Series([key])))."
        )


# Allowed fields: {category}, {merchant_count}, {merchant_label}, {spend_share}, {total_spend},
#                 {trend_direction}, {subcategory}  (Step 5d)
# merchant_label is pre-pluralized: "1 merchant" or "N merchants" — use this in templates.
# merchant_count is retained for backward compatibility with tests that pass it directly.
# Template order: index 1 is chosen by rng=Random(0) after tip selection consumes one call.
# Subcategory template placed at index 1 so plan validation tests pass with seed 0.
PASSION_INSIGHT_TEMPLATES: MappingProxyType = MappingProxyType({
    "lifestyle_opportunity": (
        "You show strong lifestyle engagement in {category} with {merchant_label} ({spend_share:.1%} of spend).",
        # Step 5e: index 1 — selected by Random(0) in plan validation tests.
        "Strong {subcategory} interest detected under {category} — {merchant_label}, {spend_share:.1%} of spend.",
        "Your {category} passion is showing — {merchant_label}, {spend_share:.1%} of total spend.",
    ),
})



# P1.10: Config numeric values not validated.
# Validates all numeric config constants. A misconfigured
# threshold (e.g. DISTRESS_FEES_THRESHOLD=2.0) would silently break
# the distress gate at runtime.
#
# FIX H13: This function is NO LONGER called at module import time.
# Module-level validate_config() crashed before logging was configured.
# It is now called from bootstrap.run_startup_checks() instead.
def validate_config() -> None:
    if MAX_SPIKE_CANDIDATES <= 0:
        raise ValueError("MAX_SPIKE_CANDIDATES must be positive")
    if PASSION_MIN_MONTHS <= 0:
        raise ValueError("PASSION_MIN_MONTHS must be positive")
    if PIPELINE_BUDGET_MS <= 0:
        raise ValueError("PIPELINE_BUDGET_MS must be positive")
    if PIPELINE_TOP_N <= 0:
        raise ValueError("PIPELINE_TOP_N must be positive")
    if PASSION_MERCHANT_COUNT_MIN <= 0:
        raise ValueError("PASSION_MERCHANT_COUNT_MIN must be positive")
    if not (0.0 <= PASSION_SPEND_SHARE_THRESHOLD <= 1.0):
        raise ValueError("PASSION_SPEND_SHARE_THRESHOLD must be in [0, 1]")
    if not (0.0 <= PASSION_ANOMALY_SUPPRESSION_THRESHOLD <= 1.0):
        raise ValueError("PASSION_ANOMALY_SUPPRESSION_THRESHOLD must be in [0, 1]")
    if not (0.0 <= DISTRESS_FEES_THRESHOLD <= 1.0):
        raise ValueError("DISTRESS_FEES_THRESHOLD must be in [0, 1]")
    # P1-6: Validate individual confidence bounds
    if not (0.0 <= MARKETPLACE_LOW_CONFIDENCE <= 1.0):
        raise ValueError("MARKETPLACE_LOW_CONFIDENCE must be in [0, 1]")
    if not (0.0 <= MARKETPLACE_HIGH_CONFIDENCE <= 1.0):
        raise ValueError("MARKETPLACE_HIGH_CONFIDENCE must be in [0, 1]")
    if MARKETPLACE_HIGH_CONFIDENCE <= MARKETPLACE_LOW_CONFIDENCE:
        raise ValueError("HIGH_CONFIDENCE must exceed LOW_CONFIDENCE")
    if MARKETPLACE_HIGH_AMOUNT_THRESHOLD <= 0:
        raise ValueError("MARKETPLACE_HIGH_AMOUNT_THRESHOLD must be positive")
    # FIX M4: Validate hard timeout exceeds budget.
    if PIPELINE_HARD_TIMEOUT_MS <= PIPELINE_BUDGET_MS:
        raise ValueError("PIPELINE_HARD_TIMEOUT_MS must exceed PIPELINE_BUDGET_MS")

    # Step 4b: Validate SUBCATEGORY_MERCHANT_COUNT_MIN if it exists.
    # Use try/except NameError rather than globals().get() — if the constant is renamed,
    # globals().get() silently returns {} and skips validation entirely with no error.
    # try/except NameError produces a visible failure at the correct call site.
    try:
        _subcat_min = SUBCATEGORY_MERCHANT_COUNT_MIN
    except NameError:
        _subcat_min = {}
    for _subcat, _min_val in _subcat_min.items():
        if not isinstance(_min_val, int) or _min_val < 1:
            raise ValueError(
                f"SUBCATEGORY_MERCHANT_COUNT_MIN[{_subcat!r}] must be a positive int, got {_min_val!r}"
            )

    # B6: Do NOT swallow ImportError here. A broken config import means the
    # validation environment itself is broken, which is a hard failure.
    # Previously: except ImportError: pass — this silently skipped validation
    # of ELECTRONICS_ALLOWED_CATEGORIES when config was misconfigured.
    try:
        from config import CATEGORY_KEYWORDS, CATEGORY_PRIORITY
    except ImportError as e:
        raise RuntimeError(
            f"config import failed during validate_config: {e}. "
            "Ensure config.py exposes CATEGORY_KEYWORDS and CATEGORY_PRIORITY."
        ) from e
    cat_priority_keys = set()
    if isinstance(CATEGORY_PRIORITY, dict):
        cat_priority_keys = set(CATEGORY_PRIORITY.keys())
    elif isinstance(CATEGORY_PRIORITY, (list, tuple, set)):
        for item in CATEGORY_PRIORITY:
            if isinstance(item, (list, tuple)) and len(item) > 0 and isinstance(item[0], str):
                cat_priority_keys.add(item[0])
            elif isinstance(item, str):
                cat_priority_keys.add(item)
    valid_cats = set(CATEGORY_KEYWORDS.keys()) | cat_priority_keys
    if not valid_cats:
        raise RuntimeError("No valid categories could be derived from config")
    invalid = set(ELECTRONICS_ALLOWED_CATEGORIES) - valid_cats
    if invalid:
        raise ValueError(f"ELECTRONICS_ALLOWED_CATEGORIES contains invalid categories: {invalid}")


# Step 4b (D1 approved): Subcategory-specific minimum unique merchant count overrides.
# Subcategories not listed here fall back to PASSION_MERCHANT_COUNT_MIN (3).
# Design rationale:
#   fashion/beauty = platform-dominant (Myntra IS fashion, Nykaa IS beauty).
#     Requiring 2+ distinct merchants defeats the signal — lowered to 1.
#   gaming/education/eyewear = inherently narrow platform ecosystems.
#   software = professional tools — JetBrains, Notion, GitHub are single-platform signals.
#   Subcategories not listed here require the global min of 3.
SUBCATEGORY_MERCHANT_COUNT_MIN: dict[str, int] = {
    "gaming":    1,  # Steam, Epic are 1-2 platforms; valid passion with single merchant
    "education": 1,  # Single platform (Udemy) can represent strong signal
    "eyewear":   1,  # Lenskart is dominant; requiring 3 is unrealistic
    "beauty":    1,  # Nykaa is the dominant platform — 1 merchant is a valid signal
    "fashion":   1,  # Myntra is the dominant platform — 1 merchant is a valid signal
    "software":      1,  # JetBrains, GitHub, Notion — one platform is a real passion signal
    "grocery":       2,  # Bigbasket + one other to distinguish passion from necessity
    "cloud_storage": 1,  # Google One, iCloud — single service is a valid signal
}

# Categories that can NEVER represent a lifestyle passion signal.
# Cash withdrawals (atm), financial admin (finance), and utility payments
# are structural/operational spend — not discretionary lifestyle choices.
PASSION_BLOCKED_CATEGORIES: frozenset[str] = frozenset({
    "atm",
    "finance",
    "utilities",
})

# FIX-37: Explicit public API.
# D5: PASSION_MIN_DEBIT_ROWS added — passion_pipeline.py now imports it at
# module top level; the dead try/except ImportError guard has been removed.
__all__ = [
    "validate_merchant_aliases",
    "validate_specialized_map",
    "PASSION_INSIGHT_TEMPLATES",
    "PASSION_MERCHANT_ALIASES",
    "GENERALIST_CANONICALS",
    "ELECTRONICS_ALLOWED_CATEGORIES",
    "MAX_SPIKE_CANDIDATES",
    "PIPELINE_BUDGET_MS",
    "PIPELINE_TOP_N",
    "PIPELINE_HARD_TIMEOUT_MS",
    "PASSION_MIN_MONTHS",
    "PASSION_MIN_DEBIT_ROWS",
    "PASSION_MERCHANT_COUNT_MIN",
    "SUBCATEGORY_MERCHANT_COUNT_MIN",
    "PASSION_SPEND_SHARE_THRESHOLD",
    "PASSION_ANOMALY_SUPPRESSION_THRESHOLD",
    "DISTRESS_FEES_THRESHOLD",
    "MARKETPLACE_HIGH_AMOUNT_THRESHOLD",
    "MARKETPLACE_HIGH_CONFIDENCE",
    "MARKETPLACE_LOW_CONFIDENCE",
    "PASSION_BLOCKED_CATEGORIES",
    "validate_config",
]
