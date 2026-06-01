import os
import string
import sys
from contracts import INSIGHT_TEMPLATES, TIP_CORPUS
from types import MappingProxyType

# FIX H11: Use structured logger from logger_factory instead of plain logging.
from logger_factory import get_logger

logger = get_logger(__name__)

# FIX-37: Explicit public API.
__all__ = ["run_startup_checks", "validate_template_fields", "ALLOWED_FORMAT_SPECS"]

ALLOWED_FORMAT_SPECS = frozenset({
    "",
    ".0%",
    ".1%",
    ".0f",
    ".1f",
    ".2f",
    ",.0f",
    ",.2f",
    "d",
    ",d",
})

# FIX M6 + FIX-14: validate_template_fields now rejects attribute/index access,
# conversions (!r, !s, !a), nested format specs, and positional fields.
# FIX-14: string.Formatter().parse() yields (literal, field, format_spec, conversion).
# Previous code ignored conversion and format_spec — "{x!r}" or "{x:{y}}" passed silently.
def validate_template_fields(template: str, allowed: set[str]) -> None:
    try:
        parsed = list(string.Formatter().parse(template))
    except (ValueError, KeyError) as e:
        raise ValueError(f"Malformed template {template!r}: {e}") from e
    fields = set()
    for _, field, format_spec, conversion in parsed:
        if field is None:
            continue
        if conversion is not None:
            raise ValueError(f"Conversions (!r, !s, !a) are forbidden in template: {template!r}")
        if "{" in (format_spec or "") or "}" in (format_spec or ""):
            raise ValueError(f"Nested format specs are forbidden in template: {template!r}")
        if (format_spec or "") not in ALLOWED_FORMAT_SPECS:
            raise ValueError(f"Format spec {format_spec!r} not in allowed list: {ALLOWED_FORMAT_SPECS}")
        if field == "" or field.isdigit():
            raise ValueError(f"Positional format fields are forbidden in template: {template!r}")
        if any(c in field for c in (".", "[", "]")):
            raise ValueError(
                f"Field names must be simple identifiers, got {field!r} in {template!r}"
            )
        fields.add(field)
    if unknown := fields - allowed:
        raise ValueError(f"Unknown template fields: {sorted(unknown)}")


def _validate_python_version() -> None:
    if sys.version_info < (3, 11):
        raise RuntimeError(
            f"Insight Engine requires Python 3.11 or later. Running: {sys.version}"
        )


# FIX #17 + P1.6: _validate_schema_columns.
# P1.6: Use vars(Col) instead of dir(Col). dir() includes inherited object
# attrs like __init__, __repr__ etc. vars() returns only the class's own namespace.
# FIX #17: Reject None and empty-string values before the lowercase check.
# FIX H6: Check for duplicate Col values — two constants sharing the same
# string value would cause DataFrames to silently overwrite each other's columns.
# FIX-26: Accept optional _col_cls parameter for testability — tests can now call
# _validate_schema_columns(mock_col) without patching bootstrap.Col.
def _validate_schema_columns(_col_cls=None) -> None:
    from schema import Col as _default_col
    col_cls = _col_cls or _default_col

    required = {
        "AMOUNT", "CLEANED_REMARKS", "DATE", "PREDICTED_CATEGORY",
        "IS_ANOMALY", "IS_RECURRING", "INSIGHT_SCORE", "IS_KNOWN_PERSON",
        "RECURRING_FREQUENCY", "INFERRED_SUBCATEGORY", "SUBCATEGORY_CONFIDENCE",
    }
    # P1.6: vars(col_cls) returns only class's own __dict__, not inherited attrs.
    present = {
        a for a, v in vars(col_cls).items()
        if not a.startswith("_") and isinstance(v, str)
    }
    missing = required - present
    if missing:
        raise RuntimeError(f"schema.Col missing required constants: {sorted(missing)}")
    for attr in required:
        val = getattr(col_cls, attr, None)
        if val is None:
            raise RuntimeError(f"schema.Col.{attr} is missing")
        if not val:
            raise RuntimeError(f"schema.Col.{attr} value must be non-empty")
        if val != val.lower():
            raise RuntimeError(
                f"schema.Col.{attr} value must be lowercase, got {val!r}"
            )

    # FIX H6 + FIX L5: Detect duplicate string values across all Col constants.
    from collections import Counter
    all_values = [
        v for a, v in vars(col_cls).items()
        if not a.startswith("_") and isinstance(v, str)
    ]
    duplicates = [item for item, count in Counter(all_values).items() if count > 1]
    if duplicates:
        raise RuntimeError(f"schema.Col has duplicate string values: {duplicates}")


def _validate_insight_templates() -> None:
    if not isinstance(INSIGHT_TEMPLATES, MappingProxyType):
        raise TypeError(f"INSIGHT_TEMPLATES must be MappingProxyType, got {type(INSIGHT_TEMPLATES)}")
    insight_allowed_fields = {"merchant", "amount", "category", "merchant_count", "spend_share",
                              "date", "pct", "frequency"}
    for category, templates in INSIGHT_TEMPLATES.items():
        if not isinstance(templates, tuple):
            raise TypeError(f"INSIGHT_TEMPLATES['{category}'] must be tuple, got {type(templates)}")
        for t in templates:
            if not isinstance(t, str):
                raise TypeError(f"INSIGHT_TEMPLATES['{category}'] entry must be str, got {type(t)}")
            validate_template_fields(t, insight_allowed_fields)


def _validate_passion_templates() -> None:
    from config_passion import PASSION_INSIGHT_TEMPLATES
    if not isinstance(PASSION_INSIGHT_TEMPLATES, MappingProxyType):
        raise TypeError(f"PASSION_INSIGHT_TEMPLATES must be MappingProxyType, got {type(PASSION_INSIGHT_TEMPLATES)}")
    if "lifestyle_opportunity" not in PASSION_INSIGHT_TEMPLATES:
        raise ValueError("PASSION_INSIGHT_TEMPLATES must contain key 'lifestyle_opportunity'")
    passion_allowed_fields = {"category", "merchant_count", "spend_share", "trend_direction", "total_spend"}
    for key, templates in PASSION_INSIGHT_TEMPLATES.items():
        if not isinstance(templates, tuple):
            raise TypeError(f"PASSION_INSIGHT_TEMPLATES['{key}'] must be tuple, got {type(templates)}")
        for t in templates:
            if not isinstance(t, str):
                raise TypeError(f"PASSION_INSIGHT_TEMPLATES['{key}'] entry must be str, got {type(t)}")
            validate_template_fields(t, passion_allowed_fields)




    

def _validate_tip_corpus() -> None:
    # Fix 13: All business rule validation is now owned by contracts._freeze_tip_corpus,
    # which runs at import time. This function only asserts the frozen MappingProxyType
    # shape that contracts guarantees, and delegates template rendering to _dry_render_templates.
    if not isinstance(TIP_CORPUS, MappingProxyType):
        raise TypeError(f"TIP_CORPUS must be MappingProxyType, got {type(TIP_CORPUS)}")
    for tip_id, tip_data in TIP_CORPUS.items():
        if not isinstance(tip_data, MappingProxyType):
            raise TypeError(
                f"TIP_CORPUS['{tip_id}'] inner value must be MappingProxyType, got {type(tip_data)} "
                "(contracts._freeze_tip_corpus should have enforced this at import time)"
            )
        # Defensive: assert required keys are present (contracts enforces this, belt-and-suspenders).
        for req in ("text", "categories", "insights"):
            if req not in tip_data:
                raise ValueError(
                    f"TIP_CORPUS['{tip_id}'] missing required key '{req}' "
                    "(contracts._freeze_tip_corpus invariant violated)"
                )


def _validate_secret() -> None:
    from log_utils import _get_secret
    _get_secret()


# FIX-28: Dry-render all templates at startup to catch format errors early.
# A template like "{merchant} spent {ammount}" (typo) would only crash at
# runtime when a real insight is rendered. Dry-rendering catches it here.
def _dry_render_templates() -> None:
    # Blocker 17: Add total_spend and trend_direction to dry-render sample values
    _SAMPLE_VALUES = {
        "merchant": "sample_merchant", "amount": 1.0, "category": "food",
        "merchant_count": 1, "spend_share": 0.1, "date": "2025-01-01",
        "pct": 10, "frequency": "monthly",
        "total_spend": 100.0, "trend_direction": "non_declining",
    }
    from config_passion import PASSION_INSIGHT_TEMPLATES
    all_corpora = [
        ("INSIGHT_TEMPLATES", INSIGHT_TEMPLATES),
        ("PASSION_INSIGHT_TEMPLATES", PASSION_INSIGHT_TEMPLATES),
    ]
    for corpus_name, corpus in all_corpora:
        for key, templates in corpus.items():
            for t in templates:
                try:
                    t.format(**_SAMPLE_VALUES)
                except (KeyError, ValueError, TypeError) as e:
                    raise ValueError(
                        f"{corpus_name}['{key}'] template fails dry render: {t!r} -- {e}"
                    ) from e
    for tip_id, tip_data in TIP_CORPUS.items():
        t = tip_data.get("text", "")
        try:
            t.format(**_SAMPLE_VALUES)
        except (KeyError, ValueError, TypeError) as e:
            raise ValueError(
                f"TIP_CORPUS['{tip_id}']['text'] fails dry render: {t!r} -- {e}"
            ) from e

    # Fix #8: Context-specific dry-render per insight type.
    # The superset pass above catches unknown fields; these passes catch fields
    # that are valid in the superset but missing in a specific render context.
    # e.g. {trend_direction} passes the superset but fails subscription context.
    _SUBSCRIPTION_VALUES = {
        "merchant": "sample_merchant",
        "amount": 1.0,
        "category": "food",
        "pct": 10,
        "frequency": "monthly",
        "date": "2025-01-01",
    }
    _LIFESTYLE_VALUES = {
        "category": "food",
        "merchant_count": 1,
        "spend_share": 0.1,
        "trend_direction": "non_declining",
        "total_spend": 100.0,
    }
    for tip_id, tip_data in TIP_CORPUS.items():
        t = tip_data.get("text", "")
        insights = tip_data.get("insights", ())
        for insight_type in insights:
            ctx = _LIFESTYLE_VALUES if insight_type == "lifestyle_opportunity" else _SUBSCRIPTION_VALUES
            try:
                t.format(**ctx)
            except (KeyError, ValueError, TypeError) as e:
                raise ValueError(
                    f"TIP_CORPUS[{tip_id!r}]['text'] fails context render "
                    f"for insight_type={insight_type!r}: {t!r} -- {e}"
                ) from e


def run_startup_checks(env: str | None = None) -> None:
    _validate_python_version()
    env = (env or os.environ.get("ENV", "development")).strip().lower()
    secret = os.environ.get("INSIGHT_ENGINE_SECRET")
    if env in ("production", "prod", "staging") and not secret:
        raise RuntimeError("CRITICAL: INSIGHT_ENGINE_SECRET missing in production/staging.")
    _validate_secret()
    _validate_schema_columns()
    from config_passion import validate_merchant_aliases
    validate_merchant_aliases()
    _validate_insight_templates()
    _validate_passion_templates()
    _validate_tip_corpus()
    # FIX-28: Catch template format errors at startup, not at runtime.
    _dry_render_templates()

    # FIX H13: config_passion.validate_config() was called at module import time,
    # which crashes before logging is configured. Now deferred to bootstrap.
    from config_passion import validate_config as _validate_passion_config
    _validate_passion_config()

    logger.info("Insight Engine startup checks passed successfully.")
