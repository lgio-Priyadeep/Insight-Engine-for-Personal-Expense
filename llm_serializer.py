"""
llm_serializer.py — LLM Export Serialization Utilities
=======================================================
Responsibilities:
  - _json_safe      : Coerce numpy scalars, NaN, Inf, pd.NA, pd.NaT → JSON-safe types.
  - _mask_merchant  : Apply PII masking gate via log_safe_merchant when pii_safe=True.
  - serialize_to_json: Serialize a context dict to a JSON string.
  - write_json_atomic: Write JSON to disk atomically (POSIX-atomic via os.replace).

PII gate rule: pii_safe = not config.ENABLE_PII_DEBUG_LOGS
"""

from __future__ import annotations

import json
import logging
import math
import os
from typing import Any

import pandas as pd

import config
from log_utils import log_safe_merchant

logger = logging.getLogger(__name__)


def _json_safe(obj: Any) -> Any:
    """
    JSON default handler — coerces types that are not natively JSON-serializable.

    Handles:
        - numpy integer types → int
        - numpy float types   → float (NaN/Inf → None)
        - Python float NaN/Inf → None
        - pd.NA               → None
        - pd.NaT              → None
        - Everything else     → str(obj) as a last-resort fallback
    """
    # pandas NA sentinel
    if obj is pd.NA or obj is pd.NaT:
        return None

    # numpy integers
    try:
        import numpy as np
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            val = float(obj)
            if math.isnan(val) or math.isinf(val):
                return None
            return val
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except ImportError:
        pass  # numpy not available — skip numpy checks

    # Python float NaN / Inf
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj

    # Fallback: stringify unknown types rather than raising TypeError
    logger.warning(
        "json_safe_fallback",
        extra={"type": type(obj).__name__, "value_repr": repr(obj)[:80]},
    )
    return str(obj)


def _mask_merchant(name: str, pii_safe: bool) -> str:
    """
    Return a PII-safe version of a merchant name when pii_safe=True.

    When pii_safe=False (ENABLE_PII_DEBUG_LOGS mode), the raw name is returned
    unchanged — this mode is ONLY for local debugging and must never reach
    production export files.

    Args:
        name:     Merchant name string (may be empty).
        pii_safe: True → apply HMAC-keyed token; False → passthrough.

    Returns:
        str — masked token or original name.
    """
    if not isinstance(name, str) or not name:
        return ""
    if pii_safe:
        return log_safe_merchant(name)
    return name


def serialize_to_json(context: dict, indent: int = 2) -> str:
    """
    Serialize a context dict to a JSON string.

    Uses _json_safe as the default handler so numpy scalars, NaN, and Inf
    are transparently coerced to JSON-safe equivalents.

    Args:
        context: Dict to serialize (produced by build_llm_context / export_to_json).
        indent:  JSON indentation level (default 2).

    Returns:
        str — UTF-8 JSON string.

    Raises:
        TypeError: if a type cannot be handled even by _json_safe.
    """
    return json.dumps(context, indent=indent, default=_json_safe, ensure_ascii=False)


def write_json_atomic(context: dict, path: str) -> None:
    """
    Write a JSON context dict to disk atomically.

    Writes to a <path>.tmp file first, then uses os.replace() for a POSIX-atomic
    rename. This prevents partial writes from corrupting the output file on
    crash or interruption.

    Args:
        context: Dict to serialize (produced by build_llm_context / export_to_json).
        path:    Final destination path (absolute or relative to cwd).

    Raises:
        OSError: on file system errors (directory not found, permission denied, etc.).
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp_path = path + ".tmp"
    try:
        json_str = serialize_to_json(context)
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(json_str)
        os.replace(tmp_path, path)
        logger.debug(
            "json_written_atomic",
            extra={"path": path, "bytes": len(json_str.encode("utf-8"))},
        )
    except Exception:
        # Attempt cleanup of tmp file on failure — do not suppress the original error
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# Module-level PII gate helper — used by aggregator and export modules
# ---------------------------------------------------------------------------

def get_pii_safe_flag() -> bool:
    """Return True when PII masking is active (production default)."""
    return not getattr(config, "ENABLE_PII_DEBUG_LOGS", False)
