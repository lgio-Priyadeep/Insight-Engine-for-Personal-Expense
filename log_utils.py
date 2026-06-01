import os
import hmac
import hashlib
import threading
import numpy as np
import pandas as pd
from typing import Any

# FIX H11: Use structured logger from logger_factory instead of plain logging.
from logger_factory import get_logger

__all__ = ["log_safe_merchant", "log_safe_text", "verify_merchant_token"]
logger = get_logger(__name__)

# FIX L5: Renamed for clarity. _DEV_SECRET_FALLBACK is the constant.
_DEV_SECRET_FALLBACK = b"dev-only-not-for-production-use!"
_active_secret: bytes | None = None
_secret_lock = threading.Lock()

# E2: Provide _reset_secret_cache for tests
def _reset_secret_cache() -> None:
    global _active_secret
    with _secret_lock:
        _active_secret = None

def _get_secret() -> bytes:
    global _active_secret
    if _active_secret is not None:
        return _active_secret
    with _secret_lock:
        if _active_secret is not None:
            return _active_secret

        env_name = os.environ.get("ENV", "development").strip().lower()
        env_val = os.environ.get("INSIGHT_ENGINE_SECRET")
        if env_val:
            secret_stripped = env_val.strip()
            if not secret_stripped:
                raise RuntimeError("INSIGHT_ENGINE_SECRET is set but blank.")
            secret_bytes = secret_stripped.encode('utf-8')
            if len(secret_bytes) < 32:
                raise RuntimeError(f"INSIGHT_ENGINE_SECRET must be at least 32 bytes, got {len(secret_bytes)}.")
            _active_secret = secret_bytes
        else:
            # E1: Never fall back in prod/staging
            if env_name in ("production", "prod", "staging"):
                raise RuntimeError("CRITICAL: INSIGHT_ENGINE_SECRET missing in production/staging.")
            logger.warning(
                "insight_engine_dev_secret_active: HMAC tokens are "
                "deterministic and NOT secure. Set INSIGHT_ENGINE_SECRET "
                "before deploying to production."
            )
            _active_secret = _DEV_SECRET_FALLBACK

        return _active_secret

def _hmac_hex(value: str) -> str:
    return hmac.new(_get_secret(), value.encode('utf-8'), hashlib.sha256).hexdigest()[:32]

def _is_safe_scalar(val: Any) -> bool:
    return isinstance(val, (str, float, int, type(None), np.generic))

def log_safe_merchant(merchant: Any) -> str:
    if not _is_safe_scalar(merchant): return ""
    try:
        if pd.isna(merchant) or not str(merchant).strip(): return ""
    except (TypeError, ValueError): return ""
    return f"merchant:{_hmac_hex(str(merchant))}"

def log_safe_text(text: Any) -> str:
    if not _is_safe_scalar(text): return ""
    try:
        if pd.isna(text) or not str(text).strip(): return ""
    except (TypeError, ValueError): return ""
    return f"text:{_hmac_hex(str(text))}"


# P3.3: verify_merchant_token — constant-time HMAC comparison.
# Use whenever comparing HMAC tokens to prevent timing side-channels.
def verify_merchant_token(token: str, merchant: str) -> bool:
    """Constant-time token verification. Use whenever comparing HMAC tokens."""
    expected = log_safe_merchant(merchant)
    # FIX L8: Catch TypeError in compare_digest (e.g. if token is None or wrong type)
    try:
        return hmac.compare_digest(token, expected)
    except TypeError:
        return False
