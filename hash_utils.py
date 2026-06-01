import warnings
import hashlib

# FIX-37: Explicit public API.
__all__ = ["stable_hash"]


def stable_hash(value: str) -> str:
    """
    DEPRECATED: This function produces reversible, non-keyed hashes.
    Use log_utils.log_safe_merchant() for HMAC-keyed PII masking instead.
    """
    warnings.warn(
        "stable_hash is deprecated and produces reversible hashes. "
        "Use log_utils.log_safe_merchant instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
