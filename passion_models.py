from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Final
import numpy as np

__all__ = ["PassionSignal"]

# Fix 11: Module-level epsilon for float comparison tolerance.
_EPS = 1e-9

# Blocker 2 & 13: Both anomaly and distress map to "suppressed".
_VALID_TRENDS: frozenset[str] = frozenset({
    "non_declining", "declining", "insufficient_history", "suppressed"
})

# H1: frozen=True, kw_only=True, NO slots=True
@dataclass(frozen=True, kw_only=True)
class PassionSignal:
    category: str
    merchant_list: tuple[str, ...]
    total_spend: float
    merchant_count: int
    spend_share: float
    trend_direction: str
    is_suppressed: bool = False
    suppression_reason: str = ""
    latest_ts: int = 0
    original_index: int = 0
    active_months: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.category, str) or not self.category.strip():
            raise ValueError("category must be a non-empty string")

        if isinstance(self.merchant_list, str):
            raise TypeError("merchant_list must be a tuple/list of str, not a bare string")

        if not isinstance(self.merchant_list, (tuple, list)):
            raise TypeError("merchant_list must be a tuple/list of str")

        object.__setattr__(self, "merchant_list", tuple(self.merchant_list))

        for m in self.merchant_list:
            if not isinstance(m, str):
                raise TypeError(f"merchant_list elements must be str, got {type(m)}")
            if not m.strip():
                raise ValueError("merchant_list elements must be non-empty str")
        if not isinstance(self.total_spend, (int, float, np.integer, np.floating)):
            raise TypeError(f"total_spend must be numeric, got {type(self.total_spend)}")
        if math.isnan(float(self.total_spend)) or math.isinf(float(self.total_spend)):
            raise ValueError("total_spend must be finite")
        # H2: raise ValueError without clamping for total_spend < 0
        if float(self.total_spend) < 0:
            raise ValueError("total_spend must be non-negative")
        if not isinstance(self.merchant_count, (int, np.integer)):
            raise TypeError(f"merchant_count must be int")
        # H2: raise ValueError without clamping for merchant_count != len(merchant_list)
        if int(self.merchant_count) != len(self.merchant_list):
            raise ValueError(f"merchant_count ({self.merchant_count}) != len(merchant_list) ({len(self.merchant_list)})")
        if not isinstance(self.spend_share, (int, float, np.integer, np.floating)):
            raise TypeError("spend_share must be numeric")

        # H2: raise ValueError without clamping for spend_share not in [0, 1]
        if not (-_EPS <= float(self.spend_share) <= 1.0 + _EPS):
            raise ValueError(f"spend_share must be in [0, 1], got {self.spend_share}")
        # NOTE: Do NOT clamp to clean value, raise ValueError if it's truly out of range.

        # H2: trend_direction not in allowed set
        if self.trend_direction not in _VALID_TRENDS:
            raise ValueError(f"trend_direction must be one of {sorted(_VALID_TRENDS)}, got {self.trend_direction!r}")
        if not isinstance(self.is_suppressed, bool):
            raise TypeError("is_suppressed must be bool")
        if not isinstance(self.suppression_reason, str):
            raise TypeError("suppression_reason must be str")

        # H2: is_suppressed True with empty suppression_reason
        if self.is_suppressed and not self.suppression_reason:
            raise ValueError("suppressed signals must have a non-empty suppression_reason")
        if not self.is_suppressed and self.suppression_reason:
            raise ValueError("non-suppressed signals must have empty suppression_reason")

        if not isinstance(self.latest_ts, (int, np.integer)) or self.latest_ts < 0:
            raise ValueError("latest_ts must be a non-negative integer")
        if not isinstance(self.original_index, (int, np.integer)) or self.original_index < 0:
            raise ValueError("original_index must be a non-negative integer")

        # P0-2: active_months validation
        if not isinstance(self.active_months, (int, np.integer)) or self.active_months < 0:
            raise ValueError("active_months must be a non-negative integer")

        # Fix 11: Normalize scalar fields to native Python types after all validation.
        # Prevents numpy scalars from leaking into logs, JSON, and downstream consumers.
        object.__setattr__(self, "total_spend", float(self.total_spend))
        object.__setattr__(self, "merchant_count", int(self.merchant_count))
        object.__setattr__(self, "spend_share", float(self.spend_share))
        object.__setattr__(self, "latest_ts", int(self.latest_ts))
        object.__setattr__(self, "original_index", int(self.original_index))
        object.__setattr__(self, "active_months", int(self.active_months))
        object.__setattr__(self, "is_suppressed", bool(self.is_suppressed))
        object.__setattr__(self, "category", self.category.strip().lower())
        object.__setattr__(self, "suppression_reason", str(self.suppression_reason).strip())
