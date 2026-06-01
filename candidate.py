from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Any
import numpy as np

__all__ = ["Candidate"]

def _coerce_finite_float(val: Any, name: str) -> float:
    # 1. Unpack numpy generic scalar types
    if isinstance(val, np.generic):
        val = val.item()
    # 2. Reject boolean types explicitly (since bool inherits from int)
    if isinstance(val, bool):
        raise TypeError(f"{name} must be numeric, got bool")
    # 3. Check for standard numeric types
    if not isinstance(val, (int, float)):
        raise TypeError(f"{name} must be numeric, got {type(val).__name__}")
    # 4. Check for NaN/Inf
    try:
        coerced = float(val)
    except (TypeError, ValueError) as e:
        raise TypeError(f"{name} must be numeric, got {type(val).__name__}") from e
    if math.isnan(coerced) or math.isinf(coerced):
        raise ValueError(f"{name} must be finite")
    return coerced

@dataclass(frozen=True, slots=True)
class Candidate:
    score: float
    category: str
    insight_type: str
    merchant: str
    amount: float
    sort_key_ts: int

    # H4, H5: Preserve signal metadata and add normalized_score
    merchant_count: int = 1
    spend_share: float = 0.0
    total_spend: float = 0.0
    trend_direction: str = ""
    suppression_reason: str = ""
    normalized_score: float = 0.0
    original_index: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, 'score', _coerce_finite_float(self.score, 'score'))
        object.__setattr__(self, 'amount', _coerce_finite_float(self.amount, 'amount'))
        object.__setattr__(self, 'normalized_score', _coerce_finite_float(self.normalized_score, 'normalized_score'))

        # Standard integer validation for sort_key_ts
        val_ts = self.sort_key_ts
        if isinstance(val_ts, np.generic):
            val_ts = val_ts.item()
        if isinstance(val_ts, bool) or not isinstance(val_ts, int):
            raise TypeError(f"sort_key_ts must be int, got {type(val_ts).__name__}")
        if val_ts < 0:
            raise ValueError("sort_key_ts must be a non-negative integer")
        object.__setattr__(self, 'sort_key_ts', val_ts)

        # Standard integer validation for original_index
        val_idx = self.original_index
        if isinstance(val_idx, np.generic):
            val_idx = val_idx.item()
        if isinstance(val_idx, bool) or not isinstance(val_idx, int) or val_idx < 0:
            raise ValueError("original_index must be a non-negative integer")
        object.__setattr__(self, 'original_index', val_idx)

        if not isinstance(self.category, str) or not self.category.strip():
            raise ValueError("category must be non-empty str")
        if not isinstance(self.merchant, str) or not self.merchant.strip():
            raise ValueError("merchant must be non-empty str")
        if not isinstance(self.insight_type, str) or not self.insight_type.strip():
            raise ValueError("insight_type must be non-empty str")

    # H6: Candidate sort order must be deterministic
    def __lt__(self, other: "Candidate") -> bool:
        if not isinstance(other, Candidate): return NotImplemented

        type_priority = {"subscription": 0, "spending_spike": 1, "lifestyle_opportunity": 2}
        p1 = type_priority.get(self.insight_type, 99)
        p2 = type_priority.get(other.insight_type, 99)

        SCORE_TOLERANCE = 1e-12
        if abs(self.normalized_score - other.normalized_score) > SCORE_TOLERANCE:
            return self.normalized_score > other.normalized_score
        if p1 != p2:
            return p1 < p2
        if self.category != other.category:
            return self.category < other.category
        if self.merchant != other.merchant:
            return self.merchant < other.merchant
        if self.sort_key_ts != other.sort_key_ts:
            return self.sort_key_ts > other.sort_key_ts
        return self.original_index < other.original_index

    # H4: Candidate.passion(signal) classmethod
    @classmethod
    def passion(
        cls,
        signal,
        sort_key_ts: int | None = None,
        normalized_score: float | None = None,
        original_index: int | None = None,
    ) -> "Candidate":
        from passion_models import PassionSignal

        if not isinstance(signal, PassionSignal):
            raise TypeError("Must provide a PassionSignal")

        if sort_key_ts is None:
            sort_key_ts = signal.latest_ts
        if normalized_score is None:
            normalized_score = signal.spend_share
        if original_index is None:
            original_index = signal.original_index

        return cls(
            score=0.0,
            category=signal.category,
            insight_type="lifestyle_opportunity",
            merchant=signal.merchant_list[0] if signal.merchant_list else "unknown",
            amount=signal.total_spend,
            sort_key_ts=sort_key_ts,
            merchant_count=signal.merchant_count,
            spend_share=signal.spend_share,
            total_spend=signal.total_spend,
            trend_direction=signal.trend_direction,
            suppression_reason=signal.suppression_reason,
            normalized_score=normalized_score,
            original_index=original_index,
        )

    @classmethod
    def subscription(cls, score: float, category: str, merchant: str,
                     amount: float, sort_key_ts: int, normalized_score: float = 0.0, original_index: int = 0) -> "Candidate":
        return cls(
            score=score,
            category=category,
            insight_type="subscription",
            merchant=merchant,
            amount=amount,
            sort_key_ts=sort_key_ts,
            normalized_score=normalized_score,
            original_index=original_index,
        )

    @classmethod
    def spending_spike(cls, score: float, category: str, merchant: str,
                       amount: float, sort_key_ts: int, normalized_score: float = 0.0, original_index: int = 0) -> "Candidate":
        return cls(
            score=score,
            category=category,
            insight_type="spending_spike",
            merchant=merchant,
            amount=amount,
            sort_key_ts=sort_key_ts,
            normalized_score=normalized_score,
            original_index=original_index,
        )
