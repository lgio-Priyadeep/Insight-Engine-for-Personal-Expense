from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING
import pandas as pd

if TYPE_CHECKING:
    from candidate import Candidate
    from passion_models import PassionSignal

# B1: PassionSignal is NOT imported at module top-level.
# pipeline_result.py is imported by core pipeline paths. A top-level sidecar
# import would make the optional Passion Engine mandatory for the entire core.
# TYPE_CHECKING guard above satisfies type checkers; runtime uses lazy import
# inside __post_init__ only.

__all__ = ["PassionResult"]


# M2: PassionResult frozen=True, kw_only=True, NO slots=True.
# Remove personal_insights and personal_summary (core owns these).
@dataclass(frozen=True, kw_only=True)
class PassionResult:
    """
    Immutable container for passion pipeline output.

    DataFrames are defensive copies.
    Call .copy(deep=True) before mutating.
    """
    debits: pd.DataFrame
    candidates: tuple[Candidate, ...]
    insights: tuple[str, ...]
    passion_signals: tuple[PassionSignal, ...]
    passion_debits: pd.DataFrame = field(default_factory=pd.DataFrame)

    def __post_init__(self) -> None:
        from candidate import Candidate as _Candidate

        # B1: Lazy import — imported here (not at module level) so that importing
        # pipeline_result.py does not force passion_models to load on the core path.
        from passion_models import PassionSignal as _PassionSignal

        if not isinstance(self.debits, pd.DataFrame):
            raise TypeError(f"debits must be pd.DataFrame, got {type(self.debits)}")
        object.__setattr__(self, "debits", self.debits.copy(deep=True))

        if not isinstance(self.passion_debits, pd.DataFrame):
            raise TypeError(f"passion_debits must be pd.DataFrame, got {type(self.passion_debits)}")
        object.__setattr__(self, "passion_debits", self.passion_debits.copy(deep=True))

        field_element_types = {
            'candidates': _Candidate,
            'insights': str,
            'passion_signals': _PassionSignal,
        }

        for field_name, elem_type in field_element_types.items():
            val = getattr(self, field_name)
            # M3: Reject str, dict, set, frozenset
            if isinstance(val, (str, dict, set, frozenset)):
                raise TypeError(f"'{field_name}' must be tuple or list, not {type(val).__name__}")
            if not hasattr(val, '__iter__'):
                raise TypeError(f"'{field_name}' must be iterable, got {type(val)}")

            # M3: Convert list to tuple
            if not isinstance(val, tuple):
                val = tuple(val)
                object.__setattr__(self, field_name, val)

            if val and not all(isinstance(x, elem_type) for x in val):
                bad = [type(x).__name__ for x in val if not isinstance(x, elem_type)][:3]
                raise TypeError(
                    f"'{field_name}' elements must be {elem_type.__name__}, got {bad}"
                )
