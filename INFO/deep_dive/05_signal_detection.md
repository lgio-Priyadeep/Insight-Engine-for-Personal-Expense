# Subsystem: 05 Signal Detection

## Files in Scope
- `anomaly_detector.py` (CORE_LOGIC)
- `recurring_detector.py` (CORE_LOGIC)
- `tests/test_phase3.py` (TEST)

## 1. Responsibility
- **Owns**: Translating the continuous features generated in Phase 3/4 (Z-scores, expected amounts, rolling dates) into binary boolean flags (`IS_ANOMALY`, `IS_RECURRING`).
- **Does NOT own**: The raw statistical calculations themselves (these were done by `feature_engineer` and `expected_spend_model`).

## 2. Public Interface

### `detect_anomalies(df, zscore_threshold=3.0, pct_dev_threshold=0.5) → pd.DataFrame`
- **File**: `anomaly_detector.py`
- **Preconditions**: Requires `Col.AMOUNT_ZSCORE` and `Col.PERCENT_DEVIATION` to exist.
- **Postconditions**: Appends `Col.IS_ANOMALY` boolean column.
- **Failure modes**: Fails if columns missing.
- **Side effects**: None.

### `find_recurring_transactions(df, group_col=Col.CLEANED_REMARKS) → pd.DataFrame`
- **File**: `recurring_detector.py`
- **Preconditions**: Requires `Col.DATE`, `Col.AMOUNT`, and the group column.
- **Postconditions**: Appends `Col.IS_RECURRING` boolean column.
- **Failure modes**: Fails if grouping fails due to dtype mismatches.
- **Side effects**: None.

## 3. Internal Design
- **Key algorithms**:
    - *Anomaly Dual-Gate*: A transaction is anomalous ONLY IF it is historically unusual (`Z-score > 3.0`) AND the ML model failed to predict it accurately (`percent_deviation > 50%`). This acts as a high-precision, lower-recall filter to prevent spam.
    - *Recurring Scoring Equation*: Groups transactions by `cleaned_remarks`. Computes a bounded score `S = 0.4*A + 0.4*T + 0.2*V` where A is amount consistency, T is temporal consistency, and V is volume (count). If `S > threshold`, all items in that group are flagged as recurring.
- **Owned data structures**: Pandas column primitives.
- **State model**: Stateless.
- **Concurrency model**: Synchronous Pandas `.groupby()` and `.apply()`.
- **Non-obvious implementation decisions**:
    - The recurring detector groups strictly by `CLEANED_REMARKS`. If a subscription changes its billing reference slightly (e.g. `Netflix 1` vs `Netflix 2`), they fall into separate groups and likely fail the volume check `V`, missing the recurring flag.

## 4. Data Contracts
| Field | Type | Required | Validation Applied | Notes |
|-------|------|----------|-------------------|-------|
| `Col.IS_ANOMALY` | bool | Output | None | Sourced by dual-gate. |
| `Col.IS_RECURRING` | bool | Output | None | Sourced by group scoring. |

## 5. Dependencies
### Internal
| Subsystem | Why Used | Could It Fail? | Failure Handled? |
|-----------|----------|---------------|-----------------|
| `schema.py` | Columns | Yes | Yes. |
| `config.py` | `RECURRING_CONFIG` params | No | N/A |

### External
| Library/Service | Why Used | Could It Fail? | Failure Handled? |
|----------------|----------|---------------|-----------------|
| `pandas` / `numpy` | Grouping, standard deviation | Low | N/A |

## 6. Test Coverage Assessment
- **Tested behaviors**: Core dual-gate anomaly detection and basic recurring group calculations.
- **Untested behaviors**: Recurring logic when `amount_std` inside a group is exactly `0.0`.
- **Missing edge cases**: Time intervals bridging leap years for recurring detectors.
- **Mock/stub concerns**: Tests manually construct the upstream ML features (zscore, deviation) to isolate the logic.

## 7. Issues & Risks
- **SMELL**: `recurring_detector.py` (lines 58-61) group-by logic.
    - *What it does*: Groups by `CLEANED_REMARKS`.
    - *Why it is a problem*: If preprocessing leaves trailing numbers (e.g., date stamps in the reference), the group is shattered.
    - *Suggested fix*: Group by `PREDICTED_CATEGORY` or a fuzzier string match (or the output of a dedicated merchant resolution pass) rather than exact `CLEANED_REMARKS`.
- **RISK**: `anomaly_detector.py` (lines 20-30) dual-gate hardcoded thresholds.
    - *What it does*: Uses `zscore > 3.0` and `pct_dev > 0.5`.
    - *Why it is a problem*: Z-scores on short rolling windows (e.g. 7 days) are highly volatile. A threshold of 3.0 on a 7-day window is mathematically impossible to reach in some configurations (max z-score for N=7 is roughly 2.449).
    - *Suggested fix*: The threshold `3.0` makes sense for global populations, but not for small rolling windows. The threshold should be dynamically scaled against the `window_size`, or the feature should revert to a global Z-score.

## 8. Improvement Candidates
1. Fix the mathematical impossibility of hitting `zscore > 3.0` on small rolling windows as detailed above.
2. Allow `detect_anomalies` to flag massive dips (negative anomalies / missing expected income) rather than just spikes, by checking `abs(zscore)`.
