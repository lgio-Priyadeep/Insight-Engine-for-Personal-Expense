# Subsystem: 03 Feature Engineering

## Files in Scope
- `feature_engineer.py` (CORE_LOGIC)

## 1. Responsibility
- **Owns**: Enhancing the core transactions with derived statistical and temporal features needed by the downstream machine learning models (Categorization, Expected Spend, Insight Scoring). Crucially, owns preventing "target leakage" during rolling aggregations.
- **Does NOT own**: Any ML inference or data cleaning. Expects a fully cleaned dataset as input.

## 2. Public Interface

### `engineer_features(df, global_amount_mean, global_amount_std, df_context) → pd.DataFrame`
- **File**: `feature_engineer.py`
- **Preconditions**: `df` must be sorted chronologically and contain `Col.DATE` and `Col.SIGNED_AMOUNT`. `global_amount_mean` and `std` must be provided (to prevent data leakage on the first row).
- **Postconditions**: Returns DataFrame with added columns: `IS_WEEKEND`, `WEEK_OF_MONTH`, `MONTH_SIN/COS`, `DOW_SIN/COS`, `ROLLING_7D_MEAN/STD`, `ROLLING_30D_MEAN/STD`, `AMOUNT_LOG`, `AMOUNT_ZSCORE`.
- **Failure modes**:
    - Raises `ValueError` if required columns missing.
    - Raises `ValueError` if `df` is empty.
- **Side effects**: None (operates on a copy).

## 3. Internal Design
- **Key algorithms**:
    - *Cyclical Time Encodings*: Uses sine/cosine transforms on months and days to prevent the model from seeing December (12) and January (1) as far apart.
    - *Leak-Free Rolling Windows*: Uses `.shift(1).rolling(window).mean()`. This ensures that the calculation for Row 10 only sees data from Row 0 to 9, never its own amount.
    - *Z-Score Clipping*: Z-scores are clamped to `[-5, 5]` to prevent short-window variance explosions from crashing models.
- **Owned data structures**: Pure Pandas DataFrames.
- **State model**: Stateless per transaction. Dependent on provided global states (`global_amount_mean`).
- **Concurrency model**: None (Vectorized Pandas).
- **Non-obvious implementation decisions**:
    - The rolling window is row-based (`window=7`), NOT time-based (`window='7D'`). This means it looks at the last 7 *transactions*, regardless of if they happened in 1 day or 3 months.

## 4. Data Contracts
| Field | Type | Required | Validation Applied | Notes |
|-------|------|----------|-------------------|-------|
| `Col.SIGNED_AMOUNT` | float | Yes | Used for all rolling features | Must be present. |
| (Various Derived) | float/int | Output | None | Mapped to ML features. |

## 5. Dependencies
### Internal
| Subsystem | Why Used | Could It Fail? | Failure Handled? |
|-----------|----------|---------------|-----------------|
| `schema.py` | Column names and validation | Yes (missing cols) | Raised to caller. |

### External
| Library/Service | Why Used | Could It Fail? | Failure Handled? |
|----------------|----------|---------------|-----------------|
| `pandas` | `.rolling`, `.shift` | Low | N/A |
| `numpy` | `np.sin`, `np.cos`, `np.log1p` | Low | N/A |

## 6. Test Coverage Assessment
- **Tested behaviors**: Tested implicitly in `tests/test_phase1.py` and `test_e2e.py`. Evaluates if columns are generated and no NaNs remain.
- **Untested behaviors**: Checking if the Z-score calculation properly handles 0-standard-deviation windows (where division by zero occurs) without emitting `inf`.
- **Missing edge cases**: Timezones are ignored. Dates are treated as abstract intervals.
- **Mock/stub concerns**: None.

## 7. Issues & Risks
- **SMELL**: Rolling window definition (`rolling(window=7)` (lines 92, 98)).
    - *What it does*: Calculates mean/std of the last N *rows*.
    - *Why it is a problem*: A user who makes 10 transactions a day will have a `ROLLING_30D_MEAN` that covers 3 days. A user making 1 transaction a week will have a mean covering 7 months. This severely confuses time-series models.
    - *Suggested fix*: Change to Pandas temporal rolling windows: `rolling('7D', on=Col.DATE)` after ensuring the date column is properly indexed.

## 8. Improvement Candidates
1. Address the row-based vs time-based rolling window issue described above.
2. The `AMOUNT_ZSCORE` can divide by zero if the rolling std is exactly 0.0 (e.g., 7 identical transactions). Verify that `numpy` `inf` behavior is explicitly caught and coerced to `0.0`.
