# Subsystem: 10 Schemas & Contracts

## Files in Scope
- `schema.py` (SCHEMA)
- `contracts.py` (SCHEMA)

## 1. Responsibility
- **Owns**:
    1. Centralizing all column name strings into a single `Col` class to prevent typos.
    2. Defining the required input schema (`Col.raw_input()`).
    3. Type-coercing the raw input dataframe into strict Pandas dtypes.
    4. Housing the Insight generation text templates (`INSIGHT_TEMPLATES`) and advice corpus (`TIP_CORPUS`).
- **Does NOT own**: Execution logic.

## 2. Public Interface

### `Col` Class
- **File**: `schema.py`
- Holds static string constants (e.g. `Col.DATE = 'date'`, `Col.CLEANED_REMARKS = 'cleaned_remarks'`).

### `coerce_and_validate_types(df) → pd.DataFrame`
- **File**: `schema.py`
- **Preconditions**: DataFrame must have the raw input columns.
- **Postconditions**: `amount` is `float64`, `remarks` is `string`. Drops rows with `NaN` in critical columns (`amount`, `remarks`, `date`).
- **Failure modes**: None (drops silently instead of raising).

### `INSIGHT_TEMPLATES` & `TIP_CORPUS`
- **File**: `contracts.py`
- Dictionaries mapping signal types (anomaly, recurring, passion) to human-readable format strings.

## 3. Internal Design
- **Key algorithms**:
    - `contracts.py` uses an internal `_freeze_insight_templates` function to deeply freeze the nested dictionary structures using `frozenset` and `MappingProxyType`, guaranteeing that downstream pipeline stages cannot accidentally mutate the global templates.
- **State model**: Immutable constants.

## 4. Issues & Risks
- **RISK**: `coerce_and_validate_types` (lines 154-159) drops rows silently.
    - *What it does*: `df = df.dropna(subset=[Col.DATE, Col.AMOUNT, Col.REMARKS])`
    - *Why it is a problem*: If a bank feed has a glitch and drops the amount column for half the rows, the pipeline will just silently analyze the remaining half, leading to wildly inaccurate financial advice.
    - *Suggested fix*: If the number of dropped rows exceeds 0, log a critical warning. If it exceeds some percentage (e.g., 5%), raise a `ValueError` aborting the pipeline.
