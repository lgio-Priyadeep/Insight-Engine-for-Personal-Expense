# Subsystem: 08 Pipeline Orchestration

## Files in Scope
- `pipeline.py` (ENTRYPOINT)
- `pipeline_result.py` (SCHEMA)

### 9.1 Responsibility
- **Owns**:
    1. Being the top-level executor for the entire Insight Engine.
    2. Passing data safely between preprocessing, modeling, detection, and insight generation stages.
    3. Managing the lifecycle of ML models (training them if not provided, passing them around if they are).
    4. Orchestrating the "soft fail" attachment of the Passion Engine results.
    5. Writing diagnostic snapshots (Crash Dumps) if the pipeline fails catastrophically.
- **Does NOT own**: The internal logic of any specific stage.

## 2. Public Interface

### `run_pipeline(raw_df, state=None, generate_insights=True, strict_mode=False) → PipelineResult`
- **File**: `pipeline.py`
- **Preconditions**: `raw_df` must be a Pandas DataFrame matching the raw schema.
- **Postconditions**: Returns a fully populated `PipelineResult` dataclass.
- **Failure modes**:
    - Raises `ValueError` if the data is fundamentally invalid (e.g., completely unparseable).
    - If a crash occurs and `INSIGHT_ENGINE_CRASH_TEST` is enabled, dumps a Parquet snapshot before raising.
- **Side effects**: Heavy CPU/Memory usage. May perform disk I/O to load the Insight Ranker model or write crash dumps.

### `train_models(spend_debits) → InsightModelState`
- **File**: `pipeline.py`
- **Preconditions**: Needs clean `spend_debits` with seed labels.
- **Postconditions**: Returns the trained ML pipelines bundled in `InsightModelState`.
- **Failure modes**: Propagates scikit-learn training errors.

## 3. Internal Design
- **Key algorithms**:
    - *Masked Execution*: The pipeline extensively uses Boolean masks (e.g., `spend_mask`, `personal_mask`) to avoid splitting and joining dataframes constantly. E.g., Categorization only runs on rows where `IS_KNOWN_PERSON == False`.
    - *Memory Optimization*: Explicitly calls `_optimize_memory_footprint` (lines 710-725) to downcast float64 -> float32 and int64 -> int32 before returning, significantly reducing the memory footprint of the final object.
    - *Passion Soft-Attachment*: Calls `_attach_passion_results` (lines 665-685) in a broad `try...except Exception`. If it fails, it logs an error but ensures the core pipeline continues and returns valid core data.
- **Owned data structures**: `PipelineResult` (Dataclass grouping dataframes and strings).
- **State model**: Stateful if `state` is not provided (trains models), Stateless if `state` is injected.
- **Concurrency model**: Purely synchronous.
- **Non-obvious implementation decisions**:
    - The pipeline explicitly avoids deleting the `raw_df` from memory so that it can attach the raw metadata back into the final payload if needed, meaning memory usage peaks at >2x the input size.

## 4. Data Contracts
| Field | Type | Required | Validation Applied | Notes |
|-------|------|----------|-------------------|-------|
| `raw_df` | pd.DataFrame | Yes | Schema validation | Passed into Stage 1. |
| `PipelineResult` | Dataclass | Output | None | Contains `debits`, `credits`, `insights`, `state`. |

## 5. Dependencies
### Internal
| Subsystem | Why Used | Could It Fail? | Failure Handled? |
|-----------|----------|---------------|-----------------|
| (All subsystems) | Execution | Yes | Often unhandled in core pipeline. |
| `passion_pipeline.py` | Secondary analysis | Yes | Explicitly caught and swallowed. |

### External
| Library/Service | Why Used | Could It Fail? | Failure Handled? |
|----------------|----------|---------------|-----------------|
| `pandas` | Everything | High | Propagates |

## 6. Test Coverage Assessment
- **Tested behaviors**: End-to-end execution, logging safety, stress limits.
- **Untested behaviors**: The `_dump_crash_snapshot` function is difficult to trigger cleanly in unit tests without mocking the internal OS state.
- **Missing edge cases**: Extremely sparse dataframes (e.g. 5 valid rows, 95 invalid rows) might pass the early `len(df) > 5` checks but fail deep inside ML inference.

## 7. Issues & Risks
- **RISK / SMELL**: Exception handling in `_attach_passion_results` (lines 665-685).
    - *What it does*: `except Exception as e:` and swallows it entirely.
    - *Why it is a problem*: Catching `Exception` catches `MemoryError`, `KeyboardInterrupt` (in some older pythons if not strictly `BaseException`), and `SystemExit`. If the passion engine triggers an OOM killer or a user hits Ctrl+C, the pipeline tries to swallow it and continue.
    - *Suggested fix*: Change to a more specific list of exception types (e.g., `ValueError`, `KeyError`, `pd.errors.PandasError`), or explicitly exclude standard terminal exceptions.
- **RISK**: `_optimize_memory_footprint` (lines 710-725).
    - *What it does*: Downcasts floats and ints.
    - *Why it is a problem*: Downcasting `float64` to `float32` destroys precision. For large currency values or ML deviations, this precision loss can flip boolean thresholds downstream or cause discrepancies in database synchronization.
    - *Suggested fix*: Only downcast boolean and categorical columns. Leave `Col.AMOUNT` and expected spend as `float64`.

## 8. Improvement Candidates
1. Fix the precision loss in memory optimization by skipping currency columns.
2. Refine the exception handling around the Passion Engine attachment.
3. Streamline the masks. Passing `spend_debits` into models, getting it back, and writing it back into `debits` using `.loc[spend_mask]` is computationally expensive compared to a simpler split-apply-combine.


## 9. Testing Subsystem

### Files in Scope
- `tests/test_e2e.py` (TEST)
- `tests/run_smoke.py` (TEST)
- `tests/run_stress_heavy.py` (TEST)
- `tests/run_stress_legacy.py` (TEST)
- `tests/run_tests_legacy.py` (TEST)

### 9.1 Responsibility
- **Owns**: Verifying that the entire orchestration loop completes without crashing, handles varied edge-case synthetic data, and respects performance / memory bounds.
- **Does NOT own**: The granular verification of specific algorithms (e.g. TF-IDF math).

### 9.2 Test Architecture
- **E2E Tests** (`test_e2e.py`): The primary regression suite. Uses small, explicitly crafted dataframes designed to trigger specific branches (e.g., triggering known persons, recurring, and anomalies in a single 10-row dataset).
- **Smoke Tests** (`run_smoke.py`): A fast sanity check that simply runs the pipeline on a static CSV file and asserts the output object is a valid `PipelineResult`.
- **Stress Tests** (`run_stress_heavy.py`, `run_stress_legacy.py`): Generates massive synthetic datasets (e.g. 100,000 rows) and runs the pipeline while tracking time and memory.

### 9.3 Findings & Risks
- **RISK**: Legacy Test Suites.
    - `run_stress_legacy.py` and `run_tests_legacy.py` appear to use outdated import paths or schemas. Having multiple overlapping definitions of "stress" tests leads to CI confusion.
    - *Actionable*: Deprecate and remove the `_legacy.py` files if `test_e2e.py` and `run_stress_heavy.py` supersede them.
- **SMELL**: `test_e2e.py` assertion depth.
    - The E2E tests verify that columns *exist* (e.g. `assert Col.PREDICTED_CATEGORY in df.columns`), but rarely assert the *correctness* of the output values for complex ML interactions.
    - *Actionable*: Add golden-path fixture files where the exact expected output dataframe is compared against the generated output dataframe using `pd.testing.assert_frame_equal`.

### 9.4 Performance Profile
Based on the stress tests, the pipeline is highly memory-bound. A 100k row dataframe will copy itself approximately 4 times during preprocessing, feature engineering, and inference, leading to peak memory usage easily exceeding 1GB for what should be a 10MB text file.
