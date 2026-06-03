# Subsystem: 04 ML Inference

## Files in Scope
- `categorization_model.py` (CORE_LOGIC)
- `expected_spend_model.py` (CORE_LOGIC)
- `insight_model.py` (CORE_LOGIC)
- `model_state.py` (SCHEMA/STATE)
- `tests/test_phase2.py`, `tests/test_ml_integration.py`, `tests/test_model_security.py` (TEST)

## 1. Responsibility
- **Owns**:
    1. **Categorization**: Training a fast Logistic Regression on the fly (or using cached state) to extrapolate `PSEUDO_LABEL` categories to unseen transactions.
    2. **Expected Spend**: Training a RidgeCV regressor on the fly to predict what a transaction *should* have cost based on rolling averages and category.
    3. **Insight Ranking**: Loading a pre-trained offline LightGBM model from disk (`models/insight_ranker.pkl`) to score the final insight worthiness of a transaction.
- **Does NOT own**: Rule-based detection (done in Phase 3), or generating the initial `PSEUDO_LABEL` ground truth.

## 2. Public Interface

### `predict_categories(df, state=None) → Tuple[pd.DataFrame, Pipeline]`
- **File**: `categorization_model.py`
- **Preconditions**: Requires `Col.CLEANED_REMARKS`, `Col.AMOUNT_LOG`, and `Col.PSEUDO_LABEL` (if training).
- **Postconditions**: Appends `Col.PREDICTED_CATEGORY` and `Col.CATEGORY_CONFIDENCE`. Returns the mutated dataframe and the scikit-learn Pipeline.
- **Failure modes**: Returns fallbacks if dataframe has < 5 rows.
- **Side effects**: Trains a scikit-learn model in memory if `state` is None.

### `predict_expected_spend(df, state=None) → Tuple[pd.DataFrame, Pipeline]`
- **File**: `expected_spend_model.py`
- **Preconditions**: Requires `Col.PREDICTED_CATEGORY`, temporal, and rolling features.
- **Postconditions**: Appends `Col.EXPECTED_AMOUNT`, `Col.RESIDUAL`, `Col.PERCENT_DEVIATION`. Returns mutated df and pipeline.
- **Failure modes**: Fails over to `EXPECTED_AMOUNT = AMOUNT` if < 10 rows.
- **Side effects**: Trains RidgeCV in memory.

### `predict_insight_scores(df, model, df_context) → pd.DataFrame`
- **File**: `insight_model.py`
- **Preconditions**: `model` is a loaded scikit-learn pipeline. Requires a vast array of numeric and categorical features.
- **Postconditions**: Appends `Col.INSIGHT_SCORE`.
- **Failure modes**: Model predicts `0.0` if features are totally invalid.
- **Side effects**: None.

### `load_insight_ranker(model_path, checksum_path) → Pipeline`
- **File**: `insight_model.py`
- **Preconditions**: Files must exist on disk.
- **Postconditions**: Deserializes a pickle file safely.
- **Failure modes**: Raises `ModelSecurityError` if checksum mismatches or path traversal is attempted. Raises standard I/O errors if missing.
- **Side effects**: Disk I/O.

## 3. Internal Design
- **Key algorithms**:
    - *Categorization*: `TfidfVectorizer` on remarks + `StandardScaler` on amount -> `LogisticRegression(class_weight='balanced')`.
    - *Expected Spend*: `OneHotEncoder` on category + `StandardScaler` on rolling metrics -> `RidgeCV`.
    - *Insight Ranker*: Pre-trained offline model (LightGBM standard) executed via scikit-learn predict_proba interface.
- **Owned data structures**: `InsightModelState` (dataclass holding the two online-trained pipelines).
- **State model**:
    - Category & Spend models are **Online/Ephemeral**. They train on the current user's dataset specifically.
    - Insight model is **Offline/Global**. It is loaded from a static `.pkl` file.
- **Concurrency model**: Synchronous scikit-learn inference.
- **Non-obvious implementation decisions**:
    - The expected spend model calculates `PERCENT_DEVIATION = RESIDUAL / EXPECTED_AMOUNT`. It specifically clips `EXPECTED_AMOUNT` to a minimum of `0.01` to avoid division by zero.
    - The categorization model predicts over the whole dataset using a model trained on the *same* dataset. This is technically in-sample prediction, meaning the confidence scores for seed-labelled transactions will be artificially high.

## 4. Data Contracts
| Field | Type | Required | Validation Applied | Notes |
|-------|------|----------|-------------------|-------|
| `Col.PREDICTED_CATEGORY` | str | Output | Categorical | Extrapolated from pseudo-labels. |
| `Col.CATEGORY_CONFIDENCE`| float | Output | `[0.0, 1.0]` | From `predict_proba().max()`. |
| `Col.EXPECTED_AMOUNT` | float | Output | Numeric | Ridge regression output. |
| `Col.INSIGHT_SCORE` | float | Output | `[0.0, 1.0]` | Model probability of "insightful". |

## 5. Dependencies
### Internal
| Subsystem | Why Used | Could It Fail? | Failure Handled? |
|-----------|----------|---------------|-----------------|
| `schema.py` | Columns | Yes | Yes (raises ValueError). |

### External
| Library/Service | Why Used | Could It Fail? | Failure Handled? |
|----------------|----------|---------------|-----------------|
| `sklearn` | TF-IDF, Ridge, Logistic, Pipelines | Yes (Singular matrix, etc) | No. Unhandled sklearn exceptions will crash pipeline. |
| `joblib`/`pickle` | Serialization | Yes (Security bounds) | Yes, caught by `ModelSecurityError`. |

## 6. Test Coverage Assessment
- **Tested behaviors**: Checksum verification is explicitly tested in `test_model_security.py`. Basic pipeline fits are tested in `test_phase2.py`.
- **Untested behaviors**: Singular matrices during RidgeCV if a user has perfectly collinear rolling averages.
- **Missing edge cases**: If a single pseudo-label exists for the whole dataset, LogisticRegression cannot perform cross-validation (or stratified splitting if it were used, though it currently fits whole dataset).
- **Mock/stub concerns**: None.

## 7. Issues & Risks
- **RISK**: `categorization_model.py` (lines 53-57) and `expected_spend_model.py` (lines 52-56)
    - *What it does*: Trains `LogisticRegression` and `RidgeCV` on the user's data payload directly.
    - *Why it is a problem*: If the user provides a dataset of 6 transactions, the model tries to fit a regression with 15+ TF-IDF features and one-hot encodings. This guarantees overfitting, rank-deficient matrices, and potentially scikit-learn warnings/crashes.
    - *Suggested fix*: The current fallback check is `len(df) < 5` for Categorization and `< 10` for Spend. These thresholds are far too low for ML models with dozens of features. They should be raised to `N > 50`, falling back to rule-based logic otherwise.
- **SECURITY / SMELL**: `insight_model.py` (lines 49-74) `load_insight_ranker`.
    - *What it does*: Uses `pickle.load()` to deserialize the ML model.
    - *Why it is a problem*: `pickle` is inherently unsafe and can execute arbitrary code upon load. Even with SHA-256 validation, if the checksum file is also swapped by an attacker, RCE is possible.
    - *Suggested fix*: Migrate from `pickle` to `safetensors`, `ONNX`, or `skops.io` which do not execute arbitrary code on load.

## 8. Improvement Candidates
1. Address the `pickle` security vulnerability mentioned above.
2. In `expected_spend_model.py`, `Col.PERCENT_DEVIATION` is calculated as `residual / max(expected, 0.01)`. If `expected` is very small (e.g. 0.01), a residual of 5.0 yields a 50,000% deviation. This should be capped or smoothed (e.g. `log-deviation` or clamped to max 1000%).
