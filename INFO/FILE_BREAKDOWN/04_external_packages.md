# External Package Map

## Manifest vs Reality

### Declared and Used
| Package | Declared In | Imported By Files | Scope |
|---------|------------|------------------|-------|
| `catboost` | requirements/pyproject | 1 files (model_benchmark.py) | single file |
| `joblib` | requirements/pyproject | 1 files (model_state.py) | single file |
| `lightgbm` | requirements/pyproject | 2 files (model_benchmark.py, train_and_save_models.py) | core only |
| `numpy` | requirements/pyproject | 23 files (marketplace_subcategory.py, pipeline.py, recurr...) | widespread |
| `pandas` | requirements/pyproject | 35 files (marketplace_subcategory.py, tutorial_real_data....) | widespread |
| `psutil` | requirements/pyproject | 1 files (tests/run_stress_legacy.py) | single file |
| `pytest` | requirements/pyproject | 12 files (tests/test_phase3.py, tests/test_ml_integration...) | widespread |
| `scikit-learn` | requirements/pyproject | 10 files (pipeline.py, model_benchmark.py, train_and_save...) | widespread |
| `xgboost` | requirements/pyproject | 1 files (model_benchmark.py) | single file |

### Declared but Never Imported
| Package | Declared In | Notes |
|---------|------------|-------|
| `fastparquet` | requirements/pyproject | **UNUSED** |
| `openpyxl` | requirements/pyproject | **UNUSED** |
| `pyarrow` | requirements/pyproject | **UNUSED** |
| `scipy` | requirements/pyproject | **UNUSED** |

### Imported but Not Declared
| Package | Imported By | Notes |
|---------|------------|-------|

## Package Concentration Risk
- **`pandas`**: Imported by 35 files. If this breaks, blast radius is wide.
- **`numpy`**: Imported by 23 files. If this breaks, blast radius is wide.
- **`pytest`**: Imported by 12 files. If this breaks, blast radius is wide.
- **`scikit-learn`**: Imported by 10 files. If this breaks, blast radius is wide.

## Package → File Reverse Map
| Package | Files That Import It | Symbols Used |
|---------|---------------------|-------------|
| `catboost` | `model_benchmark.py` | [CatBoostClassifier] |
| `joblib` | `model_state.py` | [module] |
| `lightgbm` | `model_benchmark.py` | [LGBMClassifier] |
| `lightgbm` | `train_and_save_models.py` | [LGBMClassifier] |
| `numpy` | `marketplace_subcategory.py` | [module] |
| `numpy` | `pipeline.py` | [module] |
| `numpy` | `recurring_detector.py` | [module] |
| `numpy` | `model_benchmark.py` | [module] |
| `numpy` | `training_data_generator.py` | [module] |
| `numpy` | `model_state.py` | [module] |
| `numpy` | `preprocessor.py` | [module] |
| `numpy` | `log_utils.py` | [module] |
| `numpy` | `feature_engineer.py` | [module] |
| `numpy` | `passion_models.py` | [module] |
| `numpy` | `passion_utils.py` | [module] |
| `numpy` | `candidate.py` | [module] |
| `numpy` | `expected_spend_model.py` | [module] |
| `numpy` | `passion_pipeline.py` | [module] |
| `numpy` | `passion_detector.py` | [module] |
| `numpy` | `tests/run_stress_heavy.py` | [module] |
| `numpy` | `tests/test_known_persons.py` | [module] |
| `numpy` | `tests/test_phase2.py` | [module] |
| `numpy` | `tests/test_passion_engine.py` | [module] |
| `numpy` | `tests/test_benchmark.py` | [module] |
| `numpy` | `tests/test_phase1.py` | [module] |
| `numpy` | `tests/test_e2e.py` | [module] |
| `numpy` | `tests/run_stress_legacy.py` | [module] |
| `pandas` | `marketplace_subcategory.py` | [module] |
| `pandas` | `tutorial_real_data.py` | [module] |
| `pandas` | `pipeline.py` | [module] |
| `pandas` | `recurring_detector.py` | [module] |
| `pandas` | `anomaly_detector.py` | [module] |
| `pandas` | `known_persons.py` | [module] |
| `pandas` | `summary_utils.py` | [module] |
| `pandas` | `demo.py` | [module] |
| `pandas` | `schema.py` | [module] |
| `pandas` | `seed_labeler.py` | [module] |
| `pandas` | `model_benchmark.py` | [module] |
| `pandas` | `insight_generator.py` | [module] |
| `pandas` | `categorization_model.py` | [module] |
| `pandas` | `training_data_generator.py` | [module] |
| `pandas` | `insight_model.py` | [module] |
| `pandas` | `preprocessor.py` | [module] |
| `pandas` | `log_utils.py` | [module] |
| `pandas` | `feature_engineer.py` | [module] |
| `pandas` | `passion_utils.py` | [module] |
| `pandas` | `pipeline_result.py` | [module] |
| `pandas` | `expected_spend_model.py` | [module] |
| `pandas` | `passion_pipeline.py` | [module] |
| `pandas` | `passion_detector.py` | [module] |
| `pandas` | `tests/run_stress_heavy.py` | [module] |
| `pandas` | `tests/test_phase3.py` | [module] |
| `pandas` | `tests/test_ml_integration.py` | [module] |
| `pandas` | `tests/test_known_persons.py` | [module] |
| `pandas` | `tests/test_phase2.py` | [module] |
| `pandas` | `tests/test_passion_engine.py` | [module] |
| `pandas` | `tests/test_benchmark.py` | [module] |
| `pandas` | `tests/test_logging_safety.py` | [module] |
| `pandas` | `tests/test_phase1.py` | [module] |
| `pandas` | `tests/test_e2e.py` | [module] |
| `pandas` | `tests/run_smoke.py` | [module] |
| `pandas` | `tests/run_stress_legacy.py` | [module] |
| `psutil` | `tests/run_stress_legacy.py` | [module] |
| `pytest` | `tests/test_phase3.py` | [module] |
| `pytest` | `tests/test_ml_integration.py` | [module] |
| `pytest` | `tests/test_known_persons.py` | [module] |
| `pytest` | `tests/conftest.py` | [module] |
| `pytest` | `tests/test_phase2.py` | [module] |
| `pytest` | `tests/test_passion_engine.py` | [module] |
| `pytest` | `tests/test_benchmark.py` | [module] |
| `pytest` | `tests/test_model_security.py` | [module] |
| `pytest` | `tests/test_logging_safety.py` | [module] |
| `pytest` | `tests/test_phase1.py` | [module] |
| `pytest` | `pipeline.py` | [module] |
| `pytest` | `passion_pipeline.py` | [module] |
| `scikit-learn` | `pipeline.py` | [Pipeline] |
| `scikit-learn` | `model_benchmark.py` | [recall_score, StandardScaler, classification_report, precision_score, accuracy_score, AdaBoostClassifier, OneHotEncoder, LogisticRegression, ColumnTransformer, f1_score, KNeighborsClassifier, DecisionTreeClassifier, CalibratedClassifierCV, RandomForestClassifier, ExtraTreesClassifier, StratifiedKFold, MLPClassifier, LabelEncoder, LinearSVC, GradientBoostingClassifier, Pipeline, cross_validate, confusion_matrix] |
| `scikit-learn` | `train_and_save_models.py` | [ColumnTransformer, Pipeline, StandardScaler, OneHotEncoder] |
| `scikit-learn` | `categorization_model.py` | [Pipeline, StandardScaler, LogisticRegression, ColumnTransformer, TfidfVectorizer] |
| `scikit-learn` | `training_data_generator.py` | [train_test_split] |
| `scikit-learn` | `insight_model.py` | [Pipeline] |
| `scikit-learn` | `model_state.py` | [Pipeline] |
| `scikit-learn` | `expected_spend_model.py` | [RidgeCV, Pipeline, StandardScaler, OneHotEncoder, ColumnTransformer] |
| `scikit-learn` | `tests/test_ml_integration.py` | [Pipeline] |
| `scikit-learn` | `tests/test_benchmark.py` | [Pipeline, StandardScaler, OneHotEncoder, LogisticRegression, ColumnTransformer] |
| `xgboost` | `model_benchmark.py` | [XGBClassifier] |
