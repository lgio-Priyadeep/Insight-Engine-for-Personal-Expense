# Deep Dive Index

## Subsystem Registry
| # | Name | Files Included | Purpose |
|---|------|---------------|---------|
| 01 | Preprocessing | `preprocessor.py`, `tests/test_phase1.py` | Data normalization and cleaning. |
| 02 | Identity & Labeling | `known_persons.py`, `seed_labeler.py`, `tests/test_known_persons.py` | Tagging recurring entities and assigning initial categories. |
| 03 | Feature Engineering | `feature_engineer.py` | Enriching transaction data with aggregations and patterns. |
| 04 | ML Inference | `categorization_model.py`, `expected_spend_model.py`, `insight_model.py`, `model_state.py`, `tests/test_phase2.py`, `tests/test_ml_integration.py`, `tests/test_model_security.py` | Evaluating models to predict categories, spend, and insight scores. |
| 05 | Signal Detection | `anomaly_detector.py`, `recurring_detector.py`, `tests/test_phase3.py` | Rule-based engines finding anomalies and recurring transfers. |
| 06 | Insight Generation | `insight_generator.py` | Producing human-readable financial insights from processed signals. |
| 07a | Passion Engine (Core) | `passion_pipeline.py`, `passion_utils.py`, `tests/test_passion_engine.py` | Orchestration and shared utilities for detecting hobbies and interests. |
| 07b | Passion Engine (Detection) | `passion_detector.py`, `passion_insight_generator.py`, `marketplace_subcategory.py`, `passion_models.py`, `candidate.py`, `banned_content.py` | Logic for evaluating marketplace spend and creating specific passion insights. |
| 08 | Pipeline Orchestration | `pipeline.py`, `pipeline_result.py`, `tests/test_e2e.py`, `tests/run_smoke.py`, `tests/run_stress_heavy.py`, `tests/run_stress_legacy.py`, `tests/run_tests_legacy.py` | The central conductor coordinating all insights engine stages and its end-to-end regression and stress tests. |
| 09a | Training & Data Generation | `training_data_generator.py`, `train_and_save_models.py`, `tests/test_benchmark.py` | Synthesizing data and fitting production ML models. |
| 09b | Model Benchmarking | `model_benchmark.py` | Running exhaustive algorithm comparisons for the core models. |
| 10 | Schemas & Contracts | `schema.py`, `contracts.py` | Core data dictionaries, column definitions, and template text. |
| 11a | Configuration | `config.py`, `config_passion.py` | Global constants, parameters, rulesets, and thresholds. |
| 11b | Utilities & Logging | `log_utils.py`, `logger_factory.py`, `summary_utils.py`, `hash_utils.py`, `bootstrap.py`, `tests/test_logging_safety.py`, `tests/conftest.py` | Cross-cutting infrastructure for logging, hashing, and bootstrapping. |
| 12 | Entrypoints | `demo.py`, `tutorial_real_data.py`, `refactor_pipeline.py` | Consumer scripts running the engine top-to-bottom. |

## Reading Order
For a new engineer joining the project, the recommended reading path is:
1. **10 Schemas & Contracts**: Understand `schema.py` and what data looks like.
2. **11a Configuration**: See how thresholds and rules are managed.
3. **08 Pipeline Orchestration**: Read `pipeline.py` to grasp the high-level execution flow.
4. **01 Preprocessing & 02 Identity**: See how raw data is prepared and labelled.
5. **03 Feature Engineering & 04 ML Inference**: Understand the core predictive logic.
6. **07a/07b Passion Engine**: Dive into the secondary analysis subsystem.

## High-Risk Subsystems
- **08 Pipeline Orchestration**: `pipeline.py` is nearly 1,000 lines long, orchestrating heavy Pandas dataframe transformations and exception swallowing (`_attach_passion_results`). Memory and performance regressions are likely here.
- **11a Configuration**: `config.py` is massive (700+ lines) and combines pure configurations with hardcoded keyword mappings and alias dictionaries. A typo here breaks seed labeling directly.
- **07a Passion Engine (Core)**: Huge footprint. High complexity for conditional branching depending on strict vs. soft error handling. `tests/test_passion_engine.py` is exceptionally long (2.4K lines).
