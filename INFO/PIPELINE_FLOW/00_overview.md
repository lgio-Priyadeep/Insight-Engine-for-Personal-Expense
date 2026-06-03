# Pipeline Overview

## Entry Points
| Name | File | Trigger | Input |
|------|------|---------|-------|
| Real Data Tutorial | `tutorial_real_data.py` | `if __name__ == "__main__":` block | CSV/JSON/Parquet file path |
| Demo Pipeline | `demo.py` | Direct script execution | `scrubbed.csv` (Hardcoded path) |
| Core Pipeline | `pipeline.py` | `run_pipeline(raw_df)` function | pandas DataFrame (`raw_df`) |
| Passion Pipeline | `passion_pipeline.py` | `process_pipeline(df_raw)` via `_attach_passion_results` | pandas DataFrame (`debits`) |
| Model Trainer | `train_and_save_models.py` | `if __name__ == "__main__":` calling `train_and_save()` | None (Generates synthetic data) |
| Model Benchmark | `model_benchmark.py` | `if __name__ == "__main__":` calling `main()` | None (Generates synthetic data) |

## Pipeline Index
| # | Name | Entry Point | Stage Count | File |
|---|------|-------------|-------------|------|
| 1 | Insight Engine Core | `pipeline.py` | 7 | `01_insight_engine.md` |
| 2 | Passion Engine | `passion_pipeline.py` | 6 | `02_passion_engine.md` |
| 3 | Model Trainer | `train_and_save_models.py` | 4 | `03_model_trainer.md` |
| 4 | Model Benchmark | `model_benchmark.py` | 4 | `04_model_benchmark.md` |
| 5 | Real Data Tutorial | `tutorial_real_data.py` | 4 | `05_tutorial.md` |
| 6 | Demo Pipeline | `demo.py` | 5 | `06_demo.md` |
| 7 | Core execution | `pipeline.py` | 7 | `07_pipeline.md` |

## External Surface
| Type | Identifier | Access Pattern | Auth |
|------|-----------|---------------|------|
| File System | Input Data Files | Read (CSV/JSON/Parquet via pandas) | None |
| File System | `models/insight_ranker.pkl` | Write (Model Trainer) / Read (Core) | None |
| File System | `models/insight_ranker.pkl.sha256` | Write (Model Trainer) / Read (Core) | None |
| File System | Crash Dumps | Write (on pipeline failure if enabled) | None |
| Environment | `INSIGHT_ENGINE_PASSION_ENABLED` | Read | None |
| Environment | `INSIGHT_ENGINE_PASSION_MAX_ROWS` | Read | None |
| Environment | `INSIGHT_ENGINE_PASSION_STRICT_ATTACH` | Read | None |
| Environment | `INSIGHT_ENGINE_CRASH_TEST` | Read | None |
| Environment | `ENV` | Read | None |
| Environment | `INSIGHT_ENGINE_SECRET` | Read | None |
| Environment | `INSIGHT_ENGINE_SKIP_STARTUP_CHECKS`| Read | None |

## Shared Code Segments
- **Pre-processing and Labeling (`schema.py`, `preprocessor.py`, `seed_labeler.py`)**: Used by the Core Pipeline, Real Data Tutorial, Demo Pipeline, and multiple tests.
- **Model Inference (`categorization_model.py`, `expected_spend_model.py`, `insight_model.py`)**: Shared across Core Pipeline and Model Trainer/Benchmark contexts (as imported functionality).
- **Synthetic Data Generation (`training_data_generator.py`)**: Shared by Model Trainer and Model Benchmark.

## Failure Analysis Summary
| Pipeline | Stage | Failure Type | Handled? | Blast Radius |
|----------|-------|-------------|----------|--------------|
| Core Pipeline | Core Execution | Unhandled exceptions | Yes (Logged, crash dump) | Complete pipeline failure, but writes snapshot data to disk. |
| Passion Engine | Subcategory Enrichment | General exceptions | Yes (Fail-fast in soft mode) | Returns neutral passion result, downstream passion steps skipped. Core pipeline unaffected. |
| Passion Engine | Passion Detection | General exceptions | Yes (Fail-fast in soft mode) | Returns neutral passion result. Core pipeline unaffected. |
| Passion Engine | Insight Generation | General exceptions | Yes (Fail-fast in soft mode) | Returns neutral passion result. Core pipeline unaffected. |
| Core Pipeline | Passion Attachment | Exception in Passion Engine | Yes (Swallowed by `_attach_passion_results`) | Passion insights missing, but Core pipeline succeeds. |
