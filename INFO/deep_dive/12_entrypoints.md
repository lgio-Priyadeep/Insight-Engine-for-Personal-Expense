# Subsystem: 12 Entrypoints

## Files in Scope
- `demo.py` (ENTRYPOINT)
- `tutorial_real_data.py` (ENTRYPOINT)
- `refactor_pipeline.py` (lines 1-140) (ENTRYPOINT - likely deprecated)

## 1. Responsibility
- **Owns**: Consumer-facing scripts that invoke the `pipeline.py` orchestration logic. Designed to be run directly by humans or cron jobs.
- **Does NOT own**: Internal pipeline logic.

## 2. Public Interface

### `demo.py`
- Runs the pipeline against a hardcoded `scrubbed.csv` file.
- Disables model training (uses rule-based fallbacks) or expects `models/insight_ranker.pkl` to exist.
- Prints the output using `summary_utils.py`.

### `tutorial_real_data.py`
- Exposes a CLI-like interface to accept arbitrary CSV files.
- Demonstrates how to handle the `PipelineResult` dataclass.

## 3. Issues & Risks
- **SMELL**: `refactor_pipeline.py` (lines 1-140).
    - This file appears to be an orphaned script from a previous architectural migration. It contains duplicated or incomplete pipeline logic.
    - *Suggested fix*: Delete the file if it's no longer used to avoid confusing new engineers.
