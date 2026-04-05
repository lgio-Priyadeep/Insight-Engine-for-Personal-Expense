<![CDATA[# Contributing & Developer Guide

> *How to work on this codebase without breaking it for everyone else.*

---

## Before You Write a Single Line

1. **Read [`ARCHITECTURE.md`](ARCHITECTURE.md)**. Not skim — read. The data flow contract and leakage prevention strategy are non-negotiable constraints.
2. **Run the full test suite**. If tests fail before your changes, you have a broken environment — fix that first.
3. **Understand the column lifecycle**. Know which module creates each column and which modules depend on it.

---

## Setting Up Your Environment

```bash
# Clone and enter the project
git clone <your-repo-url>
cd "Insight engine"

# Create a virtual environment (Python 3.10+ required)
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install pandas numpy scikit-learn pytest

# Verify everything works
pytest -v
python test_e2e.py
```

---

## The Golden Rules

### 1. Never Mutate Input DataFrames

Every function that receives a DataFrame must call `.copy()` before modifying it. This is not a suggestion — it's the pipeline's core invariant.

```python
# ✅ Correct
def my_function(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["new_col"] = compute_something(df)
    return df

# ❌ Wrong — mutates the caller's DataFrame
def my_function(df: pd.DataFrame) -> pd.DataFrame:
    df["new_col"] = compute_something(df)
    return df
```

**Why?** Because DataFrames are passed by reference. If you mutate the input, every variable that points to the same DataFrame is silently corrupted. This creates bugs that are invisible at the point of mutation and catastrophic at the point of use.

### 2. Never Introduce Data Leakage

If you add a new feature:

- Does it use the current row's value in its own computation? → **Leakage**. Use `shift(1)`.
- Does it use test-set statistics? → **Leakage**. Accept external parameters from the caller.
- Does it peek ahead in the time series? → **Leakage**. Ensure chronological sort + shift.

### 3. Always Add Tests

No untested code. The test suite catches regressions that manual testing misses.

- Unit test for the new function
- Edge case test (empty input, None, single-row DataFrame)
- Input immutability test (assert the input DataFrame's columns haven't changed)
- If it's a feature: NaN-safety test (assert no NaN in output)

### 4. Configuration Changes Go in `config.py` Only

If you need to add a merchant, category, keyword, or threshold:

- Add it to `config.py`
- Do not hardcode domain knowledge in module code
- Do not create new config files — there is one source of truth

---

## Code Style

### General

- **Type hints** on all function signatures (Python 3.10+ style: `dict[str, str]` not `Dict[str, str]`)
- **Docstrings** on all public functions (Google-style or numpy-style, be consistent within a file)
- **Structured logging** via `logging.getLogger(__name__)` — never `print()`
- **Defensive guards** at function entry (validate inputs, check types, raise early)

### Naming Conventions

| Entity | Convention | Example |
|---|---|---|
| Public function | `snake_case` | `engineer_features()` |
| Private function | `_snake_case` | `_normalize_flag()` |
| Constants | `UPPER_SNAKE_CASE` | `FALLBACK_DEBIT_LABEL` |
| Module-level compiled regex | `_UPPER_PATTERN` | `_LONG_DIGIT_PATTERN` |
| DataFrame column | `snake_case` | `cleaned_remarks`, `rolling_7d_mean` |

### Logging Levels

| Level | When to Use |
|---|---|
| `logger.info()` | Pipeline progress milestones ("Preprocessing complete. 50 debits, 12 credits") |
| `logger.warning()` | Recoverable oddities (invalid flags, low coverage, auto-computed stats) |
| `logger.debug()` | Verbose detail (individual subscription detection, per-row matching) |
| `logger.error()` | Not currently used — failures raise exceptions instead |

---

## Adding a New Module

If you're adding a new analysis stage (e.g., a savings recommender, budget forecaster):

### 1. Create the Module

```python
"""
savings_recommender.py — Monthly Savings Opportunity Estimator
==============================================================
[What this module does, why it exists, what it depends on.]
"""

import logging
import pandas as pd

logger = logging.getLogger(__name__)


def recommend_savings(df: pd.DataFrame, ...) -> pd.DataFrame:
    """
    [Full docstring with Args, Returns, Raises]
    """
    df = df.copy()  # NEVER SKIP THIS
    # ... your logic ...
    return df
```

### 2. Add Tests

Create `test_savings.py` with:

- Happy path test
- Edge case tests (empty DF, single row, missing columns)
- Input immutability test
- Output schema test (correct columns exist, correct types)

### 3. Document It

- Add to the [ARCHITECTURE.md](ARCHITECTURE.md) data flow diagram
- Add to the [API_REFERENCE.md](API_REFERENCE.md) function listing
- Add to the project structure in [README.md](../README.md)
- Update the column lifecycle table

### 4. Integrate into E2E

Add a call to your module in `test_e2e.py` to ensure it works in the full pipeline context.

---

## Adding a New Merchant (Most Common Change)

This is the most frequent type of contribution. No code changes needed.

1. Open `config.py`
2. Add a regex pattern to `MERCHANT_ALIASES`:
   ```python
   r"newmerchant":  "New Merchant",
   ```
3. Add the lowercased merchant name to the appropriate category in `CATEGORY_KEYWORDS`:
   ```python
   "food": [
       "swiggy", "zomato", ..., "new merchant",
   ],
   ```
4. Run tests: `pytest test_phase1.py -v`

**Tips for regex patterns**:
- Use `\b` word boundaries to prevent false positives: `r"ola\b"` won't match "scholarship"
- Use `\s?` for optional spaces: `r"pizza\s?hut"` matches both "PizzaHut" and "Pizza Hut"
- Use `'?s?` for possessives: `r"haldiram'?s?"` matches "Haldiram", "Haldirams", "Haldiram's"

---

## Pull Request Checklist

Before submitting:

- [ ] All existing tests pass: `pytest -v`
- [ ] E2E test passes: `python test_e2e.py`
- [ ] New tests added for new functionality
- [ ] No `.copy()` violations (input DataFrames not mutated)
- [ ] No data leakage introduced (rolling features use shift(1), NaN fills use external stats)
- [ ] Domain knowledge in `config.py`, not hardcoded in modules
- [ ] Structured logging used (no `print()`)
- [ ] Docstrings present on all public functions
- [ ] Type hints present on all function signatures
- [ ] Documentation updated (ARCHITECTURE.md, API_REFERENCE.md, README.md)

---

## Troubleshooting Common Development Issues

### `ModuleNotFoundError: No module named 'config'`

You're running from the wrong directory. The pipeline expects to be run from the project root:

```bash
cd "Insight engine"
pytest -v
```

### `TypeError: 'date' column must be datetime64`

You called `add_time_features()` or `engineer_features()` without running `preprocess()` first. The preprocessor converts `date` from string to datetime. Always run `preprocess()` first.

### Tests pass locally but fail in CI

Check your Python version. The codebase uses `dict[str, str]` (Python 3.10+), not `Dict[str, str]` (older `typing` style). CI must use Python 3.10 or later.

### `SettingWithCopyWarning`

This usually means you're modifying a DataFrame slice without `.copy()`. Ensure you're working on a copy:

```python
df = df.copy()  # Add this before any modification
```

---

*The goal is not to write perfect code. The goal is to write code that the next person can understand, trust, and extend without fear.*
]]>
