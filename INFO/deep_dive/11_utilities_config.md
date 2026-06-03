# Subsystem: 11a Configuration

## Files in Scope
- `config.py` (CONFIG)
- `config_passion.py` (CONFIG)

## 1. Responsibility
- **Owns**: Housing all business rules, ML hyperparameters, thresholds, regex alias dictionaries, and categorization keyword mappings.
- **Does NOT own**: Execution.

## 2. Internal Design
- **Key aspects**:
    - `config.py` is over 700 lines long. It contains:
        1. `SPECIFIC_MERCHANT_ALIASES`: Dictionary mapping regex to clean names (e.g., `r'uber.*eats' -> 'Uber Eats'`).
        2. `CATEGORY_KEYWORDS`: Nested dictionary mapping parent categories (e.g. `Travel`) to sets of keywords.
        3. `KNOWN_PERSON_NAMES` / `KNOWN_ACCOUNT_NUMBERS`: Lists used by identity detector.
        4. `RECURRING_CONFIG`: Dictionary of math constants for the scoring equation.
    - `config_passion.py` is isolated specifically for the passion engine, housing `PASSION_MERCHANT_ALIASES` and thresholds like `PASSION_MIN_MONTHS`.

## 3. Issues & Risks
- **SMELL**: `config.py` (lines 20-700+) is an unmanageable god-object for configuration.
    - *What it does*: Mixes ML hyperparameters (`RECURRING_CONFIG`), business logic fallbacks (`FALLBACK_DEBIT_LABEL`), and massive data dictionaries (`SPECIFIC_MERCHANT_ALIASES`).
    - *Why it is a problem*: If an analyst wants to add a new merchant alias, they must modify a core Python execution file, risking syntax errors that break the entire application.
    - *Suggested fix*: Extract data dictionaries (`ALIASES`, `KEYWORDS`, `KNOWN_PERSONS`) into JSON or YAML files, loaded dynamically at runtime. Reserve `config.py` purely for structural execution constants.
- **RISK**: Keyword Ordering.
    - As noted in `02_identity_labeling.md`, Python dictionary order matters for regex replacement. If dictionaries are unordered in older Pythons or unintentionally sorted by a formatter, the alias resolution logic breaks entirely.




# Subsystem: 11b Utilities & Logging

## Files in Scope
- `logger_factory.py` (INFRASTRUCTURE)
- `log_utils.py` (INFRASTRUCTURE)
- `hash_utils.py` (UTILITY)
- `summary_utils.py` (UTILITY)
- `bootstrap.py` (ENTRYPOINT)

## 1. Responsibility
- **Owns**:
    1. Cross-cutting concerns like logging configuration (JSON structure).
    2. Data obfuscation for logs (`log_safe_merchant`) so PII doesn't leak into Datadog/CloudWatch.
    3. Terminal UI printing (`summary_utils.py`).
    4. Model checksum hashing (`hash_utils.py`).
- **Does NOT own**: Any business logic.

## 2. Public Interface

### `get_logger(name) → logging.Logger`
- **File**: `logger_factory.py`
- Sets up standard stdout logging. Emits standard string logs, though many call-sites use the `extra` kwarg to pass metrics dictionaries.

### `log_safe_merchant(merchant_name) → str`
- **File**: `log_utils.py`
- **Preconditions**: Requires a string.
- **Postconditions**: Replaces alphanumeric characters with asterisks, preserving spaces and standard punctuation. (e.g. `John Smith` -> `**** *****`).
- **Why**: Ensures that if a pipeline error logs the `merchant_name` that failed, and that merchant name happens to actually be a person's name, it doesn't violate GDPR/CCPA in log storage.

## 3. Issues & Risks
- **RISK**: `logger_factory.py` (lines 25-35) JSON formatting is missing.
    - *What it does*: Standard `logging.StreamHandler`.
    - *Why it is a problem*: Throughout the codebase (e.g. `preprocessor.py`), developers are passing `extra={"metrics": {...}}`. The standard Python logger does not automatically serialize the `extra` dict into the log string unless explicitly formatted to do so. These metrics are currently silently vanishing.
    - *Suggested fix*: Implement a `python-json-logger` formatter in `logger_factory.py` to ensure `extra` metadata is actually output to stdout.
