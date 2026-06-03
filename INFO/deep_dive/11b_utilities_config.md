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
