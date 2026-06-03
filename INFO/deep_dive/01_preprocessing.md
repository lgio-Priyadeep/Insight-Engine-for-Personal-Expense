# Subsystem: 01 Preprocessing

## Files in Scope
- `preprocessor.py` (CORE_LOGIC)
- `tests/test_phase1.py` (TEST - partially tests this subsystem)

## 1. Responsibility
- **Owns**: Translating the `raw_input` schema into the internal data model. Validating columns, parsing dates into Pandas `datetime`, normalising amounts / `DR/CR` flags, creating the core derived column `SIGNED_AMOUNT`, cleaning transaction `REMARKS` (including applying alias matching/noise reduction), removing 0-amount rows and duplicates, and finally partitioning the single dataset into two isolated streams: `debits` and `credits`.
- **Does NOT own**: Tagging transactions with specific identities (e.g. personal names), labelling categories (seed labels are deferred to stage 2), or performing aggregated calculations (e.g., finding rolling 7 day spend means).

## 2. Public Interface

### `preprocess(df) → Tuple[pd.DataFrame, pd.DataFrame]`
- **File**: `preprocessor.py`
- **Preconditions**: The input `df` must conform to `Col.raw_input()` (having date, amount, amount_flag, remarks, etc.).
- **Postconditions**: Returns `(debits, credits)` split dataframes. `date` is strongly typed. Drops exact duplicate transactions and zero-amount transactions. Both output frames are sorted by date chronologically and reindexed. Adds `CLEANED_REMARKS` and `SIGNED_AMOUNT` columns.
- **Failure modes**:
    - Raises `ValueError` if `df` is missing required columns.
    - Raises `ValueError` if `df[Col.DATE]` is completely unparseable (non-ISO 8601).
- **Side effects**: None

### `clean_remark(remark) → str`
- **File**: `preprocessor.py`
- **Preconditions**: `remark` must be a scalar (string, int, etc.).
- **Postconditions**: Returns a cleaned, lowercase, space-delimited string of tokens with special characters, 10+ digit runs, emails, and `NOISE_TOKENS` stripped.
- **Failure modes**: Returns empty string `""` for NaN or empty string inputs.
- **Side effects**: None

### `normalize(text) → str`
- **File**: `preprocessor.py`
- **Preconditions**: `text` must be a scalar.
- **Postconditions**: A strictly alphanumeric (+ spaces), lowercase version of the string, dropping special chars. Used heavily across the codebase for regex bound (`\b`) safety.
- **Failure modes**: Returns `""` on non-string input.
- **Side effects**: None

## 3. Internal Design
- **Key algorithms**:
    - *Regex Sequence Replacement*: Text cleaning applies a cascading set of pre-compiled regex mappings to resolve generic bank references (`GENERIC_ROUTER_ALIASES` like 'UPI/', 'NEFT/') and explicit named entities (`SPECIFIC_MERCHANT_ALIASES` like Uber, Netflix). Order matters: specific is run first, returning immediately. If only generic is run, it is replaced with empty space and the rest of the text is evaluated.
    - *Fallback Handling*: `amount_flag` values that are missing or invalid are silently defaulted to `DR` (Debit) to ensure the system processes them defensively rather than crashing or skipping the row.
- **Owned data structures**: Pure Pandas.
- **State model**: Stateless.
- **Concurrency model**: None (Synchronous Pandas).
- **Non-obvious implementation decisions**:
    - Deduplication uses `subset=[Col.DATE, Col.AMOUNT, Col.REMARKS, Col.AMOUNT_FLAG]` meaning if there are *actually* two identical transactions in the real bank account on the exact same date (e.g., buying two identical coffees on the same day with the same remark), the second will be silently dropped. The schema does not enforce a unique transaction ID.

## 4. Data Contracts
| Field | Type | Required | Validation Applied | Notes |
|-------|------|----------|-------------------|-------|
| `Col.DATE` | str | Yes | Parsed to datetime64[ns] | Fails hard if unparseable. |
| `Col.AMOUNT_FLAG` | str | Yes | Forced to 'DR' or 'CR' | Defaulted to 'DR' if invalid. |
| `Col.AMOUNT` | float | Yes | `coerce_and_validate_types` | 0-amount rows dropped. |
| `Col.REMARKS` | str | Yes | String conversion | Processed into `CLEANED_REMARKS`. |

## 5. Dependencies
### Internal
| Subsystem | Why Used | Could It Fail? | Failure Handled? |
|-----------|----------|---------------|-----------------|
| `schema.py` | Schema validation & core strings | Raises if columns missing. | Raises to caller. |
| `config.py` | Aliases and `NOISE_TOKENS` | Safe (Dict/Lists). | N/A |
| `logger_factory.py` | Auditing dropped rows | Exceptionally unlikely. | N/A |

### External
| Library/Service | Why Used | Could It Fail? | Failure Handled? |
|----------------|----------|---------------|-----------------|
| `pandas` | Core data transformations | Yes (e.g. dtype coercions). | Propagated up. |
| `numpy` | `np.where` for sign vectorization | No. | N/A |

## 6. Test Coverage Assessment
- **Tested behaviors**: Core end-to-end `preprocess()` functionality, handling of valid cases, and error raising on schema failures are covered in `test_phase1.py` and `test_e2e.py`.
- **Untested behaviors**: The specific mechanics of `normalize()` vs `clean_remark()` edge cases are loosely covered; e.g. what happens when multiple consecutive spaces exist, or complex generic router fallbacks.
- **Missing edge cases**: Real-world date formatting edge cases (e.g., US format `MM-DD-YYYY` vs standard `YYYY-MM-DD` when the system expects ISO 8601).
- **Mock/stub concerns**: None, Pandas runs natively.

## 7. Issues & Risks
- **RISK / SMELL**: `preprocessor.py` (lines 135-151) (`_deduplicate`).
    - *What it does*: Drops duplicate transactions based on Date, Amount, Remarks, and Flag.
    - *Why it is a problem*: If a user buys two identical items from the same store on the same day, the second charge is silently dropped. This corrupts total spend values.
    - *Suggested fix*: Modify schema to require a unique `TRANSACTION_ID` string on input and use that for deduplication, or restrict deduplication only if the file is explicitly labelled as potentially overlapping.
- **RISK**: `preprocessor.py` (lines 88-109) (`_compute_signed_amount`).
    - *What it does*: Defaults missing/invalid `AMOUNT_FLAG` rows to `DR`.
    - *Why it is a problem*: If a bank export uses `+ / -` or `Credit / Debit` instead of `CR / DR` (which is common globally), ALL transactions will become debits, ruining all insight and passion generation down the line.
    - *Suggested fix*: Attempt to infer `AMOUNT_FLAG` from `AMOUNT` sign if available, or raise a hard error instead of failing silently to default debits.

## 8. Improvement Candidates
1. `clean_remark` relies on hardcoded aliases from `config.py` via Regex. These lists (`GENERIC_ROUTER_ALIASES` and `SPECIFIC_MERCHANT_ALIASES`) are bounded now, but compiling hundreds of regexes at startup will eventually cause memory and initialization lag. Switch to a Trie-based string matcher or Aho-Corasick algorithm for performance.
2. The `_parse_and_sort_dates` explicitly calls `format="%Y-%m-%d"`. If input CSVs have timestamps (e.g., `2023-01-01T14:30:00Z`), this may fail or strip the time depending on Pandas version behaviour. Enforce a robust `pd.to_datetime(utc=True)` parse with `errors='coerce'` instead.
3. Remove the implicit dependency on the `np.where` and `abs()` usage in `_compute_signed_amount`. If `Col.AMOUNT` is already correctly signed but missing a flag, the `abs()` call actively destroys the existing signal.
