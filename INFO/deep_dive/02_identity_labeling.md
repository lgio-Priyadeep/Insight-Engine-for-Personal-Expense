# Subsystem: 02 Identity & Labeling

## Files in Scope
- `known_persons.py` (DATA_LAYER)
- `seed_labeler.py` (DATA_LAYER)
- `tests/test_known_persons.py` (TEST)

## 1. Responsibility
- **Owns**:
    1. Identifying personal or self-transfers (via predefined aliases and account numbers in config) inside transaction remarks (`known_persons.py`).
    2. Tagging recognized transactions to prevent them from skewing predictive models and aggregating recurring self-transfers into distinct insights.
    3. Performing the first pass of categorization (`seed_labeler.py`), assigning deterministic "pseudo-labels" (e.g., Food, Travel) based on regex keyword matching from `config.py`.
- **Does NOT own**: Machine learning predictions (handled by Category/Spend models in Phase 2). The seed labels generated here are the "ground truth" used *by* the ML models if state isn't loaded, or used as fallback predictions.

## 2. Public Interface

### `tag_known_persons(df) → pd.DataFrame`
- **File**: `known_persons.py`
- **Preconditions**: Requires `Col.CLEANED_REMARKS` and `Col.REMARKS`. Reads lists of names and accounts from `config.py`.
- **Postconditions**: Returns DataFrame with `Col.IS_KNOWN_PERSON` (boolean), `Col.KNOWN_PERSON_NAME` (string/NaN).
- **Failure modes**: Swallows regex compilation errors (silently drops the pattern).
- **Side effects**: None.

### `detect_personal_patterns(df) → List[str]`
- **File**: `known_persons.py`
- **Preconditions**: Requires a DataFrame already passed through `recurring_detector` and `tag_known_persons`.
- **Postconditions**: Returns string insight templates regarding recurring personal transfers (e.g. "You transfer £X regularly to Y").
- **Failure modes**: Handled gracefully.
- **Side effects**: None.

### `label_debits(df) → pd.DataFrame` & `label_credits(df) → pd.DataFrame`
- **File**: `seed_labeler.py`
- **Preconditions**: Requires `Col.CLEANED_REMARKS`. Reads priorities and keywords from `config.py`.
- **Postconditions**: Assigns `Col.PSEUDO_LABEL`. Generates debug metadata columns (`Col.PSEUDO_LABEL_REASON`, etc.). Fallbacks to 'Other Spend' / 'Other Income' if no match.
- **Failure modes**: Raises `ValueError` if columns are missing.
- **Side effects**: Logs coverage metrics (warnings if too many fallbacks).

## 3. Internal Design
- **Key algorithms**:
    - *Regex Compilations & Bounds*: `seed_labeler.py` transforms all keywords in `config.py` into word-bounded regexes (`\bkeyword\b`).
    - *Multi-Tier Scoring*: `known_persons.py` uses a bespoke string matching algorithm. It looks for exact name matches, account number fragments, and concatenated names (e.g., `JOHN SMITH` → `JOHNSMITH`). It uses a scoring threshold `score >= 3` to classify a match, where different signal types (transfer contexts like 'ref', partial names, accounts) provide varying points.
- **Owned data structures**: `SignalBundle` and `CompiledKeyword` Dataclasses to represent internal state during matching.
- **State model**: Stateless, driven entirely by `config.py` mappings.
- **Concurrency model**: Synchronous Pandas `.apply()`.
- **Non-obvious implementation decisions**:
    - The `known_persons.py` logic explicitly has an invariant: "Double Scoring: Overlap between exact partial match and concat partial match gives +2 score on the same word component." This means names like "Smith" in a remark "Transfer SmithSmith" might artificially clear the threshold.

## 4. Data Contracts
| Field | Type | Required | Validation Applied | Notes |
|-------|------|----------|-------------------|-------|
| `Col.IS_KNOWN_PERSON` | bool | Output | Created | Used heavily downstream as a mask. |
| `Col.KNOWN_PERSON_NAME` | str/NaN | Output | Created | Attached to insights. |
| `Col.PSEUDO_LABEL` | str | Output | Created | The definitive label used for modelling. |

## 5. Dependencies
### Internal
| Subsystem | Why Used | Could It Fail? | Failure Handled? |
|-----------|----------|---------------|-----------------|
| `config.py` | Sources all names, account nums, and category keywords. | Yes, if empty lists exist. | Handled gracefully. |

### External
| Library/Service | Why Used | Could It Fail? | Failure Handled? |
|----------------|----------|---------------|-----------------|
| `pandas` | Map / Apply ops | Low | N/A |

## 6. Test Coverage Assessment
- **Tested behaviors**: Core matching of known persons vs merchants. Seed label prioritization (ensuring a tier 1 keyword overrides a tier 2 keyword).
- **Untested behaviors**: Boundary conditions in `known_persons.py` relating to exact scoring math limits (e.g., what if a merchant has a personal name *and* an account fragment overlaps?).
- **Missing edge cases**: Seed labeler does not handle negations (e.g., "Not Uber" would be labelled as Uber).
- **Mock/stub concerns**: Tests rely heavily on the real `config.py` dictionary. If `config.py` changes, the tests in `test_phase1.py` break immediately.

## 7. Issues & Risks
- **SMELL**: `seed_labeler.py` (lines 43-48) `_compile_keywords` function.
    - *What it does*: Iterates over lists of tuples to build `\b` regexes.
    - *Why it is a problem*: Keyword overlaps (e.g. `car` vs `car insurance`) are entirely dependent on iteration order in `config.py`. If `config.py` uses unordered dicts, or someone sorts them alphabetically, precedence rules break.
    - *Suggested fix*: Explicitly sort the keyword dictionary by length descending during `_compile_keywords` so that `car insurance` is always evaluated before `car`.
- **RISK**: `known_persons.py` (lines 201-224) `_score_remark`.
    - *What it does*: Sums arbitrary signal points to hit a `score >= 3`.
    - *Why it is a problem*: Highly heuristic. Changing any noise token or suffix in `config.py` could cascade into thousands of false positives.
    - *Suggested fix*: Introduce a regression test specifically for a large static dataset of known merchants to ensure none ever score >= 3.

## 8. Improvement Candidates
1. `seed_labeler.py` uses a pure Pandas `.apply(_match_remark, axis=1)` over strings, executing regex hundreds of times per row. For 50k transactions, this is severely CPU-bound. Refactor using `Series.str.contains(regex)` vectorized operations instead.
2. The known persons `_MERCHANT_INDICATOR_TOKENS` inside `known_persons.py` (lines 28-29) duplicate generic alias concepts from `preprocessor.py`. Consolidate these lists into `config.py` to ensure uniform updates.
