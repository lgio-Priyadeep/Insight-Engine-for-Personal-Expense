# Subsystem: 06 Insight Generation

## Files in Scope
- `insight_generator.py` (CORE_LOGIC)

## 1. Responsibility
- **Owns**: Translating boolean flags (`IS_ANOMALY`, `IS_RECURRING`) and numeric values (`INSIGHT_SCORE`, `AMOUNT`) into human-readable text strings using templates from `contracts.py`.
- **Does NOT own**: The business logic determining *if* something is an anomaly or recurring. It merely renders the text.

## 2. Public Interface

### `generate_human_insights(df, min_score=0.7, max_insights=3, rng=None) → List[str]`
- **File**: `insight_generator.py`
- **Preconditions**: Requires a fully populated dataframe (all ML flags, predicted categories, and cleaned remarks). `INSIGHT_SCORE` must exist.
- **Postconditions**: Returns a list of strings, capped at `max_insights`. Insights are ordered descending by `INSIGHT_SCORE`.
- **Failure modes**: Swallows string formatting errors (drops the specific insight and logs a warning). Returns empty list if no rows exceed `min_score`.
- **Side effects**: None.

## 3. Internal Design
- **Key algorithms**:
    - *Score Thresholding & Sorting*: Filters `df` to `INSIGHT_SCORE >= min_score`, sorts descending, and iterates.
    - *Template Filling*: Identifies the primary signal (`IS_ANOMALY` vs `IS_RECURRING`). Looks up the template from `INSIGHT_TEMPLATES`, formats it with Pandas row data (`merchant`, `amount`, `category`), and appends a tip.
    - *Seeded Random Tip Selection*: Uses `lookup_matching_tip_ids` and a seeded random number generator (`rng`) to select a piece of advice (a "tip") appended to the insight.
- **Owned data structures**: Pure string manipulation.
- **State model**: Stateless, purely determined by input `df` and `rng` seed.
- **Concurrency model**: Synchronous row iteration.
- **Non-obvious implementation decisions**:
    - If a transaction is BOTH an anomaly and recurring, it currently defaults to the `anomaly` template branch first.

## 4. Data Contracts
| Field | Type | Required | Validation Applied | Notes |
|-------|------|----------|-------------------|-------|
| (Inputs) | Various | Yes | Implicit via iteration | Relies on all prior pipeline steps succeeding. |
| (Outputs) | List[str] | Output | None | Ready for presentation. |

## 5. Dependencies
### Internal
| Subsystem | Why Used | Could It Fail? | Failure Handled? |
|-----------|----------|---------------|-----------------|
| `contracts.py` | `TIP_CORPUS`, `INSIGHT_TEMPLATES` | Yes (missing keys) | Caught, warning logged, tip omitted. |

### External
| Library/Service | Why Used | Could It Fail? | Failure Handled? |
|----------------|----------|---------------|-----------------|
| `pandas` | Row iteration | No | N/A |

## 6. Test Coverage Assessment
- **Tested behaviors**: Generating insights, obeying the `max_insights` cap, filtering by `min_score`, handling missing tips gracefully.
- **Untested behaviors**: Formatting edge cases (e.g. `amount` being `NaN` or infinity).
- **Missing edge cases**: Extremely long merchant names breaking UX constraints (no truncation applied here).
- **Mock/stub concerns**: Tests pass static `rng` seeds to ensure reproducible tip selection.

## 7. Issues & Risks
- **RISK**: `generate_human_insights` (lines 62-67) iteration logic.
    - *What it does*: Sorts by score and takes the top N.
    - *Why it is a problem*: If an anomaly and a recurring subscription both have a score of 0.99, and the user has 3 subscriptions and 1 anomaly, the top 3 insights might *all* be subscriptions, drowning out the high-priority anomaly. There is no diversity constraint on the output.
    - *Suggested fix*: Implement grouped stratification. Take max 1 anomaly, max 1 recurring, etc., before filling the rest by raw score.

## 8. Improvement Candidates
1. Add diversity constraints to the insight output (as noted above).
2. Format the currency amounts properly. Currently, it just prints the float (e.g. `£45.123456`), which looks terrible in a UI. Format to `.2f`.
