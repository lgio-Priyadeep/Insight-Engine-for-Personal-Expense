<![CDATA[# Decision Log & Design Rationale

> *Why things are the way they are. Read this before proposing changes.*

This document records the significant design decisions made during development, the alternatives we considered, and why we chose what we chose. It exists so that future developers don't re-litigate settled debates or accidentally revert decisions that were made for good reasons.

---

## Decision 1: Regex-Based Merchant Aliasing

**Date**: March 2026  
**Context**: Bank statement `remarks` fields contain raw UPI routing metadata mixed with merchant names. We needed a way to extract "Swiggy" from `"UPI/9812345678/swiggy@paytm/OrderID-3847593"`.

### Options Considered

| Option | Pros | Cons |
|---|---|---|
| **Regex patterns** (chosen) | Deterministic, auditable, zero false positives for known patterns, easy to extend | Requires manual pattern curation, won't match truly novel merchants |
| Fuzzy string matching (Levenshtein) | Handles typos, can match unknown merchants | High false positive rate in finance (confuses "Airtel" with "Airbnb"), non-deterministic |
| NLP entity extraction (NER) | Generalizes to unknown merchants | Requires training data we don't have, adds heavy dependencies (spaCy/transformers), overkill for structured remark fields |
| Exact string matching | Simplest implementation | Fails on any variation ("swiggy" vs "SWIGGY" vs "Swiggy Instamart") |

### Decision
Regex. The financial domain cannot tolerate false positives. If "CRED" is fuzzily matched to "credit", the user's CRED payment gets categorized as a generic credit — destroying the insight. Regex gives us zero-ambiguity matching with full control.

### Consequence
We must manually curate the merchant list. Currently 200+ Indian merchants are covered. Unmapped merchants fall through to standard text cleaning and may be categorized by the ML model.

---

## Decision 2: `shift(1)` for Rolling Feature Leakage Prevention

**Date**: March 2026  
**Context**: Rolling mean/std/sum features are standard in time-series ML. But naively applying `.rolling()` to a DataFrame lets row *i*'s training target leak into its own features.

### The Problem

```python
# This is WRONG — row 5's rolling mean includes row 5's value
df["rolling_mean"] = df["amount"].rolling(7).mean()
```

In production, when processing a new transaction, you don't know the current transaction's amount yet — it's what you're trying to predict. If the model was trained with leaky features, it learned to "cheat" by relying on the current value being in the rolling window, and it will fail catastrophically when that value is absent.

### Decision
Apply `shift(1)` **before** `.rolling()`:

```python
shifted = df["amount"].shift(1)
df["rolling_mean"] = shifted.rolling(7, min_periods=1).mean()
```

Row *i*'s rolling window now contains only rows `0..i-1`. Row 0's window is empty (NaN). This is correct.

### Trade-off
Row 0 (and early rows) have NaN features that must be filled. We fill with externally provided training-set statistics (`global_mean`, `global_std`). This is a minor information loss but eliminates a major correctness bug.

---

## Decision 3: RidgeCV Over Random Forest for Expected Spend

**Date**: March 2026  
**Context**: We need a model that predicts "how much would this user normally spend on this type of transaction?" to establish an anomaly baseline.

### Why Not Random Forest?

Tree-based models (Random Forest, Gradient Boosting, XGBoost) **cannot extrapolate**. They partition the feature space into rectangles and assign the mean of training targets within each rectangle. The predicted value is bounded by `[min(y_train), max(y_train)]`.

**Example**: If the maximum training transaction is ₹500, and a user spends ₹2,500:
- Random Forest predicts: ≤ ₹500 (the anomaly is invisible)
- Ridge predicts: ~₹2,500 linearly extrapolated (the anomaly is detected)

This is verified by `test_phase2.py::test_expected_spend_model_extrapolates`, which explicitly asserts that the model predicts >₹500 when given ₹1,000 rolling means against a training max of ₹42.

### Decision
RidgeCV (linear regression with built-in cross-validated regularization). Linear models extrapolate naturally. The alpha is auto-tuned from `[0.1, 1.0, 10.0, 100.0]`.

### Trade-off
Ridge cannot capture non-linear patterns (e.g., "weekends have 2x spending but only in certain months"). If this becomes a problem, consider Gradient Boosting with a fallback linear extrapolation layer.

---

## Decision 4: Dual-Gate Anomaly Detection

**Date**: March 2026  
**Context**: A single z-score threshold produces too many false positives for low-value transactions.

### The Problem with Single-Gate

| Transaction | Z-Score | Should Flag? |
|---|---|---|
| ₹30 chai (usual ₹20) | 3.5 | **No** — ₹10 is trivial |
| ₹2,500 Amazon (usual ₹120) | 8.2 | **Yes** — massive spike |

Both exceed a z-score threshold of 3.0, but only one is genuinely concerning.

### Decision
Require **both** conditions:
- `|amount_zscore| > 3.0` (statistically unusual)
- `|percent_deviation| > 0.5` (ML model says this is 50%+ above expected baseline)

The percent deviation gate adds **contextual significance**. A ₹10 increase on ₹20 is 50% but the z-score alone might flag it. The expected spend model, which considers category, time, and history, provides a more holistic "this doesn't look right" signal.

### Trade-off
The dual gate may *miss* anomalies that are modest in absolute terms but significant relative to the user's financial situation. This is acceptable — false negatives are less harmful than false positives in spending advice.

---

## Decision 5: Heuristic-Based Recurring Detection (Not ML)

**Date**: March 2026  
**Context**: We need to identify subscriptions (Netflix, Spotify, gym memberships) from transaction history.

### Why Not ML?

- **No labeled training data**: We don't have a dataset of "this is a subscription" / "this is not"
- **Explainability requirement**: Users need to trust the detection. "Flagged because 3 occurrences at 30-day intervals with <5% amount drift" is verifiable. "ML model says 87% probability" is not.
- **Simplicity**: The heuristic is 82 lines. An ML approach would require feature engineering, training, evaluation, serialization, and maintenance.

### Decision
Rule-based detection:
1. Group by `cleaned_remarks`
2. Require ≥2 occurrences
3. Check time-gap consistency (monthly: 27–33 days, weekly: 6–8 days)
4. Check amount stability (≤5% drift)

### Known Limitation
Only monthly and weekly frequencies are detected. Biweekly, quarterly, and annual subscriptions are missed. Extending is straightforward (add new gap ranges) and should be done when the need arises.

---

## Decision 6: Separate Debit and Credit Pipelines

**Date**: March 2026  
**Context**: Should we process debits and credits through the same keyword map and ML model?

### Decision
No. They are separated at the preprocessing stage and use entirely different:
- Keyword maps (`CATEGORY_KEYWORDS` vs. `CREDIT_KEYWORDS`)
- Priority orders (`CATEGORY_PRIORITY` vs. `CREDIT_PRIORITY`)
- Fallback labels (`"uncategorized"` vs. `"other_credit"`)

### Rationale
A "salary" is a credit keyword. If run through the debit map, it hits nothing and becomes "uncategorized." A "refund" is semantically a credit concept. Mixing them would require every keyword map to handle both directions, creating ambiguity and bloat.

---

## Decision 7: `class_weight='balanced'` in Logistic Regression

**Date**: March 2026  
**Context**: Transaction categories are inherently imbalanced. "Food" and "shopping" dominate; "health" and "entertainment" are rare.

### The Problem
Without balancing, the model learns to predict "food" for everything (high training accuracy, useless in practice). Minority classes get zero predictions.

### Decision
`class_weight='balanced'` inversely weights each class by its frequency. Rare categories get higher loss penalties, forcing the model to learn them.

### Trade-off
May slightly reduce accuracy on the majority class ("food"). This is acceptable — correctly identifying a rare health expense is more valuable than slightly better food classification.

---

## Decision 8: Excluding Fallback Labels from Training

**Date**: March 2026  
**Context**: Should the ML model train on rows labeled `"uncategorized"` or `"other_credit"`?

### Decision
No. These rows are excluded before training.

### Rationale
Fallback labels carry **no signal** — they mean "the keyword map didn't match anything." Including them would:
1. Teach the model that "uncategorized" is a valid prediction, reducing useful predictions
2. Pollute the feature space with noise (the remarks that got fallback labels are by definition unusual/unknown)
3. Counteract `class_weight='balanced'` by adding a large, meaningless class

---

## Decision 9: `sparse_threshold=1.0` in ColumnTransformer

**Date**: March 2026  
**Context**: TF-IDF produces a sparse matrix. StandardScaler produces a dense array. When ColumnTransformer combines them, it may convert the sparse TF-IDF output to dense, causing massive memory allocation.

### Decision
Set `sparse_threshold=1.0` to force the output to remain sparse. Also use `StandardScaler(with_mean=False)` to allow the numeric features to stay in sparse format (centering requires dense representation).

### Trade-off
`with_mean=False` means the scaled numeric features are not zero-centered. For Logistic Regression with regularization, this has negligible impact.

---

## Decision 10: Defensive `.copy()` Everywhere

**Date**: March 2026  
**Context**: Should functions modify DataFrames in-place for performance, or create copies for safety?

### Decision
Copy. Always. Every function.

### Rationale
In-place modification is a constant source of subtle bugs in pandas:

```python
debits = preprocess(raw)[0]
labeled = label_debits(debits)

# If label_debits() mutated in-place, `debits` now has pseudo_label too.
# Three weeks later, someone adds code between these lines that assumes
# `debits` doesn't have pseudo_label. Silent data corruption ensues.
```

The performance cost of `.copy()` is negligible for the dataset sizes this pipeline handles (thousands of rows, not millions). The debugging cost of not copying is unbounded.

---

*If you make a significant design decision while working on this project, add it here. Future you — and future teammates — will thank you.*
]]>
