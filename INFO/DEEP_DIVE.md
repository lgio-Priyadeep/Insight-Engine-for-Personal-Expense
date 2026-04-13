# Insight Engine — Deep Dive: Complex Architecture Components

> **Scope**: All non-obvious design decisions, tricky algorithms, and security mechanisms.
> **Audience**: Engineers who need to modify, extend, or audit this code.

---

## Table of Contents

1. [Data Leakage Prevention System](#1-data-leakage-prevention-system)
2. [Merchant Alias Resolution Pipeline](#2-merchant-alias-resolution-pipeline)
3. [Keyword Priority Tiebreaking Algorithm](#3-keyword-priority-tiebreaking-algorithm)
4. [Recurring Transaction Scoring Equation](#4-recurring-transaction-scoring-equation)
5. [Dual-Gate Anomaly Detection](#5-dual-gate-anomaly-detection)
6. [Insight Diversity Ranking Algorithm](#6-insight-diversity-ranking-algorithm)
7. [Model Security Architecture](#7-model-security-architecture)
8. [Inference vs. Training Pipeline Divergence](#8-inference-vs-training-pipeline-divergence)
9. [Synthetic Data Generator Design](#9-synthetic-data-generator-design)
10. [PII Scrubbing Strategy](#10-pii-scrubbing-strategy)
11. [Schema Contract Enforcement](#11-schema-contract-enforcement)
12. [Memory Optimization Strategies](#12-memory-optimization-strategies)
13. [Observability Architecture](#13-observability-architecture)
14. [Identified Technical Debt & Risks](#14-identified-technical-debt--risks)

---

## 1. Data Leakage Prevention System

**Location**: `feature_engineer.py`, lines 74–110 (training) and 246–303 (inference)

### The Problem

In time-series financial data, data leakage occurs when features computed for transaction at time `t` contain information from transactions at time `t` or later. This inflates training metrics and causes catastrophic performance degradation in production.

### The Solution: Two-Layer Defense

**Layer 1 — Training-Time Shift Gate (`engineer_features`)**

```python
# feature_engineer.py:89–91
shifted = df["amount"].shift(1)
rolling_7d_mean = shifted.rolling(window=7, min_periods=1).mean()
```

The `shift(1)` call moves every value down by one row. Row `i`'s rolling window now only contains values from rows `0` through `i−1`. The current row's value is NEVER visible to its own features.

**Mathematical proof of correctness**:
```
Original:  [100, 200, 300, 400, 500]
Shifted:   [NaN, 100, 200, 300, 400]

Row 0: rolling(NaN)       = NaN    ← no prior data (correct)
Row 1: rolling(100)       = 100    ← sees only row 0 (correct)
Row 2: rolling(100, 200)  = 150    ← sees rows 0,1 (correct)
Row 3: rolling(100,200,300) = 200  ← sees rows 0,1,2 (correct)
None of these see their own row's value.
```

**Layer 2 — Inference-Time History Stitching (`engineer_features_inference`)**

```python
# feature_engineer.py:268–275
new_txns["__is_new_txn__"] = True
history["__is_new_txn__"] = False
combined = pd.concat([history, new_txns])
combined = combined.sort_values("date").reset_index(drop=True)
```

At inference, new transactions are merged with historical data. Rolling features are computed on the combined set (with shift(1)), then only new-transaction rows are extracted via the tag column. This guarantees:
- New transaction features reflect the ACTUAL historical pattern
- No future-looking information is included
- Backdated transactions are handled correctly (the tag column, not position, determines extraction)

**Why NOT use `tail(n)`?** Tag-based extraction handles backdated transactions. If a new transaction's date is earlier than some historical rows, a naive `tail()` would return the wrong rows.

### NaN Filling Strategy

Early rows lack sufficient history for rolling windows. The pipeline fills these NaNs with **training-set global statistics** (`global_mean`, `global_std`), NOT with test-set or inference-set statistics. This is stored and reused across the training/inference boundary via `InsightModelState`.

```python
# feature_engineer.py:131–143
df[Col.ROLLING_7D_MEAN] = df[Col.ROLLING_7D_MEAN].fillna(global_mean)
df[Col.ROLLING_7D_STD]  = df[Col.ROLLING_7D_STD].fillna(safe_std)
# safe_std = max(abs(global_std), 1.0)  ← prevents div-by-zero downstream
```

---

## 2. Merchant Alias Resolution Pipeline

**Location**: `preprocessor.py`, lines 144–204; `config.py`, lines 15–223

### The Problem

Bank statement remarks are highly inconsistent. `"UPI/9876543210/SWIGGY ORDER 42"`, `"POS SWIGGY-DLY"`, and `"SWIGGY BLR"` should all resolve to `"swiggy"`.

### The Solution: Two-Tier Alias System

**Tier 1 — Specific Merchant Patterns** (early return):
```python
# If regex matches AND the alias is NOT a generic router:
#   Return the canonical merchant name immediately.
#   Example: r"swiggy" matches → return "swiggy"
```

**Tier 2 — Generic Router Patterns** (strip and continue):
```python
# If regex matches BUT the alias IS a generic router (UPI, Paytm, PhonePe, etc.):
#   Strip the matched router text from the remark.
#   Continue to standard cleanup.
#   Example: "UPI/1234/RajuTeaStall" → strip "UPI" → clean → "rajuteastall"
```

### How Generic vs. Specific is Determined

```python
# preprocessor.py:171
# A pattern is "generic" if its canonical name matches one of:
_GENERIC_ROUTERS = {"upi transfer", "paytm", "phonepe", "google pay", "bhim", ...}
```

### Dictionary Order Dependency

The iteration order of `MERCHANT_ALIASES` matters. Python 3.7+ guarantees insertion-order iteration. The config defines specific patterns (like "swiggy") BEFORE generic routers (like "upi"). If a remark contains both `"swiggy"` and `"upi"`, "swiggy" matches first and returns.

**Risk**: If `MERCHANT_ALIASES` is ever loaded from a database or JSON without preserving order, this behavior could change. The code does not explicitly enforce ordering.

### Pre-Compilation for Performance

```python
# preprocessor.py:39–43
_COMPILED_ALIASES = [
    (re.compile(pattern, re.IGNORECASE), canonical_name)
    for pattern, canonical_name in MERCHANT_ALIASES.items()
]
```

Regex compilation happens once at import time. This avoids re-compiling ~150 patterns for every single transaction.

---

## 3. Keyword Priority Tiebreaking Algorithm

**Location**: `seed_labeler.py`, lines 75–106

### The Problem

A remark like `"emi hospital bill ola cab"` matches keywords from three categories:
- `"emi"` → finance (HIGH priority)
- `"hospital"` → health (HIGH priority)
- `"cab"` → transport (MEDIUM priority)

Which category wins?

### The Algorithm

```
Step 1: Collect ALL keyword matches
  matches = [
    CompiledKeyword(text="emi",      category="finance",   priority=3, confidence=0.90),
    CompiledKeyword(text="hospital", category="health",    priority=3, confidence=0.90),
    CompiledKeyword(text="cab",      category="transport", priority=2, confidence=0.70),
  ]

Step 2: Find maximum priority
  max_priority = 3

Step 3: Filter to only max-priority matches
  same_tier = [
    CompiledKeyword(text="emi",      priority=3),
    CompiledKeyword(text="hospital", priority=3),
  ]

Step 4: Sort by (keyword_length DESC, keyword_text ASC)
  sorted = [
    CompiledKeyword(text="hospital", len=8),  ← longer wins
    CompiledKeyword(text="emi",      len=3),
  ]

Step 5: Return first element
  → category="health", keyword="hospital"
```

**Why longest keyword?** Longer keywords are more specific. `"cash withdrawal"` (14 chars) is more precise than `"cash"` (4 chars).

**Why alphabetical tiebreak?** Determinism. Without it, same-length keywords could produce different results across runs due to hash-map ordering.

### Priority Tiers

```python
TIER_MAPPING = {
    "HIGH":   {"priority": 3, "confidence": 0.90},  # finance, health, utilities
    "MEDIUM": {"priority": 2, "confidence": 0.70},  # food, transport, shopping, entertainment
    "LOW":    {"priority": 1, "confidence": 0.50},  # atm, transfer
}
```

---

## 4. Recurring Transaction Scoring Equation

**Location**: `recurring_detector.py`, lines 21–117

### The Problem

Identify subscriptions (Netflix ₹15.99/month) and recurring bills (electricity ~₹800/month) from raw transaction data, without any external subscription database.

### The Scoring Model

For each merchant group with ≥3 transactions:

**1. Amount Score (A)** — How consistent are the amounts?
```
amount_drift = max(|amount_i − mean_amount|) / mean_amount  ∀ i
A = clamp(1.0 − (amount_drift / amount_tolerance), 0.0, 1.0)
```
Where `amount_tolerance = 0.20` (20% max variation allowed).

**Example**:
```
Amounts: [15.99, 15.99, 15.99]
mean = 15.99, max_deviation = 0.0
drift = 0.0, A = 1.0 (perfect)

Amounts: [800, 820, 780]
mean = 800, max_deviation = 20
drift = 20/800 = 0.025, A = 1.0 − (0.025/0.20) = 0.875
```

**2. Temporal Score (T)** — Do the time gaps match a known frequency?
```python
RECURRING_CONFIG = {
    "monthly":   {"min_gap": 27, "max_gap": 33},
    "weekly":    {"min_gap": 6,  "max_gap": 8},
    "biweekly":  {"min_gap": 13, "max_gap": 16},
    "quarterly": {"min_gap": 85, "max_gap": 95},
}
```

```
mean_gap = mean of time deltas between consecutive transactions (in days)
If mean_gap ∈ [min_gap, max_gap] for any frequency:
    gap_variance = std(gaps) / mean_gap
    T = clamp(1.0 − (gap_variance / allowed_variance), 0.0, 1.0)
    assigned_frequency = that frequency
Else:
    T = 0.0, assigned_frequency = None
```

**3. Volume Score (V)** — How many occurrences?
```
V = clamp(n_occurrences / 12, 0.0, 1.0)
```
12 = a full year of monthly billing = maximum confidence.

**4. Final Composite Score**:
```
score = 0.4 × A + 0.4 × T + 0.2 × V
```

**5. Rejection Gate**:
```python
if T == 0.0 or A == 0.0:
    # Reject entirely — not recurring
    # T=0 means doesn't match any known frequency
    # A=0 means amounts are wildly inconsistent
```

**6. Confidence Penalty** (amount_drift > 10%):
```python
if amount_drift > penalty_threshold:  # 0.10
    confidence = confidence * (1.0 - amount_drift)
```

---

## 5. Dual-Gate Anomaly Detection

**Location**: `anomaly_detector.py`, lines 16–48

### The Problem

A ₹10 coffee might have a z-score of 4.0 if you usually buy ₹5 coffees. That's statistically unusual but financially irrelevant. Conversely, a ₹5000 purchase when your ML model expected ₹4800 has a low z-score but a meaningful deviation — but it's still within normal range.

### The Solution: Two Independent Gates

```
is_anomaly = (|amount_zscore| > Z_THRESHOLD) AND (|percent_deviation| > D_THRESHOLD)
```

**Gate 1 — Statistical Z-Score** (from rolling window):
```
z = (amount − rolling_7d_mean) / rolling_7d_std
Threshold: |z| > 3.0
```
This catches transactions that deviate from the user's recent spending pattern.

**Gate 2 — ML Deviation** (from RidgeCV expected spend model):
```
deviation = (actual − expected) / max(|expected|, 1.0)
Threshold: |deviation| > 0.5
```
This catches transactions that deviate from what the ML model predicts based on temporal features, category, and rolling statistics.

Both must fire. This prevents:
- Low-value statistical outliers from triggering alerts (Gate 2 filters them)
- High-value but expected purchases from triggering alerts (Gate 1 filters them)

### Why RidgeCV for Expected Spend (Not Tree-Based)?

`expected_spend_model.py` uses RidgeCV specifically because it can **extrapolate** beyond the training range. Tree-based models (RandomForest, LightGBM) cap predictions at the maximum training value. If a user's purchases suddenly double, a tree model wouldn't predict beyond the historical max. Ridge regression will, enabling meaningful anomaly detection for out-of-distribution amounts.

This design choice is validated in `test_phase2.py:test_expected_spend_model_extrapolates()`.

---

## 6. Insight Diversity Ranking Algorithm

**Location**: `insight_generator.py`, lines 118–155

### The Problem

If the top 5 insights by ML score are all spending spikes, the user gets a repetitive output. They should see a mix of insight types.

### The Solution: Type-First Diversity Pass

```
All candidates: [(type, text, score), ...]
Sorted by score descending

Pass 1 — Type Seeding:
  seen_types = set()
  selected = []
  for candidate in candidates:
      if candidate.type not in seen_types:
          selected.append(candidate)
          seen_types.add(candidate.type)
  # After pass 1: At most ONE of each type, always the highest-scoring

Pass 2 — Score Fill:
  remaining = [c for c in candidates if c not in selected]
  for candidate in remaining:
      if len(selected) < top_n:
          selected.append(candidate)

Final: Sort selected by score descending
```

**Example**:
```
Candidates (sorted by score):
  1. spending_spike  "₹2500 at Amazon"     score=0.95
  2. spending_spike  "₹1800 at Flipkart"   score=0.88
  3. subscription    "Netflix ₹16/month"    score=0.85
  4. spending_spike  "₹900 at Croma"        score=0.72
  5. subscription    "Spotify ₹149/month"   score=0.65

Pass 1 (type seeding):
  → spending_spike: "₹2500 at Amazon" (0.95)
  → subscription: "Netflix ₹16/month" (0.85)

Pass 2 (fill to top_n=5):
  → spending_spike: "₹1800 at Flipkart" (0.88)
  → spending_spike: "₹900 at Croma" (0.72)
  → subscription: "Spotify ₹149/month" (0.65)

Final (sorted by score):
  1. ₹2500 at Amazon (0.95)
  2. ₹1800 at Flipkart (0.88)
  3. Netflix ₹16/month (0.85)
  4. ₹900 at Croma (0.72)
  5. Spotify ₹149/month (0.65)
```

---

## 7. Model Security Architecture

**Location**: `insight_model.py`, lines 40–154

### Threat Model

The pre-trained LightGBM model is loaded via `pickle.load()`. Pickle deserialization is a known Remote Code Execution (RCE) vector. An attacker who swaps the model file can execute arbitrary code.

### Three-Layer Defense

**Layer 1 — Path Validation** (`_validate_model_path`):
```python
canonical = os.path.realpath(filepath)
models_dir = os.path.realpath(_MODELS_DIR)
if not canonical.startswith(models_dir):
    raise ModelSecurityError("Path resolves outside the allowed models/ directory")
```
- Prevents path traversal (`../../etc/passwd`)
- Resolves symlinks before checking (prevents symlink-based bypass)
- All model loading is constrained to the `models/` directory

**Layer 2 — SHA-256 Checksum Verification** (`_verify_checksum`):
```python
computed = _compute_checksum(model_path)    # SHA-256 of actual file bytes
expected = open(checksum_path).read().strip()  # companion .sha256 file
if computed != expected:
    raise ModelSecurityError("integrity check FAILED")
```
- Model file must have a companion `.sha256` file written at training time
- If the model file is modified after training, the checksum won't match
- Missing checksum file → model rejected gracefully (returns None)

**Layer 3 — Graceful Degradation**:
```python
# If no model file exists: return None → fallback to rule-based scoring
# If checksum missing: return None → refuse to load unsigned model
# If checksum fails: raise ModelSecurityError → hard stop
```

### Security Test Coverage

`test_model_security.py` validates:
- Checksum determinism and hex format
- Tampered file detection
- Path traversal rejection
- Symlink resolution and rejection
- Missing checksum rejection
- Valid signed model acceptance
- Missing file graceful handling

---

## 8. Inference vs. Training Pipeline Divergence

**Location**: `pipeline.py`, lines 93–258

### The Problem

Training (`run_pipeline`) and inference (`run_inference`) share 90% of the same logic but diverge in critical ways. Getting these divergences wrong causes the most insidious production bugs (training/serving skew).

### Divergence Map

| Step | Training (`run_pipeline`) | Inference (`run_inference`) |
|------|---------------------------|-----------------------------|
| **Preprocessing** | Identical | Identical |
| **Seed Labeling** | Identical | Identical |
| **Global Stats** | Computed from current data | Loaded from `InsightModelState` |
| **Feature Engineering** | `engineer_features(df, gm, gs)` | `engineer_features_inference(new, history, gm, gs)` |
| **Model Training** | Trains cat, spend, ranker | **Skipped entirely** |
| **Categorization** | Train + Predict | Predict only (pre-trained pipeline) |
| **Expected Spend** | Train + Predict | Predict only (pre-trained pipeline) |
| **Anomaly/Recurring** | Identical | Identical |
| **Insight Scoring** | Identical | Identical |
| **Output** | PipelineResult with trained models | PipelineResult with pre-trained models |

### Critical Design Decision: Feature Engineering Divergence

Training uses `engineer_features()` which operates on the complete dataset (features are computed over all history).

Inference uses `engineer_features_inference()` which stitches new transactions into a historical window. This ensures rolling features for new transactions are computed from real history, not just the new transactions in isolation.

Without this, a single new transaction would have:
- `rolling_7d_mean = NaN` (no prior rows)
- `rolling_7d_std = NaN`
- `amount_zscore = NaN`

These would be filled with `global_mean` — the average of ALL historical amounts — which is a much weaker signal than the actual recent trend.

---

## 9. Synthetic Data Generator Design

**Location**: `training_data_generator.py`

### Purpose

The LightGBM Insight Ranker needs labeled training data, but there are no ground-truth labels. The system uses a synthetic data generator that:
1. Generates feature vectors mimicking real pipeline output distributions
2. Assigns labels using rule-based logic that mirrors the insight generator
3. **Crucially**: Adjusts feature values to be CONSISTENT with labels

### Feature-Label Consistency Enforcement

```python
# After assigning "spending_spike" label randomly:
spike_mask = df["insight_type"] == "spending_spike"
df.loc[spike_mask, "amount_zscore"] = rng.uniform(3.0, 5.0, size=spike_count)
df.loc[spike_mask, "is_anomaly"] = 1
df.loc[spike_mask, "amount"] = rng.lognormal(mean=8.0, sigma=0.8, size=spike_count)
```

Without this step, a transaction labeled "spending_spike" might have a z-score of 0.1 and `is_anomaly=False`, confusing the model. The generator ensures features are causally consistent with their labels.

### Edge Case Augmentation

5 adversarial patterns are injected (~500 samples default):
1. **Borderline z-scores (2.9–3.1)**: Tests the anomaly decision boundary
2. **High z-score + tiny amount (<₹50)**: Statistically unusual but trivial — should be no_action
3. **Recurring-looking + variable amounts**: Looks like subscription but isn't
4. **Weekend spending spikes**: Context-dependent (food/entertainment on weekends)
5. **Low-confidence categorization + anomaly features**: Tests robustness to classifier uncertainty

### Indian Market Calibration

Amount distributions use `lognormal(mean=6.0, sigma=1.2)`, centered around ~₹500 with a long tail up to ₹100,000. Category weights reflect typical Indian spending patterns: food 22%, shopping 15%, transport 12%, etc. Subscription amounts include common Indian price points: ₹99, ₹149, ₹199, ₹299, ₹499, ₹799, ₹999, ₹1499.

---

## 10. PII Scrubbing Strategy

**Location**: `preprocessor.py`, lines 144–204 (`clean_remark`)

### What Gets Removed

| Pattern | Example | Rationale |
|---------|---------|-----------|
| 4+ digit sequences | `9876543210` | Phone numbers, account numbers, UPI refs |
| Email addresses | `user@email.com` | Personal identifiers |
| Special characters | `@`, `#`, `/`, `!` | Structural noise from banking systems |
| Noise tokens | "payment", "ref", "txn", "cr", "dr" | Generic banking verbiage with no semantic value |
| Single-character tokens | `a`, `x` | Residual noise after cleaning |

### What Gets Preserved

- Merchant names (via alias resolution or organic text)
- Numbers with fewer than 4 digits (e.g., "shop 123")
- Alphanumeric tokens with length ≥ 2

### Order of Operations

The order matters. Merchant alias matching happens BEFORE PII removal. This ensures patterns like `"NETFLIX.COM SUBSCRIPTION 9876543210"` are resolved to `"netflix"` immediately, without the PII removal potentially corrupting the merchant name.

---

## 11. Schema Contract Enforcement

**Location**: `schema.py`

### Design Philosophy

The system uses a **column name contract** pattern. Every column name is a constant on the `Col` class. Every module validates its required columns at entry using `require_columns()`.

### Enforcement Chain

```
preprocessor.py:274    → require_columns(df, Col.raw_input())
feature_engineer.py:55 → require_columns(df, Col.feature_engineer_input())
seed_labeler.py:150    → require_columns(df, Col.seed_labeler_input())
anomaly_detector.py:25 → require_columns(df, Col.anomaly_detector_input())
```

Each `Col.*_input()` method returns a `FrozenSet[str]` — immutable, hashable, and prevents accidental mutation.

### Type Coercion

`coerce_and_validate_types()` (lines 168–193) uses `pd.to_numeric(errors="coerce")` to handle garbage data in the `amount` column. Unparseable values become NaN, then those rows are dropped and logged. This prevents `ValueError` crashes deep in the pipeline.

---

## 12. Memory Optimization Strategies

**Location**: `pipeline.py`, lines 64–71

### Post-Pipeline Optimization

After all ML computations are complete:
```python
df["predicted_category"] = df["predicted_category"].astype("category")
df["is_weekend"] = df["is_weekend"].astype(bool)
```

**Why after, not before?** Some sklearn transformers (especially `ColumnTransformer` with `OneHotEncoder`) don't handle pandas categorical types correctly. The optimization is deferred to avoid breaking the ML pipeline.

### Sparse Matrix Preservation

In `categorization_model.py`:
```python
# sparse_threshold=1.0 forces ColumnTransformer to keep output sparse
ColumnTransformer(..., sparse_threshold=1.0)
```

TF-IDF produces sparse matrices. Without `sparse_threshold=1.0`, the ColumnTransformer would convert the entire matrix to dense, causing O(n × 2000) memory allocation for every prediction.

---

## 13. Observability Architecture

**Location**: `logger_factory.py`

### Structured Logging

Every log message includes:
```json
{
  "timestamp": "2026-04-13T07:00:00.000000+00:00",
  "level": "INFO",
  "logger": "preprocessor",
  "message": "Dropped 3 rows with zero amount",
  "pipeline_run_id": "run_a1b2c3d4"
}
```

### Pipeline Run ID Tracing

```python
pipeline_run_id_ctx = contextvars.ContextVar("pipeline_run_id", default="UNKNOWN_RUN")
```

At the start of `run_pipeline()`, `generate_new_run_id()` creates a unique ID (e.g., `run_a1b2c3d4`) and stores it in a `ContextVar`. Every log message from every module in that execution automatically includes this ID. This enables:
- Correlating all log messages from a single pipeline run
- Distinguishing concurrent pipeline executions in multi-threaded deployments
- Post-mortem debugging by filtering logs on `pipeline_run_id`

### Event Type & Metrics

Optional structured fields:
```python
logger.info("Phase 2 complete", extra={
    "event_type": "phase_complete",
    "metrics": {"rows_labeled": 450, "coverage": 0.82}
})
```

---

## 14. Identified Technical Debt & Risks

### Risk 1: Dictionary Order Dependency in Merchant Aliases
**File**: `preprocessor.py:160–182`
**Issue**: `clean_remark()` iterates `_COMPILED_ALIASES` in insertion order. Specific merchants MUST appear before generic routers. This is guaranteed by Python 3.7+ CPython dict ordering, but:
- JSON loading with `json.loads()` preserves order (safe)
- Database loading may NOT preserve order (unsafe)
- If `MERCHANT_ALIASES` is ever refactored to a database-backed config, the matching behavior could silently change
**Recommendation**: Add an explicit `order` field to each alias or sort by specificity at compile time.

### Risk 2: `fix_detector.py` is a Legacy Artifact
**File**: `fix_detector.py` (6 lines)
**Issue**: This is a one-shot patch script that directly manipulates source code via file read/write. It modifies `recurring_detector.py` by replacing a threshold condition. It should not be in the repository.
**Recommendation**: Delete the file. The fix has already been applied.

### Risk 3: Pickle Deserialization Despite Checksum
**File**: `insight_model.py:144`
**Issue**: Even with SHA-256 verification, `pickle.load()` is inherently dangerous. A compromised training pipeline could produce a valid checksum for a malicious model. The system trusts the training pipeline completely.
**Recommendation**: Consider switching to a safer serialization format (ONNX, joblib with trusted-only mode) or adding allowlist-based pickle unpickling.

### Risk 4: No Explicit Column Ordering in Feature Matrix
**File**: `insight_model.py:173–188`
**Issue**: The `predict_insight_scores()` function passes a DataFrame to `pipeline.predict()`. If the column order in the prediction DataFrame differs from the training DataFrame, some models (especially older versions of XGBoost) may produce incorrect results. The current code relies on the `ColumnTransformer` to handle this via column names, which is safe for sklearn 1.8+ but fragile.
**Recommendation**: Explicitly reorder columns before prediction.

### Risk 5: No Rate Limiting on Inference Path
**File**: `pipeline.py:184–258`
**Issue**: `run_inference()` has no guard against being called in a tight loop with millions of single-transaction calls. Each call re-executes the full feature engineering pipeline (including history concatenation and rolling computation).
**Recommendation**: Add batch inference support and/or caching for repeated history windows.

### Risk 6: Hardcoded `pipeline_version = "1.0.0"` in model_state
**File**: `model_state.py:33`
**Issue**: `save_model_state()` hardcodes `"1.0.0"` regardless of what `state.pipeline_version` contains. This means the version on the state object is ignored during save.
**Recommendation**: Use `state.pipeline_version` instead of the hardcoded string, or remove the field from the dataclass if it's always "1.0.0".

### Risk 7: Non-Deterministic E2E Test
**File**: `tests/test_e2e.py:66`
**Issue**: The E2E test uses `np.random.rand()` without a seed for generating random noise transactions. This means the number of random transactions varies per run, which could occasionally cause test flakiness.
**Recommendation**: Set `np.random.seed()` at the start of the test.
