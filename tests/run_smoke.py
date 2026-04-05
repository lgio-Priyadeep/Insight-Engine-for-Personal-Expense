"""
run_test.py — Pipeline smoke-test against test-data/scrubbed.csv
================================================================
Runs run_pipeline() end-to-end and prints a structured summary of
each phase's output without modifying any source files.
"""

import logging
import sys
import traceback
from pathlib import Path

import pandas as pd
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ── Setup logging ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_test")

CSV_PATH = Path("../test-data/scrubbed.csv")

# ── Load CSV ──────────────────────────────────────────────────────────────────
logger.info(f"Loading CSV: {CSV_PATH}")
if not CSV_PATH.exists():
    logger.error(f"File not found: {CSV_PATH}")
    sys.exit(1)

raw_df = pd.read_csv(CSV_PATH)

print("\n" + "=" * 60)
print("  RAW DATA SUMMARY")
print("=" * 60)
print(f"  Rows          : {len(raw_df)}")
print(f"  Columns       : {list(raw_df.columns)}")
print(f"  Date range    : {raw_df['date'].min()}  →  {raw_df['date'].max()}")
print(f"  DR rows       : {(raw_df['amount_flag'].str.upper() == 'DR').sum()}")
print(f"  CR rows       : {(raw_df['amount_flag'].str.upper() == 'CR').sum()}")
print(f"  Total amount  : {raw_df['amount'].sum():,.2f}")
print()

# ── Run pipeline ──────────────────────────────────────────────────────────────
try:
    from pipeline import run_pipeline, PipelineResult
    logger.info("Starting pipeline...")
    result: PipelineResult = run_pipeline(raw_df)
except Exception as e:
    print("\n[PIPELINE ERROR]")
    traceback.print_exc()
    sys.exit(1)

# ── Print results ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  PIPELINE RESULTS")
print("=" * 60)

deb = result.debits
cred = result.credits

print(f"\n  Debits processed   : {len(deb)}")
print(f"  Credits processed  : {len(cred)}")
print(f"  Insights generated : {len(result.insights)}")

# ── Phase 2: Category distribution ───────────────────────────────────────────
if "pseudo_label" in deb.columns:
    print("\n── Pseudo-label distribution (seed labeler) ─────────────────────")
    print(deb["pseudo_label"].value_counts().to_string())

# ── Phase 4: Predicted categories ─────────────────────────────────────────────
if "predicted_category" in deb.columns:
    print("\n── Predicted category distribution (ML model) ───────────────────")
    print(deb["predicted_category"].value_counts().to_string())

# ── Phase 5: Anomaly + recurring flags ────────────────────────────────────────
if "is_anomaly" in deb.columns:
    anomaly_count = deb["is_anomaly"].sum()
    print(f"\n── Anomalies detected  : {anomaly_count}")
    if anomaly_count > 0:
        print(deb[deb["is_anomaly"] == True][["date", "amount", "cleaned_remarks", "amount_zscore"]].head(5).to_string(index=False))

if "is_recurring" in deb.columns:
    recurring_count = deb["is_recurring"].sum()
    print(f"\n── Recurring txns      : {recurring_count}")
    if recurring_count > 0:
        print(deb[deb["is_recurring"] == True][["date", "amount", "cleaned_remarks", "recurring_frequency"]].head(5).to_string(index=False))

# ── Phase 6: Insights ─────────────────────────────────────────────────────────
print("\n── Generated Insights ───────────────────────────────────────────────")
if result.insights:
    for i, insight in enumerate(result.insights, 1):
        print(f"  [{i:02d}] {insight}")
else:
    print("  (no insights generated)")

# ── Final column inventory ────────────────────────────────────────────────────
print("\n── Final debit DataFrame columns ────────────────────────────────────")
print(f"  {list(deb.columns)}")

print("\n" + "=" * 60)
print("  PIPELINE TEST COMPLETE — OK")
print("=" * 60 + "\n")
