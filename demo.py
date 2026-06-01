import os
# Programmatically enable the Passion Engine before importing and running the pipeline
os.environ["INSIGHT_ENGINE_PASSION_ENABLED"] = "true"

import pandas as pd
from schema import Col
from pipeline import run_pipeline
from summary_utils import print_summary

# 1. Load your raw bank statement
my_data = pd.read_csv("/home/priyadeep/Desktop/Insight engine/test-data/scrubbed.csv")

# 2. Map your columns to the strict Schema
my_data = my_data.rename(columns={
    "Transaction Date": Col.DATE,
    "Debit/Credit": Col.AMOUNT_FLAG,
    "Value": Col.AMOUNT,
    "Bank Narration": Col.REMARKS
})

# 3. Run the engine!
results = run_pipeline(my_data)

# 4. View your ranked insights
print("\n=== CORE INSIGHTS ===")
for insight in results.insights:
    print(f"- {insight}")

# View Passion Insights & Signals if successfully processed
passion_status = results.stats.get("passion_status", "disabled")
if passion_status == "success":
    print("\n=== PASSION INSIGHTS ===")
    if results.passion_insights:
        for insight in results.passion_insights:
            print(f"- {insight}")
    else:
        print("No active passion insights detected for this statement.")
    
    if results.passion_signals:
        print("\n=== DETECTED PASSION SIGNALS ===")
        for signal in results.passion_signals:
            merchants_str = ", ".join(signal.merchant_list)
            print(f"- {signal.category.upper()}: Spend Share {signal.spend_share:.1%}, Merchants: {merchants_str}")
else:
    print(f"\n[Passion Engine Status: {passion_status}]")
    if passion_status == "disabled":
        print("To enable passion detection, set INSIGHT_ENGINE_PASSION_ENABLED=true")
    elif passion_status == "skipped":
        print("Passion detection was skipped (possibly due to empty data or insufficient rows).")

# 5. Print Executive Summary and Detailed Personal Log
print("\n=== SUMMARY ===")
print_summary(results)

# 6. Save known person transactions separately
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

if not results.personal_debits.empty or not results.personal_credits.empty:
    if not results.personal_debits.empty:
        out_path = os.path.join(output_dir, "known_person_debits.csv")
        results.personal_debits.to_csv(out_path, index=False)
        print(f"Saved personal debits to {out_path}")
        
    if not results.personal_credits.empty:
        out_path = os.path.join(output_dir, "known_person_credits.csv")
        results.personal_credits.to_csv(out_path, index=False)
        print(f"Saved personal credits to {out_path}")

# 7. Save passion-enriched transaction logs (debits containing subcategories)
if passion_status == "success" and not results.passion_debits.empty:
    passion_out_path = os.path.join(output_dir, "passion_enriched_debits.csv")
    results.passion_debits.to_csv(passion_out_path, index=False)
    print(f"Saved passion enriched debits to {passion_out_path}")