import pandas as pd
import os
from schema import Col
from pipeline import run_pipeline
from summary_utils import print_summary

# 1. Load your raw bank statement
my_data = pd.read_csv("test-data/scrubbed.csv")

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
for insight in results.insights:
    print(insight)

# 5. Print Executive Summary and Detailed Personal Log
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