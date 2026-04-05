import pandas as pd
from schema import Col
from pipeline import run_pipeline

# 1. Load your raw bank statement
my_data = pd.read_csv("path/my_bank_export.csv")

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