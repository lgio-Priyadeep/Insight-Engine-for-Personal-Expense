import pandas as pd
from schema import Col
from dataclasses import dataclass

def print_summary(results):
    """
    Prints a rich executive summary of the pipeline run.
    """
    # 1. Calculate Global Summary
    total_inflow_amt = results.credits[Col.AMOUNT].sum()
    total_inflow_count = len(results.credits)
    
    total_outflow_amt = results.debits[Col.AMOUNT].sum()
    total_outflow_count = len(results.debits)
    
    # Expense refers to the actual spending filtered for ML (excluding known persons)
    expense_mask = ~results.debits[Col.IS_KNOWN_PERSON].fillna(False)
    expense_df = results.debits.loc[expense_mask]
    expense_amt = expense_df[Col.AMOUNT].sum()
    expense_count = len(expense_df)
    
    net_change = total_inflow_amt - total_outflow_amt
    
    # 2. Known Person Summary
    sent_kp_amt = results.personal_debits[Col.AMOUNT].sum()
    sent_kp_count = len(results.personal_debits)
    
    received_kp_amt = results.personal_credits[Col.AMOUNT].sum()
    received_kp_count = len(results.personal_credits)
    
    kp_net_change = received_kp_amt - sent_kp_amt
    
    print("\n" + "="*60)
    print(" FINANCIAL EXECUTIVE SUMMARY")
    print("="*60)
    
    print(f"{'Metric':<25} | {'Amount (₹)':<15} | {'Transactions':<12}")
    print("-" * 60)
    print(f"{'Total Inflow':<25} | {total_inflow_amt:>15.2f} | {total_inflow_count:>12}")
    print(f"{'Total Outflow':<25} | {total_outflow_amt:>15.2f} | {total_outflow_count:>12}")
    print(f"{'Expense (Actual Spend)':<25} | {expense_amt:>15.2f} | {expense_count:>12}")
    print("-" * 60)
    print(f"{'NET CASH FLOW':<25} | {net_change:>15.2f} |")
    
    print("\n" + "="*60)
    print(" KNOWN PERSONS & SELF-TRANSFER SUMMARY")
    print("="*60)
    print(f"{'Total Sent':<25} | {sent_kp_amt:>15.2f} | {sent_kp_count:>12}")
    print(f"{'Total Received':<25} | {received_kp_amt:>15.2f} | {received_kp_count:>12}")
    print(f"{'KP Net Change':<25} | {kp_net_change:>15.2f} |")
    
    # Detailed Log Table (Alias-Centric)
    all_personal = pd.concat([
        results.personal_debits.assign(Type="SENT"),
        results.personal_credits.assign(Type="RCVD")
    ]).sort_values([Col.KNOWN_PERSON_ALIAS, Col.DATE])
    
    if not all_personal.empty:
        print("\n" + "="*60)
        print(" DETAILED PERSONAL TRANSACTION LOG (GROUPED BY ALIAS)")
        print("="*60)
        header = f"{'Alias':<15} | {'Date':<12} | {'Type':<5} | {'Amount':<10} | {'Remark'}"
        print(header)
        print("-" * len(header))
        
        for _, row in all_personal.iterrows():
            alias = str(row[Col.KNOWN_PERSON_ALIAS])[:15]
            date = str(row[Col.DATE])[:10]
            type_str = row["Type"]
            amount = f"₹{row[Col.AMOUNT]:.2f}"
            remark = str(row[Col.REMARKS])[:30]
            print(f"{alias:<15} | {date:<12} | {type_str:<5} | {amount:<10} | {remark}")
    
    print("="*60 + "\n")
