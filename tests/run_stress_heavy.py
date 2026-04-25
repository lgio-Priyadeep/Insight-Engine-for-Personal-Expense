import time
import pandas as pd
import numpy as np
import logging
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline import run_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(name)-15s | %(message)s")

def generate_stress_data(base_df: pd.DataFrame, multiplier: int) -> pd.DataFrame:
    """Duplicate the base DataFrame and add slight noise to avoid perfect deduplication."""
    print(f"Generating stress data (x{multiplier})...")
    dfs = []
    for i in range(multiplier):
        df_copy = base_df.copy()
        # Add slight time noise
        df_copy['date'] = pd.to_datetime(df_copy['date']) + pd.to_timedelta(np.random.randint(-5, 5, len(df_copy)), unit='d')
        # Add slight amount noise so dedup doesn't kill it
        df_copy['amount'] = df_copy['amount'] + np.random.uniform(-0.5, 0.5, len(df_copy))
        dfs.append(df_copy)
    return pd.concat(dfs, ignore_index=True)

if __name__ == "__main__":
    base_csv = os.path.join(os.path.dirname(__file__), "..", "test-data", "scrubbed.csv")
    base_df = pd.read_csv(base_csv)
    
    # 581 rows * 100 = 58,100 rows (~5-10 years of data for a heavy user)
    # We will do 3 tiers: 10x (5.8k), 50x (29k), 100x (58k)
    tiers = [10, 50, 100]
    
    print("============================================================")
    print("  INSIGHT ENGINE STRESS TEST")
    print("============================================================")
    
    for multiplier in tiers:
        big_df = generate_stress_data(base_df, multiplier)
        rows = len(big_df)
        print(f"\n--- Running Tier: {multiplier}x ({rows:,} rows) ---")
        
        start_time = time.time()
        try:
            # We explicitly silence info logs to only measure pipeline time without IO overhead
            logging.getLogger('pipeline').setLevel(logging.WARNING)
            logging.getLogger('preprocessor').setLevel(logging.WARNING)
            logging.getLogger('seed_labeler').setLevel(logging.WARNING)
            logging.getLogger('feature_engineer').setLevel(logging.WARNING)
            logging.getLogger('categorization_model').setLevel(logging.WARNING)
            logging.getLogger('expected_spend_model').setLevel(logging.WARNING)
            logging.getLogger('anomaly_detector').setLevel(logging.WARNING)
            logging.getLogger('recurring_detector').setLevel(logging.WARNING)
            logging.getLogger('insight_model').setLevel(logging.WARNING)
            logging.getLogger('insight_generator').setLevel(logging.WARNING)

            res = run_pipeline(big_df)
            elapsed = time.time() - start_time
            print(f"✅ Success! Pipeline took {elapsed:.2f} seconds.")
            print(f"   Throughput: {rows/elapsed:.0f} rows/second")
            
        except Exception as e:
            print(f"❌ Failed during tier x{multiplier}: {e}")
            break
            
    print("\nStress test complete.")
