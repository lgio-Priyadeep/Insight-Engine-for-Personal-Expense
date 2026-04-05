"""
stress_test.py
==============
Evaluates the Financial Insight Engine's performance and memory scaling
on a synthetic dataset of 50,000 rows.

Usage:
  ./venv/bin/python stress_test.py
"""

import time
import logging
import psutil
import os
import gc
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline import run_pipeline
from schema import Col

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def generate_large_dataset(n_rows: int = 50000) -> pd.DataFrame:
    logger.info(f"Generating synthetic dataset with {n_rows} rows...")
    base_date = datetime(2020, 1, 1)
    
    # Pre-allocate for speed
    dates = []
    amounts = np.random.lognormal(mean=4.0, sigma=1.0, size=n_rows).round(2)
    flags = np.random.choice(["DR", "CR"], size=n_rows, p=[0.8, 0.2])
    
    # Pick from a set of common remarks
    merchants = ["Zomato Order", "Amazon Purchase", "Netflix Sub", "Uber Ride", "Salary Credit", "Unknown Merchant TXN"]
    remarks = np.random.choice(merchants, size=n_rows)
    
    for i in range(n_rows):
        dates.append(base_date + timedelta(days=i % 1000, hours=i % 24))
        
    df = pd.DataFrame({
        Col.DATE: dates,
        Col.AMOUNT: amounts,
        Col.AMOUNT_FLAG: flags,
        Col.REMARKS: remarks
    })
    
    # Ensure chronology
    df = df.sort_values(by=Col.DATE).reset_index(drop=True)
    return df

def run_stress_test():
    n_rows = 50000
    df = generate_large_dataset(n_rows)
    
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / (1024 * 1024)  # MB
    
    logger.info(f"Starting pipeline on {n_rows} rows.")
    logger.info(f"Memory before pipeline: {mem_before:.2f} MB")
    
    start_time = time.time()
    
    result = run_pipeline(df)
    
    end_time = time.time()
    mem_after = process.memory_info().rss / (1024 * 1024)
    
    execution_time = end_time - start_time
    
    logger.info(f"Pipeline finished in {execution_time:.2f} seconds.")
    logger.info(f"Memory after pipeline: {mem_after:.2f} MB (Delta: {mem_after - mem_before:.2f} MB)")
    logger.info(f"Total Insights Generated (Top-N): {len(result.insights)}")
    
    for i, insight in enumerate(result.insights, 1):
        print(f"{i}. {insight}")
    
    if execution_time > 20.0:
        logger.warning(f"Pipeline is scaling slower than optimal (>20s for {n_rows} rows).")
    else:
        logger.info(f"Stress test PASSED: Scaling is within optimal boundaries (<20s for {n_rows} rows).")

if __name__ == "__main__":
    run_stress_test()
