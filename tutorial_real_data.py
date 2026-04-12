from logger_factory import get_logger
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

from schema import Col
from pipeline import run_pipeline

from logger_factory import get_logger

logger = get_logger(__name__)

def run_real_data_tutorial(file_path: str, file_type: str = "csv"):
    """
    Demonstrates ingesting real bank statements.
    
    Args:
        file_path: Path to your bank statement.
        file_type: 'csv', 'json', or 'parquet'.
    """
    logger.info(f"Loading {file_type.upper()} bank statement from {file_path}")
    
    # 1. Load the data using Pandas
    if file_type == "csv":
        # Adjust read_csv parameters as needed for your bank's format
        raw_df = pd.read_csv(file_path)
    elif file_type == "json":
        raw_df = pd.read_json(file_path)
    elif file_type == "parquet":
        raw_df = pd.read_parquet(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")
        
    logger.info(f"Loaded {len(raw_df)} raw transactions.")
    
    # 2. Schema Mapping
    # The Insight Engine expects specific column names defined in `schema.py`.
    # You MUST rename your bank's columns to match `schema.py`'s expected inputs:
    # 
    # Col.DATE:          Transaction date (datetime parsable or string)
    # Col.AMOUNT:        Absolute transaction amount (float)
    # Col.AMOUNT_FLAG:   "CR" for credits, "DR" for debits
    # Col.REMARKS:       Raw transaction narrative/merchant name
    
    # Example mapping (adjust according to your bank's header):
    mapping = {
        "Txn Date": Col.DATE,
        "Transaction Amount": Col.AMOUNT,
        "Cr/Dr": Col.AMOUNT_FLAG,
        "Description": Col.REMARKS
    }
    
    # Rename columns and assert requirements exist
    raw_df = raw_df.rename(columns=mapping)
    
    missing = set([Col.DATE, Col.AMOUNT, Col.AMOUNT_FLAG, Col.REMARKS]) - set(raw_df.columns)
    if missing:
        raise ValueError(f"Mapping failed! Missing columns: {missing}")
        
    # 3. Privacy & Processing
    # Pass the correctly-mapped DataFrame directly to the pipeline.
    # The preprocessor will automatically strip PII (Phone numbers, Cards, VPA IDs)
    # before any ML models ever see the data.
    logger.info("Passing mapped data into the Insight Engine...")
    result = run_pipeline(raw_df)
    
    # 4. Extract Insights
    logger.info("\n" + "="*60)
    logger.info("   GENERATED FINANCIAL INSIGHTS (ML RANKED)")
    logger.info("="*60)
    
    for i, insight in enumerate(result.insights, 1):
        print(f"\n{i}. {insight}")
        
    print("\n" + "="*60)
    
if __name__ == "__main__":
    # Create a minimal synthetic 'real_data.csv' mock for demonstration
    mock_data = pd.DataFrame({
        "Txn Date": ["2023-10-01", "2023-10-02", "2023-10-03", "2023-10-04", "2023-10-05"],
        "Transaction Amount": [500.0, 15000.0, 200.0, 200.0, 50000.0],
        "Cr/Dr": ["DR", "DR", "DR", "DR", "CR"],
        "Description": ["Swiggy Order 123", "Apple Store Purchase", "Netflix Sub", "Spotify", "Salary"]
    })
    
    mock_file = "mock_real_data.csv"
    mock_data.to_csv(mock_file, index=False)
    
    try:
        run_real_data_tutorial(mock_file, file_type="csv")
    finally:
        import os
        if os.path.exists(mock_file):
            os.remove(mock_file)
