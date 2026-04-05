"""
anomaly_detector.py — Composite Statistical Anomaly Flagging
=============================================================
Flags unusual transactions based on a composite heuristic to balance
statistical outliers (Z-scores) with contextual deviations (ML Residuals).
"""

import logging
import pandas as pd

from schema import Col, require_columns

logger = logging.getLogger(__name__)


def detect_anomalies(
    df: pd.DataFrame, 
    zscore_threshold: float = 3.0, 
    pct_dev_threshold: float = 0.5
) -> pd.DataFrame:
    """
    Identifies anomalous spending using a dual-gate requirement:
      1. amount_zscore > zscore_threshold  (Historically unusual)
      2. percent_deviation > pct_dev_threshold (ML expected spend failure)
      
    This composite prevents low-value expenses from triggering alarms 
    just because they technically miss mathematical expectations.
    """
    logger.info("Executing composite anomaly detection pipeline...")
    
    df = df.copy()
    
    require_columns(df, Col.anomaly_detector_input(), "anomaly_detector")

    # We typically care about positive anomalies (spending spikes)
    # If negative anomalies (unusually low spend) are needed, we can use .abs()
    is_spike = (
        (df[Col.AMOUNT_ZSCORE].abs() > zscore_threshold)
        & (df[Col.PERCENT_DEVIATION].abs() > pct_dev_threshold)
    )
    
    df[Col.IS_ANOMALY] = is_spike
    
    anomaly_count = df[Col.IS_ANOMALY].sum()
    logger.info(f"Anomaly scan complete. Flagged {anomaly_count} transactions.")

    return df

