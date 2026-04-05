"""
recurring_detector.py — Rule-based Recurring Transaction Identifier
===================================================================
Uses time deltas and semantic similarity to automatically flag subscriptions, 
recurring bills, and standing orders.

Current Logic is heuristics-based to ensure explainability.
"""

import logging
import pandas as pd
import numpy as np

from config import RECURRING_AMOUNT_TOLERANCE, RECURRING_FLUCTUATION_PENALTY_THRESHOLD
from schema import Col, require_columns

logger = logging.getLogger(__name__)


def find_recurring_transactions(
    df: pd.DataFrame, 
    group_col: str = Col.CLEANED_REMARKS,
    amount_tolerance: float = RECURRING_AMOUNT_TOLERANCE
) -> pd.DataFrame:
    """
    Groups transactions and identifies stable frequencies indicating subscriptions.
    
    A group is deemed recurring if:
      1. It has >= 2 occurrences.
      2. The time gap between sequential events remains roughly:
         - Monthly: 27–33 days
         - Weekly: 6–8 days
      3. The cost variation falls within `amount_tolerance` (default 5%).
    """
    logger.info("Executing recurring transaction detection...")
    
    df = df.copy()
    
    require_columns(df, Col.recurring_detector_input(), "recurring_detector")
    
    # Needs to be sorted chronologically within groups to accurately measure diff
    df = df.sort_values(by=[Col.DATE]).reset_index(drop=True)
    
    df[Col.IS_RECURRING] = False
    df[Col.RECURRING_FREQUENCY] = None
    df[Col.RECURRING_CONFIDENCE] = 0.0
    
    grouped = df.groupby(group_col)
    
    recurring_indices = []
    freq_map = {}
    conf_map = {}
    
    for identifier, group in grouped:
        if len(group) < 3:
            continue
        
        # Calculate time diff in days
        time_diffs = group[Col.DATE].diff().dt.days.dropna()
        
        # Calculate amount consistency
        amounts = group[Col.AMOUNT]
        amount_drift = (amounts.max() - amounts.min()) / amounts.mean()
        
        if amount_drift > amount_tolerance:
            continue
            
        mean_gap = time_diffs.mean()
        var = 0.0 if len(time_diffs) == 1 else time_diffs.var()
        
        assigned_freq = None
        if 27 <= mean_gap <= 33 and var < 10:
            assigned_freq = "monthly"
        elif 13 <= mean_gap <= 16 and var < 5:
            assigned_freq = "biweekly"
        elif 6 <= mean_gap <= 8 and var < 3:
            assigned_freq = "weekly"
        elif 85 <= mean_gap <= 95 and var < 20:
            assigned_freq = "quarterly"
            
        if assigned_freq:
            confidence = 1.0
            if amount_drift > RECURRING_FLUCTUATION_PENALTY_THRESHOLD:
                confidence = 0.5
                
            logger.debug(f"Detected {assigned_freq} sub for {identifier} (~{amounts.mean():.2f}) (conf {confidence})")
            recurring_indices.extend(group.index.tolist())
            freq_map.update({idx: assigned_freq for idx in group.index})
            conf_map.update({idx: confidence for idx in group.index})
            
    # Apply tags back to the main dataframe safely
    if recurring_indices:
        df.loc[recurring_indices, Col.IS_RECURRING] = True
        df[Col.RECURRING_FREQUENCY] = df.index.map(freq_map)
        df[Col.RECURRING_CONFIDENCE] = df.index.map(conf_map).fillna(0.0)
        
    logger.info(f"Flagged {len(recurring_indices)} recurring transactions across {len(set(freq_map.values()))} unique subscriptions.")
    return df
