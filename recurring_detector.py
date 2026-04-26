"""
recurring_detector.py — Rule-based Recurring Transaction Identifier
===================================================================
Uses time deltas and semantic similarity to automatically flag subscriptions, 
recurring bills, and standing orders.

Current Logic leverages deterministic scaled scoring equations:
score = 0.4*A + 0.4*T + 0.2*V
"""

import logging
import pandas as pd
import numpy as np

import config
from config import RECURRING_CONFIG
from schema import Col, require_columns
from hash_utils import stable_hash

logger = logging.getLogger(__name__)


def find_recurring_transactions(
    df: pd.DataFrame, 
    group_col: str = Col.CLEANED_REMARKS,
) -> pd.DataFrame:
    """
    Groups transactions and identifies stable frequencies indicating subscriptions.
    Utilizes clamped scoring calculations A, T, V guaranteeing bounding loops.
    """
    logger.info("Executing recurring transaction detection...")
    
    df = df.copy()
    require_columns(df, Col.recurring_detector_input(), "recurring_detector")
    
    # Needs to be sorted chronologically within groups to accurately measure diff
    df = df.sort_values(by=[Col.DATE])
    
    df[Col.IS_RECURRING] = False
    df[Col.RECURRING_FREQUENCY] = None
    df[Col.RECURRING_CONFIDENCE] = 0.0
    df[Col.RECURRING_SCORE] = 0.0
    
    grouped = df.groupby(group_col)
    
    recurring_indices = []
    freq_map = {}
    conf_map = {}
    
    amount_tolerance = RECURRING_CONFIG["global"]["amount_tolerance"]
    min_occ = RECURRING_CONFIG["global"]["min_occurrences"]
    clamp = lambda x: max(0.0, min(1.0, float(x)))
    
    for identifier, group in grouped:
        if len(group) < min_occ:
            continue
            
        time_diffs = group[Col.DATE].diff().dt.days.dropna()
        if len(time_diffs) == 0:
            continue
            
        amounts = group[Col.AMOUNT]
        mean_amt = amounts.mean()
        if mean_amt == 0:
            continue
            
        amount_drift = (amounts.max() - amounts.min()) / mean_amt
        
        # 1. Amount Score [0, 1]
        raw_A = 1.0 - (amount_drift / amount_tolerance) if amount_tolerance > 0 else 1.0
        A = clamp(raw_A)
        
        # 2. Temporal Score [0, 1]
        mean_gap = time_diffs.mean()
        var = time_diffs.var() if len(time_diffs) > 1 else 0.0
        
        assigned_freq = None
        raw_T = 0.0
        
        for k, v in RECURRING_CONFIG.items():
            if k == "global": continue
            if v["min_gap"] <= mean_gap <= v["max_gap"]:
                assigned_freq = v["type"]
                raw_T = 1.0 - (var / v["var"]) if v["var"] > 0 else 1.0
                break
                
        T = clamp(raw_T)
        
        # 3. Volume Score [0, 1]
        V = clamp(len(group) / 12.0) # Assume 12 is a perfect solid year of hits
        
        if T == 0.0 or A == 0.0:
            merchant_ref = identifier if config.ENABLE_PII_DEBUG_LOGS else stable_hash(identifier)
            logger.debug(
                "Rejected recurring pattern for %s: A=%.2f, T=%.2f",
                merchant_ref, A, T,
                extra={"event_type": "recurring_detection_metrics", "stage": "phase_5"}
            )
            continue

        score = (0.4 * A) + (0.4 * T) + (0.2 * V)
        merchant_ref = identifier if config.ENABLE_PII_DEBUG_LOGS else stable_hash(identifier)
        logger.debug(
            "Assessed recurring transaction pattern for %s. Components -> A:%.2f, T:%.2f, V:%.2f, Final:%.2f",
            merchant_ref, A, T, V, score,
            extra={"event_type": "recurring_detection_metrics", "stage": "phase_5"}
        )
        
        if assigned_freq:
            recurring_indices.extend(group.index.tolist())
            for idx in group.index:
                freq_map[idx] = assigned_freq
                conf_map[idx] = score
            
    # Defensive Vectorization mapping without NaNs sneaking in implicitly
    mask_idx = df.index.isin(recurring_indices)
    if mask_idx.any():
        df.loc[mask_idx, Col.IS_RECURRING] = True
        
        # Safely map objects natively bypassing pandas coerce mechanics
        freq_series = df.index.map(freq_map)
        conf_series = df.index.map(conf_map)
        
        df.loc[freq_series.notna(), Col.RECURRING_FREQUENCY] = freq_series[freq_series.notna()].astype("object")
        df.loc[conf_series.notna(), Col.RECURRING_CONFIDENCE] = conf_series[conf_series.notna()].astype("float")
        df.loc[conf_series.notna(), Col.RECURRING_SCORE] = conf_series[conf_series.notna()].astype("float")

    logger.info(f"Flagged {len(recurring_indices)} recurring transactions across unique subscriptions.")
    return df
