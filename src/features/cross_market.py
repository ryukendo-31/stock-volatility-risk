# src/features/cross_market.py
import numpy as np
import pandas as pd

def calculate_cross_market_features(df):
    """
    Calculates features exploiting the time-zone difference between India (NSE) and US (NYSE).
    Pillar 5: Cross-Market Asian Session Signals.
    
    Note: Nifty_Ret is now calculated natively inside loader.py to avoid 
    holiday/calendar misalignment. This function acts as a safety pass-through 
    and verification step.
    """
    if 'Nifty_Ret' not in df.columns:
        raise KeyError(
            "Nifty_Ret is missing from the raw data. "
            "Ensure that src/data/loader.py is executed first."
        )
        
    return df