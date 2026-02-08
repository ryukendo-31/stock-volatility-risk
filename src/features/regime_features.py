import pandas as pd

def calculate_volatility_ratios(df):
    """
    Adds ratio features comparing short-term vs long-term vol.
    Pillar 2: Acceleration - Detecting Regime Shifts.
    """
    # 1. Panic Signal (Short Term)
    # If 5d > 21d (Ratio > 1), risk is accelerating quickly.
    df['Vol_Ratio_5_21'] = df['Vol_5d'] / df['Vol_21d']
    
    # 2. Trend Signal (Medium Term)
    # If 21d > 63d (Ratio > 1), we are likely in a sustained downtrend/bear market.
    df['Vol_Ratio_21_63'] = df['Vol_21d'] / df['Vol_63d']
    
    return df