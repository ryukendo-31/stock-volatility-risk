import numpy as np
import pandas as pd

def calculate_cross_market_features(df):
    """
    Calculates features exploiting the time-zone difference between India (NSE) and US (NYSE).
    Pillar 5: Cross-Market Asian Session Signals.
    """
    # Asian Session Momentum (How did the Indian market close today?)
    # This gives the XGBoost model the exact trajectory of the Asian/European morning session.
    df['Nifty_Ret'] = np.log(df['NIFTY'] / df['NIFTY'].shift(1))
    
    return df