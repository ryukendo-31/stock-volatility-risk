import numpy as np
import pandas as pd

def calculate_log_returns(df, price_col='Price'):
    """Calculates Log Returns: ln(Pt / Pt-1)."""
    df['Log_Ret'] = np.log(df[price_col] / df[price_col].shift(1))
    return df

def calculate_rolling_volatility(df, window_sizes=[5, 10, 21, 63]):
    """
    Adds rolling volatility columns.
    Pillar 1: Inertia
    """
    for w in window_sizes:
        col_name = f'Vol_{w}d'
        # Std Dev * Sqrt(252) -> Annualized
        df[col_name] = df['Log_Ret'].rolling(window=w).std() * np.sqrt(252)
    return df