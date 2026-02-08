import numpy as np 
import pandas as pd 

import pandas as pd

def calculate_vix_gap(df):
    """
    Adds VIX Premium (Implied vs Realized).
    Pillar 4: Fear Gap - Market Psychology.
    """
    # VIX is in % (e.g. 20.0). Vol_21d is decimal (0.20).
    # We multiply Vol by 100 to compare them fairly.
    # Positive Gap = Fear (Insurance is expensive).
    # Negative Gap = Complacency (Insurance is suspiciously cheap).
    df['VIX_Gap'] = df['VIX'] - (df['Vol_21d'] * 100)
    
    return df