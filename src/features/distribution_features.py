import numpy as np 
import pandas as pd

def calculate_tail_risk(df, window =21):
    ''' add kurtosis and skewness to measure difference in tail width and black swan events '''
    df[f'Skew_{window}'] = df['Log_Ret'].rolling(window=window).skew()
    df[f'Kurt_{window}'] = df['Log_Ret'].rolling(window=window).kurt()
    
    return df