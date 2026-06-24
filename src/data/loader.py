# src/data/loader.py
import yfinance as yf
import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")

def fetch_data():
    """Downloads US and Indian market data natively, avoiding holiday distortions."""
    start_date = "2000-01-01" 
    
    if not os.path.exists(RAW_DIR):
        os.makedirs(RAW_DIR)
    
    print(f"📂 Downloading Global Market Data to: {RAW_DIR}")
    
    # Download symbols individually to process returns on native calendars
    print("   Fetching S&P 500 (^GSPC)...")
    sp500 = yf.download('^GSPC', start=start_date, progress=False)
    if isinstance(sp500.columns, pd.MultiIndex):
        sp500.columns = sp500.columns.get_level_values(0)
    
    print("   Fetching VIX (^VIX)...")
    vix = yf.download('^VIX', start=start_date, progress=False)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)
        
    print("   Fetching Nifty 50 (^NSEI)...")
    nifty = yf.download('^NSEI', start=start_date, progress=False)
    if isinstance(nifty.columns, pd.MultiIndex):
        nifty.columns = nifty.columns.get_level_values(0)

    # Clean individual Series
    sp500_close = sp500[['Close']].rename(columns={'Close': 'Price'})
    vix_close = vix[['Close']].rename(columns={'Close': 'VIX'})
    nifty_close = nifty[['Close']].rename(columns={'Close': 'NIFTY'})

    # Compute returns natively on original market calendars to avoid holiday artifacts
    print("   Computing returns on native market calendars...")
    sp500_close['Log_Ret'] = np.log(sp500_close['Price'] / sp500_close['Price'].shift(1))
    nifty_close['Nifty_Ret'] = np.log(nifty_close['NIFTY'] / nifty_close['NIFTY'].shift(1))

    # Join datasets relative to the US trading calendar
    print("   Merging datasets relative to the S&P 500 calendar...")
    master_df = sp500_close.join(vix_close, how='left')
    master_df = master_df.join(nifty_close[['Nifty_Ret']], how='left')
    
    # Safely carry forward previous values on holidays
    master_df['VIX'] = master_df['VIX'].ffill()
    master_df['Nifty_Ret'] = master_df['Nifty_Ret'].ffill()
    
    # Drop rows without initial valid returns/prices
    master_df.dropna(subset=['Log_Ret', 'VIX'], inplace=True)
    
    save_path = os.path.join(RAW_DIR, "global_markets.csv")
    master_df.to_csv(save_path)
    
    print(f"   ✅ Saved Global Data. Final shape: {master_df.shape}")
    print(f"   Note: The dataset runs from {master_df.index.min().date()} to {master_df.index.max().date()}")

if __name__ == "__main__":
    fetch_data()