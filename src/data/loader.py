import yfinance as yf
import pandas as pd
import os

# Define paths relative to this script
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")

def fetch_data():
    """Downloads US and Indian market data from your original start date."""
    start_date = "2000-01-01" 
    
    if not os.path.exists(RAW_DIR):
        os.makedirs(RAW_DIR)
    
    print(f"📂 Downloading Global Market Data to: {RAW_DIR}")
    print(f"   Using your specified start date: {start_date}")
    
    symbols = {
        'Price': '^GSPC',      # S&P 500
        'VIX': '^VIX',         # US Volatility
        'NIFTY': '^NSEI'       # India Nifty 50 (India VIX removed to unlock 2000-2008 data)
    }
    
    dfs = []
    for name, ticker in symbols.items():
        print(f"   Fetching {name} ({ticker})...")
        tmp = yf.download(ticker, start=start_date, progress=False)
        
        if isinstance(tmp.columns, pd.MultiIndex):
            tmp.columns = tmp.columns.get_level_values(0)
            
        tmp = tmp[['Close']].rename(columns={'Close': name})
        dfs.append(tmp)
    
    print("   Merging global dates...")
    master_df = pd.concat(dfs, axis=1)
    
    # Forward-fill to handle holidays
    master_df.ffill(inplace=True)
    
    # Drop NaNs (This will now successfully keep the year 2000+ data)
    master_df.dropna(inplace=True) 
    
    save_path = os.path.join(RAW_DIR, "global_markets.csv")
    master_df.to_csv(save_path)
    
    print(f"   ✅ Saved Global Data. Final shape: {master_df.shape}")
    print(f"   Note: The dataset now runs from {master_df.index.min().date()} to {master_df.index.max().date()}")

if __name__ == "__main__":
    fetch_data()