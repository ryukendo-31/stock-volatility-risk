import yfinance as yf
import pandas as pd
import os

# Define paths relative to this script
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")

def fetch_data():
    """Downloads S&P 500 and VIX data cleanly."""
    start_date = "2000-01-01"
    
    if not os.path.exists(RAW_DIR):
        os.makedirs(RAW_DIR)
    
    print(f"📂 Downloading data to: {RAW_DIR}")
    
    # --- 1. S&P 500 ---
    print("   Fetching S&P 500 (^GSPC)...")
    sp500 = yf.download("^GSPC", start=start_date, progress=False)
    
    # FIX: Flatten MultiIndex columns (e.g. ('Close', '^GSPC') -> 'Close')
    if isinstance(sp500.columns, pd.MultiIndex):
        sp500.columns = sp500.columns.get_level_values(0)
        
    sp500_path = os.path.join(RAW_DIR, "sp500_daily.csv")
    sp500.to_csv(sp500_path)
    print(f"   ✅ Saved S&P 500: {sp500.shape}")

    # --- 2. VIX ---
    print("   Fetching VIX (^VIX)...")
    vix = yf.download("^VIX", start=start_date, progress=False)
    
    # FIX: Flatten MultiIndex columns
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)
        
    vix_path = os.path.join(RAW_DIR, "vix_daily.csv")
    vix.to_csv(vix_path)
    print(f"   ✅ Saved VIX: {vix.shape}")
    
    print("\n🎉 Data download complete.")

if __name__ == "__main__":
    fetch_data()