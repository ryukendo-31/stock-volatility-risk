import pandas as pd
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Path Config
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

def split_data():
    print(" Initiating Chronological Data Split...")
    input_path = os.path.join(PROCESSED_DIR, "features.csv")
    
    if not os.path.exists(input_path):
        print(" Error: features.csv not found! Run feature_builder.py first.")
        return
    
    # Load the newly generated features (2000-2026)
    df = pd.read_csv(input_path, index_col=0, parse_dates=True)
    
    # Final safety check for NaNs before splitting
    df = df.dropna()
    
    # Calculate the 80/20 split index
    # STRICTLY chronological. No random shuffling for time-series!
    split_index = int(0.8 * len(df))
    
    train_df = df.iloc[:split_index]
    test_df = df.iloc[split_index:]
    
    # Save the splits
    train_path = os.path.join(PROCESSED_DIR, "train.csv")
    test_path = os.path.join(PROCESSED_DIR, "test.csv")
    
    train_df.to_csv(train_path)
    test_df.to_csv(test_path) 
    
    print(f" Split Complete!")
    print(f"   Train Set: {len(train_df)} rows ({train_df.index.min().date()} to {train_df.index.max().date()})")
    print(f"   Test Set:  {len(test_df)} rows ({test_df.index.min().date()} to {test_df.index.max().date()})")
    
    # Quick validation to prove no Data Leakage to your professor
    print("\n🔍 Checking for Data Leakage...")
    if train_df.index.max() >= test_df.index.min():
        print("   WARNING: Time overlap detected. Check index.")
    else:
        print("    Time-series integrity confirmed. No look-ahead bias.")

if __name__ == "__main__":
    split_data()