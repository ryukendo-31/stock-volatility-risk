import pandas as pd
import numpy as np
import os
import sys

# Add project root to path so Python can find our new modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import our Specialist Modules
from src.features.volatility import calculate_log_returns, calculate_rolling_volatility
from src.features.regime_features import calculate_volatility_ratios
from src.features.distribution_features import calculate_tail_risk
from src.features.vix_features import calculate_vix_gap

# Path Config
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

class FeaturePipeline:
    def __init__(self):
        self.df = None

    def load_data(self):
        print("⏳ Loading raw data...")
        sp500 = pd.read_csv(os.path.join(RAW_DIR, "sp500_daily.csv"), index_col=0, parse_dates=True)
        vix = pd.read_csv(os.path.join(RAW_DIR, "vix_daily.csv"), index_col=0, parse_dates=True)
        
        # Merge Price and VIX
        self.df = pd.DataFrame()
        self.df['Price'] = sp500['Close']
        self.df['VIX'] = vix['Close']
        self.df.dropna(inplace=True)
        return self

    def apply_feature_engineering(self):
        print("⚙️ Pipeline Running: Applying Pillars 1-4...")
        
        # 1. Base Calculations (Pillar 1)
        self.df = calculate_log_returns(self.df)
        self.df = calculate_rolling_volatility(self.df)
        
        # 2. Regime Features (Pillar 2)
        self.df = calculate_volatility_ratios(self.df)
        
        # 3. Tail Risk (Pillar 3)
        self.df = calculate_tail_risk(self.df)
        
        # 4. Fear Context (Pillar 4)
        self.df = calculate_vix_gap(self.df)
        
        # 5. Lags (Memory)
        # We must give the model "Yesterday's Data" to predict "Tomorrow"
        print("   Adding Lag Features...")
        for col in ['Log_Ret', 'Vol_5d', 'VIX']:
            for lag in [1, 2]:
                self.df[f'{col}_Lag_{lag}'] = self.df[col].shift(lag)

        # 6. THE TARGET (Future Reality)
        # We predict the volatility of the NEXT 5 days.
        print("   Generating Target (Next 5 Days Volatility)...")
        indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=5)
        self.df['Target_Vol_Next_5d'] = self.df['Log_Ret'].rolling(window=indexer).std() * np.sqrt(252)
        
        # Drop NaNs (created by lags and rolling windows)
        self.df.dropna(inplace=True)
        return self

    def save(self):
        if not os.path.exists(PROCESSED_DIR):
            os.makedirs(PROCESSED_DIR)
            
        save_path = os.path.join(PROCESSED_DIR, "features.csv")
        self.df.to_csv(save_path)
        print(f"✅ Success! Data saved to: {save_path}")
        print(f"   Final Shape: {self.df.shape}")
        print(f"   Columns Created: {self.df.shape[1]}")

if __name__ == "__main__":
    # Run the Pipeline
    pipeline = FeaturePipeline()
    pipeline.load_data().apply_feature_engineering().save()