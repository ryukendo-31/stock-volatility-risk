# --- START OF FILE run_pipeline.py ---
import os
import sys

# Ensure the 'src' directory is in the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the main functions from your scripts
from src.data.loader import fetch_data
from src.features.feature_builder import FeaturePipeline
from src.data.splitter import split_data

def main():
    """
    Executes the entire data processing pipeline:
    1. Downloads raw data.
    2. Builds features.
    3. Splits data into training and testing sets.
    """
    print(" ========== STARTING DATA PIPELINE ========== ")
    
    # Step 1: Download raw data
    print("\n[STEP 1/3] Fetching latest market data...")
    fetch_data()
    
    # Step 2: Engineer features
    print("\n[STEP 2/3] Building features from raw data...")
    pipeline = FeaturePipeline()
    pipeline.load_data().apply_feature_engineering().save()
    
    # Step 3: Split data for modeling
    print("\n[STEP 3/3] Splitting data into train and test sets...")
    split_data()
    
    print("\n ========== DATA PIPELINE COMPLETED SUCCESSFULLY ========== ")
    print(" now run 'run_baselines.py' to train and evaluate models.")

if __name__ == "__main__":
    main()
# --- END OF FILE run_pipeline.py ---