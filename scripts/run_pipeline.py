# scripts/run_pipeline.py
from src.data.loader import fetch_data
from src.features.feature_builder import FeaturePipeline


def main():
    """Download raw data and build features. Evaluation is walk-forward only (no static split)."""
    print(" ========== STARTING DATA PIPELINE ========== ")
    print("\n[STEP 1/2] Fetching latest market data...")
    fetch_data()
    print("\n[STEP 2/2] Building features from raw data...")
    FeaturePipeline().load_data().apply_feature_engineering().save()
    print("\n ========== DATA PIPELINE COMPLETED ========== ")
    print(" Next: python scripts/calibrate.py, then python scripts/run_walkforward_hybrid.py")


if __name__ == "__main__":
    main()