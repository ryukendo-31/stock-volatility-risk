# scripts/test_shap_gate.py
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.xgboost_vol import HybridXGBoostVol
from src.decision.shap_gate import ShapSafetyGate

def main():
    print("⏳ Loading pre-processed data...")
    try:
        train_df = pd.read_csv("data/processed/train.csv", index_col=0, parse_dates=True)
        test_df = pd.read_csv("data/processed/test.csv", index_col=0, parse_dates=True)
    except FileNotFoundError:
        print("❌ Pre-split data not found. Run scripts/run_pipeline.py first.")
        return

    # Train the base hybrid model
    print("⏳ Fitting the hybrid volatility engine...")
    hybrid = HybridXGBoostVol(max_depth=2, learning_rate=0.01, n_estimators=1500)
    results_df, _, _ = hybrid.fit_and_predict(train_df, test_df)

    # Feature columns used by XGBoost
    cols_to_exclude = ['Target_Vol_Next_5d', 'Log_Ret', 'Nifty_Ret']
    feature_cols = [col for col in train_df.columns if col not in cols_to_exclude]
    
    # Extract training feature matrix
    X_train = train_df[feature_cols]
    
    # Take the first test row for testing
    test_row = test_df[feature_cols].iloc[[0]]
    test_date = test_df.index[0].date()

    print("\n" + "="*65)
    print("RUNNING SHAP SAFETY GATE CALIBRATION")
    print("="*65)

    # Instantiate our safety gate
    gate = ShapSafetyGate()
    
    # Fit Explainer & Compute Baselines (Phases 2d through 2g)
    gate.fit_explainer(hybrid.xgb_model, X_train)
    
    # -------------------------------------------------------------
    # PHASE 2f: Comparative Regime Analysis
    # -------------------------------------------------------------
    if gate.shap_means is not None and gate.covid_shap_means is not None:
        comparison_df = pd.DataFrame({
            '|SHAP| Normal Mean': gate.shap_means,
            '|SHAP| COVID Mean': gate.covid_shap_means,
            'Expansion Ratio (COVID/Normal)': gate.covid_shap_means / gate.shap_means
        })
        print("\nPhase 2f: Comparative Regime Attribution Analysis:")
        print(comparison_df.sort_values(by='Expansion Ratio (COVID/Normal)', ascending=False).to_string())
        print("-" * 65)

    # -------------------------------------------------------------
    # PHASE 2g: Domain Overrides Calibration Metrics
    # -------------------------------------------------------------
    if gate.covid_raw_medians is not None and gate.covid_raw_90th is not None:
        # We calculate the overall training standard deviations to compare magnitudes
        raw_train_stds = X_train.std()
        raw_train_means = X_train.mean()
        
        overrides_calibration_df = pd.DataFrame({
            'Train Mean (Normal)': raw_train_means,
            'COVID Median (50th)': gate.covid_raw_medians,
            'COVID Peak (90th)': gate.covid_raw_90th,
            'COVID Peak Z-Score': (gate.covid_raw_90th - raw_train_means) / raw_train_stds
        })
        print("\nPhase 2g: Raw Feature Stress Percentiles (Calibration Boundaries):")
        print(overrides_calibration_df.sort_values(by='COVID Peak Z-Score', ascending=False).to_string())
        print("-" * 65)

    # Calculate SHAP values for the first test row (Phase 2d verification)
    shap_vals = gate.compute_shap_values(test_row)
    row_shap = shap_vals[0]

    # Verify additivity
    reconstructed_pred = gate.base_value + np.sum(row_shap)
    actual_xgb_pred = hybrid.xgb_model.predict(test_row)[0]
    diff = np.abs(reconstructed_pred - actual_xgb_pred)
    
    print(f"First Test Day ({test_date}) Predictions:")
    print(f"  Reconstructed (Base + SHAP Sum):  {reconstructed_pred:.6f}")
    print(f"  Actual XGBoost Output:            {actual_xgb_pred:.6f}")
    print(f"  Difference:                       {diff:.2e}")
    
    if diff < 1e-5:
        print(" Calibration Successful!")
    else:
        print(" Verification Failed.")

if __name__ == "__main__":
    main()