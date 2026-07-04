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
    print(" Loading pre-processed data...")
    try:
        train_df = pd.read_csv("data/processed/train.csv", index_col=0, parse_dates=True)
        test_df = pd.read_csv("data/processed/test.csv", index_col=0, parse_dates=True)
    except FileNotFoundError:
        print(" Pre-split data not found. Run scripts/run_pipeline.py first.")
        return

    # Train the base hybrid model
    print(" Fitting the hybrid volatility engine...")
    hybrid = HybridXGBoostVol(max_depth=2, learning_rate=0.01, n_estimators=1500)
    results_df, _, _ = hybrid.fit_and_predict(train_df, test_df)

    # Feature columns used by XGBoost
    cols_to_exclude = ['Target_Vol_Next_5d', 'Log_Ret', 'Nifty_Ret']
    feature_cols = [col for col in train_df.columns if col not in cols_to_exclude]
    
    # Extract training feature matrix
    X_train = train_df[feature_cols]

    # Instantiate our safety gate. Calibrated threshold_std to 4.0 to prevent false positives.
    gate = ShapSafetyGate(max_absolute_adj=0.03, max_relative_adj=0.25, threshold_std=4.0, max_concentration_ratio=0.45)
    
    # Fit explainer and baselines
    gate.fit_explainer(hybrid.xgb_model, X_train)

    print("\n" + "="*75)
    print("RUNNING ACTIVE GATE SIMULATION")
    print("="*75)

    # 1. Simulate a Standard Day
    test_row = test_df[feature_cols].iloc[[0]]
    egarch_pred = results_df['EGARCH_Base'].iloc[0]
    xgb_adjustment = results_df['XGB_Adjustment'].iloc[0]

    status, reason, diags = gate.evaluate_prediction_safety(test_row, egarch_pred, xgb_adjustment)
    print(f"Scenario 1: Standard Day ({test_df.index[0].date()})")
    print(f"  Decision Status: **{status}** | Reason Code: {reason}")
    print(f"  Diagnostics: {diags}")
    print("-" * 75)

    # 2. Simulate an Extreme Domain Override Day
    anomalous_row = test_row.copy()
    anomalous_row['VIX_Lag_1'] = 55.0  # Exceeds our hardcoded limit of 50.0

    status, reason, diags = gate.evaluate_prediction_safety(anomalous_row, egarch_pred, xgb_adjustment)
    print("Scenario 2: Extreme Crash Day (VIX Spikes to 55.0)")
    print(f"  Decision Status: **{status}** | Reason Code: {reason}")
    print(f"  Diagnostics: {diags}")
    print("-" * 75)

    # 3. Simulate an Extreme Volatility Correction Day
    inflated_adjustment = 0.04  # Exceeds our absolute max limit of 0.03

    status, reason, diags = gate.evaluate_prediction_safety(test_row, egarch_pred, inflated_adjustment)
    print("Scenario 3: Unstable Volatility Correction Day (Adjustment: +4.0% Vol)")
    print(f"  Decision Status: **{status}** | Reason Code: {reason}")
    print(f"  Diagnostics: {diags}")
    print("-" * 75)

    # 4. Simulate a SHAP OOD Day (Phase 2i)
    ood_row = test_row.copy()
    ood_row['Vol_10d'] = 0.45 

    status, reason, diags = gate.evaluate_prediction_safety(ood_row, egarch_pred, xgb_adjustment)
    print("Scenario 4: SHAP Out-of-Distribution Day (Attribution Anomalous)")
    print(f"  Decision Status: **{status}** | Reason Code: {reason}")
    print(f"  Diagnostics: {diags}")
    print("-" * 75)

    # 5. Simulate a SHAP Concentration Violation Day (Phase 2j)
    concentrated_row = test_row.copy()
    concentrated_row['VIX_Gap'] = 9.5  # Substantial but under absolute override limit

    status, reason, diags = gate.evaluate_prediction_safety(concentrated_row, egarch_pred, xgb_adjustment)
    print("Scenario 5: SHAP Concentration Violation Day (Single Feature Dominated)")
    print(f"  Decision Status: **{status}** | Reason Code: {reason}")
    print(f"  Diagnostics: {diags}")
    print("-" * 75)

    if status == "REJECTED" and reason == "SHAP_CONCENTRATION_LIMIT":
        print(" Phase 2j Active Verification Successful! Concentration Guard fired correctly.")
    else:
        print(" Verification Failed.")

if __name__ == "__main__":
    main()