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
    print("Loading pre-processed data...")
    try:
        train_df = pd.read_csv("data/processed/train.csv", index_col=0, parse_dates=True)
        test_df = pd.read_csv("data/processed/test.csv", index_col=0, parse_dates=True)
    except FileNotFoundError:
        print("Pre-split data not found. Run scripts/run_pipeline.py first.")
        return

    # Train the base hybrid model
    print("Fitting the hybrid volatility engine...")
    hybrid = HybridXGBoostVol(max_depth=2, learning_rate=0.01, n_estimators=1500)
    results_df, _, _ = hybrid.fit_and_predict(train_df, test_df)

    # Feature columns used by XGBoost
    cols_to_exclude = ['Target_Vol_Next_5d', 'Log_Ret', 'Nifty_Ret']
    feature_cols = [col for col in train_df.columns if col not in cols_to_exclude]
    
    # Extract training feature matrix
    X_train = train_df[feature_cols]

    # Instantiate our safety gate. Calibrated max_concentration_ratio to 0.70 to accommodate VIX_Gap dominance.
    gate = ShapSafetyGate(
        max_absolute_adj=0.03, 
        max_relative_adj=0.25, 
        threshold_std=4.0, 
        max_concentration_ratio=0.70, 
        min_rank_correlation=0.40
    )
    
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
    
    print(f"First Test Day Predictions:")
    print(f"  Decision Status:  {status}")
    print(f"  Reason Code:      {reason}")
    print(f"  Diagnostics:      {diags}")
    print("-" * 65)

    # 2. PHASE 2k: Verify Rank Instability Guard
    print("Scenario 6: Simulating Rank Instability over 10-day sliding window...")
    
    # Reset the sliding history queue
    gate.shap_history.clear()
    
    # We populate the first 9 days of history with a highly warped importance ranking 
    # (reversing feature importance order: making low-impact variables highly active, and VIX_Gap zero)
    n_features = len(feature_cols)
    warped_attribution = np.linspace(0.000001, 0.05, n_features) # Flipped ranking order
    
    for day in range(9):
        gate.shap_history.append(warped_attribution)
        
    # Evaluate on the 10th day with standard features. The combined average ranking of the sliding 
    # window will be completely decoupled from the training baseline, triggering a bypass.
    status, reason, diags = gate.evaluate_prediction_safety(test_row, egarch_pred, xgb_adjustment)
    print(f"Scenario 6: 10th Day Evaluation Outcome:")
    print(f"  Decision Status: **{status}** | Reason Code: {reason}")
    print(f"  Diagnostics: {diags}")
    print("-" * 75)

    if status == "REJECTED" and reason == "SHAP_RANK_INSTABILITY":
        print("Phase 2k Active Verification Successful! Rank Instability Guard fired correctly.")
    else:
        print("Verification Failed.")

if __name__ == "__main__":
    main()