# scripts/test_shap_gate.py
import pandas as pd
import numpy as np
import sys
import os

from src.models.xgboost_vol import HybridXGBoostVol
from src.decision.shap_gate import ShapSafetyGate

def run_bulk_simulation(gate, df, hybrid, feature_cols, label, egarch_train_preds):
    total_days = len(df)
    if total_days == 0:
        print(f"Skipping {label} - No valid overlapping days found in this slice.")
        return 0.0, 0.0
    
    egarch_preds = egarch_train_preds.loc[df.index]
    xgb_adjustments = hybrid.xgb_model.predict(df[feature_cols])
    
    approved_count = 0
    rejected_count = 0
    reasons_breakdown = {}
    
    gate.shap_history.clear()
    
    for idx in range(total_days):
        row = df[feature_cols].iloc[[idx]]
        eg_pred = egarch_preds.iloc[idx]
        xgb_adj = xgb_adjustments[idx]
        
        status, reason, diags = gate.evaluate_prediction_safety(row, eg_pred, xgb_adj)
        
        if status == "APPROVED":
            approved_count += 1
        else:
            rejected_count += 1
            reasons_breakdown[reason] = reasons_breakdown.get(reason, 0) + 1
            
    approval_rate = (approved_count / total_days) * 100
    bypass_rate = (rejected_count / total_days) * 100
    
    print("\n" + "="*65)
    print(f"BULK SIMULATION REPORT: {label}")
    print("="*65)
    print(f"Total Evaluated Days: {total_days}")
    print(f"Approved Days:        {approved_count} ({approval_rate:.2f}%)")
    print(f"Bypassed Days:        {rejected_count} ({bypass_rate:.2f}%)")
    
    if rejected_count > 0:
        print("\nBypass Trigger Breakdown (Reason Codes):")
        for reason, count in reasons_breakdown.items():
            prop = (count / rejected_count) * 100
            print(f"  - {reason:<25}: {count:>3} days ({prop:.1f}%)")
            
    return approval_rate, bypass_rate

def main():
    print("Loading pre-processed data...")
    try:
        train_df = pd.read_csv("data/processed/train.csv", index_col=0, parse_dates=True)
        test_df = pd.read_csv("data/processed/test.csv", index_col=0, parse_dates=True)
    except FileNotFoundError:
        print("Pre-split data not found. Run scripts/run_pipeline.py first.")
        return

    print("Fitting the hybrid volatility engine...")
    hybrid = HybridXGBoostVol(max_depth=2, learning_rate=0.01, n_estimators=1500)
    
    _, _, _ = hybrid.fit_and_predict(train_df, test_df)

    print("Generating simulated historical 5-day forecasts for gate evaluation...")
    
    np.random.seed(42)
    burn_in = train_df.index[252]
    eg_train_forecasts = hybrid.egarch_fit.forecast(start=burn_in, horizon=5, align='origin', method='simulation', simulations=1000)
    train_mean_variance = eg_train_forecasts.variance.mean(axis=1)
    egarch_train_preds = np.sqrt(train_mean_variance) / 100 * np.sqrt(252)

    cols_to_exclude = ['Target_Vol_Next_5d', 'Log_Ret', 'Nifty_Ret']
    feature_cols = [col for col in train_df.columns if col not in cols_to_exclude]
    
    # FIX: Ensure we only keep indices that exist in BOTH train_df and the predictions
    valid_idx = egarch_train_preds.dropna().index.intersection(train_df.index)
    X_train = train_df.loc[valid_idx, feature_cols]

    gate = ShapSafetyGate(
        max_absolute_adj=0.05,        
        max_relative_adj=0.45,        
        threshold_std=4.5,            
        max_concentration_ratio=0.90, 
        min_rank_correlation=0.40     
    )
    
    gate.fit_explainer(hybrid.xgb_model, X_train)

    covid_df = train_df.loc['2020-02-01':'2020-09-30']
    # Intersect again to prevent KeyErrors for dates lost in the burn-in
    covid_df = covid_df[covid_df.index.isin(valid_idx)]
    
    covid_approval, covid_bypass = run_bulk_simulation(
        gate=gate,
        df=covid_df,
        hybrid=hybrid,
        feature_cols=feature_cols,
        label="Phase 2l - COVID-19 Stress Regime",
        egarch_train_preds=egarch_train_preds
    )

    calm_df = train_df.loc['2017-01-01':'2019-12-31']
    calm_df = calm_df[calm_df.index.isin(valid_idx)]
    
    calm_approval, calm_bypass = run_bulk_simulation(
        gate=gate,
        df=calm_df,
        hybrid=hybrid,
        feature_cols=feature_cols,
        label="Phase 2m - Normal/Calm Market Regime",
        egarch_train_preds=egarch_train_preds
    )

    print("\n" + "="*65)
    print("FINAL CASCADE SAFETY CHECKS VERIFICATION")
    print("="*65)
    
    checks_passed = True
    
    if covid_bypass >= 60.0:
        print(f"Check 1 Passed: COVID Bypass Rate is {covid_bypass:.2f}% (Target: >= 60.0%)")
    else:
        print(f"Check 1 Failed: COVID Bypass Rate is {covid_bypass:.2f}% (Target: >= 60.0%)")
        checks_passed = False
        
    if calm_approval >= 85.0:
        print(f"Check 2 Passed: Calm Approval Rate is {calm_approval:.2f}% (Target: >= 85.0%)")
    else:
        print(f"Check 2 Failed: Calm Approval Rate is {calm_approval:.2f}% (Target: >= 85.0%)")
        checks_passed = False

    if checks_passed:
        print("\nPhase 2l & 2m Bulk Verification Successful! Gate is balanced.")
    else:
        print("\nVerification Failed. Gate thresholds require recalibration.")

if __name__ == "__main__":
    main()