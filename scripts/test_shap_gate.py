# scripts/test_shap_gate.py
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.xgboost_vol import HybridXGBoostVol

def print_gate_report(df_slice, label):
    total_days = len(df_slice)
    if total_days == 0:
        print(f"Skipping {label} - No valid days found.")
        return 0.0, 0.0
        
    rejected_count = sum(df_slice['Gate_Decision'] == "REJECTED")
    approved_count = total_days - rejected_count
    
    approval_rate = (approved_count / total_days) * 100
    bypass_rate = (rejected_count / total_days) * 100
    
    print("\n" + "="*65)
    print(f"OUT-OF-SAMPLE SIMULATION: {label}")
    print("="*65)
    print(f"Total Evaluated Days: {total_days}")
    print(f"Approved Days:        {approved_count} ({approval_rate:.2f}%)")
    print(f"Bypassed Days:        {rejected_count} ({bypass_rate:.2f}%)")
    
    if rejected_count > 0:
        print("\nBypass Trigger Breakdown (Reason Codes):")
        reasons = df_slice[df_slice['Gate_Decision'] == "REJECTED"]['Gate_Reason'].value_counts()
        for reason, count in reasons.items():
            prop = (count / rejected_count) * 100
            print(f"  - {reason:<25}: {count:>3} days ({prop:.1f}%)")
            
    return approval_rate, bypass_rate

def main():
    print("Loading pre-processed data...")
    try:
        # Load the raw features directly so we can make a custom chronological split
        features_df = pd.read_csv("data/processed/features.csv", index_col=0, parse_dates=True)
        features_df = features_df.dropna(subset=['Log_Ret', 'Nifty_Ret', 'VIX_Gap', 'Target_Vol_Next_5d'])
    except FileNotFoundError:
        print("features.csv not found. Run scripts/run_pipeline.py first.")
        return

    # ----------------------------------------------------------------------
    # STRICT OUT-OF-SAMPLE SPLIT
    # Train only up to 2016. The model has never seen 2017 or COVID (2020).
    # ----------------------------------------------------------------------
    train_df = features_df.loc[:'2016-12-31']
    test_df = features_df.loc['2017-01-01':'2020-12-31']

    print(f"Training on historical data: {train_df.index[0].date()} to {train_df.index[-1].date()}")
    print(f"Testing on UNSEEN future data: {test_df.index[0].date()} to {test_df.index[-1].date()}")

    hybrid = HybridXGBoostVol(
        max_depth=2, 
        learning_rate=0.01, 
        n_estimators=1500,
        threshold_std=4.5,            
        max_concentration_ratio=0.90, 
        min_rank_correlation=0.40
    )
    
    # Run the engine. This outputs the predictions and gate decisions for the UNSEEN test set.
    results_df, _, _ = hybrid.fit_and_predict(train_df, test_df)

    # -------------------------------------------------------------
    # PHASE 2l: COVID Stress Fold (Unseen Black Swan)
    # -------------------------------------------------------------
    covid_results = results_df.loc['2020-02-01':'2020-09-30']
    covid_approval, covid_bypass = print_gate_report(
        df_slice=covid_results,
        label="Phase 2l - COVID-19 Stress Regime (Strictly OOS)"
    )

    # -------------------------------------------------------------
    # PHASE 2m: Normal Regime (Unseen Calm Market)
    # -------------------------------------------------------------
    calm_results = results_df.loc['2017-01-01':'2019-12-31']
    calm_approval, calm_bypass = print_gate_report(
        df_slice=calm_results,
        label="Phase 2m - Normal/Calm Market Regime (Strictly OOS)"
    )

    # -------------------------------------------------------------
    # FINAL METRICS VERIFICATION CHECKS
    # -------------------------------------------------------------
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
        print("\nVerification Successful! Gate correctly isolates unseen Black Swans.")
    else:
        print("\nVerification Failed. Gate thresholds require recalibration.")

if __name__ == "__main__":
    main()