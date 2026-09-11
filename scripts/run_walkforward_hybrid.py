# scripts/run_walkforward_hybrid.py
import pandas as pd
import numpy as np
import os
import mlflow
from sklearn.metrics import mean_squared_error, mean_absolute_error

from src.models.xgboost_vol import HybridXGBoostVol
from src.evaluation.diagnostics import diebold_mariano_test, compute_qlike_loss

def main():
    print("Starting the Expanding-Window Walk-Forward Hybrid Backtest...")
    
    # Configure MLflow
    mlflow.set_experiment("Walk_Forward_Volatility_Prediction")

    try:
        features_df = pd.read_csv("data/processed/features.csv", index_col=0, parse_dates=True)
    except FileNotFoundError:
        print(" features.csv not found. Run scripts/run_pipeline.py first.")
        return

    # Ensure contiguous, non-NaN records are used
    features_df = features_df.dropna(subset=['Log_Ret', 'Nifty_Ret', 'VIX_Gap', 'Target_Vol_Next_5d'])
    n_records = len(features_df)
    
    # Configuration
    start_window = 3000
    step_size = 252
    
    print(f"   Total Records: {n_records} | Initial Train: {start_window} | Step Size: {step_size}")
    print("-" * 75)

    # Calibrated production parameters
    max_depth = 2
    learning_rate = 0.01
    n_estimators = 1500
    reg_alpha = 5.0
    reg_lambda = 10.0
    threshold_std = 3.5            
    max_concentration_ratio = 0.80 

    all_results = []
    
    fold = 1
    for start_idx in range(start_window, n_records, step_size):
        train_slice = features_df.iloc[:start_idx]
        test_slice = features_df.iloc[start_idx : start_idx + step_size]
        
        if len(test_slice) == 0:
            break
            
        print(f"\n=======================================================")
        print(f" Processing Fold {fold} | Out-of-sample: {test_slice.index[0].date()} to {test_slice.index[-1].date()} ({len(test_slice)} days)")
        print(f"=======================================================")
        
        hybrid_engine = HybridXGBoostVol(
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            threshold_std=threshold_std,
            max_concentration_ratio=max_concentration_ratio
        )
        
        # We suppress some of the repetitive print statements from inside the class during the loop
        fold_results, egarch_base, hybrid_preds = hybrid_engine.fit_and_predict(train_slice, test_slice)
        all_results.append(fold_results)
        
        fold += 1

    # Concatenate all walk-forward out-of-sample predictions
    final_wf_results = pd.concat(all_results)
    
    # Save the master walk-forward predictions
    os.makedirs("results", exist_ok=True)
    final_wf_results.to_csv("results/walkforward_hybrid_predictions.csv")

    actuals = final_wf_results['Actual']
    egarch_preds = final_wf_results['EGARCH_Base']
    hybrid_preds = final_wf_results['Hybrid_Final']

    # Calculate global performance metrics across all folds
    eg_rmse = np.sqrt(mean_squared_error(actuals, egarch_preds))
    eg_mae = mean_absolute_error(actuals, egarch_preds)
    eg_qlike = compute_qlike_loss(actuals, egarch_preds)
    
    hybrid_rmse = np.sqrt(mean_squared_error(actuals, hybrid_preds))
    hybrid_mae = mean_absolute_error(actuals, hybrid_preds)
    hybrid_qlike = compute_qlike_loss(actuals, hybrid_preds)
    
    rmse_lift = (eg_rmse - hybrid_rmse) / eg_rmse * 100
    
    total_days = len(final_wf_results)
    bypassed_days = sum(final_wf_results['Gate_Decision'] == "REJECTED")
    bypass_rate = (bypassed_days / total_days) * 100
    
    print("\n" + "="*75)
    print(" MASTER WALK-FORWARD HYBRID PERFORMANCE REPORT")
    print("="*75)
    print(f"Total Out-of-Sample Days Evaluated: {total_days}")
    print(f"EGARCH Base -> RMSE: {eg_rmse:.5f} | MAE: {eg_mae:.5f} | QLIKE: {eg_qlike:.5f}")
    print(f"Hybrid Final-> RMSE: {hybrid_rmse:.5f} | MAE: {hybrid_mae:.5f} | QLIKE: {hybrid_qlike:.5f}")
    print("-" * 75)
    print(f"Overall Bypassed Days: {bypassed_days}")
    print(f"Overall Safety Gate Bypass Rate: {bypass_rate:.2f}%")
    print(f"True Out-of-sample Performance Lift: {rmse_lift:+.2f}%")
    
    print("\nActive Gate Bypass Breakdown (Reason Codes across all folds):")
    reasons_counts = final_wf_results['Gate_Reason'].value_counts()
    for reason, count in reasons_counts.items():
        if reason != "NONE":
            prop = (count / bypassed_days) * 100
            print(f"  - {reason:<25}: {count:>3} days ({prop:.1f}%)")

    # MLflow Logging
    with mlflow.start_run(run_name="WF_Integrated_Hybrid_Run"):
        mlflow.log_params({
            "xgb_max_depth": max_depth,
            "xgb_learning_rate": learning_rate,
            "xgb_n_estimators": n_estimators,
            "xgb_reg_alpha": reg_alpha,
            "xgb_reg_lambda": reg_lambda,
            "gate_threshold_std": threshold_std,
            "gate_max_concentration": max_concentration_ratio,
            "backtest_type": "Walk_Forward_Expanding"
        })
        
        mlflow.log_metrics({
            "EGARCH_RMSE": eg_rmse,
            "EGARCH_MAE": eg_mae,
            "EGARCH_QLIKE": eg_qlike,
            "Hybrid_RMSE": hybrid_rmse,
            "Hybrid_MAE": hybrid_mae,
            "Hybrid_QLIKE": hybrid_qlike,
            "RMSE_Lift_Pct": rmse_lift,
            "Gate_Bypasses": bypassed_days,
            "Gate_Bypass_Rate_Pct": bypass_rate
        })
        
        dm_stat, dm_pvalue = diebold_mariano_test(actuals, egarch_preds, hybrid_preds, h=5)
        print("-" * 75)
        print(f"Diebold-Mariano Test (EGARCH vs Hybrid) -> Stat: {dm_stat:.4f} | p-value: {dm_pvalue:.4f}")
        
        mlflow.log_metrics({
            "DM_Stat": dm_stat,
            "DM_pvalue": dm_pvalue
        })
        
        if dm_pvalue < 0.05:
            better_model = "Hybrid" if dm_stat > 0 else "EGARCH"
            print(f"Conclusion: {better_model} is statistically superior across the entire walk-forward test!")
        else:
            print("Conclusion: No statistically significant difference in accuracy detected over the long run.")

if __name__ == "__main__":
    main()