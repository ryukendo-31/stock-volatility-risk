# scripts/run_hybrid.py
import pandas as pd
import numpy as np
import sys
import os
import mlflow
from sklearn.metrics import mean_squared_error, mean_absolute_error

from src.models.xgboost_vol import HybridXGBoostVol
from src.evaluation.diagnostics import diebold_mariano_test, compute_qlike_loss

def main():
    print("Starting the Integrated Hybrid Volatility Prediction Engine...")
    
    # Use SQLite (required by modern MLflow). It is safely ignored by Git via .gitignore!
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Hybrid_Vol_Prediction_Engine")

    try:
        train_df = pd.read_csv("data/processed/train.csv", index_col=0, parse_dates=True)
        test_df = pd.read_csv("data/processed/test.csv", index_col=0, parse_dates=True)
    except FileNotFoundError:
        print("Data not found. Run scripts/run_pipeline.py first.")
        return

    print(f"   Train Features Shape: {train_df.shape} | Test Features Shape: {test_df.shape}")
    print("-" * 75)

    max_depth = 2
    learning_rate = 0.01
    n_estimators = 1500
    reg_alpha = 5.0
    reg_lambda = 10.0
    threshold_std = 3.5            
    max_concentration_ratio = 0.80 

    run_name = "Integrated_Hybrid_Production_Run"

    with mlflow.start_run(run_name=run_name):
        hybrid_engine = HybridXGBoostVol(
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            threshold_std=threshold_std,
            max_concentration_ratio=max_concentration_ratio
        )
        
        results_df, egarch_base, hybrid_preds = hybrid_engine.fit_and_predict(train_df, test_df)
        
        actuals = test_df['Target_Vol_Next_5d']
        
        eg_rmse = np.sqrt(mean_squared_error(actuals, egarch_base))
        eg_mae = mean_absolute_error(actuals, egarch_base)
        eg_qlike = compute_qlike_loss(actuals, egarch_base)
        
        hybrid_rmse = np.sqrt(mean_squared_error(actuals, hybrid_preds))
        hybrid_mae = mean_absolute_error(actuals, hybrid_preds)
        hybrid_qlike = compute_qlike_loss(actuals, hybrid_preds)
        
        rmse_lift = ((eg_rmse - hybrid_rmse) / eg_rmse * 100) if eg_rmse > 0 else 0.0
        
        total_days = len(results_df)
        bypassed_days = sum(results_df['Gate_Decision'] == "REJECTED")
        bypass_rate = (bypassed_days / total_days) * 100
        
        print("\n" + "="*65)
        print("PRODUCTION HYBRID PERFORMANCE REPORT")
        print("="*65)
        print(f"EGARCH Base -> RMSE: {eg_rmse:.5f} | MAE: {eg_mae:.5f} | QLIKE: {eg_qlike:.5f}")
        print(f"Hybrid Final-> RMSE: {hybrid_rmse:.5f} | MAE: {hybrid_mae:.5f} | QLIKE: {hybrid_qlike:.5f}")
        print("-" * 65)
        print(f"Total Forecast Days:           {total_days}")
        print(f"Bypassed Days (EGARCH Fallback): {bypassed_days}")
        print(f"Active Safety Gate Bypass Rate: {bypass_rate:.2f}%")
        print(f"Out-of-sample Performance Lift: {rmse_lift:+.2f}%")
        
        print("\nActive Gate Bypass Breakdown (Reason Codes):")
        reasons_counts = results_df['Gate_Reason'].value_counts()
        for reason, count in reasons_counts.items():
            if reason != "NONE":
                prop = (count / bypassed_days) * 100
                print(f"  - {reason:<25}: {count:>3} days ({prop:.1f}%)")

        mlflow.log_params({
            "xgb_max_depth": max_depth,
            "xgb_learning_rate": learning_rate,
            "xgb_n_estimators": n_estimators,
            "xgb_reg_alpha": reg_alpha,
            "xgb_reg_lambda": reg_lambda,
            "gate_threshold_std": threshold_std,
            "gate_max_concentration": max_concentration_ratio,
            "backtest_type": "Static_Evaluation_With_Safety_Gate"
        })
        
        mlflow.log_metrics({
            "EGARCH_RMSE": eg_rmse,
            "EGARCH_MAE": eg_mae,
            "EGARCH_QLIKE": eg_qlike,
            "Hybrid_RMSE": hybrid_rmse,
            "Hybrid_MAE": hybrid_mae,
            "Hybrid_QLIKE": hybrid_qlike,
            "RMSE_Lift_Pct": rmse_lift,
            "Gate_Bypasses": bypassed_days, # FIX B7: Removed broken reference
            "Gate_Bypass_Rate_Pct": bypass_rate
        })
        
        dm_stat, dm_pvalue = diebold_mariano_test(actuals, egarch_base, hybrid_preds, h=5)
        print("-" * 65)
        print(f"Diebold-Mariano Test (EGARCH vs Hybrid) -> Stat: {dm_stat:.4f} | p-value: {dm_pvalue:.4f}")
        
        mlflow.log_metrics({
            "DM_Stat": dm_stat,
            "DM_pvalue": dm_pvalue
        })
        
        if dm_pvalue < 0.05:
            better_model = "Hybrid" if dm_stat > 0 else "EGARCH"
            print(f"Conclusion: The machine learning residuals adjustment is STATISTICALLY SIGNIFICANT. Model favored: {better_model}")
        else:
            print("Conclusion: The machine learning residuals adjustment did not result in a statistically significant change.")
            
    print("\nHybrid training and evaluation logged successfully!")

if __name__ == "__main__":
    main()