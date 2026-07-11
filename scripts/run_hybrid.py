# scripts/run_hybrid.py
import pandas as pd
import numpy as np
import sys
import os
import mlflow
from sklearn.metrics import mean_squared_error, mean_absolute_error

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.xgboost_vol import HybridXGBoostVol
from src.evaluation.diagnostics import diebold_mariano_test, compute_qlike_loss

def main():
    print("Starting the Integrated Hybrid Volatility Prediction Engine...")
    
    # Configure local SQLite database tracking
    mlflow.set_tracking_uri("sqlite:///mlflow.db")#initiate the mlflow database
    mlflow.set_experiment("Hybrid_Vol_Prediction_Engine")# creating an experiment file

    # Load pre-split data
    try:
        train_df = pd.read_csv("data/processed/train.csv", index_col=0, parse_dates=True)
        test_df = pd.read_csv("data/processed/test.csv", index_col=0, parse_dates=True)
    except FileNotFoundError:
        print("Data not found. Run scripts/run_pipeline.py first.")
        return

    print(f"   Train Features Shape: {train_df.shape} | Test Features Shape: {test_df.shape}")
    print("-" * 75)

    # Calibrated production parameters
    max_depth = 2
    learning_rate = 0.01
    n_estimators = 1500
    reg_alpha = 5.0
    reg_lambda = 10.0
    threshold_std = 3.5            # 3.5 std for OOD SHAP
    max_concentration_ratio = 0.80 # 80% concentration limit

    run_name = "Integrated_Hybrid_Production_Run"

    with mlflow.start_run(run_name=run_name):
        # Initialize and run the integrated hybrid engine
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
        
        # Calculate performance metrics
        eg_rmse = np.sqrt(mean_squared_error(actuals, egarch_base))
        eg_mae = mean_absolute_error(actuals, egarch_base)
        eg_qlike = compute_qlike_loss(actuals, egarch_base)
        
        hybrid_rmse = np.sqrt(mean_squared_error(actuals, hybrid_preds))
        hybrid_mae = mean_absolute_error(actuals, hybrid_preds)
        hybrid_qlike = compute_qlike_loss(actuals, hybrid_preds)
        
        rmse_lift = (eg_rmse - hybrid_rmse) / eg_rmse * 100
        
        # Calculate Gate Statistics (Phase 2o Auditing)
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
        
        # Breakdown the bypass reasons to verify which guards are active
        print("\nActive Gate Bypass Breakdown (Reason Codes):")
        reasons_counts = results_df['Gate_Reason'].value_counts()
        for reason, count in reasons_counts.items():
            if reason != "NONE":
                prop = (count / bypassed_days) * 100
                print(f"  - {reason:<25}: {count:>3} days ({prop:.1f}%)")

        # Log hyperparameters to MLflow
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
        
        # Log performance and auditing metrics to MLflow
        mlflow.log_metrics({
            "EGARCH_RMSE": eg_rmse,
            "EGARCH_MAE": eg_mae,
            "EGARCH_QLIKE": eg_qlike,
            "Hybrid_RMSE": hybrid_rmse,
            "Hybrid_MAE": hybrid_mae,
            "Hybrid_QLIKE": hybrid_qlike,
            "RMSE_Lift_Pct": rmse_lift,
            "Gate_Bypasses": bypassed_predictions if 'bypassed_predictions' in locals() else bypassed_days,
            "Gate_Bypass_Rate_Pct": bypass_rate
        })
        
        # Compute and log Diebold-Mariano significance values
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