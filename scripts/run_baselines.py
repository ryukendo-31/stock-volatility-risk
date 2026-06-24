# scripts/run_baselines.py
import pandas as pd
import numpy as np
import sys
import os
import mlflow

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.naive import NaiveModel
from src.models.linear_lag import LinearBaseline
from src.models.garch import GarchModel
from src.models.egarch import EgarchModel
from src.models.garch_x import GarchXModel
from src.evaluation.diagnostics import calculate_statistical_diagnostics, diebold_mariano_test, compute_qlike_loss

def run_fight():
    print("🚀 Starting the Model Comparison & MLflow Logging...")
    
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Volatility_Prediction_Models")

    try:
        train_df = pd.read_csv("data/processed/train.csv", index_col=0, parse_dates=True)
        test_df = pd.read_csv("data/processed/test.csv", index_col=0, parse_dates=True)
    except FileNotFoundError:
        print("❌ Data not found. Run scripts/run_pipeline.py first.")
        return

    print(f"   Train: {len(train_df)} | Test: {len(test_df)}")
    print("-" * 65)

    garch_preds = None
    egarch_preds = None

    # 1. Naive Model
    with mlflow.start_run(run_name="Baseline_Naive"):
        print("\n[1/5] Running Naive Model...")
        naive = NaiveModel()
        metrics = naive.evaluate(test_df)
        qlike_val = compute_qlike_loss(test_df['Target_Vol_Next_5d'], naive.predict(test_df))
        
        print(f"      RMSE:  {metrics['RMSE']:.5f} | QLIKE: {qlike_val:.5f}")
        mlflow.log_param("model_family", "Baseline")
        mlflow.log_metric("RMSE", metrics['RMSE'])
        mlflow.log_metric("MAE", metrics['MAE'])
        mlflow.log_metric("QLIKE", qlike_val)

    # 2. Linear Model
    with mlflow.start_run(run_name="Baseline_Linear_Lag"):
        print("\n[2/5] Running Linear Model...")
        linear = LinearBaseline()
        linear.train(train_df)
        metrics = linear.evaluate(test_df)
        qlike_val = compute_qlike_loss(test_df['Target_Vol_Next_5d'], linear.predict(test_df))
        
        print(f"      RMSE:  {metrics['RMSE']:.5f} | QLIKE: {qlike_val:.5f}")
        mlflow.log_param("model_family", "Regression")
        mlflow.log_metric("RMSE", metrics['RMSE'])
        mlflow.log_metric("MAE", metrics['MAE'])
        mlflow.log_metric("QLIKE", qlike_val)

    # 3. Standard GARCH(1,1) Model (Student-t)
    garch_fit_res = None
    if 'Log_Ret' in train_df.columns:
        with mlflow.start_run(run_name="Baseline_GARCH_1_1"):
            print("\n[3/5] Running Standard GARCH Model (t-distribution)...")
            garch = GarchModel()
            metrics, garch_fit_res, garch_preds = garch.evaluate(train_df, test_df)
            print(f"      RMSE:  {metrics['RMSE']:.5f}")
            
            mlflow.log_param("model_family", "Volatility")
            mlflow.log_param("variant", "Standard_GARCH")
            mlflow.log_metric("RMSE", metrics['RMSE'])
            
            garch_diags = calculate_statistical_diagnostics(garch_fit_res, test_df['Target_Vol_Next_5d'], garch_preds)
            mlflow.log_metric("AIC", garch_diags["AIC"])
            mlflow.log_metric("BIC", garch_diags["BIC"])
            mlflow.log_metric("MAE", garch_diags["MAE"])
            mlflow.log_metric("QLIKE", garch_diags["QLIKE"])
            mlflow.log_metric("LjungBox_Stat", garch_diags["LjungBox_Stat"])
            mlflow.log_metric("LjungBox_pvalue", garch_diags["LjungBox_pvalue"])
            mlflow.log_metric("ARCH_LM_Stat", garch_diags["ARCH_LM_Stat"])
            mlflow.log_metric("ARCH_LM_pvalue", garch_diags["ARCH_LM_pvalue"])
            mlflow.log_metric("JarqueBera_Stat", garch_diags["JarqueBera_Stat"])
            mlflow.log_metric("JarqueBera_pvalue", garch_diags["JarqueBera_pvalue"])

    # 4. EGARCH(1,1,1) Model (Student-t)
    egarch_fit_res = None
    if 'Log_Ret' in train_df.columns:
        with mlflow.start_run(run_name="Baseline_EGARCH_1_1_1"):
            print("\n[4/5] Running EGARCH Model (t-distribution)...")
            egarch = EgarchModel()
            metrics, egarch_fit_res, egarch_preds = egarch.evaluate(train_df, test_df)
            print(f"      RMSE:  {metrics['RMSE']:.5f}")
            
            mlflow.log_param("model_family", "Volatility")
            mlflow.log_param("variant", "EGARCH")
            mlflow.log_metric("RMSE", metrics['RMSE'])
            
            egarch_diags = calculate_statistical_diagnostics(egarch_fit_res, test_df['Target_Vol_Next_5d'], egarch_preds)
            mlflow.log_metric("AIC", egarch_diags["AIC"])
            mlflow.log_metric("BIC", egarch_diags["BIC"])
            mlflow.log_metric("MAE", egarch_diags["MAE"])
            mlflow.log_metric("QLIKE", egarch_diags["QLIKE"])
            mlflow.log_metric("LjungBox_Stat", egarch_diags["LjungBox_Stat"])
            mlflow.log_metric("LjungBox_pvalue", egarch_diags["LjungBox_pvalue"])
            mlflow.log_metric("ARCH_LM_Stat", egarch_diags["ARCH_LM_Stat"])
            mlflow.log_metric("ARCH_LM_pvalue", egarch_diags["ARCH_LM_pvalue"])
            mlflow.log_metric("JarqueBera_Stat", egarch_diags["JarqueBera_Stat"])
            mlflow.log_metric("JarqueBera_pvalue", egarch_diags["JarqueBera_pvalue"])

    # 5. Comparative Statistical Verification
    print("\n" + "="*65)
    print("📊 STATISTICAL DIAGNOSTIC REPORT (GARCH vs EGARCH)")
    print("="*65)
    
    if garch_fit_res is not None and egarch_fit_res is not None:
        dm_stat, dm_pvalue = diebold_mariano_test(test_df['Target_Vol_Next_5d'], garch_preds, egarch_preds, h=5)
        
        with mlflow.start_run(run_name="GARCH_vs_EGARCH_DM_Test"):
            mlflow.log_metric("DM_Stat", dm_stat)
            mlflow.log_metric("DM_pvalue", dm_pvalue)

        print(f"Goodness of Fit (t-distribution):")
        print(f"  GARCH(1,1)  -> AIC: {garch_diags['AIC']:.2f} | BIC: {garch_diags['BIC']:.2f}")
        print(f"  EGARCH(1,1) -> AIC: {egarch_diags['AIC']:.2f} | BIC: {egarch_diags['BIC']:.2f}")
        
        print(f"\nResidual Diagnostics (p-values > 0.05 indicates well-specified residuals):")
        print(f"  Ljung-Box (Squared Standardized Residuals):")
        print(f"    GARCH(1,1)  p-value: {garch_diags['LjungBox_pvalue']:.4f}")
        print(f"    EGARCH(1,1) p-value: {egarch_diags['LjungBox_pvalue']:.4f}")
        
        print(f"  ARCH LM Test (Heteroskedasticity remaining):")
        print(f"    GARCH(1,1)  p-value: {garch_diags['ARCH_LM_pvalue']:.4f}")
        print(f"    EGARCH(1,1) p-value: {egarch_diags['ARCH_LM_pvalue']:.4f}")

        print(f"  Jarque-Bera Test (p-value):")
        print(f"    GARCH(1,1)  p-value: {garch_diags['JarqueBera_pvalue']:.4f} (Stat: {garch_diags['JarqueBera_Stat']:.4f})")
        print(f"    EGARCH(1,1) p-value: {egarch_diags['JarqueBera_pvalue']:.4f} (Stat: {garch_diags['JarqueBera_Stat']:.4f})")

        print(f"\nPredictive Metrics (Out-of-sample Test Set):")
        print(f"  GARCH(1,1)  -> MAE: {garch_diags['MAE']:.5f} | QLIKE: {garch_diags['QLIKE']:.5f}")
        print(f"  EGARCH(1,1) -> MAE: {egarch_diags['MAE']:.5f} | QLIKE: {egarch_diags['QLIKE']:.5f}")
        
        print(f"\nDiebold-Mariano Test (H0: Forecast accuracy is identical):")
        print(f"  DM Statistic: {dm_stat:.4f} | p-value: {dm_pvalue:.4f}")
        if dm_pvalue < 0.05:
            better_model = "EGARCH" if dm_stat > 0 else "GARCH"
            print(f"  Conclusion: The predictive difference is statistically significant. Model favoring: {better_model}")
        else:
            print("  Conclusion: No statistically significant predictive performance difference detected.")

    print("-" * 65)
    print("\n[5/5] EXECUTING EXOGENOUS NATIVE GARCH-X PIPELINE 🔥")
    
    garch_scenarios = [
        (False, False), # Base Model
        (True, False),  # Nifty Only
        (False, True),  # VIX Only
        (True, True)    # Nifty + VIX (Hybrid)
    ]
    
    for use_nifty, use_vix in garch_scenarios:
        garch_x = GarchXModel(use_nifty=use_nifty, use_vix=use_vix)
        with mlflow.start_run(run_name=garch_x.model_name):
            results, _, _ = garch_x.evaluate(train_df, test_df)
            print(f"      >> FINAL RMSE: {results['RMSE']:.5f}\n")
            
            mlflow.log_param("model_family", "GARCH-X")
            mlflow.log_param("uses_nifty", use_nifty)
            mlflow.log_param("uses_vix", use_vix)
            mlflow.log_params(results["Params"])
            mlflow.log_metric("RMSE", results["RMSE"])

    print("\nAll models finished and logged to local MLflow database!")

if __name__ == "__main__":
    run_fight()