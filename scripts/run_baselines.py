# scripts/run_baselines.py
import pandas as pd
import numpy as np
import sys
import os
import mlflow
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.naive import NaiveModel
from src.models.linear_lag import LinearBaseline
from src.models.garch import GarchModel
from src.models.egarch import EgarchModel
from src.models.garch_x import GarchXModel
from src.evaluation.diagnostics import calculate_statistical_diagnostics, diebold_mariano_test, compute_qlike_loss

def run_fight():
    print("Starting expanding-window Walk-Forward Backtest (Annual Retraining)...")
    
    # Configure local SQLite database tracking
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Walk_Forward_Volatility_Prediction")

    # Load complete feature dataset instead of static splits
    try:
        features_df = pd.read_csv("data/processed/features.csv", index_col=0, parse_dates=True)
    except FileNotFoundError:
        print(" features.csv not found. Run scripts/run_pipeline.py first.")
        return

    # Ensure contiguous, non-NaN records are used
    features_df = features_df.dropna(subset=['Log_Ret', 'Nifty_Ret', 'VIX_Gap', 'Target_Vol_Next_5d'])
    n_records = len(features_df)
    
    # Configuration: Start training with ~12 years of data (3000 days), retrain annually (252 days)
    start_window = 3000
    step_size = 252
    
    print(f"   Total Records: {n_records} | Initial Train: {start_window} | Step Size: {step_size}")
    print("-" * 75)

    # Initialize dictionaries to collect aggregated out-of-sample predictions
    model_preds = {
        "Naive": [],
        "Linear": [],
        "GARCH": [],
        "EGARCH": [],
        "GARCH-X_Base_(No_Exogenous)": [],
        "GARCH-X_Asian_Shock_Only": [],
        "GARCH-X_US_Fear_Only": [],
        "GARCH-X_Hybrid_(Nifty+VIX)": []
    }
    actual_targets = []
    
    # Collect diagnostics over all retraining folds to log averages (AIC, Ljung-Box, etc.)
    fold_diagnostics = {k: [] for k in [
        "GARCH", "EGARCH", 
        "GARCH-X_Base_(No_Exogenous)", 
        "GARCH-X_Asian_Shock_Only", 
        "GARCH-X_US_Fear_Only", 
        "GARCH-X_Hybrid_(Nifty+VIX)"
    ]}

    # Walk-forward loop
    fold = 1
    for start_idx in range(start_window, n_records, step_size):
        train_slice = features_df.iloc[:start_idx]
        test_slice = features_df.iloc[start_idx : start_idx + step_size]
        
        if len(test_slice) == 0:
            break
            
        print(f"\n Processing Fold {fold} | Out-of-sample: {test_slice.index[0].date()} to {test_slice.index[-1].date()} ({len(test_slice)} days)")
        actual_targets.extend(test_slice['Target_Vol_Next_5d'].values)
        
        # 1. Naive Model
        naive = NaiveModel()
        model_preds["Naive"].extend(naive.predict(test_slice).values)
        
        # 2. Linear Model
        linear = LinearBaseline()
        linear.train(train_slice)
        model_preds["Linear"].extend(linear.predict(test_slice))
        
        # 3. Standard GARCH(1,1)
        garch = GarchModel()
        _, g_res, g_preds = garch.evaluate(train_slice, test_slice)
        model_preds["GARCH"].extend(g_preds.values)
        fold_diagnostics["GARCH"].append(calculate_statistical_diagnostics(g_res, test_slice['Target_Vol_Next_5d'], g_preds))
        
        # 4. EGARCH(1,1,1)
        egarch = EgarchModel()
        _, eg_res, eg_preds = egarch.evaluate(train_slice, test_slice)
        model_preds["EGARCH"].extend(eg_preds.values)
        fold_diagnostics["EGARCH"].append(calculate_statistical_diagnostics(eg_res, test_slice['Target_Vol_Next_5d'], eg_preds))
        
        # 5. GARCH-X Models (Native ARX with flat-forward lead matrix projection)
        garch_x_scenarios = [
            ("GARCH-X_Base_(No_Exogenous)", False, False),
            ("GARCH-X_Asian_Shock_Only", True, False),
            ("GARCH-X_US_Fear_Only", False, True),
            ("GARCH-X_Hybrid_(Nifty+VIX)", True, True)
        ]
        for name, use_nifty, use_vix in garch_x_scenarios:
            g_x = GarchXModel(use_nifty=use_nifty, use_vix=use_vix)
            _, gx_res, gx_preds = g_x.evaluate(train_slice, test_slice)
            model_preds[name].extend(gx_preds.values)
            fold_diagnostics[name].append(calculate_statistical_diagnostics(gx_res, test_slice['Target_Vol_Next_5d'], gx_preds))
            
        fold += 1

    # Convert accumulated actuals to Pandas Series for testing
    actual_targets = pd.Series(actual_targets)
    
    print("\n" + "="*75)
    print(" WALK-FORWARD ACCURACY METRICS (Aggregate Out-of-Sample)")
    print("="*75)

    # Compute and log aggregate out-of-sample metrics for all models
    for model_name, pred_list in model_preds.items():
        preds_series = pd.Series(pred_list)
        
        rmse = np.sqrt(mean_squared_error(actual_targets, preds_series))
        mae = mean_absolute_error(actual_targets, preds_series)
        qlike = compute_qlike_loss(actual_targets, preds_series)
        
        print(f"{model_name:<28} -> RMSE: {rmse:.5f} | MAE: {mae:.5f} | QLIKE: {qlike:.5f}")
        
        # Log to MLflow
        with mlflow.start_run(run_name=f"WF_Baseline_{model_name}"):
            mlflow.log_param("backtest_type", "Walk_Forward_Expanding")
            mlflow.log_param("initial_train_days", start_window)
            mlflow.log_param("retrain_step_days", step_size)
            
            mlflow.log_metric("RMSE", rmse)
            mlflow.log_metric("MAE", mae)
            mlflow.log_metric("QLIKE", qlike)
            
            # Log average in-sample fit and diagnostics across all retrained folds
            if model_name in fold_diagnostics:
                diags_list = fold_diagnostics[model_name]
                mlflow.log_metric("Avg_AIC", np.mean([d["AIC"] for d in diags_list]))
                mlflow.log_metric("Avg_BIC", np.mean([d["BIC"] for d in diags_list]))
                mlflow.log_metric("Avg_LjungBox_Stat", np.mean([d["LjungBox_Stat"] for d in diags_list]))
                mlflow.log_metric("Avg_LjungBox_pvalue", np.mean([d["LjungBox_pvalue"] for d in diags_list]))
                mlflow.log_metric("Avg_ARCH_LM_Stat", np.mean([d["ARCH_LM_Stat"] for d in diags_list]))
                mlflow.log_metric("Avg_ARCH_LM_pvalue", np.mean([d["ARCH_LM_pvalue"] for d in diags_list]))
                mlflow.log_metric("Avg_JarqueBera_Stat", np.mean([d["JarqueBera_Stat"] for d in diags_list]))
                mlflow.log_metric("Avg_JarqueBera_pvalue", np.mean([d["JarqueBera_pvalue"] for d in diags_list]))

    # Perform Diebold-Mariano test across the aggregated walk-forward predictions
    g_wf_preds = pd.Series(model_preds["GARCH"])
    eg_wf_preds = pd.Series(model_preds["EGARCH"])
    dm_stat, dm_pvalue = diebold_mariano_test(actual_targets, g_wf_preds, eg_wf_preds, h=5)
    
    print("\n" + "="*75)
    print(" COMPARATIVE STATISTICAL SIGNIFICANCE (WALK-FORWARD)")
    print("="*75)
    print(f"Diebold-Mariano Statistic (GARCH vs EGARCH): {dm_stat:.4f} | p-value: {dm_pvalue:.4f}")
    
    with mlflow.start_run(run_name="WF_GARCH_vs_EGARCH_DM_Test"):
        mlflow.log_metric("DM_Stat", dm_stat)
        mlflow.log_metric("DM_pvalue", dm_pvalue)
        
    if dm_pvalue < 0.05:
        print(f"Conclusion: EGARCH is statistically superior at the 5% significance level.")
    else:
        print("Conclusion: No statistically significant difference in accuracy detected.")

    print("\n Walk-forward backtesting successfully logged to local MLflow database!")

if __name__ == "__main__":
    run_fight()