import pandas as pd
import sys
import os
import mlflow

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.naive import NaiveModel
from src.models.linear_lag import LinearBaseline
from src.models.garch import GarchModel
# NEW: Import your custom GARCH-X Model
from src.models.garch_x import GarchXModel

def run_fight():
    print(" Starting the Model Comparison & MLflow Logging...")
    
    # --- MLFLOW SETUP ---
    # Since you are running `mlflow ui` on port 5000:
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Volatility_Prediction_Models")

    # Load Data
    try:
        train_df = pd.read_csv("data/processed/train.csv", index_col=0, parse_dates=True)
        test_df = pd.read_csv("data/processed/test.csv", index_col=0, parse_dates=True)
    except FileNotFoundError:
        print("Data not found. Run scripts/run_pipeline.py first.")
        return

    print(f"   Train: {len(train_df)} | Test: {len(test_df)}")
    print("-" * 50)

    # -----------------------------------------
    # 1. Naive Model
    # -----------------------------------------
    with mlflow.start_run(run_name="Baseline_Naive"):
        print("\n[1/4] Running Naive Model...")
        naive = NaiveModel()
        metrics = naive.evaluate(test_df)
        print(f"      RMSE:  {metrics['RMSE']:.5f}")
        
        mlflow.log_param("model_family", "Baseline")
        mlflow.log_metric("RMSE", metrics['RMSE'])

    # -----------------------------------------
    # 2. Linear Model
    # -----------------------------------------
    with mlflow.start_run(run_name="Baseline_Linear_Lag"):
        print("\n[2/4] Running Linear Model...")
        linear = LinearBaseline()
        linear.train(train_df)
        metrics = linear.evaluate(test_df)
        print(f"      RMSE:  {metrics['RMSE']:.5f}")
        
        mlflow.log_param("model_family", "Regression")
        mlflow.log_metric("RMSE", metrics['RMSE'])

    # -----------------------------------------
    # 3. Standard GARCH(1,1) Model
    # -----------------------------------------
    if 'Log_Ret' in train_df.columns:
        with mlflow.start_run(run_name="Baseline_GARCH_1_1"):
            print("\n[3/4] Running Standard GARCH Model...")
            garch = GarchModel()
            metrics = garch.evaluate(train_df, test_df)
            print(f"      RMSE:  {metrics['RMSE']:.5f}")
            
            mlflow.log_param("model_family", "Volatility")
            mlflow.log_param("variant", "Standard")
            mlflow.log_metric("RMSE", metrics['RMSE'])

    print("-" * 50)
    print("\n[4/4]  EXECUTING CUSTOM GARCH-X PIPELINE 🔥")
    
    # -----------------------------------------
    # 4. GARCH-X (The 4 Combinations)
    # -----------------------------------------
    garch_scenarios = [
        (False, False), # Base Model
        (True, False),  # Nifty Only
        (False, True),  # VIX Only
        (True, True)    # Nifty + VIX (Hybrid)
    ]
    
    for use_nifty, use_vix in garch_scenarios:
        # Initialize your custom model
        garch_x = GarchXModel(use_nifty=use_nifty, use_vix=use_vix)
        
        with mlflow.start_run(run_name=garch_x.model_name):
            # Evaluate directly (the optimizer output is printed via your class)
            results = garch_x.evaluate(train_df, test_df)
            
            print(f"      >> FINAL RMSE: {results['RMSE']:.5f}\n")
            
            # Log to MLflow
            mlflow.log_param("model_family", "GARCH-X")
            mlflow.log_param("uses_nifty", use_nifty)
            mlflow.log_param("uses_vix", use_vix)
            
            # Log the optimized parameters you generated
            mlflow.log_params(results["Params"])
            
            # Log the performance
            mlflow.log_metric("RMSE", results["RMSE"])

    print("\n All models finished and logged to MLflow!")

if __name__ == "__main__":
    run_fight()