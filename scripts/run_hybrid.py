# scripts/run_hybrid.py
import pandas as pd
import numpy as np
import sys
import os
import mlflow
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.xgboost_vol import HybridXGBoostVol
from src.evaluation.diagnostics import diebold_mariano_test, compute_qlike_loss

def main():
    print("🚀 Starting the Expanding-Window Walk-Forward Hybrid Backtest...")
    
    # Configure local SQLite database tracking
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Hybrid_Vol_Prediction_Engine")

    # Load complete feature dataset (instead of static splits) to enable walk-forward slicing
    try:
        features_df = pd.read_csv("data/processed/features.csv", index_col=0, parse_dates=True)
    except FileNotFoundError:
        print("❌ features.csv not found. Run scripts/run_pipeline.py first.")
        return

    # Clean the dataset to ensure contiguous records
    features_df = features_df.dropna(subset=['Log_Ret', 'Nifty_Ret', 'VIX_Gap', 'Target_Vol_Next_5d'])
    n_records = len(features_df)
    
    # Configuration: Start training with 1800 days (~7 years) to put the beginning of the 
    # out-of-sample test window in late 2007 (allowing us to capture GFC 2008). Retrain annually (252 days).
    start_window = 3000
    step_size = 252
    
    print(f"   Total Records: {n_records} | Initial Train: {start_window} | Step Size: {step_size}")
    print("-" * 75)

    # Initialize collections for walk-forward predictions
    actual_targets = []
    egarch_preds = []
    hybrid_preds = []
    test_dates = []

    # XGBoost hyperparameters
    max_depth = 2
    learning_rate = 0.01
    n_estimators = 1500
    reg_alpha = 5.0
    reg_lambda = 10.0

    fold = 1
    for start_idx in range(start_window, n_records, step_size):
        train_slice = features_df.iloc[:start_idx]
        test_slice = features_df.iloc[start_idx : start_idx + step_size]
        
        if len(test_slice) == 0:
            break
            
        print(f"⚡ Processing Fold {fold} | Out-of-sample: {test_slice.index[0].date()} to {test_slice.index[-1].date()}")
        
        # Initialize and run the hybrid engine for this fold
        hybrid_engine = HybridXGBoostVol(
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda
        )
        
        _, eg_base, hy_preds = hybrid_engine.fit_and_predict(train_slice, test_slice)
        
        actual_targets.extend(test_slice['Target_Vol_Next_5d'].values)
        egarch_preds.extend(eg_base.values)
        hybrid_preds.extend(hy_preds.values)
        test_dates.extend(test_slice.index)
        
        fold += 1

    # Create unified evaluation DataFrame
    wf_df = pd.DataFrame({
        'Actual': actual_targets,
        'EGARCH_Pred': egarch_preds,
        'Hybrid_Pred': hybrid_preds
    }, index=pd.to_datetime(test_dates))

    print("\n" + "="*75)
    print("📊 MARKET REGIME BREAKDOWN ANALYSIS (GARCH vs Hybrid)")
    print("="*75)

    regimes = {
        'GFC_2008':   ('2007-10-01', '2009-06-30'),
        'COVID_2020': ('2020-02-01', '2020-09-30'),
        'Rates_2022': ('2022-01-01', '2022-12-31'),
        'Normal':     None
    }

    # Run the regime breakdown loop
    for regime, dates in regimes.items():
        if dates:
            mask = (wf_df.index >= dates[0]) & (wf_df.index <= dates[1])
        else:
            stress_mask = (
                ((wf_df.index >= '2007-10-01') & (wf_df.index <= '2009-06-30')) |
                ((wf_df.index >= '2020-02-01') & (wf_df.index <= '2020-09-30')) |
                ((wf_df.index >= '2022-01-01') & (wf_df.index <= '2022-12-31'))
            )
            mask = ~stress_mask

        subset = wf_df[mask]
        
        if len(subset) == 0:
            print(f"{regime:15s} | ❌ No overlapping data points in this walk-forward test set.")
            continue
            
        egarch_rmse = np.sqrt(mean_squared_error(subset['Actual'], subset['EGARCH_Pred']))
        hybrid_rmse = np.sqrt(mean_squared_error(subset['Actual'], subset['Hybrid_Pred']))
        lift = (egarch_rmse - hybrid_rmse) / egarch_rmse * 100

        print(f"{regime:15s} | EGARCH: {egarch_rmse:.5f} | "
              f"Hybrid: {hybrid_rmse:.5f} | Lift: {lift:+.2f}% | Sample size: {len(subset)} days")

    # Evaluate overall out-of-sample metrics
    overall_eg_rmse = np.sqrt(mean_squared_error(wf_df['Actual'], wf_df['EGARCH_Pred']))
    overall_hy_rmse = np.sqrt(mean_squared_error(wf_df['Actual'], wf_df['Hybrid_Pred']))
    overall_lift = (overall_eg_rmse - overall_hy_rmse) / overall_eg_rmse * 100
    
    overall_eg_qlike = compute_qlike_loss(wf_df['Actual'], wf_df['EGARCH_Pred'])
    overall_hy_qlike = compute_qlike_loss(wf_df['Actual'], wf_df['Hybrid_Pred'])

    print("\n" + "="*75)
    print("📈 AGGREGATE WALK-FORWARD METRICS SUMMARY")
    print("="*75)
    print(f"Overall EGARCH -> RMSE: {overall_eg_rmse:.5f} | QLIKE: {overall_eg_qlike:.5f}")
    print(f"Overall Hybrid -> RMSE: {overall_hy_rmse:.5f} | QLIKE: {overall_hy_qlike:.5f}")
    print(f"Overall Walk-forward Performance Lift: {overall_lift:+.2f}%")

    # Log the complete walk-forward hybrid experiment to MLflow
    run_name = f"WF_Hybrid_d{max_depth}_lr{learning_rate}_a{reg_alpha}_l{reg_lambda}"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "xgb_max_depth": max_depth,
            "xgb_learning_rate": learning_rate,
            "xgb_n_estimators": n_estimators,
            "xgb_reg_alpha": reg_alpha,
            "xgb_reg_lambda": reg_lambda,
            "backtest_type": "Walk_Forward_Expanding",
            "initial_train_days": start_window,
            "retrain_step_days": step_size
        })
        
        mlflow.log_metrics({
            "Overall_EGARCH_RMSE": overall_eg_rmse,
            "Overall_Hybrid_RMSE": overall_hy_rmse,
            "Overall_EGARCH_QLIKE": overall_eg_qlike,
            "Overall_Hybrid_QLIKE": overall_hy_qlike,
            "Overall_Lift_Pct": overall_lift
        })
        
        # Calculate DM significance across the full walk-forward series
        dm_stat, dm_pvalue = diebold_mariano_test(wf_df['Actual'], wf_df['EGARCH_Pred'], wf_df['Hybrid_Pred'], h=5)
        print(f"\nDiebold-Mariano Test (WF EGARCH vs WF Hybrid) -> Stat: {dm_stat:.4f} | p-value: {dm_pvalue:.4f}")
        mlflow.log_metric("DM_Stat", dm_stat)
        mlflow.log_metric("DM_pvalue", dm_pvalue)

    print("\n✅ Walk-forward Hybrid backtest and regime analysis logged successfully!")

if __name__ == "__main__":
    main()