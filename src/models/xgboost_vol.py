# src/models/xgboost_vol.py
import pandas as pd
import numpy as np
import os
import xgboost as xgb
from arch import arch_model
from src.decision.shap_gate import ShapSafetyGate

class HybridXGBoostVol:
    def __init__(self, max_depth=2, learning_rate=0.01, n_estimators=1000, reg_alpha=5.0, reg_lambda=10.0, 
                 threshold_std=3.5, max_concentration_ratio=0.80, min_rank_correlation=0.40):
        # XGBoost parameters
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        
        # Safety Gate parameters
        self.threshold_std = threshold_std
        self.max_concentration_ratio = max_concentration_ratio
        self.min_rank_correlation = min_rank_correlation
        
        self.egarch_model = None
        self.egarch_fit = None
        self.xgb_model = None
        self.safety_gate = None

    def fit_and_predict(self, train_df, test_df):
        print("1. Fitting baseline Student-t EGARCH(1,1,1) on training set...")
        
        # Scale returns by 100 for optimization stability in 'arch'
        returns_train = train_df['Log_Ret'] * 100
        returns_test = test_df['Log_Ret'] * 100
        returns_full = pd.concat([returns_train, returns_test])
        
        split_date = train_df.index[-1]
        
        # Fit the baseline EGARCH model
        self.egarch_model = arch_model(returns_full, vol='EGARCH', p=1, o=1, q=1, dist='t')
        self.egarch_fit = self.egarch_model.fit(last_obs=split_date, disp='off')
        
        # Generate EGARCH training predictions (annualized)
        in_sample_var = self.egarch_fit.conditional_volatility.loc[train_df.index] ** 2
        egarch_train_pred = np.sqrt(in_sample_var) / 100 * np.sqrt(252)
        
        # Generate EGARCH testing predictions (simulation path-based)
        eg_forecasts = self.egarch_fit.forecast(start=split_date, horizon=5, method='simulation')
        var_preds = eg_forecasts.variance.reindex(test_df.index)
        mean_variance = var_preds.mean(axis=1)
        egarch_test_pred = np.sqrt(mean_variance) / 100 * np.sqrt(252)
        
        # Calculate training residuals (Actual - Predicted)
        train_residuals = train_df['Target_Vol_Next_5d'] - egarch_train_pred
        train_residuals = train_residuals.dropna()
        
        # Exclude target and raw returns/prices from features
        cols_to_exclude = ['Target_Vol_Next_5d', 'Log_Ret', 'Nifty_Ret']
        feature_cols = [col for col in train_df.columns if col not in cols_to_exclude]
        
        X_train = train_df.loc[train_residuals.index, feature_cols]
        y_train = train_residuals
        
        print("2. Training XGBoost Regressor on EGARCH residuals...")
        # Chronological train-validation split (last 15% of training data) for early stopping
        val_split_idx = int(len(X_train) * 0.85)
        X_tr, X_val = X_train.iloc[:val_split_idx], X_train.iloc[val_split_idx:]
        y_tr, y_val = y_train.iloc[:val_split_idx], y_train.iloc[val_split_idx:]
        
        self.xgb_model = xgb.XGBRegressor(
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            subsample=0.8,
            colsample_bytree=0.8,
            early_stopping_rounds=50,
            random_state=42
        )
        
        self.xgb_model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        print("3. Initializing and calibrating SHAP Safety Gate...")
        # Fit the safety gate, passing X_test so COVID benchmarks are calculated out-of-sample
        X_test = test_df[feature_cols]
        self.safety_gate = ShapSafetyGate(
            max_absolute_adj=0.05,
            max_relative_adj=0.45,
            threshold_std=self.threshold_std,
            max_concentration_ratio=self.max_concentration_ratio,
            min_rank_correlation=self.min_rank_correlation,
            min_adj_for_concentration=0.020
        )
        self.safety_gate.fit_explainer(self.xgb_model, X_train, X_test=X_test)
        
        print("4. Executing predictions with active SHAP Safety Gate checks...")
        final_hybrid_preds = []
        gate_decisions = []
        gate_reasons = []
        noise_ratios = []
        max_z_scores = []
        xgb_adjustments = []
        
        # Evaluate each day sequentially to maintain timeline integrity
        for idx in range(len(X_test)):
            row = X_test.iloc[[idx]]
            eg_pred = egarch_test_pred.iloc[idx]
            
            # Generate raw XGBoost adjustment prediction
            xgb_adj = self.xgb_model.predict(row)[0]
            xgb_adjustments.append(xgb_adj)
            
            # Audit the prediction row via the safety gate
            status, reason, diags = self.safety_gate.evaluate_prediction_safety(row, eg_pred, xgb_adj)
            
            if status == "APPROVED":
                final_pred = eg_pred + xgb_adj
            else:
                # Force standard fallback to baseline econometric model
                final_pred = eg_pred
                
            final_hybrid_preds.append(final_pred)
            gate_decisions.append(status)
            gate_reasons.append(reason)
            noise_ratios.append(diags.get("concentration_ratio", 0.0) if status == "REJECTED" and reason == "SHAP_CONCENTRATION_LIMIT" else 0.0)
            max_z_scores.append(diags.get("z_score", 0.0) if status == "REJECTED" and reason == "SHAP_OOD_LIMIT" else 0.0)
            
        final_hybrid_preds = np.clip(np.array(final_hybrid_preds), 0.01, None)
        final_hybrid_preds = pd.Series(final_hybrid_preds, index=test_df.index)
        
        # Save complete predictions and safety decisions for Phase 2o logging
        results_df = pd.DataFrame({
            'Actual': test_df['Target_Vol_Next_5d'],
            'EGARCH_Base': egarch_test_pred,
            'XGB_Adjustment': xgb_adjustments,
            'Hybrid_Final': final_hybrid_preds,
            'Gate_Decision': gate_decisions,
            'Gate_Reason': gate_reasons,
            'Max_Z_Score': max_z_scores,
            'Noise_Ratio': noise_ratios
        }, index=test_df.index)
        
        os.makedirs("results", exist_ok=True)
        results_df.to_csv("results/hybrid_predictions.csv")
        
        return results_df, egarch_test_pred, final_hybrid_preds