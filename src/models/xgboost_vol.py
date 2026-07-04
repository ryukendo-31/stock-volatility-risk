# src/models/xgboost_vol.py
import pandas as pd
import numpy as np
import os
import xgboost as xgb
from arch import arch_model
from sklearn.metrics import mean_squared_error

class HybridXGBoostVol:
    def __init__(self, max_depth=2, learning_rate=0.01, n_estimators=1000, reg_alpha=5.0, reg_lambda=10.0, subsample=0.8, colsample_bytree=0.8):
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        
        self.egarch_model = None
        self.egarch_fit = None
        self.xgb_model = None

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
        
        # Initialize XGBoost. Pass early_stopping_rounds here to remain fully compliant with v1.6+ and v2.0+
        self.xgb_model = xgb.XGBRegressor(
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            early_stopping_rounds=50,  # Correct placement for modern APIs
            random_state=42
        )
        
        # Fit the model with early stopping enabled
        self.xgb_model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        # 5. Predict out-of-sample residuals
        X_test = test_df[feature_cols]
        xgb_test_residual_pred = self.xgb_model.predict(X_test)
        xgb_test_residual_pred = pd.Series(xgb_test_residual_pred, index=test_df.index)
        
        # 6. Combine predictions: EGARCH base + XGBoost residual adjustment
        hybrid_test_pred = egarch_test_pred + xgb_test_residual_pred
        
        # Clip to prevent negative volatility forecasts
        hybrid_test_pred = np.clip(hybrid_test_pred, 0.01, None)
        
        # Save predictions to disk
        results_df = pd.DataFrame({
            'Actual': test_df['Target_Vol_Next_5d'],
            'EGARCH_Base': egarch_test_pred,
            'XGB_Adjustment': xgb_test_residual_pred,
            'Hybrid_Final': hybrid_test_pred
        }, index=test_df.index)
        
        os.makedirs("results", exist_ok=True)
        results_df.to_csv("results/hybrid_predictions.csv")
        
        return results_df, egarch_test_pred, hybrid_test_pred