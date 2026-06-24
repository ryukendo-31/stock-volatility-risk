# src/models/garch_x.py
import pandas as pd
import numpy as np
import os
from arch import arch_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

class GarchXModel:
    def __init__(self, use_nifty=False, use_vix=False):
        self.use_nifty = use_nifty
        self.use_vix = use_vix
        
        if use_nifty and use_vix:
            self.model_name = "GARCH-X_Hybrid_(Nifty+VIX)"
        elif use_nifty:
            self.model_name = "GARCH-X_Asian_Shock_Only"
        elif use_vix:
            self.model_name = "GARCH-X_US_Fear_Only"
        else:
            self.model_name = "GARCH-X_Base_(No_Exogenous)"

    def evaluate(self, train_df, test_df):
        print(f"\n⚙️ Training Native {self.model_name}...")
        
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        raw_df = pd.read_csv(os.path.join(BASE_DIR, "data", "raw", "global_markets.csv"), index_col=0, parse_dates=True)
        
        full_df = pd.concat([train_df, test_df])
        if 'VIX' not in full_df.columns:
            full_df = full_df.join(raw_df[['VIX']], how='left')
            
        full_df = full_df.dropna(subset=['Log_Ret', 'Nifty_Ret', 'VIX', 'Target_Vol_Next_5d'])
        
        # Standardize exogenous regressors
        exog_features = []
        if self.use_nifty:
            exog_features.append('Nifty_Ret')
        if self.use_vix:
            exog_features.append('VIX')
            
        exog_data = None
        if exog_features:
            scaler = StandardScaler()
            scaled_vals = scaler.fit_transform(full_df[exog_features])
            exog_data = pd.DataFrame(scaled_vals, index=full_df.index, columns=exog_features)
            
        # Re-split
        df_train = full_df.loc[full_df.index.isin(train_df.index)]
        df_test = full_df.loc[full_df.index.isin(test_df.index)]
        
        returns_full = full_df['Log_Ret'] * 100
        split_date = train_df.index[-1]
        
        # Configure native GARCH-X
        if exog_data is not None:
            am = arch_model(
                returns_full, 
                x=exog_data.reindex(full_df.index),
                mean='ARX',
                vol='Garch', p=1, q=1, dist='t'
            )
        else:
            am = arch_model(
                returns_full, 
                mean='Constant',
                vol='Garch', p=1, q=1, dist='t'
            )
            
        res = am.fit(last_obs=split_date, disp='off')
        
        # Forecast out-of-sample with strict look-ahead protection
        if exog_data is not None:
            forecast_dates = full_df.loc[split_date:].index
            n_forecasts = len(forecast_dates)
            
            x_dict = {}
            for col in exog_features:
                col_series = exog_data[col].reindex(full_df.index)
                last_known = col_series.reindex(forecast_dates)  # Value AT the forecast origin t
                
                # Project the known contemporaneous value flat forward across the 5 steps
                horizon_matrix = np.tile(last_known.values.reshape(-1, 1), (1, 5))
                x_dict[col] = horizon_matrix
                
            forecasts = res.forecast(start=split_date, x=x_dict, horizon=5)
        else:
            forecasts = res.forecast(start=split_date, horizon=5)
            
        var_preds = forecasts.variance.reindex(df_test.index)
        
        mean_variance = var_preds.mean(axis=1)
        vol_preds = np.sqrt(mean_variance) / 100 * np.sqrt(252)
        
        rmse = np.sqrt(mean_squared_error(df_test['Target_Vol_Next_5d'], vol_preds))
        
        # Extract parameter coefficients and sanitize keys for MLflow
        parameters_dict = {}
        for param, val in res.params.items():
            param_clean = param.replace('[', '_').replace(']', '')
            parameters_dict[param_clean] = val
            
        return {"RMSE": rmse, "Params": parameters_dict}, res, vol_preds