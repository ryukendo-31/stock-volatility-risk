# src/models/egarch.py
import pandas as pd
import numpy as np
import os
from arch import arch_model
from sklearn.metrics import mean_squared_error

class EgarchModel:
    def __init__(self):
        self.model = None

    def evaluate(self, train_df, test_df):
        print("Configuring EGARCH(1,1,1) with Student-t distribution...")

        full_df = pd.concat([train_df, test_df])
        returns = full_df['Log_Ret'] * 100 
        split_date = train_df.index[-1]
        
        am = arch_model(returns, vol='EGARCH', p=1, o=1, q=1, dist='t')
        res = am.fit(last_obs=split_date, disp='off')
        
        # FIX: Seed NumPy globally so the internal Student-t simulation is reproducible
        np.random.seed(42)
        
        forecasts = res.forecast(start=split_date, horizon=5, method='simulation', simulations=1000)
        var_preds = forecasts.variance.reindex(test_df.index)
        
        mean_variance = var_preds.mean(axis=1)
        vol_preds = np.sqrt(mean_variance) / 100 * np.sqrt(252)
        
        results_df = pd.DataFrame({
            'EGARCH_Pred': vol_preds,
            'Actual_Vol': test_df['Target_Vol_Next_5d']
        }, index=test_df.index)
        
        for col in ['^VIX', 'VIX_Close', 'VIX']:
            if col in test_df.columns:
                results_df['VIX_Market'] = test_df[col] / 100.0 
                results_df['Spread_Signal'] = results_df['EGARCH_Pred'] - results_df['VIX_Market']
                break
        
        os.makedirs("results", exist_ok=True)
        mode = 'a' if os.path.exists("results/egarch_predictions.csv") else 'w'
        header = not os.path.exists("results/egarch_predictions.csv")
        results_df.to_csv("results/egarch_predictions.csv", mode=mode, header=header)
        
        eval_df = results_df.dropna(subset=['Actual_Vol', 'EGARCH_Pred'])
        if len(eval_df) == 0:
            return {"RMSE": 999.0}, res, vol_preds

        rmse = np.sqrt(mean_squared_error(eval_df['Actual_Vol'], eval_df['EGARCH_Pred']))
        return {"RMSE": rmse}, res, vol_preds