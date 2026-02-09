import pandas as pd
import numpy as np
import os  # Added to save the results file
from arch import arch_model
from sklearn.metrics import mean_squared_error

class GarchModel:
    def __init__(self):
        self.model = None

    def evaluate(self, train_df, test_df):
        print("Configuring GARCH(1,1)...")

        full_df = pd.concat([train_df, test_df]) #create a complete dataframe by joining the split data
        
        #Scale returns 
        returns = full_df['Log_Ret'] * 100 #GARCH cannot handle small values, makes calculation faster
        
        split_date = train_df.index[-1] # We want to train up to the last day of the Training Set.
        
        #Train (Fit only on the Training period)
        # last_obs=split_date ensures the model doesn't "see" the future.
        # used the Normal assumption for the GARCH baseline as per standard practice also we know that the distribution has fatt tails and is leptokurtosis
        am = arch_model(returns, vol='Garch', p=1, q=1, dist='Normal') #we take 1,1 model as it is the industry standard and is almost never beaten because looking at 2 days ago makes the model overfit
        
        res = am.fit(last_obs=split_date, disp='off')
        
        #Forecast
        # start=split_date means "Start generating forecasts from this date onwards"
        forecasts = res.forecast(start=split_date, horizon=5)
        
        #Extract & Align
        #We specifically ask for the variance on the dates present in test_df
        #this fixes the "Empty Array" bug by forcing alignment.
        var_preds = forecasts.variance.reindex(test_df.index)
        
        #Convert Variance -> Annualized Volatility
        #Mean of the 5-day variance forecast
        mean_variance = var_preds.mean(axis=1)
        
        #Sqrt(Variance) / 100 (undo scaling) * Sqrt(252) (annualize)
        vol_preds = np.sqrt(mean_variance) / 100 * np.sqrt(252)
        
        print("\nFUTURE FORECAST (Next 5 Days):")
        print(vol_preds.tail(5))  # Print the last 5 rows of predictions
        print("-" * 30)

        # --- NEW SECTION: SAVE RESULTS & COMPARE WITH VIX ---
        # Create a Master DataFrame for analysis
        results_df = pd.DataFrame({
            'GARCH_Pred': vol_preds,
            'Actual_Vol': test_df['Target_Vol_Next_5d']
        }, index=test_df.index)
        
        # Check if VIX exists in the test data to calculate the "Spread"
        # The column might be named '^VIX', 'VIX', or 'VIX_Close' depending on loader
        for col in ['^VIX', 'VIX_Close', 'VIX']:
            if col in test_df.columns:
                # VIX is typically 0-100, we divide by 100 to match our 0.0-1.0 format
                results_df['VIX_Market'] = test_df[col] / 100.0 
                # Spread = Model Prediction - Market Prediction
                results_df['Spread_Signal'] = results_df['GARCH_Pred'] - results_df['VIX_Market']
                break
        
        # Save to Disk (This includes the future days where Actual_Vol is NaN)
        os.makedirs("results", exist_ok=True)
        save_path = "results/garch_predictions.csv"
        results_df.to_csv(save_path)
        print(f"Predictions saved to {save_path}")
        # -----------------------------------------------------

        #Create a Clean DataFrame for Evaluation
        #We use the results_df we just created, but drop NaNs for the math
        eval_df = results_df.dropna(subset=['Actual_Vol', 'GARCH_Pred'])
        
        if len(eval_df) == 0:
            print(" Error: No overlapping data found between Preds and Actuals.")
            return {"RMSE": 999.0}

        # Calculate Metrics
        rmse = np.sqrt(mean_squared_error(eval_df['Actual_Vol'], eval_df['GARCH_Pred']))
        
        return {"RMSE": rmse}