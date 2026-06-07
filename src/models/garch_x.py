import pandas as pd
import numpy as np
import os
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error

class GarchXModel:
    def __init__(self, use_nifty=False, use_vix=False):
        self.use_nifty = use_nifty
        self.use_vix = use_vix
        
        # Automatically name the model based on the combination
        if use_nifty and use_vix:
            self.model_name = "GARCH-X_Hybrid_(Nifty+VIX)"
        elif use_nifty:
            self.model_name = "GARCH-X_Asian_Shock_Only"
        elif use_vix:
            self.model_name = "GARCH-X_US_Fear_Only"
        else:
            self.model_name = "GARCH-X_Base_(No_Exogenous)"

    def _garch_x_loglikelihood(self, params, r_sp, r_nifty, vix):
        mu_sp, mu_nifty, omega, alpha, beta, gamma_nifty, gamma_vix = params
        T = len(r_sp)
        
        eps_sp2 = (r_sp - mu_sp) ** 2
        eps_nifty2 = (r_nifty - mu_nifty) ** 2
        
        sigma2 = np.zeros(T)
        sigma2[0] = np.var(r_sp)
        
        for t in range(1, T):
            var_t = omega + alpha * eps_sp2[t-1] + beta * sigma2[t-1]
            if self.use_nifty: 
                var_t += gamma_nifty * eps_nifty2[t-1]
            if self.use_vix:   
                var_t += gamma_vix * vix[t-1]
            sigma2[t] = var_t
            
        if np.any(sigma2 <= 0): 
            return 1e10 
        
        LL = -0.5 * np.sum(np.log(2 * np.pi) + np.log(sigma2) + (eps_sp2 / sigma2))
        return -LL

    def evaluate(self, train_df, test_df):
        print(f"\n⚙️ Training {self.model_name}...")
        
        # 1. Fetch VIX from Raw Data since it was dropped in features building
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        raw_df = pd.read_csv(os.path.join(BASE_DIR, "data", "raw", "global_markets.csv"), index_col=0, parse_dates=True)
        
        full_df = pd.concat([train_df, test_df])
        if 'VIX' not in full_df.columns:
            full_df = full_df.join(raw_df[['VIX']], how='left')
            
        full_df = full_df.dropna(subset=['Log_Ret', 'Nifty_Ret', 'VIX', 'Target_Vol_Next_5d'])
        
        # Separate back into train and test
        df_train = full_df.loc[full_df.index.isin(train_df.index)]
        df_test = full_df.loc[full_df.index.isin(test_df.index)]

        # Arrays for Training
        r_sp_train = df_train['Log_Ret'].values * 100
        r_nifty_train = df_train['Nifty_Ret'].values * 100
        vix_train = df_train['VIX'].values

        # 2. RUN OPTIMIZER ON TRAINING DATA
        initial_guess = [0.05, 0.05, 0.02, 0.10, 0.85, 0.01, 0.01]
        bounds = ((None, None), (None, None), (1e-6, None), (1e-6, 0.999), (1e-6, 0.999), (0, None), (0, None))
        constraints = ({'type': 'ineq', 'fun': lambda x: 0.999 - (x[3] + x[4])})
        
        result = minimize(
            self._garch_x_loglikelihood, initial_guess, 
            args=(r_sp_train, r_nifty_train, vix_train),
            method='SLSQP', bounds=bounds, constraints=constraints, 
            options={'disp': False, 'maxiter': 500}
        )
        
        mu_sp, mu_nifty, omega, alpha, beta, gamma_nifty, gamma_vix = result.x
        
        # 3. FORECAST ON TEST DATA
        r_sp_test = df_test['Log_Ret'].values * 100
        r_nifty_test = df_test['Nifty_Ret'].values * 100
        vix_test = df_test['VIX'].values
        
        eps_sp2_test = (r_sp_test - mu_sp) ** 2
        eps_nifty2_test = (r_nifty_test - mu_nifty) ** 2
        
        T_test = len(r_sp_test)
        sigma2_test = np.zeros(T_test)
        
        # Seed the first value using the last known variance from training
        sigma2_test[0] = np.var(r_sp_train) 
        
        for t in range(1, T_test):
            var_t = omega + alpha * eps_sp2_test[t-1] + beta * sigma2_test[t-1]
            if self.use_nifty: 
                var_t += gamma_nifty * eps_nifty2_test[t-1]
            if self.use_vix:   
                var_t += gamma_vix * vix_test[t-1]
            sigma2_test[t] = var_t

        # Convert back to Annualized Volatility for comparison
        vol_preds = np.sqrt(sigma2_test) / 100 * np.sqrt(252)
        
        # Calculate RMSE
        rmse = np.sqrt(mean_squared_error(df_test['Target_Vol_Next_5d'], vol_preds))
        
        # Dictionary of parameters to log in MLflow
        parameters_dict = {
            "Omega": omega, "Alpha": alpha, "Beta": beta,
            "Gamma_Nifty": gamma_nifty if self.use_nifty else 0,
            "Gamma_VIX": gamma_vix if self.use_vix else 0
        }
        
        return {"RMSE": rmse, "Params": parameters_dict}