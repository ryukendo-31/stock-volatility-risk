# src/evaluation/diagnostics.py
import numpy as np
import pandas as pd
from scipy.stats import jarque_bera, norm
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from sklearn.metrics import mean_squared_error, mean_absolute_error

def compute_qlike_loss(y_true, y_pred):
    """Computes QLIKE loss."""
    y_true = np.clip(np.array(y_true), 1e-6, None)
    y_pred = np.clip(np.array(y_pred), 1e-6, None)
    return np.mean(np.log(y_pred) + (y_true / y_pred) - 1)

def calculate_statistical_diagnostics(fit_result, actual_vol, predicted_vol):
    """Computes goodness-of-fit, residual tests, and out-of-sample loss metrics."""
    params = fit_result.params
    pvalues = fit_result.pvalues
    std_errs = fit_result.std_err
    tvalues = fit_result.tvalues
    
    std_residuals = fit_result.resid / fit_result.conditional_volatility
    std_residuals = std_residuals.dropna()
    std_residuals_sq = std_residuals ** 2
    
    # Ljung-Box Test
    lb_test = acorr_ljungbox(std_residuals_sq, lags=[10])
    lb_stat = lb_test.iloc[0, 0] if isinstance(lb_test, pd.DataFrame) else lb_test[0][0]
    lb_pvalue = lb_test.iloc[0, 1] if isinstance(lb_test, pd.DataFrame) else lb_test[1][0]
    
    # ARCH LM Test
    arch_lm_test = het_arch(std_residuals)
    arch_lm_stat = arch_lm_test[0]
    arch_lm_pvalue = arch_lm_test[1]
    
    # Jarque-Bera Test: jb_stat is index 0, jb_pvalue is index 1
    jb_stat, jb_pvalue = jarque_bera(std_residuals)
    
    common_idx = actual_vol.index.intersection(predicted_vol.index)
    act = actual_vol.loc[common_idx].values
    pred = predicted_vol.loc[common_idx].values
    
    rmse = np.sqrt(mean_squared_error(act, pred))
    mae = mean_absolute_error(act, pred)
    qlike = compute_qlike_loss(act, pred)
    
    diagnostics = {
        "AIC": fit_result.aic,
        "BIC": fit_result.bic,
        "LjungBox_Stat": lb_stat,
        "LjungBox_pvalue": lb_pvalue,
        "ARCH_LM_Stat": arch_lm_stat,
        "ARCH_LM_pvalue": arch_lm_pvalue,
        "JarqueBera_Stat": jb_stat,
        "JarqueBera_pvalue": jb_pvalue,
        "RMSE": rmse,
        "MAE": mae,
        "QLIKE": qlike,
        "Params": params.to_dict(),
        "PValues": pvalues.to_dict(),
        "StdErrors": std_errs.to_dict(),
        "TValues": tvalues.to_dict()
    }
    
    return diagnostics

def diebold_mariano_test(y_true, y_pred1, y_pred2, h=5, power=2):
    """Computes the Diebold-Mariano test for predictive accuracy using Newey-West variance."""
    common_idx = y_true.index.intersection(y_pred1.index).intersection(y_pred2.index)
    y_true = y_true.loc[common_idx].values
    y_pred1 = y_pred1.loc[common_idx].values
    y_pred2 = y_pred2.loc[common_idx].values
    
    e1 = np.abs(y_true - y_pred1) ** power
    e2 = np.abs(y_true - y_pred2) ** power
    
    d = e1 - e2
    d_mean = np.mean(d)
    T = len(d)
    
    if T <= h:
        return 0.0, 1.0
        
    gamma = np.zeros(h)
    gamma[0] = np.var(d)
    for lag in range(1, h):
        if T - lag > 0:
            gamma[lag] = np.correlate(d[:-lag] - d_mean, d[lag:] - d_mean)[0] / T
            
    lr_var = gamma[0] + 2 * np.sum([(1 - (k / h)) * gamma[k] for k in range(1, h)])
    
    if lr_var <= 0:
        lr_var = 1e-8
        
    dm_stat = d_mean / np.sqrt(lr_var / T)
    p_value = 2 * (1 - norm.cdf(np.abs(dm_stat)))
    
    return dm_stat, p_value