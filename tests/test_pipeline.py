# tests/test_pipeline.py
import pandas as pd
import numpy as np
from src.features.feature_builder import FeaturePipeline
from src.evaluation.diagnostics import compute_qlike_loss, diebold_mariano_test

def test_target_leakage():
    """
    PROVES C1 FIX: Ensures the Target (Future Volatility) does not leak the current day's return.
    If we change today's return, today's features should change, but today's target MUST remain identical.
    """
    from src.features.volatility import calculate_log_returns
    
    dates = pd.date_range("2020-01-01", periods=100, freq="B")
    np.random.seed(42)
    fake_prices = pd.DataFrame({
        'Price': np.cumprod(1 + np.random.normal(0, 0.01, 100)) * 100,
        'VIX': np.random.uniform(15, 25, 100),
        'NIFTY': np.cumprod(1 + np.random.normal(0, 0.01, 100)) * 100,
        'Nifty_Ret': np.random.normal(0, 0.01, 100),
        'Nifty_Closed': np.zeros(100)
    }, index=dates)

    pipeline1 = FeaturePipeline()
    pipeline1.df = fake_prices.copy()
    pipeline1.apply_feature_engineering()
    
    target_date = pipeline1.df.index[10]
    original_target = pipeline1.df.loc[target_date, 'Target_Vol_Next_5d']

    # Create a second pipeline, but we will hijack the data right after returns are calculated
    pipeline2 = FeaturePipeline()
    pipeline2.df = fake_prices.copy()
    
    # 1. Calculate base returns
    pipeline2.df = calculate_log_returns(pipeline2.df, price_col='Price')
    
    # 2. Perturb the return for ONLY the target_date
    pipeline2.df.loc[target_date, 'Log_Ret'] *= 5.0 
    
    # 3. Manually run the rest of the pipeline's logic
    from src.features.volatility import calculate_rolling_volatility
    from src.features.regime_features import calculate_volatility_ratios
    from src.features.distribution_features import calculate_tail_risk
    from src.features.vix_features import calculate_vix_gap
    from src.features.cross_market import calculate_cross_market_features

    pipeline2.df = calculate_rolling_volatility(pipeline2.df)
    pipeline2.df = calculate_volatility_ratios(pipeline2.df)
    pipeline2.df = calculate_tail_risk(pipeline2.df)
    pipeline2.df = calculate_vix_gap(pipeline2.df)
    pipeline2.df = calculate_cross_market_features(pipeline2.df)
    
    for col in ['Log_Ret', 'Vol_5d', 'VIX', 'Nifty_Ret']:
        for lag in [1, 2]:
            pipeline2.df[f'{col}_Lag_{lag}'] = pipeline2.df[col].shift(lag)

    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=5)
    pipeline2.df['Target_Vol_Next_5d'] = pipeline2.df['Log_Ret'].shift(-1).rolling(window=indexer).std() * np.sqrt(252)
    pipeline2.df.dropna(inplace=True)
    
    perturbed_target = pipeline2.df.loc[target_date, 'Target_Vol_Next_5d']

    # Now, changing today's return should have NO impact on the next 5 days of volatility.
    assert np.isclose(original_target, perturbed_target), "CRITICAL: Target Leakage Detected!"

def test_qlike_loss_uses_variance():
    """PROVES C8 FIX: QLIKE must strictly calculate loss on Variance (Vol^2)."""
    y_true = np.array([0.20, 0.30]) # 20% and 30% volatility
    y_pred_perfect = np.array([0.20, 0.30])
    
    loss = compute_qlike_loss(y_true, y_pred_perfect)
    
    # QLIKE of perfect predictions should theoretically be log(y_true^2)
    expected_loss = np.mean(np.log(y_true**2) + (y_true**2 / y_true**2) - 1)
    assert np.isclose(loss, expected_loss)

def test_diebold_mariano_sign():
    """PROVES B1 FIX: Ensure the DM stat correctly identifies the winning model."""
    y_true = pd.Series([0.2, 0.3, 0.4, 0.5, 0.6])
    
    # Model 1 has huge errors
    y_pred_bad = pd.Series([0.9, 0.9, 0.9, 0.9, 0.9])
    
    # Model 2 has tiny errors
    y_pred_good = pd.Series([0.21, 0.29, 0.41, 0.49, 0.61])
    
    dm_stat, p_val = diebold_mariano_test(y_true, y_pred_bad, y_pred_good, h=1)
    
    # Because Model 1 (bad) has larger errors than Model 2 (good), e1 - e2 is positive.
    # Therefore the DM Stat should be strictly positive.
    assert dm_stat > 0, "DM Stat sign logic is reversed!"