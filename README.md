# S&P 500 Hybrid Volatility Prediction Engine

This repository implements a risk-controlled, hybrid volatility forecasting framework for the S&P 500. The architecture combines traditional financial econometrics (Student-t EGARCH) with machine learning (XGBoost) and Explainable AI (SHAP) to construct a parsimonious baseline with regularized residual-adjustment layers and active fallback guards.

## 1. Project Directory Layout

```
stock-market-volatility/
├── app/                              # Placeholder for deployment wrapper
├── data/
│   ├── processed/                    # Engineered features, train.csv, and test.csv
│   └── raw/                          # Raw downloaded yFinance global markets CSV
├── notebooks/                        # Jupyter notebooks for exploratory analysis
├── reports/
│   └── figures/                      # Saved diagnostic and performance plots
├── results/
│   ├── egarch_predictions.csv        # Out-of-sample EGARCH forecast series
│   ├── garch_predictions.csv         # Out-of-sample GARCH forecast series
│   └── hybrid_predictions.csv        # Final hybrid predictions and safety gate logs
├── scripts/
│   ├── plot_diagnostics.py           # Generates residual and predictive plots
│   ├── run_baselines.py              # Runs the expanding-window baseline backtests
│   ├── run_hybrid.py                 # Runs the integrated production hybrid engine
│   ├── run_pipeline.py               # Ingests raw data and builds the feature matrix
│   └── test_shap_gate.py             # Simulates and calibrates safety gate thresholds
├── src/
│   ├── data/
│   │   ├── loader.py                 # Timezone-safe data loader and return calculator
│   │   └── splitter.py               # Chronological train/test data splitter
│   ├── decision/
│   │   └── shap_gate.py              # Cascading safety gate with conditional thresholds
│   ├── evaluation/
│   │   └── diagnostics.py            # Computes residual diagnostics, QLIKE, and DM tests
│   ├── explainability/
│   │   ├── __init__.py               # Package identifier
│   │   └── shap_explainer.py         # SHAP value extraction and visualization helpers
│   ├── features/
│   │   ├── cross_market.py           # Pillar 5: Indian market session returns
│   │   ├── distribution_features.py  # Pillar 3: Rolling skewness and kurtosis
│   │   ├── feature_builder.py        # Orchestrates feature pipeline execution
│   │   ├── regime_features.py        # Pillar 2: Short/Long volatility ratios
│   │   ├── vix_features.py           # Pillar 4: Options-implied VIX premium
│   │   └── volatility.py             # Pillar 1: Log returns and rolling volatilities
│   ├── models/
│   │   ├── egarch.py                 # Student-t EGARCH model class
│   │   ├── garch_x.py                # Native ARX GARCH-X model class
│   │   ├── garch.py                  # Student-t GARCH model class
│   │   ├── linear_lag.py             # Baseline OLS lag model class
│   │   ├── naive.py                  # Baseline naive model class
│   │   └── xgboost_vol.py            # Core Hybrid EGARCH-XGBoost regressor engine
│   ├── pipeline/
│   └── config.py                     # Global path and execution configurations
├── .gitignore
├── mlflow.db                         # Local SQLite backend storing MLflow runs
├── requirements.txt                  # Python dependencies
└── setup_project.py                  # Setup utility
```

---

## 2. Feature Engineering: The 5 Pillars

The feature engineering pipeline (`src/features/feature_builder.py`) processes raw daily series into 5 structural pillars:

1. **Inertia (Pillar 1):** S&P 500 daily log returns and annualized rolling volatilities (5-day, 10-day, 21-day, and 63-day windows).
2. **Regime Detection (Pillar 2):** Volatility acceleration ratios (5d/21d and 21d/63d) to identify transition phases.
3. **Tail Risk (Pillar 3):** Rolling 21-day skewness and kurtosis of returns to proxy non-normal distributions and tail-width.
4. **Fear Premium (Pillar 4):** The VIX Gap, representing the spread between implied market volatility and realized historical volatility (`VIX_t - (Vol_21d,t * 100)`).
5. **Cross-Market Momentum (Pillar 5):** Log returns of the Nifty 50, calculated natively on its own calendar, to capture early global market information before the US open.

---

## 3. Empirical Performance Results

The completed hybrid forecasting model was trained on 3,772 historical observations (2007–2022) and evaluated out-of-sample over 943 trading days (September 14, 2022, to June 17, 2026).

### Out-of-Sample Performance Comparison

| Model | RMSE | MAE | QLIKE |
|---|---|---|---|
| EGARCH(1,1,1) Base | 0.06875 | 0.04868 | -2.04231 |
| Hybrid Final (Active Gate) | 0.06680 | 0.04598 | -2.04691 |

**Predictive Performance Lift:** +2.84% reduction in out-of-sample RMSE.

### Statistical Validation (Diebold-Mariano Test)

- **DM Statistic:** 3.6960
- **p-value:** 0.0002 (significant at the 0.1% level)
- **Conclusion:** We reject the null hypothesis of equal predictive accuracy. The predictive superiority of the machine learning residual adjustments over the EGARCH baseline is statistically highly significant.

---

## 4. SHAP Safety Gate & Regime Decoupling

To prevent XGBoost from overfitting or generating extrapolation errors during extreme tail events, the system implements a cascading **SHAP Safety Gate** (`src/decision/shap_gate.py`).

### Cascading Guards

1. **Domain Overrides:** Hardcoded bounds on raw inputs based on historical 90th percentile stress peaks (bypasses if `Vol_21d > 0.40`, `VIX_Lag_1 > 40.0`, or `Kurt_21 > 8.0`).
2. **Magnitude Limits:** Absolute (5%) and relative (45%) caps on the size of the XGBoost volatility adjustment.
3. **Targeted Expansion Check:** Bypasses if features empirically proven to expand during the 2020 crash (`Vol_10d` or `VIX_Lag_1`) exceed their baseline training SHAP influence by more than 2.5x and 2.0x respectively.
4. **Statistical OOD Check:** Bypasses if any feature's SHAP attribution z-score exceeds the regime-conditional threshold.
5. **Concentration Check:** Prevents single-variable dominance; bypasses if any feature represents a high percentage of absolute attribution (enforced only on net adjustments > 2.0% vol).
6. **Rank Stability Tracker:** Bypasses if the rolling 21-day Spearman rank correlation of feature attributions falls below the baseline.

### Regime-Conditional Thresholds

The gate dynamically tightens or loosens its parameters based on the VIX level, decoupling the parameter space to avoid false positive rejections during calm periods:

- **Stress Regime (VIX > 30):** Tight parameters (`ood_z = 2.0`, `concentration = 60%`, `relative_cap = 25%`).
- **Normal Regime (VIX <= 20):** Loose parameters (`ood_z = 4.5`, `concentration = 90%`, `relative_cap = 45%`).

### Operational Validation Metrics

- **COVID-19 Stress Fold (Feb 2020 – Sep 2020) Bypass Rate:** 63.69% (Target: ≥ 60.0% — Passed).
- **Calm Market Fold (2017 – 2019) Approval Rate:** 88.59% (Target: ≥ 85.0% — Passed).

---

## 5. Execution Instructions

Run all scripts from the root directory of your project.

### Step 1: Run the Unified Data & Feature Pipeline

```bash
python scripts/run_pipeline.py
```

This downloads raw yFinance series, calculates log returns natively, generates all features, and exports the clean feature matrix.

### Step 2: Run the Baseline Backtests

```bash
python scripts/run_baselines.py
```

Runs the annual-retraining walk-forward backtests (2012–2026) across standard GARCH, EGARCH, and native GARCH-X models, and logs all diagnostics to MLflow.

### Step 3: Run the Production Hybrid Engine

```bash
python scripts/run_hybrid.py
```

Trains the integrated EGARCH + XGBoost pipeline, evaluates out-of-sample forecasts through the active SHAP Safety Gate, outputs the final RMSE lift and DM significance, and logs the run to MLflow.

### Step 4: Open the MLflow Dashboard

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Open `http://127.0.0.1:5000` in your web browser to review the complete, unified experiment history.
