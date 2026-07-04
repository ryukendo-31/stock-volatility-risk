# src/decision/shap_gate.py
import numpy as np
import pandas as pd
import shap
from collections import deque

class ShapSafetyGate:
    def __init__(self, max_absolute_adj=0.05, max_relative_adj=0.45, threshold_std=4.5, max_concentration_ratio=0.90, min_rank_correlation=0.40, min_adj_for_concentration=0.020):
        """
        max_absolute_adj: Default absolute volatility correction cap (5.0% vol).
        max_relative_adj: Default relative adjustment cap (45%).
        threshold_std: Default standard deviation threshold for SHAP OOD check (4.5).
        max_concentration_ratio: Default single-feature concentration cap (90%).
        min_rank_correlation: Default minimum Spearman rank stability (40%).
        min_adj_for_concentration: Minimum net adjustment to run concentration checks (2.0% vol).
        """
        self.explainer = None
        self.base_value = None
        self.best_iteration = None
        
        # Phase 2e baselines
        self.shap_means = None
        self.shap_stds = None
        self.baseline_rankings = None
        
        # Phase 2f baselines
        self.covid_shap_means = None
        self.covid_raw_medians = None
        self.covid_raw_90th = None

        # Static Gate parameters (used as fallback defaults)
        self.max_absolute_adj = max_absolute_adj
        self.max_relative_adj = max_relative_adj
        self.threshold_std = threshold_std
        self.max_concentration_ratio = max_concentration_ratio
        self.min_rank_correlation = min_rank_correlation
        self.min_adj_for_concentration = min_adj_for_concentration

        # Phase 2k: Sliding window of the last 21 trading days
        self.shap_history = deque(maxlen=21)

        # Phase 2g hardcoded Domain Overrides (Kurt_21 raised to 8.0 to prevent false positives)
        self.overrides = {
            'Vol_21d': 0.40,       
            'VIX_Lag_1': 40.0,     
            'Kurt_21': 8.0         
        }

        # Phase 2f Target Expansion check thresholds
        self.crisis_features = {
            'Vol_10d': 2.5,   
            'VIX_Lag_1': 2.0   
        }

    def fit_explainer(self, xgb_model, X_train=None, X_test=None):
        """
        Fit TreeExplainer and compute baselines (Phases 2d, 2e, & 2f).
        Accepts X_test separately to extract COVID features out-of-sample.
        """
        self.best_iteration = getattr(xgb_model, "best_iteration", None)
        self.explainer = shap.TreeExplainer(xgb_model, data=X_train)
        
        raw_expected = self.explainer.expected_value
        if isinstance(raw_expected, np.ndarray):
            self.base_value = float(raw_expected[0]) if raw_expected.ndim > 0 else float(raw_expected)
        else: 
            self.base_value = float(raw_expected)
            
        print(f"Phase 2d: TreeExplainer fitted. Base Value: {self.base_value:.6f}")
        
        if X_train is not None:
            print("calculate |SHAP| mean and standard deviation")
            train_shap = self.compute_shap_values(X_train)
            abs_train_shap = np.abs(train_shap)
            
            self.shap_means = pd.Series(np.mean(abs_train_shap, axis=0), index=X_train.columns)
            self.shap_stds = pd.Series(np.std(abs_train_shap, axis=0), index=X_train.columns).replace(0, 1e-6)
            self.baseline_rankings = self.shap_means.rank(ascending=False)
            print("attributes for shap calculated")
            
            # Extract COVID baselines from X_test if provided to prevent in-sample bias
            target_df = X_test if X_test is not None else X_train
            target_shap = self.compute_shap_values(target_df)
            self.compute_covid_baselines(target_df, target_shap)
        else:
            print("attribution calculation failed!!!")
            
        return self
    
    def compute_covid_baselines(self, X_data, shap_values):
        """Compute COVID-fold baseline metrics (Phases 2f & 2g)."""
        covid_mask = (X_data.index >= '2020-02-01') & (X_data.index <= '2020-09-30')
        covid_indices = np.where(covid_mask)[0]
        
        if len(covid_indices) > 0:
            covid_shap = shap_values[covid_indices]
            abs_covid_shap = np.abs(covid_shap)
            
            mean_abs_attribs = np.mean(abs_covid_shap, axis=0)
            self.covid_shap_means = pd.Series(mean_abs_attribs, index=X_data.columns)
            print(f"Phase 2f: COVID-fold SHAP analysis complete ({len(covid_indices)} days).")
            
            covid_features = X_data.iloc[covid_indices]
            self.covid_raw_medians = covid_features.median()
            self.covid_raw_90th = covid_features.quantile(0.90)
            print("Phase 2g: Raw feature percentiles computed for COVID-fold calibration.")
        else:
            print("Warning: No historical COVID-19 dates found in the data index.")

    def _get_regime_thresholds(self, row_dict):
        """
        Phase 2n: Dynamically scales gate thresholds based on current VIX.
        Decouples the parameter space entirely.
        """
        vix = row_dict.get('VIX_Lag_1', row_dict.get('VIX_Gap', 20.0))
        
        if vix > 30:  # Stress Regime (Tighten safety, default to EGARCH sooner)
            return {
                'ood_z': 2.0,
                'concentration': 0.60,
                'relative_cap': 0.25,
                'absolute_cap': 0.03,
                'min_rank_rho': 0.55,
                'min_adj_for_concentration': 0.025  
            }
        elif vix > 20:  # Transitional Regime
            return {
                'ood_z': 2.5,
                'concentration': 0.68,
                'relative_cap': 0.32,
                'absolute_cap': 0.04,
                'min_rank_rho': 0.47,
                'min_adj_for_concentration': 0.015
            }
        else:  # Calm Regime (Loosen limits to protect normal-day approval rate)
            return {
                'ood_z': self.threshold_std,
                'concentration': self.max_concentration_ratio,
                'relative_cap': self.max_relative_adj,
                'absolute_cap': self.max_absolute_adj,
                'min_rank_rho': self.min_rank_correlation,
                'min_adj_for_concentration': self.min_adj_for_concentration
            }

    def evaluate_prediction_safety(self, X_row, egarch_pred, xgb_adjustment):
        """
        Runs the cascading safety checks sequentially using regime-conditional limits.
        """
        if isinstance(X_row, pd.Series):
            X_row_df = pd.DataFrame([X_row])
        else:
            X_row_df = X_row
            
        row_dict = X_row_df.iloc[0].to_dict()
        
        # Load VIX-dependent thresholds dynamically
        thresholds = self._get_regime_thresholds(row_dict)

        # -------------------------------------------------------------
        # Phase 2g: Hardcoded Domain Overrides Guard
        # -------------------------------------------------------------
        for feat, limit in self.overrides.items():
            if feat in row_dict and row_dict[feat] > limit:
                return "REJECTED", "DOMAIN_OVERRIDE", {
                    "trigger_feature": feat, "feature_value": row_dict[feat], "limit": limit
                }

        # -------------------------------------------------------------
        # Phase 2h: Magnitude Gate Guard
        # -------------------------------------------------------------
        abs_adj = np.abs(xgb_adjustment)
        
        # Absolute Check
        if abs_adj > thresholds['absolute_cap']:
            return "REJECTED", "MAGNITUDE_ABS_LIMIT", {
                "xgb_adjustment": xgb_adjustment, "max_absolute_adj": thresholds['absolute_cap']
            }
            
        # Relative Check
        relative_ratio = abs_adj / egarch_pred if egarch_pred > 0 else 0
        if relative_ratio > thresholds['relative_cap']:
            return "REJECTED", "MAGNITUDE_REL_LIMIT", {
                "xgb_adjustment": xgb_adjustment, "relative_ratio": relative_ratio, "max_relative_adj": thresholds['relative_cap']
            }

        # Calculate SHAP values for downstream checks
        row_shap = self.compute_shap_values(X_row_df)[0]
        abs_row_shap = np.abs(row_shap)

        # -------------------------------------------------------------
        # Phase 2f Gap: Empirical Attribution Expansion Check
        # -------------------------------------------------------------
        if self.shap_means is not None:
            for feat, expansion_cap in self.crisis_features.items():
                if feat in X_row_df.columns:
                    feat_idx = list(X_row_df.columns).index(feat)
                    ratio = abs_row_shap[feat_idx] / (self.shap_means[feat] + 1e-9)
                    if ratio > expansion_cap:
                        return "REJECTED", "SHAP_EXPANSION_RATIO", {
                            "trigger_feature": feat,
                            "expansion_ratio": ratio,
                            "cap": expansion_cap,
                        }

        # -------------------------------------------------------------
        # Phase 2i: SHAP-based Out-of-Distribution (OOD) Guard
        # -------------------------------------------------------------
        shap_z_scores = (abs_row_shap - self.shap_means.values) / self.shap_stds.values
        max_shap_z = np.max(shap_z_scores)
        
        most_anomalous_idx = np.argmax(shap_z_scores)
        most_anomalous_feat = X_row_df.columns[most_anomalous_idx]
        
        if max_shap_z > thresholds['ood_z']:
            return "REJECTED", "SHAP_OOD_LIMIT", {
                "trigger_feature": most_anomalous_feat,
                "shap_value": row_shap[most_anomalous_idx],
                "z_score": max_shap_z,
                "threshold": thresholds['ood_z']
            }

        # -------------------------------------------------------------
        # Phase 2j: SHAP Attribution Concentration Guard
        # -------------------------------------------------------------
        total_abs_attribution = np.sum(abs_row_shap)
        
        if total_abs_attribution > 0:
            max_attrib = np.max(abs_row_shap)
            concentration_ratio = max_attrib / total_abs_attribution
            max_attrib_idx = np.argmax(abs_row_shap)
            max_attrib_feat = X_row_df.columns[max_attrib_idx]
        else:
            concentration_ratio = 0.0
            max_attrib_feat = "None"
            
        if abs_adj > thresholds['min_adj_for_concentration']:
            if concentration_ratio > thresholds['concentration']:
                return "REJECTED", "SHAP_CONCENTRATION_LIMIT", {
                    "trigger_feature": max_attrib_feat,
                    "concentration_ratio": concentration_ratio,
                    "limit": thresholds['concentration'],
                    "total_abs_attribution": total_abs_attribution
                }

        # -------------------------------------------------------------
        # Phase 2k: Rank Stability Guard (21-day sliding memory)
        # -------------------------------------------------------------
        self.shap_history.append(abs_row_shap)
        
        if len(self.shap_history) == 21:
            rolling_avg_shap = np.mean(self.shap_history, axis=0)
            rolling_series = pd.Series(rolling_avg_shap, index=X_row_df.columns)
            rolling_rankings = rolling_series.rank(ascending=False)
            
            rank_correlation = rolling_rankings.corr(self.baseline_rankings, method='spearman')
            
            if rank_correlation < thresholds['min_rank_rho']:
                return "REJECTED", "SHAP_RANK_INSTABILITY", {
                    "rank_correlation": rank_correlation,
                    "limit": thresholds['min_rank_rho']
                }

        # If all active guards pass
        return "APPROVED", "NONE", {}

    def compute_shap_values(self, X):
        """Helper to extract raw SHAP value array."""
        if self.explainer is None:
            raise ValueError("Explainer is not fitted. Call fit_explainer first.")
            
        kwargs = {}
        if self.best_iteration is not None:
            kwargs['tree_limit'] = self.best_iteration + 1
            
        raw_shap = self.explainer.shap_values(X, **kwargs)
        if isinstance(raw_shap, list):
            return raw_shap[0]
        return raw_shap