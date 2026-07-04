# src/decision/shap_gate.py
import numpy as np
import pandas as pd
import shap

class ShapSafetyGate:
    def __init__(self, max_absolute_adj=0.05, max_relative_adj=0.35, threshold_std=3.0, max_concentration_ratio=0.45):
        """
        max_absolute_adj: Maximum allowed absolute volatility correction (default 5.0% vol).
        max_relative_adj: Maximum allowed percentage correction relative to EGARCH baseline (default 35%).
        threshold_std: Maximum allowed standard deviations for a SHAP attribution z-score (default 3.0).
        max_concentration_ratio: Maximum allowed attribution proportion for a single feature (default 45%).
        """
        self.explainer = None
        self.base_value = None
        self.best_iteration = None
        
        # Phase 2e baselines
        self.shap_means = None
        self.shap_stds = None
        
        # Phase 2f baselines
        self.covid_shap_means = None
        self.covid_raw_medians = None
        self.covid_raw_90th = None

        # Phase 2h Magnitude Gate parameters
        self.max_absolute_adj = max_absolute_adj
        self.max_relative_adj = max_relative_adj
        
        # Phase 2i SHAP OOD threshold
        self.threshold_std = threshold_std
        
        # Phase 2j SHAP concentration limit
        self.max_concentration_ratio = max_concentration_ratio

        # Phase 2g hardcoded Domain Overrides (derived from COVID-19 90th percentile peaks)
        self.overrides = {
            'Vol_21d': 0.80,       # Rejects if 21-day realized volatility exceeds 80%
            'VIX_Lag_1': 50.0,     # Rejects if yesterday's VIX exceeds 50.0
            'Kurt_21': 4.5         # Rejects if excess kurtosis exceeds 4.5
        }

    def fit_explainer(self, xgb_model, X_train=None):
        """Fit TreeExplainer and compute baselines (Phases 2d, 2e, & 2f)."""
        self.best_iteration = getattr(xgb_model, "best_iteration", None)
        self.explainer = shap.TreeExplainer(xgb_model, data=X_train)
        
        raw_expected = self.explainer.expected_value
        if isinstance(raw_expected, np.ndarray):
            self.base_value = float(raw_expected[0]) if raw_expected.ndim > 0 else float(raw_expected)
        else: 
            self.base_value = float(raw_expected)
            
        print(f" Phase 2d: TreeExplainer fitted. Base Value: {self.base_value:.6f}")
        
        if X_train is not None:
            print("calculate |SHAP| mean and standard deviation")
            train_shap = self.compute_shap_values(X_train)
            abs_train_shap = np.abs(train_shap)
            
            self.shap_means = pd.Series(np.mean(abs_train_shap, axis=0), index=X_train.columns)
            self.shap_stds = pd.Series(np.std(abs_train_shap, axis=0), index=X_train.columns).replace(0, 1e-6)
            print("attributes for shap calculated")
            
            self.compute_covid_baselines(X_train, train_shap)
            
        return self
    
    def compute_covid_baselines(self, X_train, train_shap):
        """Compute COVID-fold baseline metrics (Phases 2f & 2g)."""
        covid_mask = (X_train.index >= '2020-02-01') & (X_train.index <= '2020-09-30')
        covid_indices = np.where(covid_mask)[0]
        
        if len(covid_indices) > 0:
            covid_shap = train_shap[covid_indices]
            abs_covid_shap = np.abs(covid_shap)
            
            mean_abs_attribs = np.mean(abs_covid_shap, axis=0)
            self.covid_shap_means = pd.Series(mean_abs_attribs, index=X_train.columns)
            print(f"Phase 2f: COVID-fold SHAP analysis complete ({len(covid_indices)} days).")
            
            covid_features = X_train.iloc[covid_indices]
            self.covid_raw_medians = covid_features.median()
            self.covid_raw_90th = covid_features.quantile(0.90)
            print(" Phase 2g: Raw feature percentiles computed for COVID-fold calibration.")
        else:
            print(" Warning: No historical COVID-19 dates found in the X_train index.")

    def evaluate_prediction_safety(self, X_row, egarch_pred, xgb_adjustment):
        """
        Runs the cascading safety checks sequentially.
        Returns: status ("APPROVED" or "REJECTED"), active reason code, and diagnostic values.
        """
        if isinstance(X_row, pd.Series):
            X_row_df = pd.DataFrame([X_row])
        else:
            X_row_df = X_row
            
        row_dict = X_row_df.iloc[0].to_dict()

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
        
        # Absolute Magnitude Check
        if abs_adj > self.max_absolute_adj:
            return "REJECTED", "MAGNITUDE_ABS_LIMIT", {
                "xgb_adjustment": xgb_adjustment, "max_absolute_adj": self.max_absolute_adj
            }
            
        # Relative Magnitude Check
        relative_ratio = abs_adj / egarch_pred if egarch_pred > 0 else 0
        if relative_ratio > self.max_relative_adj:
            return "REJECTED", "MAGNITUDE_REL_LIMIT", {
                "xgb_adjustment": xgb_adjustment, "relative_ratio": relative_ratio, "max_relative_adj": self.max_relative_adj
            }

        # -------------------------------------------------------------
        # Phase 2i: SHAP-based Out-of-Distribution (OOD) Guard
        # -------------------------------------------------------------
        row_shap = self.compute_shap_values(X_row_df)[0]
        abs_row_shap = np.abs(row_shap)
        
        # Compute z-scores for each feature's contribution
        shap_z_scores = (abs_row_shap - self.shap_means.values) / self.shap_stds.values
        max_shap_z = np.max(shap_z_scores)
        
        most_anomalous_idx = np.argmax(shap_z_scores)
        most_anomalous_feat = X_row_df.columns[most_anomalous_idx]
        
        if max_shap_z > self.threshold_std:
            return "REJECTED", "SHAP_OOD_LIMIT", {
                "trigger_feature": most_anomalous_feat,
                "shap_value": row_shap[most_anomalous_idx],
                "z_score": max_shap_z,
                "threshold": self.threshold_std
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
            
        if concentration_ratio > self.max_concentration_ratio:
            return "REJECTED", "SHAP_CONCENTRATION_LIMIT", {
                "trigger_feature": max_attrib_feat,
                "concentration_ratio": concentration_ratio,
                "limit": self.max_concentration_ratio,
                "total_abs_attribution": total_abs_attribution
            }

        # If all active guards pass
        return "APPROVED", "NONE", {}

    def compute_shap_values(self, X):
        """Helper to extract raw SHAP value array for input features X."""
        if self.explainer is None:
            raise ValueError("Explainer is not fitted. Call fit_explainer first.")
            
        kwargs = {}
        if self.best_iteration is not None:
            kwargs['tree_limit'] = self.best_iteration + 1
            
        raw_shap = self.explainer.shap_values(X, **kwargs)
        if isinstance(raw_shap, list):
            return raw_shap[0]
        return raw_shap