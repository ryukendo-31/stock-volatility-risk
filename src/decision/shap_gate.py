# src/decision/shap_gate.py
import numpy as np
import pandas as pd
import shap

class ShapSafetyGate:
    def __init__(self):
        self.explainer = None
        self.base_value = None
        self.best_iteration = None
        
        # Placeholders for training set absolute SHAP distributions (Phase 2e)
        self.shap_means = None
        self.shap_stds = None
        
        # Placeholders for COVID-19 fold distributions (Phase 2f)
        self.covid_shap_means = None

        # Placeholders for raw feature percentiles during COVID stress (Phase 2g calibration)
        self.covid_raw_medians = None
        self.covid_raw_90th = None


    def fit_explainer(self, xgb_model, X_train=None):
        """
        Fit SHAP TreeExplainer on the trained XGBoost model.
        Accepts X_train as a background dataset to enforce exact path attributions.
        """
        # Retrieve the best iteration from early stopping if it exists
        self.best_iteration = getattr(xgb_model, "best_iteration", None)
        
        # Initialize TreeExplainer with background data for exact scaling
        self.explainer = shap.TreeExplainer(xgb_model, data=X_train)
        
        # Extract expected value safely
        raw_expected = self.explainer.expected_value
        if isinstance(raw_expected, np.ndarray):
            self.base_value = float(raw_expected[0]) if raw_expected.ndim > 0 else float(raw_expected)
        else: 
            self.base_value = float(raw_expected)
            
        print(f" TreeExplainer fitted. Base Value: {self.base_value:.6f}")
        if self.best_iteration is not None:
            print(f"   Note: Model early stopping active. Truncating SHAP to first {self.best_iteration + 1} trees.")

        '''
        The below blocks calculate the shap attributes 
        to see how normally each variable behaves (Phase 2e & 2f)
        '''
        if X_train is not None:
            print("calculate |SHAP| mean and standard deviation")
            train_shap = self.compute_shap_values(X_train)

            # Phase 2e: Calculate absolute values to find effect regardless of direction
            abs_train_shap = np.abs(train_shap) 
            self.shap_means = pd.Series(np.mean(abs_train_shap, axis=0), index=X_train.columns)
            self.shap_stds = pd.Series(np.std(abs_train_shap, axis=0), index=X_train.columns).replace(0, 1e-6)
            print("attributes for shap calculated")
            
            #  Call the COVID-fold baseline computation (Phase 2f)
            self.compute_covid_baselines(X_train, train_shap)
        else:
            print("attribution calculation failed!!!")
            
        return self
    
    def compute_covid_baselines(self, X_train, train_shap):
        """
        Phase 2f: Slices the historical COVID-19 stress window from training features
        and computes the average absolute SHAP values during this crisis.
        """
        covid_mask = (X_train.index >= '2020-02-01') & (X_train.index <= '2020-09-30')
        covid_indices = np.where(covid_mask)[0]
        
        if len(covid_indices) > 0:
            # Extract SHAP values for the COVID period rows
            covid_shap = train_shap[covid_indices]
            abs_covid_shap = np.abs(covid_shap)
            
            # CORRECTED: Compute average absolute SHAP values using clean, highly stable numpy mean
            mean_abs_attribs = np.mean(abs_covid_shap, axis=0)
            self.covid_shap_means = pd.Series(mean_abs_attribs, index=X_train.columns)
            print(f" COVID-fold analysis complete. Found {len(covid_indices)} overlapping crisis days.")

            covid_features = X_train.iloc[covid_indices]
            self.covid_raw_medians = covid_features.median()
            self.covid_raw_90th = covid_features.quantile(0.90)
            #features and percentile from covid era calculated and done here
        else:
            print(" Warning: No historical COVID-19 dates found in the X_train index.")
            self.covid_shap_means = None
            self.covid_raw_medians = None
            self.covid_raw_90th = None

    def compute_shap_values(self, X):
        """Helper to extract raw SHAP value array for input features X."""
        if self.explainer is None:
            raise ValueError("Explainer is not fitted. Call fit_explainer first.")
            
        # If early stopping was active, restrict tree evaluation to match prediction limits
        kwargs = {}
        if self.best_iteration is not None:
            kwargs['tree_limit'] = self.best_iteration + 1
            
        raw_shap = self.explainer.shap_values(X, **kwargs)
        
        # Handle cases where SHAP returns a list containing [shap_values_array]
        if isinstance(raw_shap, list):
            return raw_shap[0]
        return raw_shap