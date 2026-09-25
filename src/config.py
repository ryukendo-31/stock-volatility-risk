"""Single source of truth for pipeline constants and model settings.

Tunable values here are only *starting points*. Every number used in a reported
walk-forward run must come from configs/frozen_params.json, which is written by
scripts/calibrate.py using data that ends before the first walk-forward test date.
"""
import json
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
FEATURES_PATH = PROCESSED_DIR / "features.csv"
RESULTS_DIR = BASE_DIR / "results"
REPORTS_DIR = BASE_DIR / "reports"
FROZEN_PARAMS_PATH = BASE_DIR / "configs" / "frozen_params.json"
MLFLOW_TRACKING_URI = "sqlite:///" + (BASE_DIR / "mlflow.db").as_posix()

TARGET = "Target_Vol_Next_5d"
HORIZON = 5  # the target at t covers returns t+1 .. t+5 ,predicting 5 trading days into the future
PURGE_ROWS = HORIZON  # labels whose window reaches into the test fold are dropped
START_WINDOW = 3000  # rows in the first walk-forward training window, nearly 12 years allows the model to understand atleast one market cycle bull and bear
STEP_SIZE = 252  # rows per walk-forward test fold
SEED = 42
N_SIMULATIONS = 1000

# Legacy values, used only as the base that calibrate.py searches around.
DEFAULT_XGB_PARAMS = {
    "max_depth": 2,
    "learning_rate": 0.01,
    "n_estimators": 1500,
    "reg_alpha": 5.0,
    "reg_lambda": 10.0,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "early_stopping_rounds": 50,
}

DEFAULT_GATE_CONFIG = {
    # Fixed a priori (not searched): features below this share of mean |SHAP| cannot trigger
    # SHAP-based vetoes, and |SHAP| std is floored at this fraction of the feature's mean |SHAP|.
    "min_feature_share": 0.05,
    "std_floor_frac": 0.10,
    # Searched by calibrate.py.
    "input_quantile": 0.995,  # veto if a material feature leaves the training [1-q, q] range
    "ood_z": 3.5,
    "stress_multiplier": 0.75,  # 1.0 = no regime tightening; < 1 tightens every limit when VIX is highWhen the VIX is high (market panic), we multiply our safety thresholds by 0.75 (tightening them). This forces the ML to behave even more conservatively during crashes.
    # Legacy calm-regime values, held fixed during calibration.
    "max_concentration": 0.80,
    "min_adj_for_concentration": 0.02,
    "max_absolute_adj": 0.05,
    "max_relative_adj": 0.45,
    "min_rank_rho": 0.40,
    "rank_window": 21,
    "vix_transition": 20.0,
    "vix_stress": 30.0,
}


def load_frozen_params(path=FROZEN_PARAMS_PATH):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python scripts/calibrate.py` first: reported runs must use "
            "parameters calibrated only on data before the first walk-forward test date."
        )
    with path.open() as f:
        return json.load(f)


def assert_calibration_precedes(params, first_test_date):
    """Raise unless every return used in calibration labels predates `first_test_date`."""
    try:
        last_used = pd.Timestamp(params["provenance"]["calibration_last_label_return_date"])
    except (KeyError, TypeError) as exc:
        raise ValueError("Frozen parameters carry no calibration provenance; re-run calibrate.py.") from exc
    if last_used >= pd.Timestamp(first_test_date):
        raise ValueError(
            f"Calibration used returns up to {last_used.date()}, but evaluation starts "
            f"{pd.Timestamp(first_test_date).date()}. The walk-forward would not be out-of-sample."
        )