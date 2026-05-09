"""
ClimateShield AI — Dengue Risk Forecasting Model

Predicts district-level dengue outbreak risk 7 days ahead using:
- Temperature and humidity (ERA5)
- Precipitation / standing water proxy (GPM)
- Historical case counts (Pakistan NIH / WHO EMRO)
- Lag and rolling window features

Model: Gradient Boosted Trees (XGBoost) with SHAP explanations
Target: dengue_risk_label (0=low, 1=medium, 2=high, 3=emergency)

References:
  - Hii et al. (2012) "Forecast of Dengue Incidence Using Temperature and Rainfall"
    https://doi.org/10.1371/journal.pntd.0001908
  - Stolerman et al. (2019) dengue forecasting via climate signals
"""

import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ─── Feature specification ────────────────────────────────────────────────────

CLIMATE_FEATURES = [
    "temp_c",           "temp_c_lag1d",   "temp_c_lag3d",   "temp_c_lag7d",
    "temp_c_lag14d",    "temp_c_roll7d_mean",  "temp_c_roll14d_mean",
    "rh",               "rh_lag1d",       "rh_lag7d",       "rh_roll7d_mean",
    "precip_mm",        "precip_mm_lag1d","precip_mm_lag3d","precip_mm_lag7d",
    "precip_mm_roll7d_mean", "precip_mm_roll14d_mean", "precip_mm_roll30d_mean",
]

TEMPORAL_FEATURES = [
    "month", "week_of_year", "is_monsoon",
]

AQ_FEATURES = [
    "pm25", "aqi_score",  # dengue outbreaks can co-occur with smoke (open burning)
]

VULNERABILITY_FEATURES = [
    "vulnerability_score",      # UNICEF MICS child vulnerability index
    "children_under12_pct",     # % population under 12 per district
]

ALL_FEATURES = CLIMATE_FEATURES + TEMPORAL_FEATURES + AQ_FEATURES + VULNERABILITY_FEATURES
TARGET = "dengue_risk_label"


# ─── Risk thresholds ─────────────────────────────────────────────────────────

@dataclass
class RiskLevel:
    label: int
    name: str
    color: str
    child_alert: bool
    description: str

RISK_LEVELS = [
    RiskLevel(0, "Low",       "#22c55e", False, "Normal conditions. Routine precautions."),
    RiskLevel(1, "Medium",    "#eab308", False, "Elevated risk. Schools advised to check drainage."),
    RiskLevel(2, "High",      "#f97316", True,  "High risk. Clinics should pre-stock ORS and antivirals."),
    RiskLevel(3, "Emergency", "#ef4444", True,  "Outbreak likely. Alert health authorities immediately."),
]


# ─── Model ────────────────────────────────────────────────────────────────────

class DengueRiskModel:
    """
    XGBoost-based dengue risk forecasting model.

    Designed for district-level, 7-day-ahead prediction
    calibrated for child health impact.
    """

    def __init__(self, n_estimators: int = 500, max_depth: int = 6,
                 learning_rate: float = 0.05, random_state: int = 42):
        self.params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 5,
            "scale_pos_weight": 3.0,  # address class imbalance (outbreaks are rare)
            "random_state": random_state,
            "use_label_encoder": False,
            "eval_metric": "mlogloss",
            "n_jobs": -1,
        }
        self.model = None
        self.feature_names: list[str] = []
        self.calibrator = None

    # ── Training ──────────────────────────────────────────────────────────

    def train(self, df: pd.DataFrame, eval_df: Optional[pd.DataFrame] = None):
        """Train on feature matrix. df must include TARGET column."""
        try:
            from xgboost import XGBClassifier
            from sklearn.model_selection import StratifiedKFold
        except ImportError:
            raise ImportError("pip install xgboost scikit-learn")

        available = [f for f in ALL_FEATURES if f in df.columns]
        missing = set(ALL_FEATURES) - set(available)
        if missing:
            log.warning(f"Missing features (will use NaN): {missing}")

        X = df[available].copy()
        y = df[TARGET].values
        self.feature_names = available

        log.info(f"Training DengueRiskModel on {len(X)} samples, {len(available)} features")
        log.info(f"Class distribution: {pd.Series(y).value_counts().to_dict()}")

        # Fill missing features with median
        X = X.fillna(X.median())

        self.model = XGBClassifier(**self.params)

        if eval_df is not None:
            X_eval = eval_df[available].fillna(X.median())
            y_eval = eval_df[TARGET].values
            self.model.fit(X, y, eval_set=[(X_eval, y_eval)], verbose=50)
        else:
            self.model.fit(X, y)

        log.info("Training complete.")
        self._log_feature_importance()

    # ── Inference ─────────────────────────────────────────────────────────

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict risk level for each row.
        Returns df with added columns:
          - predicted_risk_label (0–3)
          - risk_name
          - risk_proba_low/medium/high/emergency
          - child_alert (bool)
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() or load_model() first.")

        X = df[self.feature_names].fillna(df[self.feature_names].median())
        probas = self.model.predict_proba(X)
        labels = self.model.predict(X)

        result = df.copy()
        result["predicted_risk_label"] = labels
        result["risk_name"] = [RISK_LEVELS[l].name for l in labels]
        result["risk_color"] = [RISK_LEVELS[l].color for l in labels]
        result["child_alert"] = [RISK_LEVELS[l].child_alert for l in labels]

        for i, level in enumerate(RISK_LEVELS):
            col = f"proba_{level.name.lower()}"
            result[col] = probas[:, i] if i < probas.shape[1] else 0.0

        return result

    def predict_risk_score(self, features: dict) -> dict:
        """
        Single-location, single-time prediction.
        Input: dict of feature values
        Returns: risk assessment dict
        """
        df = pd.DataFrame([features])
        result = self.predict(df).iloc[0]
        label = int(result["predicted_risk_label"])
        return {
            "risk_label": label,
            "risk_name": RISK_LEVELS[label].name,
            "risk_color": RISK_LEVELS[label].color,
            "child_alert": bool(RISK_LEVELS[label].child_alert),
            "description": RISK_LEVELS[label].description,
            "probabilities": {
                lvl.name.lower(): float(result.get(f"proba_{lvl.name.lower()}", 0))
                for lvl in RISK_LEVELS
            },
        }

    # ── SHAP Explanations ─────────────────────────────────────────────────

    def explain(self, df: pd.DataFrame, max_display: int = 15):
        """
        Generate SHAP feature importance plot.
        Helps health workers understand WHY the risk score is high.
        """
        try:
            import shap
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("pip install shap matplotlib")

        X = df[self.feature_names].fillna(df[self.feature_names].median())
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X)

        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X, max_display=max_display, show=False)
        plt.title("ClimateShield AI — Dengue Risk Drivers")
        plt.tight_layout()
        plt.savefig("dengue_shap_importance.png", dpi=150)
        log.info("SHAP plot saved to dengue_shap_importance.png")
        return shap_values

    # ── Evaluation ────────────────────────────────────────────────────────

    def evaluate(self, df: pd.DataFrame) -> dict:
        """Return classification metrics on a held-out set."""
        from sklearn.metrics import (
            classification_report, confusion_matrix, roc_auc_score
        )

        result = self.predict(df)
        y_true = df[TARGET].values
        y_pred = result["predicted_risk_label"].values

        report = classification_report(
            y_true, y_pred,
            target_names=[lvl.name for lvl in RISK_LEVELS],
            output_dict=True,
        )
        cm = confusion_matrix(y_true, y_pred)

        log.info(f"\n{classification_report(y_true, y_pred, target_names=[l.name for l in RISK_LEVELS])}")
        return {"report": report, "confusion_matrix": cm.tolist()}

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, path: str):
        import joblib
        data = {"model": self.model, "feature_names": self.feature_names, "params": self.params}
        joblib.dump(data, path)
        log.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str) -> "DengueRiskModel":
        import joblib
        data = joblib.load(path)
        instance = cls()
        instance.model = data["model"]
        instance.feature_names = data["feature_names"]
        instance.params = data["params"]
        log.info(f"Model loaded from {path}")
        return instance

    # ── Internal helpers ──────────────────────────────────────────────────

    def _log_feature_importance(self):
        if self.model is None:
            return
        importances = self.model.feature_importances_
        fi = pd.Series(importances, index=self.feature_names).sort_values(ascending=False)
        log.info(f"Top 10 features:\n{fi.head(10)}")


# ─── Training script ─────────────────────────────────────────────────────────

def train_pakistan_model(data_path: str, model_out: str = "dengue_model_pak.joblib"):
    """Train and save the Pakistan dengue model from processed features."""
    log.info("Loading features...")
    df = pd.read_parquet(data_path)

    if TARGET not in df.columns:
        raise ValueError(
            f"Column '{TARGET}' not found. "
            "Ensure dengue case labels are merged into the feature matrix. "
            "See docs/datasets/DATASETS.md for Pakistan NIH data sources."
        )

    # Temporal train/test split (never leak future into past)
    df["date"] = pd.to_datetime(df["date"])
    split_date = df["date"].max() - pd.Timedelta(days=90)
    train_df = df[df["date"] <= split_date]
    test_df  = df[df["date"] >  split_date]

    log.info(f"Train: {len(train_df)} | Test: {len(test_df)}")

    model = DengueRiskModel()
    model.train(train_df, eval_df=test_df)
    metrics = model.evaluate(test_df)
    model.save(model_out)

    return model, metrics


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python dengue_model.py <features.parquet>")
        sys.exit(1)
    train_pakistan_model(sys.argv[1])
