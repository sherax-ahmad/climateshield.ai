"""
ClimateShield AI — Unit Tests

Tests for core model logic using synthetic data (no real datasets required).
Run: pytest tests/unit/ -v
"""

import numpy as np
import pandas as pd
import pytest

from models.heat_risk.heat_risk_model import (
    ChildHeatRiskCalculator,
    estimate_wbgt,
    CHILD_HEAT_THRESHOLDS,
)
from models.air_quality.aqi_respiratory_model import AQIRespiratoryRiskModel


# ─── Heat Risk Tests ──────────────────────────────────────────────────────────

class TestChildHeatRiskCalculator:
    def setup_method(self):
        self.calc = ChildHeatRiskCalculator()

    def test_safe_conditions(self):
        result = self.calc.realtime_alert(temp_c=22.0, rh=50.0)
        assert result["heat_risk_label"] == "Safe"
        assert result["health_alert"] is False

    def test_karachi_heatwave(self):
        """Karachi June 2015-style heatwave: 46°C, 35% RH, urban."""
        result = self.calc.realtime_alert(temp_c=46.0, rh=35.0, is_urban=True)
        assert result["heat_risk_label"] == "Emergency"
        assert result["health_alert"] is True
        assert result["heat_risk_score"] > 80

    def test_danger_zone(self):
        result = self.calc.realtime_alert(temp_c=38.0, rh=65.0)
        assert result["heat_risk_label"] in ("Danger", "Emergency")

    def test_urban_heat_island_increases_risk(self):
        rural = self.calc.realtime_alert(temp_c=35.0, rh=50.0, is_urban=False)
        urban = self.calc.realtime_alert(temp_c=35.0, rh=50.0, is_urban=True)
        assert urban["heat_risk_score"] >= rural["heat_risk_score"]

    def test_wbgt_formula_range(self):
        temps = pd.Series([20.0, 30.0, 40.0, 45.0])
        rh = pd.Series([60.0, 70.0, 40.0, 30.0])
        wbgt = estimate_wbgt(temps, rh)
        assert (wbgt >= 0).all()
        assert (wbgt <= 60).all()

    def test_thresholds_are_ordered(self):
        for i in range(len(CHILD_HEAT_THRESHOLDS) - 1):
            assert CHILD_HEAT_THRESHOLDS[i].wbgt_max == CHILD_HEAT_THRESHOLDS[i + 1].wbgt_min

    def test_dataframe_input(self):
        df = pd.DataFrame({
            "temp_c": [20.0, 30.0, 40.0, 45.0],
            "rh":     [60.0, 70.0, 50.0, 35.0],
            "is_urban": [False, True, True, True],
        })
        result = self.calc.calculate(df)
        assert "heat_risk_label" in result.columns
        assert len(result) == 4
        assert result["heat_risk_label"].iloc[-1] == "Emergency"


# ─── AQI Respiratory Risk Tests ───────────────────────────────────────────────

class TestAQIRespiratoryModel:
    def setup_method(self):
        self.model = AQIRespiratoryRiskModel()

    def test_clean_air_is_low_risk(self):
        result = self.model.simple_rule_based_score(pm25=5.0)
        assert result["respiratory_risk_label"] == "Low"
        assert result["school_outdoor_ban"] is False

    def test_lahore_smog_is_emergency(self):
        result = self.model.simple_rule_based_score(pm25=300.0)
        assert result["respiratory_risk_label"] == "Emergency"
        assert result["school_outdoor_ban"] is True

    def test_who_threshold_flag(self):
        below = self.model.simple_rule_based_score(pm25=10.0)
        above = self.model.simple_rule_based_score(pm25=20.0)
        assert below["who_child_threshold_exceeded"] is False
        assert above["who_child_threshold_exceeded"] is True

    def test_moderate_risk_range(self):
        result = self.model.simple_rule_based_score(pm25=25.0)
        assert result["respiratory_risk_label"] == "Moderate"

    def test_risk_score_increases_with_pm25(self):
        scores = [
            self.model.simple_rule_based_score(pm25=p)["respiratory_risk_score"]
            for p in [5, 20, 50, 100]
        ]
        assert scores == sorted(scores), "Risk score should increase with PM2.5"


# ─── Feature Engineering Tests ────────────────────────────────────────────────

class TestFeatureEngineering:
    def test_lag_features_created(self):
        from data.scripts.preprocess import add_lag_features
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=30),
            "temp_c": np.random.uniform(20, 45, 30),
        })
        result = add_lag_features(df, "temp_c", lags=[1, 7])
        assert "temp_c_lag1d" in result.columns
        assert "temp_c_lag7d" in result.columns
        assert "temp_c_roll7d_mean" in result.columns

    def test_aqi_score_range(self):
        from data.scripts.preprocess import _calc_aqi
        df = pd.DataFrame({"pm25": [0, 12, 35, 55, 150, 300]})
        aqi = _calc_aqi(df)
        assert (aqi >= 0).all()
        assert (aqi <= 500).all()
        assert aqi.iloc[0] == 0.0
        assert aqi.iloc[-1] == 500.0

    def test_dewpoint_to_rh(self):
        from data.scripts.preprocess import _dewpoint_to_rh
        # At saturation, T == Td → RH = 100%
        t = pd.Series([300.0])   # ~27°C in K
        td = pd.Series([300.0])  # same
        rh = _dewpoint_to_rh(t, td)
        assert abs(float(rh.iloc[0]) - 100.0) < 1.0
