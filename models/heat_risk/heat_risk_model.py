"""
ClimateShield AI — Child Heatstroke Risk Model

Estimates heatstroke and heat-related illness risk specifically for
children under 12, using child-adjusted physiological thresholds.

Children are NOT small adults:
  - Core temp rises 3–5x faster than adults in heat
  - Limited sweat gland density (especially under 6)
  - Heat cramps begin at lower exertion levels
  - School uniforms and outdoor activity increase exposure

Inputs:
  - ERA5 2m temperature (°C)
  - ERA5 relative humidity (%)
  - MODIS land surface temperature (urban heat island)
  - Time of day / school hours
  - UNICEF MICS child vulnerability index

Output:
  - heat_risk_score (0–100)
  - heat_risk_label (safe / caution / danger / emergency)
  - recommended_action (text)

References:
  - American Academy of Pediatrics (AAP) heat illness guidelines
  - WHO Heat-Health Action Plans
  - Basu & Ostro (2008) child-specific heat mortality thresholds
    https://doi.org/10.1289/ehp.11159
"""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ─── Child-specific thresholds ────────────────────────────────────────────────
# Wet Bulb Globe Temperature (WBGT) thresholds adjusted for children
# Source: AAP Policy Statement on Climatic Heat Stress (2011)

@dataclass
class HeatThreshold:
    wbgt_min: float
    wbgt_max: float
    label: str
    color: str
    school_action: str
    health_alert: bool

CHILD_HEAT_THRESHOLDS = [
    HeatThreshold(0,    26,   "Safe",      "#22c55e",
        "Normal outdoor activity permitted.",
        False),
    HeatThreshold(26,   28,   "Caution",   "#eab308",
        "Limit strenuous activity. Ensure water breaks every 20 min.",
        False),
    HeatThreshold(28,   30,   "Danger",    "#f97316",
        "Cancel outdoor PE classes. Move children indoors. Alert parents.",
        True),
    HeatThreshold(30,   100,  "Emergency", "#ef4444",
        "School closure recommended. Emergency cooling stations needed. "
        "Children under 5 face lethal risk within 2 hours.",
        True),
]


# ─── WBGT Estimation ─────────────────────────────────────────────────────────

def estimate_wbgt(temp_c: pd.Series, rh_pct: pd.Series,
                  solar_radiation: pd.Series = None) -> pd.Series:
    """
    Estimate Wet Bulb Globe Temperature (WBGT) from standard met variables.

    Uses the Liljegren et al. (2008) approximation — validated for
    outdoor conditions in South Asian climates.

    Args:
        temp_c:          Air temperature (°C)
        rh_pct:          Relative humidity (%)
        solar_radiation: Solar radiation (W/m²), optional

    Returns:
        WBGT estimate (°C)
    """
    # Wet bulb temperature via Stull (2011) formula
    # Reference: https://doi.org/10.1175/JAMC-D-11-0143.1
    tw = (temp_c * np.arctan(0.151977 * (rh_pct + 8.313659) ** 0.5)
          + np.arctan(temp_c + rh_pct)
          - np.arctan(rh_pct - 1.676331)
          + 0.00391838 * rh_pct ** 1.5 * np.arctan(0.023101 * rh_pct)
          - 4.686035)

    if solar_radiation is not None:
        # Globe temperature approximation under solar load
        tg = temp_c + 0.0146 * solar_radiation ** 0.75
        wbgt = 0.7 * tw + 0.2 * tg + 0.1 * temp_c
    else:
        # Indoor / shade approximation
        wbgt = 0.7 * tw + 0.3 * temp_c

    return wbgt.clip(0, 60)


def wbgt_to_risk(wbgt: pd.Series) -> pd.DataFrame:
    """Map WBGT values to child-specific risk levels."""
    labels, colors, actions, alerts = [], [], [], []

    for val in wbgt:
        matched = CHILD_HEAT_THRESHOLDS[-1]  # default: emergency
        for threshold in CHILD_HEAT_THRESHOLDS:
            if threshold.wbgt_min <= val < threshold.wbgt_max:
                matched = threshold
                break
        labels.append(matched.label)
        colors.append(matched.color)
        actions.append(matched.school_action)
        alerts.append(matched.health_alert)

    return pd.DataFrame({
        "wbgt_c": wbgt,
        "heat_risk_label": labels,
        "heat_risk_color": colors,
        "school_action": actions,
        "health_alert": alerts,
        "heat_risk_score": _wbgt_to_score(wbgt),
    })


def _wbgt_to_score(wbgt: pd.Series) -> pd.Series:
    """Normalize WBGT to 0–100 risk score."""
    return ((wbgt - 20) / (40 - 20) * 100).clip(0, 100)


# ─── Urban Heat Island Adjustment ────────────────────────────────────────────

def apply_uhi_adjustment(temp_c: pd.Series, is_urban: pd.Series) -> pd.Series:
    """
    Adjust temperature for urban heat island effect.
    Urban areas (schools in cities) can be 2–4°C hotter than ERA5 grid cell.

    Source: Imhoff et al. (2010) — urban heat island magnitude in Asia
    """
    uhi_delta = np.where(is_urban, 3.0, 0.0)  # conservative 3°C urban excess
    return temp_c + uhi_delta


# ─── School-Hours Weighting ───────────────────────────────────────────────────

def school_hours_weight(hour: pd.Series) -> pd.Series:
    """
    Weight risk by school hours (peak child exposure 8am–3pm).
    Pakistan school hours: 8:00–14:00 (summer), 8:00–15:00 (winter)
    """
    weights = pd.Series(0.3, index=hour.index)
    weights[(hour >= 8) & (hour <= 15)] = 1.0
    weights[(hour >= 11) & (hour <= 14)] = 1.3  # peak heat hours
    return weights


# ─── Main Risk Calculator ─────────────────────────────────────────────────────

class ChildHeatRiskCalculator:
    """
    Compute child-specific heat risk scores for a location/date grid.
    Designed for real-time use by the API layer.
    """

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Input df must have: temp_c, rh, [solar_radiation], [is_urban], [hour]
        Returns df with heat risk columns added.
        """
        required = ["temp_c", "rh"]
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        solar = df.get("solar_radiation_wm2")
        temp = df["temp_c"].copy()

        # Urban heat island adjustment
        if "is_urban" in df.columns:
            temp = apply_uhi_adjustment(temp, df["is_urban"])

        # WBGT calculation
        wbgt = estimate_wbgt(temp, df["rh"], solar)

        # Risk classification
        risk_df = wbgt_to_risk(wbgt)

        # School hours weighting
        if "hour" in df.columns:
            weight = school_hours_weight(df["hour"])
            risk_df["heat_risk_score"] = (risk_df["heat_risk_score"] * weight).clip(0, 100)

        # Vulnerability multiplier (UNICEF MICS)
        if "vulnerability_score" in df.columns:
            vuln = df["vulnerability_score"].fillna(0.5)
            risk_df["vulnerability_adjusted_score"] = (
                risk_df["heat_risk_score"] * (1 + 0.5 * vuln)
            ).clip(0, 100)

        return pd.concat([df.reset_index(drop=True), risk_df.reset_index(drop=True)], axis=1)

    def realtime_alert(self, temp_c: float, rh: float,
                       is_urban: bool = False,
                       vulnerability_score: float = 0.5) -> dict:
        """
        Single-location real-time heat risk assessment.
        Used by the mobile app and API endpoint.
        """
        df = pd.DataFrame([{
            "temp_c": temp_c,
            "rh": rh,
            "is_urban": is_urban,
            "vulnerability_score": vulnerability_score,
        }])
        result = self.calculate(df).iloc[0]
        return {
            "wbgt_c": round(float(result["wbgt_c"]), 1),
            "heat_risk_label": result["heat_risk_label"],
            "heat_risk_color": result["heat_risk_color"],
            "heat_risk_score": round(float(result["heat_risk_score"]), 1),
            "school_action": result["school_action"],
            "health_alert": bool(result["health_alert"]),
        }


# ─── Quick test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    calc = ChildHeatRiskCalculator()

    # Karachi June heatwave scenario
    karachi_heatwave = calc.realtime_alert(
        temp_c=46.0, rh=35.0, is_urban=True, vulnerability_score=0.7
    )
    print("Karachi heatwave scenario:")
    for k, v in karachi_heatwave.items():
        print(f"  {k}: {v}")

    # Lahore normal winter day
    lahore_winter = calc.realtime_alert(
        temp_c=18.0, rh=60.0, is_urban=True, vulnerability_score=0.5
    )
    print("\nLahore winter scenario:")
    for k, v in lahore_winter.items():
        print(f"  {k}: {v}")
