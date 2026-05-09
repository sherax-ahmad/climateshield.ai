#!/usr/bin/env python3
"""
ClimateShield AI — Feature Engineering Pipeline

Merges and preprocesses all raw datasets into a unified feature matrix
ready for model training.

Output: data/processed/features_{country}_{year}.parquet

Usage:
    python preprocess.py --country PAK --years 2018-2024
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)


# ─── ERA5 Climate Features ────────────────────────────────────────────────────

def load_era5(country: str, years: list) -> pd.DataFrame:
    """Load ERA5 NetCDF, aggregate to daily district-level features."""
    try:
        import xarray as xr
    except ImportError:
        raise ImportError("pip install xarray netcdf4")

    frames = []
    for year in years:
        fpath = RAW_DIR / "era5" / country / f"era5_{country}_{year}.nc"
        if not fpath.exists():
            log.warning(f"ERA5 {year} not found: {fpath}")
            continue

        ds = xr.open_dataset(fpath)
        # Convert to daily means
        df = ds[["t2m", "tp", "d2m"]].resample(time="1D").mean().to_dataframe()
        df = df.reset_index()
        # Kelvin → Celsius
        df["temp_c"] = df["t2m"] - 273.15
        # Dewpoint → relative humidity
        df["rh"] = _dewpoint_to_rh(df["t2m"], df["d2m"])
        df["precip_mm"] = df["tp"] * 1000  # m → mm
        df["year"] = year
        frames.append(df[["time", "latitude", "longitude", "temp_c", "rh", "precip_mm"]])

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _dewpoint_to_rh(t_k: pd.Series, td_k: pd.Series) -> pd.Series:
    """Magnus formula: relative humidity from temperature and dewpoint."""
    t = t_k - 273.15
    td = td_k - 273.15
    rh = 100 * np.exp((17.625 * td) / (243.04 + td)) / np.exp((17.625 * t) / (243.04 + t))
    return rh.clip(0, 100)


# ─── OpenAQ Features ──────────────────────────────────────────────────────────

def load_openaq(country: str) -> pd.DataFrame:
    """Load OpenAQ JSON measurements, aggregate to daily station averages."""
    import json

    aq_dir = RAW_DIR / "openaq" / country
    if not aq_dir.exists():
        log.warning(f"OpenAQ data not found: {aq_dir}")
        return pd.DataFrame()

    records = []
    for fpath in aq_dir.glob("measurements_*.json"):
        with open(fpath) as f:
            data = json.load(f)
        for m in data:
            records.append({
                "date":      m.get("date", {}).get("local", "")[:10],
                "parameter": m.get("parameter"),
                "value":     m.get("value"),
                "latitude":  m.get("coordinates", {}).get("latitude"),
                "longitude": m.get("coordinates", {}).get("longitude"),
            })

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "value"])

    # Pivot: one row per (date, lat, lon)
    pivoted = df.pivot_table(
        index=["date", "latitude", "longitude"],
        columns="parameter",
        values="value",
        aggfunc="mean",
    ).reset_index()
    pivoted.columns.name = None

    # Standardize column names
    col_map = {"pm25": "pm25", "pm10": "pm10", "no2": "no2", "o3": "o3"}
    for src, dst in col_map.items():
        if src not in pivoted.columns:
            pivoted[dst] = np.nan
        elif src != dst:
            pivoted.rename(columns={src: dst}, inplace=True)

    # AQI composite score (simplified EPA formula)
    pivoted["aqi_score"] = _calc_aqi(pivoted)
    return pivoted


def _calc_aqi(df: pd.DataFrame) -> pd.Series:
    """Simplified AQI from PM2.5 (μg/m³) using EPA breakpoints."""
    pm = df.get("pm25", pd.Series(dtype=float)).fillna(0)
    breakpoints = [
        (0, 12, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 500.4, 301, 500),
    ]
    aqi = pd.Series(index=df.index, dtype=float)
    for c_lo, c_hi, i_lo, i_hi in breakpoints:
        mask = (pm >= c_lo) & (pm <= c_hi)
        aqi[mask] = (i_hi - i_lo) / (c_hi - c_lo) * (pm[mask] - c_lo) + i_lo
    return aqi.fillna(0).clip(0, 500)


# ─── Lag & Rolling Features ───────────────────────────────────────────────────

def add_lag_features(df: pd.DataFrame, target_col: str, lags: list[int]) -> pd.DataFrame:
    """Add lag and rolling window features for time-series forecasting."""
    df = df.sort_values("date").copy()
    for lag in lags:
        df[f"{target_col}_lag{lag}d"] = df[target_col].shift(lag)
    for window in [7, 14, 30]:
        df[f"{target_col}_roll{window}d_mean"] = (
            df[target_col].shift(1).rolling(window).mean()
        )
        df[f"{target_col}_roll{window}d_max"] = (
            df[target_col].shift(1).rolling(window).max()
        )
    return df


# ─── Child Vulnerability Index ────────────────────────────────────────────────

def load_child_vulnerability(country: str) -> pd.DataFrame:
    """
    Load UNICEF MICS-derived child vulnerability scores per district.
    Source: https://mics.unicef.org/surveys

    Columns: district_code, stunting_rate, water_access_pct,
             healthcare_access_pct, vulnerability_score (0–1)
    """
    # Placeholder: replace with actual MICS data after download
    # The vulnerability_score weights final risk alerts
    fpath = RAW_DIR / "mics" / f"child_vulnerability_{country}.csv"
    if fpath.exists():
        return pd.read_csv(fpath)
    log.warning("MICS vulnerability data not found — using uniform weights")
    return pd.DataFrame()


# ─── Feature Matrix Assembly ──────────────────────────────────────────────────

def build_feature_matrix(country: str, years: list) -> pd.DataFrame:
    """
    Join all data sources into a single feature matrix.
    One row = one (district, date) observation.
    """
    log.info("Loading ERA5 climate data...")
    climate = load_era5(country, years)

    log.info("Loading OpenAQ air quality data...")
    air_quality = load_openaq(country)

    if climate.empty and air_quality.empty:
        log.error("No data loaded. Run download_all.py first.")
        return pd.DataFrame()

    # Merge on nearest date + spatial grid cell
    # Simplified: merge on date for now (spatial join in full implementation)
    if not climate.empty:
        climate["date"] = pd.to_datetime(climate["time"]).dt.date
        climate = climate.drop(columns=["time"])

    if not air_quality.empty and not climate.empty:
        df = pd.merge(
            climate.assign(date=pd.to_datetime(climate["date"])),
            air_quality.rename(columns={"date": "date"}),
            on=["date", "latitude", "longitude"],
            how="left",
        )
    elif not climate.empty:
        df = climate.copy()
    else:
        df = air_quality.copy()

    # Add temporal features
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df["month"] = df["date"].dt.month
        df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
        df["is_monsoon"] = df["month"].between(6, 9).astype(int)

    # Add lag features for key predictors
    for col in ["temp_c", "precip_mm", "pm25"]:
        if col in df.columns:
            df = add_lag_features(df, col, lags=[1, 3, 7, 14])

    log.info(f"Feature matrix: {df.shape[0]} rows × {df.shape[1]} columns")
    return df


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--country", default="PAK")
    parser.add_argument("--years", default="2018-2024")
    parser.add_argument("--output", default=str(PROC_DIR))
    args = parser.parse_args()

    start, end = map(int, args.years.split("-"))
    years = list(range(start, end + 1))

    log.info(f"Building feature matrix for {args.country} ({years[0]}–{years[-1]})")
    df = build_feature_matrix(args.country, years)

    if df.empty:
        log.error("Feature matrix is empty. Check raw data.")
        return

    out_path = Path(args.output) / f"features_{args.country}_{years[0]}_{years[-1]}.parquet"
    df.to_parquet(out_path, index=False)
    log.info(f"Saved features to {out_path}")

    # Summary stats
    log.info(f"\nFeature summary:\n{df.describe()}")


if __name__ == "__main__":
    main()
