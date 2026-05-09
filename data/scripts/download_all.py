#!/usr/bin/env python3
"""
ClimateShield AI — Master Dataset Downloader
Downloads all required datasets for the Pakistan pilot.

Usage:
    python download_all.py --country PAK --years 2018-2024
    python download_all.py --country BGD --years 2020-2024 --datasets era5 openaq
"""

import argparse
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# Country bounding boxes [N, W, S, E]
COUNTRY_BBOX = {
    "PAK": [37.1, 60.9, 23.6, 77.8],   # Pakistan
    "BGD": [26.6, 88.0, 20.7, 92.7],   # Bangladesh
    "NGA": [13.9, 2.7,  4.3, 14.7],    # Nigeria
}

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"


def parse_args():
    p = argparse.ArgumentParser(description="Download ClimateShield AI datasets")
    p.add_argument("--country", default="PAK", choices=COUNTRY_BBOX.keys())
    p.add_argument("--years", default="2018-2024", help="Year range, e.g. 2018-2024")
    p.add_argument(
        "--datasets",
        nargs="+",
        default=["era5", "openaq", "firms", "gpm", "modis", "flood", "boundaries"],
        help="Which datasets to download",
    )
    p.add_argument("--dry-run", action="store_true", help="Print steps without downloading")
    return p.parse_args()


def year_range(years_str: str):
    start, end = map(int, years_str.split("-"))
    return list(range(start, end + 1))


# ─── Individual downloaders ──────────────────────────────────────────────────

def download_era5(country: str, years: list[int], dry_run: bool):
    """
    ERA5-Land daily climate data via CDS API.
    Docs: https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land
    Auth: ~/.cdsapirc  (key from https://cds.climate.copernicus.eu/user/login)
    """
    try:
        import cdsapi
    except ImportError:
        log.error("cdsapi not installed. Run: pip install cdsapi")
        return

    bbox = COUNTRY_BBOX[country]
    out_dir = RAW_DIR / "era5" / country
    out_dir.mkdir(parents=True, exist_ok=True)

    variables = [
        "2m_temperature",
        "2m_dewpoint_temperature",
        "total_precipitation",
        "surface_pressure",
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
    ]

    if dry_run:
        log.info(f"[DRY RUN] ERA5: Would download {len(years)} years for {country}")
        return

    c = cdsapi.Client()
    for year in years:
        out_file = out_dir / f"era5_{country}_{year}.nc"
        if out_file.exists():
            log.info(f"ERA5 {year}: already exists, skipping")
            continue
        log.info(f"ERA5: downloading {year} for {country}...")
        c.retrieve(
            "reanalysis-era5-land",
            {
                "variable": variables,
                "year": str(year),
                "month": [f"{m:02d}" for m in range(1, 13)],
                "day": [f"{d:02d}" for d in range(1, 32)],
                "time": ["00:00", "06:00", "12:00", "18:00"],
                "area": bbox,
                "format": "netcdf",
            },
            str(out_file),
        )
        log.info(f"ERA5 {year}: saved to {out_file}")


def download_openaq(country: str, years: list[int], dry_run: bool):
    """
    OpenAQ air quality measurements (PM2.5, PM10, NO2, O3).
    API Docs: https://docs.openaq.org/
    No auth required for basic access.
    """
    import requests, json, time

    # ISO country codes for OpenAQ
    country_codes = {"PAK": "PK", "BGD": "BD", "NGA": "NG"}
    cc = country_codes.get(country, country[:2])

    out_dir = RAW_DIR / "openaq" / country
    out_dir.mkdir(parents=True, exist_ok=True)

    if dry_run:
        log.info(f"[DRY RUN] OpenAQ: Would download stations for {country} ({cc})")
        return

    # Step 1: get station list
    log.info(f"OpenAQ: fetching stations for {cc}...")
    resp = requests.get(
        "https://api.openaq.org/v3/locations",
        params={"country_id": cc, "limit": 200, "page": 1},
        headers={"X-API-Key": os.getenv("OPENAQ_API_KEY", "")},
        timeout=30,
    )
    resp.raise_for_status()
    stations = resp.json().get("results", [])
    log.info(f"OpenAQ: found {len(stations)} stations")

    # Save station index
    with open(out_dir / f"stations_{country}.json", "w") as f:
        json.dump(stations, f, indent=2)

    # Step 2: download measurements per station
    for station in stations:
        sid = station["id"]
        name = station.get("name", str(sid)).replace("/", "_")
        out_file = out_dir / f"measurements_{name}_{sid}.json"
        if out_file.exists():
            continue

        log.info(f"OpenAQ: station {name}...")
        measurements = []
        for year in years:
            r = requests.get(
                f"https://api.openaq.org/v3/locations/{sid}/measurements",
                params={
                    "date_from": f"{year}-01-01T00:00:00Z",
                    "date_to": f"{year}-12-31T23:59:59Z",
                    "limit": 10000,
                    "parameters": "pm25,pm10,no2,o3",
                },
                timeout=60,
            )
            if r.status_code == 200:
                measurements.extend(r.json().get("results", []))
            time.sleep(0.3)  # rate limiting

        with open(out_file, "w") as f:
            json.dump(measurements, f)

    log.info("OpenAQ: done.")


def download_firms(country: str, years: list[int], dry_run: bool):
    """
    NASA FIRMS active fire + heat anomalies (VIIRS 375m).
    API Docs: https://firms.modaps.eosdis.nasa.gov/api/
    Key: https://firms.modaps.eosdis.nasa.gov/api/area/ (free)
    """
    import requests

    api_key = os.getenv("FIRMS_API_KEY", "")
    if not api_key and not dry_run:
        log.warning("FIRMS_API_KEY not set. Get one free at https://firms.modaps.eosdis.nasa.gov/api/area/")

    out_dir = RAW_DIR / "firms" / country
    out_dir.mkdir(parents=True, exist_ok=True)

    if dry_run:
        log.info(f"[DRY RUN] FIRMS: Would download VIIRS data for {country}")
        return

    # Country code mapping for FIRMS
    firms_country = {"PAK": "PAK", "BGD": "BGD", "NGA": "NGA"}
    fc = firms_country.get(country, country)

    for year in years:
        out_file = out_dir / f"viirs_{country}_{year}.csv"
        if out_file.exists():
            log.info(f"FIRMS {year}: already exists")
            continue

        # FIRMS bulk country download (annual)
        url = f"https://firms.modaps.eosdis.nasa.gov/api/country/csv/{api_key}/VIIRS_SNPP_NRT/{fc}/365"
        log.info(f"FIRMS: downloading {year}...")
        r = requests.get(url, timeout=120)
        if r.status_code == 200:
            out_file.write_text(r.text)
            log.info(f"FIRMS: saved {out_file}")
        else:
            log.warning(f"FIRMS {year}: HTTP {r.status_code}")


def download_gpm(country: str, years: list[int], dry_run: bool):
    """
    NASA GPM IMERG precipitation data.
    Product: GPM_3IMERGDF v07 (daily, 0.1 degree)
    Docs: https://disc.gsfc.nasa.gov/datasets/GPM_3IMERGDF_07/summary
    Auth: NASA EarthData URS (https://urs.earthdata.nasa.gov/)
    """
    import requests
    from requests.auth import HTTPBasicAuth

    username = os.getenv("EARTHDATA_USERNAME", "")
    password = os.getenv("EARTHDATA_PASSWORD", "")

    if dry_run:
        log.info(f"[DRY RUN] GPM: Would download daily precipitation for {country} {years}")
        return

    if not username or not password:
        log.error("Set EARTHDATA_USERNAME and EARTHDATA_PASSWORD env vars.")
        log.error("Register free at https://urs.earthdata.nasa.gov/")
        return

    bbox = COUNTRY_BBOX[country]
    out_dir = RAW_DIR / "gpm" / country
    out_dir.mkdir(parents=True, exist_ok=True)

    # GPM IMERG daily data is accessed via OPeNDAP or direct HDF5 download
    # Using the GES DISC HTTPS server
    base_url = "https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/GPM_3IMERGDF.07"
    auth = HTTPBasicAuth(username, password)

    log.info(f"GPM: downloading for {country} — this may take a while...")
    log.info("GPM: See full download guide at docs/datasets/DATASETS.md")
    log.info(f"GPM: Base URL: {base_url}")
    # Full implementation uses earthaccess library:
    # pip install earthaccess
    # earthaccess.login(); earthaccess.search_data(...)


def download_flood_database(country: str, years: list[int], dry_run: bool):
    """
    Dartmouth Flood Observatory — Global Active Archive of Large Flood Events.
    URL: https://floodobservatory.colorado.edu/Archives/index.html
    Format: Excel / CSV
    """
    import requests

    out_dir = RAW_DIR / "floods"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "dfo_flood_archive.xlsx"

    if dry_run:
        log.info("[DRY RUN] DFO Floods: Would download global flood archive")
        return

    if out_file.exists():
        log.info("DFO Floods: already exists")
        return

    url = "https://floodobservatory.colorado.edu/Archives/MasterListrev.xlsx"
    log.info("DFO Floods: downloading global flood archive...")
    r = requests.get(url, timeout=60)
    if r.status_code == 200:
        out_file.write_bytes(r.content)
        log.info(f"DFO Floods: saved to {out_file}")
    else:
        log.warning(f"DFO Floods: HTTP {r.status_code}. Download manually from:")
        log.warning("https://floodobservatory.colorado.edu/Archives/index.html")


def download_admin_boundaries(country: str, dry_run: bool):
    """
    OCHA COD Administrative Boundaries via Humanitarian Data Exchange.
    Pakistan: https://data.humdata.org/dataset/cod-ab-pak
    Bangladesh: https://data.humdata.org/dataset/cod-ab-bgd
    Nigeria: https://data.humdata.org/dataset/cod-ab-nga
    """
    import requests

    hdx_urls = {
        "PAK": "https://data.humdata.org/dataset/cod-ab-pak",
        "BGD": "https://data.humdata.org/dataset/cod-ab-bgd",
        "NGA": "https://data.humdata.org/dataset/cod-ab-nga",
    }

    out_dir = RAW_DIR / "boundaries" / country
    out_dir.mkdir(parents=True, exist_ok=True)

    if dry_run:
        log.info(f"[DRY RUN] Boundaries: Would download from {hdx_urls.get(country)}")
        return

    log.info(f"Admin Boundaries: download from HDX: {hdx_urls.get(country, 'unknown')}")
    log.info("Note: HDX requires browsing to download GeoJSON. See DATASETS.md for instructions.")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    years = year_range(args.years)
    country = args.country
    dry = args.dry_run

    log.info(f"=== ClimateShield AI Dataset Downloader ===")
    log.info(f"Country: {country} | Years: {years[0]}–{years[-1]} | Dry run: {dry}")

    dispatch = {
        "era5":       lambda: download_era5(country, years, dry),
        "openaq":     lambda: download_openaq(country, years, dry),
        "firms":      lambda: download_firms(country, years, dry),
        "gpm":        lambda: download_gpm(country, years, dry),
        "flood":      lambda: download_flood_database(country, years, dry),
        "boundaries": lambda: download_admin_boundaries(country, dry),
    }

    for ds in args.datasets:
        if ds in dispatch:
            log.info(f"--- Downloading: {ds} ---")
            try:
                dispatch[ds]()
            except Exception as e:
                log.error(f"{ds} failed: {e}")
        else:
            log.warning(f"Unknown dataset: {ds}")

    log.info("=== Download complete ===")
    log.info(f"Raw data saved to: {RAW_DIR}")
    log.info("Next step: python data/scripts/preprocess.py")


if __name__ == "__main__":
    main()
