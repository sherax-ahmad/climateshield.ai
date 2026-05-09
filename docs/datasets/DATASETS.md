# 📦 Dataset Documentation

All datasets used in ClimateShield AI are **publicly available, free, and reproducible**. This document provides direct links, access instructions, and usage notes for each data source.

---

## 1. ERA5-Land Climate Reanalysis (Primary Climate Data)

**Provider:** European Centre for Medium-Range Weather Forecasts (ECMWF) via Copernicus Climate Data Store  
**URL:** https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land  
**Variables Used:**
- `2m_temperature` — Air temperature at 2 meters (°C)
- `2m_dewpoint_temperature` — For humidity calculation
- `total_precipitation` — Daily rainfall accumulation (mm)
- `surface_pressure` — Atmospheric pressure

**Spatial Resolution:** 0.1° × 0.1° (~9 km)  
**Temporal Coverage:** 1950–present (we use 2000–2024)  
**Access:** Free account required at https://cds.climate.copernicus.eu/  
**Download Script:** `data/scripts/download_era5.py`

```python
# Example API call
import cdsapi
c = cdsapi.Client()
c.retrieve('reanalysis-era5-land', {
    'variable': ['2m_temperature', 'total_precipitation'],
    'year': '2023', 'month': '07', 'day': list(range(1,32)),
    'time': [f'{h:02d}:00' for h in range(24)],
    'area': [37, 60, 23, 77],  # Pakistan bounding box
    'format': 'netcdf'
}, 'era5_pakistan_2023_07.nc')
```

---

## 2. NASA FIRMS — Fire Information for Resource Management System

**Provider:** NASA LANCE / FIRMS  
**URL:** https://firms.modaps.eosdis.nasa.gov/  
**API Docs:** https://firms.modaps.eosdis.nasa.gov/api/  
**Variables Used:**
- `VIIRS_I-Band_375m_Active_Fire` — High-resolution heat anomalies
- `MODIS_NRT` — Near real-time land surface temperature anomalies

**Access:** Free. API key available at https://firms.modaps.eosdis.nasa.gov/api/area/  
**Download Script:** `data/scripts/download_firms.py`

```bash
# Direct HTTP access (no auth for bulk historical)
curl "https://firms.modaps.eosdis.nasa.gov/api/country/csv/YOUR_MAP_KEY/VIIRS_SNPP_NRT/PAK/7"
```

---

## 3. OpenAQ — Global Air Quality Data

**Provider:** OpenAQ  
**URL:** https://openaq.org/  
**API Docs:** https://docs.openaq.org/  
**Variables Used:**
- `pm25` — Fine particulate matter (μg/m³) — primary asthma trigger
- `pm10` — Coarse particulate
- `no2` — Nitrogen dioxide (combustion indicator)
- `o3` — Ozone

**Pakistan Stations:** Lahore, Karachi, Islamabad, Faisalabad  
**Access:** Free REST API — no key required for basic use  
**Download Script:** `data/scripts/download_openaq.py`

```python
import requests
resp = requests.get(
    "https://api.openaq.org/v3/locations",
    params={"country_id": "PK", "limit": 100}
)
```

---

## 4. NASA MODIS Land Surface Temperature (LST)

**Provider:** NASA EarthData / LP DAAC  
**URL:** https://earthdata.nasa.gov/  
**Product:** MOD11A1 (Terra, daily, 1km resolution)  
**Direct Link:** https://lpdaac.usgs.gov/products/mod11a1v061/  
**Variables Used:**
- `LST_Day_1km` — Daytime land surface temperature
- `LST_Night_1km` — Nighttime temperature

**Access:** Free EarthData account: https://urs.earthdata.nasa.gov/  
**Download Script:** `data/scripts/download_modis_lst.py`

---

## 5. NASA GPM — Global Precipitation Measurement

**Provider:** NASA Goddard Space Flight Center  
**URL:** https://gpm.nasa.gov/data  
**Product:** GPM IMERG Final (30-min, 0.1° resolution)  
**Direct Link:** https://disc.gsfc.nasa.gov/datasets/GPM_3IMERGDF_07/summary  
**Variables Used:**
- `precipitationCal` — Calibrated precipitation rate (mm/hr)

Used for: mosquito breeding site estimation (stagnant water after rain events)

**Access:** Free EarthData account required  
**Download Script:** `data/scripts/download_gpm.py`

---

## 6. Pakistan Dengue Surveillance Data

**Primary Source:** Pakistan National Institute of Health (NIH)  
**URL:** https://www.nih.org.pk/  
**WHO EMRO Pakistan:** https://www.emro.who.int/pak/pakistan-infocus/dengue.html  
**HealthMap (supplementary):** https://healthmap.org/en/  
**IDSP / DHIS2:** Province-level weekly case counts

**Variables:**
- Weekly confirmed dengue case counts by district
- Hospitalization and fatality counts
- Child (under-15) stratification where available

**Note:** Raw district-level data may require formal data-sharing agreements with Punjab Health Department. WHO EMRO aggregate data is publicly accessible.

---

## 7. Global Active Archive of Large Flood Events (Dartmouth Flood Observatory)

**Provider:** Dartmouth Flood Observatory, University of Colorado  
**URL:** https://floodobservatory.colorado.edu/  
**Direct Data:** https://floodobservatory.colorado.edu/Archives/index.html  
**Variables Used:**
- Flood event dates, durations, locations
- Affected area (km²)
- Displacement counts

Used for: post-flood waterborne disease risk modeling (cholera, typhoid)

---

## 8. Pakistan Administrative Boundaries (COD-AB)

**Provider:** OCHA Humanitarian Data Exchange (HDX)  
**URL:** https://data.humdata.org/dataset/cod-ab-pak  
**Format:** GeoJSON / Shapefile  
**Levels:** Province, District, Tehsil

Used for: spatial mapping, school/clinic geocoding, risk map rendering

---

## 9. WHO Disease Outbreak News (DON)

**Provider:** World Health Organization  
**URL:** https://www.who.int/emergencies/disease-outbreak-news  
**API:** https://www.who.int/rss-feeds/news-releases-emergency-operations.xml (RSS)

Used for: supplementary outbreak event labels for model training

---

## 10. UNICEF MICS — Multiple Indicator Cluster Surveys

**Provider:** UNICEF  
**URL:** https://mics.unicef.org/  
**Pakistan MICS 2019-20:** https://mics.unicef.org/surveys  
**Variables Used:**
- Child nutrition status (stunting, wasting)
- Access to safe drinking water
- Healthcare access indices

Used for: **child vulnerability weighting** — higher-vulnerability children receive elevated risk alerts

---

## 11. Pakistan School Locations

**Provider:** OSMPH / OpenStreetMap  
**URL:** https://www.openstreetmap.org/  
**HDX Schools:** https://data.humdata.org/dataset/hotosm_pak_education_facilities  
**Columns:** name, latitude, longitude, district, enrollment count

Used for: proximity-weighted risk scores (children in schools near high-risk zones get priority alerts)

---

## 12. NOAA Global Surface Summary of Day (GSOD)

**Provider:** NOAA National Centers for Environmental Information  
**URL:** https://www.ncei.noaa.gov/access/metadata/landing-page/bin/iso?id=gov.noaa.ncdc:C00516  
**Variables:** Temperature, dew point, wind speed, precipitation (station-based)

Used as: ground-truth validation for ERA5 satellite-derived estimates

---

## Data Storage & Reproducibility

All raw data is **not committed to Git** (too large). Instead:

```bash
# Download everything needed for Pakistan pilot
python data/scripts/download_all.py --country PAK --years 2018-2024
```

Processed features ready for model training:
```bash
python data/scripts/preprocess.py --output data/processed/
```

Estimated download size: ~12 GB raw, ~800 MB processed  
Estimated time: 45–90 minutes depending on connection

---

## License Summary

| Dataset | License |
|---------|---------|
| ERA5 | Copernicus License (free, attribution required) |
| NASA FIRMS / MODIS / GPM | NASA Open Data |
| OpenAQ | CC BY 4.0 |
| DFO Flood | Public domain |
| OCHA COD-AB | CC BY-IGO 3.0 |
| WHO DON | WHO Terms (non-commercial OK) |
| UNICEF MICS | UNICEF open data |
| OSM Schools | ODbL |
