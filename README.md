# 🌍 ClimateShield AI

**An Open-Source AI Early Warning System for Climate-Related Child Health Risks**

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Open Source](https://img.shields.io/badge/Open%20Source-Yes-brightgreen)](https://github.com/)
[![UNICEF Venture Fund](https://img.shields.io/badge/UNICEF-Venture%20Fund%202026-1cabe2)](https://www.unicef.org/innovation/call-for-application-climate-and-health-2026)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![Flutter](https://img.shields.io/badge/Mobile-Flutter-02569B)](https://flutter.dev)

> **"Protecting children before climate disasters become health disasters."**

ClimateShield AI predicts climate-driven disease risks — dengue, heatstroke, respiratory illness, waterborne disease — for children in vulnerable communities across Pakistan and emerging markets. It ingests real-time satellite, weather, and pollution data, runs AI forecasting models, and delivers hyperlocal alerts to schools, community health workers, clinics, and parents — even offline.

---

## 🎯 Problem

Over **1 billion children** live in high-risk climate zones. In Pakistan alone:
- **Lahore** regularly records AQI > 300 (hazardous for children)
- **Dengue outbreaks in Punjab** killed 300+ children in recent years
- **2022 floods** displaced 8 million children; waterborne disease surged 400%
- **Karachi heatwaves** exceed 48°C — children under 5 face lethal risk

Existing early-warning systems are:
- Built for adults, not children
- Siloed (weather OR health — not integrated)
- Inaccessible in low-resource, offline settings
- Not open-source or locally deployable

---

## 💡 Solution

ClimateShield AI is a **modular, open-source platform** with four components:

| Component | What It Does |
|-----------|-------------|
| **AI Forecasting Engine** | Predicts dengue, AQI-linked respiratory risk, heatstroke, and waterborne disease 7 days ahead |
| **Hyperlocal Risk Maps** | Schools, clinics, and communities mapped with child-specific risk scores |
| **Offline Mobile App** | Flutter app for health workers — works with no internet, syncs when connected |
| **AI Health Assistant** | Tiny on-device LLM answers health queries in Urdu/English for parents & workers |

---

## 📊 Real Datasets Used

All data sources are **free, public, and reproducible**. See [`docs/datasets/DATASETS.md`](docs/datasets/DATASETS.md) for full details.

| Dataset | Source | Used For |
|---------|--------|----------|
| ERA5 Reanalysis (temperature, humidity, rainfall) | [ECMWF Copernicus](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land) | Climate features |
| FIRMS Active Fire & Heat Anomalies | [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/) | Heatwave detection |
| OpenAQ Air Quality | [OpenAQ](https://openaq.org/) | AQI / PM2.5 / NO2 |
| Pakistan Dengue Surveillance | [Pakistan DHIS2 / WHO EMRO](https://www.emro.who.int/pak/pakistan-infocus/dengue.html) | Disease labels |
| Global Flood Database | [DFO Dartmouth](https://floodobservatory.colorado.edu/) | Flood features |
| MODIS Land Surface Temperature | [NASA EarthData](https://earthdata.nasa.gov/) | Urban heat islands |
| GPM Precipitation | [NASA GPM](https://gpm.nasa.gov/data) | Rainfall / mosquito breeding |
| Pakistan Census / Admin Boundaries | [HDX OCHA](https://data.humdata.org/dataset/cod-ab-pak) | Spatial mapping |
| WHO Disease Outbreak News | [WHO DON](https://www.who.int/emergencies/disease-outbreak-news) | Training labels |
| UNICEF MICS Child Vulnerability Index | [UNICEF MICS](https://mics.unicef.org/) | Child vulnerability weighting |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     DATA INGESTION LAYER                     │
│  NASA FIRMS │ ERA5 Climate │ OpenAQ │ GPM Rain │ WHO DON    │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                   FEATURE ENGINEERING                        │
│  Lag features │ Rolling averages │ Spatial interpolation    │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                  AI FORECASTING ENGINE                       │
│  Dengue Risk Model (XGBoost) │ AQI-Asthma (LSTM)           │
│  Heatstroke Risk (Random Forest) │ Flood Disease (GNN)     │
└─────────────────────┬───────────────────────────────────────┘
                      │
          ┌───────────┴────────────┐
          │                        │
┌─────────▼────────┐    ┌─────────▼────────────┐
│   FastAPI Backend │    │   Risk Map Dashboard  │
│   REST + WebSocket│    │   Leaflet.js + React  │
└─────────┬────────┘    └──────────────────────┘
          │
┌─────────▼────────────────────────────────────┐
│           Flutter Offline Mobile App          │
│  SQLite sync │ Urdu/English │ Push alerts     │
└──────────────────────────────────────────────┘
```

Full architecture diagram: [`docs/architecture/ARCHITECTURE.md`](docs/architecture/ARCHITECTURE.md)

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Docker & Docker Compose
- Flutter SDK (for mobile)
- Node.js 18+ (for dashboard)

### 1. Clone & Setup
```bash
git clone https://github.com/YOUR_ORG/climateshield-ai.git
cd climateshield-ai
cp .env.example .env
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download Data
```bash
python data/scripts/download_datasets.py
```

### 4. Run the API
```bash
docker-compose up
```

### 5. Access Dashboard
Open `http://localhost:3000`

Full deployment guide: [`docs/deployment/DEPLOYMENT.md`](docs/deployment/DEPLOYMENT.md)

---

## 📁 Project Structure

```
climateshield-ai/
├── data/
│   ├── scripts/           # Dataset download + preprocessing scripts
│   ├── raw/               # Raw data (gitignored, download via scripts)
│   └── processed/         # Cleaned features
├── models/
│   ├── disease_prediction/ # Dengue + waterborne forecasting
│   ├── air_quality/        # AQI → respiratory risk model
│   └── heat_risk/          # Heatstroke risk for children <12
├── api/
│   ├── routes/             # FastAPI route handlers
│   ├── schemas/            # Pydantic models
│   └── utils/              # Helpers, alerting
├── dashboard/              # React + Leaflet risk map
├── mobile/                 # Flutter offline app
├── docs/
│   ├── architecture/       # System diagrams
│   ├── datasets/           # Data source documentation
│   └── deployment/         # Docker, cloud, offline deployment
└── tests/                  # Unit + integration tests
```

---

## 🌐 Deployment Targets

| Environment | Method |
|------------|--------|
| Cloud (primary) | Docker Compose on any VPS |
| Low-resource edge | Raspberry Pi 4 + SQLite |
| Community health posts | Android APK (offline-first) |
| National health system | REST API integration |

---

## 🌍 Target Countries (Phase 1)

- 🇵🇰 **Pakistan** — Dengue (Punjab), smog (Lahore), floods (Sindh/KPK)
- 🇧🇩 **Bangladesh** — Cyclones, waterborne disease
- 🇳🇬 **Nigeria** — Malaria, extreme heat, flooding

---

## 👶 Child-Specific Design

All risk scores are calibrated for **children under 12**, accounting for:
- Higher respiratory sensitivity to PM2.5
- Lower heat tolerance thresholds
- School location proximity
- Vulnerability weighting from UNICEF MICS data

---

## 📄 License

MIT License — fully open-source. See [LICENSE](LICENSE).

---

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). We welcome contributions from developers, epidemiologists, and public health researchers.

---

## 📬 Contact

Built for the **UNICEF Venture Fund Climate & Health 2026** call.  
Questions: [open an issue](https://github.com/YOUR_ORG/climateshield-ai/issues)
