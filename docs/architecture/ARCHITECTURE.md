# ClimateShield AI — System Architecture

## Overview

ClimateShield AI is designed as a **modular, offline-capable, open-source** platform. Each component can be deployed independently, making it suitable for everything from national health ministry deployments to single Raspberry Pi installations in rural clinics.

---

## Component Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                         EXTERNAL DATA SOURCES                           │
│                                                                         │
│  ERA5 (ECMWF)    NASA FIRMS    OpenAQ     NASA GPM    WHO DON          │
│  Climate/Temp    Heat Anomaly  Air Quality Rainfall    Outbreak News    │
└──────────────────────────────┬─────────────────────────────────────────┘
                               │  Pull every 6 hours
┌──────────────────────────────▼─────────────────────────────────────────┐
│                      DATA INGESTION SERVICE                              │
│  Python + APScheduler   |   Error handling + retry   |   Raw → S3/disk │
└──────────────────────────────┬─────────────────────────────────────────┘
                               │
┌──────────────────────────────▼─────────────────────────────────────────┐
│                     FEATURE ENGINEERING PIPELINE                         │
│  Spatial interpolation  |  Lag/rolling features  |  WBGT calculation   │
│  Urban heat island adj  |  Child vulnerability weights (UNICEF MICS)   │
└──────────────────────────────┬─────────────────────────────────────────┘
                               │
┌──────────────────────────────▼─────────────────────────────────────────┐
│                      AI FORECASTING ENGINE                               │
│                                                                         │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐              │
│  │  Dengue Risk  │  │   Heat Risk   │  │ Respiratory   │              │
│  │  XGBoost      │  │  Threshold    │  │ Risk (LSTM)   │              │
│  │  7-day ahead  │  │  WBGT-based   │  │  AQI → asthma │              │
│  └───────┬───────┘  └───────┬───────┘  └───────┬───────┘              │
│          └──────────────────┴──────────────────┘                        │
│                               │                                         │
│                    Composite Child Risk Score                            │
└──────────────────────────────┬─────────────────────────────────────────┘
                               │
          ┌────────────────────┼────────────────────┐
          │                    │                    │
┌─────────▼──────┐   ┌─────────▼──────┐   ┌────────▼───────┐
│  FastAPI REST  │   │  React Dashboard│   │  Flutter App   │
│  + WebSocket   │   │  Leaflet.js maps│   │  Offline-first │
│                │   │                 │   │  Urdu/English  │
│  /v1/risk/*    │   │  localhost:3000  │   │  Android APK   │
│  /v1/alerts/*  │   │  Risk heatmaps  │   │  SQLite sync   │
└────────────────┘   └─────────────────┘   └────────────────┘
          │
┌─────────▼───────────────────────────────────────────────────────┐
│                      PostgreSQL + PostGIS                         │
│  District boundaries  |  Historical risk scores  |  Alert logs   │
└─────────────────────────────────────────────────────────────────┘
```

---

## AI Models

### 1. Dengue Risk Model (XGBoost)
- **Algorithm:** Gradient Boosted Trees (XGBoost multi-class)
- **Input:** 40+ climate features with 1–14 day lags
- **Output:** 4-class risk label (Low / Medium / High / Emergency)
- **Forecast horizon:** 7 days ahead
- **Calibration:** Child-specific (under-12 vulnerability weighting)
- **Explainability:** SHAP values — shows health workers *why* risk is high

### 2. Heat Risk Calculator
- **Algorithm:** Physics-based (WBGT formula) + vulnerability weighting
- **Input:** Temperature, humidity, solar radiation, urban flag
- **Thresholds:** AAP pediatric guidelines (not adult thresholds)
- **Key insight:** Children's heat tolerance is ~3°C lower than adults

### 3. Respiratory Risk Model (LSTM)
- **Algorithm:** LSTM (sequence model for AQI time series)
- **Input:** PM2.5, PM10, NO2, O3 from OpenAQ — rolling 14-day window
- **Output:** Asthma/respiratory risk score for children
- **Clinically grounded:** WHO AQI → child respiratory harm mapping

---

## Data Flow

```
6-hourly schedule:
  1. Pull ERA5, OpenAQ, FIRMS, GPM
  2. Run feature engineering pipeline
  3. Run AI model inference (all districts)
  4. Store risk scores in PostgreSQL
  5. Publish alerts via Redis pub/sub → WebSocket → mobile app
  6. Update GeoJSON map cache
```

---

## Offline Architecture (Mobile)

The Flutter app is designed for **zero-connectivity environments**:

```
Mobile App (Flutter)
  └─ SQLite (local)
       ├─ Last downloaded risk scores (per district)
       ├─ Symptom log (health worker entries)
       ├─ Pending sync queue
       └─ Offline AI model (TensorFlow Lite)
            └─ Basic heat risk — runs on device without internet
```

Sync happens whenever connectivity is available (WiFi, 2G, 3G).

---

## Deployment Options

| Scenario | Stack | Min Spec |
|---------|-------|----------|
| National health ministry | Cloud (Docker + PostgreSQL) | 4 vCPU, 8GB RAM |
| Provincial health office | VPS + Docker | 2 vCPU, 4GB RAM |
| District hospital | Local server | 2 vCPU, 4GB RAM |
| Rural clinic (offline) | Raspberry Pi 4 + SQLite | 4GB RAM Pi |
| Community health worker | Android phone | Android 8+ |

---

## Security & Privacy

- No personally identifiable information (PII) collected
- All risk scores are **aggregate** (district/tehsil level), never individual
- HTTPS enforced; API keys rotated monthly
- Open-source: full audit trail on GitHub
- GDPR / Pakistan PDPA compliant

---

## Open Source Principles

Per UNICEF Venture Fund requirements:

- **MIT License** — maximum permissibility for health system adoption
- **Public GitHub** — all code, documentation, and model weights
- **Reproducible** — one command to download data and retrain models
- **Documented APIs** — Swagger/OpenAPI spec at `/docs`
- **No vendor lock-in** — runs on any Linux server or cloud provider
