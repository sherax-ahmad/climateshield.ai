# Deployment Guide

ClimateShield AI can be deployed in three configurations depending on available infrastructure.

---

## Option 1: Cloud Deployment (Recommended for National / Provincial Use)

### Requirements
- Linux server (Ubuntu 22.04+)
- Docker + Docker Compose
- 4 vCPU, 8 GB RAM minimum
- 50 GB disk (for data storage)

### Steps

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_ORG/climateshield-ai.git
cd climateshield-ai

# 2. Configure environment
cp .env.example .env
# Edit .env: add API keys (OpenAQ, FIRMS, NASA EarthData)

# 3. Start all services
docker-compose up -d

# 4. Download initial datasets (runs once)
docker-compose exec api python data/scripts/download_all.py --country PAK --years 2018-2024

# 5. Train models
docker-compose exec api python models/disease_prediction/dengue_model.py \
    data/processed/features_PAK_2018_2024.parquet

# 6. Access
# Dashboard:  http://your-server:3000
# API docs:   http://your-server:8000/docs
# API health: http://your-server:8000/v1/health
```

---

## Option 2: Low-Resource Edge Deployment (Raspberry Pi 4)

For rural health posts with no reliable cloud connectivity.

### Requirements
- Raspberry Pi 4 (4GB RAM)
- 64GB microSD or USB SSD
- Raspberry Pi OS 64-bit
- Occasional internet for data sync (weekly is enough)

### Steps

```bash
# Install dependencies
sudo apt-get update && sudo apt-get install -y python3-pip docker.io

# Clone and configure
git clone https://github.com/YOUR_ORG/climateshield-ai.git
cd climateshield-ai
cp .env.example .env

# Use lightweight SQLite config (no PostgreSQL needed)
cp docker-compose.edge.yml docker-compose.override.yml

# Start (uses ~1.5GB RAM)
docker-compose up -d

# Pre-load district data
python data/scripts/download_all.py --country PAK --districts Lahore,Faisalabad

# Dashboard available on local network: http://192.168.1.X:3000
```

---

## Option 3: Android APK (Community Health Workers)

The Flutter mobile app works fully offline.

### Build from source

```bash
cd mobile/
flutter pub get
flutter build apk --release
# Output: build/app/outputs/flutter-apk/app-release.apk
```

### Pre-configure for a district

```bash
# Bake in district data for offline use
flutter build apk --dart-define=DEFAULT_DISTRICT=PK-PB-LAH \
    --dart-define=API_URL=http://your-server:8000
```

---

## Environment Variables

Copy `.env.example` to `.env` and fill in:

```bash
# Database
DB_PASSWORD=your_secure_password

# API Keys (all free to obtain)
OPENAQ_API_KEY=        # https://openaq.org/developers/api-keys/
FIRMS_API_KEY=         # https://firms.modaps.eosdis.nasa.gov/api/area/
EARTHDATA_USERNAME=    # https://urs.earthdata.nasa.gov/
EARTHDATA_PASSWORD=

# CDS API for ERA5
# Create ~/.cdsapirc file: https://cds.climate.copernicus.eu/user/login

# Optional: push notifications
FCM_SERVER_KEY=        # Firebase Cloud Messaging for mobile alerts

# Security
ALLOWED_ORIGINS=http://localhost:3000,http://your-domain.com
SECRET_KEY=            # generate: python -c "import secrets; print(secrets.token_hex(32))"
```

---

## Data Refresh

By default, the scheduler pulls new data every 6 hours. To change:

```bash
# In .env:
REFRESH_INTERVAL_HOURS=6
```

Or trigger manually:

```bash
docker-compose exec scheduler python -m api.scheduler --run-now
```

---

## Monitoring

Basic health check:
```bash
curl http://localhost:8000/v1/health
```

Full system status (all data sources, model freshness):
```bash
curl http://localhost:8000/v1/health/detailed
```

---

## Updating

```bash
git pull
docker-compose build
docker-compose up -d
```

Model retraining (recommended monthly):
```bash
docker-compose exec api python models/disease_prediction/dengue_model.py \
    data/processed/features_PAK_latest.parquet
```
