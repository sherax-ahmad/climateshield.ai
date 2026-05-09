"""
ClimateShield AI — FastAPI Backend

REST API + WebSocket server for:
  - Real-time risk scores (heat, dengue, AQI-respiratory)
  - Hyperlocal district risk maps
  - Alert management
  - Offline sync for mobile app

Base URL: https://api.climateshield.ai/v1
Docs:     https://api.climateshield.ai/docs  (auto-generated Swagger)
"""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.routes import health, risk, alerts, districts, sync
from api.utils.model_registry import ModelRegistry

log = logging.getLogger(__name__)

# ─── App lifespan (load models on startup) ───────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load AI models into memory on startup."""
    log.info("ClimateShield AI: loading models...")
    registry = ModelRegistry()
    await registry.load_all()
    app.state.models = registry
    log.info("Models ready.")
    yield
    log.info("ClimateShield AI: shutting down.")


# ─── App ─────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="ClimateShield AI API",
    description=(
        "Open-source AI Early Warning System for Climate-Related Child Health Risks. "
        "Predicts dengue, heatstroke, and respiratory illness risks for children "
        "in vulnerable communities. Built for UNICEF Venture Fund 2026."
    ),
    version="0.1.0",
    license_info={"name": "MIT", "url": "https://opensource.org/licenses/MIT"},
    contact={
        "name": "ClimateShield AI",
        "url": "https://github.com/YOUR_ORG/climateshield-ai",
    },
    lifespan=lifespan,
)

# CORS — allow mobile app and dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Routers ─────────────────────────────────────────────────────────────────

app.include_router(health.router,    prefix="/v1/health",     tags=["Health"])
app.include_router(risk.router,      prefix="/v1/risk",       tags=["Risk Scores"])
app.include_router(alerts.router,    prefix="/v1/alerts",     tags=["Alerts"])
app.include_router(districts.router, prefix="/v1/districts",  tags=["Districts"])
app.include_router(sync.router,      prefix="/v1/sync",       tags=["Offline Sync"])


# ─── Root ─────────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root():
    return {
        "name": "ClimateShield AI",
        "version": "0.1.0",
        "description": "Climate-Health Early Warning for Children",
        "docs": "/docs",
        "github": "https://github.com/YOUR_ORG/climateshield-ai",
        "license": "MIT",
    }


# ─── WebSocket for real-time alerts ──────────────────────────────────────────

@app.websocket("/v1/ws/alerts/{district_code}")
async def alerts_websocket(websocket: WebSocket, district_code: str):
    """
    Real-time alert stream for a district.
    Mobile app connects here for live push notifications.

    Message format:
    {
      "type": "risk_update",
      "district": "PKR-LAH-001",
      "timestamp": "2024-07-15T12:00:00Z",
      "risks": {
        "dengue": {"label": "High", "score": 72},
        "heat":   {"label": "Danger", "score": 85},
        "aqi":    {"label": "Medium", "score": 45}
      },
      "child_alert": true,
      "message_ur": "لاہور میں ڈینگی کا خطرہ زیادہ ہے",
      "message_en": "High dengue risk in Lahore"
    }
    """
    await websocket.accept()
    log.info(f"WebSocket connected: district={district_code}")
    try:
        while True:
            data = await websocket.receive_text()
            # Echo back with acknowledgement (full implementation uses Redis pub/sub)
            await websocket.send_json({"status": "connected", "district": district_code})
    except Exception as e:
        log.info(f"WebSocket closed: {e}")
