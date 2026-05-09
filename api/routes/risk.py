"""
ClimateShield AI — Risk Score API Routes

GET  /v1/risk/district/{district_code}     — Current risk for a district
GET  /v1/risk/forecast/{district_code}     — 7-day forecast
POST /v1/risk/point                        — Risk for arbitrary lat/lon
GET  /v1/risk/map                          — GeoJSON risk map for dashboard
"""

from datetime import date, datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

router = APIRouter()


# ─── Request / Response schemas ───────────────────────────────────────────────

class PointRiskRequest(BaseModel):
    latitude: float = Field(..., ge=-90, le=90, example=31.558)
    longitude: float = Field(..., ge=-180, le=180, example=74.351)
    date: Optional[date] = Field(None, description="Defaults to today")
    include_forecast_days: int = Field(0, ge=0, le=7)


class RiskScore(BaseModel):
    label: str         = Field(example="High")
    score: float       = Field(ge=0, le=100, example=72.4)
    color: str         = Field(example="#f97316")
    child_alert: bool  = Field(example=True)
    description: str   = Field(example="High dengue risk. Clinics should pre-stock treatments.")


class DistrictRiskResponse(BaseModel):
    district_code: str
    district_name: str
    country: str
    timestamp: datetime
    dengue:      RiskScore
    heat:        RiskScore
    respiratory: RiskScore
    flood_disease: Optional[RiskScore]
    composite_child_risk: float = Field(ge=0, le=100, description="Weighted composite for children under 12")
    alert_message_en: str
    alert_message_ur: str
    data_freshness_hours: float = Field(description="How old is the underlying climate data")


class ForecastDay(BaseModel):
    date: date
    dengue:      RiskScore
    heat:        RiskScore
    respiratory: RiskScore
    composite_child_risk: float


class ForecastResponse(BaseModel):
    district_code: str
    generated_at: datetime
    forecast_days: list[ForecastDay]


# ─── Routes ──────────────────────────────────────────────────────────────────

@router.get(
    "/district/{district_code}",
    response_model=DistrictRiskResponse,
    summary="Get current risk scores for a district",
    description=(
        "Returns real-time AI-predicted risk scores for dengue, heat, "
        "and respiratory illness for children in the specified district. "
        "District codes follow the Pakistan COD-AB system (e.g. PK-PB-Lahore)."
    ),
)
async def get_district_risk(
    district_code: str,
    request: Request,
):
    models = request.app.state.models

    try:
        risk_data = await models.get_district_risk(district_code)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"District '{district_code}' not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model inference failed: {str(e)}")

    return risk_data


@router.get(
    "/forecast/{district_code}",
    response_model=ForecastResponse,
    summary="Get 7-day child health risk forecast",
)
async def get_district_forecast(
    district_code: str,
    days: int = Query(default=7, ge=1, le=7),
    request: Request = None,
):
    models = request.app.state.models

    try:
        forecast = await models.get_district_forecast(district_code, days=days)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"District '{district_code}' not found.")

    return forecast


@router.post(
    "/point",
    response_model=DistrictRiskResponse,
    summary="Get risk score for any lat/lon point",
    description="Useful for mobile app — pass device GPS location.",
)
async def get_point_risk(
    payload: PointRiskRequest,
    request: Request,
):
    models = request.app.state.models

    try:
        risk_data = await models.get_point_risk(
            lat=payload.latitude,
            lon=payload.longitude,
            target_date=payload.date,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return risk_data


@router.get(
    "/map",
    summary="GeoJSON risk map for all districts",
    description=(
        "Returns a GeoJSON FeatureCollection with current risk scores "
        "for all districts. Used by the Leaflet.js dashboard. "
        "Cached and updated every 6 hours."
    ),
)
async def get_risk_map(
    country: str = Query(default="PAK", description="ISO3 country code"),
    risk_type: str = Query(
        default="composite",
        enum=["composite", "dengue", "heat", "respiratory"],
        description="Which risk to show",
    ),
    request: Request = None,
):
    models = request.app.state.models

    geojson = await models.get_risk_map_geojson(country=country, risk_type=risk_type)
    return geojson
