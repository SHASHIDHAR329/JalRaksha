"""JalRaksha weather helper.

Provides a stable get_current_weather(latitude, longitude) function for the
Streamlit dashboard. Weather is contextual only; it is not fed into the
current FNO model.
"""
from __future__ import annotations

from typing import Any

import requests

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"


def get_current_weather(latitude: float, longitude: float) -> dict[str, Any]:
    """Return current weather from Open-Meteo for a coordinate pair."""
    lat = float(latitude)
    lon = float(longitude)
    if not -90.0 <= lat <= 90.0:
        raise ValueError("Latitude must be between -90 and 90 degrees.")
    if not -180.0 <= lon <= 180.0:
        raise ValueError("Longitude must be between -180 and 180 degrees.")

    params = {
        "latitude": lat,
        "longitude": lon,
        "current": (
            "temperature_2m,relative_humidity_2m,precipitation,"
            "wind_speed_10m,weather_code"
        ),
        "timezone": "auto",
    }
    response = requests.get(OPEN_METEO_URL, params=params, timeout=10)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict) or "current" not in payload:
        raise RuntimeError("Weather provider returned an unexpected response.")
    return payload


__all__ = ["get_current_weather"]
