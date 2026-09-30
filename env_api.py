"""
Live environment capture: a place name goes in, the three environment channels
the model needs come out.

    C_PM   PM2.5, ug/m3          measured, straight from the feed
    C_O3   surface ozone, ug/m3  measured, straight from the feed
    I_UV   UV dose, normalised   measured UV index / UVI_REF, and UVI_REF is ours

Source is Open-Meteo (air quality + geocoding). Free, no key, no account, CORS
open, which is why it survives a Streamlit Community Cloud deploy where a keyed
feed would need secrets. Data behind it is CAMS for Europe and the CAMS global
model elsewhere, so a city reading is a model reanalysis at the nearest grid
cell, not a kerbside monitor. Treat it as the exposure a person in that city
plausibly saw, not as an instrument reading.

No Streamlit import here on purpose: the module is usable from the CLI and from
a notebook.

    python env_api.py Delhi
    python env_api.py "Sao Paulo" --uv now
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import requests

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
AIRQ_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
TIMEOUT_S = 20

# The one constant this file invents. Evidence class C, same convention as the
# model: nobody measured a UV dose next to a skin endpoint we could reuse, so
# I_UV is a ratio and this is its denominator. 8.0 is the bottom of the WHO
# "very high" band, i.e. a clear summer noon at low latitude, which is the same
# kind of day C_PM_ref = 80 and C_O3_ref = 60 describe. Move it and every I_UV
# moves with it; nothing else in the model depends on it.
UVI_REF = 8.0
UVI_REF_EVIDENCE = "C"

# what the feed calls things -> what we call them
_CURRENT_VARS = (
    "pm2_5,pm10,nitrogen_dioxide,sulphur_dioxide,carbon_monoxide,ozone,"
    "uv_index,european_aqi,us_aqi"
)


class EnvLookupError(RuntimeError):
    """Geocoding or the air-quality feed failed, or returned nothing usable."""


@dataclass(frozen=True)
class Place:
    name: str
    lat: float
    lon: float
    country: str = ""
    admin1: str = ""
    timezone: str = ""

    @property
    def label(self) -> str:
        bits = [self.name, self.admin1, self.country]
        seen, out = set(), []
        for b in bits:
            if b and b not in seen:
                seen.add(b)
                out.append(b)
        return ", ".join(out)


@dataclass(frozen=True)
class LiveEnvironment:
    """One observation, in the feed's own units. Nothing here is model input
    yet; model_inputs() does that conversion and is the only place that knows
    about UVI_REF."""

    place: Place
    observed_at: str
    pm25: float
    o3: float
    uv_now: float
    uv_peak_today: float
    pm10: float = 0.0
    no2: float = 0.0
    so2: float = 0.0
    co: float = 0.0
    european_aqi: Optional[float] = None
    us_aqi: Optional[float] = None
    extras: Dict[str, float] = field(default_factory=dict)

    def uv_index(self, mode: str = "peak") -> float:
        """"peak" is today's forecast maximum, "now" is the current hour.

        Peak is the default the app uses. A product applied in the morning is
        on the face for the whole day, and "now" reads 0.0 at night, which
        would silently recommend a UV screen of nothing to anyone opening the
        app after dark.
        """
        if mode not in ("peak", "now"):
            raise ValueError("mode must be 'peak' or 'now'")
        return self.uv_peak_today if mode == "peak" else self.uv_now

    def model_inputs(self, uv_mode: str = "peak", uvi_ref: float = UVI_REF) -> Dict[str, float]:
        """The three channels, in the units Environment() wants."""
        if uvi_ref <= 0:
            raise ValueError("uvi_ref must be positive")
        return {
            "C_PM": float(self.pm25),
            "C_O3": float(self.o3),
            "I_UV": float(self.uv_index(uv_mode)) / float(uvi_ref),
        }


def _get(url: str, params: dict) -> dict:
    try:
        r = requests.get(url, params=params, timeout=TIMEOUT_S)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as exc:
        raise EnvLookupError(f"{url.split('/')[2]} did not answer: {exc}") from exc
    except ValueError as exc:
        raise EnvLookupError(f"{url.split('/')[2]} returned something that is not JSON") from exc


def geocode(query: str, count: int = 5, language: str = "en") -> List[Place]:
    """Place name -> candidates, best match first. Empty list is a normal
    answer for a typo, not an error."""
    query = (query or "").strip()
    if not query:
        return []
    data = _get(GEOCODE_URL, {"name": query, "count": max(1, min(count, 10)),
                              "language": language, "format": "json"})
    out = []
    for h in data.get("results") or []:
        out.append(Place(
            name=h.get("name", query),
            lat=float(h["latitude"]),
            lon=float(h["longitude"]),
            country=h.get("country", "") or "",
            admin1=h.get("admin1", "") or "",
            timezone=h.get("timezone", "") or "",
        ))
    return out


def _num(v, default: float = 0.0) -> float:
    return default if v is None else float(v)


def fetch_live_environment(place: Place) -> LiveEnvironment:
    """Current hour plus today's UV profile for one place."""
    data = _get(AIRQ_URL, {
        "latitude": place.lat, "longitude": place.lon,
        "current": _CURRENT_VARS,
        "hourly": "uv_index",
        "forecast_days": 1,
        "timezone": "auto",
    })
    cur = data.get("current")
    if not cur:
        raise EnvLookupError("the air-quality feed answered without a current reading")

    hourly = (data.get("hourly") or {}).get("uv_index") or []
    uv_hours = [float(v) for v in hourly if v is not None]
    uv_now = _num(cur.get("uv_index"))
    uv_peak = max(uv_hours) if uv_hours else uv_now

    return LiveEnvironment(
        place=place,
        observed_at=str(cur.get("time", "")),
        pm25=_num(cur.get("pm2_5")),
        o3=_num(cur.get("ozone")),
        uv_now=uv_now,
        uv_peak_today=uv_peak,
        pm10=_num(cur.get("pm10")),
        no2=_num(cur.get("nitrogen_dioxide")),
        so2=_num(cur.get("sulphur_dioxide")),
        co=_num(cur.get("carbon_monoxide")),
        european_aqi=None if cur.get("european_aqi") is None else float(cur["european_aqi"]),
        us_aqi=None if cur.get("us_aqi") is None else float(cur["us_aqi"]),
    )


def fetch_hourly_profile(place: Place, days: int = 1) -> Dict[str, list]:
    """Hour-by-hour PM2.5, ozone and UV index for one place, today onward.

    Same feed and units as fetch_live_environment, so an hour of this profile
    can go straight into the model. Missing hours come back as None and are
    left for the caller to drop.
    """
    data = _get(AIRQ_URL, {
        "latitude": place.lat, "longitude": place.lon,
        "hourly": "pm2_5,ozone,uv_index",
        "forecast_days": max(1, min(days, 5)),
        "timezone": "auto",
    })
    h = data.get("hourly") or {}
    times = h.get("time") or []
    if not times:
        raise EnvLookupError("the air-quality feed answered without an hourly profile")
    return {
        "time": list(times),
        "pm25": list(h.get("pm2_5") or [None] * len(times)),
        "o3": list(h.get("ozone") or [None] * len(times)),
        "uv_index": list(h.get("uv_index") or [None] * len(times)),
    }


def lookup(query: str) -> LiveEnvironment:
    """geocode + fetch in one call, taking the first candidate."""
    hits = geocode(query, count=1)
    if not hits:
        raise EnvLookupError(f"no place matched {query!r}")
    return fetch_live_environment(hits[0])


# ---------------------------------------------------------------------------
# out-of-domain warnings. The model is a set of fitted curves with a stated
# validity range; a live feed will happily hand it a day outside that range.
# ---------------------------------------------------------------------------

def domain_warnings(inputs: Dict[str, float]) -> List[str]:
    msgs = []
    if inputs["C_PM"] > 300:
        msgs.append(
            f"PM2.5 is {inputs['C_PM']:.0f} ug/m3. The TEWL slope alpha_PM was fitted "
            "on a far narrower range, so the barrier channel is extrapolating."
        )
    if inputs["C_O3"] > 200:
        msgs.append(
            f"Ozone is {inputs['C_O3']:.0f} ug/m3, past anything the ozone-to-TEWL "
            "ratio was matched against."
        )
    if inputs["I_UV"] > 2.0:
        msgs.append(
            f"I_UV is {inputs['I_UV']:.2f}, i.e. more than twice the reference day. "
            "The UV channel is linear by assumption and was never tested there."
        )
    if inputs["I_UV"] < 0.05:
        msgs.append(
            "UV is essentially zero for this place today. The UV screen is then "
            "protecting against nothing, and pulcherrimin is recommended only for "
            "its chelation."
        )
    return msgs


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="live environment for one place")
    ap.add_argument("place", nargs="+")
    ap.add_argument("--uv", choices=["peak", "now"], default="peak")
    ap.add_argument("--uvi-ref", type=float, default=UVI_REF)
    a = ap.parse_args()

    live = lookup(" ".join(a.place))
    mi = live.model_inputs(a.uv, a.uvi_ref)
    print(f"{live.place.label}  ({live.place.lat:.3f}, {live.place.lon:.3f})  {live.observed_at}")
    print(f"  PM2.5 {live.pm25:6.1f} ug/m3   PM10 {live.pm10:6.1f}   NO2 {live.no2:6.1f}   "
          f"O3 {live.o3:6.1f}   SO2 {live.so2:6.1f}")
    print(f"  UV index now {live.uv_now:.1f}, peak today {live.uv_peak_today:.1f}"
          f"   EAQI {live.european_aqi}   US AQI {live.us_aqi}")
    print(f"  -> C_PM {mi['C_PM']:.1f}   C_O3 {mi['C_O3']:.1f}   "
          f"I_UV {mi['I_UV']:.3f}  (UV {a.uv} / {a.uvi_ref:g})")
    for w in domain_warnings(mi):
        print(f"  ! {w}")
