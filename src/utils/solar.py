"""Solar-elevation based day/night classification for the VIS channel.

Filename timestamps are UTC. Solar elevation depends only on the absolute
instant and the observer location, so it is computed from the UTC time
directly; ``utc_to_ist`` is provided only for human-readable manifest columns.
"""
import datetime as dt

from astral import Observer
from astral.sun import elevation as _elevation

IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
UTC = dt.timezone.utc


def _as_utc(t):
    """Return a timezone-aware UTC datetime."""
    if t.tzinfo is None:
        return t.replace(tzinfo=UTC)
    return t.astimezone(UTC)


def utc_to_ist(t_utc):
    """Convert a (naive-UTC or aware) datetime to Indian Standard Time."""
    return _as_utc(t_utc).astimezone(IST)


def solar_elevation_deg(t_utc, lat, lon):
    """Solar elevation angle (degrees) above the horizon at the given instant."""
    observer = Observer(latitude=lat, longitude=lon)
    return float(_elevation(observer, _as_utc(t_utc)))


def is_daytime(t_utc, lat, lon, threshold_deg=0.0):
    """True when the sun is above ``threshold_deg`` => VIS frame is usable."""
    return solar_elevation_deg(t_utc, lat, lon) > threshold_deg
