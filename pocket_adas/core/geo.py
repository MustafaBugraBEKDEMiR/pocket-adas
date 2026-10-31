"""Local metric <-> WGS84 conversion (equirectangular, fine for a few km)."""
import math

# Default origin: Hacettepe University Beytepe Campus, Ankara
ORIGIN = (39.8680, 32.7350)
_M_PER_DEG = 111_320.0


def xy_to_latlon(x: float, y: float, origin=ORIGIN):
    lat0, lon0 = origin
    lat = lat0 + y / _M_PER_DEG
    lon = lon0 + x / (_M_PER_DEG * math.cos(math.radians(lat0)))
    return lat, lon


def latlon_to_xy(lat: float, lon: float, origin=ORIGIN):
    lat0, lon0 = origin
    y = (lat - lat0) * _M_PER_DEG
    x = (lon - lon0) * _M_PER_DEG * math.cos(math.radians(lat0))
    return x, y
