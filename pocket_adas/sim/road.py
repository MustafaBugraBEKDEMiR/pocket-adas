"""Worlds the car drives in.

Every world offers the same small interface, used by the simulation, the
autopilot and the renderer:

* ``origin`` (lat, lon) of the local x/y frame, ``bounds``
* ``speed_limit_at(x, y)``, ``offroad(x, y)``, ``height_at(x, y)``, ``features``
* ``start_pose(rng)``, ``new_path(x, y, heading, rng)``, ``extend_path(path, i, rng)``
* ``segments()`` and ``signs()`` for drawing

``Track`` is the built-in 2 km test loop; ``OsmWorld`` (osm_world.py) loads real streets.
"""
import math
from dataclasses import dataclass

import numpy as np

from ..core.geo import ORIGIN
from .path import Path, chaikin, resample

# Rough loop (metres, x = east, y = north)
CONTROL_POINTS = [
    (0, 0), (380, 0), (470, 40), (500, 150), (460, 260), (360, 300),
    (280, 250), (200, 250), (130, 320), (20, 330), (-80, 280),
    (-120, 170), (-90, 60),
]
# (start fraction, end fraction, limit km/h, name)
ZONES = [
    (0.00, 0.17, 50, "urban"),
    (0.17, 0.33, 30, "school"),
    (0.33, 0.62, 70, "boulevard"),
    (0.62, 1.00, 50, "urban"),
]


@dataclass
class Feature:
    kind: str        # "hump" or "pothole"
    x: float
    y: float
    tx: float        # unit tangent of the road at the feature
    ty: float
    width: float     # across the road
    length: float    # along the road
    height: float    # + up (hump), - down (pothole)
    synthetic: bool = False


def feature_height(features, x, y):
    h = 0.0
    for f in features:
        dx, dy = x - f.x, y - f.y
        if abs(dx) > 6 or abs(dy) > 6:
            continue
        along = dx * f.tx + dy * f.ty
        across = -dx * f.ty + dy * f.tx
        if abs(along) < f.length / 2 and abs(across) < f.width / 2:
            h += f.height * 0.5 * (1 + math.cos(2 * math.pi * along / f.length))
    return h


class Track:
    """Closed 2 km test loop with urban, school and boulevard zones."""
    name = "test track"
    width = 8.0

    def __init__(self, control_points=CONTROL_POINTS, zones=ZONES):
        pts, _, total, _ = resample(chaikin(control_points, 4, closed=True), closed=True)
        n = len(pts)
        self.zones = [(a * total, b * total, lim, name) for a, b, lim, name in zones]
        limits = np.array([self._zone(i)[0] for i in range(n)], float)
        self.path = Path(pts, limits, closed=True, lane_offset=-2.0)
        self.origin = ORIGIN
        p = self.path
        self.bounds = (p.pts[:, 0].min(), p.pts[:, 1].min(), p.pts[:, 0].max(), p.pts[:, 1].max())
        self.features = []
        for frac in (0.21, 0.29):
            self._add("hump", frac, 0.0, self.width, 3.0, 0.07)
        for frac, lat in ((0.45, -2.0), (0.70, -2.2), (0.86, -1.8)):
            self._add("pothole", frac, lat, 1.4, 0.9, -0.06)

    def _zone(self, s):
        for a, b, lim, name in self.zones:
            if a <= s < b:
                return lim, name
        return self.zones[-1][2], self.zones[-1][3]

    def _add(self, kind, frac, lateral, width, length, height):
        p = self.path
        i = int(frac * len(p.pts))
        x, y = p.pts[i] + lateral * p.normal[i]
        tx, ty = p.tangent[i]
        self.features.append(Feature(kind, x, y, tx, ty, width, length, height))

    # ----------------------------------------------------------- interface
    def _nearest(self, x, y):
        d = self.path.pts - (x, y)
        i = int(np.argmin(np.einsum("ij,ij->i", d, d)))
        lat = float(np.dot((x, y) - self.path.pts[i], self.path.normal[i]))
        return i, lat

    def speed_limit_at(self, x, y):
        i, lat = self._nearest(x, y)
        return None if abs(lat) > self.width / 2 + 6 else float(self.path.limits[i])

    def offroad(self, x, y):
        return abs(self._nearest(x, y)[1]) > self.width / 2 + 0.5

    def height_at(self, x, y):
        return feature_height(self.features, x, y)

    def start_pose(self, rng=None):
        x, y = self.path.lane_point(5)
        tx, ty = self.path.tangent[5]
        return float(x), float(y), math.atan2(ty, tx)

    def new_path(self, x, y, heading, rng=None):
        self.path._last = self._nearest(x, y)[0]
        return self.path

    def extend_path(self, path, i, rng=None):
        return path

    def segments(self):
        """(points, width m, closed, style) for drawing."""
        p, out = self.path, []
        for a, b, lim, name in self.zones:
            ids = np.arange(int(a), min(int(b) + 2, len(p.pts)))
            out.append((p.pts[ids], self.width, False, name))
        return out

    areas = []

    def labels(self):
        return []

    def signs(self):
        p = self.path
        out = []
        for a, b, lim, name in self.zones:
            i = int(a) % len(p.pts)
            x, y = p.pts[i] + (self.width / 2 + 5) * p.normal[i]
            out.append((x, y, lim, "SCHOOL ZONE" if name == "school" else ""))
        return out


# backwards-compatible name
Road = Track
