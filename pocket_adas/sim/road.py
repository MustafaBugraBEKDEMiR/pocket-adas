"""A closed test track with speed zones, speed humps and potholes."""
import math
from dataclasses import dataclass

import numpy as np

# Rough loop (metres, x = east, y = north). Smoothed with Chaikin's algorithm.
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
    s: float         # arc length along the track (m)
    x: float
    y: float
    tx: float        # unit tangent
    ty: float
    lateral: float   # offset from centre line, + = left
    width: float     # across the road
    length: float    # along the road
    height: float    # + up (hump), - down (pothole)


def _chaikin(pts, iters=4):
    p = np.asarray(pts, dtype=float)
    for _ in range(iters):
        q = 0.75 * p + 0.25 * np.roll(p, -1, axis=0)
        r = 0.25 * p + 0.75 * np.roll(p, -1, axis=0)
        p = np.empty((2 * len(q), 2))
        p[0::2], p[1::2] = q, r
    return p


def _resample(p, step=1.0):
    closed = np.vstack([p, p[:1]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    total = s[-1]
    ss = np.arange(0, total, step)
    x = np.interp(ss, s, closed[:, 0])
    y = np.interp(ss, s, closed[:, 1])
    return np.column_stack([x, y]), ss, total


class Road:
    width = 8.0          # two lanes of 4 m
    lane_offset = -2.0   # drive on the right

    def __init__(self, control_points=CONTROL_POINTS, zones=ZONES):
        self.pts, self.s, self.length = _resample(_chaikin(control_points))
        d = np.roll(self.pts, -1, axis=0) - np.roll(self.pts, 1, axis=0)
        self.tangent = d / np.linalg.norm(d, axis=1, keepdims=True)
        self.normal = np.column_stack([-self.tangent[:, 1], self.tangent[:, 0]])  # left
        heading = np.unwrap(np.arctan2(self.tangent[:, 1], self.tangent[:, 0]))
        self.curvature = np.abs(np.gradient(heading, self.s))
        self.zones = [(a * self.length, b * self.length, lim, name) for a, b, lim, name in zones]
        self.features = []
        for frac in (0.21, 0.29):
            self._add("hump", frac, 0.0, self.width, 3.0, 0.07)
        for frac, lat in ((0.45, -2.0), (0.70, -2.2), (0.86, -1.8)):
            self._add("pothole", frac, lat, 1.4, 0.9, -0.06)

    def _add(self, kind, frac, lateral, width, length, height):
        i = int(frac * len(self.pts))
        x, y = self.pts[i] + lateral * self.normal[i]
        tx, ty = self.tangent[i]
        self.features.append(Feature(kind, self.s[i], x, y, tx, ty, lateral, width, length, height))

    # --------------------------------------------------------------- queries
    def nearest(self, x, y):
        """Index of the closest centre-line point and signed lateral offset."""
        d = self.pts - (x, y)
        i = int(np.argmin(np.einsum("ij,ij->i", d, d)))
        lat = float(np.dot((x, y) - self.pts[i], self.normal[i]))
        return i, lat

    def zone_at_s(self, s):
        s %= self.length
        for a, b, lim, name in self.zones:
            if a <= s < b:
                return lim, name
        return self.zones[-1][2], self.zones[-1][3]

    def speed_limit_at(self, x, y):
        i, lat = self.nearest(x, y)
        if abs(lat) > self.width / 2 + 6:
            return None  # off the road map
        return self.zone_at_s(self.s[i])[0]

    def height_at(self, x, y):
        h = 0.0
        for f in self.features:
            dx, dy = x - f.x, y - f.y
            along = dx * f.tx + dy * f.ty
            across = -dx * f.ty + dy * f.tx
            if abs(along) < f.length / 2 and abs(across) < f.width / 2:
                h += f.height * 0.5 * (1 + math.cos(2 * math.pi * along / f.length))
        return h

    def lane_point(self, i, offset=None):
        i %= len(self.pts)
        off = self.lane_offset if offset is None else offset
        return self.pts[i] + off * self.normal[i]

    def start_pose(self):
        x, y = self.lane_point(5)
        tx, ty = self.tangent[5]
        return float(x), float(y), math.atan2(ty, tx)
