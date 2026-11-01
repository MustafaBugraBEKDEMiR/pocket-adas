"""A drivable centre line resampled at 1 m, with per-point speed limits."""
import numpy as np


def chaikin(p, iters=2, closed=False):
    p = np.asarray(p, dtype=float)
    for _ in range(iters):
        if closed:
            q = 0.75 * p + 0.25 * np.roll(p, -1, axis=0)
            r = 0.25 * p + 0.75 * np.roll(p, -1, axis=0)
            p = np.empty((2 * len(q), 2))
            p[0::2], p[1::2] = q, r
        else:
            if len(p) < 3:
                return p
            q = 0.75 * p[:-1] + 0.25 * p[1:]
            r = 0.25 * p[:-1] + 0.75 * p[1:]
            mid = np.empty((2 * len(q), 2))
            mid[0::2], mid[1::2] = q, r
            p = np.vstack([p[:1], mid, p[-1:]])
    return p


def resample(p, values=None, step=1.0, closed=False):
    """Resample a polyline at ``step`` metres; ``values`` (per vertex) are carried
    along by nearest-previous-vertex."""
    pts = np.vstack([p, p[:1]]) if closed else np.asarray(p, float)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    total = s[-1]
    ss = np.arange(0, total, step) if closed else np.arange(0, total + 1e-9, step)
    out = np.column_stack([np.interp(ss, s, pts[:, 0]), np.interp(ss, s, pts[:, 1])])
    vals = None
    if values is not None:
        v = np.concatenate([values, values[:1]]) if closed else np.asarray(values)
        idx = np.clip(np.searchsorted(s, ss, side="right") - 1, 0, len(v) - 1)
        vals = v[idx]
    return out, ss, total, vals


class Path:
    """Centre line a driver follows. ``limits`` = km/h per point."""

    def __init__(self, pts, limits, closed=False, lane_offset=-2.0):
        self.closed = closed
        self.pts = np.asarray(pts, float)
        self.limits = np.asarray(limits, float)
        n = len(self.pts)
        self.s = np.arange(n, dtype=float)
        self.length = float(n)
        if closed:
            d = np.roll(self.pts, -1, axis=0) - np.roll(self.pts, 1, axis=0)
        else:
            d = np.gradient(self.pts, axis=0)
        self.tangent = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
        self.normal = np.column_stack([-self.tangent[:, 1], self.tangent[:, 0]])
        heading = np.unwrap(np.arctan2(self.tangent[:, 1], self.tangent[:, 0]))
        k = np.convolve(np.abs(np.gradient(heading)), np.ones(3) / 3, mode="same")
        # conservative for speed planning: worst curvature within +-5 m
        pad = np.pad(k, 5, mode="wrap" if closed else "edge")
        self.curvature = np.lib.stride_tricks.sliding_window_view(pad, 11).max(axis=1)
        self.lane_offset = np.full(n, lane_offset) if np.isscalar(lane_offset) else np.asarray(lane_offset, float)
        self._last = 0

    def idx(self, i):
        n = len(self.pts)
        return i % n if self.closed else max(0, min(n - 1, i))

    def nearest(self, x, y, window=60):
        """Closest index; searches near the previous answer first (paths may cross)."""
        n = len(self.pts)
        lo, hi = self._last - window // 3, self._last + window
        ids = np.arange(lo, hi) % n if self.closed else np.arange(max(0, lo), min(n, hi))
        d = self.pts[ids] - (x, y)
        dd = np.einsum("ij,ij->i", d, d)
        j = int(np.argmin(dd))
        if dd[j] > 15 ** 2:  # lost: global search
            d = self.pts - (x, y)
            i = int(np.argmin(np.einsum("ij,ij->i", d, d)))
        else:
            i = int(ids[j])
        self._last = i
        return i

    def lane_point(self, i, offset=None):
        i = self.idx(i)
        off = self.lane_offset[i] if offset is None else offset
        return self.pts[i] + off * self.normal[i]

    def remaining(self, i):
        return float("inf") if self.closed else len(self.pts) - 1 - i
