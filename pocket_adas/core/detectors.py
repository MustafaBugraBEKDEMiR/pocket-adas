"""Streaming event detectors."""
from collections import deque
from dataclasses import dataclass

from .events import Event, BUMP, POTHOLE


@dataclass
class DetectorConfig:
    # thresholds in m/s^2 (1 g = 9.81); typical telematics values are 0.25-0.35 g
    harsh_brake: float = 3.0
    harsh_accel: float = 2.5
    harsh_turn: float = 3.0
    sudden_move: float = 3.5      # used only before the forward axis is known
    vertical: float = 3.5         # bump / pothole jolt
    min_duration_s: float = 0.25
    cooldown_s: float = 1.0
    speeding_tolerance_kmh: float = 5.0
    speeding_min_s: float = 2.0
    smoothing_tau_s: float = 0.25
    alert_hold_s: float = 2.5     # how long a warning stays on screen
    danger_ratio: float = 1.5     # severity at which a warning becomes DANGER


class ThresholdDetector:
    """Fires once when ``value`` stays above ``threshold`` for ``min_duration``.

    The returned Event object keeps being updated (peak, severity, duration)
    until the value falls back below 70 % of the threshold.
    """

    def __init__(self, kind, threshold, min_duration, cooldown):
        self.kind = kind
        self.thr = threshold
        self.min_dur = min_duration
        self.cooldown = cooldown
        self._above_since = None
        self._last_end = -1e9
        self.active = None

    def update(self, t, value, **ctx):
        if self.active is not None:
            ev = self.active
            if value > ev.peak:
                ev.peak = value
                ev.severity = value / self.thr
            if value < 0.7 * self.thr:
                ev.duration = t - ev.t
                self.active = None
                self._last_end = t
            return None
        if value >= self.thr:
            if self._above_since is None:
                self._above_since = t
            if t - self._above_since >= self.min_dur and t - self._last_end >= self.cooldown:
                ev = Event(t=t, kind=self.kind, peak=value, severity=value / self.thr, **ctx)
                self.active = ev
                self._above_since = None
                return ev
        else:
            self._above_since = None
        return None


class VerticalDetector:
    """Detects short vertical jolts and tells bumps from potholes.

    A bump first pushes the car body up (positive), a pothole first drops it
    (negative) - the same idea as Fazeen et al. (2012).
    """

    def __init__(self, threshold, fs, cooldown=0.8, lookback_s=0.3):
        self.thr = threshold
        self.cooldown = cooldown
        self._buf = deque(maxlen=max(3, int(lookback_s * fs)))
        self._last = -1e9
        self.active = None

    def update(self, t, value, **ctx):
        self._buf.append(value)
        if self.active is not None and t - self.active.t > 0.4:
            self.active.duration = t - self.active.t
            self.active = None
        if self.active is not None:
            if abs(value) > self.active.peak:
                self.active.peak = abs(value)
                self.active.severity = abs(value) / self.thr
            return None
        if abs(value) < self.thr or t - self._last < self.cooldown:
            return None
        first = next((v for v in self._buf if abs(v) > 0.5 * self.thr), value)
        kind = BUMP if first > 0 else POTHOLE
        self._last = t
        ev = Event(t=t, kind=kind, peak=abs(value), severity=abs(value) / self.thr, **ctx)
        self.active = ev
        return ev


class SpeedingDetector:
    def __init__(self, tolerance_kmh, min_s):
        self.tol = tolerance_kmh
        self.min_s = min_s
        self._since = None
        self.active = None

    def update(self, t, speed_kmh, limit_kmh, **ctx):
        ctx["speed_kmh"] = speed_kmh
        if limit_kmh is None:
            self._since = None
            return None
        over = speed_kmh - limit_kmh
        if self.active is not None:
            self.active.peak = max(self.active.peak, speed_kmh)
            self.active.severity = max(1.0, (self.active.peak - limit_kmh) / 10.0)
            if over <= 0:
                self.active.duration = t - self.active.t
                self.active = None
            return None
        if over > self.tol:
            if self._since is None:
                self._since = t
            if t - self._since >= self.min_s:
                self._since = None
                self.active = Event(t=t, kind="SPEEDING", peak=speed_kmh,
                                    severity=max(1.0, over / 10.0),
                                    limit_kmh=limit_kmh, **ctx)
                return self.active
        else:
            self._since = None
        return None
