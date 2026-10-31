"""The on-phone ADAS pipeline.

Feed it raw phone samples (accelerometer + gyroscope at ~50 Hz, GPS at ~1 Hz)
and it returns events and a status for the screen. It does not know whether
the samples come from a real phone, a CSV file or the simulator.
"""
from typing import Callable, Optional

import numpy as np

from . import events as E
from .calibration import MountCalibrator
from .detectors import DetectorConfig, ThresholdDetector, VerticalDetector, SpeedingDetector
from .filters import EMA, HighPass
from .scoring import trip_score

class ADASPipeline:
    def __init__(self, fs: float = 50.0, config: Optional[DetectorConfig] = None,
                 speed_limit_fn: Optional[Callable[[dict], Optional[float]]] = None):
        """``speed_limit_fn(gps) -> km/h or None``; on a real phone this would be
        an OpenStreetMap ``maxspeed`` lookup, in the simulator it asks the road."""
        self.fs = fs
        self.dt = 1.0 / fs
        self.cfg = config or DetectorConfig()
        self.speed_limit_fn = speed_limit_fn
        self.reset()

    def reset(self):
        c, dt = self.cfg, self.dt
        self.cal = MountCalibrator(self.fs)
        self._lon = EMA(c.smoothing_tau_s, dt)
        self._lat = EMA(c.smoothing_tau_s, dt)
        self._hmag = EMA(c.smoothing_tau_s, dt)
        self._vert = HighPass(0.5, dt)
        self.det_brake = ThresholdDetector(E.HARSH_BRAKE, c.harsh_brake, c.min_duration_s, c.cooldown_s)
        self.det_accel = ThresholdDetector(E.HARSH_ACCEL, c.harsh_accel, c.min_duration_s, c.cooldown_s)
        self.det_turn = ThresholdDetector(E.HARSH_TURN, c.harsh_turn, c.min_duration_s, c.cooldown_s)
        self.det_move = ThresholdDetector(E.SUDDEN_MOVE, c.sudden_move, c.min_duration_s, c.cooldown_s)
        self.det_vert = VerticalDetector(c.vertical, self.fs)
        self.det_speed = SpeedingDetector(c.speeding_tolerance_kmh, c.speeding_min_s)
        self.events = []
        self.status = E.Status()
        self.t = 0.0
        self.speed_ms = 0.0          # GPS speed fused with forward acceleration
        self.limit_kmh = None
        self.lat = self.lon = None
        self.distance_m = 0.0
        self.signals = {"lon": 0.0, "lat": 0.0, "vert": 0.0, "yaw": 0.0, "lon_raw": 0.0, "lat_raw": 0.0}

    # ----------------------------------------------------------------- public
    @property
    def score(self) -> int:
        return trip_score(self.events, self.distance_m)

    @property
    def speed_kmh(self) -> float:
        return self.speed_ms * 3.6

    def update(self, t, acc, gyro, gps: Optional[dict] = None):
        """Process one sample. ``gps`` = {lat, lon, speed(m/s)} when a fix arrives."""
        acc = np.asarray(acc, dtype=float)
        gyro = np.asarray(gyro, dtype=float)
        dt = self.dt if self.t == 0 else max(1e-3, t - self.t)
        self.t = t
        new = []

        if gps is not None:
            self.cal.on_gps(t, gps["speed"])
            self.speed_ms = gps["speed"]
            self.lat, self.lon = gps.get("lat"), gps.get("lon")
            if self.speed_limit_fn is not None:
                self.limit_kmh = self.speed_limit_fn(gps)
        self.cal.update(acc, gyro)

        if self.cal.gravity is None:
            self.status = E.Status(E.CALIBRATING, "CALIBRATING - KEEP STILL")
            return new

        ctx = {"speed_kmh": self.speed_kmh, "lat": self.lat, "lon": self.lon}
        up = self.cal.up
        a_lin = acc - self.cal.gravity
        vert = float(np.dot(a_lin, up))
        self.signals["yaw"] = float(np.dot(gyro, up))
        self.signals["vert"] = v_hp = self._vert(vert)

        fwd = self.cal.forward
        if fwd is not None:
            lon_raw = float(np.dot(a_lin, fwd))
            lat_raw = float(np.dot(a_lin, self.cal.left))
            self.signals["lon_raw"], self.signals["lat_raw"] = lon_raw, lat_raw
            lon, lat = self._lon(lon_raw), self._lat(lat_raw)
            self.signals["lon"], self.signals["lat"] = lon, lat
            # dead-reckon speed between GPS fixes
            if gps is None:
                self.speed_ms = max(0.0, self.speed_ms + lon * dt)
            for det, val in ((self.det_brake, -lon), (self.det_accel, lon), (self.det_turn, abs(lat))):
                ev = det.update(t, val, **ctx)
                if ev:
                    new.append(ev)
        else:
            h = a_lin - vert * up
            hm = self._hmag(float(np.linalg.norm(h)))
            self.signals["lon"], self.signals["lat"] = hm, 0.0
            ev = self.det_move.update(t, hm, **ctx)
            if ev:
                new.append(ev)

        if self.speed_ms > 1.0:   # ignore door slams and phone handling at rest
            ev = self.det_vert.update(t, v_hp, **ctx)
            if ev:
                new.append(ev)
        ctx.pop("speed_kmh")
        ev = self.det_speed.update(t, self.speed_kmh, self.limit_kmh, **ctx)
        if ev:
            new.append(ev)

        self.distance_m += self.speed_ms * dt
        self.events.extend(new)
        self._update_status(t)
        return new

    # ---------------------------------------------------------------- helpers
    def _update_status(self, t):
        c = self.cfg
        actives = {id(d.active) for d in (self.det_brake, self.det_accel, self.det_turn,
                                          self.det_move, self.det_vert, self.det_speed)
                   if d.active is not None}
        recent = [e for e in self.events[-12:] if id(e) in actives or t - e.t < c.alert_hold_s]
        if not recent:
            msg = "YOU ARE SAFE" if self.cal.stage == "done" else "YOU ARE SAFE (calibrating)"
            self.status = E.Status(E.SAFE, msg)
            return
        worst = max(recent, key=lambda e: e.severity)
        danger = worst.severity >= c.danger_ratio or (
            worst.kind == E.SPEEDING and (worst.limit_kmh or 99) <= 30)
        self.status = E.Status(E.DANGER if danger else E.WARNING, worst.message, worst)
