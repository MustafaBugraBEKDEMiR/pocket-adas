"""Glue: road + vehicle + virtual phone + ADAS pipeline, stepped at a fixed rate."""
import csv
import json
import os
import time

import numpy as np

from ..core import ADASPipeline
from ..core.geo import latlon_to_xy
from ..core.scoring import event_counts
from .phone import VirtualPhone
from .road import Road
from .vehicle import Vehicle


class Simulation:
    def __init__(self, seed=None, mount="dashboard", fs=50.0, record=True):
        self.rng = np.random.default_rng(seed)
        self.fs = fs
        self.dt = 1.0 / fs
        self.road = Road()
        self.phone = VirtualPhone(self.rng, mount, fs)
        self.pipeline = ADASPipeline(fs, speed_limit_fn=self._limit_from_gps)
        self.record = record
        self.reset_vehicle()

    def reset_vehicle(self):
        x, y, h = self.road.start_pose()
        self.vehicle = Vehicle(x, y, h)
        self.t = 0.0
        self.phone._next_gps = 0.0
        self.pipeline.reset()
        self.samples = []      # recorded phone log
        self.truth = []        # true vehicle state, for evaluation

    def remount_phone(self, mount=None):
        """Move the phone to a new random position; the core must re-calibrate."""
        self.phone.remount(mount or self.phone.mount)
        self.pipeline.cal.reset()

    def _limit_from_gps(self, gps):
        x, y = latlon_to_xy(gps["lat"], gps["lon"])
        return self.road.speed_limit_at(x, y)

    def step(self, throttle, steer):
        veh = self.vehicle
        _, lat = self.road.nearest(veh.x, veh.y)
        off_road = abs(lat) > self.road.width / 2 + 0.5
        veh.step(throttle, steer, self.dt, self.road.height_at, extra_drag=1.5 if off_road else 0.0)
        self.t += self.dt
        s = self.phone.sample(self.t, veh)
        self.last_sample = s
        events = self.pipeline.update(self.t, s["acc"], s["gyro"], s["gps"])
        if self.record:
            g = s["gps"] or {}
            self.samples.append((round(self.t, 3), *np.round(s["acc"], 4), *np.round(s["gyro"], 5),
                                 g.get("lat", ""), g.get("lon", ""),
                                 round(g["speed"], 3) if g else ""))
            self.truth.append((round(self.t, 3), round(veh.x, 2), round(veh.y, 2), round(veh.v, 3),
                               round(veh.a_long, 3), round(veh.a_lat, 3), round(veh.a_vert, 3)))
        return events

    # --------------------------------------------------------------- output
    def save_trip(self, out_dir="trips", tag="manual"):
        """Write sensors.csv (Sensor-Logger-like), truth.csv, events.csv, summary.json."""
        if not self.samples:
            return None
        d = os.path.join(out_dir, time.strftime("%Y%m%d_%H%M%S") + "_" + tag)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "sensors.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["seconds_elapsed", "acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z",
                        "latitude", "longitude", "speed"])
            w.writerows(self.samples)
        with open(os.path.join(d, "truth.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t", "x", "y", "v", "a_long", "a_lat", "a_vert"])
            w.writerows(self.truth)
        p = self.pipeline
        with open(os.path.join(d, "events.csv"), "w", newline="") as f:
            keys = ["t", "kind", "peak", "severity", "speed_kmh", "lat", "lon", "duration", "limit_kmh"]
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for e in p.events:
                w.writerow(e.to_dict())
        summary = {
            "tag": tag, "duration_s": round(self.t, 1), "distance_m": round(p.distance_m, 1),
            "score": p.score, "events": event_counts(p.events),
            "mount": self.phone.mount,
            "calibration": p.cal.stage,
        }
        with open(os.path.join(d, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        return d
