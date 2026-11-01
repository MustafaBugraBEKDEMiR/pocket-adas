"""Scripted drivers: pure-pursuit steering plus a speed planner.

``calm`` keeps every acceleration gentle and respects limits; ``aggressive``
speeds, brakes late, takes humps fast and throws in sudden brakes and swerves.
They are used for hands-free demos and (phase 2) for labelled training data.
"""
import math
from dataclasses import dataclass

import numpy as np


@dataclass
class DriverProfile:
    speed_factor: float
    a_lat_max: float
    acc_max: float
    dec_max: float
    slow_for_humps: bool
    disturb_every_s: float   # mean time between sudden manoeuvres (0 = never)
    swerve_amp: float = 3.0  # lateral offset of a swerve (m)
    swerve_time: float = 1.6


PROFILES = {
    "calm": DriverProfile(0.9, 1.6, 1.4, 1.8, True, 0.0),
    # in between: mostly fine, sometimes crosses a threshold - the hard cases
    "normal": DriverProfile(1.05, 3.0, 2.6, 3.4, True, 30.0, 2.0, 2.2),
    "aggressive": DriverProfile(1.3, 6.5, 3.8, 7.5, False, 18.0),
}


class Autopilot:
    def __init__(self, profile: str, rng):
        self.name = profile
        self.p = PROFILES[profile]
        self.rng = rng
        self._brake_until = -1.0
        self._swerve_t0 = -10.0
        self._swerve_dir = 1.0
        self.path = None
        self._humps = []   # path indices of speed humps

    # ------------------------------------------------------------- route
    def _set_path(self, path, world):
        self.path = path
        self._humps = []
        for f in world.features:
            if f.kind != "hump":
                continue
            d = path.pts - (f.x, f.y)
            j = int(np.argmin(np.einsum("ij,ij->i", d, d)))
            if np.hypot(*d[j]) < 6:
                self._humps.append(j)

    def _locate(self, sim):
        veh, world = sim.vehicle, sim.world
        if self.path is None:
            self._set_path(world.new_path(veh.x, veh.y, veh.heading, self.rng), world)
        i = self.path.nearest(veh.x, veh.y)
        if np.hypot(*(self.path.pts[i] - (veh.x, veh.y))) > 20:   # e.g. after manual driving
            self._set_path(world.new_path(veh.x, veh.y, veh.heading, self.rng), world)
            i = self.path.nearest(veh.x, veh.y)
        new = world.extend_path(self.path, i, self.rng)
        if new is not self.path:
            self._set_path(new, world)
            i = self.path.nearest(veh.x, veh.y)
        return i

    def _target_speed(self, i):
        p, path = self.p, self.path
        n = len(path.pts)
        best = path.limits[i] / 3.6 * p.speed_factor
        dec = p.dec_max * (0.8 if self.name != "aggressive" else 1.0)
        for d in range(0, 120, 2):
            if not path.closed and i + d >= n:
                break
            j = path.idx(i + d)
            k = path.curvature[j]
            v_pt = math.sqrt(p.a_lat_max / k) if k > 1e-4 else 99.0
            v_pt = min(v_pt, path.limits[j] / 3.6 * p.speed_factor)
            best = min(best, math.sqrt(v_pt ** 2 + 2 * dec * d))
        if p.slow_for_humps:
            for j in self._humps:
                d = (j - i) % n if path.closed else j - i
                if 0 <= d < 60:
                    best = min(best, math.sqrt((15 / 3.6) ** 2 + 2 * dec * max(0.0, d - 3)))
        return best

    def control(self, sim, dt):
        veh, t = sim.vehicle, sim.t
        p = self.p
        i = self._locate(sim)
        path = self.path

        # random sudden manoeuvres
        if p.disturb_every_s > 0 and veh.v > 8 and self.rng.random() < dt / p.disturb_every_s:
            if self.rng.random() < 0.5:
                self._brake_until = t + self.rng.uniform(0.8, 1.5)
            else:
                self._swerve_t0, self._swerve_dir = t, self.rng.choice([-1.0, 1.0])

        # lane offset (+ a quick swerve)
        look = max(6.0, 0.9 * veh.v)
        j = path.idx(i + int(look))
        offset = path.lane_offset[j]
        ts = t - self._swerve_t0
        if ts < p.swerve_time:
            offset += p.swerve_amp * self._swerve_dir * math.sin(math.pi * ts / p.swerve_time)

        # pure pursuit
        tx, ty = path.lane_point(j, offset)
        alpha = math.atan2(ty - veh.y, tx - veh.x) - veh.heading
        alpha = (alpha + math.pi) % (2 * math.pi) - math.pi
        delta = math.atan2(2 * veh.p.wheelbase * math.sin(alpha), look)
        steer = delta / veh.p.max_steer

        # speed
        v_t = self._target_speed(i)
        a_des = max(-p.dec_max, min(p.acc_max, 1.5 * (v_t - veh.v)))
        if t < self._brake_until:
            a_des = -p.dec_max
        vp = veh.p
        if a_des >= 0:
            resist = vp.rolling + vp.drag * veh.v ** 2
            throttle = (a_des + resist) / (vp.max_accel * max(0.05, 1 - veh.v / vp.max_speed))
        else:
            throttle = a_des / vp.max_brake
        return throttle, steer
