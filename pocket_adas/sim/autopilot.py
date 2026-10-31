"""Scripted drivers: pure-pursuit steering plus a speed planner.

``calm`` keeps every acceleration gentle and respects limits; ``aggressive``
speeds, brakes late, takes humps fast and throws in sudden brakes and swerves.
They are used for hands-free demos and (phase 2) for labelled training data.
"""
import math
from dataclasses import dataclass


@dataclass
class DriverProfile:
    speed_factor: float
    a_lat_max: float
    acc_max: float
    dec_max: float
    slow_for_humps: bool
    disturb_every_s: float   # mean time between sudden manoeuvres (0 = never)


PROFILES = {
    "calm": DriverProfile(0.9, 1.6, 1.4, 1.8, True, 0.0),
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

    def _target_speed(self, road, i, v):
        p = self.p
        n = len(road.pts)
        limit, _ = road.zone_at_s(road.s[i])
        best = limit / 3.6 * p.speed_factor
        dec = p.dec_max * (0.8 if self.name == "calm" else 1.0)
        for d in range(5, 120, 3):
            j = (i + d) % n
            k = road.curvature[j]
            v_pt = math.sqrt(p.a_lat_max / k) if k > 1e-4 else 99.0
            lim_j, _ = road.zone_at_s(road.s[j])
            v_pt = min(v_pt, lim_j / 3.6 * p.speed_factor)
            best = min(best, math.sqrt(v_pt ** 2 + 2 * dec * d))
        if p.slow_for_humps:
            for f in road.features:
                if f.kind != "hump":
                    continue
                d = (f.s - road.s[i]) % road.length
                if d < 60:
                    best = min(best, math.sqrt((15 / 3.6) ** 2 + 2 * dec * max(0.0, d - 3)))
        return best

    def control(self, sim, dt):
        veh, road, t = sim.vehicle, sim.road, sim.t
        p = self.p
        i, _ = road.nearest(veh.x, veh.y)

        # random sudden manoeuvres for the aggressive driver
        if p.disturb_every_s > 0 and veh.v > 8 and self.rng.random() < dt / p.disturb_every_s:
            if self.rng.random() < 0.5:
                self._brake_until = t + self.rng.uniform(0.8, 1.5)
            else:
                self._swerve_t0, self._swerve_dir = t, self.rng.choice([-1.0, 1.0])

        # lane offset (+ a quick swerve)
        offset = road.lane_offset
        ts = t - self._swerve_t0
        if ts < 1.6:
            offset += 3.0 * self._swerve_dir * math.sin(math.pi * ts / 1.6)

        # pure pursuit
        look = max(6.0, 0.9 * veh.v)
        tx, ty = road.lane_point(i + int(look), offset)
        alpha = math.atan2(ty - veh.y, tx - veh.x) - veh.heading
        alpha = (alpha + math.pi) % (2 * math.pi) - math.pi
        delta = math.atan2(2 * veh.p.wheelbase * math.sin(alpha), look)
        steer = delta / veh.p.max_steer

        # speed
        v_t = self._target_speed(road, i, veh.v)
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
