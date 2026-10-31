"""Kinematic bicycle model with a quarter-car vertical model."""
import math
from dataclasses import dataclass

G = 9.80665


@dataclass
class VehicleParams:
    wheelbase: float = 2.6
    length: float = 4.3
    width: float = 1.8
    max_steer: float = math.radians(32)
    steer_rate: float = math.radians(90)   # actuator limit, rad/s
    max_accel: float = 3.8                 # full throttle at low speed
    max_brake: float = 9.0
    max_speed: float = 50.0                # m/s
    rolling: float = 0.15
    drag: float = 0.0005                   # a = drag * v^2
    grip: float = 0.9                      # friction coefficient
    susp_freq: float = 1.3                 # body natural frequency, Hz
    susp_damping: float = 0.45
    wheel_harshness: float = 0.12          # share of wheel acceleration felt in the cabin


class Vehicle:
    def __init__(self, x, y, heading, params: VehicleParams = None):
        self.p = params or VehicleParams()
        self.x, self.y, self.heading = x, y, heading
        self.v = 0.0
        self.steer = 0.0
        self.a_long = self.a_lat = self.a_vert = 0.0
        self.yaw_rate = self.pitch_rate = self.roll_rate = 0.0
        self.odometer = 0.0
        self.sliding = False
        self._z = self._zd = 0.0        # body heave
        self._h_prev = None             # (h_front, h_rear)
        self._hd_prev = (0.0, 0.0)
        self._alat_prev = 0.0

    def axle_positions(self):
        c, s = math.cos(self.heading), math.sin(self.heading)
        hb = self.p.wheelbase / 2
        return (self.x + hb * c, self.y + hb * s), (self.x - hb * c, self.y - hb * s)

    def step(self, throttle, steer_cmd, dt, height_fn=None, extra_drag=0.0):
        p = self.p
        throttle = max(-1.0, min(1.0, throttle))
        steer_cmd = max(-1.0, min(1.0, steer_cmd))

        # --- longitudinal
        if throttle >= 0:
            a = throttle * p.max_accel * max(0.0, 1 - self.v / p.max_speed)
        else:
            a = throttle * p.max_brake
        if self.v > 0.05:
            a -= p.rolling + p.drag * self.v ** 2 + extra_drag
        a = max(-p.grip * G, min(p.grip * G, a))
        v_new = max(0.0, self.v + a * dt)
        self.a_long = (v_new - self.v) / dt
        self.v = v_new

        # --- steering + lateral (grip-limited)
        target = steer_cmd * p.max_steer
        d = max(-p.steer_rate * dt, min(p.steer_rate * dt, target - self.steer))
        self.steer += d
        yaw = self.v / p.wheelbase * math.tan(self.steer)
        a_max = p.grip * G
        self.sliding = abs(self.v * yaw) > a_max
        if self.sliding:
            yaw = math.copysign(a_max / max(self.v, 0.1), yaw)
        self.yaw_rate = yaw
        self.a_lat = self.v * yaw
        self.roll_rate = 0.015 * (self.a_lat - self._alat_prev) / dt
        self._alat_prev = self.a_lat

        self.heading += yaw * dt
        self.x += self.v * math.cos(self.heading) * dt
        self.y += self.v * math.sin(self.heading) * dt
        self.odometer += self.v * dt

        # --- vertical (quarter car driven by the mean axle height)
        if height_fn is None:
            self.a_vert = self.pitch_rate = 0.0
            return
        (fx, fy), (rx, ry) = self.axle_positions()
        hf, hr = height_fn(fx, fy), height_fn(rx, ry)
        if self._h_prev is None:
            self._h_prev = (hf, hr)
        hfd, hrd = (hf - self._h_prev[0]) / dt, (hr - self._h_prev[1]) / dt
        hfdd, hrdd = (hfd - self._hd_prev[0]) / dt, (hrd - self._hd_prev[1]) / dt
        self._h_prev, self._hd_prev = (hf, hr), (hfd, hrd)
        h, hd, hdd = (hf + hr) / 2, (hfd + hrd) / 2, (hfdd + hrdd) / 2
        w = 2 * math.pi * p.susp_freq
        zdd = -w * w * (self._z - h) - 2 * p.susp_damping * w * (self._zd - hd)
        self._zd += zdd * dt
        self._z += self._zd * dt
        self.a_vert = zdd + p.wheel_harshness * hdd
        self.pitch_rate = 0.5 * (hfd - hrd) / p.wheelbase
