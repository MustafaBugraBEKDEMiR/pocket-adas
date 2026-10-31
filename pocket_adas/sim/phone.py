"""A virtual smartphone mounted somewhere in the car.

It turns the true vehicle motion into what a real phone would report:
accelerometer (specific force, *including gravity*, Android convention),
gyroscope and a 1 Hz GPS - all rotated into the phone's own axes and noisy.
"""
import math

import numpy as np

from ..core.geo import xy_to_latlon

G = 9.80665


def _rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


# Rows = phone x/y/z axes written in vehicle coordinates (x fwd, y left, z up)
_DASHBOARD = np.array([[0, -1, 0],   # phone x -> right of car
                       [0, 0, 1],    # phone y -> up
                       [-1, 0, 0]])  # phone z (screen) -> towards driver
_FLAT = np.array([[0, -1, 0],        # lying on the console, top pointing forward
                  [1, 0, 0],
                  [0, 0, 1]])

MOUNTS = ("dashboard", "flat", "random")


def mount_rotation(kind: str, rng: np.random.Generator) -> np.ndarray:
    """Vehicle -> phone rotation matrix for a mount style."""
    if kind == "dashboard":
        return _rx(math.radians(rng.uniform(10, 30))) @ _DASHBOARD @ _rz(math.radians(rng.uniform(-10, 10)))
    if kind == "flat":
        return _FLAT @ _rz(rng.uniform(0, 2 * math.pi))
    # uniformly random orientation
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q @ np.diag(np.sign(np.diag(r)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


class VirtualPhone:
    def __init__(self, rng: np.random.Generator, mount: str = "dashboard",
                 fs: float = 50.0, gps_rate: float = 1.0):
        self.rng = rng
        self.fs = fs
        self.gps_period = 1.0 / gps_rate
        self.remount(mount)
        self._next_gps = 0.0
        self._gps_err = np.zeros(2)

    def remount(self, mount: str):
        self.mount = mount
        self.R = mount_rotation(mount, self.rng)
        self.acc_bias = self.rng.normal(0, 0.05, 3)
        self.gyro_bias = self.rng.normal(0, 0.002, 3)

    def sample(self, t: float, veh) -> dict:
        rng = self.rng
        f_v = np.array([veh.a_long, veh.a_lat, veh.a_vert + G])
        vib = rng.normal(0, 1, 3) * np.array([0.35, 0.35, 1.0]) * (0.03 + 0.012 * veh.v)
        acc = self.R @ (f_v + vib) + self.acc_bias + rng.normal(0, 0.03, 3)
        w_v = np.array([veh.roll_rate, veh.pitch_rate, veh.yaw_rate])
        gyro = self.R @ w_v + self.gyro_bias + rng.normal(0, 0.004 + 0.0004 * veh.v, 3)
        out = {"t": t, "acc": acc, "gyro": gyro, "gps": None}
        if t + 1e-9 >= self._next_gps:
            self._next_gps += self.gps_period
            # slowly wandering position error, like a real receiver
            self._gps_err = 0.8 * self._gps_err + rng.normal(0, 0.8, 2)
            gx, gy = veh.x + self._gps_err[0], veh.y + self._gps_err[1]
            lat, lon = xy_to_latlon(gx, gy)
            speed = max(0.0, veh.v + rng.normal(0, 0.12)) if veh.v > 0.05 else 0.0
            out["gps"] = {"lat": lat, "lon": lon, "speed": speed}
        return out
