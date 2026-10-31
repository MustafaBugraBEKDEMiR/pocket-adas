"""Automatic phone-mount calibration.

The phone can sit in the car at any angle. To read "forward / left / up"
accelerations we need the rotation between the phone frame and the vehicle
frame. It is estimated in two stages, with no input from the driver:

1. **Up axis (gravity).** While the phone is still (low accelerometer variance,
   no rotation, GPS speed ~0) the mean accelerometer reading is the gravity
   vector. This is also what the 2023 version was missing: its
   ``|x|+|y|+|z| >= 15`` rule mixed gravity into every reading.
2. **Forward axis.** Between two GPS fixes the speed change tells us how much
   the car accelerated forward. Correlating that with the horizontal part of
   the measured acceleration (while driving straight) gives the forward
   direction; ``left = up x forward``.
"""
from collections import deque

import numpy as np

G = 9.80665


class MountCalibrator:
    def __init__(self, fs: float = 50.0, still_window_s: float = 1.5,
                 min_forward_fixes: int = 3):
        self.fs = fs
        n = max(5, int(still_window_s * fs))
        self._acc = deque(maxlen=n)
        self._gyro = deque(maxlen=n)
        self.min_forward_fixes = min_forward_fixes
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self):
        self._acc.clear()
        self._gyro.clear()
        self.gravity = None          # phone-frame gravity reading (m/s^2)
        self._fwd_sum = np.zeros(3)
        self._fwd_n = 0
        self._interval_acc = np.zeros(3)
        self._interval_yaw = 0.0
        self._interval_n = 0
        self._last_fix = None        # (t, speed m/s)
        self.gps_speed = None

    @property
    def up(self):
        return None if self.gravity is None else self.gravity / np.linalg.norm(self.gravity)

    @property
    def forward(self):
        if self.gravity is None or self._fwd_n < self.min_forward_fixes:
            return None
        u = self.up
        f = self._fwd_sum - np.dot(self._fwd_sum, u) * u
        n = np.linalg.norm(f)
        return None if n < 1e-6 else f / n

    @property
    def left(self):
        f = self.forward
        return None if f is None else np.cross(self.up, f)

    @property
    def stage(self) -> str:
        if self.gravity is None:
            return "gravity"
        if self.forward is None:
            return "forward"
        return "done"

    def rotation(self):
        """3x3 matrix whose rows are vehicle forward/left/up in phone coords."""
        f = self.forward
        if f is None:
            return None
        return np.vstack([f, self.left, self.up])

    # ----------------------------------------------------------------- update
    def is_still(self) -> bool:
        if len(self._acc) < self._acc.maxlen:
            return False
        a = np.asarray(self._acc)
        w = np.asarray(self._gyro)
        if a.std(axis=0).max() > 0.20 or np.abs(w).mean() > 0.04:
            return False
        if abs(np.linalg.norm(a.mean(axis=0)) - G) > 0.6:
            return False
        if self.gps_speed is not None and self.gps_speed > 0.8:
            return False
        return True

    def update(self, acc: np.ndarray, gyro: np.ndarray):
        self._acc.append(acc)
        self._gyro.append(gyro)
        if self.is_still():
            g = np.asarray(self._acc).mean(axis=0)
            if self.gravity is None:
                self.gravity = g
            else:  # slowly track small mount changes while stopped
                self.gravity = 0.98 * self.gravity + 0.02 * g
        if self.gravity is not None:
            self._interval_acc += acc - self.gravity
            self._interval_yaw += abs(float(np.dot(gyro, self.up)))
            self._interval_n += 1

    def on_gps(self, t: float, speed: float):
        """Feed a GPS fix (speed in m/s)."""
        self.gps_speed = speed
        if self._last_fix is not None and self._interval_n > 0 and self.gravity is not None:
            t0, v0 = self._last_fix
            dt = t - t0
            if 0.3 < dt < 3.0:
                dvdt = (speed - v0) / dt
                mean_a = self._interval_acc / self._interval_n
                mean_yaw = self._interval_yaw / self._interval_n
                if abs(dvdt) > 0.6 and mean_yaw < 0.06 and max(speed, v0) > 1.0:
                    u = self.up
                    h = mean_a - np.dot(mean_a, u) * u
                    self._fwd_sum += h * dvdt
                    self._fwd_n += 1
        self._last_fix = (t, speed)
        self._interval_acc = np.zeros(3)
        self._interval_yaw = 0.0
        self._interval_n = 0
