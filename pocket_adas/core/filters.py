"""Tiny streaming filters (no SciPy, so the core also runs in Pydroid)."""


class EMA:
    """First-order low-pass filter with time constant ``tau`` seconds."""

    def __init__(self, tau: float, dt: float):
        self.alpha = 1.0 if tau <= 0 else dt / (tau + dt)
        self.y = None

    def __call__(self, x):
        self.y = x if self.y is None else self.y + self.alpha * (x - self.y)
        return self.y

    def reset(self):
        self.y = None


class HighPass:
    """x minus its slow trend."""

    def __init__(self, tau: float, dt: float):
        self.lp = EMA(tau, dt)

    def __call__(self, x):
        return x - self.lp(x)

    def reset(self):
        self.lp.reset()
