"""Labelled data from the simulator.

Every trip records, per 50 Hz sample: the raw phone signals, the core's
calibrated signals, the rule-based events, and the *true* vehicle motion.
Labels come only from the true motion, so both the rule-based detector and
the ML model are scored against the same independent ground truth.

Windows: 2 s long, 0.5 s hop. A window gets the label of a true event that
overlaps its central second (priority: the event that exceeds its threshold
the most); otherwise NORMAL.
"""
from dataclasses import dataclass

import numpy as np

from ..core import events as E
from ..core.detectors import DetectorConfig
from ..core.filters import EMA, HighPass
from ..sim import Simulation
from ..sim.autopilot import Autopilot

CLASSES = ["NORMAL", E.HARSH_BRAKE, E.HARSH_ACCEL, E.HARSH_TURN, E.BUMP, E.POTHOLE]
WIN_S, HOP_S = 2.0, 0.5


@dataclass
class Trip:
    profile: str
    mount: str
    seed: int
    fs: float
    t: np.ndarray
    raw: np.ndarray        # (n, 6) phone acc xyz + gyro xyz (gravity included)
    cal: np.ndarray        # (n, 4) forward, left, vertical(high-passed), yaw rate
    speed: np.ndarray      # (n,) core's speed estimate, m/s
    cal_done: np.ndarray   # (n,) bool
    truth: np.ndarray      # (n, 6) x, y, v, a_long, a_lat, a_vert
    rule_events: list
    features: list         # road features (for bump/pothole ground truth)


def simulate_trip(profile, mount, seed, seconds=300.0) -> Trip:
    sim = Simulation(seed=seed, mount=mount, record=False)
    ap = Autopilot(profile, sim.rng)
    n = int(seconds * sim.fs)
    raw = np.zeros((n, 6)); cal = np.zeros((n, 4)); truth = np.zeros((n, 6))
    t = np.zeros(n); speed = np.zeros(n); done = np.zeros(n, bool)
    p = sim.pipeline
    for k in range(n):
        th, st = ap.control(sim, sim.dt) if sim.t > 3.0 else (0.0, 0.0)
        sim.step(th, st)
        s, v = sim.last_sample, sim.vehicle
        t[k] = sim.t
        raw[k, :3], raw[k, 3:] = s["acc"], s["gyro"]
        sg = p.signals
        cal[k] = (sg["lon_raw"], sg["lat_raw"], sg["vert"], sg["yaw"])
        speed[k] = p.speed_ms
        done[k] = p.cal.stage == "done"
        truth[k] = (v.x, v.y, v.v, v.a_long, v.a_lat, v.a_vert)
    return Trip(profile, mount, seed, sim.fs, t, raw, cal, speed, done, truth,
                list(p.events), list(sim.road.features))


# ------------------------------------------------------------------ labels
def _runs(mask, t, min_len_s, fs):
    """Contiguous True runs of at least min_len_s, as (start, end) times."""
    out = []
    m = np.concatenate([[False], mask, [False]])
    d = np.diff(m.astype(int))
    starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
    for a, b in zip(starts, ends):
        if (b - a) / fs >= min_len_s:
            out.append((t[a], t[b - 1]))
    return out


def true_intervals(trip: Trip, cfg: DetectorConfig = DetectorConfig()):
    """[(start, end, kind, ratio)] from the true vehicle motion."""
    fs, t = trip.fs, trip.t
    sm_lon, sm_lat, hp = EMA(cfg.smoothing_tau_s, 1 / fs), EMA(cfg.smoothing_tau_s, 1 / fs), HighPass(0.5, 1 / fs)
    lon = np.array([sm_lon(a) for a in trip.truth[:, 3]])
    lat = np.array([sm_lat(a) for a in trip.truth[:, 4]])
    vert = np.array([hp(a) for a in trip.truth[:, 5]])
    out = []
    for kind, sig, thr in ((E.HARSH_BRAKE, -lon, cfg.harsh_brake), (E.HARSH_ACCEL, lon, cfg.harsh_accel),
                           (E.HARSH_TURN, np.abs(lat), cfg.harsh_turn)):
        for a, b in _runs(sig >= thr, t, cfg.min_duration_s, fs):
            sel = (t >= a) & (t <= b)
            out.append((a, b, kind, sig[sel].max() / thr))
    # vertical: jolts while moving, classified by the road feature under the car
    jolt = (np.abs(vert) >= cfg.vertical) & (trip.truth[:, 2] > 1.0)
    for a, b in _runs(jolt, t, 0.0, fs):
        i = np.searchsorted(t, a)
        x, y = trip.truth[i, 0], trip.truth[i, 1]
        near = min(trip.features, key=lambda f: (f.x - x) ** 2 + (f.y - y) ** 2)
        if (near.x - x) ** 2 + (near.y - y) ** 2 > 8 ** 2:
            continue
        kind = E.BUMP if near.kind == "hump" else E.POTHOLE
        sel = (t >= a - 0.3) & (t <= b + 0.3)
        out.append((a - 0.1, b + 0.3, kind, np.abs(vert[sel]).max() / cfg.vertical))
    return _merge_vertical(out)


def _merge_vertical(intervals):
    """A hump hits both axles: merge vertical intervals closer than 0.8 s."""
    vert = sorted([iv for iv in intervals if iv[2] in (E.BUMP, E.POTHOLE)])
    other = [iv for iv in intervals if iv[2] not in (E.BUMP, E.POTHOLE)]
    merged = []
    for iv in vert:
        if merged and iv[2] == merged[-1][2] and iv[0] - merged[-1][1] < 0.8:
            a, b, k, r = merged[-1]
            merged[-1] = (a, max(b, iv[1]), k, max(r, iv[3]))
        else:
            merged.append(iv)
    return other + merged


def rule_intervals(trip: Trip, cfg: DetectorConfig = DetectorConfig()):
    out = []
    for e in trip.rule_events:
        if e.kind not in CLASSES:
            continue
        if e.kind in (E.BUMP, E.POTHOLE):
            out.append((e.t - 0.1, e.t + 0.3, e.kind, e.severity))
        else:
            end = e.t + e.duration if e.duration > 0 else trip.t[-1]
            out.append((e.t - cfg.min_duration_s, end, e.kind, e.severity))
    return out


def window_label(intervals, c0, c1, min_overlap=0.15):
    best, best_r = "NORMAL", 0.0
    for a, b, kind, r in intervals:
        ov = min(b, c1) - max(a, c0)
        if ov >= min(min_overlap, b - a) and ov > 0 and r > best_r:
            best, best_r = kind, r
    return best


# ---------------------------------------------------------------- features
def _stats(x):
    """mean, std, min, max, max |jerk| per column."""
    j = np.abs(np.diff(x, axis=0)).max(axis=0)
    return np.concatenate([x.mean(0), x.std(0), x.min(0), x.max(0), j])


def window_features(trip: Trip, i0, i1):
    sp = trip.speed[i0:i1]
    extra = [sp.mean(), sp[-1] - sp[0]]
    return (np.concatenate([_stats(trip.cal[i0:i1]), extra]),
            np.concatenate([_stats(trip.raw[i0:i1]), extra]))


def CAL_NAMES():
    cols = ["forward", "left", "vertical", "yaw"]
    return [f"{c}_{s}" for s in ("mean", "std", "min", "max", "jerk") for c in cols] + ["speed_mean", "speed_delta"]


def windows(trip: Trip):
    """Yield (X_cal, X_raw, y_true, y_rule) for every fully calibrated window."""
    fs = trip.fs
    w, h = int(WIN_S * fs), int(HOP_S * fs)
    truth_iv, rule_iv = true_intervals(trip), rule_intervals(trip)
    Xc, Xr, yt, yr = [], [], [], []
    for i0 in range(0, len(trip.t) - w, h):
        i1 = i0 + w
        if not trip.cal_done[i0]:
            continue
        c0 = trip.t[i0] + (WIN_S - 1.0) / 2
        c1 = c0 + 1.0
        fc, fr = window_features(trip, i0, i1)
        Xc.append(fc); Xr.append(fr)
        yt.append(window_label(truth_iv, c0, c1))
        yr.append(window_label(rule_iv, c0, c1))
    return np.array(Xc), np.array(Xr), np.array(yt), np.array(yr)
