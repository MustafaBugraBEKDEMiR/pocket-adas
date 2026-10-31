import numpy as np
import pytest

from pocket_adas.core import ADASPipeline, events as E
from pocket_adas.sim import Simulation
from pocket_adas.sim.autopilot import Autopilot

G = 9.80665


def drive(profile, mount, seed=1, seconds=180):
    sim = Simulation(seed=seed, mount=mount, record=False)
    ap = Autopilot(profile, sim.rng)
    for _ in range(int(seconds * sim.fs)):
        th, st = ap.control(sim, sim.dt) if sim.t > 3 else (0.0, 0.0)
        sim.step(th, st)
    return sim


@pytest.mark.parametrize("mount", ["dashboard", "flat", "random"])
def test_calibration_finds_forward_axis(mount):
    sim = drive("calm", mount, seconds=40)
    cal = sim.pipeline.cal
    assert cal.stage == "done"
    true_forward = sim.phone.R[:, 0]          # vehicle x-axis in phone coordinates
    angle = np.degrees(np.arccos(np.clip(np.dot(cal.forward, true_forward), -1, 1)))
    assert angle < 8


@pytest.mark.parametrize("mount", ["dashboard", "random"])
def test_calm_driver_gets_no_behaviour_warnings(mount):
    sim = drive("calm", mount)
    kinds = {e.kind for e in sim.pipeline.events}
    assert kinds <= {E.POTHOLE}             # potholes are road defects, not driver faults
    assert sim.pipeline.score >= 90


def test_aggressive_driver_is_caught():
    sim = drive("aggressive", "random")
    kinds = {e.kind for e in sim.pipeline.events}
    assert {E.HARSH_BRAKE, E.HARSH_TURN, E.SPEEDING} <= kinds
    assert sim.pipeline.score < 50


def test_still_phone_is_safe_at_any_angle():
    rng = np.random.default_rng(0)
    for _ in range(5):
        g = rng.normal(size=3)
        g = g / np.linalg.norm(g) * G
        p = ADASPipeline()
        for k in range(300):
            p.update(k / 50, g + rng.normal(0, 0.03, 3), rng.normal(0, 0.003, 3),
                     {"lat": 0, "lon": 0, "speed": 0.0} if k % 50 == 0 else None)
        assert p.status.level == E.SAFE
        assert not p.events
