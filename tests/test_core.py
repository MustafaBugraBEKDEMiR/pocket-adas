import os

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


def test_ml_labels_from_ground_truth():
    from pocket_adas.ml.dataset import simulate_trip, windows
    trip = simulate_trip("aggressive", "random", seed=0, seconds=90)
    Xc, Xr, yt, yr = windows(trip)
    assert len(Xc) == len(yt) == len(yr) > 100
    assert {E.HARSH_BRAKE, E.HARSH_TURN} <= set(yt)
    assert (yt == yr).mean() > 0.85   # rules agree with ground truth on most windows


def test_osm_world_routes_and_detects():
    import os
    from pocket_adas.sim.osm_world import OsmWorld
    path = os.path.join(os.path.dirname(__file__), "..", "maps", "ankara_sihhiye.osm")
    world = OsmWorld(path)
    assert len(world.routable) > 50
    assert world.speed_limit_at(*world.start_pose(np.random.default_rng(0))[:2]) is not None
    sim = Simulation(seed=2, mount="random", world=world, record=False)
    ap = Autopilot("calm", sim.rng)
    for _ in range(120 * 50):
        th, st = ap.control(sim, sim.dt) if sim.t > 3 else (0.0, 0.0)
        sim.step(th, st)
    assert sim.pipeline.distance_m > 600
    assert sim.pipeline.cal.stage == "done"
    assert not [e for e in sim.pipeline.events if e.kind != E.POTHOLE]


def test_replay_matches_live_run(tmp_path):
    from pocket_adas.core.scoring import event_counts
    from pocket_adas.replay import load_recording, replay
    sim = Simulation(seed=1, mount="random")
    ap = Autopilot("aggressive", sim.rng)
    for _ in range(90 * 50):
        th, st = ap.control(sim, sim.dt) if sim.t > 3 else (0.0, 0.0)
        sim.step(th, st)
    d = sim.save_trip(out_dir=str(tmp_path), tag="t")
    p = replay(load_recording(os.path.join(d, "sensors.csv")))
    # no map was given to the replay, so speed limits (and SPEEDING) are unknown there
    motion = lambda evs: event_counts([e for e in evs if e.kind != E.SPEEDING])
    assert motion(p.events) == motion(sim.pipeline.events)
    assert os.path.exists(os.path.join(d, "report.html"))
