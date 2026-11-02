import json
import os
import threading
import urllib.request

import numpy as np

from pocket_adas.core import events as E
from pocket_adas.core.scoring import event_counts
from pocket_adas.live import LiveSession, make_server, parse_push
from pocket_adas.replay import load_recording, replay
from pocket_adas.sim import Simulation
from pocket_adas.sim.autopilot import Autopilot

EPOCH_NS = 1_700_000_000 * 10**9


def _sim_trip(tmp_path, seconds=90):
    sim = Simulation(seed=1, mount="random")
    ap = Autopilot("aggressive", sim.rng)
    for _ in range(int(seconds * 50)):
        th, st = ap.control(sim, sim.dt) if sim.t > 3 else (0.0, 0.0)
        sim.step(th, st)
    return sim.save_trip(out_dir=str(tmp_path), tag="t")


def _xyz(name, ns, v):
    return {"name": name, "time": ns, "values": {"x": v[0], "y": v[1], "z": v[2]}}


def _to_push_messages(rec, batch_s=1.0, mode="android"):
    """Recording -> list of Sensor Logger HTTP-push JSON bodies (in time order).

    mode="android": 'accelerometeruncalibrated' (gravity included, m/s^2).
    mode="ios":     'accelerometer' without gravity in g + 'gravity' in m/s^2,
                    i.e. the iOS default when "Standardise Units & Frames" is off.
    """
    items = []
    acc = rec["acc"]
    if mode == "ios":
        k = np.ones(50) / 50.0                       # 1-s moving average as gravity
        grav = np.column_stack([np.convolve(acc[:, j], k, mode="same") for j in range(3)])
        lin_g = (acc - grav) / 9.80665
    for i, t in enumerate(rec["t"]):
        ns = EPOCH_NS + int(round(t * 1e9))
        if mode == "ios":
            items.append((t, _xyz("accelerometer", ns, lin_g[i])))
            items.append((t, _xyz("gravity", ns, grav[i])))
        else:
            items.append((t, _xyz("accelerometeruncalibrated", ns, acc[i])))
        items.append((t, _xyz("gyroscope", ns, rec["gyro"][i])))
    for t, lat, lon, spd in rec["gps"]:
        items.append((t, {"name": "location", "time": EPOCH_NS + int(round(t * 1e9)),
                          "values": {"latitude": lat, "longitude": lon, "speed": spd}}))
    items.sort(key=lambda x: x[0])
    msgs, cur, edge = [], [], batch_s
    for t, it in items:
        if t >= edge and cur:
            msgs.append(json.dumps({"messageId": len(msgs), "payload": cur}))
            cur, edge = [], edge + batch_s
        cur.append(it)
    msgs.append(json.dumps({"messageId": len(msgs), "payload": cur}))
    return msgs


def _motion(evs):
    return event_counts([e for e in evs if e.kind != E.SPEEDING])


def test_parse_push_formats():
    body = {"payload": [
        {"name": "accelerometerUncalibrated", "time": EPOCH_NS, "values": {"x": 0.1, "y": 9.8, "z": 0.2}},
        {"name": "gyroscope", "time": EPOCH_NS + 20_000_000, "values": {"x": 0, "y": 0, "z": 0.01}},
        {"name": "location", "time": EPOCH_NS, "values": {"latitude": 39.93, "longitude": 32.85, "speed": 4.2}},
        {"name": "magnetometer", "time": EPOCH_NS, "values": {"x": 1, "y": 2, "z": 3}},
        {"name": "gyroscope", "values": {"x": 0, "y": 0, "z": 0}},          # no time -> skipped
    ]}
    out = parse_push(json.dumps(body).encode())
    assert [s["kind"] for s in out] == ["uncal", "gyro", "gps"]
    assert abs(out[1]["t"] - out[0]["t"] - 0.02) < 1e-6
    assert out[2]["v"] == (39.93, 32.85, 4.2)


def test_live_matches_replay(tmp_path):
    d = _sim_trip(tmp_path)
    rec = load_recording(os.path.join(d, "sensors.csv"))
    ref = replay(rec)
    s = LiveSession()
    for m in _to_push_messages(rec):
        s.ingest(parse_push(m))
    assert s.pipeline.cal.stage == "done"
    assert _motion(s.pipeline.events) == _motion(ref.events)
    assert sum(_motion(ref.events).values()) > 0
    summary = s.finish(str(tmp_path / "live"))
    assert os.path.exists(tmp_path / "live" / "report.html")
    assert summary["source"] == "sensor logger (live)"


def test_live_ios_streams_match_replay(tmp_path):
    d = _sim_trip(tmp_path)
    rec = load_recording(os.path.join(d, "sensors.csv"))
    ref = replay(rec)
    s = LiveSession(accel_unit="g")
    for m in _to_push_messages(rec, mode="ios"):
        s.ingest(parse_push(m))
    assert s.scale == 1.0                        # gravity stream detected as m/s^2
    assert _motion(s.pipeline.events) == _motion(ref.events)


def test_live_accepts_g_units_and_out_of_order(tmp_path):
    d = _sim_trip(tmp_path, seconds=40)
    rec = load_recording(os.path.join(d, "sensors.csv"))
    rec_g = dict(rec, acc=rec["acc"] / 9.80665)
    msgs = _to_push_messages(rec_g, batch_s=0.5)
    msgs[3], msgs[4] = msgs[4], msgs[3]          # one late packet (messageId out of order)
    s = LiveSession()
    for m in msgs:
        s.ingest(parse_push(m))
    assert s.scale > 9.0
    assert s.pipeline.cal.stage == "done"


def test_http_server_roundtrip(tmp_path):
    d = _sim_trip(tmp_path, seconds=20)
    rec = load_recording(os.path.join(d, "sensors.csv"))
    session = LiveSession()
    srv = make_server(session, "127.0.0.1", 0, verbose=False)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    url = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        for m in _to_push_messages(rec, batch_s=2.0):
            req = urllib.request.Request(url + "/data", data=m.encode(), method="POST",
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req) as r:
                assert json.loads(r.read())["ok"]
        with urllib.request.urlopen(url + "/status") as r:
            st = json.loads(r.read())
    finally:
        srv.shutdown()
        srv.server_close()
    assert st["seconds"] > 15
    assert st["level"] in (E.SAFE, E.WARNING, E.DANGER, E.CALIBRATING)
