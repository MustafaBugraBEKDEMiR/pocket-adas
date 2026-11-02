"""Live mode: stream a real phone into the ADAS core over the local network.

    python -m pocket_adas.live --port 8000 [--map maps/ankara_sihhiye.osm]

On the phone, open the free **Sensor Logger** app, enable *HTTP Push* with the URL
``http://<laptop-ip>:8000/data`` (gear icon on the Logger page) and switch on Accelerometer + Gravity (or
Accelerometer Uncalibrated), Gyroscope and Location. Turning on *Settings > Sensor
Configuration > Standardise Units & Frames* makes iOS and Android send m/s^2;
otherwise iOS sends accelerometer values in g (use ``--accel-unit g``). The laptop prints the driver status as it
changes; ``GET /status`` returns it as JSON. Ctrl-C ends the trip and writes the
same trip folder and report as ``pocket_adas.replay``.

This replaces the 2023 path (phone -> API Gateway -> Lambda -> Kafka on EC2 ->
consumer, see ``legacy/2023``): no cloud account, no mobile data, and the samples
go through the same calibrated core as the simulator and the replay mode.
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import time
from bisect import bisect_right
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from .core import ADASPipeline
from .core.geo import latlon_to_xy

G = 9.80665
FS = 50.0

# Sensor Logger stream names -> internal kinds (awesome-sensor-logger PUSHING.md / UNITS.md)
_KINDS = {
    "totalacceleration": "total",           # gravity included (CSV-export name, kept for safety)
    "accelerometeruncalibrated": "uncal",   # raw reading, gravity included
    "accelerometer": "lin",                 # user acceleration, gravity removed
    "gravity": "grav",                      # always m/s^2
    "gyroscope": "gyro",                    # rad/s
    "location": "gps",                      # speed in m/s, negative = invalid
}
_ACC_PRIORITY = ("total", "uncal")


def parse_push(body) -> list[dict]:
    """Parse one Sensor Logger HTTP-push message.

    ``body`` is the raw request body (bytes/str) or the decoded dict. Returns a list
    of ``{"kind", "t" [s since Unix epoch], "v"}``; unknown streams are skipped.
    """
    if isinstance(body, (bytes, str)):
        body = json.loads(body)
    out = []
    for item in body.get("payload", []):
        kind = _KINDS.get(str(item.get("name", "")).lower())
        vals = item.get("values") or {}
        if kind is None or "time" not in item:
            continue
        t = float(item["time"]) / 1e9          # Sensor Logger sends nanoseconds
        if kind == "gps":
            if vals.get("latitude") is None or vals.get("longitude") is None:
                continue
            spd = vals.get("speed")
            v = (float(vals["latitude"]), float(vals["longitude"]),
                 float(spd) if spd is not None else -1.0)
        else:
            try:
                v = (float(vals["x"]), float(vals["y"]), float(vals["z"]))
            except (KeyError, TypeError, ValueError):
                continue
        out.append({"kind": kind, "t": t, "v": v})
    return out


class _Stream:
    def __init__(self):
        self.t, self.v = [], []

    def add(self, t, v):
        if self.t and t < self.t[-1]:          # late packet: insert in order
            i = bisect_right(self.t, t)
            self.t.insert(i, t)
            self.v.insert(i, v)
        else:
            self.t.append(t)
            self.v.append(v)

    def at(self, tq):
        tt = np.asarray(self.t)
        vv = np.asarray(self.v)
        return np.array([np.interp(tq, tt, vv[:, k]) for k in range(vv.shape[1])])

    @property
    def end(self):
        return self.t[-1] if self.t else None


class LiveSession:
    """Turns irregular pushed samples into a 50 Hz stream for :class:`ADASPipeline`.

    Uses the same resampling (linear interpolation onto a 50 Hz grid) and GPS hand-
    over rule as ``replay.py``, so a live run and a replay of the same data agree.
    The grid only advances to the newest time that every needed stream has reached.
    Out-of-order messages are inserted in time order; a sample older than the
    already processed grid is kept for the trip file but cannot change past output.
    """

    def __init__(self, world=None, fs: float = FS, accel_unit: str = "ms2"):
        """``accel_unit`` applies only to the gravity-free ``accelerometer`` stream,
        whose unit cannot be inferred from the data ("ms2" or "g"); streams that
        include gravity are detected automatically from their magnitude."""
        self.fs = fs
        self.lin_scale = G if accel_unit == "g" else 1.0
        self.world = world
        limit_fn = None
        if world is not None:
            def limit_fn(g):
                return world.speed_limit_at(*latlon_to_xy(g["lat"], g["lon"], world.origin))
        self.pipeline = ADASPipeline(fs, speed_limit_fn=limit_fn)
        self.s = {k: _Stream() for k in ("total", "uncal", "lin", "grav", "gyro")}
        self.gps = []                    # (t_abs, lat, lon, speed)
        self._gi = 0
        self._k = 0                      # next grid index
        self.t0 = None
        self.scale = None                # 1.0 (m/s^2) or G for the gravity-including stream
        self.rec_t, self.rec_acc, self.rec_gyro = [], [], []
        self.lock = threading.Lock()

    # ---------------------------------------------------------------- input
    def ingest(self, samples):
        with self.lock:
            for s in samples:
                if s["kind"] == "gps":
                    self.gps.append((s["t"], *s["v"]))
                else:
                    self.s[s["kind"]].add(s["t"], s["v"])
            self.gps.sort()
            return self._process()

    def _acc_stream(self):
        for k in _ACC_PRIORITY:
            if self.s[k].t:
                return k
        if self.s["lin"].t and self.s["grav"].t:
            return "lin"
        return None

    def _acc_at(self, t):
        src = self._acc_stream()
        if src == "lin":
            return self.s["lin"].at(t) * self.lin_scale + self.s["grav"].at(t) * self.scale
        return self.s[src].at(t) * self.scale

    def _ready_until(self):
        src = self._acc_stream()
        if src is None or not self.s["gyro"].t:
            return None
        ends = [self.s[src].end, self.s["gyro"].end]
        if src == "lin":
            ends.append(self.s["grav"].end)
        return min(ends)

    def _speed(self, i):
        t, lat, lon, spd = self.gps[i]
        if spd >= 0 or i == 0:
            return max(0.0, spd)
        t0, lat0, lon0, _ = self.gps[i - 1]
        x0, y0 = latlon_to_xy(lat0, lon0, (lat0, lon0))
        x1, y1 = latlon_to_xy(lat, lon, (lat0, lon0))
        return float(np.hypot(x1 - x0, y1 - y0) / max(t - t0, 1e-3))

    def _process(self):
        new = []
        end = self._ready_until()
        if end is None:
            return new
        if self.t0 is None:
            src = self._acc_stream()
            self.t0 = max(self.s[src].t[0], self.s["gyro"].t[0])
        if self.scale is None:
            src = self._acc_stream()
            ref = "grav" if src == "lin" else src     # the stream that carries gravity
            if len(self.s[ref].t) < 25:
                return new
            norm = float(np.nanmedian(np.linalg.norm(np.asarray(self.s[ref].v), axis=1)))
            self.scale = G if norm < 3.0 else 1.0
        while True:
            t = self.t0 + self._k / self.fs
            if t > end:
                break
            fix = None
            while self._gi < len(self.gps) and self.gps[self._gi][0] <= t:
                _, lat, lon, _ = self.gps[self._gi]
                fix = {"lat": lat, "lon": lon, "speed": self._speed(self._gi)}
                self._gi += 1
            acc, gyro = self._acc_at(t), self.s["gyro"].at(t)
            new += self.pipeline.update(t - self.t0 + 1e-3, acc, gyro, fix)
            self.rec_t.append(t - self.t0)
            self.rec_acc.append(acc)
            self.rec_gyro.append(gyro)
            self._k += 1
        return new

    # --------------------------------------------------------------- output
    def status(self) -> dict:
        p = self.pipeline
        return {"level": p.status.level, "message": p.status.message, "score": p.score,
                "speed_kmh": round(p.speed_kmh, 1), "limit_kmh": p.limit_kmh,
                "calibration": p.cal.stage, "events": len(p.events),
                "seconds": round(self.rec_t[-1], 1) if self.rec_t else 0.0}

    def recording(self) -> dict:
        t0 = self.t0 or 0.0
        return dict(t=np.asarray(self.rec_t), acc=np.asarray(self.rec_acc),
                    gyro=np.asarray(self.rec_gyro), source="sensor logger (live)",
                    gps=[(g[0] - t0, g[1], g[2], self._speed(i)) for i, g in enumerate(self.gps)
                         if self.rec_t and t0 <= g[0] <= t0 + self.rec_t[-1]])

    def finish(self, out_dir):
        from .replay import write_trip
        with self.lock:
            return write_trip(self.pipeline, self.recording(), out_dir, "live",
                              self.world.name if self.world else "")


def make_server(session: LiveSession, host="0.0.0.0", port=8000, verbose=True):
    last = {"msg": None}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):          # keep the console for status lines
            pass

        def _reply(self, code, obj):
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            try:
                events = session.ingest(parse_push(self.rfile.read(n)))
            except (ValueError, json.JSONDecodeError) as exc:
                return self._reply(400, {"error": str(exc)})
            st = session.status()
            if verbose:
                for e in events:
                    print(f"  {st['seconds']:7.1f} s  {e.kind:<14} peak {e.peak:5.2f}")
                if st["message"] != last["msg"]:
                    print(f"[{st['level']}] {st['message']}  ({st['speed_kmh']} km/h, score {st['score']})")
                    last["msg"] = st["message"]
            self._reply(200, {"ok": True, **st})

        def do_GET(self):
            self._reply(200, session.status())

    return ThreadingHTTPServer((host, port), Handler)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--map", default=None, help="OSM file for speed limits")
    ap.add_argument("--out", default=None)
    ap.add_argument("--accel-unit", choices=["ms2", "g"], default="ms2",
                    help="unit of the gravity-free 'accelerometer' stream (iOS default: g)")
    args = ap.parse_args()
    world = None
    if args.map:
        from .sim.osm_world import OsmWorld
        world = OsmWorld(args.map)
    session = LiveSession(world, accel_unit=args.accel_unit)
    srv = make_server(session, args.host, args.port)
    print(f"Listening on http://{args.host}:{args.port}/data  (Ctrl-C to end the trip)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
    if not session.rec_t:
        print("no data received")
        return
    out = args.out or os.path.join("trips", time.strftime("%Y%m%d_%H%M%S") + "_live")
    s = session.finish(out)
    print(f"trip: {s['duration_s']:.0f} s, {s['distance_m'] / 1000:.2f} km, score={s['score']}")
    print("events:", s["events"])
    print("report:", os.path.join(out, "report.html"))


if __name__ == "__main__":
    main()
