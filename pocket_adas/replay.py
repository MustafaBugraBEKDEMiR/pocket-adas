"""Replay a recorded drive through the ADAS core and write a trip report.

    python -m pocket_adas.replay recording.zip                  # Sensor Logger export
    python -m pocket_adas.replay recording_folder/ --map maps/ankara_sihhiye.osm
    python -m pocket_adas.replay trips/<simulated trip>/sensors.csv

Accepted inputs
* **Sensor Logger** (iOS/Android) export, as a .zip or an unzipped folder with
  ``TotalAcceleration.csv`` *or* ``Accelerometer.csv`` (+ ``Gravity.csv``),
  ``Gyroscope.csv`` and ``Location.csv``.
* A ``sensors.csv`` written by the simulator.

Units are detected from the data (g vs m/s^2). The core only needs the *gravity-
including* acceleration; a global sign flip between platforms does not matter
because the mount calibration learns the axes from the data itself.
With ``--map`` the speed limit at each GPS fix is looked up in the OSM file.
"""
import argparse
import csv
import io
import json
import os
import time
import zipfile

import numpy as np

from .core import ADASPipeline
from .core.geo import latlon_to_xy
from .core.scoring import event_counts

G = 9.80665
FS = 50.0


# ------------------------------------------------------------------ reading
def _open_source(path):
    """Return {lower-case file name: text} for a zip, a folder or a single csv."""
    files = {}
    if os.path.isdir(path):
        for name in os.listdir(path):
            if name.lower().endswith(".csv"):
                with open(os.path.join(path, name), encoding="utf-8-sig") as f:
                    files[name.lower()] = f.read()
    elif path.lower().endswith(".zip"):
        with zipfile.ZipFile(path) as z:
            for name in z.namelist():
                if name.lower().endswith(".csv"):
                    files[os.path.basename(name).lower()] = z.read(name).decode("utf-8-sig")
    else:
        with open(path, encoding="utf-8-sig") as f:
            files[os.path.basename(path).lower()] = f.read()
    return files


def _table(text):
    rows = list(csv.DictReader(io.StringIO(text)))
    return rows


def _col(rows, *names):
    for n in names:
        if rows and n in rows[0]:
            out = []
            for r in rows:
                try:
                    out.append(float(r[n]))
                except (TypeError, ValueError):
                    out.append(np.nan)
            return np.array(out)
    return None


def _time(rows):
    t = _col(rows, "seconds_elapsed")
    if t is None:
        ns = _col(rows, "time")
        t = (ns - ns[0]) / 1e9
    return t


def _xyz(rows):
    return np.column_stack([_col(rows, "x"), _col(rows, "y"), _col(rows, "z")])


def load_recording(path):
    """-> dict(t, acc (n,3), gyro (n,3), gps list of (t, lat, lon, speed), source)"""
    files = _open_source(path)
    if "sensors.csv" in files and len(files) == 1:  # simulator log
        rows = _table(files["sensors.csv"])
        t = _col(rows, "seconds_elapsed")
        acc = np.column_stack([_col(rows, "acc_x"), _col(rows, "acc_y"), _col(rows, "acc_z")])
        gyro = np.column_stack([_col(rows, "gyro_x"), _col(rows, "gyro_y"), _col(rows, "gyro_z")])
        lat, lon, spd = _col(rows, "latitude"), _col(rows, "longitude"), _col(rows, "speed")
        gps = [(t[i], lat[i], lon[i], spd[i]) for i in range(len(t)) if not np.isnan(lat[i])]
        return dict(t=t, acc=acc, gyro=gyro, gps=gps, source="simulator")

    def find(*names):
        for n in names:
            if n in files:
                return _table(files[n])
        return None

    total = find("totalacceleration.csv")
    if total is not None:
        ta, acc = _time(total), _xyz(total)
    else:
        lin = find("accelerometer.csv")
        if lin is None:
            raise SystemExit("No TotalAcceleration.csv or Accelerometer.csv found")
        ta, acc = _time(lin), _xyz(lin)
        grav = find("gravity.csv")
        if grav is not None:
            tg, g = _time(grav), _xyz(grav)
            acc = acc + np.column_stack([np.interp(ta, tg, g[:, k]) for k in range(3)])
    norm = np.nanmedian(np.linalg.norm(acc, axis=1))
    if norm < 3.0:          # recorded in g
        acc = acc * G
        norm *= G
    if norm < 5.0:
        raise SystemExit("Acceleration has no gravity component; export TotalAcceleration or Gravity too")

    gy = find("gyroscope.csv")
    if gy is None:
        raise SystemExit("Gyroscope.csv not found")
    tg, gyro = _time(gy), _xyz(gy)

    gps = []
    loc = find("location.csv")
    if loc is not None:
        tl = _time(loc)
        lat, lon = _col(loc, "latitude"), _col(loc, "longitude")
        spd = _col(loc, "speed")
        for i in range(len(tl)):
            s = spd[i] if spd is not None else np.nan
            if (np.isnan(s) or s < 0) and i > 0:  # derive from positions
                x0, y0 = latlon_to_xy(lat[i - 1], lon[i - 1], (lat[0], lon[0]))
                x1, y1 = latlon_to_xy(lat[i], lon[i], (lat[0], lon[0]))
                s = np.hypot(x1 - x0, y1 - y0) / max(tl[i] - tl[i - 1], 1e-3)
            gps.append((tl[i], lat[i], lon[i], 0.0 if np.isnan(s) else max(0.0, s)))

    # resample IMU to a common 50 Hz clock
    t0, t1 = max(ta[0], tg[0]), min(ta[-1], tg[-1])
    t = np.arange(t0, t1, 1.0 / FS)
    acc_r = np.column_stack([np.interp(t, ta, acc[:, k]) for k in range(3)])
    gyro_r = np.column_stack([np.interp(t, tg, gyro[:, k]) for k in range(3)])
    return dict(t=t - t0, acc=acc_r, gyro=gyro_r,
                gps=[(g[0] - t0, *g[1:]) for g in gps if t0 <= g[0] <= t1], source="sensor logger")


# ------------------------------------------------------------------ replay
def replay(rec, world=None):
    limit_fn = None
    if world is not None:
        def limit_fn(g):
            return world.speed_limit_at(*latlon_to_xy(g["lat"], g["lon"], world.origin))
    p = ADASPipeline(FS, speed_limit_fn=limit_fn)
    gps = sorted(rec["gps"])
    gi = 0
    for i, t in enumerate(rec["t"]):
        fix = None
        while gi < len(gps) and gps[gi][0] <= t:
            _, lat, lon, spd = gps[gi]
            fix = {"lat": lat, "lon": lon, "speed": spd}
            gi += 1
        p.update(float(t) + 1e-3, rec["acc"][i], rec["gyro"][i], fix)
    return p


def write_trip(p, rec, out_dir, tag, world_name=""):
    from .report import build_report
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "sensors.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seconds_elapsed", "latitude", "longitude", "speed"])
        for t, lat, lon, spd in rec["gps"]:
            w.writerow([round(t, 3), lat, lon, round(spd, 3)])
    keys = ["t", "kind", "peak", "severity", "speed_kmh", "lat", "lon", "duration", "limit_kmh"]
    with open(os.path.join(out_dir, "events.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for e in p.events:
            w.writerow(e.to_dict())
    summary = {"tag": tag, "duration_s": round(float(rec["t"][-1]), 1), "distance_m": round(p.distance_m, 1),
               "score": p.score, "events": event_counts(p.events), "mount": "unknown (recorded)",
               "calibration": p.cal.stage, "world": world_name, "source": rec["source"]}
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    build_report(out_dir)
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recording")
    ap.add_argument("--map", default=None, help="OSM file for speed limits")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rec = load_recording(args.recording)
    world = None
    if args.map:
        from .sim.osm_world import OsmWorld
        world = OsmWorld(args.map)
    p = replay(rec, world)
    out = args.out or os.path.join("trips", time.strftime("%Y%m%d_%H%M%S") + "_replay")
    s = write_trip(p, rec, out, "replay", world.name if world else "")
    print(f"{rec['source']}: {s['duration_s']:.0f} s, {s['distance_m'] / 1000:.2f} km, "
          f"calibration={s['calibration']}, score={s['score']}")
    print("events:", s["events"])
    print("report:", os.path.join(out, "report.html"))


if __name__ == "__main__":
    main()
