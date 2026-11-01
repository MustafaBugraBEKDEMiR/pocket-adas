# Pocket ADAS: Technical Report

*Smartphone-sensor driver assistance, evaluated in a simulator on synthetic and OpenStreetMap roads.*
Mustafa Buğra Bekdemir · 2026 revision of a 2023 Geomatics Engineering graduation project (Hacettepe University)

---

## 1. Problem

Advanced driver-assistance systems (ADAS) such as harsh-braking or speeding warnings are standard in
expensive cars and rare in low-segment ones. Almost every driver, though, carries a smartphone with
an accelerometer, a gyroscope and GPS. The question is how much of a driving coach can be built from
those sensors alone. The target is a phone app that warns the driver in real time, works offline,
and does not care how the phone is mounted.

The same idea runs a commercial market. Smartphone-telematics providers such as Cambridge Mobile
Telematics (DriveWell) and Zendrive (acquired by Intuit in 2024) score drivers for insurers. In Turkey,
insurer apps such as Groupama Kaskopilot and fleet systems such as Arvento and Mobiliz rate harsh
acceleration, braking and cornering. This project does not compete with them. It rebuilds the core
technical pieces in the open and measures them.

## 2. What went wrong in 2023

The 2023 prototype streamed raw accelerometer samples from a phone app (Sensor Logger) through AWS
API Gateway, Lambda and Kafka to a Python consumer. The consumer beeped when `|x| + |y| + |z| ≥ 15 m/s²`.
A Keras classifier trained on the Kaggle *Driving Behavior* dataset was dropped from the live loop
because its predictions were unstable.

The root cause of both problems was the same: **the signal was never put into the car's frame.**

* Raw accelerometer values contain gravity (≈ 9.81 m/s²). Depending on the phone's tilt, a car
  standing still reads anywhere from 9.8 to 17 on `|x|+|y|+|z|`. A 15 m/s² threshold therefore fires on
  how the phone is held, not on how the car moves.
* A model trained on raw phone axes learns the mounting of the phones in its training set.
  Section 6 measures this directly.

## 3. System

```
accelerometer 50 Hz ─┐   ┌───────────── ADAS core (NumPy only) ─────────────┐
gyroscope     50 Hz ─┼──▶│ mount calibration → car-frame acceleration        │──▶ SAFE / WARNING / DANGER
GPS            1 Hz ─┘   │ smoothing → detectors → speed-limit lookup → score │    events + trip score
                         └────────────────────────────────────────────────────┘
```

The core (`pocket_adas/core`) has no knowledge of where samples come from. The same code runs on
live simulator data, on replayed recordings, and (in principle) on a phone.

### 3.1 Automatic mount calibration

Let **a** be the accelerometer reading (specific force, phone frame) and **ω** the gyroscope reading.

1. **Up axis.** While the phone is still (per-axis standard deviation < 0.2 m/s² over 1.5 s,
   mean |ω| < 0.04 rad/s, GPS speed < 0.8 m/s, |mean **a**| ≈ g), the mean reading is the gravity
   vector **g**. Then **u** = **g** / |**g**|, and the linear acceleration is **a**ₗ = **a** − **g**.
2. **Forward axis.** Between consecutive GPS fixes, the speed change Δv/Δt is the car's mean
   longitudinal acceleration. For intervals that are straight (mean |ω·**u**| < 0.06 rad/s) and
   informative (|Δv/Δt| > 0.6 m/s²), the horizontal part **h** of the mean linear acceleration is
   accumulated as **S** += **h** · Δv/Δt. The forward axis **f** is **S** made orthogonal to **u** and
   normalised, available after three such intervals. Left = **u** × **f**.

This needs no user action, and a global sign flip of all axes (iOS vs. Android conventions) does not
matter. In the simulator the forward axis is found within about 6 s of driving, with errors of 0.1–3°,
for dashboard, flat and uniformly random mounts.

### 3.2 Detectors

Car-frame signals are smoothed with a 0.25 s first-order filter. The vertical signal is high-passed
(τ = 0.5 s).

| Event | Rule (default) |
|---|---|
| Harsh braking | forward ≤ −3.0 m/s² for ≥ 0.25 s |
| Harsh acceleration | forward ≥ +2.5 m/s² for ≥ 0.25 s |
| Sharp turn | \|lateral\| ≥ 3.0 m/s² for ≥ 0.25 s |
| Bump / pothole | \|vertical\| ≥ 3.5 m/s² while moving; the sign of the first swing separates a bump (up first) from a pothole (down first), following Fazeen et al. (2012) |
| Speeding | GPS speed > limit + 5 km/h for ≥ 2 s |
| Sudden manoeuvre | horizontal magnitude ≥ 3.5 m/s², used only until the forward axis is known |

An event becomes **DANGER** when its peak is ≥ 1.5× the threshold, or when speeding happens in a
zone of ≤ 30 km/h. Between GPS fixes, speed is dead-reckoned from the forward acceleration.
Trip score = 100 − Σ(penalty × severity) per km, with trips shorter than 2 km scored as 2 km.

## 4. Simulator

* **Vehicle:** kinematic bicycle model with steering-rate and tyre-grip limits, plus a quarter-car
  suspension (1.3 Hz, ζ = 0.45) excited by the road height at both axles.
* **Virtual phone:** rotates the true specific force (including gravity) into the phone frame for a
  dashboard, flat or random mount. It adds bias, white noise, speed-dependent vibration and a
  wandering 1 Hz GPS error.
* **Drivers:** pure-pursuit steering with a look-ahead speed planner. The planner uses the worst-case
  curvature within ±5 m and braking distance to slower points ahead.
  * *calm*: 0.9 × limit, lateral ≤ 1.6 m/s², slows to 15 km/h for humps
  * *normal*: 1.05 × limit, occasional mild swerves and brakes
  * *aggressive*: 1.3 × limit, late braking, ignores humps, frequent sudden brakes and swerves
* **Worlds:**
  * a 2 km test loop with 50/30/70 km/h zones, two humps and three potholes
  * real streets from an OpenStreetMap export:
    * the drivable ways become a directed graph (one-ways and roundabouts respected)
    * dead ends are pruned and routing is restricted to the largest strongly connected component
    * the speed limit is the way's `maxspeed` tag, or a Turkish urban default (50 km/h on through roads, 30 km/h on minor roads) when the tag is missing
    * roads within 60 m of a school become 30 km/h zones
    * humps come from `traffic_calming`; when a map has none, they are placed on some pedestrian crossings and marked *synthetic*
* **Outputs:**
  * `sensors.csv` (Sensor Logger-like) and `truth.csv`
  * `events.csv` and `summary.json`
  * `events.geojson` and `route.geojson`
  * `report.html`: a Leaflet map with the score and event counts
* **Replay:** `python -m pocket_adas.replay` runs any simulator log or Sensor Logger export (zip or
  folder) through the same core.
  * Units (g or m/s²) are detected from the data.
  * Linear acceleration and gravity are added back together when exported separately.
  * Speed limits come from an OSM file when one is given.
  * A synthetic iPhone-style export (g units, flipped axes, separate gravity file, jittered sampling)
    reproduces the live event counts exactly.

## 5. Ground truth and data set

Labels come from the **true** vehicle motion, never from the phone. The data set has 54 trips of
300 s: {calm, normal, aggressive} × {dashboard, flat, random mount} × 6 seeds. The trips are cut into
2 s windows with a 0.5 s hop; only windows after calibration is complete are used.

A window takes the label of the true event that overlaps its central second most strongly (or
NORMAL). True events are defined like the detectors, but on the noise-free car-frame signals.
Bumps and potholes are labelled by the road feature under the car.

Features per window:

* *car-axis* set: mean, standard deviation, min, max and maximum jerk of the forward, lateral,
  vertical and yaw signals, plus mean speed and speed change (22 features)
* *raw* set: the same statistics on the six raw phone axes (32 features)

The classifier is a random forest (300 trees, balanced class weights).

## 6. Results

Macro-F1 over {normal, harsh brake, harsh accel, sharp turn, bump, pothole}:

| Method | A. unseen trips | B. unseen mount (train dashboard + flat → test random) |
|---|---|---|
| Rule-based core (no training) | 0.91 | 0.90 |
| Random forest, car-axis features | **0.93** | **0.95** |
| Random forest, raw phone-axis features | 0.81 | 0.46 |

Full per-class results and confusion matrices: [`results/ml_results.md`](results/ml_results.md).

Findings:

1. **Calibration matters more than the model.** On an unseen mount, the raw-axis forest falls to 0.46.
   Its recall is 0.28 for harsh braking, 0.00 for bumps and 0.28 for potholes. The car-axis forest keeps 0.95. This is the
   quantitative explanation of the 2023 instability.
2. **The rules are a strong baseline.** They need no training data and reach ≥ 0.90. Their errors are
   mostly borderline false alarms around the thresholds, and some missed bumps and potholes
   (recall 0.80 and 0.83 on the unseen mount). The
   forest mainly improves the vertical events, so the rules stay the default and the forest is optional.
3. **Behaviour separates cleanly** (240 s, random mount):
   * test track: calm 97 (only road defects logged), normal 71, aggressive 0
   * Sıhhiye OSM streets: calm 100, normal 78, aggressive 0

## 7. Limitations

* **All numbers come from a simulator** whose sensor model I wrote, and the truth labels use the same
  thresholds as the rules. This favours the rules and says nothing about field accuracy. Real phones
  add things the simulator lacks: handling of the phone while driving, loose mounts, GPS dropouts in
  tunnels and urban canyons, and different sensor qualities.
* The OSM speed limits are mostly defaults, because the sample area has almost no `maxspeed` tags,
  and the humps in the sample map are synthetic.
* There is no phone-use (distraction) detection, which commercial systems treat as a major risk factor.
* There is no on-device app yet. The core is NumPy-only so that it can run in Pydroid 3, but live
  sensor access on the phone is not implemented.

## 8. Next steps

1. Record real drives with Sensor Logger, replay them, and hand-label a few events to measure field
   precision and recall.
2. Retune the thresholds on real data, possibly per vehicle class.
3. Build a thin phone front-end (Pydroid 3 + plyer, or a small Kotlin app) around the same core.
4. Aggregate bump and pothole events from many trips into a road-condition map (GeoJSON / PostGIS).

## References

1. Fazeen, M., Gozick, B., Dantu, R., Bhukhiya, M., González, M. C. (2012). Safe driving using mobile phones. *IEEE Trans. Intelligent Transportation Systems*.
2. Vaitkus, V., Lengvenis, P., Žylius, G. (2014). Driving style classification using long-term accelerometer information.
3. Goregaonkar, R. K., Bhosale, S. A. (2014). Assistance to driver and monitoring the accidents on road by using three axis accelerometer and GPS system.
4. Whipple, J., Arensman, W., Boler, M. S. (2009). A public safety application of GPS-enabled smartphones and the Android operating system.
5. Paefgen, J., Staake, T., Thiesse, F. (2013). Evaluation and aggregation of pay-as-you-drive insurance rate factors: a classification analysis approach. *Decision Support Systems*.
6. Cojocaru, I. et al. (2022). *Driving Behavior* dataset, Kaggle (used in the 2023 prototype).
7. OpenStreetMap contributors. Map data under the Open Database License.
