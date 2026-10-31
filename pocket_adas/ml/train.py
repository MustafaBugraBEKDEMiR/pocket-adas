"""Generate simulator data, train classifiers and compare them with the rules.

    python -m pocket_adas.ml.train                # full run (~54 trips)
    python -m pocket_adas.ml.train --quick        # smaller, for a smoke test

Two experiments:

A. **Unseen trips** - train on seeds 0-3, test on seeds 4-5 (all mounts).
B. **Unseen mount** - train on dashboard + flat mounts, test on random mounts.
   This is where the 2023 idea (a model on raw phone axes) breaks down and
   the calibrated features keep working.

Each experiment compares: the rule-based detector of the core, a random forest
on calibrated (car-axis) features, and a random forest on raw phone-axis features.
"""
import argparse
import json
import os
import pickle
import time

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from .dataset import CLASSES, CAL_NAMES, simulate_trip, windows

PROFILES = ["calm", "normal", "aggressive"]
MOUNTS = ["dashboard", "flat", "random"]


def build(seeds, seconds, cache):
    if cache and os.path.exists(cache):
        with open(cache, "rb") as f:
            return pickle.load(f)
    rows = []
    t0 = time.time()
    for seed in seeds:
        for prof in PROFILES:
            for mount in MOUNTS:
                trip = simulate_trip(prof, mount, seed, seconds)
                Xc, Xr, yt, yr = windows(trip)
                rows.append(dict(seed=seed, profile=prof, mount=mount, Xc=Xc, Xr=Xr, yt=yt, yr=yr))
        print(f"  seed {seed} done ({time.time() - t0:.0f} s)")
    if cache:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        with open(cache, "wb") as f:
            pickle.dump(rows, f)
    return rows


def stack(rows, key):
    return np.concatenate([r[key] for r in rows])


def rf():
    return RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced_subsample",
                                  n_jobs=-1, random_state=0)


def evaluate(name, train, test):
    yt_tr, yt_te = stack(train, "yt"), stack(test, "yt")
    res = {"train_windows": int(len(yt_tr)), "test_windows": int(len(yt_te)), "models": {}}
    preds = {"rules": stack(test, "yr")}
    models = {}
    for feat, key in (("rf_calibrated", "Xc"), ("rf_raw_phone_axes", "Xr")):
        m = rf().fit(stack(train, key), yt_tr)
        preds[feat] = m.predict(stack(test, key))
        models[feat] = m
    labels = [c for c in CLASSES if c in set(yt_te) | set(yt_tr)]
    for k, p in preds.items():
        res["models"][k] = {
            "macro_f1": round(f1_score(yt_te, p, labels=labels, average="macro", zero_division=0), 3),
            "report": classification_report(yt_te, p, labels=labels, output_dict=True, zero_division=0),
            "confusion": confusion_matrix(yt_te, p, labels=labels).tolist(),
        }
    res["labels"] = labels
    print(f"\n== {name}: {len(yt_tr)} train / {len(yt_te)} test windows")
    for k, v in res["models"].items():
        print(f"  {k:<20} macro-F1 {v['macro_f1']:.3f}")
    return res, models


def plot(results, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    short = {"NORMAL": "normal", "HARSH_BRAKE": "brake", "HARSH_ACCEL": "accel",
             "HARSH_TURN": "turn", "BUMP": "bump", "POTHOLE": "pothole"}
    titles = {"rules": "Rule-based core", "rf_calibrated": "RF, car axes",
              "rf_raw_phone_axes": "RF, raw phone axes"}
    for exp, res in results.items():
        labels = [short[c] for c in res["labels"]]
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
        for ax, (k, v) in zip(axes, res["models"].items()):
            cm = np.array(v["confusion"], float)
            cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
            ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
            for i in range(len(labels)):
                for j in range(len(labels)):
                    ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=8,
                            color="white" if cmn[i, j] > 0.6 else "black")
            ax.set_xticks(range(len(labels)), labels, rotation=40, ha="right", fontsize=9)
            ax.set_yticks(range(len(labels)), labels, fontsize=9)
            ax.set_xlabel("predicted"); ax.set_ylabel("true")
            ax.set_title(f"{titles[k]}  (macro-F1 {v['macro_f1']:.2f})", fontsize=11)
        fig.suptitle({"A_unseen_trips": "A. Unseen trips (all mounts)",
                      "B_unseen_mount": "B. Unseen mount (train: dashboard+flat, test: random)"}[exp])
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"confusion_{exp}.png"), dpi=110)
        plt.close(fig)


def write_markdown(results, importances, out_dir, meta):
    names = {"rules": "Rule-based core (no training)", "rf_calibrated": "Random forest, car-axis features",
             "rf_raw_phone_axes": "Random forest, raw phone-axis features"}
    lines = ["# Rule-based detector vs. machine learning", "",
             f"Generated by `python -m pocket_adas.ml.train` on {meta['date']}: {meta['trips']} simulated trips "
             f"({', '.join(PROFILES)} drivers × {', '.join(MOUNTS)} mounts × {meta['seeds']} seeds, "
             f"{meta['seconds']:.0f} s each), 2 s windows with 0.5 s hop. Labels come from the true vehicle "
             "motion, not from the phone.", ""]
    for exp, res in results.items():
        title = {"A_unseen_trips": "A. Unseen trips (all mounts)",
                 "B_unseen_mount": "B. Unseen mount (train: dashboard + flat, test: random)"}[exp]
        lines += [f"## {title}", "", f"{res['train_windows']} training / {res['test_windows']} test windows.", "",
                  "| Method | Macro-F1 | " + " | ".join(f"{c} recall" for c in res["labels"][1:]) + " |",
                  "|---|---|" + "---|" * (len(res["labels"]) - 1)]
        for k, v in res["models"].items():
            rec = [f"{v['report'].get(c, {}).get('recall', 0):.2f}" for c in res["labels"][1:]]
            lines.append(f"| {names[k]} | **{v['macro_f1']:.2f}** | " + " | ".join(rec) + " |")
        lines += ["", f"![confusion matrices](confusion_{exp}.png)", ""]
    lines += ["## Most useful features (random forest, car axes)", ""]
    lines += [f"{i + 1}. `{n}` ({v:.3f})" for i, (n, v) in enumerate(importances[:8])]
    lines.append("")
    with open(os.path.join(out_dir, "ml_results.md"), "w") as f:
        f.write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seconds", type=float, default=300.0)
    ap.add_argument("--out", default="docs/results")
    ap.add_argument("--cache", default="data/windows.pkl")
    ap.add_argument("--save-model", default=None, help="path for the car-axis model (pickle)")
    args = ap.parse_args()

    seeds = list(range(3 if args.quick else 6))
    seconds = 120.0 if args.quick else args.seconds
    print(f"simulating {len(seeds) * 9} trips of {seconds:.0f} s ...")
    rows = build(seeds, seconds, None if args.quick else args.cache)
    test_seeds = seeds[-2:]
    results, importances = {}, None
    res, models = evaluate("A. unseen trips",
                           [r for r in rows if r["seed"] not in test_seeds],
                           [r for r in rows if r["seed"] in test_seeds])
    results["A_unseen_trips"] = res
    imp = models["rf_calibrated"].feature_importances_
    importances = sorted(zip(CAL_NAMES(), imp), key=lambda x: -x[1])
    if args.save_model:
        with open(args.save_model, "wb") as f:
            pickle.dump(models["rf_calibrated"], f)
    res, _ = evaluate("B. unseen mount",
                      [r for r in rows if r["mount"] != "random"],
                      [r for r in rows if r["mount"] == "random"])
    results["B_unseen_mount"] = res

    os.makedirs(args.out, exist_ok=True)
    meta = {"date": time.strftime("%Y-%m-%d"), "trips": len(rows), "seeds": len(seeds), "seconds": seconds}
    with open(os.path.join(args.out, "ml_results.json"), "w") as f:
        json.dump({"meta": meta, "results": results}, f, indent=1)
    plot(results, args.out)
    write_markdown(results, importances, args.out, meta)
    print(f"\nwritten to {args.out}/")


if __name__ == "__main__":
    main()
