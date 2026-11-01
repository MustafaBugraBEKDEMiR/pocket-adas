"""Pocket ADAS simulator.

    python run_sim.py                         # drive yourself
    python run_sim.py --driver aggressive     # watch the aggressive autopilot
    python run_sim.py --mount random          # phone mounted at a random angle
    python run_sim.py --map maps/ankara_sihhiye.osm   # real streets (OpenStreetMap)
    python run_sim.py --headless --driver calm --duration 240
"""
import argparse

from pocket_adas.core.scoring import event_counts


def load_world(path, seed=None):
    if not path:
        return None
    from pocket_adas.sim.osm_world import OsmWorld
    return OsmWorld(path, seed=seed or 0)


def headless(args):
    from pocket_adas.sim import Simulation
    from pocket_adas.sim.autopilot import Autopilot

    driver = "calm" if args.driver == "manual" else args.driver
    sim = Simulation(seed=args.seed, mount=args.mount, world=load_world(args.map, args.seed))
    ap = Autopilot(driver, sim.rng)
    for _ in range(int(args.duration * sim.fs)):
        th, st = ap.control(sim, sim.dt) if sim.t > 3.0 else (0.0, 0.0)
        sim.step(th, st)
    p = sim.pipeline
    print(f"driver={driver} mount={args.mount} distance={p.distance_m / 1000:.2f} km "
          f"score={p.score} calibration={p.cal.stage}")
    print("events:", event_counts(p.events))
    if args.save:
        print("saved to", sim.save_trip(tag=driver))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--driver", choices=["manual", "calm", "normal", "aggressive"], default="manual")
    ap.add_argument("--mount", choices=["dashboard", "flat", "random"], default="dashboard")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--map", default=None, help="OpenStreetMap .osm file (default: built-in test track)")
    ap.add_argument("--headless", action="store_true", help="no window, autopilot only")
    ap.add_argument("--duration", type=float, default=240.0, help="headless run length (s)")
    ap.add_argument("--save", action="store_true", help="headless: save the trip folder")
    args = ap.parse_args()
    if args.headless:
        headless(args)
    else:
        from pocket_adas.sim.app import App
        App(mount=args.mount, seed=args.seed, driver=args.driver, world=load_world(args.map, args.seed)).run()


if __name__ == "__main__":
    main()
