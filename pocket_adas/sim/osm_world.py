"""Real streets from an OpenStreetMap extract (.osm XML from openstreetmap.org > Export).

* Drivable ways become a directed graph (``oneway`` and roundabouts respected).
* Speed limit: the way's ``maxspeed`` tag; if missing, Turkish urban defaults
  (50 km/h on through roads, 30 km/h on residential/service roads). Points within
  ``SCHOOL_RADIUS`` of a school are capped at 30 km/h.
* ``traffic_calming`` nodes become speed humps. When a map has none (common in
  Turkey), humps are placed on a share of pedestrian crossings of minor roads and
  marked ``synthetic``. A few synthetic potholes are added as well.
* The autopilot follows a random route through the graph and extends it as it goes.

Map data © OpenStreetMap contributors, ODbL.
"""
import math
import xml.etree.ElementTree as ET
from collections import defaultdict

import numpy as np

from ..core.geo import latlon_to_xy
from .path import Path, chaikin, resample
from .road import Feature, feature_height

DRIVABLE = {
    "motorway": (3, 90), "trunk": (3, 82), "primary": (3, 50), "secondary": (2, 50),
    "tertiary": (2, 50), "unclassified": (1, 50), "residential": (1, 30),
    "living_street": (1, 20), "service": (1, 30), "motorway_link": (1, 70),
    "trunk_link": (1, 50), "primary_link": (1, 50), "secondary_link": (1, 50),
    "tertiary_link": (1, 50),
}  # highway: (default lanes per direction, default limit km/h)
LANE_W = 3.3
SCHOOL_RADIUS = 60.0
MINOR = {"residential", "service", "tertiary", "unclassified", "living_street"}


def _maxspeed(v):
    try:
        return float(str(v).split()[0])
    except (ValueError, TypeError):
        return None


class Edge:
    __slots__ = ("u", "v", "pts", "limits", "width", "oneway", "name", "highway", "school")

    def __init__(self, u, v, pts, limits, width, oneway, name, highway, school):
        self.u, self.v, self.pts, self.limits = u, v, pts, limits
        self.width, self.oneway, self.name, self.highway, self.school = width, oneway, name, highway, school


class OsmWorld:
    def __init__(self, path, seed=0):
        self.name = path.replace("\\", "/").split("/")[-1]
        rng = np.random.default_rng(seed)
        root = ET.parse(path).getroot()
        b = root.find("bounds")
        if b is not None:
            lat0 = (float(b.get("minlat")) + float(b.get("maxlat"))) / 2
            lon0 = (float(b.get("minlon")) + float(b.get("maxlon"))) / 2
        else:
            lats = [float(n.get("lat")) for n in root.iter("node")]
            lons = [float(n.get("lon")) for n in root.iter("node")]
            lat0, lon0 = float(np.mean(lats)), float(np.mean(lons))
        self.origin = (lat0, lon0)
        clip = None
        if b is not None:
            x0, y0 = latlon_to_xy(float(b.get("minlat")), float(b.get("minlon")), self.origin)
            x1, y1 = latlon_to_xy(float(b.get("maxlat")), float(b.get("maxlon")), self.origin)
            clip = (x0 - 40, y0 - 40, x1 + 40, y1 + 40)

        nodes, ntags = {}, {}
        for n in root.iter("node"):
            nid = n.get("id")
            nodes[nid] = latlon_to_xy(float(n.get("lat")), float(n.get("lon")), self.origin)
            t = {x.get("k"): x.get("v") for x in n.findall("tag")}
            if t:
                ntags[nid] = t

        ways, schools, areas = [], [], []
        for w in root.iter("way"):
            t = {x.get("k"): x.get("v") for x in w.findall("tag")}
            refs = [nd.get("ref") for nd in w.findall("nd") if nd.get("ref") in nodes]
            if len(refs) >= 3 and refs[0] == refs[-1]:
                kind = ("building" if "building" in t else
                        "park" if t.get("leisure") in ("park", "garden", "pitch") or
                        t.get("landuse") in ("grass", "forest", "recreation_ground") or t.get("natural") == "wood"
                        else "school" if t.get("amenity") in ("school", "kindergarten", "university") else None)
                if kind:
                    areas.append((np.array([nodes[r] for r in refs]), kind))
            if t.get("amenity") in ("school", "kindergarten"):
                schools.extend(nodes[r] for r in refs)
            hw = t.get("highway")
            if hw in DRIVABLE and len(refs) >= 2 and t.get("area") != "yes":
                ways.append((t, refs))
        for nid, t in ntags.items():
            if t.get("amenity") in ("school", "kindergarten"):
                schools.append(nodes[nid])
        self.schools = np.array(schools) if schools else np.zeros((0, 2))

        # intersections = nodes used by more than one drivable way, plus way ends
        use = defaultdict(int)
        for t, refs in ways:
            for r in refs:
                use[r] += 1
        self.edges, self.out = [], defaultdict(list)
        for t, refs in ways:
            hw = t["highway"]
            lanes_dir, default_lim = DRIVABLE[hw]
            oneway = 1 if (t.get("oneway") in ("yes", "true", "1") or t.get("junction") == "roundabout"
                           or hw.endswith("motorway")) else (-1 if t.get("oneway") == "-1" else 0)
            if oneway == -1:
                refs, oneway = refs[::-1], 1
            lanes = _maxspeed(t.get("lanes"))
            width = (lanes * LANE_W) if lanes else (lanes_dir * LANE_W * (1 if oneway else 2))
            width = max(width, 4.0)
            limit = _maxspeed(t.get("maxspeed")) or default_lim
            cut = [0] + [i for i in range(1, len(refs) - 1) if use[refs[i]] > 1] + [len(refs) - 1]
            for a, b in zip(cut[:-1], cut[1:]):
                poly = np.array([nodes[r] for r in refs[a:b + 1]])
                pts, _, length, _ = resample(poly, step=2.0)
                if length < 1.0:
                    continue
                if clip is not None:
                    inside = ((pts[:, 0] > clip[0]) & (pts[:, 0] < clip[2]) &
                              (pts[:, 1] > clip[1]) & (pts[:, 1] < clip[3]))
                    if inside.mean() < 0.5:
                        continue
                lim = np.full(len(pts), float(limit))
                school = False
                if len(self.schools):
                    d = np.sqrt(((pts[:, None, :] - self.schools[None]) ** 2).sum(-1)).min(1)
                    near = d < SCHOOL_RADIUS
                    lim[near] = np.minimum(lim[near], 30.0)
                    school = bool(near.any())
                self.edges.append(Edge(refs[a], refs[b], pts, lim, width, oneway, t.get("name", ""), hw, school))
        self._build_routing_graph()
        if clip is not None:
            areas = [(a, k) for a, k in areas if clip[0] < a[:, 0].mean() < clip[2] and clip[1] < a[:, 1].mean() < clip[3]]
        self.areas = areas

        # flat arrays for nearest-road queries
        P, L, Wd = [], [], []
        for e in self.edges:
            P.append(e.pts); L.append(e.limits); Wd.append(np.full(len(e.pts), e.width / 2))
        self._pts, self._lim, self._halfw = np.vstack(P), np.concatenate(L), np.concatenate(Wd)
        self.bounds = (self._pts[:, 0].min(), self._pts[:, 1].min(), self._pts[:, 0].max(), self._pts[:, 1].max())

        # humps: tagged traffic calming, else synthetic on crossings of minor roads
        self.features = []
        node_edge = {}
        for k, e in enumerate(self.edges):
            for r in (e.u, e.v):
                node_edge.setdefault(r, k)
        calming = [nid for nid, t in ntags.items() if t.get("traffic_calming") in ("hump", "bump", "table", "cushion")]
        synthetic = not calming
        if synthetic:
            crossings = [nid for nid, t in ntags.items() if t.get("highway") == "crossing"]
            calming = [c for c in crossings if rng.random() < 0.45]
        for nid in calming:
            x, y = nodes[nid]
            i = self._nearest_idx(x, y)
            if np.hypot(*(self._pts[i] - (x, y))) > 3:
                continue
            e = self._edge_of(i)
            if synthetic and e.highway not in MINOR:
                continue
            tx, ty = self._tangent(e, x, y)
            self.features.append(Feature("hump", x, y, tx, ty, e.width, 3.0, 0.07, synthetic))
        minor = [e for e in self.edges if e.highway in MINOR or e.highway == "secondary"]
        for _ in range(min(14, len(minor))):
            e = minor[int(rng.integers(len(minor)))]
            j = int(rng.integers(len(e.pts)))
            tx, ty = self._tangent(e, *e.pts[j])
            side = -e.width / 4 if not e.oneway else rng.uniform(-1, 1)
            x, y = e.pts[j] + side * np.array([-ty, tx])
            self.features.append(Feature("pothole", x, y, tx, ty, 1.4, 0.9, -0.06, True))

    # ------------------------------------------------------------- helpers
    def _build_routing_graph(self):
        """Keep only the largest strongly connected part for routing, so a
        random walk can never get stuck in a dead end or a cut-off one-way."""
        succ = defaultdict(set)
        for e in self.edges:
            succ[e.u].add(e.v)
            if not e.oneway:
                succ[e.v].add(e.u)
        nodes = set(succ) | {v for vs in succ.values() for v in vs}
        # prune dead ends (they would force U-turns)
        while True:
            und = defaultdict(set)
            for u in nodes:
                for v in succ[u]:
                    if v in nodes:
                        und[u].add(v)
                        und[v].add(u)
            dead = {n for n in nodes if len(und[n]) <= 1}
            if not dead:
                break
            nodes -= dead
        succ = defaultdict(set, {u: {v for v in vs if v in nodes} for u, vs in succ.items() if u in nodes})
        pred = defaultdict(set)
        for u, vs in succ.items():
            for v in vs:
                pred[v].add(u)
        # Kosaraju, iterative
        order, seen = [], set()
        for s0 in nodes:
            if s0 in seen:
                continue
            seen.add(s0)
            stack = [(s0, iter(succ[s0]))]
            while stack:
                n, it = stack[-1]
                nxt = next((v for v in it if v not in seen), None)
                if nxt is None:
                    order.append(n)
                    stack.pop()
                else:
                    seen.add(nxt)
                    stack.append((nxt, iter(succ[nxt])))
        comp, best = {}, None
        for s0 in reversed(order):
            if s0 in comp:
                continue
            members, stack = [s0], [s0]
            comp[s0] = s0
            while stack:
                n = stack.pop()
                for v in pred[n]:
                    if v not in comp:
                        comp[v] = s0
                        members.append(v)
                        stack.append(v)
            if best is None or len(members) > len(best):
                best = members
        core = set(best or [])
        self.out = defaultdict(list)
        self.routable = []
        for k, e in enumerate(self.edges):
            if e.u in core and e.v in core:
                self.routable.append(k)
                self.out[e.u].append((k, +1))
                if not e.oneway:
                    self.out[e.v].append((k, -1))

    def _nearest_idx(self, x, y):
        d = self._pts - (x, y)
        return int(np.argmin(np.einsum("ij,ij->i", d, d)))

    def _edge_of(self, flat_idx):
        k = 0
        for e in self.edges:
            if flat_idx < k + len(e.pts):
                return e
            k += len(e.pts)
        return self.edges[-1]

    @staticmethod
    def _tangent(e, x, y):
        d = e.pts - (x, y)
        j = int(np.argmin(np.einsum("ij,ij->i", d, d)))
        a, b = e.pts[max(0, j - 1)], e.pts[min(len(e.pts) - 1, j + 1)]
        t = b - a
        n = np.linalg.norm(t)
        return (t / n) if n > 0 else np.array([1.0, 0.0])

    # ----------------------------------------------------------- interface
    def speed_limit_at(self, x, y):
        i = self._nearest_idx(x, y)
        if np.hypot(*(self._pts[i] - (x, y))) > self._halfw[i] + 6:
            return None
        return float(self._lim[i])

    def offroad(self, x, y):
        i = self._nearest_idx(x, y)
        return np.hypot(*(self._pts[i] - (x, y))) > self._halfw[i] + 1.0

    def height_at(self, x, y):
        return feature_height(self.features, x, y)

    def start_pose(self, rng=None):
        rng = rng or np.random.default_rng()
        routable = [self.edges[k] for k in self.routable]
        main = [e for e in routable if e.highway in ("primary", "secondary") and len(e.pts) > 40] or routable
        e = main[int(rng.integers(len(main)))]
        j = len(e.pts) // 3
        t = e.pts[j + 1] - e.pts[j]
        off = 0.0 if e.oneway else -e.width / 4
        n = np.array([-t[1], t[0]]) / max(np.linalg.norm(t), 1e-9)
        x, y = e.pts[j] + off * n
        return float(x), float(y), math.atan2(t[1], t[0])

    # ------------------------------------------------------------- routing
    def _oriented(self, k, d):
        e = self.edges[k]
        if d > 0:
            return e.pts, e.limits, e.v, e
        return e.pts[::-1], e.limits[::-1], e.u, e

    def _walk(self, node, prev_edge, rng, length=600.0):
        """Random walk without U-turns; returns list of (edge, dir)."""
        route, total = [], 0.0
        for _ in range(60):
            opts = [(k, d) for k, d in self.out.get(node, []) if k != prev_edge]
            if not opts:
                opts = list(self.out.get(node, []))
            if not opts:
                break
            # prefer staying on bigger roads a bit
            w = np.array([3.0 if self.edges[k].highway not in MINOR else 1.0 for k, _ in opts])
            k, d = opts[int(rng.choice(len(opts), p=w / w.sum()))]
            route.append((k, d))
            pts, _, node, e = self._oriented(k, d)
            prev_edge = k
            total += len(pts) * 2.0
            if total > length:
                break
        return route

    def _build(self, head_pts, head_lim, head_off, route):
        P, L, O = [head_pts], [head_lim], [head_off]
        for k, d in route:
            pts, lim, _, e = self._oriented(k, d)
            P.append(pts); L.append(lim)
            O.append(np.full(len(pts), 0.0 if e.oneway else -e.width / 4))
        pts = np.vstack([p for p in P if len(p)])
        lim = np.concatenate([x for x in L if len(x)])
        off = np.concatenate([x for x in O if len(x)])
        keep = np.concatenate([[True], np.linalg.norm(np.diff(pts, axis=0), axis=1) > 0.2])
        pts, lim, off = pts[keep], lim[keep], off[keep]
        # round junction corners to a drivable radius (~8 m)
        smooth = chaikin(pts[::4], 3) if len(pts) > 12 else pts
        rp, _, _, _ = resample(smooth, step=1.0)
        # carry limits/offsets by nearest original point
        idx = np.clip(np.searchsorted(np.cumsum(np.r_[0, np.linalg.norm(np.diff(pts, axis=0), axis=1)]),
                                      np.arange(len(rp)), side="right") - 1, 0, len(pts) - 1)
        path = Path(rp, lim[idx], closed=False, lane_offset=off[idx])
        path.route = route
        return path

    def new_path(self, x, y, heading, rng):
        best = None
        for k in self.routable:
            d = self.edges[k].pts - (x, y)
            dd = np.einsum("ij,ij->i", d, d)
            j0 = int(np.argmin(dd))
            if best is None or dd[j0] < best[0]:
                best = (dd[j0], k, j0)
        _, k, j = best
        e = self.edges[k]
        hv = np.array([math.cos(heading), math.sin(heading)])
        t = e.pts[min(j + 1, len(e.pts) - 1)] - e.pts[max(j - 1, 0)]
        forward = np.dot(t, hv) >= 0 or e.oneway
        if forward:
            head, lim, node = e.pts[j:], e.limits[j:], e.v
        else:
            head, lim, node = e.pts[:j + 1][::-1], e.limits[:j + 1][::-1], e.u
        off = np.full(len(head), 0.0 if e.oneway else -e.width / 4)
        return self._build(head, lim, off, self._walk(node, k, rng))

    def extend_path(self, path, i, rng):
        """When fewer than 150 m remain, drop the driven part and add more route."""
        if path.remaining(i) > 150:
            return path
        start = max(0, i - 20)
        if not path.route:
            x, y = path.pts[i]
            t = path.tangent[i]
            return self.new_path(x, y, math.atan2(t[1], t[0]), rng)
        k, d = path.route[-1]
        _, _, node, _ = self._oriented(k, d)
        new = self._build(path.pts[start:], path.limits[start:], path.lane_offset[start:],
                          self._walk(node, k, rng))
        new.route = path.route[-1:] + new.route
        return new

    # ------------------------------------------------------------ drawing
    def segments(self):
        order = {"service": 0, "residential": 1, "living_street": 1, "unclassified": 1}
        es = sorted(self.edges, key=lambda e: order.get(e.highway, 2))
        return [(e.pts, e.width, False, "school" if e.school and e.highway in MINOR else e.highway) for e in es]

    def signs(self):
        out = []
        for e in self.edges:
            if e.school and len(e.pts) > 10:
                x, y = e.pts[len(e.pts) // 2]
                out.append((x, y, 30, "SCHOOL"))
        return out[::3]

    def labels(self):
        seen, out = set(), []
        for e in sorted(self.edges, key=lambda e: -len(e.pts)):
            if e.name and e.name not in seen and len(e.pts) > 60:
                seen.add(e.name)
                x, y = e.pts[len(e.pts) // 2]
                out.append((x, y, e.name))
        return out
