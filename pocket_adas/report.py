"""Trip report: GeoJSON layers and a self-contained HTML map.

    python -m pocket_adas.report trips/<trip folder>

Writes ``route.geojson`` (GPS track), ``events.geojson`` (one point per event,
open them in QGIS or geojson.io) and ``report.html`` (Leaflet map + summary).
"""
import csv
import json
import os
import sys

from .core.events import MESSAGES

COLOURS = {
    "HARSH_BRAKE": "#e64646", "HARSH_ACCEL": "#f09632", "HARSH_TURN": "#c85adc",
    "BUMP": "#e6c81e", "POTHOLE": "#a07850", "SPEEDING": "#50aaff", "SUDDEN_MOVE": "#888888",
}


def _read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_geojson(trip_dir):
    sensors = _read_csv(os.path.join(trip_dir, "sensors.csv"))
    coords = [[float(r["longitude"]), float(r["latitude"])] for r in sensors if r["latitude"]]
    route = {"type": "FeatureCollection", "features": [{
        "type": "Feature", "properties": {"name": "GPS track"},
        "geometry": {"type": "LineString", "coordinates": coords}}]}
    events = []
    for e in _read_csv(os.path.join(trip_dir, "events.csv")):
        if not e["lat"]:
            continue
        props = {k: e[k] for k in ("t", "kind", "peak", "severity", "speed_kmh", "limit_kmh")}
        props["message"] = MESSAGES.get(e["kind"], e["kind"])
        events.append({"type": "Feature", "properties": props,
                       "geometry": {"type": "Point", "coordinates": [float(e["lon"]), float(e["lat"])]}})
    ev = {"type": "FeatureCollection", "features": events}
    with open(os.path.join(trip_dir, "route.geojson"), "w") as f:
        json.dump(route, f)
    with open(os.path.join(trip_dir, "events.geojson"), "w") as f:
        json.dump(ev, f)
    return route, ev


HTML = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pocket ADAS trip report</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
 body{margin:0;font:14px system-ui,sans-serif;display:flex;height:100vh;color:#222}
 #side{width:320px;padding:18px;overflow:auto;background:#f6f6f4;border-right:1px solid #ddd}
 #map{flex:1}
 h1{font-size:20px;margin:0 0 4px} .muted{color:#777}
 .score{font-size:56px;font-weight:700;margin:10px 0 0}
 table{border-collapse:collapse;width:100%;margin-top:12px} td{padding:4px 2px;border-bottom:1px solid #e4e4e0}
 .dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px}
</style></head><body>
<div id="side">
 <h1>Pocket ADAS trip report</h1>
 <div class="muted">__TAG__ driver · __MOUNT__ mount · __WORLD__</div>
 <div class="score">__SCORE__</div><div class="muted">score / 100</div>
 <p>__KM__ km · __MIN__ min</p>
 <table>__ROWS__</table>
 <p class="muted" style="margin-top:16px">Click a marker for details. Map data © OpenStreetMap contributors.</p>
</div>
<div id="map"></div>
<script>
const route = __ROUTE__, events = __EVENTS__, colours = __COLOURS__;
const map = L.map('map');
L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png',
  {maxZoom: 19, attribution: '&copy; OpenStreetMap contributors'}).addTo(map);
const line = L.geoJSON(route, {style: {color: '#2a6fdb', weight: 4, opacity: .8}}).addTo(map);
L.geoJSON(events, {pointToLayer: (f, ll) => L.circleMarker(ll, {radius: 7, color: '#222', weight: 1,
   fillColor: colours[f.properties.kind] || '#999', fillOpacity: .95}),
  onEachFeature: (f, l) => { const p = f.properties;
   l.bindPopup(`<b>${p.message}</b><br>t = ${(+p.t).toFixed(1)} s<br>peak ${(+p.peak).toFixed(1)}` +
     `${p.kind === 'SPEEDING' ? ' km/h (limit ' + p.limit_kmh + ')' : ' m/s²'}<br>speed ${(+p.speed_kmh).toFixed(0)} km/h`); }
}).addTo(map);
try { map.fitBounds(line.getBounds(), {padding: [20, 20]}); } catch (e) { map.setView([39.93, 32.86], 15); }
</script></body></html>"""


def write_html(trip_dir, route, events):
    with open(os.path.join(trip_dir, "summary.json")) as f:
        s = json.load(f)
    rows = "".join(
        f'<tr><td><span class="dot" style="background:{COLOURS.get(k, "#999")}"></span>'
        f'{MESSAGES.get(k, k).title()}</td><td style="text-align:right">{n}</td></tr>'
        for k, n in sorted(s["events"].items(), key=lambda kv: -kv[1])) or "<tr><td>No events</td></tr>"
    html = (HTML.replace("__TAG__", s.get("tag", "")).replace("__MOUNT__", s.get("mount", ""))
            .replace("__WORLD__", s.get("world", "")).replace("__SCORE__", str(s["score"]))
            .replace("__KM__", f'{s["distance_m"] / 1000:.2f}').replace("__MIN__", f'{s["duration_s"] / 60:.1f}')
            .replace("__ROWS__", rows).replace("__ROUTE__", json.dumps(route))
            .replace("__EVENTS__", json.dumps(events)).replace("__COLOURS__", json.dumps(COLOURS)))
    path = os.path.join(trip_dir, "report.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    return path


def build_report(trip_dir):
    route, events = write_geojson(trip_dir)
    return write_html(trip_dir, route, events)


if __name__ == "__main__":
    for d in sys.argv[1:]:
        print(build_report(d))
