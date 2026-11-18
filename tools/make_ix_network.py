"""Make a one-intersection LPSim network for testing the full-intersection control.

    python tools/make_ix_network.py out_dir --legs 4 --lanes 2 --control signal --vph 600 --hours 1

Creates nodes.csv / edges.csv / od_demand.csv (LPSim format) for a single 4-leg or T
intersection: every leg has a feeder link (vehicles spawn there) and a 200 m approach / exit
link pair. ``--curve`` bends every leg (edge_attrs.csv gets a WKT ``geometry``), ``--skew`` rotates legs
away from the compass points, and ``--turn-lanes`` sets an OSM turn:lanes value on every
approach (its lane count overrides ``--lanes`` for approaches). Demand is Poisson per approach with left / through / right shares, then
tools/lpsim_intersections.py adds the ix_*.csv design tables.
"""
import argparse
import csv
import math
import os
import random
import subprocess
import sys

TAGS = {"signal": "traffic_signals", "awsc": "stop", "twsc": "", "yield": "give_way"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out")
    p.add_argument("--legs", type=int, default=4, choices=[3, 4])
    p.add_argument("--lanes", type=int, default=2, help="lanes per direction")
    p.add_argument("--control", default="signal", choices=list(TAGS))
    p.add_argument("--vph", type=float, default=600.0, help="vehicles per hour per approach")
    p.add_argument("--shares", default="0.2,0.6,0.2", help="left,through,right")
    p.add_argument("--hours", type=float, default=1.0)
    p.add_argument("--start-hour", type=float, default=5.0)
    p.add_argument("--approach", type=float, default=200.0)
    p.add_argument("--peds", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--curve", type=float, default=0.0, help="each leg bends by this many degrees (alternating sign)")
    p.add_argument("--skew", type=float, default=0.0, help="leg i is rotated by +-skew degrees")
    p.add_argument("--turn-lanes", default="", help='OSM turn:lanes for every approach, e.g. "left|left|through|through;right"')
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    lon0, lat0 = -122.4, 37.77
    m_per_deg_lat, m_per_deg_lon = 111_320.0, 111_320.0 * math.cos(math.radians(lat0))
    angles = [0, 90, 180, 270][:a.legs] if a.legs == 4 else [0, 90, 180]
    angles = [ang + (a.skew if i % 2 else -a.skew) for i, ang in enumerate(angles)]
    to_ll = lambda x, y: (lon0 + x / m_per_deg_lon, lat0 + y / m_per_deg_lat)
    wkt = lambda pts: "LINESTRING (" + ", ".join(f"{to_ll(x, y)[0]:.8f} {to_ll(x, y)[1]:.8f}" for x, y in pts) + ")"
    n_app = len(a.turn_lanes.split("|")) if a.turn_lanes else a.lanes
    nodes = [(0, lon0, lat0, TAGS[a.control])]
    edges, uid = [], 100
    mid, feed = {}, {}
    for i, ang in enumerate(angles):
        # leg centre line: straight for 30 m, then a constant-radius bend of --curve degrees
        bend = math.radians(a.curve) * (1 if i % 2 == 0 else -1)
        pts, h, x, y, s = [(0.0, 0.0)], math.radians(ang), 0.0, 0.0, 0.0
        while s < a.approach - 1e-6:
            ds = min(5.0, a.approach - s)
            if s >= 30.0:
                h += bend * ds / max(a.approach - 30.0, 1.0)
            x, y, s = x + ds * math.cos(h), y + ds * math.sin(h), s + ds
            pts.append((x, y))
        fx, fy = x + 100.0 * math.cos(h), y + 100.0 * math.sin(h)
        for (px, py), store in (((x, y), mid), ((fx, fy), feed)):
            idx = len(nodes)
            nodes.append((idx, *to_ll(px, py), ""))
            store[i] = idx
        geo_out, geo_in = (wkt(pts), wkt(pts[::-1])) if a.curve else ("", "")
        for u, v, length, lanes, geo, tl in ((feed[i], mid[i], 100.0, n_app, "", a.turn_lanes and ""),
                                             (mid[i], 0, a.approach, n_app, geo_in, a.turn_lanes),
                                             (0, mid[i], a.approach, a.lanes, geo_out, ""),
                                             (mid[i], feed[i], 100.0, a.lanes, "", "")):
            edges.append((uid, u, v, length, lanes, 30, geo, tl))
            uid += 1
    with open(os.path.join(a.out, "nodes.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["osmid", "x", "y", "ref", "highway"])
        for n in nodes:
            w.writerow([n[0], f"{n[1]:.8f}", f"{n[2]:.8f}", "", n[3]])
    with open(os.path.join(a.out, "edges.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["uniqueid", "u", "v", "length", "lanes", "speed_mph"])
        w.writerows(e[:6] for e in edges)
    with open(os.path.join(a.out, "edge_attrs.csv"), "w", newline="") as f:   # shapes / turn lanes (sidecar)
        w = csv.writer(f)
        w.writerow(["uniqueid", "geometry", "turn_lanes"])
        w.writerows((e[0], e[6], e[7]) for e in edges if e[6] or e[7])
    # demand: Poisson arrivals per approach, movement by turning direction
    rng = random.Random(a.seed)
    shares = [float(x) for x in a.shares.split(",")]
    trips = []
    t0, t1 = a.start_hour * 3600, (a.start_hour + a.hours) * 3600
    for i, ang in enumerate(angles):
        dests = {}
        for j, bng in enumerate(angles):
            if j == i:
                continue
            turn = (bng - ang) % 360            # exit direction relative to the approach leg
            mov = {90: 2, 180: 1, 270: 0}.get(int(round(turn / 90.0)) * 90 % 360)   # right / through / left
            if mov is not None:
                dests[mov] = j
        w = [shares[m] if m in dests else 0.0 for m in range(3)]
        t = t0
        while True:
            t += rng.expovariate(a.vph / 3600.0)
            if t >= t1:
                break
            m = rng.choices(range(3), weights=w)[0]
            trips.append((round(t, 1), feed[i], feed[dests[m]]))
    trips.sort()
    with open(os.path.join(a.out, "od_demand.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dep_time", "origin", "destination"])
        w.writerows(trips)
    print(f"{len(nodes)} nodes, {len(edges)} edges, {len(trips)} trips -> {a.out}")
    tool = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lpsim_intersections.py")
    subprocess.run([sys.executable, tool, a.out, "--peds", str(a.peds)], check=True)


if __name__ == "__main__":
    main()
