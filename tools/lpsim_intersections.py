"""Build full-intersection designs for every node of an LPSim network.

    python tools/lpsim_intersections.py external/LPSim/data/networks/sf_basic [--control auto] [--peds 0]

Reads nodes.csv / edges.csv, groups the edges at each node into legs (one per neighbouring
node), allocates movements to approach lanes, chooses the control type, and calls
lcrl.intersection.build -- the same geometry / conflict / signal code the training env uses.

Optional per-edge attributes, in a sidecar edge_attrs.csv (uniqueid, geometry, turn_lanes;
written by LPSim's tools/generate_network.py --edge-attrs) or as extra edges.csv columns:
  geometry    WKT LINESTRING of the edge (lon lat, or metres) -> curved legs, curve speed caps
  turn_lanes  OSM turn:lanes of the edge's direction, e.g. "left|left;through|through|right"
              -> exact lane use, including dual / triple turn lanes
(The sidecar keeps edges.csv readable by LPSim's own CSV reader, which splits on every comma.)
Optional lane spans, edge_lanes.csv (uniqueid, lane, start_m, end_m; generate_network.py
--lane-spans): an approach lane that starts late is a turn bay, one that ends before the stop
line serves no movement, turn_lanes applies to the lanes that reach the stop line, and exits use
only the lanes that exist where the exit edge starts.
Without them legs are straight (towards the neighbouring node) and lanes are allocated by a
heuristic (outer lanes take the turns).

Writes, next to the network:

  ix_nodes.csv      node, ctrl, n_legs, n_paths, n_phase, rtor, D, cycle      (D: largest setback)
  ix_legs.csv       node, leg, angle_deg, in_edge, out_edge, major, ped_rate_ph, cross_time_s, setback_m
  ix_paths.csv      node, path, leg, lane, mov, exit_leg, exit_lane, in_edge, out_edge, length_m,
                    s_wait_m, n_stage    (staging point / places of permitted turns; -1 / 0 = none)
  ix_conflicts.csv  node, p, q, kind, p_in, p_out      (zone on path p with respect to path q)
  ix_phases.csv     node, phase, duration_s, walk_mask, states (one digit per path)
  edge_curves.csv   edge, step_m, caps          (curve speed caps in m/s along curved edges)

Edges are referenced by edges.csv ``uniqueid``; nodes by ``osmid``. Nodes with fewer than three
legs are left to LPSim's default pass-through behaviour.
"""
import argparse
import csv
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lcrl import intersection as ixd  # noqa: E402
from lcrl.config import MOV_LEFT, MOV_RIGHT, MOV_THROUGH  # noqa: E402

CTRL_NAMES = {"signal": ixd.IX_SIGNAL, "awsc": ixd.IX_AWSC, "twsc": ixd.IX_TWSC, "yield": ixd.IX_YIELD}


def read_network(path):
    nodes = {}
    with open(os.path.join(path, "nodes.csv"), newline="") as f:
        for r in csv.DictReader(f):
            nodes[int(float(r["osmid"]))] = (float(r["x"]), float(r["y"]), (r.get("highway") or "").strip())
    edges = []
    with open(os.path.join(path, "edges.csv"), newline="") as f:
        rd = csv.DictReader(f)
        u_key = "osmid_u" if "osmid_u" in rd.fieldnames else "u"
        v_key = "osmid_v" if "osmid_v" in rd.fieldnames else "v"
        for r in rd:
            edges.append(dict(id=int(float(r["uniqueid"])), u=int(float(r[u_key])), v=int(float(r[v_key])),
                              length=float(r["length"]), lanes=max(1, int(float(r["lanes"]))),
                              speed=float(r["speed_mph"]), geometry=parse_wkt(r.get("geometry")),
                              turn_lanes=(r.get("turn_lanes") or "").strip()))
    side = os.path.join(path, "edge_attrs.csv")
    if os.path.exists(side):
        by_id = {e["id"]: e for e in edges}
        with open(side, newline="") as f:
            for r in csv.DictReader(f):
                e = by_id.get(int(float(r["uniqueid"])))
                if e is not None:
                    e["geometry"] = parse_wkt(r.get("geometry")) or e["geometry"]
                    e["turn_lanes"] = (r.get("turn_lanes") or "").strip() or e["turn_lanes"]
    for e in edges:                             # lane spans: (start_m, end_m) per lane, full length by default
        e["spans"] = [(0.0, e["length"])] * e["lanes"]
    spans = os.path.join(path, "edge_lanes.csv")
    if os.path.exists(spans):
        by_id = {e["id"]: e for e in edges}
        with open(spans, newline="") as f:
            for r in csv.DictReader(f):
                e = by_id.get(int(float(r["uniqueid"])))
                k = int(r["lane"])
                if e is not None and 0 <= k < e["lanes"]:
                    e["spans"][k] = (max(0.0, float(r["start_m"])), min(e["length"], float(r["end_m"])))
    return nodes, edges


def stop_line_lanes(e):
    """Lanes of an approach edge that reach its end (the stop line)."""
    return [k for k, (s, t) in enumerate(e["spans"]) if t > s and t >= e["length"] - 1.0]


def start_lanes(e):
    """(first lane, count) of the lanes that exist where an exit edge starts."""
    ks = [k for k, (s, t) in enumerate(e["spans"]) if t > s and s <= 0.5]
    return (ks[0], ks[-1] - ks[0] + 1) if ks else (0, e["lanes"])


def parse_wkt(text):
    """[(x, y), ...] from 'LINESTRING (x y, x y, ...)', or None."""
    if not text or "(" not in text:
        return None
    body = text[text.index("(") + 1:text.rindex(")")]
    pts = [tuple(float(v) for v in p.split()[:2]) for p in body.split(",") if p.strip()]
    return pts if len(pts) >= 2 else None


def to_local(nodes, center, pts):
    """Metres relative to node ``center`` (equirectangular if coordinates look like lon/lat)."""
    x0, y0, _ = nodes[center]
    lonlat = abs(x0) <= 180 and abs(y0) <= 90
    k = 111_320.0
    out = [((x - x0) * k * math.cos(math.radians(y0)), (y - y0) * k) if lonlat else (x - x0, y - y0) for x, y in pts]
    return np.array(out, float)


def leg_shape(nodes, n, e_in, e_out):
    """Centre line of a leg from node n outward: the out-edge geometry, else the reversed
    in-edge geometry; None when no geometry is known (straight leg)."""
    if e_out is not None and e_out["geometry"]:
        return to_local(nodes, n, e_out["geometry"])
    if e_in is not None and e_in["geometry"]:
        return to_local(nodes, n, e_in["geometry"][::-1])
    return None


TURN_TOKENS = {"left": MOV_LEFT, "slight_left": MOV_LEFT, "sharp_left": MOV_LEFT, "through": MOV_THROUGH,
               "none": MOV_THROUGH, "": MOV_THROUGH, "merge_to_left": MOV_THROUGH, "merge_to_right": MOV_THROUGH,
               "right": MOV_RIGHT, "slight_right": MOV_RIGHT, "sharp_right": MOV_RIGHT}


def lanes_from_tag(tag, n, exits):
    """Exit legs per lane from an OSM turn:lanes value (left -> right), or None if unusable.
    ``exits`` are (leg, angles, turn_deg) sorted from the sharpest left."""
    parts = tag.split("|")
    if len(parts) != n:
        return None
    by_class = {}
    for e in exits:
        by_class.setdefault(ixd.turn_class(*e[1]), []).append(e[0])
    out = []
    for p in parts:
        legs = set()
        for tok in p.split(";"):
            m = TURN_TOKENS.get(tok.strip())
            if m is not None:
                legs |= set(by_class.get(m, []))
        out.append(legs)
    if not all(out):
        return None
    if set().union(*out) != {e[0] for e in exits}:   # an exit no lane serves: give it to the nearest outer lane
        for e in exits:
            if not any(e[0] in s for s in out):
                (out[0] if ixd.turn_class(*e[1]) == MOV_LEFT else out[-1]).add(e[0])
    return out


def local_xy(nodes, center, other):
    """Metres from node ``center`` to node ``other`` (equirectangular if coordinates look like lon/lat)."""
    x0, y0, _ = nodes[center]
    x1, y1, _ = nodes[other]
    if abs(x0) <= 180 and abs(y0) <= 90:
        k = 111_320.0
        return (x1 - x0) * k * math.cos(math.radians(y0)), (y1 - y0) * k
    return x1 - x0, y1 - y0


def allocate_lanes(n, exits):
    """Exit legs per approach lane, left -> right. ``exits`` is sorted from the sharpest left
    to the sharpest right turn. Outer lanes take the turns, inner lanes the straightest exit."""
    m = len(exits)
    if n == 1 or m == 1:
        return [{e[0] for e in exits} for _ in range(n)]
    turn = [ixd.turn_class(*e[1]) for e in exits]
    straight = min(range(m), key=lambda i: abs(exits[i][2]))
    lefts = [i for i in range(m) if turn[i] == MOV_LEFT]
    rights = [i for i in range(m) if turn[i] == MOV_RIGHT]
    out = [set() for _ in range(n)]
    if n == 2:
        out[0] = {i for i in range(m) if i <= straight}
        out[1] = {i for i in range(m) if i >= straight}
    else:
        out[0] = set(lefts) or {0}
        out[-1] = set(rights) or {m - 1}
        for k in range(1, n - 1):
            out[k] = {i for i in range(m) if i not in lefts and i not in rights} or {straight}
        served = set().union(*out)
        for i in range(m):                 # anything left over goes to the nearest outer lane
            if i not in served:
                out[0 if i < straight else -1].add(i)
    return [{exits[i][0] for i in s} for s in out]


def node_design(nodes, n, ins, outs, ctrl_mode, peds, kmax=8, pmax=64):
    nbrs = sorted({e["u"] for e in ins} | {e["v"] for e in outs})
    if len(nbrs) < 3:
        return None
    legs, leg_in, leg_out = [], [], []
    for m in nbrs:
        dx, dy = local_xy(nodes, n, m)
        if dx == 0 and dy == 0:
            return None
        e_in = next((e for e in ins if e["u"] == m), None)
        e_out = next((e for e in outs if e["v"] == m), None)
        shape = leg_shape(nodes, n, e_in, e_out)
        if shape is not None:                   # leg direction for turn classes: 40 m out along the shape
            cl = ixd.Centreline(shape - shape[0])
            px, py = cl.at(ixd.CLASSIFY_S)[0]
            far = math.degrees(math.atan2(py, px)) % 360.0
        else:
            far = math.degrees(math.atan2(dy, dx)) % 360.0
        legs.append(dict(angle=far, shape=None if shape is None else shape - shape[0], e_in=e_in, e_out=e_out))
    order = sorted(range(len(legs)), key=lambda i: legs[i]["angle"])
    legs = [legs[i] for i in order]
    spec = []
    for a, lg in enumerate(legs):
        if lg["e_in"] is None:
            spec.append(dict(angle=lg["angle"], shape=lg["shape"], lanes=[], n_out=start_lanes(lg["e_out"])[1], L_app=0.0,
                             L_exit=lg["e_out"]["length"]))
            continue
        exits = []
        for b, lb in enumerate(legs):
            if b != a and lb["e_out"] is not None:
                c = ixd.turn_class(lg["angle"], lb["angle"])
                if c >= 0:
                    h_in, h_out = -ixd.unit_deg(lg["angle"]), ixd.unit_deg(lb["angle"])
                    turn = math.degrees(math.atan2(h_in[0] * h_out[1] - h_in[1] * h_out[0], h_in @ h_out))
                    exits.append((b, (lg["angle"], lb["angle"]), turn))
        exits.sort(key=lambda e: -e[2])         # sharpest left first
        if not exits:
            spec.append(dict(angle=lg["angle"], shape=lg["shape"], lanes=[],
                             n_out=start_lanes(lg["e_out"])[1] if lg["e_out"] else 0,
                             L_app=0.0, L_exit=lg["e_out"]["length"] if lg["e_out"] else 0.0))
            continue
        e_in = lg["e_in"]
        at_line = [k for k in stop_line_lanes(e_in) if k < kmax]   # movements go to the lanes at the stop line
        n_lanes = len(at_line)
        alloc = lanes_from_tag(e_in["turn_lanes"], n_lanes, exits) if e_in["turn_lanes"] else None
        if alloc is None:
            alloc = allocate_lanes(n_lanes, exits)
        L = e_in["length"]
        masks = {k: sum(1 << b for b in s_) for k, s_ in zip(at_line, alloc)}
        lanes = [(s0, t0, masks.get(k, 0)) for k, (s0, t0) in enumerate(e_in["spans"][:kmax])]
        spec.append(dict(angle=lg["angle"], shape=lg["shape"], lanes=lanes,
                         n_out=start_lanes(lg["e_out"])[1] if lg["e_out"] else 0,
                         L_app=L, L_exit=lg["e_out"]["length"] if lg["e_out"] else 0.0))
    for s in spec:                              # legs without an approach or exit still need geometry
        s["n_out"] = max(s["n_out"], 0)
    # exit lanes must take every lane turning into them side by side (dual turns)
    for a, sa in enumerate(spec):
        for b, sb in enumerate(spec):
            if a != b and sb["n_out"] > 0:
                n_into = sum(1 for _, _, m in sa["lanes"] if m >> b & 1)
                sb["n_out"] = max(sb["n_out"], min(n_into, kmax))
    # control type: OSM node tag first; untagged junctions get two-way stop (widest street major)
    tag = nodes[n][2]
    widest = max(max(len(s["lanes"]), s["n_out"]) for s in spec)
    speeds = [e["speed"] for lg in legs for e in (lg["e_in"], lg["e_out"]) if e is not None]
    if ctrl_mode != "auto":
        ctrl = CTRL_NAMES[ctrl_mode]
    elif "motorway" in tag or min(speeds) >= 45:
        return None                           # freeway merge / diverge: lane changing on links handles it
    elif "signal" in tag:
        ctrl = ixd.IX_SIGNAL
    elif "stop" in tag:
        ctrl = ixd.IX_AWSC if len(spec) >= 4 and widest <= 1 else ixd.IX_TWSC
    elif "give_way" in tag or "yield" in tag or "roundabout" in tag:
        ctrl = ixd.IX_YIELD
    else:
        ctrl = ixd.IX_TWSC
    d = ixd.build(spec, ctrl, ped_rate=[peds if s["lanes"] or s["n_out"] else 0 for s in spec],
                  kmax=kmax, pmax=pmax,
                  left_mode=lambda st: "prot_perm")
    d["legs_meta"] = legs
    return d


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("network")
    p.add_argument("--control", default="auto", choices=["auto", "signal", "awsc", "twsc", "yield"])
    p.add_argument("--peds", type=float, default=0.0, help="pedestrians per hour per crosswalk")
    a = p.parse_args()
    nodes, edges = read_network(a.network)
    by_in, by_out = {}, {}
    for e in edges:
        by_in.setdefault(e["v"], []).append(e)
        by_out.setdefault(e["u"], []).append(e)
    rows = {k: [] for k in ("nodes", "legs", "paths", "conflicts", "phases")}
    curves = edge_curves(nodes, edges)
    counts = {}
    for n in nodes:
        try:
            d = node_design(nodes, n, by_in.get(n, []), by_out.get(n, []), a.control, a.peds)
        except AssertionError as err:            # too many lanes / paths for the fixed sizes
            print(f"skip node {n}: {err}")
            continue
        if d is None:
            continue
        counts[d["ctrl"]] = counts.get(d["ctrl"], 0) + 1
        nP, nph = int(d["n_paths"]), int(d["n_phase"])
        rows["nodes"].append([n, d["ctrl"], len(d["legs_meta"]), nP, nph, int(d["rtor"]), round(float(np.max(d["D"])), 3),
                              round(float(d["phase_dur"][:nph].sum()), 3)])
        for leg, lg in enumerate(d["legs_meta"]):
            rows["legs"].append([n, leg, round(lg["angle"], 3), lg["e_in"]["id"] if lg["e_in"] else -1,
                                 lg["e_out"]["id"] if lg["e_out"] else -1, int(d["major"][leg]),
                                 round(float(d["ped_rate"][leg]) * 3600, 3), round(float(d["ped_cross_time"][leg]), 3),
                                 round(float(d["D"][leg]), 3)])
        for i in range(nP):
            a_, b_ = int(d["p_leg"][i]), int(d["p_exit"][i])
            e_out = d["legs_meta"][b_]["e_out"]
            rows["paths"].append([n, i, a_, int(d["p_lane"][i]), int(d["p_mov"][i]), b_,
                                  int(d["p_exit_lane"][i]) + start_lanes(e_out)[0],
                                  d["legs_meta"][a_]["e_in"]["id"], d["legs_meta"][b_]["e_out"]["id"],
                                  round(float(d["p_len"][i]), 3), round(float(d["s_wait"][i]), 3), int(d["n_stage"][i])])
            for q in range(nP):
                if d["c_kind"][i, q]:
                    rows["conflicts"].append([n, i, q, int(d["c_kind"][i, q]), round(float(d["c_in"][i, q]), 3),
                                              round(float(d["c_out"][i, q]), 3)])
        for k in range(nph):
            walk = sum(1 << leg for leg in range(len(d["legs_meta"])) if d["phase_walk"][k, leg])
            rows["phases"].append([n, k, round(float(d["phase_dur"][k]), 3), walk,
                                   "".join(str(int(s)) for s in d["phase_state"][k, :nP])])
    headers = {
        "nodes": ["node", "ctrl", "n_legs", "n_paths", "n_phase", "rtor", "D", "cycle"],
        "legs": ["node", "leg", "angle_deg", "in_edge", "out_edge", "major", "ped_rate_ph", "cross_time_s", "setback_m"],
        "paths": ["node", "path", "leg", "lane", "mov", "exit_leg", "exit_lane", "in_edge", "out_edge", "length_m",
                  "s_wait_m", "n_stage"],
        "conflicts": ["node", "p", "q", "kind", "p_in", "p_out"],
        "phases": ["node", "phase", "duration_s", "walk_mask", "states"],
    }
    for k, rs in rows.items():
        with open(os.path.join(a.network, f"ix_{k}.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(headers[k])
            w.writerows(rs)
    with open(os.path.join(a.network, "edge_curves.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["edge", "step_m", "caps"])
        w.writerows(curves)
    names = {v: k for k, v in CTRL_NAMES.items()}
    print(f"{len(rows['nodes'])} intersections ({', '.join(f'{names[c]}: {n}' for c, n in sorted(counts.items()))}), "
          f"{len(rows['paths'])} paths, {len(rows['conflicts'])} conflict entries, {len(curves)} curved edges "
          f"-> {a.network}/ix_*.csv, edge_curves.csv")


def edge_curves(nodes, edges, step=10.0):
    """Curve speed caps (m/s, every ``step`` m from the edge start) for edges with geometry whose
    curvature limits speed below the edge's limit."""
    out = []
    for e in edges:
        if not e["geometry"]:
            continue
        pts = to_local(nodes, e["u"], e["geometry"])
        cl = ixd.Centreline(pts - pts[0], length=max(e["length"], 1.0) + 50.0)
        L = e["length"]
        n = int(L // step) + 1
        x = np.arange(n) * step
        grid = np.arange(0.0, L + 25.0, 1.0)
        h = np.unwrap(np.radians(cl.heading(grid)))
        kap = np.abs(np.interp(grid + 10.0, grid, h) - np.interp(grid - 10.0, grid, h)) / 20.0
        cap = np.where(kap > 1e-5, np.sqrt(ixd.A_LAT / np.maximum(kap, 1e-5)), ixd.VCAP_FREE)
        cap = np.minimum(np.interp(x, grid, cap), ixd.VCAP_FREE)
        for k in range(n - 2, -1, -1):
            cap[k] = min(cap[k], math.sqrt(cap[k + 1] ** 2 + 2 * ixd.B_CURVE * step))
        if cap.min() < e["speed"] * 0.44704:
            out.append([e["id"], step, ";".join(f"{c:.1f}" for c in cap)])
    return out


if __name__ == "__main__":
    main()
