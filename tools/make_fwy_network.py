"""Make a freeway LPSim network for testing lane spans, lane links and the route-aware segment.

    python tools/make_fwy_network.py out_dir [--vph 3600] [--ramp-vph 600] [--exit-share 0.15] [--hours 1]
                                     [--no-lanes]

A 3-lane freeway, eastbound, split into several edges:

    A --200-- B --600-- C --900-- D --700-- E --600-- F --200-- H
                       /                     \\
          R0 --300-- R1 (on-ramp)             G (off-ramp, 300 m)

  * B and D are plain nodes: every lane continues (the RL segment runs through them).
  * C is the on-ramp merge: C->D has a 4th lane, lanes 0-2 from B->C and lane 3 from the ramp,
    which is an acceleration lane that ends 250 m after the merge.
  * D->E has a 4th lane that starts 250 m before E: the deceleration lane of the off-ramp at E.
    Only that lane continues into the ramp, only lanes 0-2 continue on the freeway.

Every edge also has a reverse twin without demand (a westbound carriageway and reversed ramps):
LPSim's lane-map setup crashes when a node has only outgoing or only incoming edges.

Writes nodes.csv / edges.csv / od_demand.csv (LPSim format), edge_lanes.csv (lane spans) and
lane_links.csv (which lane of the next edge each lane continues into). ``--no-lanes`` leaves the
two sidecar files out: full-length lanes, same lane index across nodes (LPSim's default).
Demand: Poisson, mainline A->H and A->G (exit share), ramp R0->H.
"""
import argparse
import csv
import math
import os
import random

LON0, LAT0 = -122.3, 37.85
M_LAT, M_LON = 111_320.0, 111_320.0 * math.cos(math.radians(LAT0))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out")
    p.add_argument("--vph", type=float, default=3600.0, help="mainline vehicles per hour")
    p.add_argument("--ramp-vph", type=float, default=600.0)
    p.add_argument("--exit-share", type=float, default=0.15, help="share of mainline vehicles taking the off-ramp")
    p.add_argument("--hours", type=float, default=1.0)
    p.add_argument("--start-hour", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--no-lanes", action="store_true", help="no edge_lanes.csv / lane_links.csv")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    xy = {"A": (0, 0), "B": (200, 0), "C": (800, 0), "D": (1700, 0), "E": (2400, 0), "F": (3000, 0), "H": (3200, 0),
          "R0": (520, -120), "G": (2680, -110)}
    ids = {n: i for i, n in enumerate(xy)}
    tag = {"C": "motorway_junction", "E": "motorway_junction"}
    # (u, v, length, lanes, mph)
    fwy, ramp = 65, 45
    edges = [("A", "B", 200, 3, fwy), ("B", "C", 600, 3, fwy), ("R0", "C", 300, 1, ramp), ("C", "D", 900, 4, fwy),
             ("D", "E", 700, 4, fwy), ("E", "F", 600, 3, fwy), ("E", "G", 300, 1, ramp), ("F", "H", 200, 3, fwy)]
    uid = {(u, v): 100 + i for i, (u, v, *_) in enumerate(edges)}
    twins = [(v, u, L, n, mph) for u, v, L, n, mph in edges]          # reverse carriageway, no demand
    uid.update({(u, v): 200 + i for i, (u, v, *_) in enumerate(twins)})
    with open(os.path.join(a.out, "nodes.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["osmid", "x", "y", "ref", "highway"])
        for n, (x, y) in xy.items():
            w.writerow([ids[n], f"{LON0 + x / M_LON:.8f}", f"{LAT0 + y / M_LAT:.8f}", "", tag.get(n, "")])
    with open(os.path.join(a.out, "edges.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["uniqueid", "u", "v", "length", "lanes", "speed_mph"])
        for u, v, L, n, mph in edges + twins:
            w.writerow([uid[(u, v)], ids[u], ids[v], L, n, mph])
    for name in ("edge_lanes.csv", "lane_links.csv"):
        if os.path.exists(os.path.join(a.out, name)):
            os.remove(os.path.join(a.out, name))
    if not a.no_lanes:
        with open(os.path.join(a.out, "edge_lanes.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["uniqueid", "lane", "start_m", "end_m"])
            for k in range(3):
                w.writerow([uid[("C", "D")], k, 0, 900])
                w.writerow([uid[("D", "E")], k, 0, 700])
            w.writerow([uid[("C", "D")], 3, 0, 250])        # acceleration lane
            w.writerow([uid[("D", "E")], 3, 450, 700])      # deceleration lane
        links = [(("A", "B"), ("B", "C"), 0), (("B", "C"), ("C", "D"), 0), (("R0", "C"), ("C", "D"), 3),
                 (("C", "D"), ("D", "E"), 0), (("D", "E"), ("E", "F"), 0), (("D", "E"), ("E", "G"), -3),
                 (("E", "F"), ("F", "H"), 0)]
        with open(os.path.join(a.out, "lane_links.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["in_edge", "out_edge", "shift"])
            for e_in, e_out, sh in links:
                w.writerow([uid[e_in], uid[e_out], sh])
    rng = random.Random(a.seed)
    t0, t1 = a.start_hour * 3600, (a.start_hour + a.hours) * 3600
    trips = []
    for origin, rate in (("A", a.vph), ("R0", a.ramp_vph)):
        t = t0
        while True:
            t += rng.expovariate(rate / 3600.0)
            if t >= t1:
                break
            dest = "G" if origin == "A" and rng.random() < a.exit_share else "H"
            trips.append((round(t, 1), ids[origin], ids[dest]))
    trips.sort()
    with open(os.path.join(a.out, "od_demand.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dep_time", "origin", "destination"])
        w.writerows(trips)
    print(f"{len(xy)} nodes, {len(edges) + len(twins)} edges, {len(trips)} trips"
          f"{'' if a.no_lanes else ', lane spans and lane links'} -> {a.out}")


if __name__ == "__main__":
    main()
