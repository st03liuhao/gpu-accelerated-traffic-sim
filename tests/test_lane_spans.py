"""Lane spans and lane links written by LPSim's tools/generate_network.py --lane-spans."""
import importlib.util
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("generate_network",
                                              os.path.join(ROOT, "external", "LPSim", "tools", "generate_network.py"))
gn = importlib.util.module_from_spec(spec)
sys.modules["generate_network"] = gn       # dataclasses look the module up while it loads
spec.loader.exec_module(gn)


def test_left_turn_bay_goes_on_the_left():
    # 150 m with 2 lanes, then 60 m with 3 lanes tagged left|through|through;right
    n, spans = gn._lane_spans([(150.0, 2, ""), (60.0, 3, "left|through|through;right")], 210.0, 1)
    assert n == 3
    assert spans == ((150.0, 210.0), (0.0, 210.0), (0.0, 210.0))


def test_acceleration_lane_goes_on_the_right():
    # 250 m with 4 lanes after a merge, then 650 m with 3 lanes, no turn tags
    n, spans = gn._lane_spans([(250.0, 4, ""), (650.0, 3, "")], 900.0, 1)
    assert n == 4
    assert spans == ((0.0, 900.0), (0.0, 900.0), (0.0, 900.0), (0.0, 250.0))


def test_lane_merging_to_the_right_ends_on_the_left():
    n, spans = gn._lane_spans([(300.0, 3, "merge_to_right|none|none"), (300.0, 2, "")], 600.0, 1)
    assert n == 3
    assert spans == ((0.0, 300.0), (0.0, 600.0), (0.0, 600.0))


def test_untagged_pieces_take_a_neighbours_count():
    n, spans = gn._lane_spans([(100.0, None, ""), (100.0, 2, ""), (100.0, None, "")], 300.0, 1)
    assert n == 2 and spans == ((0.0, 300.0), (0.0, 300.0))


def _edge(u, v, length, spans, xy):
    (x0, y0), (x1, y1) = xy[u], xy[v]
    return gn.NormalizedEdge(osmid_u=u, osmid_v=v, length=length, lanes=len(spans), speed_mph=60.0, source_key="0",
                             geometry=f"LINESTRING ({x0} {y0}, {x1} {y1})", spans=tuple(spans))


def test_lane_links_at_a_merge_and_a_diverge():
    # eastbound freeway (lon grows to the east); the ramps are south of it, i.e. on the right
    d = 0.001
    xy = {1: (0.0, 0.0), 2: (5 * d, 0.0), 3: (3 * d, -d), 7: (7.5 * d, 0.0), 4: (10 * d, 0.0), 5: (12 * d, -d),
          6: (15 * d, 0.0)}
    full = lambda n, L: [(0.0, L)] * n
    edges = [_edge(1, 2, 440.0, full(3, 440.0), xy),                       # 0 mainline in
             _edge(3, 2, 200.0, full(1, 200.0), xy),                       # 1 on-ramp
             _edge(2, 7, 220.0, full(3, 220.0) + [(0.0, 150.0)], xy),      # 2 with an acceleration lane
             _edge(7, 4, 220.0, full(3, 220.0) + [(70.0, 220.0)], xy),     # 3 with a deceleration lane
             _edge(4, 6, 440.0, full(3, 440.0), xy),                       # 4 mainline out
             _edge(4, 5, 200.0, full(1, 200.0), xy)]                       # 5 off-ramp
    links = {(i, o): s for i, o, s in gn._lane_links(edges, xy)}
    assert links[(0, 2)] == 0          # merge: the mainline keeps lanes 0-2
    assert links[(1, 2)] == 3          # merge: ramp lane 0 -> lane 3, the acceleration lane on the right
    assert links[(2, 3)] == 0          # one road on: lanes 0-2 continue
    assert links[(3, 4)] == 0          # diverge: lanes 0-2 go on
    assert links[(3, 5)] == -3         # diverge: lane 3 -> off-ramp lane 0


def test_full_pipeline_on_a_synthetic_osm_graph(tmp_path):
    ox = pytest.importorskip("osmnx")
    import networkx as nx
    g = nx.MultiDiGraph(crs="epsg:4326")
    d = 0.0009          # ~100 m of latitude
    pts = {1: (0, 0), 2: (0, 1.5 * d), 3: (0, 2.1 * d), 4: (-d, 2.1 * d), 5: (d, 2.1 * d), 6: (0, 3.1 * d)}
    for n, (x, y) in pts.items():
        g.add_node(n, x=-122.3 + x, y=37.8 + y)
    def add(u, v, length, lanes, osmid, **tags):
        g.add_edge(u, v, length=length, lanes=str(lanes), oneway=True, reversed=False, osmid=osmid,
                   highway="primary", **tags)
    add(1, 2, 150.0, 2, 10)                                                 # approach, 2 lanes
    add(2, 3, 60.0, 3, 11, **{"turn:lanes": "left|through|through;right"})  # then a left-turn bay
    add(3, 4, 100.0, 1, 12); add(3, 5, 100.0, 1, 13); add(3, 6, 100.0, 2, 14)
    add(4, 3, 100.0, 1, 15); add(5, 3, 100.0, 1, 16); add(6, 3, 100.0, 2, 17)   # node 1: a dead end (kept)
    simple = ox.simplify_graph(g, track_merged=True)
    gn.generate_network(simple, tmp_path, connectivity="all", raw=g, lane_spans=True)
    import csv
    edges = {r["uniqueid"]: r for r in csv.DictReader(open(tmp_path / "edges.csv"))}
    spans = list(csv.DictReader(open(tmp_path / "edge_lanes.csv")))
    approach = next(r for r in edges.values() if r["osmid_u"] == "1" and r["osmid_v"] == "3")
    assert approach["lanes"] == "3"
    rows = sorted((int(r["lane"]), float(r["start_m"]), float(r["end_m"])) for r in spans
                  if r["uniqueid"] == approach["uniqueid"])
    assert rows == [(0, 150.0, 210.0), (1, 0.0, 210.0), (2, 0.0, 210.0)]
    attrs = {r["uniqueid"]: r for r in csv.DictReader(open(tmp_path / "edge_attrs.csv"))}
    assert attrs[approach["uniqueid"]]["turn_lanes"] == "left|through|through;right"
