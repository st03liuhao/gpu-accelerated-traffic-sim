import csv
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

from lcrl import intersection as ixd
from lcrl.config import MOV_LEFT, MOV_RIGHT, MOV_THROUGH, IxConfig
from lcrl.env import S
from lcrl.ix_env import KIND_NAMES, IntersectionEnv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_turn_classes_orthogonal():
    # from the east leg (vehicles heading west): north is a right turn, south a left turn
    assert ixd.turn_class(0, 90) == MOV_RIGHT
    assert ixd.turn_class(0, 270) == MOV_LEFT
    assert ixd.turn_class(0, 180) == MOV_THROUGH
    assert ixd.turn_class(0, 5) == -1          # U-turn


@pytest.mark.parametrize("kind", KIND_NAMES)
def test_design_invariants(kind):
    rng = np.random.default_rng(0)
    for _ in range(10):
        d = ixd.design(rng, kind)
        n = d["n_paths"]
        k = d["c_kind"][:n, :n]
        assert (k == k.T).all()                          # conflicts are symmetric
        assert (np.diag(k) == 0).all()
        for p in range(n):
            a, lane, m = d["p_leg"][p], d["p_lane"][p], d["p_mov"][p]
            assert d["lane_mov"][a, lane] >> m & 1           # path served by its lane
            assert d["legs"][d["p_exit"][p]]                 # exits into an existing leg
            assert 0 <= d["p_exit_lane"][p] < d["n_out"][d["p_exit"][p]]
            for q in range(n):                               # zones lie on the paths
                if k[p, q]:
                    assert 0 <= d["c_in"][p, q] <= d["c_out"][p, q] <= d["p_len"][p] + 1e-3
        if d["ctrl"] == ixd.IX_SIGNAL:
            st = d["phase_state"][:d["n_phase"], :n]
            # no two conflicting crossing paths are both protected green in the same phase
            for ph in st:
                g = np.nonzero(ph == ixd.SIG_G)[0]
                assert not any(k[p, q] == ixd.C_CROSS for p in g for q in g)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_ix_env_rule_baseline_safe():
    env = IntersectionEnv(IxConfig(num_envs=48, pool_per_kind=4), seed=5)
    env.reset(kind_ids=torch.arange(48) % len(KIND_NAMES))
    for _ in range(600):
        obs, mask, valid = env.observe()
        assert torch.isfinite(obs).all()
        assert mask[..., 0].all()
        env.step(*env.rule_action(obs, mask))
    st = env.stats.sum(0)
    assert st[S["exited"]] > 300
    assert st[S["collisions"]] == 0
    assert st[S["violations"]] == 0
    assert st[S["timeouts"]] == 0


def test_lpsim_tool_tables(tmp_path):
    out = tmp_path / "net"
    subprocess.run([sys.executable, os.path.join(ROOT, "tools", "make_ix_network.py"), str(out),
                    "--legs", "4", "--lanes", "2", "--control", "signal", "--vph", "100", "--hours", "0.1"],
                   check=True, capture_output=True)
    rows = lambda name: list(csv.DictReader(open(out / f"ix_{name}.csv")))
    nodes, paths, phases = rows("nodes"), rows("paths"), rows("phases")
    edges = {int(r["uniqueid"]) for r in csv.DictReader(open(out / "edges.csv"))}
    assert len(nodes) == 1 and int(nodes[0]["n_paths"]) == len(paths)
    assert all(int(p["in_edge"]) in edges and int(p["out_edge"]) in edges for p in paths)
    assert all(len(ph["states"]) == len(paths) for ph in phases)
    assert {int(p["mov"]) for p in paths} == {MOV_LEFT, MOV_THROUGH, MOV_RIGHT}


def test_signal_4leg_study_design():
    """600 m legs, shared left lane, short right bay, two 30 s phases, random entry lane."""
    for q in (50.0, 500.0, 800.0):
        d = ixd.signal_4leg(None, demand_vph=q)
        assert (d["n_in"] == 3).all() and (d["L_app"] == 600).all() and (d["L_exit"] == 600).all()
        assert (d["lane_start"][:, :2] <= 0).all() and (d["lane_start"][:, 2] == 600 - ixd.SIG4_BAY_M).all()
        assert (d["lane_mov"][:, 0] == (1 << MOV_LEFT | 1 << MOV_THROUGH)).all()
        assert (d["lane_mov"][:, 2] == 1 << MOV_RIGHT).all()
        assert d["rtor"] and (d["ped_rate"] == 0).all()
        assert list(d["phase_dur"][:d["n_phase"]:3]) == [30.0, 30.0]
        nP = d["n_paths"]
        for ph in (0, 3):                                             # every movement of a street moves
            st = d["phase_state"][ph, :nP]
            on = st > 0
            assert set(d["p_leg"][:nP][on].tolist()) in ({0, 2}, {1, 3}) and on.sum() == nP // 2
            assert (st[on & (d["p_mov"][:nP] == MOV_LEFT)] == ixd.SIG_PERM).all()
        q_mov = (d["spawn_rate"][:, :, None] * d["mov_prob"]).sum((0, 1)) * 3600 / 4
        assert np.isclose(q_mov.sum(), q, rtol=1e-4)
        assert q_mov[MOV_LEFT] < 0.10 * q and q_mov[MOV_RIGHT] < 0.15 * q
        assert (d["spawn_rate"][:, 2] == 0).all() and np.allclose(d["spawn_rate"][:, 0], d["spawn_rate"][:, 1])


def test_lpsim_tool_curves_and_turn_lanes(tmp_path):
    """Curved, skewed legs and OSM turn:lanes: dual lefts into two exit lanes, protected only."""
    out = tmp_path / "net"
    subprocess.run([sys.executable, os.path.join(ROOT, "tools", "make_ix_network.py"), str(out),
                    "--legs", "4", "--lanes", "3", "--control", "signal", "--vph", "100", "--hours", "0.1",
                    "--curve", "35", "--skew", "12", "--turn-lanes", "left|left|through|through;right"],
                   check=True, capture_output=True)
    rows = lambda name: list(csv.DictReader(open(out / f"ix_{name}.csv")))
    paths, phases = rows("paths"), rows("phases")
    for leg in range(4):
        lefts = [p for p in paths if int(p["leg"]) == leg and int(p["mov"]) == MOV_LEFT]
        assert sorted(int(p["lane"]) for p in lefts) == [0, 1]
        assert sorted(int(p["exit_lane"]) for p in lefts) == [0, 1]
    left_ids = [int(p["path"]) for p in paths if int(p["mov"]) == MOV_LEFT]
    assert all(ph["states"][i] != str(ixd.SIG_PERM) for ph in phases for i in left_ids)   # dual lefts: protected
    assert all(int(p["n_stage"]) == 0 for p in paths if int(p["path"]) in left_ids)


def test_staged_permitted_turns_do_not_deadlock():
    """Permitted lefts pull into the box and wait; 10 minutes at 700 veh/h per leg without
    timeouts, collisions or violations."""
    d = ixd.signal_4leg(None, demand_vph=700.0)
    assert (d["n_stage"][d["p_mov"] == MOV_LEFT][: d["n_paths"]] >= 0).all() and d["n_stage"].sum() == 4
    env = IntersectionEnv(IxConfig(num_envs=4, max_veh=224, episode_s=2000), seed=3, designs=[d])
    env.reset(design_ids=[0, 0, 0, 0])
    from lcrl.graph import GraphedEnv
    g = GraphedEnv(env)
    staged = torch.zeros((), device="cuda")
    for _ in range(1200):
        obs, mask, _ = g.observe()
        g.step(*env.rule_action(obs, mask))
        staged += (env.staged & env.active).sum()
    assert staged.item() > 0
    assert env.stats[:, S["timeouts"]].sum() == 0
    assert env.stats[:, S["collisions"]].sum() == 0 and env.stats[:, S["violations"]].sum() == 0
    assert env.stats[:, S["exited"]].sum() > 4 * 250
