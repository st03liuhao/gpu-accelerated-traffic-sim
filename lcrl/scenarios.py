"""Randomised road-segment scenarios used for training and testing.

Each environment is one directed segment (one LPSim edge) of length ``seg_len`` with
up to ``max_lanes`` lanes. A lane exists on [lane_start, lane_end]; lanes that end
before ``seg_len`` are dropped lanes / acceleration lanes, lanes that start after 0
are added lanes / turn bays / deceleration lanes. At the segment end each lane allows
a set of movements (bitmask over MOV_LEFT / MOV_THROUGH / MOV_RIGHT) and the
downstream control is free flow, a signal, a stop sign or a yield sign.
"""
import numpy as np

from .config import (
    CTRL_FREE, CTRL_SIGNAL, CTRL_STOP, CTRL_YIELD,
    FREEWAY_SCENARIOS, FWY_LANE_ADD, FWY_LANE_DROP, FWY_OFFRAMP, FWY_ONRAMP, FWY_WEAVE,
    MOV_LEFT, MOV_RIGHT, MOV_THROUGH, NUM_MOV,
    SIG_DEDICATED, SIG_SHARED, STOP_SIGN, YIELD_SIGN,
)

L, T, R = 1 << MOV_LEFT, 1 << MOV_THROUGH, 1 << MOV_RIGHT
NO_LANE = (1e9, -1e9)


def _empty(max_lanes):
    return dict(
        seg_len=0.0, speed_limit=0.0, ctrl=CTRL_FREE,
        lane_start=np.full(max_lanes, NO_LANE[0], np.float32),
        lane_end=np.full(max_lanes, NO_LANE[1], np.float32),
        lane_mov=np.zeros(max_lanes, np.int64),
        spawn_rate=np.zeros(max_lanes, np.float32),       # veh/s
        mov_prob=np.tile(np.eye(NUM_MOV, dtype=np.float32)[MOV_THROUGH], (max_lanes, 1)),
        cycle=60.0, g_thru=30.0, g_left=12.0, sig_offset=0.0,
        conflict_on=0.0, conflict_off=1.0,                # Markov switching per step
    )


def _main_lanes(s, lanes, seg_len, mov=T):
    for k in lanes:
        s["lane_start"][k], s["lane_end"][k], s["lane_mov"][k] = 0.0, seg_len, mov


def generate(scenario, rng, max_lanes=6, demand_scale=1.0):
    """Return a dict describing one randomised segment of the given scenario type."""
    s = _empty(max_lanes)
    s["scenario"] = scenario
    if scenario in FREEWAY_SCENARIOS:
        # segments end at a deadline (off-ramp, lane drop) are up to 2 km long, so that
        # vehicles learn to prepare from far away; the exit is known from the entry
        long_ = scenario in (FWY_OFFRAMP, FWY_LANE_DROP)
        seg_len, m = float(rng.uniform(600.0, 2000.0 if long_ else 1200.0)), int(rng.integers(2, 5))
        if scenario == FWY_WEAVE:
            x_on, weave = float(rng.uniform(100.0, 200.0)), float(rng.uniform(300.0, 900.0))
            seg_len = x_on + weave
        s["seg_len"], s["speed_limit"] = seg_len, float(rng.uniform(25.0, 33.0))
        _main_lanes(s, range(m), seg_len)
        lane_rate = rng.uniform(0.12, 0.45) * demand_scale     # 430-1620 veh/h/lane
        s["spawn_rate"][:m] = lane_rate
        if scenario == FWY_ONRAMP:
            start = float(rng.uniform(100.0, 200.0))
            s["lane_start"][m], s["lane_end"][m] = start, start + float(rng.uniform(150.0, 300.0))
            s["spawn_rate"][m] = rng.uniform(0.08, 0.3) * demand_scale
        elif scenario == FWY_OFFRAMP:
            s["lane_start"][m], s["lane_end"][m] = seg_len - float(rng.uniform(150.0, 300.0)), seg_len
            s["lane_mov"][m] = R
            exit_share = float(rng.uniform(0.15, 0.4))
            s["mov_prob"][:m] = [0.0, 1.0 - exit_share, exit_share]
        elif scenario == FWY_LANE_DROP:
            s["lane_end"][m - 1], s["lane_mov"][m - 1] = seg_len - float(rng.uniform(150.0, 300.0)), 0
        elif scenario == FWY_LANE_ADD:
            s["lane_start"][m], s["lane_end"][m], s["lane_mov"][m] = float(rng.uniform(100.0, 250.0)), seg_len, T
        elif scenario == FWY_WEAVE:
            # auxiliary lane from the on-ramp at x_on to the off-ramp at the segment end: ramp
            # vehicles going on must leave it to the left, exiting mainline vehicles enter it
            s["lane_start"][m], s["lane_end"][m], s["lane_mov"][m] = x_on, seg_len, R
            s["spawn_rate"][m] = rng.uniform(0.08, 0.3) * demand_scale
            exit_share, ramp_to_ramp = float(rng.uniform(0.1, 0.3)), float(rng.uniform(0.1, 0.3))
            s["mov_prob"][:m] = [0.0, 1.0 - exit_share, exit_share]
            s["mov_prob"][m] = [0.0, 1.0 - ramp_to_ramp, ramp_to_ramp]
        return s

    # urban approaches
    seg_len = 300.0
    s["seg_len"], s["speed_limit"] = seg_len, float(rng.uniform(11.0, 17.0))
    turn = rng.dirichlet([2.0, 6.0, 2.0]).astype(np.float32)   # left/through/right shares
    if scenario == SIG_DEDICATED:
        m = int(rng.integers(1, 4))
        _main_lanes(s, range(1, m + 1), seg_len)
        s["lane_start"][0], s["lane_end"][0], s["lane_mov"][0] = seg_len - float(rng.uniform(50.0, 120.0)), seg_len, L
        s["lane_start"][m + 1], s["lane_end"][m + 1], s["lane_mov"][m + 1] = seg_len - float(rng.uniform(40.0, 90.0)), seg_len, R
        s["spawn_rate"][1:m + 1] = rng.uniform(0.08, 0.3) * demand_scale
        s["mov_prob"][:] = turn
    else:
        m = int(rng.integers(2, 4))
        _main_lanes(s, range(m), seg_len)
        s["lane_mov"][0] |= L
        s["lane_mov"][m - 1] |= R
        s["spawn_rate"][:m] = rng.uniform(0.06, 0.25) * demand_scale
        s["mov_prob"][:] = turn
        if scenario == SIG_SHARED:
            s["ctrl"] = CTRL_SIGNAL
        elif scenario == STOP_SIGN:
            s["ctrl"] = CTRL_STOP
            s["spawn_rate"][:m] *= 0.6
        elif scenario == YIELD_SIGN:
            s["ctrl"] = CTRL_YIELD
        if scenario in (STOP_SIGN, YIELD_SIGN):
            # conflicting-stream occupancy: busy fraction ~ on / (on + off)
            s["conflict_on"], s["conflict_off"] = float(rng.uniform(0.05, 0.25)), float(rng.uniform(0.2, 0.5))
    if scenario in (SIG_DEDICATED, SIG_SHARED):
        s["ctrl"] = CTRL_SIGNAL
        s["cycle"] = float(rng.uniform(50.0, 100.0))
        s["g_left"] = float(rng.uniform(0.15, 0.25)) * s["cycle"]
        s["g_thru"] = float(rng.uniform(0.35, 0.55)) * s["cycle"]
        s["sig_offset"] = float(rng.uniform(0.0, s["cycle"]))
    return s
