"""Record vehicle behaviour in every test scenario and write an interactive HTML viewer.

    python -m lcrl.visualize --policy runs/ppo_v4/policy.pt --out viz/lane_change_viewer.html

The rule baseline and the RL policy are run on the same randomised geometry and demand
for each scenario (``--variants`` geometries per scenario). The output is one
self-contained HTML file (viz/viewer.html with the recording embedded); open it in a
browser. ``--json`` additionally writes the raw recording, which the viewer can also
load by drag-and-drop.
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from .config import LON_DELTAS, SCENARIO_NAMES, EnvConfig
from .env import LaneChangeEnv
from .graph import GraphedEnv
from .evaluate import metrics_from_stats

TEMPLATE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "viz", "viewer.html")
PLACEHOLDER = "/*__LCRL_DATA__*/null"

# per vehicle-frame flag bits (decoded in viz/viewer.html)
F_LC, F_START, F_ABORT = 1, 2, 4          # changing lanes / started this frame / aborted this frame
F_SIG_L, F_SIG_R = 8, 16                  # turn indicator
F_BLOCKED, F_COMMITTED = 32, 64           # stop line closed for this vehicle / committed to cross
LON_SHIFT = 7                             # 3 bits: longitudinal action index
# vehicle outcomes
OUT_SUCCESS, OUT_MISSED, OUT_COLLISION, OUT_TIMEOUT = 1, 2, 3, 4


def _static(env, e, scenario, variant):
    lanes = []
    for k in range(env.K):
        s, t = env.lane_start[e, k].item(), env.lane_end[e, k].item()
        if s < t:
            lanes.append({"k": k, "start": round(s, 1), "end": round(t, 1), "mov": int(env.lane_mov[e, k])})
    return {"scenario": SCENARIO_NAMES[scenario], "variant": variant, "seg_len": env.seg_len[e].item(),
            "limit": round(env.limit[e].item(), 2), "ctrl": int(env.ctrl[e]), "lanes": lanes,
            "spawn_rate_vph": [round(3600 * env.spawn_rate[e, k].item()) for k in range(env.K)]}


@torch.no_grad()
def record(policy, scenario_ids, seed=7, warmup=240, steps=400, device="cuda"):
    """Run one policy ("rule" or an ActorCritic) on one segment per entry of scenario_ids.

    Returns a list (one per segment) of dicts with static geometry, per-frame vehicle
    arrays and per-vehicle attributes. Frame f is the state at t = f*dt *before* the
    action chosen at that frame is applied.
    """
    E = len(scenario_ids)
    cfg = EnvConfig(num_envs=E, episode_s=(warmup + steps + 10) * EnvConfig.decision_dt)
    env = LaneChangeEnv(cfg, device=device, seed=seed)
    env.reset(scenario_ids=scenario_ids)
    variant = {}
    out = []
    for e, s in enumerate(scenario_ids):
        variant[s] = variant.get(s, 0) + 1
        out.append({"static": _static(env, e, s, variant[s]), "frames": [], "signal": [], "vehicles": {}})

    env = GraphedEnv(env)                    # each decision replays one CUDA graph

    def act(obs, mask):
        if policy == "rule":
            return env.rule_action(obs, mask)
        lat, lon, _ = policy.act(obs, mask, deterministic=True)
        return lat, lon

    for _ in range(warmup):
        obs, mask, _ = env.observe()
        env.step(*act(obs, mask))
    env.stats.zero_()

    uid = np.full((E, env.N), -1, np.int64)
    next_uid = np.zeros(E, np.int64)
    prev_active = np.zeros((E, env.N), bool)
    prev_age = np.zeros((E, env.N), np.int64)
    for f in range(steps):
        obs, mask, valid = env.observe()
        lat, lon = act(obs, mask)
        C = env._cache
        pre_from, pre_to = env.lane_from.clone(), env.lane_to.clone()
        snap = {k: getattr(env, k).cpu().numpy() for k in ("active", "x", "v", "prog", "age", "aggr", "info", "vdes", "mov")}
        snap["from"], snap["to"] = pre_from.cpu().numpy(), pre_to.cpu().numpy()
        need, blocked = C["need"].cpu().numpy(), C["blocked"].cpu().numpy()
        committed = env.committed.cpu().numpy()
        lon_np = lon.cpu().numpy()
        tc = torch.remainder(env.t_env * cfg.dt + env.sig_offset, env.cycle)
        g_thru = (tc < env.g_thru).cpu().numpy()
        g_left = ((tc >= env.g_thru) & (tc < env.g_thru + env.g_left)).cpu().numpy()
        busy = env.conflict_busy.cpu().numpy()

        env.step(lat, lon)
        post_from, post_to = env.lane_from.cpu().numpy(), env.lane_to.cpu().numpy()
        oc = {k: m.cpu().numpy() for k, m in env.last_outcome.items()}

        for e in range(E):
            act_e = snap["active"][e]
            new = act_e & (~prev_active[e] | (snap["age"][e] < prev_age[e]))
            for n in np.nonzero(new)[0]:
                uid[e, n] = next_uid[e]
                next_uid[e] += 1
                out[e]["vehicles"][int(uid[e, n])] = {
                    "aggr": round(float(snap["aggr"][e, n]), 2), "info": round(float(snap["info"][e, n])),
                    "vdes": round(float(snap["vdes"][e, n]), 1), "mov": int(snap["mov"][e, n]),
                    "spawn": f, "end": None, "outcome": None}
            row = []
            for n in np.nonzero(act_e)[0]:
                fr, to, p = int(snap["from"][e, n]), int(snap["to"][e, n]), float(snap["prog"][e, n])
                flags = int(lon_np[e, n]) << LON_SHIFT
                if fr != to:
                    flags |= F_LC
                started = fr == to and post_to[e, n] != post_from[e, n]
                aborted = fr != to and post_from[e, n] == to and post_to[e, n] == fr
                if started:
                    flags |= F_START
                if aborted:
                    flags |= F_ABORT
                ind = (to - fr) if fr != to else int(np.sign(need[e, n]))
                if started:
                    ind = int(post_to[e, n] - post_from[e, n])
                flags |= F_SIG_L if ind < 0 else F_SIG_R if ind > 0 else 0
                if blocked[e, n]:
                    flags |= F_BLOCKED
                if committed[e, n]:
                    flags |= F_COMMITTED
                y = fr + (to - fr) * p
                row += [int(uid[e, n]), round(float(snap["x"][e, n]), 1), round(y, 2),
                        round(float(snap["v"][e, n]), 1), flags]
                for name, code in (("success", OUT_SUCCESS), ("missed", OUT_MISSED),
                                   ("collided", OUT_COLLISION), ("timeout", OUT_TIMEOUT)):
                    if oc[name][e, n]:
                        veh = out[e]["vehicles"][int(uid[e, n])]
                        veh["end"], veh["outcome"] = f, code
            out[e]["frames"].append(row)
            out[e]["signal"].append(int(g_thru[e]) | int(g_left[e]) << 1 | int(busy[e]) << 2)
        prev_active, prev_age = snap["active"], snap["age"]

    for e in range(E):
        st = env.stats[e].tolist()
        out[e]["summary"] = metrics_from_stats(st, 1, steps * cfg.decision_dt, cfg.dt)
    return out


def build(policies, scenarios, variants=1, seed=7, warmup=240, steps=400, ix_kinds=(), sig4_levels=(),
          sig4_steps=None):
    """Record road-segment scenarios and (optionally) complete intersections for each policy.
    The 4-leg signal (kind 0) is recorded once per demand level in ``sig4_levels``
    (veh/h per leg) when given, instead of as a random design."""
    from .visualize_ix import record_ix, signal4_sweep_designs
    ids = [s for s in scenarios for _ in range(variants)]
    sweep = 0 in ix_kinds and len(sig4_levels) > 0
    kinds = [k for k in ix_kinds if not (sweep and k == 0) for _ in range(variants)]
    runs = {}
    for name, pol in policies.items():
        runs[name] = record(pol, ids, seed=seed, warmup=warmup, steps=steps) if ids else []
        for r in runs[name]:
            r["static"]["type"] = "segment"
        if sweep:
            runs[name] += record_ix(pol, seed=seed, warmup=warmup, steps=sig4_steps or steps,
                                    designs=signal4_sweep_designs(sig4_levels))
        if kinds:
            runs[name] += record_ix(pol, kinds, seed=seed, warmup=warmup, steps=steps)
    return {"meta": {"dt": EnvConfig.decision_dt, "physics_dt": EnvConfig.dt, "veh_len": EnvConfig.veh_len, "steps": steps, "warmup_s": warmup * EnvConfig.decision_dt,
                     "seed": seed, "created": time.strftime("%Y-%m-%d %H:%M"),
                     "lon_deltas": list(LON_DELTAS), "policies": list(runs),
                     "ctrl_names": ["free flow", "signal", "stop sign", "yield sign"],
                     "ix_ctrl_names": {"1": "signal", "2": "all-way stop", "3": "two-way stop", "4": "yield"},
                     "mov_names": ["left", "through", "right / exit"]},
            "runs": runs}


def write_html(data, out_path, template=TEMPLATE):
    with open(template, encoding="utf-8") as f:
        html = f.read()
    assert PLACEHOLDER in html, "viewer template is missing the data placeholder"
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html.replace(PLACEHOLDER, payload))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--policy", default="runs/ppo_v6_ix/policy.pt", help="trained policy; 'none' = rule baseline only")
    p.add_argument("--out", default="viz/lane_change_viewer.html")
    p.add_argument("--json", default=None, help="also write the raw recording here")
    p.add_argument("--scenarios", default="freeway", help="road-segment scenarios: 'freeway', 'all', 'none' or names/indices")
    p.add_argument("--intersections", default="all", help="intersection kinds: 'all', 'none' or names/indices")
    p.add_argument("--variants", type=int, default=1, help="random geometries per scenario")
    p.add_argument("--seconds", type=float, default=200.0, help="recorded duration")
    p.add_argument("--warmup", type=float, default=120.0, help="seconds simulated before recording")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--sig4-seconds", type=float, default=900.0, help="recorded duration of the 4-leg signal study")
    p.add_argument("--demand-levels", default="500",
                   help="4-leg signal demand levels in veh/h per leg (start:stop:step or a,b,c); 'none' = one random design")
    a = p.parse_args()
    if a.scenarios in ("all", "freeway", "none"):
        scen = {"all": list(range(len(SCENARIO_NAMES))), "freeway": [0, 1, 2, 3, 8], "none": []}[a.scenarios]
    else:
        scen = [int(s) if s.isdigit() else SCENARIO_NAMES.index(s) for s in a.scenarios.split(",")]
    from .visualize_ix import ix_kind_ids
    kinds = ix_kind_ids(a.intersections)
    pols = {"rule": "rule"}
    if a.policy != "none":
        from .train import load
        pols["rl"] = load(a.policy)
    dt = EnvConfig.decision_dt
    if a.demand_levels == "none":
        levels = []
    elif ":" in a.demand_levels:
        lo, hi, st = (int(x) for x in a.demand_levels.split(":"))
        levels = list(range(lo, hi + 1, st))
    else:
        levels = [int(x) for x in a.demand_levels.split(",")]
    data = build(pols, scen, a.variants, a.seed, round(a.warmup / dt), round(a.seconds / dt), kinds, levels,
                 round(a.sig4_seconds / dt))
    write_html(data, a.out)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(data, f, separators=(",", ":"))
    print(f"wrote {a.out} ({os.path.getsize(a.out) / 1e6:.1f} MB)")
