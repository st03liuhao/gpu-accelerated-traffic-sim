"""Evaluate the RL policy against the LPSim-style rule baseline on every test scenario.

    python -m lcrl.evaluate --policy runs/ppo_v1/policy.pt --out runs/ppo_v1/eval.json
"""
import argparse
import dataclasses
import json
import math

import torch

from .config import SCENARIO_NAMES, EnvConfig
from .env import S, STAT_NAMES, LaneChangeEnv
from .graph import GraphedEnv

METRICS = [  # (key, label, lower_is_better)
    ("throughput_vph", "Throughput (veh/h)", False),
    ("success_pct", "Route success (%)", False),
    ("delay_s", "Delay (s/veh)", True),
    ("collisions_per_1k", "Collisions /1k veh", True),
    ("violations_pct", "Stop-line violations (%)", True),
    ("hard_brakes_per_veh", "Hard brakes /veh", True),
    ("unsafe_s_per_veh", "TTC<1.5s (s/veh)", True),
    ("lc_per_veh", "Lane changes /veh", True),
    ("aborts_per_veh", "Aborted LCs /veh", True),
    ("induced_decel", "Lag veh max decel after cut-in (m/s2)", True),
    ("rms_jerk", "RMS jerk (m/s3)", True),
]


def metrics_from_stats(st, n_envs, seconds, dt):
    ex = max(st[S["exited"]], 1.0)
    return {
        "throughput_vph": st[S["exited"]] / n_envs / seconds * 3600.0,
        "success_pct": 100.0 * st[S["success"]] / ex,
        "delay_s": st[S["delay"]] / ex,
        "collisions_per_1k": 1000.0 * st[S["collisions"]] / ex,
        "violations_pct": 100.0 * st[S["violations"]] / ex,
        "hard_brakes_per_veh": st[S["hard_brakes"]] / ex,
        "unsafe_s_per_veh": st[S["unsafe_steps"]] * dt / ex,
        "lc_per_veh": st[S["lane_changes"]] / ex,
        "aborts_per_veh": st[S["aborts"]] / ex,
        "induced_decel": -st[S["induced_decel"]] / max(st[S["lc_events"]], 1.0),
        "rms_jerk": math.sqrt(st[S["jerk_sq"]] / max(st[S["veh_steps"]], 1.0)),
        "exited": st[S["exited"]], "timeouts": st[S["timeouts"]],
    }


@torch.no_grad()
def run(policy, envs_per_scenario=128, warmup=200, steps=1200, seed=123, fixed_aggr=None,
        fixed_info=None, device="cuda", scenarios=(0, 1, 2, 3, 8)):
    """policy: "rule" or an ActorCritic. Returns {scenario_name: metrics}."""
    scenarios = list(scenarios)
    cfg = EnvConfig(num_envs=envs_per_scenario * len(scenarios), episode_s=(warmup + steps + 10) * EnvConfig.decision_dt)
    env = LaneChangeEnv(cfg, device=device, seed=seed, fixed_aggr=fixed_aggr, fixed_info=fixed_info)
    env.reset(scenario_ids=torch.tensor(scenarios).repeat_interleave(envs_per_scenario))
    env = GraphedEnv(env)                    # each decision replays one CUDA graph
    for t in range(warmup + steps):
        if t == warmup:
            env.stats.zero_()
        obs, mask, _ = env.observe()
        if policy == "rule":
            lat, lon = env.rule_action(obs, mask)
        else:
            lat, lon, _ = policy.act(obs, mask, deterministic=True)
        env.step(lat, lon)
    per = torch.zeros(len(SCENARIO_NAMES), len(STAT_NAMES), device=device).index_add_(0, env.scenario, env.stats)
    out = {}
    for s in scenarios:
        out[SCENARIO_NAMES[s]] = metrics_from_stats(per[s].tolist(), envs_per_scenario, steps * cfg.decision_dt, cfg.dt)
    out["all"] = metrics_from_stats(per.sum(0).tolist(), envs_per_scenario * len(scenarios), steps * cfg.decision_dt, cfg.dt)
    return out


@torch.no_grad()
def run_ix(policy, envs_per_kind=64, warmup=200, steps=1200, seed=123, fixed_aggr=None, fixed_info=None,
           device="cuda", kinds=None):
    """Complete intersections (lcrl/ix_env.py), one row per intersection kind."""
    from .config import IxConfig
    from .ix_env import KIND_NAMES, IntersectionEnv
    kinds = list(range(len(KIND_NAMES))) if kinds is None else list(kinds)
    cfg = IxConfig(num_envs=envs_per_kind * len(kinds), episode_s=(warmup + steps + 10) * EnvConfig.decision_dt, pool_per_kind=32)
    env = IntersectionEnv(cfg, device=device, seed=seed, fixed_aggr=fixed_aggr, fixed_info=fixed_info)
    env.reset(kind_ids=torch.tensor(kinds).repeat_interleave(envs_per_kind))
    env = GraphedEnv(env)                    # each decision replays one CUDA graph
    for t in range(warmup + steps):
        if t == warmup:
            env.stats.zero_()
        obs, mask, _ = env.observe()
        if policy == "rule":
            lat, lon = env.rule_action(obs, mask)
        else:
            lat, lon, _ = policy.act(obs, mask, deterministic=True)
        env.step(lat, lon)
    kind_of_env = env.pool_kind[env.design_idx]
    per = torch.zeros(len(KIND_NAMES), len(STAT_NAMES), device=device).index_add_(0, kind_of_env, env.stats)
    out = {f"ix_{KIND_NAMES[k]}": metrics_from_stats(per[k].tolist(), envs_per_kind, steps * cfg.decision_dt, cfg.dt) for k in kinds}
    out["ix_all"] = metrics_from_stats(per.sum(0).tolist(), envs_per_kind * len(kinds), steps * cfg.decision_dt, cfg.dt)
    return out


SIG4_LEVELS = tuple(range(50, 801, 50))      # veh/h per leg


def sweep_signal4(policy, levels=SIG4_LEVELS, envs_per_level=32, warmup=600, steps=3000, seed=123,
                  device="cuda", n_lanes=2):
    """4-leg signal study design (lcrl.intersection.signal_4leg) at fixed demand levels.
    Every policy sees the same designs, signal timing (offset 0) and random streams."""
    from .config import IxConfig
    from .intersection import signal_4leg
    from .ix_env import IntersectionEnv
    designs = [signal_4leg(None, demand_vph=float(q), n_lanes=n_lanes) for q in levels]
    E = envs_per_level * len(levels)
    cfg = IxConfig(num_envs=E, episode_s=(warmup + steps + 10) * EnvConfig.decision_dt, max_veh=224)
    env = IntersectionEnv(cfg, device=device, seed=seed, designs=designs)
    env.reset(design_ids=torch.arange(len(levels)).repeat_interleave(envs_per_level))
    env.sig_offset.zero_()
    env = GraphedEnv(env)                    # each decision replays one CUDA graph
    for t in range(warmup + steps):
        if t == warmup:
            env.stats.zero_()
        obs, mask, _ = env.observe()
        if policy == "rule":
            lat, lon = env.rule_action(obs, mask)
        else:
            lat, lon, _ = policy.act(obs, mask, deterministic=True)
        env.step(lat, lon)
    per = env.stats.view(len(levels), envs_per_level, -1).sum(1)
    return {int(q): metrics_from_stats(per[i].tolist(), envs_per_level, steps * cfg.decision_dt, cfg.dt)
            for i, q in enumerate(levels)}


@torch.no_grad()
def lc_timing(policy, scenario, aggr_levels=(0.0, 0.5, 1.0), envs=128, warmup=100, steps=700, seed=7, device="cuda"):
    """Distance to the deadline when route-required lane changes start, per aggressiveness level:
    {aggr: (n, p10, p50, p90)} in metres. Every driver gets the same aggressiveness."""
    import numpy as np
    out = {}
    for ag in aggr_levels:
        cfg = EnvConfig(num_envs=envs, episode_s=(warmup + steps + 10) * EnvConfig.decision_dt)
        env = LaneChangeEnv(cfg, device=device, seed=seed, fixed_aggr=ag)
        env.reset(scenario_ids=torch.full((envs,), scenario))
        dists = []
        for t in range(warmup + steps):
            obs, mask, _ = env.observe()
            c = env._cache
            need, deadline, in_lc = c["need"].clone(), c["deadline"].clone(), c["in_lc"].clone()
            if policy == "rule":
                lat, lon = env.rule_action(obs, mask)
            else:
                lat, lon, _ = policy.act(obs, mask, deterministic=True)
            env.step(lat, lon)
            if t >= warmup:
                started = env.active & ~in_lc & (env.lane_from != env.lane_to)
                route = started & (need != 0) & ((env.lane_to - env.lane_from).sign() == need.sign())
                dists.append(deadline[route])
        d = torch.cat(dists).cpu().numpy()
        out[ag] = (len(d), *np.percentile(d, [10, 50, 90]).tolist()) if len(d) else (0, float("nan"), float("nan"), float("nan"))
    return out


@torch.no_grad()
def route_share(policy, scenario, aggr_levels=(0.0, 0.5, 1.0), bins=(0, 100, 200, 500, 1000, 2000), envs=128,
                warmup=100, steps=700, seed=7, device="cuda", tol=0):
    """Share of exiting vehicles (movement = right / exit) already in a route lane, by distance to
    the deadline: {aggr: [share per bin]}. ``tol``: lanes still to go that count as there. Every driver gets the same aggressiveness."""
    from .config import MOV_RIGHT
    out = {}
    edges = torch.tensor(bins, dtype=torch.float32, device=device)
    for ag in aggr_levels:
        cfg = EnvConfig(num_envs=envs, episode_s=(warmup + steps + 10) * EnvConfig.decision_dt)
        env = LaneChangeEnv(cfg, device=device, seed=seed, fixed_aggr=ag)
        env.reset(scenario_ids=torch.full((envs,), scenario))
        inside = torch.zeros(len(bins) - 1, device=device)
        total = torch.zeros(len(bins) - 1, device=device)
        for t in range(warmup + steps):
            obs, mask, _ = env.observe()
            if t >= warmup:
                c = env._cache
                sel = env.active & (env.mov == MOV_RIGHT)
                b = torch.bucketize(c["deadline"][sel], edges[1:-1])
                total += torch.bincount(b, minlength=len(bins) - 1).float()
                inside += torch.bincount(b[c["need"][sel].abs() <= tol], minlength=len(bins) - 1).float()
            if policy == "rule":
                lat, lon = env.rule_action(obs, mask)
            else:
                lat, lon, _ = policy.act(obs, mask, deterministic=True)
            env.step(lat, lon)
        out[ag] = (inside / total.clamp(min=1)).tolist()
    return out


def run_all(policy, envs=128, steps=1200, **kw):
    """Freeway scenarios and complete intersections."""
    out = run(policy, envs, steps=steps, **kw)
    out.update(run_ix(policy, max(envs // 2, 8), steps=steps, **kw))
    return out


def table(results, names):
    """Markdown table: one row per scenario, metric columns for each policy."""
    lines = []
    for key, label, _ in METRICS:
        lines.append(f"\n**{label}**\n")
        lines.append("| scenario | " + " | ".join(names) + " |")
        lines.append("|---|" + "---|" * len(names))
        for scen in list(results[names[0]].keys()):
            vals = [results[n][scen][key] for n in names]
            lines.append(f"| {scen} | " + " | ".join(f"{v:.3g}" for v in vals) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--policy", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--envs", type=int, default=128)
    p.add_argument("--steps", type=int, default=1200)
    p.add_argument("--sweeps", action="store_true", help="also sweep aggressiveness and info level")
    a = p.parse_args()
    from .train import load
    net = load(a.policy)
    res = {"rule": run_all("rule", a.envs, steps=a.steps), "rl": run_all(net, a.envs, steps=a.steps)}
    names = ["rule", "rl"]
    if a.sweeps:
        for ag in (0.1, 0.9):
            res[f"rl_aggr{ag}"] = run_all(net, a.envs, steps=a.steps, fixed_aggr=ag)
            names.append(f"rl_aggr{ag}")
        for info in (50.0, 200.0):
            res[f"rl_info{int(info)}"] = run_all(net, a.envs, steps=a.steps, fixed_info=info)
            names.append(f"rl_info{int(info)}")
    md = table(res, names)
    print(md)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)
        with open(a.out.replace(".json", ".md"), "w") as f:
            f.write(md)
