"""Record complete intersections (lcrl/ix_env.py) for the viewer (viz/viewer.html).

Vehicles are stored in world coordinates (metres, intersection centre at the origin):
frame rows are flat lists of [uid, X, Y, heading_deg, v, flags] per vehicle. Static data
holds lanes, turning paths, crosswalks and control; per-frame data holds every path's
signal state and the remaining crossing time on each crosswalk.
"""
import numpy as np
import torch

from . import intersection as ixd
from .config import LAT_LEFT, LAT_RIGHT, EnvConfig, IxConfig
from .evaluate import metrics_from_stats
from .graph import GraphedEnv
from .ix_env import KIND_NAMES, NLEG, IntersectionEnv

# flag bits shared with lcrl/visualize.py
F_LC, F_START, F_ABORT, F_SIG_L, F_SIG_R, F_BLOCKED, F_COMMITTED = 1, 2, 4, 8, 16, 32, 64
LON_SHIFT = 7
F_INBOX = 1 << 10
F_STAGED = 1 << 11            # permitted turner waiting inside the box for a gap


def _static(d, env, e, variant):
    lanes = []
    for a in range(NLEG):
        for k in range(int(d["n_in"][a])):
            lanes.append({"leg": a, "k": k, "start": round(float(d["lane_start"][a, k]), 1),
                          "end": round(float(d["lane_end"][a, k]), 1), "mov": int(d["lane_mov"][a, k])})
    paths = []
    for p in range(int(d["n_paths"])):
        pts = d["p_pts"][p][::3].round(2).tolist() + [d["p_pts"][p][-1].round(2).tolist()]
        paths.append({"leg": int(d["p_leg"][p]), "lane": int(d["p_lane"][p]), "mov": int(d["p_mov"][p]),
                      "exit": int(d["p_exit"][p]), "pts": pts})
    return {"type": "intersection", "scenario": d["kind"], "variant": variant, "ctrl": int(d["ctrl"]),
            "D": round(float(np.max(d["D"])), 2), "W": ixd.W, "legs": [bool(x) for x in d["legs"]],
            "draw": draw_primitives(d), "staging": int((d["n_stage"] > 0).sum()),
            "angle": [round(float(x), 2) for x in d["angle"]],
            "major": [bool(x) for x in d["major"]], "L_app": [round(float(x), 1) for x in d["L_app"]],
            "L_exit": [round(float(x), 1) for x in d["L_exit"]], "n_in": [int(x) for x in d["n_in"]],
            "n_out": [int(x) for x in d["n_out"]], "lanes": lanes, "paths": paths, "rtor": bool(d["rtor"]),
            "limit": round(env.limit[e].item(), 2), "ped_rate_ph": [round(3600 * float(x)) for x in d["ped_rate"]],
            "spawn_rate_vph": [round(3600 * float(d["spawn_rate"][a].sum()) * env.demand_mult[e].item())
                               for a in range(NLEG)],
            "demand_vph": round(float(d["demand_vph"]) * env.demand_mult[e].item()),
            "sweep": bool(d.get("sweep", False)), "phase_dur": [round(float(x), 1) for x in d["phase_dur"][:int(d["n_phase"])]],
            "turn_share": [round(float(x), 3) for x in _turn_share(d)]}


def _turn_share(d):
    """Left / through / right share of the entering demand."""
    q = (d["spawn_rate"][:, :, None] * d["mov_prob"]).sum((0, 1))
    return q / max(float(q.sum()), 1e-9)


def _r(a):
    return np.round(np.asarray(a, float), 2).tolist()


def _arc(D, L, near=120.0):
    """Arc lengths for drawing a link from D outward over L metres: 2 m steps near the box,
    10 m steps beyond."""
    s1 = np.arange(D, D + min(L, near), 2.0)
    s2 = np.arange(D + min(L, near), D + L, 10.0)
    return np.concatenate([s1, s2, [D + L]])


def draw_primitives(d):
    """Road geometry for the viewer, from the same centre lines the simulation uses: the box
    outline, lane polygons, lane markings, stop lines, crosswalks, signal heads and signs."""
    W, cls = ixd.W, d["centrelines"]
    out = dict(box=[], lanes=[], exits=[], dividers=[], centre=[], stop=[], cross=[], heads=[], signs=[], ped=[])
    present = [a for a in range(len(cls)) if d["legs"][a]]
    order = sorted(present, key=lambda a: float(d["angle"][a]) % 360.0)
    for a in order:
        D = float(d["D"][a])
        out["box"].append(_r(cls[a].lane_point(D, d["n_out"][a] * W, inbound=False)))
        out["box"].append(_r(cls[a].lane_point(D, d["n_in"][a] * W, inbound=True)))
    for a in present:
        cl, D = cls[a], float(d["D"][a])
        n_in, n_out = int(d["n_in"][a]), int(d["n_out"][a])
        L_app, L_exit = float(d["L_app"][a]), float(d["L_exit"][a])
        for k in range(n_in):                   # approach lanes (from the stop line outward)
            ls, le = float(d["lane_start"][a, k]), float(d["lane_end"][a, k])
            ss = _arc(D + L_app - le, le - max(ls, 0.0))
            edge0, edge1 = cl.lane_point(ss, k * W, True), cl.lane_point(ss, (k + 1) * W, True)
            out["lanes"].append(dict(leg=a, k=k, pts=_r(np.vstack([edge0, edge1[::-1]]))))
        if n_out:
            ss = _arc(D, L_exit)
            edge0, edge1 = cl.lane_point(ss, 0.0, False), cl.lane_point(ss, n_out * W, False)
            out["exits"].append(_r(np.vstack([edge0, edge1[::-1]])))
            for j in range(1, n_out):
                out["dividers"].append(_r(cl.lane_point(ss, j * W, False)))
        for k in range(1, n_in):                # dashed lines where adjacent approach lanes overlap
            lo = max(float(d["lane_start"][a, k - 1]), float(d["lane_start"][a, k]), 0.0)
            hi = min(float(d["lane_end"][a, k - 1]), float(d["lane_end"][a, k]))
            if hi > lo:
                out["dividers"].append(_r(cl.lane_point(_arc(D + L_app - hi, hi - lo), k * W, True)))
        out["centre"].append(_r(cl.at(_arc(D, max(L_app, L_exit)))[0]))
        if n_in:
            out["stop"].append(_r([cl.lane_point(D, 0.0, True), cl.lane_point(D, n_in * W, True)]))
        # crosswalk: stripes across the whole road between s = D - 4.5 and D - 0.5
        o0, o1 = -n_out * W, n_in * W
        stripes = []
        for o in np.arange(o0 + 0.5, o1, 1.2):
            q = [cl.lane_point(D - 4.5, o, True), cl.lane_point(D - 0.5, o, True),
                 cl.lane_point(D - 0.5, o + 0.6, True), cl.lane_point(D - 4.5, o + 0.6, True)]
            stripes.append(_r(q))
        out["cross"].append(dict(leg=a, stripes=stripes))
        out["ped"].append(dict(leg=a, a=_r(cl.lane_point(D - 2.5, o1, True)), b=_r(cl.lane_point(D - 2.5, o0, True))))
        for k in range(n_in):
            out["heads"].append(dict(leg=a, k=k, mov=int(d["lane_mov"][a, k]),
                                     xy=_r(cl.lane_point(D + 1.8, (k + 0.5) * W, True))))
        ctrl, minor = int(d["ctrl"]), not bool(d["major"][a])
        sign = "stop" if ctrl == ixd.IX_AWSC or (ctrl == ixd.IX_TWSC and minor) else "yield" if ctrl == ixd.IX_YIELD and minor else None
        if sign and n_in:
            out["signs"].append(dict(kind=sign, xy=_r(cl.lane_point(D + 3.0, n_in * W + 1.5, True))))
    return out


def world_xy(d, link, lane_y, x, path):
    """World position and heading (degrees) of vehicles on links or in the box."""
    n = len(link)
    X, Y, H = np.zeros(n), np.zeros(n), np.zeros(n)
    W, cls = ixd.W, d["centrelines"]
    for i in range(n):
        li = int(link[i])
        if li >= NLEG:                        # exit link
            a = li - NLEG
            sa = float(d["D"][a]) + x[i]
            pos = cls[a].lane_point(sa, (lane_y[i] + 0.5) * W, inbound=False)
            h = cls[a].at(sa)[1]
        elif li >= 0:                         # approach link
            sa = float(d["D"][li]) + float(d["L_app"][li]) - x[i]
            pos = cls[li].lane_point(sa, (lane_y[i] + 0.5) * W, inbound=True)
            h = -cls[li].at(sa)[1]
        else:                                 # box: along the path polyline
            p = int(path[i])
            s_tab, pts = d["p_s"][p], d["p_pts"][p]
            s = min(max(x[i], 0.0), float(s_tab[-1]))
            j = int(np.clip(np.searchsorted(s_tab, s) - 1, 0, len(s_tab) - 2))
            t = (s - s_tab[j]) / max(s_tab[j + 1] - s_tab[j], 1e-6)
            pos = pts[j] + t * (pts[j + 1] - pts[j])
            h = pts[j + 1] - pts[j]
        X[i], Y[i], H[i] = pos[0], pos[1], np.degrees(np.arctan2(h[1], h[0]))
    return X, Y, H


@torch.no_grad()
def record_ix(policy, kind_ids=None, seed=7, warmup=240, steps=400, device="cuda", designs=None):
    """Record random designs of the given kinds, or the explicit ``designs`` (one env each,
    signal offset 0)."""
    E = len(kind_ids) if designs is None else len(designs)
    cfg = IxConfig(num_envs=E, episode_s=(warmup + steps + 10) * EnvConfig.decision_dt, pool_per_kind=4,
                   max_veh=96 if designs is None else 224)       # 600 m legs of the study design
    env = IntersectionEnv(cfg, device=device, seed=seed, designs=designs)
    if designs is None:
        env.reset(kind_ids=kind_ids)
    else:
        env.reset(design_ids=list(range(E)))
        env.sig_offset.zero_()
        kind_ids = [KIND_NAMES.index(d["kind"]) for d in designs]
    designs = [env.pool_designs[env.design_idx[e].item()] for e in range(E)]
    env = GraphedEnv(env)                    # each decision replays one CUDA graph
    variant, out = {}, []
    for e, k in enumerate(kind_ids):
        if not designs[e].get("sweep"):
            variant[k] = variant.get(k, 0) + 1
        variant.setdefault(k, 1)
        out.append({"static": _static(designs[e], env, e, variant[k]), "frames": [], "sig": [], "peds": [],
                    "vehicles": {}})

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
        now = env.t_env.float() * cfg.dt
        snap = {k: getattr(env, k).cpu().numpy() for k in
                ("active", "x", "v", "prog", "age", "aggr", "info", "vdes", "mov", "link", "path", "granted", "staged")}
        snap["from"], snap["to"] = env.lane_from.cpu().numpy(), env.lane_to.cpu().numpy()
        need, blocked = C["need"].cpu().numpy(), C["blocked"].cpu().numpy()
        lon_np = lon.cpu().numpy()
        sig = C["ctl"]["state"].cpu().numpy()
        ped_left = (env.ped_busy_until - now[:, None]).clamp(min=0).cpu().numpy()
        env.step(lat, lon)
        post_from, post_to = env.lane_from.cpu().numpy(), env.lane_to.cpu().numpy()
        post_link = env.link.cpu().numpy()
        oc = {k: m.cpu().numpy() for k, m in env.last_outcome.items()}
        for e in range(E):
            d, R = designs[e], out[e]
            act_e = snap["active"][e]
            new = act_e & (~prev_active[e] | (snap["age"][e] < prev_age[e]))
            for n in np.nonzero(new)[0]:
                uid[e, n] = next_uid[e]
                next_uid[e] += 1
                R["vehicles"][int(uid[e, n])] = {
                    "aggr": round(float(snap["aggr"][e, n]), 2), "info": round(float(snap["info"][e, n])),
                    "vdes": round(float(snap["vdes"][e, n]), 1), "mov": int(snap["mov"][e, n]),
                    "leg": int(snap["link"][e, n]), "spawn": f, "end": None, "outcome": None}
            idx = np.nonzero(act_e)[0]
            fr, to, pg = snap["from"][e, idx], snap["to"][e, idx], snap["prog"][e, idx]
            lane_y = fr + (to - fr) * pg
            X, Y, H = world_xy(d, snap["link"][e, idx], lane_y, snap["x"][e, idx], snap["path"][e, idx])
            row = []
            for j, n in enumerate(idx):
                on_link = snap["link"][e, n] >= 0
                flags = int(lon_np[e, n]) << LON_SHIFT if on_link else 0
                same_link = post_link[e, n] == snap["link"][e, n]
                if on_link and fr[j] != to[j]:
                    flags |= F_LC
                started = on_link and same_link and fr[j] == to[j] and post_to[e, n] != post_from[e, n]
                aborted = on_link and same_link and fr[j] != to[j] and post_from[e, n] == to[j] and post_to[e, n] == fr[j]
                flags |= F_START if started else 0
                flags |= F_ABORT if aborted else 0
                ind = (to[j] - fr[j]) if fr[j] != to[j] else int(np.sign(need[e, n])) if on_link else 0
                if started:
                    ind = int(post_to[e, n] - post_from[e, n])
                flags |= F_SIG_L if ind < 0 else F_SIG_R if ind > 0 else 0
                flags |= F_BLOCKED if blocked[e, n] else 0
                flags |= F_COMMITTED if snap["granted"][e, n] else 0
                flags |= F_INBOX if not on_link else 0
                flags |= F_STAGED if snap["staged"][e, n] else 0
                row += [int(uid[e, n]), round(float(X[j]), 1), round(float(Y[j]), 1), int(round(H[j])),
                        round(float(snap["v"][e, n]), 1), flags]
                veh = R["vehicles"][int(uid[e, n])]
                if oc["crossed"][e, n]:
                    veh["outcome"] = 1 if oc["success"][e, n] else 2
                    veh["cross"] = f
                if oc["finished"][e, n] or oc["collided"][e, n] or oc["timeout"][e, n]:
                    veh["end"] = f
                    if oc["collided"][e, n]:
                        veh["outcome"] = 3
                    elif oc["timeout"][e, n]:
                        veh["outcome"] = 4
            R["frames"].append(row)
            R["sig"].append("".join(str(int(c)) for c in sig[e, :int(d["n_paths"])]))
            R["peds"].append([round(float(t), 1) for t in ped_left[e]])
        prev_active, prev_age = snap["active"], snap["age"]
    for e in range(E):
        out[e]["summary"] = metrics_from_stats(env.stats[e].tolist(), 1, steps * cfg.decision_dt, cfg.dt)
        out[e]["static"]["ped_cross_time"] = [round(float(x), 1) for x in designs[e]["ped_cross_time"]]
    return out


def signal4_sweep_designs(levels, n_lanes=2):
    """The 4-leg signal study design at each demand level (veh/h per leg), for the viewer's
    demand slider."""
    out = []
    for q in levels:
        d = ixd.signal_4leg(None, demand_vph=float(q), n_lanes=n_lanes)
        d["sweep"] = True
        out.append(d)
    return out


def ix_kind_ids(names):
    if names == "all":
        return list(range(len(KIND_NAMES)))
    if names in ("none", ""):
        return []
    return [int(s) if s.isdigit() else KIND_NAMES.index(s) for s in names.split(",")]
