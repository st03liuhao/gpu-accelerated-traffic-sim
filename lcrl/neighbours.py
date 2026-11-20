"""Neighbour search on sorted lane lists: a hand-written CUDA kernel (lcrl/csrc/neighbours.cu).

The dense neighbour search in ``_tables_dense`` compares every vehicle with every other
vehicle in every lane: [E, N, N, K] work and memory. Here the vehicles are first sorted into
one list per lane (per (link, lane) at an intersection), ordered by position. A vehicle
changing lanes is in both of its lanes' lists. Then one GPU thread per (env, vehicle, lane):
- binary-searches the vehicle's place in that lane's list;
- takes the next and previous entries as leader and follower;
- scans only the vehicles within its information range (queue count and speeds, best gap)
  and within 60 m behind (queue cost).
Work per thread is O(log n + vehicles in range) instead of O(N).

Results equal the dense version, including tie-breaking (smallest slot index first). Only
float sums (speed sum, queue cost) can differ in the last bits, from the order of additions.

The extension compiles on first use (about a minute) and is cached by PyTorch. It needs
nvcc and, on Windows, MSVC: if ``cl`` is not on PATH, the environment of ``vcvars64.bat``
(``VCVARS``, default Visual Studio 2022 Community) is loaded. ``LCRL_NB_KERNEL=0`` or a
failed build falls back to the dense PyTorch version.
"""
import os
import shutil
import subprocess
import warnings

import torch

_MOD = None
_STATE = None          # None: not tried yet, True: loaded, False: unavailable
VCVARS_DEFAULT = r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"


def _msvc_env():
    if os.name != "nt" or shutil.which("cl"):
        return
    vc = os.environ.get("VCVARS", VCVARS_DEFAULT)
    if not os.path.exists(vc):
        raise RuntimeError(f"cl.exe not on PATH and {vc} not found (set VCVARS)")
    out = subprocess.run(f'"{vc}" >nul && set', shell=True, capture_output=True, text=True, check=True).stdout
    for line in out.splitlines():
        k, sep, v = line.partition("=")
        if sep and k:
            os.environ[k] = v
    os.environ.setdefault("DISTUTILS_USE_SDK", "1")


def _load():
    global _MOD, _STATE
    if _STATE is not None:
        return _MOD
    _STATE = False
    if os.environ.get("LCRL_NB_KERNEL", "1") == "0" or not torch.cuda.is_available():
        return None
    try:
        from torch.utils.cpp_extension import load
        _msvc_env()
        d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc")
        _MOD = load(name="lcrl_neighbours", sources=[os.path.join(d, "neighbours.cpp"), os.path.join(d, "neighbours.cu")],
                    extra_cuda_cflags=["-O3"], verbose=False)
        _STATE = True
    except Exception as exc:                  # no compiler, build error: use the dense version
        warnings.warn(f"lcrl.neighbours: CUDA kernel unavailable ({exc}); using the dense neighbour search",
                      stacklevel=2)
        _MOD = None
    return _MOD


def enabled():
    return _load() is not None


# ---------------------------------------------------------------------- sorted lists
def sorted_lists(x, lid, L):
    """Sort lane memberships into lists.

    x [E, N] positions; lid [E, N, 2] list id of each vehicle's two lane memberships
    (L = not a member). Returns entries sorted by (list, x, slot) and each list's range:
    ent_slot [E, 2N] int32, ent_x [E, 2N], seg_start / seg_end [E, L] int32.
    """
    E, N, _ = lid.shape
    ids = lid.reshape(E, 2 * N)
    xs = x[:, :, None].expand(E, N, 2).reshape(E, 2 * N)
    # entries start in slot order; two stable sorts give (list, x, slot) order
    o1 = torch.sort(xs, dim=1, stable=True).indices
    o2 = torch.sort(torch.gather(ids, 1, o1), dim=1, stable=True).indices
    order = torch.gather(o1, 1, o2)
    sid = torch.gather(ids, 1, order).contiguous()
    ent_x = torch.gather(xs, 1, order).contiguous()
    ent_slot = torch.div(order, 2, rounding_mode="floor").int().contiguous()
    q = torch.arange(L, device=x.device, dtype=sid.dtype).expand(E, L).contiguous()
    seg_start = torch.searchsorted(sid, q, right=False).int().contiguous()
    seg_end = torch.searchsorted(sid, q, right=True).int().contiguous()
    return ent_slot, ent_x, seg_start, seg_end


def _memberships(active, lane_from, lane_to, base, L):
    c0 = torch.where(active, base + lane_from, L)
    c1 = torch.where(active & (lane_to != lane_from), base + lane_to, L)
    return torch.stack([c0, c1], -1)


def _queue_loss(env):
    return (0.5 + env.aggr) * (1.0 - env.v / env.vdes.clamp(min=1.0)).clamp(min=0.0)


_EMPTY = {}


def _empty(dev, dtype):
    key = (dev, dtype)
    if key not in _EMPTY:
        _EMPTY[key] = torch.empty(0, device=dev, dtype=dtype)
    return _EMPTY[key]


# ---------------------------------------------------------------------- the two envs
def freeway_tables(env, INF):
    """Same dict as LaneChangeEnv._tables_dense."""
    mod, cfg, K = _load(), env.cfg, env.K
    f32 = lambda t: t.float().contiguous()
    lid = _memberships(env.active, env.lane_from, env.lane_to, 0, K)
    ent = sorted_lists(env.x, lid, K)
    qbase = torch.zeros_like(env.lane_from, dtype=torch.int32)
    ei, ef = _empty(env.dev, torch.int32), _empty(env.dev, torch.float32)
    lead_dx, lead_idx, _, lag_dx, lag_idx, count, speed_sum, queue_cost = mod.pass1(
        *ent, f32(env.x), f32(env.info), f32(env.v), f32(_queue_loss(env)), qbase, ei, ef, ei, ei, K, INF)
    T = {"occ": env._occupancy(), "lead_idx": lead_idx, "lag_idx": lag_idx,
         "count": count, "speed_sum": speed_sum, "queue_cost": queue_cost}
    T["lead_gap"], T["lag_gap"] = env._bumper_gaps(lead_dx, lead_idx, lag_dx)
    gap_j = torch.minimum(T["lead_gap"], env.info[..., None]).contiguous()
    T["best_gap"], T["best_pos"] = mod.pass2(*ent, f32(env.x), f32(env.info), qbase, gap_j)
    return T


def ix_tables(env, INF, NLEG):
    """Link part of IntersectionEnv._tables_dense (everything except the box leaders)."""
    mod, cfg, K = _load(), env.cfg, env.K
    f32 = lambda t: t.float().contiguous()
    on_link = env.active & (env.link >= 0)
    in_box = env.active & (env.link < 0)
    L = 2 * NLEG * K                                      # one list per (link, lane)
    base = env.link.clamp(min=0) * K
    lid = _memberships(on_link, env.lane_from, env.lane_to, base, L)
    ent = sorted_lists(env.x, lid, L)
    qbase = torch.where(on_link, base, -1).int().contiguous()
    # approach vehicles follow box vehicles that started from their lane
    path_c = env.path.clamp(min=0)
    pleg = env._g1(env.D("p_leg"), path_c)
    plane = env._g1(env.D("p_lane"), path_c)
    chain_a = f32(env._per_link(env.D("link_len")) - env.x)
    chain_leg = torch.where(on_link & (env.link < NLEG), env.link, -1).int().contiguous()
    box_leg = torch.where(in_box, pleg, -2).int().contiguous()
    lead_dx, lead_idx, link_lead_dx, lag_dx, lag_idx, count, speed_sum, queue_cost = mod.pass1(
        *ent, f32(env.x), f32(env.info), f32(env.v), f32(_queue_loss(env)), qbase,
        chain_leg, chain_a, box_leg, plane.int().contiguous(), K, INF)
    k = env.lanes.view(1, 1, -1)
    occ = on_link[..., None] & ((env.lane_from[..., None] == k) | (env.lane_to[..., None] == k))
    T = {"occ": occ, "lead_idx": lead_idx, "lag_idx": lag_idx, "link_lead_dx": link_lead_dx,
         "count": count, "speed_sum": speed_sum, "queue_cost": queue_cost}
    T["lead_gap"], T["lag_gap"] = env._bumper_gaps(lead_dx, lead_idx, lag_dx)
    gap_j = torch.minimum(T["lead_gap"], env.info[..., None]).contiguous()
    T["best_gap"], T["best_pos"] = mod.pass2(*ent, f32(env.x), f32(env.info), qbase, gap_j)
    return T
