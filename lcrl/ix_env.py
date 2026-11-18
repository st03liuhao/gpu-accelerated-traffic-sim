"""Batched GPU environment of complete intersections: every approach, the box, every exit.

Implements docs/INTERSECTIONS.md. Each environment is one intersection (4-leg or T) with
8 links: approach link of leg a = link a, exit link of leg b = link 4 + b. Vehicles on links
are RL agents (same observation layout and actions as lcrl/env.py). Vehicles in the box
follow the rule-based right-of-way model and are not agents.

Vehicle position: on a link, ``x`` is metres from the link start; in the box (link = -1),
``x`` is metres along the turning path ``path``.
"""
import math
import numpy as np
import torch

from . import intersection as ixd
from .config import (CAT_AWSC, CAT_FREE, CAT_SIGNAL, CAT_TWSC_MAJOR, CAT_TWSC_MINOR, CAT_YIELD_MINOR,
                     EGO_FEATURES, LANE_FEATURES, LAT_KEEP, LAT_LEFT, LAT_RIGHT, LON_DELTAS, MOV_LEFT,
                     MOV_RIGHT, MOV_THROUGH, NUM_CAT, NUM_LAT, NUM_MOV, REL_LANES, REWARD_DT, IxConfig,
                     RewardConfig)
from .env import INF, STAT_NAMES, S, ahead_of, decision_step, scatter_slots

KIND_NAMES = list(ixd.KINDS)
NLEG, K, P, NPH = ixd.NLEG, ixd.KMAX, ixd.PMAX, ixd.NPH
LK = 2 * NLEG
TURN_SPEED = (9.0, 99.0, 6.0)        # speed cap in the box for left / through / right (m/s)
DECIDE_RANGE = 80.0                  # same-rank approaching vehicles considered for entry order (m)
GAP_RANGE = 250.0                    # higher-rank vehicles considered for gap acceptance (m)
# gap acceptance against higher-rank traffic (permitted turns, right turn on red, minor streets):
# the conflicting vehicle must arrive no sooner than the critical gap of the movement
# (+ up to 1 s for cautious drivers) and no sooner than PET_MIN + (1 - aggr) after we clear.
CRIT_GAP = (4.0, 5.5, 5.0)           # left, through, right (s)
PET_MIN = 1.0
CAND = NLEG * K + 16                 # entry decisions per intersection: lane heads + staged vehicles


class IntersectionEnv:
    def __init__(self, cfg: IxConfig = None, rcfg: RewardConfig = None, device="cuda", seed=0,
                 fixed_aggr=None, fixed_info=None, designs=None):
        """``designs``: explicit list of designs (lcrl.intersection) instead of a random pool;
        then ``reset(design_ids=...)`` pins env e to design ``design_ids[e]``."""
        self.cfg = cfg or IxConfig()
        self.rcfg = rcfg or RewardConfig()
        cfg = self.cfg
        self.dev = torch.device(device)
        self.E, self.N, self.K = cfg.num_envs, cfg.max_veh, K
        self.rng = np.random.default_rng(seed)
        self.gen = torch.Generator(device=self.dev).manual_seed(seed)
        self.fixed_aggr, self.fixed_info = fixed_aggr, fixed_info
        self._build_pool(designs)
        E, N, d = self.E, self.N, self.dev
        f = lambda *s: torch.zeros(*s, device=d)
        i = lambda *s: torch.zeros(*s, device=d, dtype=torch.long)
        b = lambda *s: torch.zeros(*s, device=d, dtype=torch.bool)
        self.design_idx, self.t_env = i(E), i(E)
        self.limit, self.sig_offset, self.demand_mult = f(E), f(E), f(E)
        self.pending = i(E, NLEG, K)
        self.ped_wait, self.ped_busy_until = f(E, NLEG), f(E, NLEG)
        # vehicles
        self.active, self.granted, self.missed, self.stopped_once = b(E, N), b(E, N), b(E, N), b(E, N)
        self.staged = b(E, N)         # permitted turner waiting inside the box for a gap
        self.link, self.path = i(E, N), i(E, N) - 1
        self.lane_from, self.lane_to, self.prog = i(E, N), i(E, N), f(E, N)
        self.x, self.v, self.a_prev = f(E, N), f(E, N), f(E, N)
        self.mov, self.age, self.stop_time = i(E, N), i(E, N), f(E, N)
        self.aggr, self.info, self.vdes, self.t_ff = f(E, N), f(E, N), f(E, N), f(E, N)
        self.acc_lc, self.acc_abort, self.acc_hb, self.acc_unsafe, self.acc_jerk = (f(E, N) for _ in range(5))
        self.stats = f(E, len(STAT_NAMES))
        self.slot = torch.arange(N, device=d)
        self.lanes = torch.arange(K, device=d)
        self.pinned = self.pinned_design = None
        if designs is not None:
            self.reset(design_ids=torch.arange(E, device=d) % len(designs))
        else:
            self.reset()

    # ------------------------------------------------------------------ designs
    def _build_pool(self, designs=None):
        if designs is not None:
            kinds = [KIND_NAMES.index(d["kind"]) for d in designs]
        else:
            designs, kinds = [], []
            for k in self.cfg.kinds:
                for _ in range(self.cfg.pool_per_kind):
                    peds = self.rng.random() < self.cfg.ped_share
                    designs.append(ixd.design(self.rng, KIND_NAMES[k], self.cfg.demand_scale, peds=peds))
                    kinds.append(k)
        st = ixd.stack(designs)
        self.pool_designs = designs                       # kept for visualisation
        self.pool_kind = torch.as_tensor(kinds, device=self.dev)
        self.pool = {}
        for key, arr in st.items():
            if arr.dtype == object or arr.dtype.kind in "US":
                continue
            t = torch.as_tensor(arr, device=self.dev)
            self.pool[key] = t.float() if t.dtype == torch.float64 else t
        p = self.pool
        # link-level lane geometry: links 0-3 approaches, 4-7 exits
        nd = len(designs)
        j = torch.arange(K, device=self.dev)
        exit_exists = j[None, None, :] < p["n_out"][:, :, None]
        p["link_len"] = torch.cat([p["L_app"], p["L_exit"]], 1)
        p["link_vcap"] = torch.cat([p["vcap_app"], p["vcap_exit"]], 1)            # [nd, LK, VCAP_N] curve caps
        p["link_ls"] = torch.cat([p["lane_start"], torch.where(exit_exists, 0.0, 1e9)], 1)
        p["link_le"] = torch.cat([p["lane_end"], torch.where(exit_exists, p["L_exit"][:, :, None], -1e9)], 1)
        kind_pos = torch.full((len(KIND_NAMES),), -1, dtype=torch.long)
        for i, k in enumerate(self.cfg.kinds):
            kind_pos[k] = i
        self.kind_pos = kind_pos.to(self.dev)
        self.kinds_t = torch.as_tensor(self.cfg.kinds, device=self.dev)
        # constant lookup tables live on the device (no per-step host-to-device copies)
        self.k_crit_gap = torch.as_tensor(CRIT_GAP, dtype=torch.float32, device=self.dev)
        self.k_turn_speed = torch.as_tensor(TURN_SPEED, dtype=torch.float32, device=self.dev)
        self.k_lon_deltas = torch.as_tensor(LON_DELTAS, dtype=torch.float32, device=self.dev)
        p["cycle"] = p["phase_dur"].sum(1).clamp(min=1.0)
        p["phase_cum"] = p["phase_dur"].cumsum(1)
        assert p["link_len"].shape == (nd, LK)

    def reset(self, kind_ids=None, design_ids=None):
        """Regenerate every environment. ``kind_ids`` pins env e to a design kind;
        ``design_ids`` pins it to one design of the pool."""
        self.pinned = None if kind_ids is None else torch.as_tensor(kind_ids, device=self.dev)
        self.pinned_design = None if design_ids is None else torch.as_tensor(design_ids, device=self.dev)
        self._reset_envs(torch.ones(self.E, dtype=torch.bool, device=self.dev))
        if kind_ids is None and design_ids is None:
            self.t_env.copy_(torch.randint(0, self.cfg.episode_steps, (self.E,), device=self.dev, generator=self.gen))
        self.stats.zero_()

    def _reset_envs(self, mask):
        """Regenerate the envs in ``mask`` [E] (masked writes on the GPU, no host sync)."""
        E, d = self.E, self.dev
        if self.pinned_design is not None:
            didx = self.pinned_design
        else:
            if self.pinned is not None:
                want = self.pinned
            else:
                want = self.kinds_t[torch.randint(0, len(self.kinds_t), (E,), device=d, generator=self.gen)]
            per = self.cfg.pool_per_kind
            didx = self.kind_pos[want] * per + torch.randint(0, per, (E,), device=d, generator=self.gen)
        self.design_idx = torch.where(mask, didx, self.design_idx)
        r = lambda: torch.rand(E, device=d, generator=self.gen)
        self.limit = torch.where(mask, 11.0 + 6.0 * r(), self.limit)
        self.sig_offset = torch.where(mask, r() * self.pool["cycle"][self.design_idx], self.sig_offset)
        jitter = self.pool["demand_jitter"][self.design_idx]
        self.demand_mult = torch.where(mask, 1.0 + jitter * (0.6 * r() - 0.3), self.demand_mult)
        self.pending = torch.where(mask[:, None, None], 0, self.pending)
        self.ped_wait = torch.where(mask[:, None], 0.0, self.ped_wait)
        self.ped_busy_until = torch.where(mask[:, None], -1.0, self.ped_busy_until)
        self.active = self.active & ~mask[:, None]
        self.t_env = torch.where(mask, 0, self.t_env)

    def D(self, key):
        """Design tensor for every env (leading dim E)."""
        return self.pool[key][self.design_idx]

    # ------------------------------------------------------------------ helpers
    def _g1(self, t, idx):
        """t [E, X], idx [E, N] -> [E, N]"""
        return torch.gather(t, 1, idx)

    def _gk(self, t, k):
        """t [E, N, K], k [E, N] -> [E, N]"""
        return torch.gather(t, 2, k.clamp(0, self.K - 1)[..., None]).squeeze(-1)

    def _per_link(self, t):
        """t [E, LK, ...] gathered at each vehicle's link -> [E, N, ...]"""
        li = self.link.clamp(min=0)
        idx = li.view(self.E, self.N, *([1] * (t.dim() - 2))).expand(-1, -1, *t.shape[2:])
        return torch.gather(t, 1, idx)

    def _pair(self, t, pi, pj):
        """t [E, P, P] -> [E, A, B] at (pi[a], pj[b]) for pi [E, A], pj [E, B]"""
        flat = t.reshape(self.E, P * P)
        idx = (pi.clamp(min=0)[:, :, None] * P + pj.clamp(min=0)[:, None, :]).reshape(self.E, -1)
        return torch.gather(flat, 1, idx).view(self.E, pi.shape[1], pj.shape[1])

    def _t_reach(self, v, d):
        """Time to cover d metres from speed v accelerating at a_max (no speed cap)."""
        a = self.cfg.a_max
        return (-v + torch.sqrt(v * v + 2 * a * d.clamp(min=0))) / a

    def _t_reach_capped(self, v, d, vmax):
        """Time to cover d metres from speed v accelerating at a_max up to vmax, then cruising."""
        a = self.cfg.a_max
        d = d.clamp(min=0)
        vmax = torch.maximum(vmax, v + 1e-3)
        d1 = (vmax * vmax - v * v) / (2 * a)
        free = (-v + torch.sqrt(v * v + 2 * a * d)) / a
        return torch.where(d <= d1, free, (vmax - v) / a + (d - d1) / vmax)

    def _idm(self, v, vdes, T_h, gap, dv):
        c = self.cfg
        s_star = c.s0 + (v * T_h + v * dv / (2 * math.sqrt(c.a_max * c.b_comf))).clamp(min=0)
        return c.a_max * (1 - (v / vdes.clamp(min=1.0)) ** 4 - (s_star / gap.clamp(min=0.1)) ** 2)

    def _krauss(self, v, gap, v_l, b=4.5):
        tau = self.cfg.dt
        return v_l + (gap - 0.5 - v_l * tau) / ((v + v_l) / (2 * b) + tau)

    def _headway(self):
        return self.cfg.T * (1.3 - 0.6 * self.aggr)

    def _center(self):
        return torch.where(self.prog >= 0.5, self.lane_to, self.lane_from)

    # ------------------------------------------------------------------ control state
    def _control(self):
        """Signal phase, per-path signal state and time to change, pedestrian walk windows."""
        dt = self.cfg.dt
        cyc = self.D("cycle")
        tc = torch.remainder(self.t_env * dt + self.sig_offset, cyc)
        cum = self.D("phase_cum")
        nph = self.D("n_phase").clamp(min=1)
        cur = (tc[:, None] >= cum).sum(1).clamp(max=NPH - 1)
        cur = torch.minimum(cur, nph - 1)
        start = torch.gather(cum, 1, cur[:, None]).squeeze(1) - torch.gather(self.D("phase_dur"), 1, cur[:, None]).squeeze(1)
        elapsed = tc - start
        state = torch.gather(self.D("phase_state"), 1, cur[:, None, None].expand(-1, 1, P)).squeeze(1)
        same_left = torch.gather(self.D("phase_same_left"), 1, cur[:, None, None].expand(-1, 1, P)).squeeze(1)
        walk = torch.gather(self.D("phase_walk"), 1, cur[:, None, None].expand(-1, 1, NLEG)).squeeze(1) & (elapsed[:, None] < 7.0)
        signal = self.D("ctrl") == ixd.IX_SIGNAL
        walk = torch.where(signal[:, None], walk, torch.ones_like(walk))
        return dict(state=state, tts=(same_left - elapsed[:, None]).clamp(min=0), walk=walk)

    def _plan(self, center):
        """Path each approach vehicle would take from its centre lane: the intended movement if
        the lane allows it, else through, right, left (that vehicle then misses its movement)."""
        on_app = self.active & (self.link >= 0) & (self.link < NLEG)
        path_of = self.D("path_of")                                              # [E, 4, K, 3]
        leg = self.link.clamp(0, NLEG - 1)
        flat = path_of.reshape(self.E, -1)
        base = (leg * K + center.clamp(0, K - 1)) * 3
        cand = [self.mov, torch.full_like(self.mov, MOV_THROUGH), torch.full_like(self.mov, MOV_RIGHT),
                torch.full_like(self.mov, MOV_LEFT)]
        plan = torch.full_like(self.mov, -1)
        for m in cand:
            p = torch.gather(flat, 1, base + m)
            plan = torch.where((plan < 0) & (p >= 0), p, plan)
        plan = torch.where(on_app, plan, -1)
        return torch.where(self.granted | (self.link < 0), self.path, plan)

    def _rank(self, path, ctl):
        """Right-of-way rank of each vehicle's (planned) path, section 2 of the spec."""
        ctrl = self.D("ctrl")[:, None]
        pm = self._g1(self.D("p_mov"), path.clamp(min=0))
        pleg = self._g1(self.D("p_leg"), path.clamp(min=0))
        major = self._g1(self.D("major").long(), pleg).bool()
        st = self._g1(ctl["state"], path.clamp(min=0))
        stopped = self.stopped_once
        minor_rank = torch.ones_like(self.v)      # minor movements take turns (FCFS by stop time)
        major_rank = torch.where(pm == MOV_LEFT, 1.5, 2.0)
        rtor = self.D("rtor")[:, None] & (pm == MOV_RIGHT) & stopped & (st == ixd.SIG_R)
        r_sig = torch.where(st == ixd.SIG_G, 2.0, torch.where(st == ixd.SIG_PERM, 1.0, torch.where(rtor, 0.5, 0.0)))
        r_awsc = torch.where(stopped, 1.0, 0.0)
        r_twsc = torch.where(major, major_rank, torch.where(stopped, minor_rank, 0.0))
        r_yield = torch.where(major, major_rank, minor_rank)
        r = torch.where(ctrl == ixd.IX_SIGNAL, r_sig, torch.where(ctrl == ixd.IX_AWSC, r_awsc,
            torch.where(ctrl == ixd.IX_TWSC, r_twsc, r_yield)))
        return torch.where(path >= 0, r, 0.0), st

    # ------------------------------------------------------------------ neighbour tables
    def _tables(self):
        cfg = self.cfg
        on_link = self.active & (self.link >= 0)
        in_box = self.active & (self.link < 0)
        k = self.lanes.view(1, 1, -1)
        occ = on_link[..., None] & ((self.lane_from[..., None] == k) | (self.lane_to[..., None] == k))
        same = (self.link[:, :, None] == self.link[:, None, :]) & on_link[:, :, None] & on_link[:, None, :]
        dx = self.x[:, None, :] - self.x[:, :, None]
        order = ahead_of(self.x, self.slot)
        ahead = order & same
        behind = order.transpose(1, 2) & same
        occ_j = occ[:, None, :, :]
        lead_dx, lead_idx = torch.where(ahead[..., None] & occ_j, dx[..., None], INF).min(2)
        lag_dx, lag_idx = torch.where(behind[..., None] & occ_j, -dx[..., None], INF).min(2)
        link_lead_dx = lead_dx.clone()                       # link vehicles only (head-of-line test)
        # chain: approach vehicles follow box vehicles that started from their lane
        path_c = self.path.clamp(min=0)
        pleg = self._g1(self.D("p_leg"), path_c)
        plane = self._g1(self.D("p_lane"), path_c)
        L_link = self._per_link(self.D("link_len"))
        match = on_link[:, :, None] & (self.link[:, :, None] < NLEG) & in_box[:, None, :] \
            & (pleg[:, None, :] == self.link[:, :, None])
        start_k = in_box[..., None] & (plane[..., None] == k)                          # [E, j, K]
        chain = (L_link - self.x)[:, :, None, None] + self.x[:, None, :, None]
        cdx, cidx = torch.where(match[..., None] & start_k[:, None], chain, INF).min(2)
        use = cdx < lead_dx
        lead_dx, lead_idx = torch.where(use, cdx, lead_dx), torch.where(use, cidx, lead_idx)
        T = {"occ": occ, "lead_idx": lead_idx, "lag_idx": lag_idx, "link_lead_dx": link_lead_dx,
             "lead_gap": torch.where(lead_dx < INF / 2, lead_dx - cfg.veh_len, INF),
             "lag_gap": torch.where(lag_dx < INF / 2, lag_dx - cfg.veh_len, INF)}
        rng_i = self.info[:, :, None, None]
        within = ahead[..., None] & occ_j & (dx[..., None] < rng_i)
        T["count"] = within.sum(2).float()
        T["speed_sum"] = (within * self.v[:, None, :, None]).sum(2)
        # lost-time rate of the queue each vehicle holds up behind it (slow vehicles within 60 m)
        loss_j = (0.5 + self.aggr) * (1.0 - self.v / self.vdes.clamp(min=1.0)).clamp(min=0.0)
        queued = behind[..., None] & occ_j & (dx[..., None] > -60.0) & (self.v[:, None, :, None] < 3.0)
        T["queue_cost"] = (queued * loss_j[:, None, :, None]).sum(2)
        gap_j = torch.minimum(T["lead_gap"], self.info[..., None])
        near = (dx.abs()[..., None] < rng_i) & occ_j & same[..., None]
        cand = torch.where(near, gap_j[:, None, :, :], -1.0)
        best, bj = cand.max(dim=2)
        pos = torch.gather(dx[..., None].expand(-1, -1, -1, K), 2, bj[:, :, None, :]).squeeze(2) \
            + torch.gather(gap_j, 1, bj) * 0.5
        none = best < 0
        T["best_gap"] = torch.where(none, self.info[..., None], best)
        T["best_pos"] = torch.where(none, 0.0, pos)
        # box leaders: same path ahead, diverging path still in the shared start, exit-lane chain
        same_path = in_box[:, :, None] & in_box[:, None, :] & (self.path[:, :, None] == self.path[:, None, :])
        div = in_box[:, :, None] & in_box[:, None, :] & (self._pair(self.D("c_kind"), self.path, self.path) == ixd.C_DIVERGE)
        div_out_j = self._pair(self.D("c_out"), self.path, self.path).transpose(1, 2)   # zone end on j's path
        div = div & (self.x[:, None, :] - cfg.veh_len < div_out_j)
        box_ahead = (same_path | div) & order
        bdx, bidx = torch.where(box_ahead, dx, INF).min(2)
        p_len = self._g1(self.D("p_len"), path_c)
        exit_link = NLEG + self._g1(self.D("p_exit"), path_c)
        exit_lane = self._g1(self.D("p_exit_lane"), path_c)
        on_exit = on_link[:, None, :] & (self.link[:, None, :] == exit_link[:, :, None]) & \
            ((self.lane_from[:, None, :] == exit_lane[:, :, None]) | (self.lane_to[:, None, :] == exit_lane[:, :, None]))
        edx = (p_len - self.x)[:, :, None] + self.x[:, None, :]
        edx, eidx = torch.where(in_box[:, :, None] & on_exit, edx, INF).min(2)
        use = edx < bdx
        T["box_lead_gap"] = torch.where(torch.minimum(bdx, edx) < INF / 2, torch.minimum(bdx, edx) - cfg.veh_len, INF)
        T["box_lead_idx"] = torch.where(use, eidx, bidx)
        return T

    def _lane_geometry(self):
        on_app = self.link < NLEG
        ls, le = self._per_link(self.D("link_ls")), self._per_link(self.D("link_le"))      # [E, N, K]
        x = self.x[..., None]
        exists = (ls <= x) & (x <= le)
        L_link = self._per_link(self.D("link_len"))
        mov_app = torch.cat([self.D("lane_mov"), torch.full_like(self.D("lane_mov"), 7)], 1)  # exits: any
        lm = self._per_link(mov_app)
        reaches = le >= (L_link[..., None] - 1.0)
        ok = reaches & (((lm >> self.mov[..., None]) & 1) == 1)
        ok = torch.where(on_app[..., None], ok, le > 0)
        diff = self.lanes.view(1, 1, 1, -1) - self.lanes.view(1, 1, -1, 1)
        dist = torch.where(ok[:, :, None, :], diff, 99)
        pick = dist.abs().argmin(-1, keepdim=True)
        lanes_to_ok = torch.gather(dist, 3, pick).squeeze(-1)
        lanes_to_ok = torch.where(lanes_to_ok.abs() >= 99, 0, lanes_to_ok)
        return exists, ok, lanes_to_ok, ls, le, L_link

    # ------------------------------------------------------------------ observation
    def _agents(self):
        return self.active & (self.link >= 0)

    def observe(self, build_obs=True):
        """Returns (obs, lat_mask, valid) and caches tables; ``build_obs=False`` only refreshes
        the cache (physics substeps)."""
        cfg, Rn = self.cfg, self.cfg.obs_norm_range
        T = self._tables()
        exists, ok, lanes_to_ok, ls_v, le_v, L_link = self._lane_geometry()
        ctl = self._control()
        c = self._center()
        plan = self._plan(c)
        rank, st = self._rank(plan, ctl)
        on_link = self.active & (self.link >= 0)
        on_app = on_link & (self.link < NLEG)
        head = on_app & ~self.granted & ~(self.lane_from != self.lane_to) & (self._gk(T["link_lead_dx"], c) >= INF / 2)
        grant_ok, stage_ok, release_ok, lane_held = self._permission(T, plan, rank, st, head)
        blocked = on_app & ~self.granted & ((rank <= 0) | (head & ~grant_ok))
        tts = torch.where(self.D("ctrl")[:, None] == ixd.IX_SIGNAL, self._g1(ctl["tts"], plan.clamp(min=0)), 0.0)
        tts = torch.where(on_app, tts, 0.0)
        g = lambda t, kk: self._gk(t, kk)
        gv = lambda t, j: self._g1(t, j)
        need = g(lanes_to_ok, c)
        deadline = L_link - self.x
        in_lc = self.lane_from != self.lane_to
        lc_dir = (self.lane_to - self.lane_from).sign()
        intent_dir = torch.where(in_lc, lc_dir, need.sign())
        intent_tgt = c + intent_dir
        urgency = torch.where(in_lc, 1.0, (1.0 - deadline / 300.0).clamp(0.1, 1.0)) * (intent_dir != 0)
        self._cache = dict(T=T, exists=exists, ok=ok, lanes_to_ok=lanes_to_ok, blocked=blocked, c=c, need=need,
                           in_lc=in_lc, lc_dir=lc_dir, plan=plan, rank=rank, sig_state=st, ctl=ctl,
                           grant_ok=grant_ok, stage_ok=stage_ok, release_ok=release_ok, lane_held=lane_held, head=head,
                           ls=ls_v, le=le_v, L_link=L_link)
        if not build_obs:
            return None
        # control category
        ctrl = self.D("ctrl")[:, None]
        pleg = self.link.clamp(0, NLEG - 1)
        major = self._g1(self.D("major").long(), pleg).bool()
        cat = torch.full_like(self.link, CAT_FREE)
        cat = torch.where(ctrl == ixd.IX_SIGNAL, CAT_SIGNAL, cat)
        cat = torch.where(ctrl == ixd.IX_AWSC, CAT_AWSC, cat)
        cat = torch.where((ctrl == ixd.IX_TWSC) | (ctrl == ixd.IX_YIELD), torch.where(major, CAT_TWSC_MAJOR,
                          torch.where(ctrl == ixd.IX_TWSC, CAT_TWSC_MINOR, CAT_YIELD_MINOR)), cat)
        cat = torch.where(on_app, cat, CAT_FREE)
        lim = self.limit[:, None]
        rank_obs = torch.where(on_app & ~self.granted, rank / 2.0, 1.0)
        ego = [self.v / lim, self.vdes / lim, self.aggr, self.info / Rn, self.prog, lc_dir.float(),
               need.clamp(-3, 3) / 3.0, (deadline / 300.0).clamp(0, 1), (deadline / 600.0).clamp(0, 1)]
        ego += [(cat == kk).float() for kk in range(NUM_CAT)]
        ego += [rank_obs, blocked.float(), (tts / 60.0).clamp(0, 1), self.stopped_once.float()]
        ego += [(self.mov == m).float() for m in range(NUM_MOV)]
        ego += [self.a_prev / 3.0]
        assert len(ego) == len(EGO_FEATURES)
        feats = []
        for d in REL_LANES:
            kk = c + d
            valid = (kk >= 0) & (kk < K)
            kc = kk.clamp(0, K - 1)
            ls, le = g(ls_v, kc), g(le_v, kc)
            ex = g(exists, kc) & valid
            lead_j, lag_j = g(T["lead_idx"], kc), g(T["lag_idx"], kc)
            lead_gap, lag_gap = g(T["lead_gap"], kc), g(T["lag_gap"], kc)
            has_lead, has_lag = lead_gap < self.info, lag_gap < self.info
            cnt = g(T["count"], kc)
            fl = [ex.float(),
                  ((le - self.x) / 300.0).clamp(0, 1) * valid,
                  ((ls - self.x) / 300.0).clamp(0, 1) * valid * (ls < INF),
                  (g(ok, kc) & valid).float(),
                  torch.where(valid, g(lanes_to_ok, kc).abs().clamp(max=3) / 3.0, 1.0),
                  torch.where(has_lead, lead_gap, self.info).clamp(-5, Rn) / Rn,
                  torch.where(has_lead, gv(self.v, lead_j) - self.v, 0.0) / 10.0,
                  torch.where(has_lead & (gv(intent_tgt, lead_j) == c), gv(urgency, lead_j), 0.0),
                  torch.where(has_lag, lag_gap, self.info).clamp(-5, Rn) / Rn,
                  torch.where(has_lag, gv(self.v, lag_j) - self.v, 0.0) / 10.0,
                  torch.where(has_lag & (gv(intent_tgt, lag_j) == c), gv(urgency, lag_j), 0.0),
                  cnt / 10.0,
                  torch.where(cnt > 0, g(T["speed_sum"], kc) / cnt.clamp(min=1), lim) / lim,
                  g(T["best_gap"], kc).clamp(max=Rn) / 100.0,
                  g(T["best_pos"], kc).clamp(-Rn, Rn) / Rn]
            assert len(fl) == len(LANE_FEATURES)
            feats += [z * valid if z.dtype != torch.bool else z for z in fl]
        obs = torch.stack([z.float() for z in ego + feats], dim=-1) * on_link[..., None]
        return obs, self._lat_mask(), on_link.clone()

    # ------------------------------------------------------------------ entry permission
    def _participant_s(self):
        """Position along the (planned) path: negative = metres before the stop line."""
        L_link = self._per_link(self.D("link_len"))
        return torch.where(self.link < 0, self.x, self.x - L_link)

    def _permission(self, T, plan, rank, st, head):
        """Entry decisions (section 3 of the spec) for the head vehicle of every approach lane and
        for every staged vehicle (a permitted turner waiting inside the box). Candidates are
        paired with every vehicle: [E, CAND, N] instead of [E, N, N].

        Returns [E, N] masks: grant_ok (a head may enter and cross), stage_ok (a head may pull
        into the box and wait at its path's staging point), release_ok (a staged vehicle may go),
        lane_held (a staged vehicle from the head's lane is waiting in the box).
        """
        cfg = self.cfg
        E, N = self.E, self.N
        on_app = self.active & (self.link >= 0) & (self.link < NLEG)
        part = self.active & ((self.link < 0) | self.granted)
        holding = self.active & self.staged
        s = self._participant_s()
        dist = (-s).clamp(min=0)
        c = self._center()
        cand = head | holding
        score = cand.float() * 2.0 - self.slot.float() / (2.0 * N)              # candidates first
        hi = torch.topk(score, min(CAND, N), dim=1).indices                       # [E, C]
        has = torch.gather(cand, 1, hi)
        G = lambda t: torch.gather(t, 1, hi)                                    # [E, N] -> [E, C]
        plan_i, v_i, s_i, rank_i, aggr_i, hold_i = G(plan), G(self.v), G(s), G(rank), G(self.aggr), G(holding)
        kind = self._pair(self.D("c_kind"), plan_i, plan)                         # [E, C, N]
        conflict = ((kind == ixd.C_CROSS) | (kind == ixd.C_MERGE)) & (plan_i[:, :, None] >= 0) & (plan[:, None, :] >= 0)
        conflict &= self.active[:, None, :] & (hi[:, :, None] != self.slot[None, None, :]) & has[:, :, None]
        i_in = self._pair(self.D("c_in"), plan_i, plan)                           # zone on i's path
        i_out = self._pair(self.D("c_out"), plan_i, plan)
        j_in = self._pair(self.D("c_in"), plan, plan_i).transpose(1, 2)           # zone on j's path
        j_out = self._pair(self.D("c_out"), plan, plan_i).transpose(1, 2)
        t_clear = self._t_reach(v_i[:, :, None], i_out - s_i[:, :, None] + cfg.veh_len)
        h = (1.0 + (1.0 - aggr_i))[:, :, None]
        s_j = s[:, None, :]
        cleared = s_j - cfg.veh_len > j_out
        inside = (s_j >= j_in) & ~cleared
        v_j = self.v[:, None, :].expand_as(j_in)
        eta_j = self._t_reach(v_j, j_in - s_j)
        eta_i = self._t_reach(v_i[:, :, None], i_in - s_i[:, :, None])
        r_i, r_j = rank_i[:, :, None], rank[:, None, :]
        # committed traffic. A staged vehicle waiting for its gap is not about to enter a zone,
        # so it blocks only zones it physically occupies; once its own phase has ended it must
        # clear first, so then it blocks like any committed vehicle (except another staged
        # vehicle's release).
        waiting_j = holding[:, None, :] & ~((r_j <= 0) & ~hold_i[:, :, None])
        # a vehicle already inside a zone (a staged turner) has priority to clear it: others wait
        # at the zone (their box walls), so their arrival times do not count
        self_inside = s_i[:, :, None] >= i_in
        block_a = conflict & part[:, None, :] & ~cleared & (inside | ((eta_j < t_clear + h) & ~waiting_j & ~self_inside))
        app_j = (on_app & ~self.granted)[:, None, :] & (dist[:, None, :] <= DECIDE_RANGE)
        # gap acceptance against higher-rank traffic: speed-capped arrival time vs critical gap
        far_j = (on_app & ~self.granted)[:, None, :] & (dist[:, None, :] <= GAP_RANGE)
        vmax_j = torch.maximum(self.vdes, self.v)[:, None, :].expand_as(j_in)
        eta_cap = self._t_reach_capped(v_j, j_in - s_j, vmax_j)
        crit = self.k_crit_gap[G(self.mov)][:, :, None] + (1.0 - aggr_i)[:, :, None]
        need_gap = torch.maximum(crit, t_clear + PET_MIN)
        block_b = conflict & far_j & (r_j > r_i + 1e-3) & (eta_cap < need_gap)
        awsc = ((self.D("ctrl") == ixd.IX_AWSC) | (self.D("ctrl") == ixd.IX_TWSC))[:, None, None]
        head_j = (on_app & ~self.granted & (self._gk(T["link_lead_dx"], c) >= INF / 2))[:, None, :]
        st_i, leg_i = G(self.stop_time)[:, :, None], G(self.link)[:, :, None]
        st_j, leg_j = self.stop_time[:, None, :], self.link[:, None, :]
        earlier = (st_j < st_i - 1e-3) | ((st_j - st_i).abs() <= 1e-3) & (leg_j < leg_i)
        same_rank = conflict & app_j & ((r_j - r_i).abs() <= 1e-3) & (r_i > 0)
        block_c = same_rank & torch.where(awsc, head_j & self.stopped_once[:, None, :] & earlier,
                                          (eta_j < eta_i) & (eta_j < t_clear + h))
        clear_only = hold_i & (rank_i <= 0)        # phase over: clear as soon as the box allows
        blocked = block_a.any(-1) | (~clear_only & (block_b | block_c).any(-1))   # [E, C]
        # crosswalks on the path must be free (a staged vehicle is already past the first one)
        busy = self.ped_busy_until > self.t_env[:, None] * cfg.dt                 # [E, 4]
        pc = plan_i.clamp(min=0)
        ped_start = self._g1(busy.long(), self._g1(self.D("p_leg"), pc)).bool()
        ped_end = self._g1(busy.long(), self._g1(self.D("p_exit"), pc)).bool()
        ped = ped_end | (ped_start & ~hold_i)
        # exit lane must have room for this vehicle (plus everyone else already heading there)
        b = self._g1(self.D("p_exit"), pc)
        j = self._g1(self.D("p_exit_lane"), pc)
        heading = part & (self.path >= 0)
        path_c = self.path.clamp(min=0)
        bj_part = self._g1(self.D("p_exit"), path_c) * K + self._g1(self.D("p_exit_lane"), path_c)
        n_head = (heading[:, None, :] & (bj_part[:, None, :] == (b * K + j)[:, :, None])).sum(-1) - hold_i.long()
        on_exit = self.active & (self.link >= NLEG)
        el = self.link - NLEG
        same_exit = on_exit[:, None, :] & (el[:, None, :] == b[:, :, None]) & \
            ((self.lane_from[:, None, :] == j[:, :, None]) | (self.lane_to[:, None, :] == j[:, :, None]))
        x_last = torch.where(same_exit, self.x[:, None, :], 1e3).min(-1).values
        room = x_last - n_head * (cfg.veh_len + cfg.s0) >= cfg.veh_len + cfg.s0
        ok = has & ~ped & room & (plan_i >= 0)
        # a staged vehicle from the same lane waits in the box: the vehicles behind it cannot go
        # (only the next staging place on the same path may be taken)
        src_leg = self._g1(self.D("p_leg"), path_c)[:, None, :]
        src_lane = self._g1(self.D("p_lane"), path_c)[:, None, :]
        same_src = holding[:, None, :] & (src_leg == G(self.link)[:, :, None]) & (src_lane == G(c)[:, :, None])             & (hi[:, :, None] != self.slot[None, None, :])
        lane_held = same_src.any(-1)
        lane_held_other = (same_src & (self.path[:, None, :] != plan_i[:, :, None])).any(-1)
        grant_c = ok & ~hold_i & (rank_i > 0) & ~blocked & ~lane_held
        release_c = ok & hold_i & ~blocked
        # staging: the movement is permitted now, the path has a staging point with a free place,
        # and nothing committed is crossing the zones before that point
        sw_i = self._g1(self.D("s_wait"), pc)
        n_st = self._g1(self.D("n_stage"), pc)
        waiting = (holding[:, None, :] & (self.path[:, None, :] == plan_i[:, :, None])).sum(-1)
        before = (block_a & (i_in < sw_i[:, :, None])).any(-1)
        stage_c = has & ~hold_i & (rank_i > 0) & (G(st) == ixd.SIG_PERM) & (n_st > waiting) & (sw_i > 0) \
            & ~before & ~ped_start & room & (plan_i >= 0) & ~lane_held_other
        dst = torch.where(has, hi, N)
        put = lambda v: torch.zeros(E, N + 1, dtype=torch.bool, device=self.dev).scatter(1, dst, v)[:, :N]
        return put(grant_c), put(stage_c), put(release_c), put(lane_held)

    def _lat_mask(self):
        C = self._cache
        in_lc, lc_dir = C["in_lc"], C["lc_dir"]
        on_link = self.active & (self.link >= 0)
        free = on_link & ~in_lc & ~self.granted
        start_l = free & self._safe_to_enter(self.lane_from - 1)
        start_r = free & self._safe_to_enter(self.lane_from + 1)
        # during a change: continuing is always allowed; aborting only once continuing became unsafe
        lc = in_lc & on_link
        abort_ok = lc & ~self._safe_to_enter(self.lane_to)
        left = start_l | (lc & (lc_dir < 0)) | (abort_ok & (lc_dir > 0))
        right = start_r | (lc & (lc_dir > 0)) | (abort_ok & (lc_dir < 0))
        mask = torch.stack([torch.ones_like(in_lc), left, right], dim=-1)
        keep_only = torch.arange(NUM_LAT, device=self.dev) == LAT_KEEP
        return torch.where(on_link[..., None], mask, keep_only)

    def _safe_to_enter(self, kk):
        cfg, C = self.cfg, self._cache
        T = C["T"]
        kc = kk.clamp(0, K - 1)
        g = lambda t: self._gk(t, kc)
        lead_gap, lag_gap = g(T["lead_gap"]), g(T["lag_gap"])
        v_lead, v_lag = self._g1(self.v, g(T["lead_idx"])), self._g1(self.v, g(T["lag_idx"]))
        b = cfg.b_lc_safe
        ok_lead = (lead_gap >= INF / 2) | ((lead_gap >= cfg.s0) & ((self.v ** 2 - v_lead ** 2) / (2 * b) <= lead_gap - cfg.s0))
        ok_lag = (lag_gap >= INF / 2) | ((lag_gap >= cfg.s0) &
                                          ((v_lag ** 2 - self.v ** 2) / (2 * b) + v_lag * cfg.dt <= lag_gap - cfg.s0))
        ls, le = g(C["ls"]), g(C["le"])
        geo = (kk >= 0) & (kk < K) & (ls <= self.x) & (le - self.x > 15.0) & (C["L_link"] - self.x > 10.0)
        # no lane changes just after the box: vehicles still in it are heading into these lanes
        geo &= (self.link < NLEG) | (self.x > 30.0)
        return geo & ok_lead & ok_lag

    # ------------------------------------------------------------------ dynamics
    def step(self, lat, lon):
        """Advance one decision interval (cfg.substeps physics steps). Returns (reward, done_rl):
        done_rl ends an agent's RL episode (it entered the box, finished, collided, timed out or
        its env was reset)."""
        return decision_step(self, lat, lon)

    def _substep(self, lat, lon):
        """One physics step of cfg.dt."""
        cfg, rc, C = self.cfg, self.rcfg, self._cache
        T, dt = C["T"], cfg.dt
        rate = dt / REWARD_DT                 # per-second reward terms are defined per 0.5 s
        a_before = self.a_prev
        E, N = self.E, self.N
        act0 = self.active.clone()
        on_link0 = act0 & (self.link >= 0)
        on_app0 = on_link0 & (self.link < NLEG)
        in_box0 = act0 & (self.link < 0)
        now = self.t_env.float() * dt

        # ---- entry permission (heads) and amber commitment
        L_link = C["L_link"]
        stop_gap = L_link - self.x
        commit_zone = (stop_gap - cfg.s0 < self.v ** 2 / (2 * 3.5)) | (stop_gap < 2.0)
        amber = (self.D("ctrl")[:, None] == ixd.IX_SIGNAL) & (C["sig_state"] == ixd.SIG_Y) & commit_zone
        decide = C["head"] & (stop_gap <= torch.maximum(torch.full_like(stop_gap, 30.0), self.v ** 2 / 4.0 + 10.0))
        new_grant = on_app0 & ~self.granted & ((decide & C["grant_ok"]) | (C["head"] & amber & ~C["lane_held"]))
        # permitted turners without a gap pull into the box and wait at the staging point
        new_stage = on_app0 & ~self.granted & ~new_grant & decide & C["stage_ok"]
        self.staged = (self.staged & ~C["release_ok"]) | new_stage
        self.granted |= new_grant | new_stage
        self.path = torch.where(new_grant | new_stage, C["plan"], self.path)

        # ---- lateral (link vehicles only)
        in_lc, lc_dir = C["in_lc"], C["lc_dir"]
        want = torch.where(lat == LAT_LEFT, -1, torch.where(lat == LAT_RIGHT, 1, 0))
        start = on_link0 & ~in_lc & ~self.granted & (want != 0)
        abort = on_link0 & in_lc & (want == -lc_dir)
        new_to = torch.where(start, self.lane_from + want, self.lane_to)
        dxp = self.x[:, None, :] - self.x[:, :, None]
        v_i, v_j = self.v[:, :, None], self.v[:, None, :]
        need = cfg.veh_len + cfg.s0 + (v_i ** 2 - v_j ** 2).clamp(min=0) / (2 * cfg.b_lc_safe) + v_i * dt
        clash = start[:, :, None] & start[:, None, :] & (new_to[:, :, None] == new_to[:, None, :]) \
            & (self.link[:, :, None] == self.link[:, None, :]) & (dxp > -1e-3) & (dxp < need) \
            & (self.slot[:, None] != self.slot[None, :])
        start = start & ~clash.any(-1)
        new_to = torch.where(start, self.lane_from + want, self.lane_to)
        k_new = new_to.clamp(0, K - 1)
        f_idx, f_gap = self._gk(T["lag_idx"], k_new), self._gk(T["lag_gap"], k_new)
        has_f = start & (f_gap < INF / 2)
        v_f = self._g1(self.v, f_idx)
        rows = lambda t: torch.gather(t, 1, f_idx[..., None].expand(-1, -1, K))
        f_lead_gap = self._gk(rows(T["lead_gap"]), k_new)
        f_lead_v = self._g1(self.v, self._gk(rows(T["lead_idx"]), k_new))
        vdes_f, T_f = self._g1(self.vdes, f_idx), self._g1(self._headway(), f_idx)
        a_f_before = torch.where(f_lead_gap < INF / 2, self._idm(v_f, vdes_f, T_f, f_lead_gap, v_f - f_lead_v),
                                 self._idm(v_f, vdes_f, T_f, torch.full_like(v_f, INF), torch.zeros_like(v_f)))
        a_f_after = self._idm(v_f, vdes_f, T_f, f_gap, v_f - self.v)
        impact = torch.where(has_f, (a_f_before - a_f_after).clamp(0, 6), 0.0)
        self.lane_to = new_to
        fr, to = self.lane_from.clone(), self.lane_to.clone()
        self.lane_from = torch.where(abort, to, fr)
        self.lane_to = torch.where(abort, fr, self.lane_to)
        self.prog = torch.where(abort, 1.0 - self.prog, self.prog)

        # ---- longitudinal
        T_h = self._headway()
        pc = self.path.clamp(min=0)
        pm = self._g1(self.D("p_mov"), pc)
        # speed caps: turn speed in the box, curve profile on links
        li = self.link.clamp(min=0)
        vi = (self.x / ixd.VCAP_DS).long().clamp(0, ixd.VCAP_N - 2) + 1
        vcap_tab = self.D("link_vcap").reshape(E, LK * ixd.VCAP_N)
        v_curve = torch.gather(vcap_tab, 1, li * ixd.VCAP_N + vi)
        vdes = torch.where(self.link < 0, torch.minimum(self.vdes, self.k_turn_speed[pm]),
                           torch.minimum(self.vdes, v_curve))
        free = torch.full_like(self.v, INF)
        a_idm = self._idm(self.v, vdes, T_h, free, torch.zeros_like(self.v))
        v_cap = torch.full_like(self.v, INF)

        def follow(on, gap, vl):
            nonlocal a_idm, v_cap
            on = on & (gap < INF / 2)
            a_idm = torch.where(on, torch.minimum(a_idm, self._idm(self.v, vdes, T_h, gap, self.v - vl)), a_idm)
            v_cap = torch.where(on, torch.minimum(v_cap, self._krauss(self.v, gap, vl)), v_cap)

        zero = torch.zeros_like(self.v)
        for kk in (self.lane_from, self.lane_to):
            gap = self._gk(T["lead_gap"], kk)
            follow(on_link0, gap, self._g1(self.v, self._gk(T["lead_idx"], kk)))
        follow(in_box0, T["box_lead_gap"], self._g1(self.v, T["box_lead_idx"]))
        # stop line for approach vehicles without permission
        follow(on_app0 & ~self.granted & (stop_gap > 0), stop_gap, zero)
        # conflict and crosswalk walls for vehicles in or committed to the box
        wall = self._box_walls(now)
        follow(act0 & ((self.link < 0) | self.granted), wall, zero)
        delta = self.k_lon_deltas[lon]
        delta = torch.where(on_link0, delta, 0.0)
        # a vehicle granted entry has accepted its gap: it commits (no braking offset)
        delta = torch.where(self.granted, delta.clamp(min=0.0), delta)
        a = (a_idm + delta).clamp(-cfg.b_max, cfg.a_max + 1.5)
        a = torch.minimum(a, (v_cap - self.v) / dt).clamp(min=-cfg.b_max)
        v_new = (self.v + a * dt).clamp(min=0.0)
        v_new = torch.minimum(v_new, self.limit[:, None] * 1.15)
        a = (v_new - self.v) / dt
        jerk = torch.where(act0, (a - self.a_prev) / dt, 0.0)
        self.x = torch.where(act0, self.x + 0.5 * (self.v + v_new) * dt, self.x)
        self.v, self.a_prev = torch.where(act0, v_new, self.v), torch.where(act0, a, self.a_prev)

        # ---- lateral progress
        moving = on_link0 & (self.lane_from != self.lane_to)
        self.prog = torch.where(moving, self.prog + dt / cfg.lc_duration, 0.0)
        done_lc = moving & (self.prog >= 1.0 - 1e-4)
        self.lane_from = torch.where(done_lc, self.lane_to, self.lane_from)
        self.prog = torch.where(done_lc, 0.0, self.prog)

        # ---- stop bookkeeping
        first_stop = on_app0 & ~self.stopped_once & (self.v < 0.5) & ((L_link - self.x) < 8.0) & C["head"]
        self.stop_time = torch.where(first_stop, now[:, None].expand_as(self.stop_time), self.stop_time)
        self.stopped_once |= first_stop

        # ---- safety events (before transitions)
        lead_gap_c = self._gk(T["lead_gap"], C["c"])
        v_lead_c = self._g1(self.v, self._gk(T["lead_idx"], C["c"]))
        closing = self.v - v_lead_c
        unsafe = on_link0 & (lead_gap_c < INF / 2) & (closing > 0.1) & (lead_gap_c / closing.clamp(min=0.1) < 1.5)
        hard = act0 & (a < -4.0)
        hard_onset = hard & (a_before >= -4.0)
        collided = act0 & self._collisions()

        # ---- transitions: stop line -> box -> exit link -> gone
        crossed = on_app0 & ~collided & (self.x >= L_link)
        violation = crossed & ~self.granted
        plan = C["plan"]
        path_new = torch.where(self.granted, self.path, plan)
        self.path = torch.where(crossed, path_new, self.path)
        pc = self.path.clamp(min=0)
        took = self._g1(self.D("p_mov"), pc)
        self.missed = torch.where(crossed, took != self.mov, self.missed)
        success, missed = crossed & ~self.missed, crossed & self.missed
        self.x = torch.where(crossed, self.x - L_link, self.x)
        self.link = torch.where(crossed, -1, self.link)
        self.prog = torch.where(crossed, 0.0, self.prog)
        self.granted = torch.where(crossed, False, self.granted)
        p_len = self._g1(self.D("p_len"), pc)
        to_exit = act0 & ~collided & (self.link < 0) & (self.x >= p_len) & ~crossed
        exit_lane = self._g1(self.D("p_exit_lane"), pc)
        self.link = torch.where(to_exit, NLEG + self._g1(self.D("p_exit"), pc), self.link)
        self.x = torch.where(to_exit, self.x - p_len, self.x)
        self.lane_from = torch.where(to_exit | crossed, exit_lane, self.lane_from)
        self.lane_to = torch.where(to_exit | crossed, exit_lane, self.lane_to)
        self.path = torch.where(to_exit, -1, self.path)
        self.staged = self.staged & ~to_exit
        self.stopped_once = torch.where(to_exit | crossed, False, self.stopped_once)
        L_now = self._per_link(self.D("link_len"))
        finished = act0 & ~collided & (self.link >= NLEG) & (self.x >= L_now)
        self.age = self.age + act0.long()
        timeout = act0 & ~finished & ~collided & (self.age >= 3 * cfg.max_age_steps)
        self.t_env += 1
        env_reset = self.t_env >= cfg.episode_steps
        truncated = act0 & env_reset[:, None] & ~finished & ~collided & ~timeout
        done = finished | collided | timeout | truncated
        done_rl = on_link0 & (crossed | done)
        self.last_outcome = dict(success=success, missed=missed, collided=collided, timeout=timeout,
                                 crossed=crossed, finished=finished, violation=violation)

        # ---- pedestrians
        self._peds(now)

        # ---- reward (RL agents = vehicles on links at the start of the step)
        aggr = self.aggr
        polite = 0.2 + 0.8 * (1.0 - aggr)
        r = -rc.efficiency * rate * (0.5 + aggr) * (1.0 - self.v / self.vdes.clamp(min=1.0)).clamp(min=0.0)
        r = r - rc.comfort * rate * ((a / 3.0) ** 2 + (jerk / 10.0) ** 2)
        r = r - rc.lc_cost * (1.5 - aggr) * start - rc.abort_cost * abort
        r = r - rc.lag_impact * polite * impact - rate * (rc.unsafe_ttc * unsafe + rc.hard_brake * hard)
        # blocking: standing in the way with free road ahead costs the lost time of the queue behind
        gap0 = self._gk(T["lead_gap"], C["c"])
        unconstrained = (gap0 > 15.0 + 1.5 * self.v) & (stop_gap > 15.0) & on_link0
        r = r - rc.blocking * rate * polite * rc.efficiency * self._gk(T["queue_cost"], C["c"]) * unconstrained
        r = r + rc.success * success + rc.missed_route * missed + rc.timeout * timeout + rc.collision * collided
        # a missed movement costs a detour, priced at the driver's own lost-time rate
        r = r - rc.efficiency * (0.5 + aggr) * (rc.detour_s / REWARD_DT) * missed
        r = r + rc.violation * violation
        r = torch.where(on_link0, r, 0.0)
        n_act = on_link0.sum(1, keepdim=True).clamp(min=1)
        r = r + rc.team_share * (1.0 - aggr) * (r.sum(1, keepdim=True) / n_act) * on_link0

        # ---- stats (trip-level: a trip ends at the end of the exit link)
        self.acc_lc += start
        self.acc_abort += abort
        self.acc_hb += hard_onset
        self.acc_unsafe += unsafe
        self.acc_jerk += jerk ** 2 * act0
        st = self.stats
        st[:, S["veh_steps"]] += act0.sum(1)
        st[:, S["violations"]] += violation.sum(1)
        tt = self.age * dt
        st[:, S["exited"]] += finished.sum(1)
        st[:, S["success"]] += (finished & ~self.missed).sum(1)
        st[:, S["missed"]] += (finished & self.missed).sum(1)
        st[:, S["collisions"]] += collided.sum(1)
        st[:, S["timeouts"]] += timeout.sum(1)
        fin = finished.float()
        st[:, S["travel_time"]] += (tt * fin).sum(1)
        st[:, S["delay"]] += ((tt - self.t_ff).clamp(min=0) * fin).sum(1)
        st[:, S["lc_events"]] += has_f.sum(1)
        st[:, S["induced_decel"]] += torch.where(has_f, a_f_after.clamp(-cfg.b_max, 0), 0.0).sum(1)
        for name, acc in (("lane_changes", self.acc_lc), ("aborts", self.acc_abort), ("hard_brakes", self.acc_hb),
                          ("unsafe_steps", self.acc_unsafe), ("jerk_sq", self.acc_jerk)):
            st[:, S[name]] += (acc * done).sum(1)

        self.active &= ~done
        self._reset_envs(env_reset)
        self._spawn()
        return r, done_rl

    def _box_walls(self, now):
        """Distance to the nearest conflict-zone or crosswalk wall for vehicles in / committed
        to the box (section 4 and 5 of the spec)."""
        cfg = self.cfg
        part = self.active & ((self.link < 0) | self.granted)
        s = self._participant_s()
        path = self.path
        kind = self._pair(self.D("c_kind"), path, path)
        conflict = ((kind == ixd.C_CROSS) | (kind == ixd.C_MERGE)) & part[:, :, None] & part[:, None, :] \
            & (self.slot[:, None] != self.slot[None, :])
        i_in = self._pair(self.D("c_in"), path, path)
        j_in = i_in.transpose(1, 2)
        j_out = self._pair(self.D("c_out"), path, path).transpose(1, 2)
        s_i, s_j = s[:, :, None], s[:, None, :]
        cleared_j = s_j - cfg.veh_len > j_out
        tol = torch.where(self.staged, 0.0, 0.5)[:, None, :]       # a staged vehicle stops 0.5 m short: not inside
        inside_j = (s_j >= j_in - tol) & ~cleared_j
        before_i = s_i < i_in - 0.5
        eta_i = self._t_reach(self.v[:, :, None], i_in - s_i)
        eta_j = self._t_reach(self.v[:, None, :].expand_as(j_in), j_in - s_j)
        tie = (eta_j - eta_i).abs() < 0.2
        j_first = ~inside_j & ~cleared_j & (eta_j < 2.0) & ((eta_j < eta_i - 0.2) | (tie & (self.slot[None, None, :] < self.slot[None, :, None])))
        j_first &= ~self.staged[:, None, :]          # a staged vehicle waits; it is not about to enter
        wall_pair = conflict & before_i & (inside_j | j_first)
        wall = torch.where(wall_pair, i_in - s_i, INF).min(-1).values
        s_wait = self._g1(self.D("s_wait"), path.clamp(min=0))
        wall = torch.where(self.staged, torch.minimum(wall, s_wait - s), wall)
        busy = self.ped_busy_until > now[:, None]
        pc = path.clamp(min=0)
        p_len = self._g1(self.D("p_len"), pc)
        b_start = self._g1(busy.long(), self._g1(self.D("p_leg"), pc)).bool() & (s < ixd.CROSSWALK[0])
        b_end = self._g1(busy.long(), self._g1(self.D("p_exit"), pc)).bool() & (s < p_len - ixd.CROSSWALK[1])
        wall = torch.where(part & b_start, torch.minimum(wall, ixd.CROSSWALK[0] - s), wall)
        wall = torch.where(part & b_end, torch.minimum(wall, p_len - ixd.CROSSWALK[1] - s), wall)
        return torch.where(part, wall, INF)

    def _peds(self, now):
        """Pedestrian arrivals and crossings (section 5)."""
        cfg = self.cfg
        arrive = torch.rand(self.E, NLEG, device=self.dev, generator=self.gen) < self.D("ped_rate") * cfg.dt
        self.ped_wait += arrive
        walk = self._cache["ctl"]["walk"]
        # crosswalk c is clear if no vehicle in/committed to the box is on its span
        part = self.active & ((self.link < 0) | self.granted)
        s = self._participant_s()
        pc = self.path.clamp(min=0)
        p_len = self._g1(self.D("p_len"), pc)
        on_start = part & (s > -1.0) & (s < ixd.CROSSWALK[1] + 1.0)
        on_end = part & (s > p_len - ixd.CROSSWALK[1] - 1.0) & (s < p_len)
        leg_s, leg_e = self._g1(self.D("p_leg"), pc), self._g1(self.D("p_exit"), pc)
        legs = torch.arange(NLEG, device=self.dev)
        occupied = ((on_start[..., None] & (leg_s[..., None] == legs)) | (on_end[..., None] & (leg_e[..., None] == legs))).any(1)
        go = (self.ped_wait > 0) & walk & ~occupied & (self.ped_busy_until <= now[:, None])
        self.ped_busy_until = torch.where(go, now[:, None] + self.D("ped_cross_time"), self.ped_busy_until)
        self.ped_wait = torch.where(go, 0.0, self.ped_wait)

    def _collisions(self):
        cfg = self.cfg
        on_link = self.active & (self.link >= 0)
        in_box = self.active & (self.link < 0)
        k = self.lanes.view(1, 1, -1)
        occ = (on_link[..., None] & ((self.lane_from[..., None] == k) | (self.lane_to[..., None] == k))).float()
        shared = (torch.einsum("enk,emk->enm", occ, occ) > 0) & (self.link[:, :, None] == self.link[:, None, :])
        dx = self.x[:, None, :] - self.x[:, :, None]
        ahead = ahead_of(self.x, self.slot)
        hit = (shared & ahead & (dx < cfg.veh_len - 0.5)).any(-1)
        same_path = in_box[:, :, None] & in_box[:, None, :] & (self.path[:, :, None] == self.path[:, None, :])
        hit |= (same_path & ahead & (dx < cfg.veh_len - 0.5)).any(-1)
        kind = self._pair(self.D("c_kind"), self.path, self.path)
        conflict = ((kind == ixd.C_CROSS) | (kind == ixd.C_MERGE)) & in_box[:, :, None] & in_box[:, None, :]
        i_in, i_out = self._pair(self.D("c_in"), self.path, self.path), self._pair(self.D("c_out"), self.path, self.path)
        inside_i = (self.x[:, :, None] >= i_in + 0.5) & (self.x[:, :, None] - cfg.veh_len <= i_out - 0.5)
        inside_j = inside_i.transpose(1, 2)
        both = conflict & inside_i & inside_j & (self.slot[None, None, :] > self.slot[None, :, None])
        hit |= both.any(-1)
        return hit

    # ------------------------------------------------------------------ demand
    def _spawn(self):
        """Vectorised entry queues: at most one vehicle per (approach, lane) per step."""
        cfg = self.cfg
        E, N = self.E, self.N
        on_link = self.active & (self.link >= 0)
        rate = self.D("spawn_rate") * self.demand_mult[:, None, None]
        arrive = torch.rand(E, NLEG, K, device=self.dev, generator=self.gen) < rate * cfg.dt
        self.pending = (self.pending + arrive.long()).clamp(max=10)
        lanes = self.lanes.view(1, 1, -1)
        occ = on_link[..., None] & ((self.lane_from[..., None] == lanes) | (self.lane_to[..., None] == lanes))  # [E,N,K]
        legs = torch.arange(NLEG, device=self.dev)
        here = occ[:, :, None, :] & (self.link[:, :, None, None] == legs[None, None, :, None])                   # [E,N,4,K]
        d_ahead, lead = torch.where(here, self.x[:, :, None, None], INF).min(1)                                  # [E,4,K]
        v_lead = torch.gather(self.v, 1, lead.view(E, -1)).view(E, NLEG, K)
        lim = self.limit[:, None, None]
        v0 = (0.9 * lim).expand(E, NLEG, K)
        room = (d_ahead - cfg.veh_len - cfg.s0 - v0 * cfg.dt).clamp(min=0)
        v0 = torch.where(d_ahead < INF / 2, torch.minimum(v0, torch.sqrt(v_lead ** 2 + 2 * cfg.b_comf * room)), v0)
        clear = (d_ahead >= INF / 2) | (d_ahead - cfg.veh_len - cfg.s0 >= (0.8 * v0).clamp(min=2.0))
        go = ((self.pending > 0) & clear).view(E, -1)                                                         # [E,4K]
        # hand out free slots in order within each env (fixed shapes, no host sync)
        free = ~self.active
        rank = go.long().cumsum(1) - 1
        go &= rank < free.sum(1, keepdim=True)
        free_sorted = torch.argsort((~free).to(torch.int8), dim=1, stable=True)                                # free slots first
        slot = torch.where(go, torch.gather(free_sorted, 1, rank.clamp(min=0)), N)                            # [E,4K]; N = none
        C4 = NLEG * K
        a_c = (torch.arange(C4, device=self.dev) // K).expand(E, -1)
        k_c = (torch.arange(C4, device=self.dev) % K).expand(E, -1)
        r = lambda: torch.rand(E, C4, device=self.dev, generator=self.gen)
        aggr = r() if self.fixed_aggr is None else torch.full((E, C4), self.fixed_aggr, device=self.dev)
        lo, hi = cfg.info_range
        info = lo + (hi - lo) * r() if self.fixed_info is None else torch.full((E, C4), self.fixed_info, device=self.dev)
        lim_e = self.limit[:, None]
        truck = r() < cfg.truck_share
        vdes = lim_e * torch.where(truck, 0.6 + 0.2 * r(), (0.9 + 0.2 * r()) * (0.95 + 0.1 * aggr))
        probs = self.D("mov_prob").reshape(-1, NUM_MOV) + 1e-6 * (torch.arange(NUM_MOV, device=self.dev) == MOV_THROUGH)
        mov = torch.multinomial(probs, 1, generator=self.gen).view(E, C4)
        L_trip = self.D("L_app")[:, :, None].expand(-1, -1, K).reshape(E, C4) + 25.0 \
            + self.D("L_exit")[:, :, None].expand(-1, -1, K).reshape(E, C4)
        put = lambda name, val: setattr(self, name, scatter_slots(getattr(self, name), slot, val))
        put("active", True)
        put("link", a_c)
        put("path", -1)
        put("x", 0.0)
        put("v", v0.reshape(E, C4))
        put("a_prev", 0.0)
        put("lane_from", k_c)
        put("lane_to", k_c)
        put("prog", 0.0)
        put("mov", mov)
        put("age", 0)
        put("aggr", aggr)
        put("info", info)
        put("vdes", vdes)
        put("granted", False)
        put("staged", False)
        put("missed", False)
        put("stopped_once", False)
        put("stop_time", 0.0)
        put("t_ff", L_trip / torch.minimum(vdes, lim_e))
        for name in ("acc_lc", "acc_abort", "acc_hb", "acc_unsafe", "acc_jerk"):
            put(name, 0.0)
        self.pending = self.pending - go.view(E, NLEG, K).long()
        self.stats[:, S["spawned"]] += go.sum(1)

    # ------------------------------------------------------------------ baseline
    def rule_action(self, obs, lat_mask):
        from .env import LaneChangeEnv
        return LaneChangeEnv.rule_action(self, obs, lat_mask)
