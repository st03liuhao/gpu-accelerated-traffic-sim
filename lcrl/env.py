"""Batched GPU multi-lane traffic environment for training the lane-changing policy.

Every vehicle is an agent controlled by one shared policy (parameter sharing), so a
batch of E segments x N vehicle slots gives E*N decisions per simulation step, all
computed with dense tensor ops on the GPU.

Vehicle model
  * longitudinal: IDM (as in LPSim) + an RL acceleration offset, then a Krauss
    safe-speed shield against every leader the vehicle can see in the lanes it
    occupies (and stop lines / lane ends), so the agent cannot drive into anyone;
  * lateral: a lane change takes ``lc_duration`` seconds. While changing, the vehicle
    occupies both lanes (it is a leader/follower in both), and it can abort by
    steering back. Initiations are masked unless the gaps are physically safe
    (no-one has to brake harder than ``b_lc_safe``); *which* safe gap to take, when,
    and how to adjust speed to reach it is learned.
"""
import math

import numpy as np
import torch

from . import scenarios
from .config import (
    CTRL_SIGNAL, CTRL_STOP, CTRL_YIELD, EGO_FEATURES, LANE_FEATURES, LAT_KEEP, LAT_LEFT,
    LAT_RIGHT, LON_DELTAS, LON_NEUTRAL, MOV_LEFT, MOV_THROUGH, CTRL_TO_CAT, NUM_CAT, NUM_LAT, NUM_MOV,
    NUM_SCENARIOS, OBS_DIM, REL_LANES, REWARD_DT, EnvConfig, RewardConfig,
)

INF = 1e6
STAT_NAMES = [
    "exited", "success", "missed", "collisions", "timeouts", "travel_time", "delay",
    "lane_changes", "aborts", "hard_brakes", "unsafe_steps", "veh_steps", "lc_events",
    "induced_decel", "jerk_sq", "spawned", "blocked_spawns", "violations",
]
S = {n: i for i, n in enumerate(STAT_NAMES)}


def scatter_slots(t, slot, val):
    """t [E, N] with t[e, slot[e, c]] = val[e, c] for every candidate c; slot == N means
    'no assignment' (written to a dummy column). Fixed shapes, no host sync."""
    ext = torch.cat([t, t[:, :1]], 1)
    ext = ext.scatter(1, slot, val.to(t.dtype) if torch.is_tensor(val) else torch.full_like(slot, val, dtype=t.dtype))
    return ext[:, :-1]


def decision_step(env, lat, lon):
    """Advance one decision interval: ``cfg.substeps`` physics steps of ``cfg.dt``.

    The lateral action applies on the first substep (later substeps continue an ongoing lane
    change); the longitudinal offset is held, while IDM and the safety shield are recomputed
    every physics step. Rewards are summed and terminations or-ed over the substeps for the
    vehicles that were agents at the start of the interval; a vehicle that spawns into a freed
    slot mid-interval starts acting at the next decision."""
    agent = env._agents()
    keep = torch.full_like(lat, LAT_KEEP)
    R = torch.zeros_like(env.v)
    D = torch.zeros_like(agent)
    outcome = None
    for k in range(env.cfg.substeps):
        if k > 0:
            env.observe(build_obs=False)
        lon_k = torch.where(agent, lon, LON_NEUTRAL)
        r, d = env._substep(lat if k == 0 else keep, lon_k)
        R = R + torch.where(agent, r, 0.0)
        D = D | (agent & d)
        agent = agent & ~d
        oc = env.last_outcome
        outcome = oc if outcome is None else {key: outcome[key] | oc[key] for key in oc}
    env.last_outcome = outcome
    return R, D


def ahead_of(x, slot):
    """[E, i, j]: vehicle j is ahead of vehicle i (ties broken by slot index, exactly)."""
    xi, xj = x[:, :, None], x[:, None, :]
    return (xj > xi) | ((xj == xi) & (slot[None, None, :] > slot[None, :, None]))


class LaneChangeEnv:
    def __init__(self, cfg: EnvConfig = EnvConfig(), rcfg: RewardConfig = RewardConfig(),
                 device="cuda", seed=0, fixed_aggr=None, fixed_info=None):
        self.cfg, self.rcfg, self.dev = cfg, rcfg, torch.device(device)
        self.E, self.N, self.K = cfg.num_envs, cfg.max_veh, cfg.max_lanes
        self.rng = np.random.default_rng(seed)
        self.gen = torch.Generator(device=self.dev).manual_seed(seed)
        self.fixed_aggr, self.fixed_info = fixed_aggr, fixed_info
        E, N, K, d = self.E, self.N, self.K, self.dev
        f = lambda *s: torch.zeros(*s, device=d)
        i = lambda *s: torch.zeros(*s, device=d, dtype=torch.long)
        b = lambda *s: torch.zeros(*s, device=d, dtype=torch.bool)
        # segment description
        self.scenario, self.ctrl = i(E), i(E)
        self.seg_len, self.limit = f(E), f(E)
        self.lane_start, self.lane_end, self.lane_mov = f(E, K), f(E, K), i(E, K)
        self.spawn_rate, self.mov_prob = f(E, K), f(E, K, NUM_MOV)
        self.cycle, self.g_thru, self.g_left, self.sig_offset = f(E), f(E), f(E), f(E)
        self.conflict_on, self.conflict_off, self.conflict_busy = f(E), f(E), b(E)
        self.pending = i(E, K)        # arrivals waiting at the entry (point queue)
        self.t_env = i(E)
        # vehicle state
        self.active, self.stopped_once = b(E, N), b(E, N)
        self.committed = b(E, N)      # entered the stop line's commitment zone while allowed to go
        self.x, self.v, self.a_prev = f(E, N), f(E, N), f(E, N)
        self.lane_from, self.lane_to, self.prog = i(E, N), i(E, N), f(E, N)
        self.mov, self.age = i(E, N), i(E, N)
        self.aggr, self.info, self.vdes, self.t_ff = f(E, N), f(E, N), f(E, N), f(E, N)
        # per-vehicle accumulators flushed into per-env stats on termination
        self.acc_lc, self.acc_abort, self.acc_hb, self.acc_unsafe = f(E, N), f(E, N), f(E, N), f(E, N)
        self.acc_jerk = f(E, N)
        # realised braking of the follower a vehicle cut in front of, tracked for 2 s
        self.cutin_f, self.cutin_t, self.cutin_min = i(E, N) - 1, i(E, N), f(E, N)
        self.stats = f(E, len(STAT_NAMES))
        self.slot = torch.arange(N, device=d)
        self.lanes = torch.arange(K, device=d)
        # constant lookup tables live on the device (no per-step host-to-device copies)
        self.k_ctrl_to_cat = torch.as_tensor(CTRL_TO_CAT, device=d)
        self.k_lon_deltas = torch.as_tensor(LON_DELTAS, dtype=torch.float32, device=d)
        self._build_pool()
        self.reset()

    def _build_pool(self):
        """Pre-generate ``pool_size`` segments of every scenario type and keep them on the GPU.
        Resets only index into this pool, so simulation and training never wait on the CPU."""
        P, cfg = self.cfg.pool_size, self.cfg
        segs = [scenarios.generate(sc, self.rng, self.K, cfg.demand_scale)
                for sc in range(NUM_SCENARIOS) for _ in range(P)]
        t = lambda key, dt=torch.float32: torch.as_tensor(np.stack([sg[key] for sg in segs]), dtype=dt, device=self.dev)
        self.pool = {"scenario": t("scenario", torch.long), "ctrl": t("ctrl", torch.long),
                     "lane_mov": t("lane_mov", torch.long), "limit": t("speed_limit")}
        for key in ("seg_len", "cycle", "g_thru", "g_left", "conflict_on", "conflict_off",
                    "lane_start", "lane_end", "spawn_rate", "mov_prob"):
            self.pool[key] = t(key)
        # the generator's conflicting-stream switching probabilities are per 0.5 s: convert to per dt
        f = cfg.dt / REWARD_DT
        for key in ("conflict_on", "conflict_off"):
            self.pool[key] = 1.0 - (1.0 - self.pool[key]).clamp(min=0.0) ** f
        self.allowed = torch.as_tensor(cfg.scenarios, device=self.dev)

    def _agents(self):
        return self.active.clone()

    # ------------------------------------------------------------------ resets
    def reset(self, scenario_ids=None):
        """Regenerate every segment. ``scenario_ids`` pins env e to scenario_ids[e]."""
        self.pinned = None if scenario_ids is None else torch.as_tensor(scenario_ids, device=self.dev)
        self._reset_envs(torch.ones(self.E, dtype=torch.bool, device=self.dev))
        # stagger episode boundaries so resets are spread over time
        if scenario_ids is None:
            self.t_env.copy_(torch.randint(0, self.cfg.episode_steps, (self.E,), device=self.dev,
                                           generator=self.gen))
        self.stats.zero_()

    def _reset_envs(self, mask):
        """Regenerate the envs in ``mask`` [E] from the GPU pool (masked writes, no host sync)."""
        E, P, d = self.E, self.cfg.pool_size, self.dev
        if self.pinned is not None:
            scen = self.pinned
        else:
            scen = self.allowed[torch.randint(0, len(self.allowed), (E,), device=d, generator=self.gen)]
        pidx = scen * P + torch.randint(0, P, (E,), device=d, generator=self.gen)
        for key, val in self.pool.items():
            new = val[pidx]
            m = mask.view(-1, *([1] * (new.dim() - 1)))
            setattr(self, key, torch.where(m, new, getattr(self, key)))
        self.sig_offset = torch.where(mask, torch.rand(E, device=d, generator=self.gen) * self.cycle, self.sig_offset)
        self.conflict_busy = self.conflict_busy & ~mask
        self.pending = torch.where(mask[:, None], 0, self.pending)
        self.active = self.active & ~mask[:, None]
        self.t_env = torch.where(mask, 0, self.t_env)

    # ---------------------------------------------------------- neighbour tables
    def _occupancy(self):
        k = self.lanes.view(1, 1, -1)
        return self.active[..., None] & ((self.lane_from[..., None] == k) | (self.lane_to[..., None] == k))

    def _center(self):
        return torch.where(self.prog >= 0.5, self.lane_to, self.lane_from)

    def _tables(self):
        """Leader / follower / density tables for every vehicle and every lane: [E, N, K]."""
        cfg = self.cfg
        occ = self._occupancy()
        dx = self.x[:, None, :] - self.x[:, :, None]           # [E, i, j] = x_j - x_i
        ahead = ahead_of(self.x, self.slot)
        occ_j = occ[:, None, :, :]                             # [E, 1, j, K]
        lead_c = torch.where(ahead[..., None] & occ_j, dx[..., None], INF)
        lead_dx, lead_idx = lead_c.min(dim=2)
        lag_c = torch.where((~ahead)[..., None] & occ_j & (self.slot[:, None] != self.slot[None, :])[None, ..., None],
                            -dx[..., None], INF)
        lag_dx, lag_idx = lag_c.min(dim=2)
        T = {"occ": occ, "lead_idx": lead_idx, "lag_idx": lag_idx,
             "lead_gap": torch.where(lead_dx < INF, lead_dx - cfg.veh_len, INF),
             "lag_gap": torch.where(lag_dx < INF, lag_dx - cfg.veh_len, INF)}
        # density / queue ahead within the vehicle's information range
        rng_i = self.info[:, :, None, None]
        within = ahead[..., None] & occ_j & (dx[..., None] < rng_i)
        T["count"] = within.sum(2).float()
        T["speed_sum"] = (within * self.v[:, None, :, None]).sum(2)
        # lost-time rate of the queue each vehicle holds up behind it (slow vehicles within 60 m)
        loss_j = (0.5 + self.aggr) * (1.0 - self.v / self.vdes.clamp(min=1.0)).clamp(min=0.0)
        queued = (~ahead)[..., None] & occ_j & (self.slot[:, None] != self.slot[None, :])[None, ..., None]             & (dx[..., None] > -60.0) & (self.v[:, None, :, None] < 3.0)
        T["queue_cost"] = (queued * loss_j[:, None, :, None]).sum(2)
        # best gap visible in each lane: gaps in front of vehicles j within range of i
        gap_j = torch.minimum(T["lead_gap"], self.info[..., None])                        # [E, j, K]
        near = (dx.abs()[..., None] < rng_i) & occ_j
        cand = torch.where(near, gap_j[:, None, :, :], -1.0)                             # [E, i, j, K]
        best, bj = cand.max(dim=2)
        pos = torch.gather(dx[..., None].expand(-1, -1, -1, self.K), 2, bj[:, :, None, :]).squeeze(2) \
            + torch.gather(gap_j, 1, bj) * 0.5
        none = best < 0
        T["best_gap"] = torch.where(none, self.info[..., None], best)
        T["best_pos"] = torch.where(none, 0.0, pos)
        return T

    # -------------------------------------------------------------- route logic
    def _lane_geometry(self):
        x = self.x[..., None]
        exists = (self.lane_start[:, None, :] <= x) & (x <= self.lane_end[:, None, :])
        reaches_end = self.lane_end >= self.seg_len[:, None] - 1.0                       # [E, K]
        ok = reaches_end[:, None, :] & (((self.lane_mov[:, None, :] >> self.mov[..., None]) & 1) == 1)
        # signed number of lanes from each lane k to the nearest OK lane
        diff = self.lanes.view(1, 1, 1, -1) - self.lanes.view(1, 1, -1, 1)               # [.., k, k']
        dist = torch.where(ok[:, :, None, :], diff, 99)
        pick = dist.abs().argmin(-1, keepdim=True)
        lanes_to_ok = torch.gather(dist, 3, pick).squeeze(-1)
        lanes_to_ok = torch.where(lanes_to_ok.abs() >= 99, 0, lanes_to_ok)
        return exists, ok, lanes_to_ok

    def _signal(self):
        """Per-vehicle: blocked at the stop line, and seconds until that changes."""
        tc = torch.remainder(self.t_env * self.cfg.dt + self.sig_offset, self.cycle)[:, None].expand(-1, self.N)
        g_thru, g_left, cyc = (z[:, None] for z in (self.g_thru, self.g_left, self.cycle))
        is_left = self.mov == MOV_LEFT
        green = torch.where(is_left, (tc >= g_thru) & (tc < g_thru + g_left), tc < g_thru)
        tts_left = torch.where(tc < g_thru, g_thru - tc,
                               torch.where(green, g_thru + g_left - tc, cyc - tc + g_thru))
        tts = torch.where(is_left, tts_left, torch.where(green, g_thru - tc, cyc - tc))
        ctrl = self.ctrl[:, None]
        busy = self.conflict_busy[:, None]
        blocked = torch.where(ctrl == CTRL_SIGNAL, ~green,
                  torch.where(ctrl == CTRL_STOP, ~self.stopped_once | busy,
                  torch.where(ctrl == CTRL_YIELD, busy, torch.zeros_like(green))))
        tts = torch.where(ctrl == CTRL_SIGNAL, tts, 0.0)
        return blocked, tts

    # ------------------------------------------------------------- observation
    def observe(self, build_obs=True):
        """Returns (obs [E,N,OBS_DIM], lat_mask [E,N,3], valid [E,N]) and caches tables.
        ``build_obs=False`` only refreshes the cached tables (physics substeps)."""
        cfg, Rn = self.cfg, self.cfg.obs_norm_range
        T = self._tables()
        exists, ok, lanes_to_ok = self._lane_geometry()
        blocked, tts = self._signal()
        c = self._center()
        g = lambda t, k: torch.gather(t, 2, k[..., None]).squeeze(-1)    # [E,N,K] at lane k
        gv = lambda t, j: torch.gather(t, 1, j)                            # [E,N] at slot j
        need = g(lanes_to_ok, c)
        lane_end_c = torch.gather(self.lane_end, 1, c)
        deadline = torch.minimum(lane_end_c, self.seg_len[:, None]) - self.x
        in_lc = self.lane_from != self.lane_to
        lc_dir = (self.lane_to - self.lane_from).sign()
        # what each vehicle signals to its neighbours (turn indicator)
        intent_dir = torch.where(in_lc, lc_dir, need.sign())
        intent_tgt = c + intent_dir
        urgency = torch.where(in_lc, 1.0, (1.0 - deadline / 300.0).clamp(0.1, 1.0)) * (intent_dir != 0)
        self._cache = dict(T=T, exists=exists, ok=ok, lanes_to_ok=lanes_to_ok, blocked=blocked,
                           c=c, need=need, deadline=deadline, in_lc=in_lc, lc_dir=lc_dir)
        if not build_obs:
            return None

        lim = self.limit[:, None]
        ego = [self.v / lim, self.vdes / lim, self.aggr, self.info / Rn, self.prog, lc_dir.float(),
               need.clamp(-3, 3) / 3.0, (deadline / 300.0).clamp(0, 1),
               ((self.seg_len[:, None] - self.x) / 600.0).clamp(0, 1)]
        cat = self.k_ctrl_to_cat[self.ctrl][:, None]
        ego += [(cat == k).float().expand(-1, self.N) for k in range(NUM_CAT)]
        # rank / 2: a single-segment stop line is either closed (0) or open with priority (1)
        ego += [(~blocked).float(), blocked.float(), (tts / 60.0).clamp(0, 1), self.stopped_once.float()]
        ego += [(self.mov == m).float() for m in range(NUM_MOV)]
        ego += [self.a_prev / 3.0]
        assert len(ego) == len(EGO_FEATURES)

        feats = []
        for d in REL_LANES:
            k = c + d
            valid = (k >= 0) & (k < self.K)
            kc = k.clamp(0, self.K - 1)
            ls, le = torch.gather(self.lane_start, 1, kc), torch.gather(self.lane_end, 1, kc)
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
                  torch.where(cnt > 0, g(T["speed_sum"], kc) / cnt.clamp(min=1), self.limit[:, None]) / lim,
                  g(T["best_gap"], kc).clamp(max=Rn) / 100.0,
                  g(T["best_pos"], kc).clamp(-Rn, Rn) / Rn]
            assert len(fl) == len(LANE_FEATURES)
            feats += [z * valid if z.dtype != torch.bool else z for z in fl]
        obs = torch.stack([z.float() for z in ego + feats], dim=-1)
        obs = obs * self.active[..., None]
        return obs, self._lat_mask(), self.active.clone()

    def _safe_to_enter(self, k):
        """Physical safety of moving into lane k (gaps in the cached tables). [E,N] bool."""
        cfg, T = self.cfg, self._cache["T"]
        kc = k.clamp(0, self.K - 1)
        g = lambda t: torch.gather(t, 2, kc[..., None]).squeeze(-1)
        lead_gap, lag_gap = g(T["lead_gap"]), g(T["lag_gap"])
        v_lead = torch.gather(self.v, 1, g(T["lead_idx"]))
        v_lag = torch.gather(self.v, 1, g(T["lag_idx"]))
        b = cfg.b_lc_safe
        ok_lead = (lead_gap >= INF / 2) | ((lead_gap >= cfg.s0) &
                   ((self.v ** 2 - v_lead ** 2) / (2 * b) <= lead_gap - cfg.s0))
        ok_lag = (lag_gap >= INF / 2) | ((lag_gap >= cfg.s0) &
                  ((v_lag ** 2 - self.v ** 2) / (2 * b) + v_lag * cfg.dt <= lag_gap - cfg.s0))
        le = torch.gather(self.lane_end, 1, kc)
        ls = torch.gather(self.lane_start, 1, kc)
        geo = (k >= 0) & (k < self.K) & (ls <= self.x) & (le - self.x > 15.0) \
            & (self.seg_len[:, None] - self.x > 10.0)
        return geo & ok_lead & ok_lag

    def _lat_mask(self):
        in_lc, lc_dir = self._cache["in_lc"], self._cache["lc_dir"]
        start_l = ~in_lc & self._safe_to_enter(self.lane_from - 1)
        start_r = ~in_lc & self._safe_to_enter(self.lane_from + 1)
        # during a change: continuing is always allowed; aborting only once continuing became unsafe
        abort_ok = in_lc & ~self._safe_to_enter(self.lane_to)
        left = start_l | (in_lc & (lc_dir < 0)) | (abort_ok & (lc_dir > 0))
        right = start_r | (in_lc & (lc_dir > 0)) | (abort_ok & (lc_dir < 0))
        mask = torch.stack([torch.ones_like(in_lc), left, right], dim=-1)
        return mask & self.active[..., None] | (~self.active[..., None] & (torch.arange(NUM_LAT, device=self.dev) == LAT_KEEP))

    # ------------------------------------------------------------------- dynamics
    def _idm(self, v, vdes, T_h, gap, dv):
        c = self.cfg
        s_star = c.s0 + (v * T_h + v * dv / (2 * math.sqrt(c.a_max * c.b_comf))).clamp(min=0)
        return c.a_max * (1 - (v / vdes.clamp(min=1.0)) ** 4 - (s_star / gap.clamp(min=0.1)) ** 2)

    def _krauss(self, v, gap, v_l, b=4.5):
        tau = self.cfg.dt
        return v_l + (gap - 0.5 - v_l * tau) / ((v + v_l) / (2 * b) + tau)

    def step(self, lat, lon):
        """Advance one decision interval. lat/lon: [E,N] long actions. Returns (reward, done) [E,N]."""
        return decision_step(self, lat, lon)

    def _substep(self, lat, lon):
        """One physics step of cfg.dt."""
        cfg, rc, C = self.cfg, self.rcfg, self._cache
        T = C["T"]
        act0 = self.active.clone()
        E, N = self.E, self.N
        dt = cfg.dt
        rate = dt / REWARD_DT                 # per-second reward terms are defined per 0.5 s
        a_before = self.a_prev

        # ---- lateral decision
        in_lc, lc_dir = C["in_lc"], C["lc_dir"]
        want = torch.where(lat == LAT_LEFT, -1, torch.where(lat == LAT_RIGHT, 1, 0))
        start = act0 & ~in_lc & (want != 0)
        abort = act0 & in_lc & (want == -lc_dir)
        new_to = torch.where(start, self.lane_from + want, self.lane_to)
        # two vehicles entering the same lane side by side in the same step cannot see
        # each other in the tables: only the one further ahead may go
        dx = self.x[:, None, :] - self.x[:, :, None]                        # [E, i, j]: j ahead of i
        v_i, v_j = self.v[:, :, None], self.v[:, None, :]
        need = cfg.veh_len + cfg.s0 + (v_i ** 2 - v_j ** 2).clamp(min=0) / (2 * cfg.b_lc_safe) + v_i * dt
        clash = start[:, :, None] & start[:, None, :] & (new_to[:, :, None] == new_to[:, None, :]) \
            & (dx > -1e-3) & (dx < need) & (self.slot[:, None] != self.slot[None, :])
        start = start & ~clash.any(-1)
        new_to = torch.where(start, self.lane_from + want, self.lane_to)
        # lag impact: decel imposed on the new follower in the target lane (MOBIL-style)
        k_new = new_to.clamp(0, self.K - 1)
        gk = lambda t: torch.gather(t, 2, k_new[..., None]).squeeze(-1)
        f_idx, f_gap = gk(T["lag_idx"]), gk(T["lag_gap"])
        has_f = start & (f_gap < INF / 2)
        v_f = torch.gather(self.v, 1, f_idx)
        # follower's own leader in the ego's target lane: table[e, f_idx, k_new]
        rows = lambda t: torch.gather(t, 1, f_idx[..., None].expand(-1, -1, self.K))
        f_lead_gap = torch.gather(rows(T["lead_gap"]), 2, k_new[..., None]).squeeze(-1)
        f_lead_v = torch.gather(self.v, 1, torch.gather(rows(T["lead_idx"]), 2, k_new[..., None]).squeeze(-1))
        vdes_f, T_f = torch.gather(self.vdes, 1, f_idx), torch.gather(self._headway(), 1, f_idx)
        a_f_before = torch.where(f_lead_gap < INF / 2, self._idm(v_f, vdes_f, T_f, f_lead_gap, v_f - f_lead_v),
                                 self._idm(v_f, vdes_f, T_f, torch.full_like(v_f, INF), torch.zeros_like(v_f)))
        a_f_after = self._idm(v_f, vdes_f, T_f, f_gap, v_f - self.v)
        impact = torch.where(has_f, (a_f_before - a_f_after).clamp(0, 6), 0.0)

        # apply: abort swaps origin/target, progress mirrors
        self.lane_to = new_to
        fr, to = self.lane_from.clone(), self.lane_to.clone()
        self.lane_from = torch.where(abort, to, fr)
        self.lane_to = torch.where(abort, fr, self.lane_to)
        self.prog = torch.where(abort, 1.0 - self.prog, self.prog)

        # ---- longitudinal: IDM over every lane occupied + obstacles, plus RL offset
        T_h = self._headway()
        a_idm = self._idm(self.v, self.vdes, T_h, torch.full_like(self.v, INF), torch.zeros_like(self.v))
        v_cap = torch.full_like(self.v, INF)
        lanes_occ = [self.lane_from, self.lane_to]
        for k in lanes_occ:
            gap = torch.gather(T["lead_gap"], 2, k.clamp(0, self.K - 1)[..., None]).squeeze(-1)
            vl = torch.gather(self.v, 1, torch.gather(T["lead_idx"], 2, k.clamp(0, self.K - 1)[..., None]).squeeze(-1))
            has = gap < INF / 2
            a_idm = torch.where(has, torch.minimum(a_idm, self._idm(self.v, self.vdes, T_h, gap, self.v - vl)), a_idm)
            v_cap = torch.where(has, torch.minimum(v_cap, self._krauss(self.v, gap, vl)), v_cap)
        # static obstacles: end of the centre lane, and the stop line when blocked
        c = self._center()
        lane_end_c = torch.gather(self.lane_end, 1, c)
        walls = [(lane_end_c < self.seg_len[:, None] - 1.0, lane_end_c - self.x)]
        stop_gap = self.seg_len[:, None] - self.x
        # amber logic: a vehicle allowed to go that can no longer stop comfortably (or is at
        # the line) is committed and may clear even if the signal / conflict state changes
        commit_zone = (stop_gap - cfg.s0 < self.v ** 2 / (2 * 3.5)) | (stop_gap < 10.0)
        self.committed |= self.active & ~C["blocked"] & commit_zone & (stop_gap > 0)
        walls.append((C["blocked"] & ~self.committed & (stop_gap > 0), stop_gap))
        zero = torch.zeros_like(self.v)
        for on, gap in walls:
            a_idm = torch.where(on, torch.minimum(a_idm, self._idm(self.v, self.vdes, T_h, gap, self.v)), a_idm)
            v_cap = torch.where(on, torch.minimum(v_cap, self._krauss(self.v, gap, zero)), v_cap)
        delta = self.k_lon_deltas[lon]
        a = (a_idm + delta).clamp(-cfg.b_max, cfg.a_max + 1.5)
        a = torch.minimum(a, (v_cap - self.v) / dt).clamp(min=-cfg.b_max)
        v_new = (self.v + a * dt).clamp(min=0.0)
        v_new = torch.minimum(v_new, self.limit[:, None] * 1.15)
        a = (v_new - self.v) / dt
        self.x = torch.where(act0, self.x + 0.5 * (self.v + v_new) * dt, self.x)
        jerk = torch.where(act0, (a - self.a_prev) / dt, 0.0)
        self.v, self.a_prev = torch.where(act0, v_new, self.v), torch.where(act0, a, self.a_prev)
        # follower reaction to cut-ins: running min of its acceleration over 2 s
        track = self.cutin_t > 0
        a_f_now = torch.gather(self.a_prev, 1, self.cutin_f.clamp(min=0))
        self.cutin_min = torch.where(track, torch.minimum(self.cutin_min, a_f_now), self.cutin_min)
        self.cutin_t = (self.cutin_t - 1).clamp(min=0)
        closed = track & (self.cutin_t == 0)
        self.stats[:, S["lc_events"]] += closed.sum(1)
        self.stats[:, S["induced_decel"]] += torch.where(closed, self.cutin_min.clamp(max=0), 0.0).sum(1)
        self.cutin_f = torch.where(has_f, f_idx, self.cutin_f)
        self.cutin_t = torch.where(has_f, round(2.0 / dt), self.cutin_t)
        self.cutin_min = torch.where(has_f, 0.0, self.cutin_min)

        # ---- lateral progress
        moving = act0 & (self.lane_from != self.lane_to)
        self.prog = torch.where(moving, self.prog + dt / cfg.lc_duration, 0.0)
        done_lc = moving & (self.prog >= 1.0 - 1e-4)
        self.lane_from = torch.where(done_lc, self.lane_to, self.lane_from)
        self.prog = torch.where(done_lc, 0.0, self.prog)

        # ---- stop-sign bookkeeping and conflicting stream for stop / yield
        near_line = (self.seg_len[:, None] - self.x) < 8.0
        self.stopped_once |= act0 & near_line & (self.v < 0.5)
        u = torch.rand(self.E, device=self.dev, generator=self.gen)
        self.conflict_busy = torch.where(self.conflict_busy, u > self.conflict_off, u < self.conflict_on)

        # ---- safety events
        lead_gap_c = torch.gather(T["lead_gap"], 2, c[..., None]).squeeze(-1)
        v_lead_c = torch.gather(self.v, 1, torch.gather(T["lead_idx"], 2, c[..., None]).squeeze(-1))
        closing = self.v - v_lead_c
        unsafe = act0 & (lead_gap_c < INF / 2) & (closing > 0.1) & (lead_gap_c / closing.clamp(min=0.1) < 1.5)
        hard = act0 & (a < -4.0)
        hard_onset = hard & (a_before >= -4.0)
        collided = act0 & self._collisions()

        # ---- terminations
        exited = act0 & ~collided & (self.x >= self.seg_len[:, None])
        # crossed the stop line on red / without the required stop / into conflicting traffic
        violation = exited & C["blocked"] & ~self.committed
        self.stats[:, S["violations"]] += violation.sum(1)
        c_new = self._center()
        lane_ok = ((torch.gather(self.lane_mov, 1, c_new) >> self.mov) & 1) == 1
        success, missed = exited & lane_ok, exited & ~lane_ok
        self.age = self.age + act0.long()
        timeout = act0 & ~exited & ~collided & (self.age >= cfg.max_age_steps)
        self.t_env += 1
        env_reset = self.t_env >= cfg.episode_steps
        truncated = act0 & env_reset[:, None] & ~exited & ~collided & ~timeout
        done = exited | collided | timeout | truncated
        self.last_outcome = dict(success=success, missed=missed, collided=collided, timeout=timeout)

        # ---- reward
        aggr = self.aggr
        polite = 0.2 + 0.8 * (1.0 - aggr)
        # lost-time penalty (never positive: a per-step bonus would reward lingering on the segment)
        r = -rc.efficiency * rate * (0.5 + aggr) * (1.0 - self.v / self.vdes.clamp(min=1.0)).clamp(min=0.0)
        r -= rc.comfort * rate * ((a / 3.0) ** 2 + (jerk / 10.0) ** 2)
        r -= rc.lc_cost * (1.5 - aggr) * start
        r -= rc.abort_cost * abort
        r -= rc.lag_impact * polite * impact
        # blocking: standing in the way with free road ahead costs the lost time of the queue behind
        c0 = C["c"]
        gap0 = torch.gather(T["lead_gap"], 2, c0[..., None]).squeeze(-1)
        unconstrained = (gap0 > 15.0 + 1.5 * self.v) & ((self.seg_len[:, None] - self.x) > 15.0)
        qcost = torch.gather(T["queue_cost"], 2, c0[..., None]).squeeze(-1)
        r -= rc.blocking * rate * polite * rc.efficiency * qcost * unconstrained
        r -= rate * (rc.unsafe_ttc * unsafe + rc.hard_brake * hard)
        r += rc.success * success + rc.missed_route * missed + rc.timeout * timeout + rc.collision * collided
        # a missed movement costs a detour, priced at the driver's own lost-time rate
        r = r - rc.efficiency * (0.5 + aggr) * (rc.detour_s / REWARD_DT) * missed
        r += rc.violation * violation
        r = torch.where(act0, r, 0.0)
        n_act = act0.sum(1, keepdim=True).clamp(min=1)
        team = r.sum(1, keepdim=True) / n_act
        r = r + rc.team_share * (1.0 - aggr) * team * act0

        # ---- stats
        self.acc_lc += start
        self.acc_abort += abort
        self.acc_hb += hard_onset
        self.acc_unsafe += unsafe
        self.acc_jerk += jerk ** 2 * act0
        st = self.stats
        st[:, S["veh_steps"]] += act0.sum(1)
        tt = self.age * dt
        for name, m in (("exited", exited), ("success", success), ("missed", missed),
                        ("collisions", collided), ("timeouts", timeout)):
            st[:, S[name]] += m.sum(1)
        fin = exited.float()
        st[:, S["travel_time"]] += (tt * fin).sum(1)
        st[:, S["delay"]] += ((tt - self.t_ff).clamp(min=0) * fin).sum(1)
        for name, acc in (("lane_changes", self.acc_lc), ("aborts", self.acc_abort),
                          ("hard_brakes", self.acc_hb), ("unsafe_steps", self.acc_unsafe), ("jerk_sq", self.acc_jerk)):
            st[:, S[name]] += (acc * done).sum(1)

        # ---- remove finished vehicles, reset finished envs, spawn new vehicles
        self.active &= ~done
        self._reset_envs(env_reset)
        self._spawn()
        return r, done

    def _headway(self):
        return self.cfg.T * (1.3 - 0.6 * self.aggr)

    def _collisions(self):
        occ = self._occupancy().float()
        shared = torch.einsum("enk,emk->enm", occ, occ) > 0
        dx = self.x[:, None, :] - self.x[:, :, None]
        ahead = ahead_of(self.x, self.slot)
        return (shared & ahead & (dx < self.cfg.veh_len - 0.5)).any(-1)

    # ------------------------------------------------------------------- demand
    def _spawn(self):
        """Entry queues for every lane at once: at most one vehicle per lane per step, placed
        into free slots with masked scatters (fixed shapes, no host sync)."""
        cfg, E, N, K, d = self.cfg, self.E, self.N, self.K, self.dev
        occ = self._occupancy()                                                   # [E,N,K]
        xs = self.lane_start.clamp(min=0.0)                                       # [E,K] entry point
        xo = torch.where(occ, self.x[..., None], INF)
        arrive = (self.spawn_rate > 0) & (torch.rand(E, K, device=d, generator=self.gen) < self.spawn_rate * cfg.dt)
        self.pending = (self.pending + arrive.long()).clamp(max=10)
        ahead = xo >= xs[:, None, :]
        d_ahead, lead = torch.where(ahead, xo - xs[:, None, :], INF).min(1)       # [E,K]
        v_lead = torch.gather(self.v, 1, lead)
        lim = self.limit[:, None]
        v0 = torch.where(xs > 0, 0.7 * lim, 0.9 * lim)
        # enter no faster than allows a comfortable stop behind the vehicle ahead
        room = (d_ahead - cfg.veh_len - cfg.s0 - v0 * cfg.dt).clamp(min=0)
        v0 = torch.where(d_ahead < INF / 2, torch.minimum(v0, torch.sqrt(v_lead ** 2 + 2 * cfg.b_comf * room)), v0)
        clear = (d_ahead >= INF / 2) | (d_ahead - cfg.veh_len - cfg.s0 >= (0.8 * v0).clamp(min=2.0))
        # vehicles that moved into a mid-segment lane (ramp) just behind the entry point
        behind = (xo < xs[:, None, :]) & (xo < INF / 2)
        d_behind = torch.where(behind, xs[:, None, :] - xo, INF)
        v_b = torch.where(behind, self.v[..., None], 0.0)
        need_b = cfg.veh_len + cfg.s0 + (v_b ** 2 - v0[:, None, :] ** 2).clamp(min=0) / (2 * cfg.b_comf) + v_b * cfg.dt
        clear &= ~(d_behind < need_b).any(1)
        go = (self.pending > 0) & clear
        free = ~self.active
        rank = go.long().cumsum(1) - 1
        go &= rank < free.sum(1, keepdim=True)
        self.stats[:, S["blocked_spawns"]] += (arrive & ~go).sum(1)
        self.stats[:, S["spawned"]] += go.sum(1)
        self.pending = self.pending - go.long()
        free_sorted = torch.argsort((~free).to(torch.int8), dim=1, stable=True)  # free slots first
        slot = torch.where(go, torch.gather(free_sorted, 1, rank.clamp(min=0)), N)   # [E,K]; N = none
        # attributes drawn for every lane (dense), written only where a vehicle enters
        r = lambda: torch.rand(E, K, device=d, generator=self.gen)
        aggr = r() if self.fixed_aggr is None else torch.full((E, K), self.fixed_aggr, device=d)
        lo, hi = cfg.info_range
        info = lo + (hi - lo) * r() if self.fixed_info is None else torch.full((E, K), self.fixed_info, device=d)
        truck = r() < cfg.truck_share
        vdes = lim * torch.where(truck, 0.55 + 0.2 * r(), (0.9 + 0.2 * r()) * (0.95 + 0.1 * aggr))
        probs = self.mov_prob.reshape(-1, NUM_MOV) + 1e-6 * (torch.arange(NUM_MOV, device=d) == MOV_THROUGH)
        mov = torch.multinomial(probs, 1, generator=self.gen).view(E, K)
        lane = self.lanes.view(1, -1).expand(E, -1)
        put = lambda name, val: setattr(self, name, scatter_slots(getattr(self, name), slot, val))
        put("active", True)
        put("x", xs)
        put("v", v0)
        put("a_prev", 0.0)
        put("lane_from", lane)
        put("lane_to", lane)
        put("prog", 0.0)
        put("mov", mov)
        put("age", 0)
        put("aggr", aggr)
        put("info", info)
        put("vdes", vdes)
        put("stopped_once", False)
        put("committed", False)
        put("t_ff", (self.seg_len[:, None] - xs) / torch.minimum(vdes, lim))
        for name in ("acc_lc", "acc_abort", "acc_hb", "acc_unsafe", "acc_jerk"):
            put(name, 0.0)

    # ---------------------------------------------------------------- baseline
    def rule_action(self, obs, lat_mask):
        """LPSim-style rule baseline: mandatory change toward the route lane as soon as
        the shield allows; discretionary change when blocked by a slower leader and the
        adjacent lane offers a longer gap; pure IDM longitudinally."""
        from .config import obs_index as oi
        need = obs[..., oi("need")]
        lead0, dv0 = obs[..., oi("lead_gap", 0)], obs[..., oi("lead_dv", 0)]
        lat = torch.full(need.shape, LAT_KEEP, dtype=torch.long, device=self.dev)
        slow = (lead0 < 0.25) & (dv0 < -0.2)
        for d, a in ((-1, LAT_LEFT), (1, LAT_RIGHT)):
            disc = slow & (obs[..., oi("lanes_to_ok", d)] == 0) & (obs[..., oi("exists", d)] > 0) \
                & (obs[..., oi("lead_gap", d)] > lead0 + 0.1)
            lat = torch.where((lat == LAT_KEEP) & disc & lat_mask[..., a], a, lat)
        lat = torch.where((need < -0.1) & lat_mask[..., LAT_LEFT], LAT_LEFT, lat)
        lat = torch.where((need > 0.1) & lat_mask[..., LAT_RIGHT], LAT_RIGHT, lat)
        lat = torch.where(obs[..., oi("lc_progress")] > 0, LAT_KEEP, lat)
        from .config import LON_NEUTRAL
        return lat, torch.full_like(lat, LON_NEUTRAL)
