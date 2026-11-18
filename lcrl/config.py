"""Shared constants and configuration for the RL lane-changing model.

Conventions follow LPSim (src/simulator/cuda_traffic_sim.cu):
  * lane 0 is the leftmost lane; "left" means lane - 1, "right" means lane + 1;
  * IDM car following, s_0 ~= 1.38 m;
  * physics step ``dt`` = 0.1 s; the policy decides every ``decision_dt`` = 0.5 s (the env
    runs ``substeps`` physics steps per decision, all on the GPU).
"""
from dataclasses import dataclass, field

# ---- scenarios -------------------------------------------------------------
FWY_ONRAMP, FWY_OFFRAMP, FWY_LANE_DROP, FWY_LANE_ADD = 0, 1, 2, 3
SIG_DEDICATED, SIG_SHARED, STOP_SIGN, YIELD_SIGN = 4, 5, 6, 7
SCENARIO_NAMES = [
    "freeway_onramp", "freeway_offramp", "freeway_lane_drop", "freeway_lane_add",
    "signal_dedicated_turn", "signal_shared_turn", "stop_sign", "yield_sign",
]
NUM_SCENARIOS = len(SCENARIO_NAMES)

# ---- downstream control at the end of the segment ---------------------------
# (single-segment env, lcrl/env.py; full intersections use lcrl/intersection.py)
CTRL_FREE, CTRL_SIGNAL, CTRL_STOP, CTRL_YIELD = 0, 1, 2, 3
NUM_CTRL = 4
# control category the policy observes (shared by both envs)
CAT_FREE, CAT_SIGNAL, CAT_AWSC, CAT_TWSC_MAJOR, CAT_TWSC_MINOR, CAT_YIELD_MINOR = range(6)
NUM_CAT = 6
CTRL_TO_CAT = (CAT_FREE, CAT_SIGNAL, CAT_TWSC_MINOR, CAT_YIELD_MINOR)   # single-segment ctrl -> category

# ---- movements at the end of the segment (bit i of a lane mask) -----------
MOV_LEFT, MOV_THROUGH, MOV_RIGHT = 0, 1, 2   # MOV_RIGHT doubles as "exit ramp"
NUM_MOV = 3

# ---- actions ---------------------------------------------------------------
# Lateral: 0 keep (continue an ongoing change), 1 go/steer left, 2 go/steer right.
# Steering opposite to an ongoing change aborts it (vehicle returns to its origin lane).
LAT_KEEP, LAT_LEFT, LAT_RIGHT = 0, 1, 2
NUM_LAT = 3
# Longitudinal: acceleration offset (m/s^2) added to the IDM acceleration, then
# clipped by the safety shield. Lets the agent open/close gaps on purpose.
LON_DELTAS = (-3.0, -1.5, 0.0, 0.75, 1.5)
NUM_LON = len(LON_DELTAS)
LON_NEUTRAL = 2

# ---- observation layout ----------------------------------------------------
EGO_FEATURES = [
    "v_rel_limit", "vdes_rel_limit", "aggr", "info_range", "lc_progress", "lc_dir",
    "need", "deadline", "dist_to_end",
    "cat_free", "cat_signal", "cat_all_way_stop", "cat_two_way_major", "cat_two_way_minor", "cat_yield_minor",
    "rank", "blocked", "time_to_switch", "stopped_once",
    "mov_left", "mov_through", "mov_right", "prev_accel",
]
LANE_FEATURES = [
    "exists", "runway", "starts_ahead", "route_ok", "lanes_to_ok",
    "lead_gap", "lead_dv", "lead_intent",
    "lag_gap", "lag_dv", "lag_intent",
    "count_ahead", "speed_ahead", "best_gap", "best_gap_pos",
]
REL_LANES = (-1, 0, 1)   # left, current, right
OBS_DIM = len(EGO_FEATURES) + len(REL_LANES) * len(LANE_FEATURES)


def obs_index(name, rel_lane=None):
    """Index of a named feature in the flat observation vector."""
    if rel_lane is None:
        return EGO_FEATURES.index(name)
    return (len(EGO_FEATURES) + REL_LANES.index(rel_lane) * len(LANE_FEATURES)
            + LANE_FEATURES.index(name))


@dataclass
class EnvConfig:
    num_envs: int = 1024          # parallel road segments
    max_veh: int = 64             # vehicle slots per segment
    max_lanes: int = 6
    dt: float = 0.1               # physics step (s)
    decision_dt: float = 0.5      # policy / rule decision interval (s); a multiple of dt
    veh_len: float = 5.0
    s0: float = 1.38              # IDM jam distance (LPSim default)
    T: float = 1.2                # IDM time headway (scaled by aggressiveness)
    a_max: float = 1.5            # IDM max acceleration
    b_comf: float = 2.0           # IDM comfortable deceleration
    b_max: float = 6.0            # physical braking capability used by the shield
    b_lc_safe: float = 4.0        # max decel a lane change may force on anyone
    lc_duration: float = 3.0      # seconds for a full lateral move
    max_age_s: float = 200.0      # vehicle timeout (deadlock guard)
    episode_s: float = 300.0      # env re-generated after this long
    obs_norm_range: float = 200.0 # normaliser for distances
    demand_scale: float = 1.0
    truck_share: float = 0.12     # slow vehicles that trigger bypass manoeuvres
    # freeway scenarios only: the single-leg intersection scenarios (4-7) are superseded by
    # the full intersections in lcrl/ix_env.py but remain available for comparison
    scenarios: tuple = (FWY_ONRAMP, FWY_OFFRAMP, FWY_LANE_DROP, FWY_LANE_ADD)
    # info level (gap search range) and aggressiveness sampled per vehicle
    info_range: tuple = (50.0, 200.0)
    aggr_range: tuple = (0.0, 1.0)
    pool_size: int = 256          # pre-generated freeway segments per scenario (kept on the GPU)

    @property
    def substeps(self):
        n = round(self.decision_dt / self.dt)
        assert n >= 1 and abs(n * self.dt - self.decision_dt) < 1e-6, "decision_dt must be a multiple of dt"
        return n

    @property
    def episode_steps(self):
        """Episode length in physics steps."""
        return round(self.episode_s / self.dt)

    @property
    def max_age_steps(self):
        return round(self.max_age_s / self.dt)


@dataclass
class IxConfig(EnvConfig):
    """Full-intersection environment (lcrl/ix_env.py)."""
    num_envs: int = 256
    max_veh: int = 160                     # 600 m legs of the 4-leg signal study design
    max_lanes: int = 5
    episode_s: float = 400.0
    pool_per_kind: int = 64
    kinds: tuple = (0, 1, 2, 3, 4, 5)      # indices into lcrl.intersection.KINDS
    ped_share: float = 0.5                 # share of designs with pedestrian demand


REWARD_DT = 0.5   # per-step reward rates below are per REWARD_DT seconds (scaled by dt / REWARD_DT)


@dataclass
class RewardConfig:
    efficiency: float = 0.05      # lost-time penalty per 0.5 s, times (1 - v / v_des), scaled by aggressiveness
    comfort: float = 0.01
    lc_cost: float = 0.04         # per lane change initiated (cheaper when aggressive)
    abort_cost: float = 0.04
    lag_impact: float = 0.10      # per m/s^2 of decel forced on the new follower
    unsafe_ttc: float = 0.05      # per 0.5 s with TTC < 1.5 s
    hard_brake: float = 0.10      # per 0.5 s with a < -4 m/s^2
    success: float = 1.0
    missed_route: float = -1.0       # flat part of the missed-movement penalty
    detour_s: float = 90.0          # plus the lost-time cost of a detour this long (rerouting after a missed turn)
    timeout: float = -1.0
    collision: float = -5.0
    violation: float = -1.0       # crossing the stop line on red / without stopping / without yielding
    blocking: float = 1.0         # x lost-time rate of the queue a vehicle holds up while the road ahead is free
    team_share: float = 0.5       # cooperative share, multiplied by politeness = 1 - aggr


@dataclass
class TrainConfig:
    iterations: int = 400
    rollout_steps: int = 32
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    epochs: int = 4
    minibatch: int = 65536
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden: int = 64
    seed: int = 0
    env: EnvConfig = field(default_factory=lambda: EnvConfig(num_envs=512))
    ix: IxConfig = field(default_factory=IxConfig)
    use_ix: bool = True
    reward: RewardConfig = field(default_factory=RewardConfig)
