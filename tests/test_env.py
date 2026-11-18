import pytest
import torch

from lcrl.config import (LON_DELTAS, FWY_LANE_ADD, FWY_OFFRAMP, LAT_KEEP, LAT_LEFT, LAT_RIGHT, LON_NEUTRAL,
                         MOV_RIGHT, MOV_THROUGH, NUM_SCENARIOS, OBS_DIM, EnvConfig)
from lcrl.env import S, LaneChangeEnv

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def make_env(scenario, n_envs=1, **kw):
    cfg = EnvConfig(num_envs=n_envs, max_veh=16, episode_s=1e5, **kw)
    env = LaneChangeEnv(cfg, device="cuda", seed=1)
    env.reset(scenario_ids=[scenario] * n_envs)
    env.spawn_rate.zero_()                      # place vehicles by hand
    return env


def place(env, slot, x, lane, v, mov=MOV_THROUGH, e=0):
    env.active[e, slot] = True
    env.x[e, slot], env.v[e, slot] = x, v
    env.lane_from[e, slot] = env.lane_to[e, slot] = lane
    env.prog[e, slot] = 0.0
    env.mov[e, slot], env.age[e, slot] = mov, 0
    env.aggr[e, slot], env.info[e, slot], env.vdes[e, slot] = 0.5, 200.0, v
    env.t_ff[e, slot] = 1.0


def step(env, lat_by_slot=None):
    obs, mask, valid = env.observe()
    lat = torch.full((env.E, env.N), LAT_KEEP, device="cuda")
    for s, a in (lat_by_slot or {}).items():
        lat[0, s] = a
    lon = torch.full_like(lat, LON_NEUTRAL)
    return obs, mask, env.step(lat, lon)


def test_lane_change_takes_lc_duration_and_occupies_both_lanes():
    env = make_env(FWY_LANE_ADD)
    place(env, 0, 50.0, 1, 25.0)
    step(env, {0: LAT_LEFT})
    assert env.lane_from[0, 0] == 1 and env.lane_to[0, 0] == 0
    occ = env._occupancy()[0, 0]
    assert occ[0] and occ[1]
    n_steps = round(env.cfg.lc_duration / env.cfg.decision_dt)
    for _ in range(n_steps - 1):
        step(env)
    assert env.lane_from[0, 0] == 0 and env.lane_to[0, 0] == 0 and env.prog[0, 0] == 0


def test_abort_returns_to_origin_lane():
    env = make_env(FWY_LANE_ADD)
    place(env, 0, 50.0, 1, 25.0)
    step(env, {0: LAT_LEFT})
    step(env)
    step(env, {0: LAT_RIGHT})                   # steer back: abort
    assert env.lane_to[0, 0] == 1
    for _ in range(10):
        step(env)
    assert env.lane_from[0, 0] == 1 and env.lane_to[0, 0] == 1
    assert env.acc_abort[0, 0] == 1


def test_shield_masks_change_into_occupied_gap():
    env = make_env(FWY_LANE_ADD)
    place(env, 0, 100.0, 1, 25.0)
    place(env, 1, 98.0, 0, 25.0)               # alongside in the left lane
    obs, mask, _ = step(env)
    assert not mask[0, 0, LAT_LEFT]
    assert mask[0, 0, LAT_KEEP]


def test_shield_masks_change_in_front_of_fast_follower():
    env = make_env(FWY_LANE_ADD)
    place(env, 0, 100.0, 1, 15.0)
    place(env, 1, 80.0, 0, 33.0)               # 15 m behind, 18 m/s faster
    _, mask, _ = step(env)
    assert not mask[0, 0, LAT_LEFT]


def test_missed_exit_is_counted():
    env = make_env(FWY_OFFRAMP)
    seg = env.seg_len[0].item()
    place(env, 0, seg - 5.0, 0, 20.0, mov=MOV_RIGHT)   # wants the exit, stuck in lane 0
    step(env)
    assert env.stats[0, S["missed"]] == 1 and env.stats[0, S["success"]] == 0


def test_rule_baseline_all_scenarios_collision_free():
    cfg = EnvConfig(num_envs=8 * NUM_SCENARIOS, episode_s=1e5)
    env = LaneChangeEnv(cfg, device="cuda", seed=3)
    env.reset(scenario_ids=torch.arange(NUM_SCENARIOS).repeat_interleave(8))
    for _ in range(300):
        obs, mask, valid = env.observe()
        assert obs.shape[-1] == OBS_DIM and torch.isfinite(obs).all()
        assert mask[..., LAT_KEEP].all()
        env.step(*env.rule_action(obs, mask))
    assert env.stats[:, S["exited"]].sum() > 500
    assert env.stats[:, S["collisions"]].sum() == 0
    assert env.stats[:, S["violations"]].sum() <= 0.001 * env.stats[:, S["exited"]].sum()


def test_red_light_cannot_be_run_by_accelerating():
    from lcrl.config import SIG_SHARED
    env = make_env(SIG_SHARED)
    env.sig_offset[0] = env.g_thru[0] + env.g_left[0] + 1.0     # all-red for movement THROUGH
    seg = env.seg_len[0].item()
    place(env, 0, seg - 80.0, 1, 12.0)
    for _ in range(40):
        obs, mask, valid = env.observe()
        lat = torch.full((env.E, env.N), LAT_KEEP, device="cuda")
        env.step(lat, torch.full_like(lat, len(LON_DELTAS) - 1))   # full throttle
        if env._signal()[0][0, 0] == 0:
            break
    assert env.stats[0, S["violations"]] == 0
