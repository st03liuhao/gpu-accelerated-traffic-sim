"""The CUDA-graph runner (lcrl/graph.py) must reproduce the eager simulation exactly, fused
rule kernels (lcrl/fuse.py) must match the unfused rules, and a decision step must never
synchronise with the host."""
import pytest
import torch

from lcrl import fuse
from lcrl.config import EnvConfig, IxConfig
from lcrl.env import S, LaneChangeEnv
from lcrl.graph import GraphedEnv
from lcrl.ix_env import IntersectionEnv

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
needs_fuse = pytest.mark.skipif(not fuse.available(), reason="needs Triton for torch.compile")

STATE = ("active", "x", "v", "lane_from", "lane_to", "prog", "mov", "aggr")


def _mk(kind):
    if kind == "freeway":
        return LaneChangeEnv(EnvConfig(num_envs=32, episode_s=40.0), device="cuda", seed=11)
    return IntersectionEnv(IxConfig(num_envs=24, pool_per_kind=4, episode_s=40.0), device="cuda", seed=12)


def _pair(kind, fused):
    eager = _mk(kind)
    if fused:
        fuse.fuse_rules(eager)
    return eager, GraphedEnv(_mk(kind), fuse=fused)


@pytest.mark.parametrize("fused", [False, pytest.param(True, marks=needs_fuse)])
@pytest.mark.parametrize("kind", ["freeway", "intersection"])
def test_graph_matches_eager(kind, fused):
    eager, graphed = _pair(kind, fused)
    for t in range(120):                      # 60 s: includes episode resets (40 s) and spawns
        outs = []
        for env in (eager, graphed):
            obs, mask, _ = env.observe()
            lat, lon = env.rule_action(obs, mask)
            outs.append((obs, lat, lon))
        assert torch.equal(outs[0][0], outs[1][0]), f"observations differ at decision {t}"
        assert torch.equal(outs[0][1], outs[1][1]) and torch.equal(outs[0][2], outs[1][2])
        r0, d0 = eager.step(outs[0][1], outs[0][2])
        r1, d1 = graphed.step(outs[1][1], outs[1][2])
        assert torch.equal(d0, d1) and torch.allclose(r0, r1)
        for name in STATE:
            assert torch.equal(getattr(eager, name), getattr(graphed.env, name)), f"{name} differs at decision {t}"
    assert torch.equal(eager.stats, graphed.env.stats)
    assert eager.stats[:, S["exited"]].sum() > 0


@needs_fuse
@pytest.mark.parametrize("kind", ["freeway", "intersection"])
def test_fused_rules_match_unfused(kind):
    """Every fused rule function gives the unfused result on the same state (sums may differ
    in the last bits because of the order of additions)."""
    env = _mk(kind)
    for _ in range(80):                       # 40 s of traffic
        obs, mask, _ = env.observe()
        env.step(*env.rule_action(obs, mask))
    plain = {name: getattr(env, name) for name in fuse.FUSED[type(env).__name__]}
    env.observe()
    ref = dict(env._cache)
    fuse.fuse_rules(env)
    env.observe()
    for key, a in ref.items():
        b = env._cache[key]
        _assert_close(a, b, key)
    a = plain["_collisions"]()
    _assert_close(a, env._collisions(), "_collisions")


def _assert_close(a, b, key):
    if isinstance(a, dict):
        for k in a:
            _assert_close(a[k], b[k], f"{key}.{k}")
    elif isinstance(a, (tuple, list)):
        for i, (x, y) in enumerate(zip(a, b)):
            _assert_close(x, y, f"{key}[{i}]")
    elif torch.is_tensor(a):
        if a.is_floating_point():
            assert torch.allclose(a, b, rtol=1e-5, atol=1e-4), key
        else:
            assert torch.equal(a, b), key


@pytest.mark.parametrize("kind", ["freeway", "intersection"])
def test_decision_step_has_no_host_sync(kind):
    env = _mk(kind)
    if fuse.available():
        fuse.fuse_rules(env)                  # the fused kernels must not synchronise either
    for _ in range(5):
        obs, mask, _ = env.observe()
        env.step(*env.rule_action(obs, mask))
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        for _ in range(90):                   # crosses an episode reset
            obs, mask, _ = env.observe()
            env.step(*env.rule_action(obs, mask))
    finally:
        torch.cuda.set_sync_debug_mode("default")
