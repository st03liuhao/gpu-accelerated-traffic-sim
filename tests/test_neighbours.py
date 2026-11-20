"""The sorted-list neighbour kernel (lcrl/neighbours.py) must give the dense neighbour
search's tables: identical indices, gaps and counts; float sums equal up to rounding."""
import pytest
import torch

from lcrl import neighbours
from lcrl.config import EnvConfig, IxConfig
from lcrl.env import LaneChangeEnv
from lcrl.intersection import signal_4leg
from lcrl.ix_env import IntersectionEnv

pytestmark = pytest.mark.skipif(not (torch.cuda.is_available() and neighbours.enabled()),
                                reason="needs CUDA and the compiled neighbour kernel")

SUMS = ("speed_sum", "queue_cost")


def _envs(kind):
    if kind == "freeway":
        env = LaneChangeEnv(EnvConfig(num_envs=64), device="cuda", seed=5)
        env.reset(scenario_ids=torch.zeros(64, dtype=torch.long))          # on-ramp
    else:
        env = IntersectionEnv(IxConfig(num_envs=32, max_veh=224), device="cuda", seed=5,
                              designs=[signal_4leg(None, demand_vph=500.0)])
        env.reset(design_ids=torch.zeros(32, dtype=torch.long))
    return env


@pytest.mark.parametrize("kind", ["freeway", "intersection"])
def test_kernel_matches_dense(kind):
    env = _envs(kind)
    checked = 0
    for t in range(300):                      # 150 s: queues, lane changes, box vehicles
        obs, mask, _ = env.observe()
        env.step(*env.rule_action(obs, mask))
        if t % 30 == 29:
            A, B = env._tables_dense(), env._tables()
            assert A.keys() == B.keys()
            for key in A:
                if key in SUMS:
                    assert torch.allclose(A[key], B[key], rtol=1e-5, atol=1e-4), key
                else:
                    assert torch.equal(A[key], B[key]), f"{key} differs at decision {t}"
            checked += 1
    assert checked == 10 and env.active.sum() > 20 * env.E
