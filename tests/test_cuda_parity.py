"""The exported CUDA policy must choose the same actions as the PyTorch policy.

Uses runs/*/policy.pt when LCRL_POLICY is set, otherwise a randomly initialised network.
Requires nvcc and MSVC (see cuda/build_test.bat)."""
import os
import shutil
import subprocess

import numpy as np
import pytest
import torch

from lcrl.config import EnvConfig, NUM_LAT
from lcrl.env import LaneChangeEnv
from lcrl.export_cuda import export
from lcrl.policy import ActorCritic

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.mark.skipif(shutil.which("nvcc") is None or not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_matches_torch(tmp_path):
    if os.environ.get("LCRL_POLICY"):
        from lcrl.train import load
        net = load(os.environ["LCRL_POLICY"], "cpu")
    else:
        torch.manual_seed(0)
        net = ActorCritic()
        torch.nn.init.normal_(net.actor[-1].weight, std=0.5)
    # export into a scratch copy of cuda/ so the repo's weights header is untouched
    cdir = tmp_path / "cuda"
    shutil.copytree(os.path.join(ROOT, "cuda"), cdir, ignore=shutil.ignore_patterns("*.exe", "*.h"))
    export(net, str(cdir / "lc_policy_weights.h"))
    exe = str(tmp_path / "test_lc_policy.exe")
    subprocess.run(["cmd", "/c", str(cdir / "build_test.bat"), exe], check=True)

    env = LaneChangeEnv(EnvConfig(num_envs=256), device="cuda", seed=7)
    for _ in range(150):
        obs, mask, valid = env.observe()
        env.step(*env.rule_action(obs, mask))
    obs, mask, valid = env.observe()
    o, m = obs[valid].cpu(), mask[valid].cpu()
    n = o.shape[0]
    o.numpy().astype(np.float32).tofile(tmp_path / "obs.bin")
    m.numpy().astype(np.uint8).tofile(tmp_path / "mask.bin")
    subprocess.run([exe, str(tmp_path / "obs.bin"), str(tmp_path / "mask.bin"), str(tmp_path / "out.bin"), str(n)],
                   check=True)
    got = np.fromfile(tmp_path / "out.bin", dtype=np.int32).reshape(n, 2)
    lat, lon, _ = net.act(o, m, deterministic=True)
    # only near-ties (float rounding between cuBLAS/CPU and the CUDA loop) may differ
    logits = net.actor(o)
    lat_l = logits[:, :NUM_LAT].masked_fill(~m, -1e9).sort(-1, descending=True).values
    lon_l = logits[:, NUM_LAT:].sort(-1, descending=True).values
    tie = ((lat_l[:, 0] - lat_l[:, 1]) < 1e-4) | ((lon_l[:, 0] - lon_l[:, 1]) < 1e-4)
    mismatch = (torch.from_numpy(got[:, 0]) != lat) | (torch.from_numpy(got[:, 1]) != lon)
    assert n > 1000
    assert int((mismatch & ~tie).sum()) == 0, f"{int(mismatch.sum())} of {n} differ"
