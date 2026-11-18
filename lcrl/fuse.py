"""Kernel fusion for the batched envs with ``torch.compile``.

The rule functions build large intermediate tensors. The intersection neighbour search, for
example, compares every vehicle with every other vehicle in every lane: [E, N, N, K] masks
that eager PyTorch writes to GPU memory and reads back, one kernel per operation.
``torch.compile`` (TorchInductor, Triton kernels) fuses each function into a few kernels
that keep those intermediates in registers. The rules themselves are unchanged.

Results match eager PyTorch except for the order of floating-point additions in sums
(differences of about 1e-6). A fused env is deterministic, so a fused env under CUDA
graphs still reproduces the fused eager env exactly.

Fusion needs Triton (``pip install "triton-windows>=3.7,<3.8"`` on Windows; bundled with
PyTorch on Linux). Without it, or with ``LCRL_FUSE=0``, ``fuse_rules`` leaves the env as is.
The first call compiles (about 30 s for the freeway env, 70 s for the intersection env);
TorchInductor caches the kernels on disk, so later runs start faster.
"""
import os
import warnings

import torch

# pure functions of the env state: no random numbers, no state writes
FUSED = {
    "LaneChangeEnv": ("_tables", "_collisions", "_lane_geometry", "_lat_mask", "_safe_to_enter", "_signal"),
    "IntersectionEnv": ("_tables", "_box_walls", "_permission", "_collisions", "_lane_geometry", "_lat_mask",
                        "_safe_to_enter", "_plan", "_rank", "_control"),
}


def available():
    if os.environ.get("LCRL_FUSE", "1") == "0" or not torch.cuda.is_available():
        return False
    try:
        import triton  # noqa: F401
    except ImportError:
        return False
    return True


def fuse_rules(env):
    """Replace the env's rule functions with fused (compiled) versions. Returns True if fused."""
    if getattr(env, "_fused", False):
        return True
    if not available():
        warnings.warn("lcrl.fuse: Triton not available (or LCRL_FUSE=0); running unfused", stacklevel=2)
        return False
    dyn = torch._dynamo.config
    for name in ("recompile_limit", "cache_size_limit"):      # one compiled graph per env instance
        if hasattr(dyn, name):
            setattr(dyn, name, max(getattr(dyn, name), 64))
    for name in FUSED[type(env).__name__]:
        setattr(env, name, torch.compile(getattr(env, name), fullgraph=True, dynamic=False))
    env._fused = True
    return True
