"""Run a batched env's decision step as one CUDA graph.

One 0.5 s decision is ``cfg.substeps`` physics steps: thousands of small kernels. Launching
them one by one from Python costs more CPU time than the GPU spends running them, so
``GraphedEnv`` captures ``env.step(lat, lon)`` followed by ``env.observe()`` once and
replays it. This needs everything the step does to stay on the GPU with fixed shapes and no
host synchronisation, which lcrl/env.py and lcrl/ix_env.py guarantee.

By default the env's rule functions are also fused into fewer, larger kernels first
(lcrl/fuse.py), which cuts the GPU time itself.

The env object keeps working as usual: its state tensors become fixed buffers that each
replay updates in place, so code that reads ``env.x``, ``env._cache`` or
``env.last_outcome`` after a step sees the new values. Eager changes to the state (a
``reset()``, placing vehicles by hand) are picked up automatically before the next replay.
"""
import torch

from .fuse import fuse_rules


def _clone_tree(d, avoid=()):
    """Fixed buffers for a nested dict of tensors (clones; never aliases a state tensor)."""
    return {k: _clone_tree(v, avoid) if isinstance(v, dict) else v.clone() for k, v in d.items()}


def _copy_tree(dst, src):
    for k, v in src.items():
        if isinstance(v, dict):
            _copy_tree(dst[k], v)
        else:
            dst[k].copy_(v)


class GraphedEnv:
    def __init__(self, env, warmup=3, fuse=True):
        if fuse:                             # fused rule kernels (lcrl/fuse.py), compiled in the warm-up
            fuse_rules(env)
        self.env = env
        self.E, self.N = env.E, env.N
        self.lat = torch.zeros(env.E, env.N, dtype=torch.long, device=env.dev)
        self.lon = torch.zeros_like(self.lat)
        self.graph = None
        self.warmup = warmup
        self._obs = None

    def __getattr__(self, name):             # everything else is the env's
        return getattr(self.env, name)

    # -------------------------------------------------------------- state handling
    def _state(self):
        return {k: v for k, v in vars(self.env).items() if torch.is_tensor(v)}

    def _rebind(self):
        """Copy any rebound state tensor back into its fixed buffer and point the env at it.
        The tables cached by ``observe()`` (``env._cache``) are state too: the first physics
        step of the next decision uses them."""
        for name, buf in self.static.items():
            cur = getattr(self.env, name)
            if cur is not buf:
                buf.copy_(cur)
                setattr(self.env, name, buf)
        if self.env._cache is not self.static_cache:
            _copy_tree(self.static_cache, self.env._cache)
            self.env._cache = self.static_cache

    def _fn(self):
        r, d = self.env.step(self.lat, self.lon)
        out = self.env.observe()
        self._rebind()
        return r, d, out

    # ------------------------------------------------------------------ public API
    def observe(self):
        if self._obs is None:
            self._obs = self.env.observe()
        return self._obs

    def reset(self, *a, **kw):
        self.env.reset(*a, **kw)
        self._obs = None

    def step(self, lat, lon):
        self.lat.copy_(lat)
        self.lon.copy_(lon)
        if self.graph is None:
            return self._capture()
        self._rebind()                       # eager edits since the last replay
        self.graph.replay()
        r, d, self._obs = self._out
        return r, d

    def _capture(self):
        env = self.env
        self.observe()
        self.static = self._state()
        self.static_cache = _clone_tree(env._cache)
        env._cache = self.static_cache
        # warm-up (allocator, cuBLAS) before capture, then roll the state back so the graphed
        # trajectory is exactly the eager one
        saved = {k: v.clone() for k, v in self.static.items()}
        saved_cache = _clone_tree(self.static_cache)
        lat, lon = self.lat.clone(), self.lon.clone()
        rng = env.gen.get_state()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(self.warmup):
                self._fn()
        torch.cuda.current_stream().wait_stream(s)
        for k, v in saved.items():
            self.static[k].copy_(v)
        _copy_tree(self.static_cache, saved_cache)
        self.lat.copy_(lat)
        self.lon.copy_(lon)
        env.gen.set_state(rng)
        self.graph = torch.cuda.CUDAGraph()
        self.graph.register_generator_state(env.gen)
        with torch.cuda.graph(self.graph):
            self._out = self._fn()
        self.graph.replay()                  # the capture itself does not execute
        r, d, self._obs = self._out
        return r, d
