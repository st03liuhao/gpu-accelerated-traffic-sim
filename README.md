# GPU RL lane-changing model

A reinforcement learning (RL) lane-changing model for large-scale traffic simulation on the
graphics processing unit (GPU), in [LPSim](https://github.com/Xuan-1998/LPSim) (Large-scale Parallel Simulator). The requirements are in
[GPU-accelerated lane-changing model design.md](GPU-accelerated%20lane-changing%20model%20design.md).
The design is in [docs/DESIGN.md](docs/DESIGN.md); intersection rules are in
[docs/INTERSECTIONS.md](docs/INTERSECTIONS.md); the LPSim port is in
[docs/LPSIM_INTEGRATION.md](docs/LPSIM_INTEGRATION.md).

- **Model.** One shared 68→64→64→8 multilayer perceptron (MLP) for all vehicles. It outputs a lateral action and
  an acceleration offset added to the Intelligent Driver Model (IDM)
  car-following acceleration:
  - lateral: keep, left or right; steering back aborts a change, which is allowed only once
    continuing has become unsafe;
  - a safety shield makes collisions impossible by construction;
  - aggressiveness and information range (50–200 m) are per-vehicle inputs;
  - vehicles have lengths (cars 5 m, trucks 12 m) and all gaps are bumper to bumper, in
    the training environments and in LPSim.
- **Training.** Proximal Policy Optimization (PPO) on two batched GPU environments at once:
  - freeway segments (`lcrl/env.py`: on-ramp, off-ramp, lane drop, lane addition);
  - complete intersections with every approach (`lcrl/ix_env.py`: 4-leg and T; signals
    with protected and permitted lefts, right turn on red, all-way and two-way stop,
    yield, pedestrians);
  - general geometry, for real road networks: curved and skewed legs, any number of lanes,
    and dual or triple turn lanes. Permitted turners pull past the stop bar and wait for a
    gap.
- **Deployment in LPSim.** Two parts, both in `tools/patches/lpsim.patch`:
  - full-intersection control generated from any network's OpenStreetMap (OSM) tags, edge shapes and
    `turn:lanes` (`generate_network.py --edge-attrs`);
  - the policy as a CUDA (NVIDIA's Compute Unified Device Architecture) `__device__` function, enabled with `RL_LANE_CHANGE=true`.
    Its segment runs along the vehicle's path through nodes where every lane continues, and edges can have turn bays and
    acceleration / deceleration lanes from OSM lane counts (`generate_network.py --lane-spans`).

**Time step and GPU execution.** Vehicle kinematics run at 0.1 s and the policy decides
every 0.5 s. Everything runs on the GPU with no host synchronisation inside a step. The rule
functions are fused into fewer GPU kernels with `torch.compile` (`lcrl/fuse.py`), the
neighbour search is a hand-written CUDA kernel on sorted lane lists (`lcrl/neighbours.py`),
and each decision is replayed as a CUDA graph. See [docs/DESIGN.md section 1.5](docs/DESIGN.md#15-time-step-and-gpu-execution).

## Setup

```bash
conda create -n lcrl python=3.12
conda activate lcrl
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install numpy pytest matplotlib
pip install "triton-windows>=3.7,<3.8"   # Windows only: Triton for fused kernels (lcrl/fuse.py); bundled on Linux
```

CUDA parity tests on Windows need `nvcc` and the Microsoft Visual C++ (MSVC)
compiler (`cuda/build_test.bat` uses Visual Studio 2022; set
`VCVARS` to point elsewhere). LPSim builds in Windows Subsystem for Linux (WSL); see
[docs/LPSIM_INTEGRATION.md](docs/LPSIM_INTEGRATION.md).

Two speed-ups compile on first use and are then cached:
- **Fused kernels** (`lcrl/fuse.py`, `torch.compile`): about 30 s for the freeway
  environment and 70 s for the intersection environment. They need Triton.
- **The neighbour-search CUDA extension** (`lcrl/neighbours.py`, `lcrl/csrc/`): about 30 s.
  It needs `nvcc` and, on Windows, MSVC. If `cl` is not on PATH, the loader runs Visual
  Studio 2022's `vcvars64.bat` (or the file `VCVARS` points to). The CUDA file includes only
  ATen headers, because `torch/extension.h` does not compile under nvcc with MSVC; the
  Python binding is in `lcrl/csrc/neighbours.cpp`.

If either fails to build, the code falls back to plain PyTorch with a warning.
`LCRL_FUSE=0` and `LCRL_NB_KERNEL=0` switch them off.

## Usage

```bash
# train on freeway segments + complete intersections, in stages (docs/DESIGN.md 1.3):
# v9 from scratch, then v10 (0.1 s physics), v11 (general geometry) and v12 (2 km look-ahead,
# weaving, preparation distance) each fine-tuned from the previous
python -m lcrl.train --iterations 500 --lr 5e-4 --minibatch 32768 --out runs/ppo_v9_ix
python -m lcrl.train --iterations 400 --lr 3e-4 --minibatch 32768 --init runs/ppo_v9_ix/policy.pt --out runs/ppo_v10_dt01
python -m lcrl.train --iterations 400 --lr 3e-4 --minibatch 32768 --init runs/ppo_v10_dt01/policy.pt --out runs/ppo_v11_geom
python -m lcrl.train --iterations 400 --lr 3e-4 --init runs/ppo_v11_geom/policy.pt --out runs/ppo_v12_prep

# evaluate vs the LPSim-style rule baseline (freeway scenarios + every intersection kind)
python -m lcrl.evaluate --policy runs/ppo_v12_prep/policy.pt --out runs/ppo_v12_prep/eval.json

# export to CUDA (repo and LPSim), then check CUDA vs PyTorch parity
python -m lcrl.export_cuda --policy runs/ppo_v12_prep/policy.pt --out cuda/lc_policy_weights.h
LCRL_POLICY=runs/ppo_v12_prep/policy.pt python -m pytest tests -q

# LPSim (WSL): a network from OSM with lane spans, its intersection tables, build, run with the policy
python external/LPSim/tools/generate_network.py --place "Berkeley, California, USA" --output <dir> --lane-spans
python tools/lpsim_intersections.py <dir>
wsl -d Ubuntu-24.04 -u root -- bash tools/wsl_build_lpsim.sh
LPSIM_RL=1 bash tools/wsl_run_lpsim.sh <network_dir> <max_trips> <start_hr> <end_hr>
# LPSIM_RL=rule: the training env's rule baseline on the same code path as RL; unset: LPSim's own rule
```

## Visualising vehicle behaviour

```bash
python -m lcrl.visualize --policy runs/ppo_v12_prep/policy.pt --out viz/lane_change_viewer.html
```

This runs the rule baseline and the RL policy on the same randomised geometry and demand,
for the four freeway scenarios and the six intersection kinds (120 s warm-up, then 200 s
recorded). It writes **one self-contained HyperText Markup Language (HTML) file**; open it in any browser.

- **Freeway segments** show as synchronised lane views.
- **Intersections** show top-down, rule and RL side by side:
  - approach and exit lanes, turn bays, crosswalks with pedestrians;
  - turning paths coloured by signal state (protected / permitted / amber);
  - per-lane signal heads, stop and yield signs;
  - vehicles turning through the box.
- **Colour by** next movement, speed, longitudinal action, aggressiveness or information
  range.
- **Lane-change detail:** turn signals, lane-change outlines, abort rings, trails, and
  ✓ / ✗ markers for turns made or missed.
- **Interaction:**
  - hover for a vehicle's state; click to select it (shows its information range);
  - scroll to zoom, drag to pan;
  - a comparison chart over time (click it to seek);
  - deep links such as `#scen=4&t=120`.

Options: `--scenarios` (freeway / all / none / names), `--intersections` (all / none / kinds),
`--variants`, `--seconds`, `--policy none`, `--json`.

## Layout

| Path | Contents |
|---|---|
| `lcrl/config.py` | constants, observation layout, env / reward / PPO configs |
| `lcrl/scenarios.py`, `lcrl/env.py` | freeway segments: generators, batched env, shield, reward, rule baseline |
| `lcrl/intersection.py` | intersection designs: geometry, turning paths, conflict zones, signal plans, crosswalks |
| `lcrl/ix_env.py` | batched env of complete intersections (right-of-way rules, box, pedestrians) |
| `lcrl/graph.py`, `lcrl/fuse.py` | CUDA-graph replay of a decision; kernel fusion with `torch.compile` |
| `lcrl/neighbours.py`, `lcrl/csrc/` | neighbour search on sorted lane lists (CUDA kernel, built on first use) |
| `lcrl/policy.py`, `lcrl/train.py` | actor-critic network; PPO on both envs |
| `lcrl/evaluate.py` | metrics, RL vs baseline, trait sweeps |
| `lcrl/export_cuda.py`, `cuda/` | weights → CUDA header; `lcPolicyAct()`; parity / timing harness |
| `lcrl/visualize.py`, `lcrl/visualize_ix.py`, `viz/viewer.html` | recorders and the HTML viewer |
| `tools/lpsim_intersections.py` | intersection tables (`ix_*.csv`) for any LPSim network |
| `tools/make_ix_network.py` | one-intersection LPSim test networks (`--left-bay` for turn bays) |
| `tools/make_fwy_network.py` | freeway LPSim test network: on-ramp with acceleration lane, off-ramp with deceleration lane |
| `tools/wsl_*.sh` | WSL toolchain setup, LPSim build, LPSim run |
| `tools/patches/lpsim.patch` | all LPSim source changes (`git apply` in an LPSim checkout) |
| `external/LPSim` | LPSim checkout with the patch applied (branch `rl-lane-change`) |
| `runs/lpsim_nets/` | LPSim test networks, comparison scripts and their outputs |
| `tests/` | env mechanics, intersection designs and env, LPSim tool, viewer, CUDA parity |

## Results (release model `runs/ppo_v11_geom`, greedy policy, 0.1 s physics)

**Training environments, rule baseline → RL.** 128 freeway segments per scenario and 64
intersections per kind, 600 s after warm-up; full tables in `runs/ppo_v11_geom/eval.json`.
- Intersection geometry is randomised: leg directions within ±12°, 40% of legs curved, and
  1–4 through lanes with single or dual turn lanes.
- Permitted turners pull into the box to wait for a gap.
- The 4-leg signal is the study design ([INTERSECTIONS.md §7](docs/INTERSECTIONS.md)).

| | Throughput (veh/h) | Route success | Delay (s/veh) | Collisions | Violations |
|---|---|---|---|---|---|
| Freeway (4 scenarios) | 3,227 → 3,341 | 99.2 → 100% | 11.6 → 3.1 | 0 → 0 | 0 → 0 |
| 4-leg signal (study design) | 1,264 → 1,317 | 99.8 → 100% | 44.4 → 29.1 | 0 → 0 | 0 → 0 |
| T signal | 2,832 → 2,940 | 90.7 → 96.1% | 56.4 → 41.8 | 0 → 0 | 0 → 0 |
| All-way stop | 616 → 702 | 99.9 → 99.5% | 136 → 114 | 0 → 0 | 0 → 0 |
| Two-way stop, 4-leg | 971 → 1,129 | 98.5 → 99.7% | 44.0 → 32.1 | 0 → 0 | 0 → 0 |
| Two-way stop, T | 1,066 → 1,108 | 96.2 → 98.6% | 53.1 → 38.0 | 0 → 0 | 0 → 0 |
| Yield, 4-leg | 894 → 1,008 | 96.6 → 99.4% | 46.0 → 39.0 | 0 → 0 | 0 → 0 |

The RL policy now raises throughput as well as cutting delay at every intersection kind;
v10 lost capacity at the T signal and the all-way stop. Some random designs get more demand
than they can serve, so both policies have timeouts there (vehicles queued for more than
10 minutes; 2,198 for the rule and 1,696 for RL across all intersection runs).

**The 4-leg signal study design** at 500 veh/h per leg, over 900 s (the viewer recording):

| Metric | Rule baseline | RL |
|---|---|---|
| Delay | 24.2 s/veh | 11.9 s/veh |
| Throughput | 2,048 veh/h | 2,080 veh/h |
| Turns completed | 100% | 100% |
| Vehicles that waited in the box for a gap | 26 | 29 |

**LPSim, one-intersection test networks** (`TIME_STEP=0.1`, training-env kinematics), LPSim's
lane-change rule → RL. Both use the new intersection control.

| Network | Median travel time | Trips finished | Missed movements |
|---|---|---|---|
| 4-leg signal, 2 shared lanes (staging) | 76 → 61 s | 100 → 100% | 0.82% → 0.05% |
| 4-leg signal, curved and skewed legs, dual lefts | 88 → 77 s | 100 → 100% | 1.41% → 0.09% |
| T signal | 60 → 54 s | 100 → 100% | 0.63% → 0% |
| All-way stop | 252 → 64 s | 100 → 100% | 0 → 0 |
| Two-way stop | 57 → 48 s | 100 → 100% | 0 → 0 |
| Yield | 52 → 44 s | 100 → 100% | 0 → 0 |

All runs have zero stop-line violations, conflict overlaps and occupancy-list overflows.

**LPSim, lane spans and the route-aware segment** (rule → RL, `runs/lpsim_nets/run_lanes.sh`;
details in [docs/LPSIM_INTEGRATION.md](docs/LPSIM_INTEGRATION.md#lane-geometry-and-the-vehicles-segment-srcsimulatorlane_geocuh)).

| Network | Median travel time | Trips finished | Wrong-lane crossings / missed movements |
|---|---|---|---|
| Freeway, on-ramp acceleration lane + off-ramp deceleration lane (3 demand seeds) | 171–194 → 107–109 s | 100 → 100% | 138–147 → 1–8 crossings |
| 4-leg signal with 60 m left-turn bays | 80 → 71 s | 100 → 100% | 1.45% → 0.55% |
| Same two networks with v12 | 172 → 107 s; 80 → 71 s | 100 → 100% | 131 → 5 crossings; 1.45% → 0.13% |

**LPSim, with vehicle lengths and the fair rule baseline** (`runs/lpsim_nets/time_rule_rl.sh`),
LPSim's rule / the training env's rule on the RL code path (`LPSIM_RL=rule`) / v12:

| Network | Median travel time | Other result |
|---|---|---|
| Freeway with ramps | 266 / 140 / 107 s | all trips finished in every mode |
| 4-leg signal with 60 m left-turn bays | 83 / 82 / 72 s | missed movements 2.39% / 0.17% / 0.64% |

Driver traits (slow trucks, desired speeds, lengths) now apply in every mode, which is why
LPSim's rule is slower than in the tables above. v12 misses more turns than the env rule
at the 4-leg signal (open item).

**LPSim, city scale** (v12, 0.1 s step, before vehicle lengths,
`runs/lpsim_nets/city_bay_v12_envrule.txt`). `sf_bay_area`: San Francisco Bay Area network,
70,891 trips over one simulated hour.

| | Early trips finished* | Mean speed (miles per hour) | Missed movements | Conflict-zone overlaps | GPU simulation loop |
|---|---|---|---|---|---|
| Intersection control + LPSim's lane-change rule | 26.4% | 43.8 | 3.18% | 9 | 79–84 s |
| Intersection control + env rule (`LPSIM_RL=rule`) | 24.3% | 42.4 | 4.25% | 0 | — |
| Intersection control + RL policy v12 | 28.0% | 51.2 | 3.67% | 43 | 130 s (153 s before the decision-step speed-up) |

\*Trips that departed in the first half hour and finished within the hour.

v12 finishes more trips and drives faster than both rules at city scale. It misses slightly
more movements than LPSim's rule, and the env rule misses the most; why is not yet
explained. Open items are listed in
[docs/DESIGN.md section 10](docs/DESIGN.md#10-limitations-and-open-items); timing details in
[docs/DESIGN.md section 9.2](docs/DESIGN.md#92-computation-speed).
