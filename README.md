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
  - aggressiveness and information range (50–200 m) are per-vehicle inputs.
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

**Time step and GPU execution.** Vehicle dynamics run at 0.1 s and the policy decides
every 0.5 s. Everything runs on the GPU with no host synchronisation inside a step, and each
decision is replayed as a CUDA graph. See [docs/DESIGN.md section 2](docs/DESIGN.md#2-time-step-and-gpu-execution).

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

## Usage

```bash
# train on freeway segments + complete intersections (about 4 s per iteration on an RTX 2080 Ti)
# (v11 was fine-tuned: v9 -> v10 (0.1 s physics) -> v11 (curves, multi-lane turns, staging),
#  each with --init <previous>/policy.pt --iterations 400 --lr 3e-4)
python -m lcrl.train --iterations 500 --lr 5e-4 --minibatch 32768 --out runs/ppo_v11_geom

# evaluate vs the LPSim-style rule baseline (freeway scenarios + every intersection kind)
python -m lcrl.evaluate --policy runs/ppo_v11_geom/policy.pt --out runs/ppo_v11_geom/eval.json

# export to CUDA (repo and LPSim), then check CUDA vs PyTorch parity
python -m lcrl.export_cuda --policy runs/ppo_v11_geom/policy.pt --out cuda/lc_policy_weights.h
LCRL_POLICY=runs/ppo_v11_geom/policy.pt python -m pytest tests -q

# LPSim (WSL): intersection tables for a network, build, run with the policy
python tools/lpsim_intersections.py external/LPSim/data/networks/sf_bay_area
wsl -d Ubuntu-24.04 -u root -- bash tools/wsl_build_lpsim.sh
LPSIM_RL=1 bash tools/wsl_run_lpsim.sh <network_dir> <max_trips> <start_hr> <end_hr>
```

## Visualising vehicle behaviour

```bash
python -m lcrl.visualize --policy runs/ppo_v11_geom/policy.pt --out viz/lane_change_viewer.html
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
| `lcrl/policy.py`, `lcrl/train.py` | actor-critic network; PPO on both envs |
| `lcrl/evaluate.py` | metrics, RL vs baseline, trait sweeps |
| `lcrl/export_cuda.py`, `cuda/` | weights → CUDA header; `lcPolicyAct()`; parity / timing harness |
| `lcrl/visualize.py`, `lcrl/visualize_ix.py`, `viz/viewer.html` | recorders and the HTML viewer |
| `tools/lpsim_intersections.py` | intersection tables (`ix_*.csv`) for any LPSim network |
| `tools/make_ix_network.py` | one-intersection LPSim test networks |
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

**LPSim, one-intersection test networks** (`TIME_STEP=0.1`, training-env dynamics), LPSim's
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

**LPSim, city scale** (v9, 0.5 s step; not yet rerun with v11). `sf_bay_area`: 155,620 controlled intersections, 70,891 trips over
one simulated hour.

| | Early trips finished* | Crossings | Missed movements | Compute for 1 h |
|---|---|---|---|---|
| Original LPSim (no intersection control) | 30.9% | n/a | n/a | 16 s |
| Intersection control + LPSim's lane-change rule | 22.9% | 1.77 M | 3.84% | 18 s |
| Intersection control + RL policy | 9.7% | 1.25 M | 0.72% | 31 s |

\*Trips that departed by 5:30 and finished by 6:00.

The RL policy at city scale is **not yet better** than LPSim's rule. It makes far fewer
missed turns, but it moves 29% fewer vehicles through intersections. Open items are listed
in [docs/DESIGN.md section 13](docs/DESIGN.md#13-limitations-and-open-items).
