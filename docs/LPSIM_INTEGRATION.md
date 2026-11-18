# LPSim integration

LPSim, the Large-scale Parallel Simulator (`external/LPSim`, branch `rl-lane-change`), now runs two
additions, built and tested in Windows Subsystem for Linux (WSL) with Ubuntu 24.04 on an
NVIDIA RTX 2080 Ti graphics processing unit (GPU):

1. **Full-intersection control.** This is the rule set of [INTERSECTIONS.md](INTERSECTIONS.md):
   movement-based signal phases, all-way and two-way stop, yield, pedestrians, turning paths,
   and conflict zones inside the box. It is active when a network has `ix_*.csv` tables.
2. **The reinforcement learning (RL) lane-changing policy.** It is enabled with `RL_LANE_CHANGE=true` in
   `command_line_options.ini`. Without it, LPSim keeps its original rule-based lane changing.

The complete source change is in [`tools/patches/lpsim.patch`](../tools/patches/lpsim.patch)
(`git apply` inside an LPSim checkout).

## What LPSim did before

- **Intersections.** Each node gave green to one incoming edge at a time, rotating every 20 s.
  But vehicles never consulted those lights: a vehicle reaching the end of an edge moved onto
  the next edge whenever the first lane-map cell was free. There were no turn phases, no stop
  or yield control, and no conflicts inside the intersection.
- **Lane changes.** These were instantaneous, from a rule:
  - a probabilistic switch into a "mandatory" mode after 40% of the edge;
  - fixed critical gaps for acceptance;
  - a check only every 5 steps, and only above 3 m/s.
- **Vehicle size.** Vehicles are points in 1 m cells, so following gaps are measured
  front-to-front.
- **A config bug (fixed).** The Intelligent Driver Model (IDM) parameters `a`, `b`, `T` and `s_0` are read from the ini
  file but were missing from its key allow-list, so setting any of them aborted the run.
- **A crash (fixed).** The memory-logger thread was started before the members it writes
  to were constructed. This gave random segmentation faults (SIGSEGV), about 1 run in 10 on small networks.
- **A latent launch failure (fixed).** The vehicle kernel could need too many registers for
  LPSim's 384-thread blocks once code was added, failing at launch with exit code 0. It now
  has `__launch_bounds__(384)`.

## Building and running

```bash
# one-time toolchain (CUDA 12.8 = NVIDIA's Compute Unified Device Architecture, Qt5, Boost, pandana's contraction-hierarchy library)
wsl -d Ubuntu-24.04 -u root -- bash tools/wsl_setup.sh
# build external/LPSim into /root/lpsim-build
wsl -d Ubuntu-24.04 -u root -- bash tools/wsl_build_lpsim.sh

# a network from OpenStreetMap, with edge shapes and turn:lanes (edge_attrs.csv)
python external/LPSim/tools/generate_network.py --place "Berkeley, California, USA" --output <dir> --edge-attrs

# intersection tables for a network (any LPSim network: nodes.csv + edges.csv [+ edge_attrs.csv])
python tools/lpsim_intersections.py <network_dir> [--peds 100]

# policy weights into LPSim, then rebuild
python -m lcrl.export_cuda --policy runs/ppo_v10_dt01/policy.pt --out external/LPSim/src/simulator/lc_policy_weights.h

# run (LPSIM_RL=1 -> RL_LANE_CHANGE=true; LPSIM_A/B/T/S0 -> IDM parameters;
#      LPSIM_DT -> TIME_STEP, default 0.1 s)
LPSIM_RL=1 bash tools/wsl_run_lpsim.sh <network_dir> <max_trips> <start_hr> <end_hr>
```

**Time step.** Runs use `TIME_STEP=0.1`, the training env's physics step. With
`RL_LANE_CHANGE`, the policy is evaluated every 0.5 s (`rl::kDecisionDt`, every 5th step),
as in training.
- Between decisions, lane changes continue and the acceleration offset is held
  (`rlLon`).
- IDM, the Krauss cap and the intersection rules run at every 0.1 s step.

LPSim's own lane-change rule already checks every 5 steps, which is now every 0.5 s.

`tools/make_ix_network.py` builds a one-intersection test network with feeder links and
turning demand. Use it for any control type, with 3 or 4 legs.
- `--curve DEG` bends the legs.
- `--skew DEG` rotates them away from the compass points.
- `--turn-lanes "left|left|through|through;right"` sets OpenStreetMap (OSM) turn lanes on every approach.

`runs/lpsim_nets/ix_signal4_curve` is such a network: curved, skewed legs, and two left-turn
lanes on every approach.

## How the intersection tables are produced

`tools/lpsim_intersections.py` runs once per network. For every node with three or more legs:

- **Legs.** Edges are grouped into legs by neighbouring node.
  - A leg's centre line is the edge shape from `edge_attrs.csv` (well-known text (WKT), longitude/latitude) when present,
    otherwise a straight line towards the neighbour.
  - Turns are classified by the direction 40 m out along it: within ±35° is through, and a
    turn beyond 160° is a U-turn and is excluded.
- **Lanes.** Movements are allocated to lanes from the edge's OSM `turn:lanes` when tagged.
  This includes dual and triple turn lanes; exit legs get enough lanes for them.
  Otherwise a heuristic is used: the outer lanes take the turns and the inner lanes the
  straightest exit, and with two lanes both share the straight exit.
- **Why a sidecar file.** Edge shapes and turn lanes live in `edge_attrs.csv`
  (`uniqueid, geometry, turn_lanes`), not in `edges.csv`. LPSim's own comma-separated values (CSV) reader splits on
  every comma, and a WKT column crashed it. `generate_network.py --edge-attrs` writes the
  file; OSMnx keeps the `turn:lanes` tags for it.
- **Control type.** Taken from the OSM node tag:
  - `traffic_signals` → signal;
  - `stop` → all-way stop on small 4-leg nodes, two-way stop otherwise;
  - `give_way` → yield;
  - `motorway_junction` and nodes where every leg is at freeway speed → left to LPSim's
    pass-through, since lane changing on the links handles freeway merges;
  - untagged → two-way stop, with the widest street as the major street.
- **Geometry.** It then calls `lcrl.intersection.build`, the same geometry, conflict and
  signal code the training env uses.
  - It adds per-leg setbacks (`ix_legs.csv`), and staging points and places for permitted
    turns (`ix_paths.csv`: `s_wait_m`, `n_stage`).
  - It also writes curve speed caps for curved edges (`edge_curves.csv`, every 10 m).

On `sf_tertiary_v2` (31,121 nodes) this produces 23,038 intersections in about 60 s:
- 5,844 signals;
- 17,155 two-way stops;
- 38 all-way stops;
- 1 yield.

They contain 130,519 turning paths and 369,898 conflict entries.

## Kernel changes (`src/simulator/ix_lpsim.cuh`, `rl_lpsim.cuh`, hooks in `cuda_traffic_sim.cu`)

Each simulation step now runs in this order:

1. **`kernel_ixControl`, once per controlled node.** It sets each path's signal state and the
   time until it changes. It also runs pedestrians: Poisson arrivals, walk windows, and the
   crosswalk-occupied test.
2. **`kernel_ixPublish`, once per vehicle.** Every vehicle in the box, committed to it, or
   within 80 m of a controlled stop line appends itself to its path's *near* list
   (16 entries per path).
   - An entry holds the vehicle's position on the path, speed, maximum expected speed
     (`max(v, desired speed)`), rank, stop time and head / granted / stopped flags.
   - Vehicles approaching 80–250 m out go on the path's *far* list instead (16 entries). It
     is used only for gap acceptance against traffic with priority, so distant queues can
     never push near or committed vehicles out of the near list.
   - Entries beyond a list's capacity are counted as overflows in the `[ix]` report line.
3. **`kernel_trafficSimulation`, the existing vehicle kernel, with these hooks:**
   - **Box vehicles** follow their turning path. Leaders are vehicles on the same or a
     diverging path and the first car of the exit lane. Conflict zones held by others and busy
     crosswalks are walls. The vehicle leaves onto its exit lane through the lane map.
   - **Approach vehicles at a controlled node** follow box vehicles that started from their
     lane, record their stop, and evaluate entry permission (section 3 of the spec) from the
     occupancy lists. This includes the critical gap of the movement (left 4.0 s, through
     5.5 s, right 5.0 s) and a 1 s post-encroachment margin against traffic with priority
     up to 250 m away. Until entry is granted, the stop line is a hard wall: the movement is
     clipped so a vehicle can never cross it.
   - **Permitted turns stage.** A permitted turner without a gap pulls in to its path's
     staging point (`stageOk`). It is released by the same permission check from its own
     position, with priority to clear once its phase ends. Vehicles behind it in its lane
     wait, and staged vehicles are flagged in the occupancy lists (`kStaged`).
   - **Curves.** A vehicle's desired speed on a curved edge is capped by `edge_curves.csv`
     (`ix::curveCap`).
   - **With `RL_LANE_CHANGE`:**
     - It builds the 68-feature observation from the lane map, plus a parallel turn-indicator
       map, and runs `lcPolicyAct`.
     - A lane change takes 3 s, can be aborted, and writes the vehicle into both lanes.
     - The acceleration offset is added to IDM and capped by the Krauss safe speed behind
       leaders in both occupied lanes.
     - Driver traits (aggressiveness, information range, desired speed) are drawn per
       vehicle, as in training.
   - **Without `RL_LANE_CHANGE`,** LPSim's own lane-change rule runs, but at designed
     intersections its "OK lanes" come from the intersection tables.

LPSim intersection runs are single-GPU only (`NUM_GPUS=1`); the hooks are disabled
otherwise. Debug output: `LPSIM_IX_DEBUG=1` prints node 0's occupancy lists every 20 s.

## Matching the training dynamics

LPSim's default IDM parameters are a = 0.56 m/s², b = 2.9 m/s², T = 0.54 s and s₀ = 1.38 m,
with point vehicles. With a = 0.56, a vehicle starting from a stop needs about 10 s to cross a
26 m box. A stop-controlled minor approach then almost never finds a big enough gap, so
two-way stops jam even at 250 veh/h.

The parity runs below use the training env's dynamics instead: a = 1.5, b = 2.0, T = 1.2, and
s₀ = 6.38. The last one is 1.38 m plus a 5 m vehicle length, so LPSim's front-to-front gaps
become the env's bumper-to-bumper gaps.

## Results

See the README and `runs/lpsim_nets/` for the test networks and scripts. The rule-vs-RL table
is in the README's LPSim section.
