# LPSim integration

LPSim, the Large-scale Parallel Simulator (`external/LPSim`, branch `rl-lane-change`), now runs two
additions, built and tested in Windows Subsystem for Linux (WSL) with Ubuntu 24.04 on an
NVIDIA RTX 2080 Ti graphics processing unit (GPU):

1. **Full-intersection control.** This is the rule set of [INTERSECTIONS.md](INTERSECTIONS.md):
   movement-based signal phases, all-way and two-way stop, yield, pedestrians, turning paths,
   and conflict zones inside the box. It is active when a network has `ix_*.csv` tables.
2. **The reinforcement learning (RL) lane-changing policy.** It is enabled with `RL_LANE_CHANGE=true` in
   `command_line_options.ini`. Without it, LPSim keeps its original rule-based lane changing.
   With `RL_LANE_CHANGE=true` and the environment variable `LPSIM_LC_RULE=1`, the training
   env's rule baseline decides instead of the policy, on the same code path
   ([section on lane-change modes](#lane-change-modes)).

Vehicles now have lengths (cars 5 m, trucks 12 m), and all gaps are bumper to bumper
([Vehicle lengths](#vehicle-lengths)).

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
- **Vehicle size.** Vehicles were points in 1 m cells, so following gaps were measured
  front-to-front. Now each vehicle has a length ([Vehicle lengths](#vehicle-lengths)).
- **Driver traits.** Every vehicle drove at the speed limit under LPSim's rule. Driver
  traits (slow trucks, desired-speed factors) are now drawn for every vehicle in every mode.
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
python -m lcrl.export_cuda --policy runs/ppo_v12_prep/policy.pt --out external/LPSim/src/simulator/lc_policy_weights.h

# run (LPSIM_RL=1 -> RL policy; LPSIM_RL=rule -> the training env's rule on the RL code path;
#      unset -> LPSim's own rule; LPSIM_A/B/T/S0 -> IDM parameters;
#      LPSIM_DT -> TIME_STEP, default 0.1 s)
LPSIM_RL=1 LPSIM_A=1.5 LPSIM_B=2.0 LPSIM_T=1.2 LPSIM_S0=1.38 \
  bash tools/wsl_run_lpsim.sh <network_dir> <max_trips> <start_hr> <end_hr>
```

**Time step.** Runs use `TIME_STEP=0.1`, the training env's physics step. With
`RL_LANE_CHANGE`, the policy is evaluated every 0.5 s (`rl::kDecisionDt`, every 5th step),
as in training.
- Between decisions, lane changes continue and the acceleration offset is held
  (`rlLon`).
- The route-aware segment and the observation are built only on decision steps. In between,
  the vehicle holds its deadline, reduced by the distance driven, and its lanes-to-go
  (`vehicle.rlDeadline`, `vehicle.rlNeed`). During a lane change it scans only the leader in
  the other lane (`rl::leadInLane`). This cut LPSim's RL run time by up to a third with the
  same traffic results ([Computation speed](#computation-speed)).
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
   - **Driver traits,** in every mode (`rl::initTraits`): aggressiveness, information
     range, desired speed factor, 12% slow trucks and vehicle length are drawn per vehicle,
     as in training. Before, they were drawn only with `RL_LANE_CHANGE`, so under LPSim's
     rule every vehicle drove at the speed limit and the comparison was not fair.
   - **Without `RL_LANE_CHANGE`,** LPSim's own lane-change rule runs, but at designed
     intersections its "OK lanes" come from the intersection tables.

LPSim intersection runs are single-GPU only (`NUM_GPUS=1`); the hooks are disabled
otherwise. Debug output: `LPSIM_IX_DEBUG=1` prints node 0's occupancy lists every 20 s.

### Lane-change modes

`tools/wsl_run_lpsim.sh` selects one of three modes with `LPSIM_RL`:

| `LPSIM_RL` | Settings | Who decides lane changes |
|---|---|---|
| unset | — | LPSim's own rule (`calculateGapsLC` and its mandatory / discretionary logic) |
| `1` | `RL_LANE_CHANGE=true` | the RL policy, `lcPolicyAct` |
| `rule` | `RL_LANE_CHANGE=true`, environment variable `LPSIM_LC_RULE=1` | the training env's rule baseline, `rl::ruleAct` in `rl_lpsim.cuh` |

In `rule` mode the rule runs on the same code path as RL: route-aware segment, observation,
safety shield, 3 s lane-change motion and give-up rule. Only the decision differs, so
`rule` against `1` isolates the policy's contribution. `rl::ruleAct` mirrors `rule_action`
in `lcrl/env.py`:
- change toward the route lane as soon as the shield allows;
- otherwise change discretionarily when held up by a close leader (lead gap under 50 m)
  and an adjacent lane that serves the route has a longer gap. "Held up" means the leader
  is more than 2 m/s slower, or the vehicle drives below 80% of its desired speed;
- no acceleration offset (pure IDM).

The 80% condition is new. Without it the rule never left a slow queue, because the leader
in a queue is no slower than the vehicle behind it, and LPSim puts every entering vehicle in
the right lane.

### Vehicle lengths

Vehicle positions are front bumpers. Each vehicle has a length `vehicle.vlen`: cars 5 m,
trucks 12 m (`rl::kCarLen`, `rl::kTruckLen`, matching `EnvConfig.veh_len` and
`EnvConfig.truck_len` in `lcrl/config.py`).
- **Length map.** `lg::View::lenMap` (`lane_geo.cuh`) holds one length byte per lane-map
  cell, beside the lane map. It is written at every lane-map write (`lg::putLen`) and read
  with `lg::lenAt` (default 5 m).
- **Gaps.** Every leader gap, LPSim's own gap acceptance (`calculateGapsLC`), the RL
  neighbour scans and the intersection occupancy entries (`Occ.len` in `ix_lpsim.cuh`) use
  the real lengths, so all gaps are bumper to bumper.
- **Spawning** needs empty cells for the vehicle's own length plus s₀ behind its front.
- **Jam distance.** The IDM jam distance is now the true s₀ = 1.38 m in all run scripts.
  Before, `LPSIM_S0=6.38` hid a 5 m length in it ([Matching the training kinematics](#matching-the-training-kinematics)).

### Bugs fixed

Both in `cuda_traffic_sim.cu`:
- **Undone moves were invisible.** A vehicle whose move was undone (its target cell was
  taken) stayed where it was but was not written into the lane map that step, so its
  followers drove into it. `lgStayVisible` now writes it.
- **Gap to a leader on the next edge.** It counted only the distance into the next edge, not
  what was left of the current edge. It now adds the rest of the current edge and subtracts
  the leader's length.

## Lane geometry and the vehicle's segment (`src/simulator/lane_geo.cuh`)

In LPSim an edge runs from one junction to the next, and every lane of an edge used to span
the whole edge. The policy used to see only the vehicle's current edge. Two additions bring
LPSim's roads closer to the training segments.

### Lanes that start or end part-way, and lane links

Two optional files next to the network describe the lanes:

| File | Columns | Meaning |
|---|---|---|
| `edge_lanes.csv` | `uniqueid, lane, start_m, end_m` | Lane `lane` (0 = leftmost) exists only on `[start_m, end_m]` of the edge: a turn bay, an acceleration or deceleration lane, a lane drop. Edges not listed keep full-length lanes. |
| `lane_links.csv` | `in_edge, out_edge, shift` | At a node without intersection control, lane k of `in_edge` continues as lane k + `shift` of `out_edge`. An in-edge with any row uses only its rows. Out-edges without a row (such as a U-turn) continue no lane. |

Without the files LPSim behaves as before: full-length lanes, and the same lane index across a
node, clamped to the next edge's lanes. With them:
- **Spawning** uses the rightmost lane that exists at the spawn point.
- **Crossing a node** without intersection control puts the vehicle in the lane its lane
  continues into. If it does not continue (wrong lane at a diverge), the vehicle goes to the
  nearest lane that exists. These forced crossings are counted.
- **The end of a lane inside an edge is a wall**, like a closed stop line. Lane changes,
  for both the RL policy and LPSim's rule, go only into lanes that exist at the vehicle's
  position.
- **LPSim's rule** treats a lane ending within 150 m as a mandatory change, at any speed.
  A vehicle stopped at a lane end moves into the next lane once that lane is free from s₀
  ahead to s₀ + 1.5 s of the follower's speed behind. (The rule otherwise never changes
  lanes below 3 m/s, and a stopped vehicle never reaches LPSim's lane-change code.)
- **Turning paths** at controlled intersections exit only into lanes that exist where the
  exit edge starts (`tools/lpsim_intersections.py` offsets the exit lane).

The files are written by `generate_network.py --lane-spans`:
- **Edges.** It downloads the unsimplified OSM graph and simplifies it with OSMnx's
  `track_merged`, so each edge keeps the OSM ways it is made of. Lane-count changes along an
  edge become lane spans.
- **Which side.** Added lanes go on the left when the turn-lane tag of the wider part starts
  with left turns, dropped lanes on the left when they merge to the right, otherwise on the
  right. A bay is at least 30 m long: OSM often splits a way only a few metres before the
  stop line.
- **Edge attributes.** `lanes` in `edges.csv` becomes the largest count along the edge, and
  `turn_lanes` comes from the edge's last OSM way (at the stop line).
- **Lane links.** Merging roads are stacked left to right by their geometry. At a diverge the
  leftmost branch keeps the left lanes and the rightmost branch the right lanes. Otherwise
  lanes are aligned on the right.

On a 545-edge area around the I-80 / University Avenue interchange in Berkeley, 38 edges get
such lanes: 36 left-turn bays from `turn:lanes` tags and 4 lanes that end part-way.

### The vehicle's segment

The policy was trained on one road segment at a time. Its deadline, lane runways and route
lanes are measured to the segment end. In LPSim the vehicle's segment is now its current edge
plus the edges ahead on its path (`rlBuildChain`), through every node where all lanes
continue and none has to be chosen. The segment ends at the first of:
- a controlled intersection (route lanes from its design);
- a diverge or lane drop: some lane does not continue, or the route uses only some lanes
  (route lanes from the lane links, else from LPSim's own lane rule);
- the end of the path;
- 2 km (the distance inputs saturate there; 600 m before v12), or 24 edges;
- a lane slot that ended in the segment and would start again (one slot can hold, say, an
  acceleration lane and then the next ramp's deceleration lane, but a training segment has
  one span per lane).

Along the segment:
- **Lane inputs.** Runway, "starts ahead", route lanes and lanes-to-route are computed per
  lane slot (current-edge lane index) through the lane links. A lane that appears only on a
  later edge, such as an on-ramp's target lane, is visible as starting ahead.
- **Neighbour search.** It runs along the segment, and up to 4 edges back on the path for
  followers.
- **Absent lane slots.** Slots that the training env would have but the road lacks (index
  below 6 on freeways, 8 at intersections) now get the env's "absent lane" inputs: a gap of
  the information range and the speed limit. Before, they were all zero.
- **Controlled intersection ahead.** A segment that ends there several edges ahead gets that
  intersection's category, movement, rank and signal time.

**Give-up rule.** A vehicle stopped at its segment end outside the route lanes for 30 s
stops braking on purpose and drives on, missing its route. Training ends every vehicle at its
age limit, so the policy never learnt to resolve two vehicles that each wait for the other's
lane. On the freeway test below, exiters in lane 2 and through vehicles in the deceleration
lane blocked each other at the gore until the end of the run. The reward prices a miss as a
90 s detour, so waiting much longer than 30 s is worse for the vehicle too.

Counters: `[lg] node crossings N, from a lane that does not continue F, vehicle-steps stopped
at a lane end S`. `LPSIM_LG_DEBUG=1` writes the unfinished vehicles (edge, lane, position,
speed) to `unfinished.csv`.

### Test networks and results

- `tools/make_fwy_network.py` builds a 3-lane freeway: an on-ramp merging through a 250 m
  acceleration lane, a plain node, and an off-ramp reached through a 250 m deceleration lane.
  - `--no-lanes` leaves out the two files.
  - Every edge has a reverse twin without demand: LPSim's lane-map setup crashes on a node
    with only outgoing or only incoming edges.
- `make_ix_network.py --left-bay 60` gives every approach of the one-intersection network a
  60 m left-turn bay; the feeder's two lanes link to the two through lanes.

`runs/lpsim_nets/run_lanes.sh` runs both with training-env kinematics, rule → RL (the v11 policy, except rows marked v12):

| Network | Median travel time | 90th percentile | Trips finished | Crossings from a lane that does not continue |
|---|---|---|---|---|
| Freeway ramps, demand seed 1 | 171 → 107 s | 356 → 135 s | 100 → 100% | 138 → 1 |
| Freeway ramps, seed 2 | 173 → 108 s | 473 → 136 s | 100 → 100% | 147 → 8 |
| Freeway ramps, seed 3 | 194 → 109 s | 596 → 138 s | 100 → 100% | 146 → 2 |
| Freeway ramps, seed 1, with v12 | 172 → 107 s | 361 → 134 s | 100 → 100% | 131 → 5 |

| Network | Median travel time | 90th percentile | Trips finished | Missed movements |
|---|---|---|---|---|
| 4-leg signal with 60 m left-turn bays | 80 → 71 s | 116 → 108 s | 100 → 100% | 1.45% → 0.55% |
| Same, with v12 (2 km segment, log distance inputs) | 80 → 71 s | 116 → 108 s | 100 → 100% | 1.45% → 0.13% |
| 4-leg signal, no bays (previous build) | 76 → 61 s | 132 → 90 s | 100 → 100% | 0.82% → 0.05% |
| 4-leg signal, no bays (this build) | 76 → 60 s | 132 → 86 s | 100 → 100% | 0.82% → 0% |

- **RL keeps to the lanes.** Under RL, 1 to 8 vehicles per run cross a node from a lane
  that does not continue (and are put in the nearest lane); under the rule, about 140.
- **Vehicles stopped at a lane end.** Rule vehicles spend about 28,000 vehicle-steps (0.1 s
  each) stopped at the end of the acceleration lane, RL vehicles 0 to 218.
- **Before the give-up rule,** RL finished only 78.5% of trips on seed 2.
- **Effect of the segment alone.** On the same freeway without lane files, RL finished 98.4%
  on the previous build (single-edge view) and 100% on this build.
- **The rule baseline is unchanged:** on the network without bays, the rule results are
  identical on both builds.
- LPSim runs are not exactly repeatable (vehicles race for lane-map cells), so rule
  runs vary by a few percent in the 90th percentile.

**With vehicle lengths and the fair rule** (`runs/lpsim_nets/time_rule_rl.sh`, current
build), LPSim's rule / env rule (`LPSIM_RL=rule`) / v12:

| Network | Median travel time | Other result |
|---|---|---|
| Freeway ramps | 266 / 140 / 107 s | all trips finished in every mode |
| 4-leg signal with 60 m left-turn bays | 83 / 82 / 72 s | missed movements 2.39% / 0.17% / 0.64% |

- LPSim's rule is much slower on the freeway than in the table above (266 s against
  171 s), because its vehicles now have slow trucks, desired-speed factors and lengths too.
- The env rule leaves slow queues and closes most of the gap to v12 on the freeway.
- At the 4-leg signal, v12 is fastest but misses more turns than the env rule. This is an
  open item ([DESIGN.md section 10](DESIGN.md#10-limitations-and-open-items)).
- v13 (`runs/ppo_v13_len`), v12 fine-tuned with vehicle lengths, did the same or slightly
  worse in LPSim, so LPSim keeps v12.

The observation dump (`LPSIM_RL_DUMP=1`) on the bay network shows both parts working:
- 149 of 662 sampled observations have a segment end beyond the vehicle's own 200 m edge
  (up to 299 m: feeder plus approach);
- for left-turners the bay appears as the left lane, starting ahead and serving the route,
  with "lanes to the route lane" −1 or −2 until they are in it.

## Matching the training kinematics

LPSim's default IDM parameters are maximum acceleration a = 0.56 m/s², comfortable
deceleration b = 2.9 m/s², time headway T = 0.54 s and jam distance s₀ = 1.38 m. With
a = 0.56, a vehicle starting from a stop needs about 10 s to cross a 26 m box. A
stop-controlled minor approach then almost never finds a big enough gap, so two-way stops
jam even at 250 veh/h.

The parity runs use the training env's vehicle kinematics instead: a = 1.5 m/s²,
b = 2.0 m/s², T = 1.2 s and s₀ = 1.38 m (`LPSIM_A=1.5 LPSIM_B=2.0 LPSIM_T=1.2 LPSIM_S0=1.38`).
Gaps are bumper to bumper, with each vehicle's own length ([Vehicle lengths](#vehicle-lengths)).
Before lengths were modelled, vehicles were points and runs set s₀ = 6.38 m (1.38 m plus a
5 m length), so that front-to-front gaps became the env's bumper-to-bumper gaps.

## Results

See the README and `runs/lpsim_nets/` for the test networks and scripts. The rule-vs-RL table
is in the README's LPSim section.

### City network

San Francisco Bay Area network `sf_bay_area` (70,891 trips, 1 h simulated, 0.1 s step,
before vehicle lengths, `runs/lpsim_nets/city_bay_v12_envrule.txt`), LPSim's rule / env rule
/ v12:

| Measure | LPSim's rule | Env rule | v12 |
|---|---|---|---|
| Trips departed in the first half hour that finished | 26.4% | 24.3% | 28.0% |
| Mean speed (miles per hour) | 43.8 | 42.4 | 51.2 |
| Missed movements | 3.18% | 4.25% | 3.67% |
| Conflict-zone overlaps (of about 1.8 million crossings) | 9 | 0 | 43 |

All three have zero stop-line violations. Why the env rule misses more movements at city
scale than on the test networks is not yet explained
([DESIGN.md section 10](DESIGN.md#10-limitations-and-open-items)).

### Computation speed

LPSim's own timer for the GPU simulation loop, before vehicle lengths, LPSim's rule / RL
before / RL after building the segment and observation only on decision steps
([Building and running](#building-and-running), "Time step"):

| Network | Rule | RL before | RL after |
|---|---|---|---|
| Freeway ramps, 1.5 h | 11.3–13.2 s | 19.2–20.1 s | 12.4–13.0 s |
| 4-leg signal with bays | 12.8–13.9 s | 18.4–19.0 s | 15.0–15.7 s |
| San Francisco Bay Area city network | 79–84 s | 153 s | 130 s |

- Ranges are repeated runs (`runs/lpsim_nets/time_rule_rl.sh`; city:
  `runs/lpsim_nets/city_bay_v12.txt` before, `city_bay_v12_fast.txt` after).
- Traffic results did not change.
- At city scale, RL costs 1.6× the rule's time, down from 1.8×.
