# GPU-Accelerated RL Lane-Changing Model: Design and Training

## 1. Purpose and scope

This document specifies a reinforcement-learning (RL) lane-changing model for large-scale traffic simulation. The model selects lane-changing intentions and longitudinal adjustments, searches for safe gaps, and adapts or cancels a maneuver when conditions change.

The design targets the graphics processing unit (GPU)-accelerated simulator [LPSim](https://github.com/Xuan-1998/LPSim) (Large-scale Parallel Simulator). A simulator adapter is needed to connect the policy to the simulator's vehicle state, lane topology, action interface, and episode controls. Confirm the exact LPSim application programming interface (API) and GPU execution path before implementation; this design does not assume a particular API.

The model does not plan routes or high-level trajectories. It receives route or lane-level objectives from the simulation and acts on the next lane-changing decision. If a desired turn or exit is missed, the external route planner handles recovery.

### Required capabilities

The policy and its safety layer must support:

- Passing a slow vehicle, including temporarily leaving the desired lane.
- Moving to a shorter queue near intersections or congested freeway sections.
- Moving proactively toward a required turn, ramp, or exit lane.
- Moving out of the current lane to create space for another vehicle to enter.
- Executing mandatory lane changes toward a desired lane.
- Finding safe target-lane gaps, and adjusting speed in the current lane while waiting for a gap.
- Avoiding collisions and limiting adverse effects on the target-lane lag vehicle.
- Accommodating cut-ins by slowing down, and creating a gap for another vehicle by speeding up when safe.
- Canceling or revising an uncompleted lane change if the maneuver becomes infeasible.

Optional extensions include aggressiveness and information-range settings. The policy should remain compact enough to support batched GPU inference and training.

## 2. Recommended system design

Use a **shared, feed-forward policy** for all controlled vehicles, paired with a deterministic **action-safety and maneuver-management layer**. Sharing one policy across vehicles limits model size and allows parallel training. The safety layer enforces hard feasibility constraints that should not be left to reward optimization alone.

```text
Simulator state
    -> observation builder
    -> shared RL policy
    -> action-safety filter and maneuver manager
    -> simulator controls
```

The policy runs at a configurable decision interval (initially 0.5–1.0 simulated seconds). The simulator may continue integrating vehicle dynamics at a smaller time step. Update observations and recheck maneuver safety at every policy decision; if the simulator supports it, run the safety check at every simulation step.

### 2.1 Policy and simulator responsibilities

The policy chooses a desired lane direction and longitudinal intent. The simulator or adapter remains responsible for vehicle dynamics, car-following constraints, lane geometry, legal movement constraints, and collision detection.

The maneuver manager:

1. Checks that the requested adjacent lane exists and is reachable.
2. Checks for sufficient front and rear gaps using predicted relative motion and configured safety limits.
3. Rejects or modifies infeasible lane-change requests.
4. Rechecks an active maneuver as surrounding traffic changes.
5. Cancels an unstarted request or transitions an active maneuver to the safest supported outcome if the safety check fails. The precise mid-maneuver behavior must match simulator capabilities; do not assume a vehicle can instantaneously return to its original lane.
6. Applies longitudinal safety limits independently of the policy.

The safety layer should expose its decisions (accepted, rejected, canceled, or modified) to logging and training. Repeatedly issuing unsafe actions should have a measurable learning cost, but policy rewards must not override a hard safety rejection.

### 2.2 Integration with car-following control

The RL policy should provide lane-change and speed **intent**, while the simulator's car-following controller remains responsible for safe longitudinal tracking. Avoid having both components issue independent acceleration commands. Use one authoritative controller or safety arbiter to produce the final applied longitudinal command, subject to validated acceleration and jerk limits for passenger comfort.

At each simulator control step:

1. The car-following controller computes its normal response to the current-lane leader and desired speed.
2. The RL policy proposes lane intent and a bounded longitudinal preference, such as a modest increase, decrease, or no change to desired speed.
3. The maneuver manager validates the lane-change request against road rules and target-lane front/rear gaps, including predicted separation over the maneuver.
4. The adapter passes the accepted speed preference to the car-following controller as a modified desired speed, or combines it with the controller output through a validated safety arbiter. The car-following and safety constraints take precedence over the RL preference.
5. The final control authority checks the resulting acceleration against vehicle limits and the jerk implied by the change from the previously applied acceleration over the control interval. Limit jerk to a validated comfort bound during normal operation. If urgent collision avoidance requires exceeding that bound, prioritize safety and log the jerk-limit override.
6. The final applied command is sent to the simulator. Lane-change feasibility and longitudinal safety are reassessed as traffic changes; reject or cancel a maneuver when necessary and supported safely by the simulator.

The current-lane car-following check alone is not sufficient to approve a lane change: assess the target-lane leader and lag vehicle as well, including the braking impact on the lag vehicle. Expose relevant car-following state to the policy observation, such as current headway, relative speed to the current leader, and whether the vehicle is following or braking, when the simulator provides it. Log the policy's requested intent, the safety/controller-adjusted action, jerk-limit adjustments or overrides, and the final applied command separately. Verify LPSim's actual control interfaces before selecting whether speed intent is expressed as desired speed or as a bounded acceleration adjustment.

## 3. Observation and action interface

Use a fixed-size, normalized observation so that vehicles can be processed in batches on the GPU. Include validity masks for missing neighbors and unavailable lanes. Use only information available at the configured information level.

### 3.1 Observation

Recommended ego features:

- Speed, acceleration, lane index normalized by road lane count, and current maneuver phase.
- Current-lane desired speed or speed limit, if available.
- Distance to the next route-relevant turn, ramp, lane drop, or lane addition; encode the lane-level objective (stay, move left, move right, or no immediate objective).
- Road type and relevant control state (for example, signal phase or stop/yield control), when exposed by the simulator.
- A configurable perception range and visibility mask.

For the current lane and each reachable adjacent lane, include the nearest relevant front and rear vehicles:

- Relative longitudinal distance and speed.
- Relative lane and vehicle class, when available and useful.
- Predicted time-to-collision or time gap, computed consistently by the adapter.
- For a candidate gap, its available length and the gap's front/rear safety margins.

Include nearby vehicles with an active lane-change intention or an observed cut-in when those states are available. Include a short summary of lane occupancy or queue length per reachable lane to support queue selection. Pad to a configured maximum neighbor count and use masks; do not treat padded entries as real vehicles.

All continuous features should have documented units, clipping bounds, and normalization constants. Training and inference must use the same preprocessing. Missing data should be represented by masks rather than ambiguous numeric sentinels.

Optional settings:

- **Information range:** expose vehicles and lane summaries only within the configured range. Candidate values from the requirements are 50 m and 200 m; evaluate intermediate values before selecting defaults.
- **Aggressiveness:** encode as a small scalar or category that affects policy preferences and selected safety margins. Keep the hard collision-avoidance checks in place at every setting.

### 3.2 GPU memory organization

Treat road topology as mostly static data and vehicle state/observations as dynamic data. When LPSim exposes GPU-resident state, keep both on the GPU and avoid copying full topology or per-vehicle data between the central processing unit (CPU) and GPU at each decision step. Confirm the simulator's actual storage and API before choosing buffer ownership.

Represent the road as a compact lane graph. A compressed sparse row (CSR) layout can store lane connections using an integer `lane_offsets` array and a contiguous `connected_lane_ids` array. Store left/right adjacency separately if that makes reachability checks simpler. Keep lane attributes such as length, speed limit, road type, and turn/ramp role in parallel arrays. Reuse this topology across decision steps; if environments share a map, share the same topology storage. For different maps, use separate topology buffers or combine graphs with per-environment offsets.

Build observations in fixed-shape, contiguous batches with one row per controlled vehicle. For example, ego features may have shape `[vehicle_batch, ego_feature]`, while neighbor features may have shape `[vehicle_batch, lane_slots, neighbor_slots, feature]`. Keep validity masks alongside the corresponding lane and neighbor dimensions. Pad to configured maximum slot counts and mask padded entries; do not duplicate the full road graph in each vehicle's observation. Use the topology to look up relevant nearby lanes and then construct only the local observation required by the policy.

Use `int32` for lane IDs and offsets unless validated map-size bounds justify another representation, boolean or byte masks, and `float32` for normalized continuous observations as the initial baseline. Consider `float16` or `bfloat16` for policy inference only after checking numerical stability and behavior; safety checks should use a validated precision. Apply identical feature units, clipping, normalization, tensor layout, and masks during training and inference.

Size buffers from the maximum vehicle batch, lane slots, neighbor slots, feature count, and rollout horizon. Profile peak GPU memory and throughput at the intended number of vehicles and parallel environments. Avoid per-vehicle allocations and CPU/GPU synchronization in the decision loop; reuse appropriately sized observation and inference buffers where supported. During Proximal Policy Optimization (PPO) training, include rollout storage, minibatches, model parameters, optimizer state, and simulator state in the memory budget. If the combined footprint exceeds available memory, reduce parallel environments or rollout length, or use minibatching before changing observation semantics.

### 3.3 Action

Use a compact hybrid action:

1. **Lane intent:** stay, request one lane left, request one lane right, or cancel a pending lane-change request. Mask directions that are not topologically available.
2. **Longitudinal intent:** prefer decelerating, maintaining, or accelerating, mapped by the adapter to a bounded desired-speed adjustment or another validated input to the car-following controller.
3. **Maneuver persistence:** if a lane change is pending or active, continue, cancel if safely possible, or defer to the maneuver manager's safe fallback.

The adapter defines and documents speed-adjustment bounds, lane-change duration, and decision timing using the simulator's supported controls. The car-following controller or shared safety arbiter produces the final acceleration command and enforces vehicle and following limits. A rejected lane request must not be silently presented to the policy as a successful lane change. Record both the requested and applied actions.

This action design supports gap search (adjust desired speed while remaining in lane), gap creation (request acceleration or deceleration through the car-following controller when safe), and lane-change cancellation without requiring the RL model to control low-level steering or generate trajectories.

## 4. Policy and safety-layer design

### 4.1 Policy network

Recommended baseline:

- Shared multilayer perceptron with two hidden layers of 128 units and rectified linear unit (ReLU) or sigmoid linear unit (SiLU) activations.
- A categorical lane-intent head, a categorical longitudinal-intent head, and a value head for PPO.
- Optional small conditioning input for aggressiveness and information range.
- A mask over invalid lane intents applied before action sampling.

This is intentionally a small baseline. Use a recurrent layer only if evaluation shows that the observation does not contain enough state to make consistent decisions; first test adding explicit maneuver phase and short-term traffic summaries. Keep the actor and critic architecture and observation preprocessing identical between training and deployment.

### 4.2 Safety and feasibility checks

Use a safety filter based on relative position, relative speed, road geometry, and configurable margins. For each candidate target-lane gap, assess both the front and rear vehicles. Consider predicted separation over the maneuver horizon, not just instantaneous distance. Reject a request if either side violates the configured safety rule, the lane is unavailable, or the maneuver is illegal for the current road state.

The safety filter should:

- Enforce collision-avoidance and legal-lane constraints as hard rules.
- Limit acceleration, deceleration, and normal-operation jerk to simulator-supported and validated bounds; allow safety-critical overrides and log them.
- Avoid creating an unsafe braking demand for the target-lane lag vehicle.
- Prefer waiting and safe gap-search speed adjustments when no gap is acceptable.
- Reassess a requested or active maneuver as the scene evolves and cancel or modify it when feasible and necessary.
- Record the reason for every rejected or canceled action.

Thresholds must be calibrated against simulator dynamics and tested across vehicle types and speeds. They are configuration values, not learned parameters. The policy should be trained with the same filter used at inference.

## 5. Reward design

Use a per-vehicle reward at each policy decision, with team or traffic-level terms added only where needed to discourage harmful external effects. Normalize and log each component separately.

Recommended components:

- **Progress and efficiency:** forward progress and speed relative to a safe desired speed, conditioned on traffic and speed limits.
- **Objective completion:** positive value for a safe, successful lane change that advances a route-level lane objective, passes a slow vehicle, or selects a meaningfully shorter queue.
- **Gap search:** modest value for making progress toward an acceptable gap; do not reward waiting indefinitely or speed changes without an eventual useful outcome.
- **Safety:** strong penalty for collisions and near-collisions; penalties for unsafe time gaps, harsh braking imposed on nearby vehicles, and unsafe lane-change requests rejected by the filter.
- **Comfort:** penalties for excessive acceleration, deceleration, and unnecessary lane changes.
- **Traffic impact:** penalty for avoidable delay or braking imposed on target-lane followers and surrounding traffic; reward safe accommodation of a cut-in or safe gap creation when it benefits another vehicle.
- **Rule compliance:** penalties for illegal maneuvers and route-objective misses attributable to a feasible but ignored lane-change opportunity. Do not penalize a miss when no safe opportunity existed.

Avoid rewarding lane changes by count. Give credit for safe completion and the outcome, not for repeated requests or oscillation. Apply a clear terminal outcome for a canceled maneuver, and distinguish a safety-driven cancellation from a policy-initiated cancellation.

Tune reward scales using component distributions and scenario outcomes, rather than relying only on the aggregate return. Collision avoidance remains the responsibility of the safety layer, regardless of reward weights.

## 6. Training plan

### 6.1 Algorithm and parallelization

Start with **Proximal Policy Optimization (PPO)** using the hybrid categorical action heads and a value function. PPO is a practical baseline for the discrete, repeated lane-change decisions in this design. Train a single shared policy over many simulated vehicles and parallel environments. Batch observations and policy inference on the GPU; use the simulator's GPU path where supported, and avoid CPU/GPU synchronization inside per-vehicle loops.

Keep the simulator adapter, observation builder, safety filter, reward calculation, and episode logging deterministic and separately testable. If the simulator cannot expose enough independent environments or GPU-resident state for efficient rollout, profile before choosing a different algorithm or moving rollout collection to CPU.

### 6.2 Training stages

1. **Adapter and baseline validation**
   - Verify observation units, masks, lane topology, speed-intent bounds, car-following interface, and vehicle identity.
   - Run scripted safe policies and confirm that the safety filter rejects deliberately invalid requests.
   - Check collision, lane-change, cancellation, and reward logging against hand-constructed scenes; verify that RL speed intent is constrained by the car-following controller and that requested, adjusted, and applied actions are logged distinctly.
2. **Basic freeway behavior**
   - Train on ordinary multilane freeway traffic with slow vehicles and both available and blocked adjacent-lane gaps.
   - Include gap search, safe passing, lane keeping, and return to the desired lane when appropriate.
3. **Interaction behavior**
   - Add cut-ins, target-lane lag vehicles, gap creation, and gap accommodation.
   - Train cancellation and recovery behavior as traffic conditions change during pending maneuvers.
4. **Route and network objectives**
   - Add ramps, lane drops/additions, queue selection, intersections, and turning-lane objectives.
   - Ensure the policy receives only lane-level objectives, not responsibility for route planning.
5. **Robustness and optional conditioning**
   - Randomize demand, vehicle speeds and lengths, driver behavior, signal timing, road geometry, and initial positions within plausible simulator ranges.
   - If required, train across aggressiveness and information-range settings, including the 50 m and 200 m examples.
   - Hold out combinations of traffic demand and geometry for evaluation.

Use curriculum progression only after the current stage meets its safety and completion gates. Mix earlier scenarios into later stages to reduce forgetting. Maintain fixed validation seeds and a separate held-out test set.

### 6.3 Domain randomization and episode setup

Randomize the scenario variables that are available and meaningful in LPSim, including traffic density, desired speeds, vehicle dimensions, surrounding-driver responses, route objectives, signal phases, and initial queue states. Use consistent random seeds for comparisons. Ensure training examples include both feasible and infeasible lane changes, near-gap cases, mandatory objectives with limited remaining distance, and traffic changes during maneuvers.

Train with the inference-time safety filter enabled. Track its intervention rate: a policy that appears safe only because most of its actions are rejected has not learned satisfactory behavior.
Use the same car-following controller, controller parameters, and action-arbitration path during training, validation, and deployment. If those change, revalidate and, where the resulting behavior differs materially, retrain or fine-tune the policy.

### 6.4 Checkpoints and reproducibility

Save policy and optimizer checkpoints, configuration, observation/action schema version, reward weights, simulator version or commit, training seeds, and evaluation summaries. Preserve a fixed validation suite and select checkpoints by a documented combination of safety and task performance, not training reward alone. A candidate that fails the safety gate is not eligible for deployment regardless of its average score.

## 7. Evaluation scenarios and metrics

Test the scenarios required by the model requirements:

1. Freeway with on- and off-ramps.
2. Freeway with lane drops.
3. Freeway with lane additions.
4. Signalized intersections with dedicated turning lanes.
5. Signalized intersections with shared turning lanes.
6. Stop-sign-controlled intersections.
7. Yield-sign-controlled intersections.

For each scenario, test multiple traffic demands, route objectives, initial positions, random seeds, and lane-change opportunities. Include cases where a gap is initially unavailable but may become available after a safe speed adjustment, as well as cases where the correct decision is to cancel or decline a maneuver.

### 7.1 Core metrics

Report per scenario, traffic-demand level, and overall, with sample counts and confidence intervals:

| Metric | Definition |
|---|---|
| Collision rate | Collisions per 1,000 controlled-vehicle-kilometers and per 1,000 episodes; report collision severity if available. |
| Near-collision rate | Safety-threshold violations or minimum time-to-collision events per 1,000 vehicle-kilometers; define the threshold before evaluation. |
| Safe lane-change success | Completed lane changes with no collision and no configured safety-margin violation, divided by attempted lane changes. |
| Objective completion | Safe completion of feasible lane objectives (pass, queue selection, turn/ramp positioning, or mandatory lane change) divided by eligible opportunities. |
| Unavailable-opportunity handling | Correctly deferred or canceled maneuvers when no safe gap exists, and successful completion when a safe gap later becomes available. Report both rates. |
| Cancellation response | Unsafe or infeasible pending maneuvers canceled before execution, divided by such opportunities; separately count avoidable cancellations. |
| Target-lane follower impact | Change in lag-vehicle braking, delay, and minimum time gap attributable to controlled-vehicle lane changes, compared with a matched baseline. |
| Gap-search efficiency | Time and distance from first lane-change request to safe completion, plus the fraction of requests abandoned or timed out. |
| Throughput and travel time | Mean and distribution of travel time, flow, and queue delay, compared with a matched baseline. |
| Comfort | Acceleration, deceleration, jerk, and lane-change frequency distributions. |
| Cut-in accommodation | Safe response rate to adjacent-lane cut-ins and rate of collisions or hard braking during those interactions. |
| Gap creation for others | Safe and successful gap-creation opportunities divided by eligible opportunities, with effects on ego and other-vehicle delay. |
| Safety-filter intervention | Rejected, modified, and canceled requests by reason, per 1,000 decisions and per attempted lane change. |
| Runtime | Policy inference throughput, end-to-end decisions per second, GPU memory use, and simulation throughput at target batch sizes. |

Compare against at least a lane-keeping baseline and, if available, the simulator's existing lane-change behavior. Use paired scenario seeds for fair comparisons. Report both means and tail behavior (for example, 95th-percentile travel time and minimum safety gap). Set numerical acceptance thresholds before final evaluation based on the simulator's validated safety limits and project goals; do not infer deployment readiness from aggregate reward.

## 8. Implementation and acceptance gates

1. **Integration gate:** deterministic observations and actions; documented units; repeatable scenario playback; no invalid lane requests.
2. **Safety gate:** safety filter passes unit and scenario tests; no collision in targeted deterministic tests; rejected actions and cancellation reasons are observable.
3. **Behavior gate:** demonstrate all required lane-change classes, safe gap search, cancellation under changing conditions, and cut-in/gap-creation responses in targeted scenarios.
4. **Training gate:** stable PPO training, reproducible checkpoints, validation performance that does not depend on excessive safety-filter intervention.
5. **Generalization gate:** report the full scenario matrix, held-out seeds, traffic-demand ranges, safety metrics, and baseline comparisons.
6. **Performance gate:** show end-to-end GPU throughput and memory at the intended number of controlled vehicles and simulation environments.

Do not begin large-scale training until the adapter, observation schema, safety checks, and metric calculations have passed their targeted tests. Do not deploy a checkpoint that fails a predeclared safety or runtime gate.

## 9. Open configuration decisions

The following values require simulator-specific validation before training:

- Policy decision interval and simulator integration step.
- Neighbor count and perception range.
- Lane-change duration and maneuver phases supported by LPSim.
- Acceleration, deceleration, and jerk bounds.
- Front/rear gap and predicted-separation safety limits.
- Reward scales and objective-priority rules when objectives conflict.
- Traffic demand and domain-randomization ranges.
- Numerical acceptance thresholds for safety, efficiency, and throughput.

Record the selected values and their rationale in the experiment configuration. The source requirements provide no fixed numeric acceptance thresholds, so these must be agreed upon before model comparison.
