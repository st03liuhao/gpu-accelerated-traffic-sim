# GPU-Accelerated RL Lane-Changing Model: Technical Specification

## 1. Purpose and relationship to the design document

This specification formalizes the proposed model structure, policy inputs and outputs, safety arbitration, reward, and Proximal Policy Optimization (PPO) training procedure for the graphics processing unit (GPU)-accelerated reinforcement learning (RL) lane-changing model. It supplements the [design and training document](./GPU-accelerated%20lane-changing%20RL%20model%20design%20and%20training.md), which remains the source for scenario coverage, implementation stages, and evaluation metrics.

The equations below define a practical baseline, not simulator-validated constants. The exact state and control interfaces of LPSim (the Large-scale Parallel Simulator), vehicle dynamics, and supported lane-change behavior must be confirmed before implementation. Thresholds, scales, and bounds identified as parameters must be calibrated against those dynamics.

The model does not plan routes or trajectories. It receives lane-level route intent from the simulator and chooses lane-change and longitudinal **intent**. The simulator's car-following controller and a deterministic maneuver/safety manager determine whether and how the intent is applied.

## 2. Problem formulation

Let $t$ index RL decision steps, separated by $\Delta t_\pi$ seconds. At decision $t$, let $s_t$ denote the joint traffic and road state, including vehicle states, lane topology, active maneuvers, and route-level lane objectives. Each controlled vehicle $i$ receives a local observation $o_t^i = O_i(s_t)$, samples a policy action, and receives a reward $r_t^i$. The simulator transitions according to:

$$
s_{t+1} \sim P(s_{t+1}\mid s_t, a_t^1,\dots,a_t^M),
$$

where $M$ is the number of controlled vehicles and the actual simulator action $a_t^i$ may differ from the requested action after safety arbitration.

This is a partially observed multi-agent control problem. The baseline uses decentralized execution with parameter sharing:

$$
a_t^i \sim \pi_\theta(\cdot\mid o_t^i),\quad i=1,\ldots,M.
$$

Every vehicle uses the same policy parameters $\theta$, but receives its own observation and produces its own action. The policy is feed-forward; any state needed across decisions, such as a pending lane-change phase or persistent driver style, must be included explicitly in $o_t^i$.

## 3. Observation representation

### 3.1 Fixed-shape per-vehicle feature vector

Construct a fixed-size observation from ego features, lane context, nearby vehicles, and optional conditioning variables:

$$
o_t^i =
\left[
e_t^i,\;
\ell_t^i,\;
n_t^i,\;
m_t^i,\;
c^i
\right].
$$

Here:

- $e_t^i\in\mathbb{R}^{F_e}$ contains ego state, for example speed, acceleration, lane index, desired speed, maneuver phase, and current-lane leader headway/relative speed.
- $\ell_t^i\in\mathbb{R}^{K\times F_\ell}$ contains features for up to $K$ relevant lanes, including lane availability, lane-level objective, queue summary, and candidate gap information.
- $n_t^i\in\mathbb{R}^{K\times N\times F_n}$ contains up to $N$ front/rear or otherwise relevant neighbors per lane, with relative distance and speed, vehicle class, and lane-change intent when available.
- $m_t^i$ contains validity masks for unavailable lanes and padded neighbor slots.
- $c^i$ optionally contains persistent aggressiveness and information-range conditioning.

The observation builder selects only features within the configured perception range. Use normalized continuous features and explicit masks; padded values must not be interpreted as real traffic. Apply the same ordering, units, clipping, normalization, and masks during training and deployment.

For a scalar feature $x$, a typical normalization is:

$$
\tilde{x} = \operatorname{clip}\left(\frac{x-\mu_x}{\sigma_x+\epsilon},-c,c\right),
$$

where $\mu_x,\sigma_x,c$ are fixed from training data or defined by physical bounds. Store these values with the checkpoint. Categorical variables should use one-hot or learned embeddings, not an arbitrary continuous ordering.

### 3.2 Static topology and GPU tensors

Store lane topology as persistent integer arrays, such as compressed sparse row (CSR) offsets and connected-lane IDs, with lane attributes in parallel arrays. Keep static topology separate from dynamic vehicle state. Where supported by LPSim, retain both on the GPU and construct contiguous observation batches without per-vehicle host-device copies.

For a batch of $B$ vehicles, representative tensors are:

$$
E\in\mathbb{R}^{B\times F_e},\quad
L\in\mathbb{R}^{B\times K\times F_\ell},\quad
N\in\mathbb{R}^{B\times K\times N_{\max}\times F_n},
$$

with corresponding boolean or byte masks. The observation builder may flatten the feature tensors after masking:

$$
X=\operatorname{flatten}(E,L,N,m,c)\in\mathbb{R}^{B\times F}.
$$

Use integer lane IDs/offsets and masks, and begin with float32 normalized observations. Lower-precision policy inference is an optimization to benchmark, not an assumption for safety calculations. Safety-critical gap checks should use a validated numerical precision. Profile memory for simulator state, observation buffers, PPO rollout storage, minibatches, model parameters, and optimizer state together.

## 4. Policy network and action distribution

### 4.1 Shared feed-forward network

The baseline actor-critic uses a shared multilayer perceptron (MLP) trunk with two hidden layers of 128 units:

$$
h_1 = \phi(W_1 X+b_1),\qquad
h_2 = \phi(W_2 h_1+b_2),
$$

where $\phi$ is rectified linear unit (ReLU) or sigmoid linear unit (SiLU). Each row of $X$ represents one vehicle. The policy has separate lane-intent and speed-intent heads, and a value head:

$$
z^{\text{lane}}=W_{\text{lane}}h_2+b_{\text{lane}},\qquad
z^{\text{speed}}=W_{\text{speed}}h_2+b_{\text{speed}},\qquad
V_\psi(o)=W_Vh_2+b_V.
$$

The actor parameters are denoted by $\theta$; the critic parameters are denoted by $\psi$. A shared trunk may be used for both, or separate actor/critic trunks may be compared if value learning interferes with policy learning. Keep the deployed actor and its observation preprocessing consistent with training.

### 4.2 Factorized hybrid action

Use two categorical action components:

$$
a_t^i=(a_{L,t}^i,a_{S,t}^i),
$$

where:

- $a_L\in\{\text{stay},\text{left},\text{right},\text{cancel}\}$ is lane intent.
- $a_S\in\{\text{decelerate},\text{maintain},\text{accelerate}\}$ is longitudinal intent.

The action probability is factorized as:

$$
\pi_\theta(a\mid o)=
\pi_\theta(a_L\mid o)\,
\pi_\theta(a_S\mid o).
$$

Mask unavailable or illegal lane directions before softmax. For lane logits $z_j$ and validity mask $q_j\in\{0,1\}$:

$$
\pi(a_L=j\mid o)=
\frac{q_j\exp(z_j)}
{\sum_k q_k\exp(z_k)}.
$$

At least one lane action (normally stay) must remain valid. Maneuver persistence and phase are managed using explicit observation state and the maneuver manager; do not allow contradictory independent action heads. The lane-change request may be canceled only when cancellation is safe and supported by the simulator.

Map speed intent to a bounded desired-speed offset:

$$
\delta v_{\text{RL}}\in\{-\Delta v,0,+\Delta v\},\qquad
v_{\text{des}}=
\operatorname{clip}(v_{\text{CF}}+\delta v_{\text{RL}},
v_{\min},v_{\max}),
$$

where $v_{\text{CF}}$ is the car-following (CF) controller's nominal desired speed and $\Delta v,v_{\min},v_{\max}$ are calibrated values. If LPSim exposes a better validated interface, such as acceleration preference, use that instead and document the mapping.

## 5. Car-following integration and final control authority

The RL action is a request, not the final simulator command. At each control update, compute a nominal car-following acceleration $a_{\text{CF}}$ using the simulator's existing controller and its current-lane leader. The RL speed preference may adjust the car-following desired speed, but must not bypass its following logic. The maneuver manager separately evaluates target-lane front and rear traffic.

The final control authority produces applied acceleration $a_t$ subject to physical acceleration/deceleration limits and normal-operation jerk limits. Given the previous applied acceleration $a_{t-1}$, define raw acceleration $a_t^{\text{raw}}$ from the car-following controller and safety arbiter. The comfort-limited command is:

$$
a_t^{\text{comfort}} =
\operatorname{clip}\left(
a_t^{\text{raw}},
a_{t-1}-J_{\max}\Delta t_c,\;
a_{t-1}+J_{\max}\Delta t_c
\right),
$$

$$
a_t =
\operatorname{clip}(a_t^{\text{comfort}},a_{\min},a_{\max}),
$$

where $\Delta t_c$ is the control update interval, $J_{\max}$ is the validated jerk bound, and $a_{\min},a_{\max}$ are vehicle limits. The implied jerk is $j_t=(a_t-a_{t-1})/\Delta t_c$. During urgent collision avoidance, safety may override the comfort jerk bound; log the override and retain physical command limits supported by the vehicle model.

The applied command sequence must have one authority. Do not have the RL policy and car-following controller independently overwrite acceleration. Log the policy request, car-following output, safety-filter decision, jerk-limited output, and final applied command.

## 6. Lane-change feasibility filter

For a candidate adjacent lane, identify the target-lane leader $F$ and follower $R$. In a longitudinal coordinate increasing in the direction of travel, let $x_E,v_E$ be ego position and speed, and $x_F,v_F,x_R,v_R$ be the target-lane leader and follower state. For a lane-change (LC) screening horizon $T_{\text{LC}}$, constant-velocity predicted separations are:

$$
d_F(\tau)=x_F-x_E+(v_F-v_E)\tau,
$$
$$
d_R(\tau)=x_E-x_R+(v_E-v_R)\tau,\qquad
0\le \tau\le T_{\text{LC}}.
$$

A simple time-headway screening requirement is:

$$
d_F(\tau)\ge d_0+h_E v_E(\tau),\qquad
d_R(\tau)\ge d_0+h_R v_R(\tau),
$$

for the relevant points in the maneuver horizon, where $d_0$ is a standstill margin and $h_E,h_R$ are calibrated headway parameters. The rear constraint protects the target-lane follower; its required gap can be extended to account for relative closing speed, braking capability, and predicted follower deceleration.

This constant-velocity calculation is only a screening approximation. The implemented safety filter should use simulator-supported vehicle dynamics or a validated prediction envelope, check legal lane connectivity, and account for uncertainty and maneuver duration. Reject or defer a lane change if the constraints fail. If traffic changes after a request, re-evaluate and cancel an unstarted maneuver or use the simulator's safest supported active-maneuver response. Hard safety rules are not replaced by a reward penalty.

## 7. Reward definition

Define a scalar per-decision reward as a weighted sum of interpretable terms:

$$
r_t^i =
w_p r_{\text{progress}}
+w_o r_{\text{objective}}
+w_g r_{\text{gap}}
-w_c C_{\text{collision}}
-w_n C_{\text{near}}
-w_f C_{\text{follower}}
-w_h C_{\text{harsh}}
-w_j C_{\text{jerk}}
-w_\ell C_{\text{lane-change}}
-w_q C_{\text{filter}}.
$$

Terms are measured or clipped to documented ranges before weighting. Example definitions include:

$$
r_{\text{progress}} =
\frac{x_{t+1}-x_t}{v_{\text{ref}}\Delta t_\pi},
\quad
C_{\text{jerk}} =
\min\left(\left|\frac{a_t-a_{t-1}}{\Delta t_c}\right|,j_{\text{clip}}\right),
$$

$$
C_{\text{follower}} =
\max(0,\; b_R-b_{\text{comfort}}),
$$

where $b_R$ is braking demand imposed on the target-lane follower as estimated by the simulator or a calibrated counterfactual, and $b_{\text{comfort}}$ is a comfort threshold. Objective reward is granted for safe completion of an eligible objective, not for issuing a request. Gap-search reward may reflect progress toward a feasible gap but should not reward indefinite waiting or unnecessary speed oscillation.

Collision and hard safety constraints remain the responsibility of the safety layer. A filter rejection may carry a learning cost for repeated unsafe requests, but policy optimization must never trade that penalty against permission to violate the filter. Log reward components independently and tune weights from scenario outcomes and component distributions.

## 8. PPO training equations

Collect rollout tuples $(o_t^i,a_t^i,r_t^i,o_{t+1}^i,d_t^i)$, where $d_t^i$ indicates episode termination. Store the action sampled by the policy and the corresponding log probability. Distinguish that requested action from the safety-adjusted action applied by the simulator.

For current policy $\pi_\theta$ and behavior policy $\pi_{\theta_{\text{old}}}$, define the likelihood ratio:

$$
\rho_t(\theta)=
\frac{\pi_\theta(a_t\mid o_t)}
{\pi_{\theta_{\text{old}}}(a_t\mid o_t)}
=\exp\left(
\log\pi_\theta(a_t\mid o_t)-
\log\pi_{\theta_{\text{old}}}(a_t\mid o_t)
\right).
$$

For factorized lane and speed actions, the joint log probability is the sum of the two categorical log probabilities.

Estimate generalized advantages using temporal-difference (TD) residuals:

$$
\delta_t=r_t+\gamma(1-d_t)V_\psi(o_{t+1})-V_\psi(o_t),
$$
$$
\hat A_t=\sum_{l=0}^{T-t-1}(\gamma\lambda)^l
\left(\prod_{k=0}^{l-1}(1-d_{t+k})\right)\delta_{t+l}.
$$

The PPO clipped actor objective is:

$$
L^{\text{clip}}(\theta)=
\mathbb{E}_t\left[
\min\left(
\rho_t(\theta)\hat A_t,\;
\operatorname{clip}(\rho_t(\theta),1-\epsilon,1+\epsilon)\hat A_t
\right)
\right].
$$

Use a value loss and entropy regularizer:

$$
L_V(\psi)=\mathbb{E}_t[(V_\psi(o_t)-\hat R_t)^2],
\quad
L_H(\theta)=\mathbb{E}_t[\mathcal H(\pi_\theta(\cdot\mid o_t))].
$$

One minimization convention for the total loss is:

$$
L(\theta,\psi)=
-L^{\text{clip}}(\theta)
+c_V L_V(\psi)
-c_H L_H(\theta),
$$

where $\hat R_t=\hat A_t+V_{\psi_{\text{old}}}(o_t)$. Normalize advantages within a minibatch if empirically stable. Tune $\gamma,\lambda,\epsilon,c_V,c_H$, rollout length, minibatch size, and learning rate through validation; these are not specified by the source requirements.

Train over many vehicles and parallel environments, batching policy inference and learning on GPU where supported. The shared policy update pools valid vehicle transitions. To avoid a few dense scenes dominating, consider normalizing or weighting contributions by scenario and environment, and report how samples are weighted. Keep the safety filter and car-following controller active during rollout collection.

## 9. Training and validation procedure

1. Validate units, masks, action mapping, car-following behavior, lane topology, and deterministic replay on hand-built scenes.
2. Verify target-lane gap checks, rejection reasons, maneuver cancellation, acceleration bounds, jerk limiting, and emergency override logging.
3. Begin training with basic multilane freeway passing and gap-search scenarios; add cut-ins and target-lane follower effects.
4. Add ramps, lane drops/additions, intersection lane objectives, queue selection, and mandatory changes with limited distance.
5. Randomize traffic demand, driver response, vehicle properties, signal/control conditions, and initial states within supported ranges; hold out seeds and scenario combinations.
6. If style conditioning is enabled, sample a persistent style value per vehicle/episode and include it in all that vehicle's observations. Do not resample style each decision unless time-varying style is an explicit modeling objective.
7. Select checkpoints using safety gates first, then objective completion, traffic impact, comfort, and efficiency. Track the safety-filter intervention rate and jerk-limit overrides.

Use the same simulator configuration, observation schema, car-following controller, safety filter, action arbiter, and jerk constraints in training, validation, and deployment. Changes to these components require revalidation and may require fine-tuning.

## 10. Parameters requiring calibration

The following are design parameters rather than universal constants:

- Policy decision interval $\Delta t_\pi$ and car-following control interval $\Delta t_c$.
- Feature bounds, normalization statistics, lane/neighbor slot counts, and perception range.
- Speed intent magnitude $\Delta v$, desired-speed bounds, and policy action timing.
- Lane-change duration $T_{\text{LC}}$, standstill gap $d_0$, headways $h_E,h_R$, and prediction uncertainty margins.
- Physical acceleration bounds $a_{\min},a_{\max}$, normal jerk limit $J_{\max}$, and emergency override rules.
- Reward weights, reward clipping/scaling, PPO hyperparameters, training randomization ranges, and evaluation acceptance thresholds.

Record calibrated values, rationale, simulator version, observation/action schema version, and random seeds in the experiment configuration and checkpoint metadata.
