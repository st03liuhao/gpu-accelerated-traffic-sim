# Full-intersection model (training env and LPSim)

The reinforcement learning (RL) lane-changing policy is trained and deployed against the
same intersection rules. This
document is the specification. `lcrl/intersection.py` (geometry, conflicts, signal plans),
`lcrl/ix_env.py` (batched training env) and the kernel of LPSim (the Large-scale Parallel Simulator) all implement it, and any
difference between them is a bug.

## 1. Geometry

- **Legs.** An intersection has 3 or more legs. Training uses up to 4 leg slots (a T drops
  one); LPSim nodes may have more. Traffic is right-hand.
- **Centre lines.** Each leg has a centre line: a polyline from the intersection centre
  outward, between its inbound and outbound lanes.
  - It may curve. From OpenStreetMap (OSM) it is the edge shape; otherwise it is straight towards the
    neighbouring node.
  - Training legs point within ±12° of the compass directions. 40% of them curve after
    30–60 m, with a radius of 100–400 m and a turn of 15–40°.
  - Inbound lanes lie on the centre line's counter-clockwise side, looking outward; outbound
    lanes on its clockwise side.
- **Links.** Each leg has an *approach* link (inbound, toward the centre) and an *exit* link
  (outbound).
  - Approach link coordinate `x` runs from 0 (upstream end) to `L_app` (the stop line).
  - Exit link coordinate runs from 0 (the box edge) to `L_exit`.
  - Lane 0 is the leftmost lane, next to the centre line.
- **Lane positions.**
  - Inbound lane `k` of leg `a` lies at arc length `D_a + L_app − x` along the centre line,
    offset `(k+½)·W` to the inbound side (perpendicular to the local tangent).
  - Outbound lane `j` of leg `b` lies at arc length `D_b + x`, offset `(j+½)·W` to the
    outbound side.
  - Lane width `W = 3.5 m`.
  - `D_a` is leg `a`'s **setback**: the distance from the centre to its stop line, computed
    per leg. It clears the corners with both neighbouring legs; a corner is where `a`'s kerb
    meets the neighbour's kerb, given the angle between them. Then add a 4.5 m crosswalk and
    3.5 m of margin.
- **Movements.** Turns are classified from each leg's overall direction, the centre line
  point 40 m out:
  - through within ±35°;
  - left or right beyond that;
  - U-turns (beyond 160°) are not modelled.
- **Lane use** at the stop line is a bitmask over the exit legs. Any number of lanes per leg
  is allowed (up to 8 in training).
  - Exclusive turn lanes: one, two or three left or right lanes. Bays are lanes that exist
    only on the last `bay_len` metres.
  - Shared lanes carry several movements.
  - From OSM, lane use comes from `turn:lanes` when tagged. Otherwise the outer lanes take
    the turns.
  - Exit legs take every lane turning into them side by side.
- **Curve speeds.** On a curved link the desired speed is capped at `sqrt(2 m/s² / curvature)`.
  The cap is lowered ahead of a curve so a vehicle can slow down at 1.5 m/s². The cap profile
  is sampled every 5 m in training and every 10 m in LPSim (`edge_curves.csv`).
- **Paths.** There is one path per (approach lane, exit leg): a cubic Bézier curve from the
  stop line to the start of the exit lane, tangent to both lanes.
  - Exit lane assignment: lefts take the leftmost exit lanes, rights the rightmost, and
    throughs map left to right in order. Dual turns therefore run side by side.
  - The path is sampled at 48 points to get its arc length `L_p`.
- **Conflicts.** For every pair of paths (p, q), the *conflict zone* on p is the arc-length
  interval of p that comes within 2.6 m of q (vehicle width plus margin).
  - **Diverge:** p and q start in the same lane. Not a reservation conflict; the vehicles
    follow each other until they separate.
  - **Merge:** p and q end in the same exit lane.
  - **Crossing:** any other overlap.
- **Crosswalks.** One per leg, located between the box and the stop line or exit. A path
  crosses its start leg's crosswalk on `s ∈ [0.5, 4.5]` and its end leg's crosswalk on
  `s ∈ [L_p−4.5, L_p−0.5]`.

## 2. Control and right-of-way rank

Every path has a **rank** at every step. A vehicle may only enter the box if rank > 0, and it
yields to conflicting traffic of higher rank.

| Control | Path | Rank |
|---|---|---|
| Signal | protected green (G) | 2 |
| | permitted green (g): left turns during the through phase | 1 |
| | amber | 2 for committed vehicles (§3), otherwise 0 |
| | red | 0 |
| | red, right turn, right-turn-on-red allowed, after a full stop | 0.5 |
| All-way stop | after a full stop | 1; ties are served first-come, first-served by stop time |
| Two-way stop | major through / right | 2 |
| | major left | 1.5 |
| | any minor movement, after a full stop | 1; minor approaches take turns, first-come, first-served by stop time |
| Yield | as two-way stop, but minor approaches need no full stop; ties go to whoever arrives first | |

An earlier version ranked minor movements right > through > left (as in the Highway Capacity Manual, HCM). A minor-street
left-turner then waited for every opposing minor-street through vehicle standing at its own stop
line, which almost always existed, and starved for 130 s or more in LPSim tests. Real drivers on
minor approaches take turns.

- **Major and minor legs.** On a 4-leg intersection, E–W is major. On a T, the two collinear
  legs are major and the stem is minor.
- **Signal plans** are phase sequences. Each phase gives a state (R / g / G) to every path.
  Every green phase is followed by amber (3.5 s) and all-red (1.5 s).
  - On a 4-leg intersection, each street pair has a left-turn mode: protected (its own left
    phase, lefts R in the through phase), permitted (no left phase, lefts g), or
    protected+permitted. An approach with two or more left-turn lanes is always protected.
  - On a T, there is a major-street phase (thru/right G, the left into the stem g or
    protected) and a stem phase.

## 3. Entering the box (decided at the stop line)

The head vehicle `i` of an approach lane, with path `p` and rank `r > 0`, is **granted entry**
when all of the following hold:

1. **Exit space.** The exit lane of `p` has room (`x_last ≥ veh_len + s0`).
2. **Crosswalks.** Neither crosswalk on `p` is occupied by pedestrians.
3. **No higher-priority conflict.** For every path `q` conflicting with `p` (not a diverge),
   and every vehicle `j` that will use `q`:
   - `j` is in the box and has not cleared the zone: it blocks if it is inside the zone, or
     if its estimated time of arrival (ETA) `ETA_j(q_in) < t_clear_i + h_i`;
   - `j` is approaching on q (not yet granted) within **250 m** of its stop line and has
     rank > r. It blocks unless the gap is large enough for `i`'s movement
     (PET: post-encroachment time):
     `ETA*_j(q_in) ≥ max(g_crit(mov_i) + c_i, t_clear_i + PET)`;
   - same rank at an all-way or two-way stop: `j` blocks if it is a head vehicle that
     stopped earlier (ties broken by leg index);
   - same rank elsewhere (yield, permitted signal movements): `j` within 80 m blocks if it
     reaches the zone first (`ETA_j < ETA_i`) and `ETA_j < t_clear_i + h_i`.

Definitions:
- `t_clear_i` is the time for `i` to travel from its position to `q_out + veh_len`,
  accelerating at `a_max`.
- `ETA_j` is the time for `j` to cover the distance, accelerating at `a_max` with no speed
  limit. This is a conservative (early) arrival estimate.
- `ETA*_j` is the same, but capped at `max(desired speed_j, v_j)`. This is a realistic
  arrival time for traffic that has priority.
- `g_crit` is the critical gap of the movement: left 4.0 s, through 5.5 s, right (including
  right turn on red) 5.0 s.
- `PET = 1 s` is the minimum post-encroachment time after `i` has cleared the zone.
- `c_i = 1 − aggressiveness` adds up to 1 s for cautious drivers.
- `h_i = 1.0 + 1.0·(1 − aggressiveness)` s is the safety gap used against committed and
  same-rank vehicles.

The rule is re-evaluated at every 0.1 s physics step.

**Permitted turns pull into the box.** Real drivers making a permitted turn drive past the
stop bar and wait for a gap.
- **Staging point.** Each permitted path gets one when it is built: just before (0.5 m) its
  first conflict zone with traffic that is green in the same phase. Its number of places is
  how many vehicles fit between the stop bar and that point, at most 2.
- **Staging.** A head vehicle with no acceptable gap may *stage* (enter and wait at the
  staging point) when all of these hold:
  - its movement is permitted now;
  - a place is free;
  - no committed vehicle is crossing the zones before the staging point;
  - its start crosswalk is free and the exit has room.
- **While it waits:**
  - it does not hold back traffic with priority; it blocks only zones it physically
    occupies;
  - vehicles behind it in its lane cannot be granted entry, even on amber; the next vehicle
    may only take the next place on the same path.
- **Release.** At every step, the staged vehicle re-checks entry from its own position, with
  the gap rule above.
- **After its phase.** Once its phase has ended, it has priority to clear: it then only
  waits for committed vehicles. Cross traffic cannot enter the zones it is in or about to
  cross.
- **Inside a zone.** A vehicle that is already inside a conflict zone has priority to clear
  it, and other vehicles' arrival times for that zone do not hold it back.

These rules came from deadlocks found in testing: staged turners waiting on through vehicles
that were stuck behind the opposite staged turner, and amber grants behind a staged turner.
`tests/test_ix.py::test_staged_permitted_turns_do_not_deadlock` covers them.

**Commitment.** A vehicle that is granted entry, or is in the amber commitment zone
(cannot stop at 3.5 m/s²), is committed. A committed vehicle never re-checks rank.
Uncommitted vehicles see the stop line as a wall. A vehicle that was not granted entry but
crosses anyway is a violation.

**Lane mismatch.** A vehicle whose lane does not allow its intended movement takes an
allowed movement (preferring through) and is counted as *missed*.

## 4. In the box (rule-driven, not RL)

- **Car following.** The Intelligent Driver Model (IDM) plus the Krauss safety shield along the path.
- **Leaders:** vehicles ahead on the same path; vehicles on diverging paths that are still
  inside the shared start segment; and the last vehicle of the exit lane (chained gap).
- **Safety backup:** a wall at `p_in` if a vehicle on a conflicting path is inside its zone,
  or will reach it first (smaller ETA, ETA < 2 s; ties broken by vehicle index).
- **Collision:** two vehicles inside their zones of a conflicting pair at the same time, or
  any overlap on the same path or lane.

## 5. Pedestrians

- Each crosswalk has a Poisson pedestrian demand (peds/h).
- **Signalized:** pedestrians may start crossing only during the first 7 s of the phase that
  serves the parallel street's through movement. The crosswalk across leg E or W walks with
  the N–S through phase, and vice versa.
- **Unsignalized:** waiting pedestrians start when no vehicle is inside that crosswalk's zone.
- A crossing occupies the crosswalk for `road_width / 1.2 m/s`.
- An occupied crosswalk is a hard wall for every path that crosses it.

## 6. What the RL policy controls

The policy acts on approach links and exit links: lane choice and the acceleration offset,
exactly as on freeway links. Box traversal and entry permission follow the rules above.

Once a vehicle has been granted entry it has accepted its gap and commits: its acceleration
offset cannot be negative, so it cannot brake below IDM. Without this, granted vehicles
sometimes stood still at the stop line in LPSim, and the rest of the junction waited for them
as committed traffic.

The policy's ego features include:
- the control category (free, signal, all-way stop, two-way-stop major, two-way-stop minor,
  yield minor);
- its movement's current rank;
- whether it is blocked;
- the time until the signal state changes;
- whether it has already stopped.

The per-lane queue features let it see, for example, permitted left-turners waiting for a gap.

## 7. The 4-leg signal study design

The `signal_4leg` kind is a fixed design for studying permitted left turns and right turn on
red (`lcrl.intersection.signal_4leg`). Unlike the other kinds, it has no random geometry or
timing.

- **Legs.** Four legs of 60 m, both the approach and the exit link.
- **Lanes.** Two lanes per approach in evaluation and in the viewer; training also draws
  one-lane approaches. Turning and through traffic share lanes:
  - the left lane serves left and through;
  - the right lane serves through and right;
  - with one lane, that lane serves every movement.

  There are no turn bays.
- **Signal.** The same plan for the baseline and the RL policy:

  | Phase | Duration | What moves |
  |---|---|---|
  | E–W green | 30 s | both directions; through and right protected, left permitted |
  | Amber, then all red | 3.5 s + 1.5 s | |
  | N–S green | 30 s | the same movements for N and S |
  | Amber, then all red | 3.5 s + 1.5 s | |

  This gives a 70 s cycle. Right turn on red is allowed. Evaluation and the viewer start
  every run at the beginning of the E–W green (offset 0).
- **Pedestrians.** None.
- **Demand.** Set per leg, from 50 to 800 veh/h. The left share is below 10% and the right
  share below 15%:
  - evaluation and the viewer use 8% left, 80% through and 12% right;
  - training draws 3–10% left and 5–15% right.

  Vehicles enter on a lane that serves their movement, and through traffic is split evenly
  over its lanes. Lane changes are therefore discretionary, for example to get around a
  left-turner waiting for a gap. Demand has no per-episode multiplier.
