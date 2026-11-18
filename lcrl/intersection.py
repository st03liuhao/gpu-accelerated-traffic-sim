"""Intersection designs: geometry, turning paths, conflict zones, signal plans, crosswalks.

Implements sections 1, 2 and 5 of docs/INTERSECTIONS.md. ``build`` turns a list of legs into a
design. Legs may have any direction, any centre-line shape (curves), any number of lanes, and
several left- or right-turn lanes. ``design`` draws random designs for training.
``tools/lpsim_intersections.py`` calls ``build`` for LPSim network nodes, so training and
deployment share this code. Everything is plain numpy and runs once per design.
"""
import numpy as np

from .config import MOV_LEFT, MOV_RIGHT, MOV_THROUGH

W = 3.5                 # lane width (m)
NLEG = 4                # training designs: 4 leg slots (a T uses 3)
KMAX = 8                # approach lanes per leg (e.g. 2 left + 4 through + 2 right)
PMAX = 48               # paths per intersection
NPH = 12                # signal phases incl. amber / all-red
CONFLICT_DIST = 2.6     # path centre-lines closer than this conflict (m)
NSAMP = 48
CROSSWALK = (0.5, 4.5)  # crosswalk span measured from each path end (m)
THROUGH_DEG, UTURN_DEG = 35.0, 160.0
CLASSIFY_S = 40.0       # a leg's direction for turn classes: centre -> centre-line point 40 m out
A_LAT = 2.0             # comfortable lateral acceleration on curves (m/s^2)
B_CURVE = 1.5           # deceleration used to anticipate a curve (m/s^2)
VCAP_DS = 5.0           # curve speed profile: one sample every 5 m of link
VCAP_N = 241            # samples per link (1200 m; beyond that the last sample holds)
VCAP_FREE = 99.0        # no curve constraint
STAGE_GAP = 0.5         # a staged permitted turner stops this far before the first conflict zone (m)
STAGE_SPACING = 5.0 + 1.38 + 0.62   # vehicle length + jam gap (+ margin) between staged vehicles
STAGE_MAX = 2           # at most this many vehicles wait inside the box per path
L_, T_, R_ = 1 << MOV_LEFT, 1 << MOV_THROUGH, 1 << MOV_RIGHT
CLASS_OFFSET = {MOV_THROUGH: 2, MOV_RIGHT: 1, MOV_LEFT: 3}   # exit leg offsets on an orthogonal design

# control types (per intersection)
IX_SIGNAL, IX_AWSC, IX_TWSC, IX_YIELD = 1, 2, 3, 4
# signal states per path
SIG_R, SIG_PERM, SIG_G, SIG_Y = 0, 1, 2, 3
# conflict kinds
C_NONE, C_CROSS, C_MERGE, C_DIVERGE = 0, 1, 2, 3

KINDS = {  # name: (control, legs)
    "signal_4leg": (IX_SIGNAL, 4), "signal_tee": (IX_SIGNAL, 3), "all_way_stop": (IX_AWSC, 4),
    "two_way_stop_4leg": (IX_TWSC, 4), "two_way_stop_tee": (IX_TWSC, 3), "yield_4leg": (IX_YIELD, 4),
}


def unit_deg(deg):
    t = np.radians(deg)
    return np.array([np.cos(t), np.sin(t)])


def unit(a):
    """Unit vector of orthogonal leg a (0 = E, 1 = N, 2 = W, 3 = S)."""
    return unit_deg(90.0 * a)


def rnormal(d):
    """Right-hand normal of direction d."""
    return np.array([d[1], -d[0]])


def _angle(v):
    return float(np.degrees(np.arctan2(v[1], v[0])))


def turn_class(angle_in, angle_out):
    """Movement class from approach leg angle to exit leg angle (degrees, outward directions):
    MOV_LEFT / MOV_THROUGH / MOV_RIGHT, or -1 for a U-turn."""
    h_in, h_out = -unit_deg(angle_in), unit_deg(angle_out)
    ang = np.degrees(np.arctan2(h_in[0] * h_out[1] - h_in[1] * h_out[0], h_in @ h_out))
    if abs(ang) > UTURN_DEG:
        return -1
    if abs(ang) <= THROUGH_DEG:
        return MOV_THROUGH
    return MOV_LEFT if ang > 0 else MOV_RIGHT


def _bezier(p0, d0, p1, d1, n=NSAMP):
    c = 0.45 * np.linalg.norm(p1 - p0)
    t = np.linspace(0, 1, n)[:, None]
    a, b = p0 + c * d0, p1 - c * d1
    pts = (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * a + 3 * (1 - t) * t ** 2 * b + t ** 3 * p1
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    return pts, np.concatenate([[0.0], np.cumsum(seg)])


# ---------------------------------------------------------------- leg centre lines
class Centreline:
    """A leg's centre line (between its inbound and outbound lanes) from the intersection centre
    outward, as a polyline with arc length. Inbound lanes lie on its left (counter-clockwise)
    side looking outward, outbound lanes on its right, as for right-hand traffic."""

    def __init__(self, shape=None, angle=0.0, length=5000.0):
        if shape is None:
            pts = np.array([[0.0, 0.0], unit_deg(angle) * length])
        else:
            pts = np.asarray(shape, float).reshape(-1, 2)
            keep = np.concatenate([[True], np.linalg.norm(np.diff(pts, axis=0), axis=1) > 1e-6])
            pts = pts[keep]
            if len(pts) < 2:
                pts = np.array([pts[0], pts[0] + unit_deg(angle)])
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        if s[-1] < length:                      # continue straight beyond the given shape
            t = (pts[-1] - pts[-2]) / max(seg[-1], 1e-9)
            pts = np.vstack([pts, pts[-1] + t * (length - s[-1])])
            s = np.append(s, length)
        self.pts, self.s = pts, s

    def at(self, s):
        """Point and outward unit tangent at arc length s (array or scalar)."""
        s = np.clip(np.asarray(s, float), 0.0, self.s[-1])
        i = np.clip(np.searchsorted(self.s, s, side="right") - 1, 0, len(self.s) - 2)
        seg = self.pts[i + 1] - self.pts[i]
        ln = np.maximum(self.s[i + 1] - self.s[i], 1e-9)
        t = ((s - self.s[i]) / ln)[..., None]
        p = self.pts[i] + t * seg
        return p, seg / ln[..., None]

    def lane_point(self, s, offset, inbound):
        """Point at arc length s, ``offset`` metres to the inbound (left) or outbound (right) side."""
        p, t = self.at(s)
        n = np.stack([-t[..., 1], t[..., 0]], -1) if inbound else np.stack([t[..., 1], -t[..., 0]], -1)
        return p + n * np.asarray(offset)[..., None] if np.ndim(offset) else p + n * offset

    def heading(self, s):
        _, t = self.at(s)
        return np.degrees(np.arctan2(t[..., 1], t[..., 0]))


def _curve_cap(cl, s0, s1, L):
    """Speed cap profile (m/s) for travel along the centre line from arc s0 to s1 over a link of
    length L (sampled every VCAP_DS m of link): sqrt(A_LAT / curvature), lowered ahead of
    curves so a vehicle can slow down at B_CURVE."""
    x = np.arange(VCAP_N) * VCAP_DS
    s = s0 + (s1 - s0) * np.clip(x, 0.0, L) / max(L, 1e-9)
    grid = np.arange(0.0, max(s0, s1) + 25.0, 1.0)
    h = np.unwrap(np.radians(cl.heading(grid)))
    win = 10.0                                    # curvature over +-10 m of arc
    kap = np.abs(np.interp(grid + win, grid, h) - np.interp(grid - win, grid, h)) / (2 * win)
    kappa = np.interp(s, grid, kap)
    cap = np.where(kappa > 1e-5, np.sqrt(A_LAT / np.maximum(kappa, 1e-5)), VCAP_FREE)
    cap = np.minimum(cap, VCAP_FREE)
    for k in range(VCAP_N - 2, -1, -1):           # anticipate: reachable by braking at B_CURVE
        cap[k] = min(cap[k], np.sqrt(cap[k + 1] ** 2 + 2 * B_CURVE * VCAP_DS))
    return cap.astype(np.float32)


def _setbacks(present, box_angle, n_in, n_out):
    """Stop-line distance from the centre for each leg: clear the kerbs of both neighbouring
    legs (a corner between leg a and its counter-clockwise neighbour b is where a's inbound
    kerb meets b's outbound kerb), plus the crosswalk and a margin."""
    NL = len(present)
    idx = [a for a in range(NL) if present[a]]
    order = sorted(idx, key=lambda a: box_angle[a] % 360.0)
    D = np.zeros(NL)
    for i, a in enumerate(order):
        for b, ccw in ((order[(i + 1) % len(order)], True), (order[i - 1], False)):
            if b == a:
                continue
            th = np.radians(((box_angle[b] - box_angle[a]) % 360.0) if ccw else ((box_angle[a] - box_angle[b]) % 360.0))
            th = np.clip(th, np.radians(25.0), np.radians(155.0))
            w_a = (n_in[a] if ccw else n_out[a]) * W        # a's kerb facing b
            w_b = (n_out[b] if ccw else n_in[b]) * W        # b's kerb facing a
            D[a] = max(D[a], (w_b + w_a * np.cos(th)) / np.sin(th))
    return np.maximum(D, 0.0) + CROSSWALK[1] + 1.5 + 2.0


def build(legs, ctrl, *, major=None, left_mode=None, rtor=True, ped_rate=None, green=None,
          kmax=KMAX, pmax=PMAX):
    """Design of an intersection from its legs.

    legs: list (one entry per leg slot, ``None`` = no leg) of dicts with
        angle    outward direction of the leg at the intersection (degrees, counter-clockwise
                 from east); optional when ``shape`` is given
        shape    optional centre-line polyline [(x, y), ...] from the intersection centre outward
                 (metres); curves are kept, and the shape is extended straight beyond its end
        lanes    approach lanes left -> right as (start_m, end_m, exit_mask) where exit_mask
                 is a bitmask over leg slots this lane may turn into; any number of lanes may
                 serve the same exit (dual / triple turn lanes)
        n_out    number of exit lanes
        L_app, L_exit   link lengths (m)
    ctrl: IX_SIGNAL / IX_AWSC / IX_TWSC / IX_YIELD.
    major: per-leg bool (two-way stop / yield priority); default: the widest street.
    left_mode(street_legs) -> "protected" | "permitted" | "prot_perm" chooses signal left-turn
        treatment (several left-turn lanes from one approach are always protected);
        green(kind) -> seconds for "left" / "through" / "stem" phases.
    """
    NL = len(legs)
    present = [lg is not None for lg in legs]
    n_in = np.array([len(lg["lanes"]) if lg else 0 for lg in legs])
    n_out = np.array([lg["n_out"] if lg else 0 for lg in legs])
    L_app = np.array([lg["L_app"] if lg else 0.0 for lg in legs], np.float32)
    L_exit = np.array([lg["L_exit"] if lg else 0.0 for lg in legs], np.float32)
    assert n_in.max() <= kmax, "too many approach lanes"
    cls_ = [Centreline(lg.get("shape"), lg.get("angle", 0.0)) if lg else None for lg in legs]
    box_angle = np.array([cls_[a].heading(0.5 * min(5.0, cls_[a].s[1])) if present[a] else 0.0 for a in range(NL)])
    far_angle = np.array([_angle(cls_[a].at(CLASSIFY_S)[0]) if present[a] else 0.0 for a in range(NL)])
    cls = np.full((NL, NL), -1, int)
    for a in range(NL):
        for b in range(NL):
            if present[a] and present[b] and a != b:
                cls[a, b] = turn_class(far_angle[a], far_angle[b])
    lane_start = np.full((NL, kmax), 1e9, np.float32)
    lane_end = np.full((NL, kmax), -1e9, np.float32)
    lane_exit = np.zeros((NL, kmax), np.int64)
    lane_mov = np.zeros((NL, kmax), np.int64)
    for a in range(NL):
        for k, (s, e, mask) in enumerate(legs[a]["lanes"] if present[a] else []):
            lane_start[a, k], lane_end[a, k], lane_exit[a, k] = s, e, mask
            for b in range(NL):
                if mask >> b & 1 and cls[a, b] >= 0:
                    lane_mov[a, k] |= 1 << cls[a, b]

    D = _setbacks(present, box_angle, n_in, n_out)
    # the stop line is perpendicular to the leg at arc length D[a]
    stop_dir = np.array([cls_[a].heading(D[a]) if present[a] else 0.0 for a in range(NL)])

    # ---- paths: one per (approach lane, exit leg)
    P = []
    for a in range(NL):
        for k in range(n_in[a]):
            for b in range(NL):
                if lane_exit[a, k] >> b & 1 and cls[a, b] >= 0:
                    P.append((a, k, b))
    assert len(P) <= pmax, len(P)
    path_of = np.full((NL, kmax, 3), -1, np.int64)
    p_leg, p_lane, p_mov, p_exit, p_exit_lane = (np.zeros(pmax, np.int64) for _ in range(5))
    p_len = np.zeros(pmax, np.float32)
    p_pts = np.zeros((pmax, NSAMP, 2), np.float32)
    p_s = np.zeros((pmax, NSAMP), np.float32)
    for i, (a, k, b) in enumerate(P):
        m = cls[a, b]
        group = [kk for kk in range(n_in[a]) if lane_exit[a, kk] >> b & 1]
        g, G = group.index(k), len(group)
        if m == MOV_RIGHT:
            j = max(n_out[b] - 1 - (G - 1 - g), 0)
        else:                                    # lefts take the leftmost lanes, throughs in order
            j = min(g, n_out[b] - 1)
        _, t_a = cls_[a].at(D[a])
        _, t_b = cls_[b].at(D[b])
        p0 = cls_[a].lane_point(D[a], (k + 0.5) * W, inbound=True)
        p1 = cls_[b].lane_point(D[b], (j + 0.5) * W, inbound=False)
        pts, s = _bezier(p0, -t_a, p1, t_b)
        if path_of[a, k, m] < 0:
            path_of[a, k, m] = i
        p_leg[i], p_lane[i], p_mov[i], p_exit[i], p_exit_lane[i] = a, k, m, b, j
        p_len[i], p_pts[i], p_s[i] = s[-1], pts, s
    nP = len(P)

    # ---- conflict zones: c_in/c_out = zone on row path p w.r.t. column path q
    c_kind = np.zeros((pmax, pmax), np.int64)
    c_in = np.zeros((pmax, pmax), np.float32)
    c_out = np.zeros((pmax, pmax), np.float32)
    for p in range(nP):
        for q in range(p + 1, nP):
            d = np.linalg.norm(p_pts[p][:, None, :] - p_pts[q][None, :, :], axis=-1)
            close = d < CONFLICT_DIST
            if not close.any():
                continue
            ip, iq = np.nonzero(close)
            same_start = p_leg[p] == p_leg[q] and p_lane[p] == p_lane[q]
            same_end = p_exit[p] == p_exit[q] and p_exit_lane[p] == p_exit_lane[q]
            if same_start:
                ck = C_DIVERGE               # shared start: from 0 until the curves separate
                sep = np.nonzero(~close.diagonal())[0]
                k_end = sep[0] if len(sep) else NSAMP - 1
                zp, zq = (0.0, p_s[p][k_end]), (0.0, p_s[q][k_end])
            else:
                ck = C_MERGE if same_end else C_CROSS
                zp = (max(p_s[p][ip.min()] - 0.5, 0.0), min(p_s[p][ip.max()] + 0.5, p_len[p]))
                zq = (max(p_s[q][iq.min()] - 0.5, 0.0), min(p_s[q][iq.max()] + 0.5, p_len[q]))
            c_kind[p, q] = c_kind[q, p] = ck
            c_in[p, q], c_out[p, q] = zp
            c_in[q, p], c_out[q, p] = zq

    # ---- streets: pairs of roughly opposite legs, then single legs
    streets, used = [], set()
    order = sorted((a for a in range(NL) if present[a]), key=lambda a: -(n_in[a] + n_out[a]))
    for a in order:
        if a in used:
            continue
        best, bd = None, 35.0
        for b in order:
            if b != a and b not in used:
                dev = abs(((far_angle[b] - far_angle[a]) % 360.0) - 180.0)
                if dev < bd:
                    best, bd = b, dev
        streets.append((a, best) if best is not None else (a,))
        used |= {a} | ({best} if best is not None else set())
    if major is None:                     # widest street is major
        major = [a in streets[0] for a in range(NL)]
    major = np.array(major, bool)

    # ---- signal plan
    phase_dur = np.zeros(NPH, np.float32)
    phase_state = np.zeros((NPH, pmax), np.int64)
    phase_walk = np.zeros((NPH, NL), bool)
    same_left = np.zeros((NPH, pmax), np.float32)
    n_phase = 0
    left_lanes = lambda a: [k for k in range(n_in[a]) if any(lane_exit[a, k] >> b & 1 and cls[a, b] == MOV_LEFT for b in range(NL))]
    left_only = lambda a, k: all(cls[a, b] == MOV_LEFT for b in range(NL) if lane_exit[a, k] >> b & 1)
    if ctrl == IX_SIGNAL:
        left_mode = left_mode or (lambda st: "permitted")
        green = green or (lambda kind: {"left": 12.0, "through": 30.0, "stem": 18.0}[kind])
        greens = []                       # (duration, {path: state}, walk legs)
        for st in streets:
            has_left = [bool(left_lanes(a)) for a in st]
            # exclusive left lanes on every approach with lefts (bays or multi-lane lefts)
            bays = all(all(left_only(a, k) for k in left_lanes(a)) and n_in[a] > len(left_lanes(a))
                       for a, hl in zip(st, has_left) if hl)
            multi = any(len(left_lanes(a)) >= 2 for a in st)
            mode = "stem" if len(st) == 1 else ("protected" if multi and bays else (left_mode(st) if bays else "permitted"))
            others = [a for a in range(NL) if present[a] and a not in st]
            if mode in ("protected", "prot_perm"):
                greens.append((green("left"), {i: SIG_G for i, (a, _, b) in enumerate(P)
                                               if a in st and cls[a, b] == MOV_LEFT}, []))
            thru = {}
            for i, (a, _, b) in enumerate(P):
                if a in st:
                    if cls[a, b] != MOV_LEFT or mode == "stem":
                        thru[i] = SIG_G
                    elif mode in ("permitted", "prot_perm"):
                        thru[i] = SIG_PERM
            greens.append((green("stem" if mode == "stem" else "through"), thru, others))
        if 3 * len(greens) > NPH:            # many-leg nodes (LPSim networks) need more phases
            nph = 3 * len(greens)
            phase_dur = np.zeros(nph, np.float32)
            phase_state = np.zeros((nph, pmax), np.int64)
            phase_walk = np.zeros((nph, NL), bool)
            same_left = np.zeros((nph, pmax), np.float32)
        for d_, st, w in greens:
            phase_dur[n_phase] = d_
            for i, s in st.items():
                phase_state[n_phase, i] = s
            phase_walk[n_phase, w] = True
            n_phase += 1
            phase_dur[n_phase] = 3.5      # amber
            for i in st:
                phase_state[n_phase, i] = SIG_Y
            n_phase += 1
            phase_dur[n_phase] = 1.5      # all red
            n_phase += 1
        for k in range(n_phase):
            for p in range(nP):
                t, j = 0.0, k
                while True:
                    t += phase_dur[j]
                    j = (j + 1) % n_phase
                    if phase_state[j, p] != phase_state[k, p] or j == k:
                        break
                same_left[k, p] = t

    # ---- staging of permitted turns: wait inside the box, just before the first zone shared
    # with traffic that has priority in the same phase (docs/INTERSECTIONS.md section 3)
    s_wait = np.full(pmax, -1.0, np.float32)
    n_stage = np.zeros(pmax, np.int64)
    for p in range(nP):
        zones = [c_in[p, q] for k in range(n_phase) if phase_state[k, p] == SIG_PERM
                 for q in range(nP) if phase_state[k, q] == SIG_G and c_kind[p, q] in (C_CROSS, C_MERGE)]
        if not zones:
            continue
        sw = min(zones) - STAGE_GAP
        first = 5.0 + 0.3                      # the first staged vehicle must be past the stop bar
        if sw >= first:
            s_wait[p] = sw
            n_stage[p] = min(STAGE_MAX, 1 + int((sw - first) // STAGE_SPACING))

    # ---- curve speed caps per link (approach a: towards the box; exit b: away from it)
    vcap_app = np.full((NL, VCAP_N), VCAP_FREE, np.float32)
    vcap_exit = np.full((NL, VCAP_N), VCAP_FREE, np.float32)
    for a in range(NL):
        if present[a]:
            if n_in[a]:
                vcap_app[a] = _curve_cap(cls_[a], D[a] + L_app[a], D[a], float(L_app[a]))
            if n_out[a]:
                vcap_exit[a] = _curve_cap(cls_[a], D[a], D[a] + L_exit[a], float(L_exit[a]))

    ped_rate = np.zeros(NL) if ped_rate is None else np.asarray(ped_rate, float)
    ped_cross_time = np.where(present, (n_in + n_out) * W / 1.2, 0.0)
    return dict(ctrl=ctrl, legs=np.array(present), major=major, angle=box_angle.astype(np.float32),
                stop_angle=stop_dir.astype(np.float32), D=D.astype(np.float32),
                streets=streets, turn_class=cls, L_app=L_app, L_exit=L_exit, n_in=n_in, n_out=n_out,
                lane_start=lane_start, lane_end=lane_end, lane_mov=lane_mov, lane_exit=lane_exit,
                n_paths=nP, path_of=path_of, p_leg=p_leg, p_lane=p_lane, p_mov=p_mov, p_exit=p_exit,
                p_exit_lane=p_exit_lane, p_len=p_len, p_pts=p_pts, p_s=p_s, c_kind=c_kind, c_in=c_in, c_out=c_out,
                n_phase=n_phase, phase_dur=phase_dur, phase_state=phase_state, phase_walk=phase_walk,
                phase_same_left=same_left, rtor=bool(rtor and ctrl == IX_SIGNAL),
                s_wait=s_wait, n_stage=n_stage, vcap_app=vcap_app, vcap_exit=vcap_exit,
                centrelines=cls_,
                ped_rate=(ped_rate / 3600.0).astype(np.float32), ped_cross_time=ped_cross_time.astype(np.float32))


SIG4_LEG_M = 600.0                # 4-leg signal study: approach and exit link length (m)
SIG4_BAY_M = 45.0                 # short right-turn bay at the end of each approach (right turn on red)
SIG4_GREEN_S = 30.0               # one green per street (E-W, then N-S); permitted lefts, RTOR
SIG4_DEMAND_VPH = 500.0           # evaluation / viewer demand per leg
SIG4_DEMAND = (50.0, 800.0)       # training range, veh/h per leg
SIG4_SHARES = (0.08, 0.12)        # default left / right share of a leg's demand (< 10% / < 15%)


def signal_4leg(rng=None, demand_vph=None, left_share=None, right_share=None, n_lanes=2):
    """The 4-leg signal study design (docs/INTERSECTIONS.md section 7).

    Legs of 600 m (approach and exit). Each approach has ``n_lanes`` full-length lanes, the
    leftmost shared by left and through traffic, plus a 45 m right-turn bay; rights turn only
    from the bay. Two 30 s phases (E-W, N-S) each serve every movement of both directions:
    permitted lefts, protected throughs and rights, right turn on red. No pedestrians.
    ``demand_vph`` is per leg. Vehicles enter on a random full-length lane whatever their
    movement, so lefts entering on the right and every right-turner make mandatory lane
    changes. Unset arguments are drawn from ``rng`` (training): demand 50-800 veh/h per leg,
    left share 3-10%, right share 5-15%; with ``rng=None`` the defaults are 500 veh/h, 8% / 12%.
    """
    if demand_vph is None:
        demand_vph = SIG4_DEMAND_VPH if rng is None else rng.uniform(*SIG4_DEMAND)
    if left_share is None:
        left_share = SIG4_SHARES[0] if rng is None else rng.uniform(0.03, 0.10)
    if right_share is None:
        right_share = SIG4_SHARES[1] if rng is None else rng.uniform(0.05, 0.15)
    L = SIG4_LEG_M
    legs = []
    for a in range(NLEG):
        ex = {m: (a + off) % 4 for m, off in CLASS_OFFSET.items()}
        lanes = [(0.0, L, 1 << ex[MOV_THROUGH] | (1 << ex[MOV_LEFT]))]
        lanes += [(0.0, L, 1 << ex[MOV_THROUGH]) for _ in range(n_lanes - 1)]
        lanes.append((L - SIG4_BAY_M, L, 1 << ex[MOV_RIGHT]))
        legs.append(dict(angle=90.0 * a, lanes=lanes, n_out=n_lanes, L_app=L, L_exit=L))
    d = build(legs, IX_SIGNAL, major=[True, False, True, False], ped_rate=np.zeros(NLEG), rtor=True,
              left_mode=lambda st: "permitted", green=lambda kind: SIG4_GREEN_S)
    d["kind"] = "signal_4leg"
    share = np.array([left_share, 1.0 - left_share - right_share, right_share], np.float32)
    spawn_rate = np.zeros((NLEG, KMAX), np.float32)
    mov_prob = np.zeros((NLEG, KMAX, 3), np.float32)
    spawn_rate[:, :n_lanes] = demand_vph / 3600.0 / n_lanes
    mov_prob[:, :n_lanes] = share
    d["spawn_rate"], d["mov_prob"] = spawn_rate, mov_prob
    d["demand_vph"] = np.float32(demand_vph)
    d["demand_jitter"] = np.float32(0.0)      # demand is exact (no per-episode multiplier)
    del d["streets"], d["turn_class"]
    return d


def _curved_shape(rng, angle, length=1300.0):
    """Centre line leaving the intersection along ``angle``: straight for 30-60 m, then an arc
    (radius 100-400 m, turning 15-40 degrees either way), then straight again."""
    s1, R = rng.uniform(30.0, 60.0), rng.uniform(100.0, 400.0)
    turn = np.radians(rng.uniform(15.0, 40.0)) * rng.choice([-1.0, 1.0])
    pts, h, p, s = [np.zeros(2)], np.radians(angle), np.zeros(2), 0.0
    ds, arc = 5.0, abs(turn) * R
    while s < length:
        if s1 <= s < s1 + arc:
            h += np.sign(turn) * ds / R
        p = p + ds * np.array([np.cos(h), np.sin(h)])
        pts.append(p)
        s += ds
    return np.array(pts)


def _leg_lanes(n_left, n_thru, n_right, exits, L_app, rng):
    """Approach lanes left -> right: ``n_left`` exclusive left-turn lanes (bays), ``n_thru``
    full-length lanes, ``n_right`` exclusive right-turn lanes. Without exclusive turn lanes the
    outermost through lanes share the turn. Returns (start, end, exit_mask) tuples."""
    lanes = []
    has = lambda m: m in exits
    bay_l = L_app - rng.uniform(50.0, 120.0)
    bay_r = L_app - rng.uniform(40.0, 90.0)
    nl = n_left if has(MOV_LEFT) else 0
    nr = n_right if has(MOV_RIGHT) else 0
    lanes += [[bay_l, L_app, {MOV_LEFT}] for _ in range(nl)]
    if has(MOV_THROUGH):
        thru = [[0.0, L_app, {MOV_THROUGH}] for _ in range(n_thru)]
        if has(MOV_LEFT) and nl == 0:
            thru[0][2].add(MOV_LEFT)
        if has(MOV_RIGHT) and nr == 0:
            thru[-1][2].add(MOV_RIGHT)
    else:                                         # T stem: left lanes on the left, right lanes on the right
        turns = [m for m in (MOV_LEFT, MOV_RIGHT) if has(m)]
        if n_thru == 1 or len(turns) == 1:
            thru = [[0.0, L_app, set(turns)] for _ in range(n_thru)]
            if len(turns) == 2 and n_thru > 1:
                thru[0][2], thru[-1][2] = {MOV_LEFT}, {MOV_RIGHT}
        else:
            nh = (n_thru + 1) // 2
            thru = [[0.0, L_app, {MOV_LEFT} if k < nh else {MOV_RIGHT}] for k in range(n_thru)]
    lanes += thru
    lanes += [[bay_r, L_app, {MOV_RIGHT}] for _ in range(nr)]
    out = [(s, e, ms) for s, e, ms in lanes if ms]
    served = set().union(*[ms for _, _, ms in out]) if out else set()
    if out and served != set(exits):
        s, e, ms = out[-1]
        out[-1] = (s, e, ms | (set(exits) - served))
    return [(s, e, sum(1 << exits[m] for m in ms)) for s, e, ms in out]


def design(rng, kind, demand_scale=1.0, peds=None):
    """One random intersection of the given kind (see KINDS) for training.

    Geometry is randomised beyond the orthogonal textbook layout: leg directions within
    +-12 degrees of the compass points, 40% of legs curved, 1-4 through lanes, and at signals
    single or dual exclusive left- and right-turn lanes. Arrays are padded to fixed sizes
    (NLEG, KMAX, PMAX, NPH) so designs can be stacked."""
    if kind == "signal_4leg":
        return signal_4leg(rng, n_lanes=int(rng.integers(1, 3)))
    ctrl, nlegs = KINDS[kind]
    present = [True] * NLEG
    missing = -1
    if nlegs == 3:
        missing = int(rng.integers(NLEG))
        present[missing] = False
    if missing < 0:
        major = [True, False, True, False]
    else:
        major = [False] * NLEG
        major[(missing + 1) % 4] = major[(missing + 3) % 4] = True
    sig = ctrl == IX_SIGNAL
    n_thru, n_left, n_right = (np.zeros(NLEG, int) for _ in range(3))
    for a in range(NLEG):
        if not present[a]:
            continue
        if ctrl == IX_AWSC:
            n_thru[a] = rng.integers(1, 3)
        elif major[a]:
            n_thru[a] = rng.choice([1, 2, 3, 4], p=[0.15, 0.4, 0.3, 0.15]) if sig else rng.integers(1, 3)
        else:
            n_thru[a] = rng.integers(1, 4) if sig else 1
        if sig:
            n_left[a] = rng.choice([0, 1, 2], p=[0.35, 0.45, 0.2]) if n_thru[a] >= 2 else rng.choice([0, 1], p=[0.4, 0.6])
            n_right[a] = rng.choice([0, 1, 2], p=[0.55, 0.35, 0.1]) if n_thru[a] >= 2 else rng.choice([0, 1], p=[0.6, 0.4])
        else:
            n_left[a] = int(rng.random() < 0.3)
            n_right[a] = int(rng.random() < 0.15)
    legs = []
    for a in range(NLEG):
        if not present[a]:
            legs.append(None)
            continue
        exits = {m: (a + off) % 4 for m, off in CLASS_OFFSET.items() if present[(a + off) % 4]}
        L_app = rng.uniform(150.0, 300.0)
        lanes = _leg_lanes(n_left[a], n_thru[a], n_right[a], exits, L_app, rng)
        # exit lanes: enough for the opposite through lanes and for multi-lane turns into this leg
        opp, from_left, from_right = (a + 2) % 4, (a + 1) % 4, (a + 3) % 4
        n_out = max(n_thru[a], n_thru[opp] if present[opp] else 0,
                    n_left[from_left] if present[from_left] else 0,
                    n_right[from_right] if present[from_right] else 0, 1)
        angle = 90.0 * a + rng.uniform(-12.0, 12.0)
        shape = _curved_shape(rng, angle) if rng.random() < 0.4 else None
        legs.append(dict(angle=angle, shape=shape, lanes=lanes, n_out=int(n_out), L_app=L_app,
                         L_exit=rng.uniform(100.0, 200.0)))
    if peds is None:
        peds = rng.random() < 0.5
    ped = rng.uniform(60, 400, NLEG) * np.array(present) if peds else np.zeros(NLEG)
    d = build(legs, ctrl, major=major, ped_rate=ped, rtor=rng.random() < 0.7,
              left_mode=lambda st: str(rng.choice(["protected", "permitted", "prot_perm"])),
              green=lambda kind: {"left": rng.uniform(8, 15), "through": rng.uniform(20, 40),
                                  "stem": rng.uniform(12, 25)}[kind])
    d["kind"] = kind
    # ---- demand: vehicles enter on any full-length lane, whatever their movement
    spawn_rate = np.zeros((NLEG, KMAX), np.float32)
    mov_prob = np.zeros((NLEG, KMAX, 3), np.float32)
    for a in range(NLEG):
        if not present[a]:
            continue
        lo, hi = {IX_SIGNAL: (0.08, 0.2), IX_AWSC: (0.04, 0.1)}.get(ctrl, (0.06, 0.16) if major[a] else (0.03, 0.08))
        for k in range(d["n_in"][a]):
            if d["lane_start"][a, k] <= 0.0:
                spawn_rate[a, k] = rng.uniform(lo, hi) * demand_scale
        share = rng.dirichlet([2.0, 6.0, 2.0])
        for m, off in CLASS_OFFSET.items():
            if not present[(a + off) % 4]:
                share[m] = 0.0
        mov_prob[a, :] = share / share.sum()
    d["spawn_rate"], d["mov_prob"] = spawn_rate, mov_prob
    d["demand_vph"] = np.float32(3600.0 * spawn_rate.sum() / max(nlegs, 1))
    d["demand_jitter"] = np.float32(1.0)      # per-episode demand multiplier 0.7-1.3
    del d["streets"], d["turn_class"]
    return d


def stack(designs):
    """Stack a list of designs into arrays with a leading batch dimension."""
    keys = [k for k, v in designs[0].items() if isinstance(v, (np.ndarray, float, int, bool, np.floating, np.integer, np.bool_))]
    return {k: np.stack([np.asarray(d[k]) for d in designs]) for k in keys}
