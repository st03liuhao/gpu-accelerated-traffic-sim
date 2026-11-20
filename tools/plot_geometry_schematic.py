"""Schematic of the intersection geometry data structure (docs/DESIGN.md section 7.2).

    python tools/plot_geometry_schematic.py [--out docs/img/geometry_data_structure.png]

Panel (a) is drawn from the real ``signal_4leg`` study design (lcrl.intersection), so lane
positions, the bay, the turning paths and the conflict zone are the stored values.
Panel (b) shows how the arrays are organised and indexed.
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lcrl import intersection as ixd                       # noqa: E402
from lcrl.config import MOV_LEFT, MOV_THROUGH               # noqa: E402

INK, INK2, MUTED = "#1f1f1d", "#55554f", "#8f8e88"
ROAD, LANE_EDGE, BAY = "#e9e8e3", "#b9b8b1", "#f3dfb4"
EGO, OPP, ZONE = "#2a78d6", "#c2410c", "#d4a017"
PATH_FAINT = "#c9c8c2"


def lane_quad(p0, p1, normal, w):
    h = normal * w / 2
    return np.array([p0 - h, p1 - h, p1 + h, p0 + h])


def panel_geometry(ax, d):
    W, D = ixd.W, float(np.max(d["D"]))
    view = D + 62.0
    # ---- links and lanes
    for a in range(4):
        u = ixd.unit_deg(float(d["angle"][a]))
        L = float(d["L_app"][a])
        for k in range(int(d["n_in"][a])):
            ls, le = float(d["lane_start"][a, k]), float(d["lane_end"][a, k])
            n = ixd.rnormal(-u)
            c = n * (k + 0.5) * W
            p0, p1 = u * (D + L - ls) + c, u * (D + L - le) + c
            bay = ls > 0
            ax.add_patch(Polygon(lane_quad(p0, p1, n, W), closed=True, fc=BAY if bay else ROAD,
                                 ec=LANE_EDGE, lw=0.6, zorder=1))
        for j in range(int(d["n_out"][a])):
            n = ixd.rnormal(u)
            c = n * (j + 0.5) * W
            p0, p1 = u * D + c, u * (D + float(d["L_exit"][a])) + c
            ax.add_patch(Polygon(lane_quad(p0, p1, n, W), closed=True, fc=ROAD, ec=LANE_EDGE, lw=0.6, zorder=1))
        # stop line
        n = ixd.rnormal(-u)
        s0, s1 = u * D, u * D + n * int(d["n_in"][a]) * W
        ax.plot([s0[0], s1[0]], [s0[1], s1[1]], color=INK, lw=2.2, zorder=3, solid_capstyle="butt")
    ax.add_patch(Polygon(np.array([[-D, -D], [D, -D], [D, D], [-D, D]]), closed=True, fc=ROAD, ec="none", zorder=0.5))

    # ---- turning paths
    nP = int(d["n_paths"])
    for p in range(nP):
        pts = d["p_pts"][p]
        ax.plot(pts[:, 0], pts[:, 1], color=PATH_FAINT, lw=0.8, zorder=2)
    # ego: permitted left from the east approach; opponent: opposing through from the west
    p_ego = int(d["path_of"][0, 0, MOV_LEFT])
    q_opp = int(d["path_of"][2, 0, MOV_THROUGH])
    for p, col in ((p_ego, EGO), (q_opp, OPP)):
        pts = d["p_pts"][p]
        ax.plot(pts[:, 0], pts[:, 1], color=col, lw=1.8, zorder=4)
        ax.scatter(pts[::4, 0], pts[::4, 1], s=6, color=col, zorder=5)
    # conflict zone on both paths
    for p, q in ((p_ego, q_opp), (q_opp, p_ego)):
        s_tab, pts = d["p_s"][p], d["p_pts"][p]
        zin, zout = float(d["c_in"][p, q]), float(d["c_out"][p, q])
        m = (s_tab >= zin) & (s_tab <= zout)
        ax.plot(pts[m, 0], pts[m, 1], color=ZONE, lw=6, alpha=0.85, zorder=3.5, solid_capstyle="butt")

    # ---- vehicles: ego i at its stop line, opponent j 40 m before its stop line
    def car(pos, heading, col):
        h = np.radians(heading)
        f, s = np.array([np.cos(h), np.sin(h)]), np.array([-np.sin(h), np.cos(h)])
        quad = np.array([pos + f * 2.5 + s * 0.95, pos - f * 2.5 + s * 0.95, pos - f * 2.5 - s * 0.95, pos + f * 2.5 - s * 0.95])
        ax.add_patch(Polygon(quad, closed=True, fc=col, ec="white", lw=0.8, zorder=6))
    u_e, u_w = ixd.unit_deg(0.0), ixd.unit_deg(180.0)
    ego_pos = u_e * (D + 3.0) + ixd.rnormal(-u_e) * 0.5 * W
    opp_pos = u_w * (D + 40.0) + ixd.rnormal(-u_w) * 0.5 * W
    car(ego_pos, 180.0, EGO)
    car(opp_pos, 0.0, OPP)

    # ---- annotations, kept in the four empty corners
    t = dict(fontsize=8, color=INK2, zorder=8)
    line = lambda col: dict(arrowstyle="-", color=col, lw=0.6)
    # north-east: the bay (label right above it), then the approach link's x axis
    bay_x = D + ixd.SIG4_BAY_M / 2
    ax.text(bay_x, 3 * W + 0.8, "lane 2 = right-turn bay: lane_mov = R\nlane_start = 555, lane_end = 600",
            ha="center", va="bottom", **t)
    ax.annotate("", xy=(D + 4, 22.0), xytext=(view - 2, 22.0), arrowprops=dict(arrowstyle="->", color=INK2, lw=0.8))
    ax.text(view - 2, 23.5, "approach link 0 (east, above centre line)\n"
            "x = 0 … L_app = 600 m at the stop line", ha="right", va="bottom", **t)
    ax.text(view - 2, 34, "lane 0: lane_mov = L | T (shared)\nlane 1: lane_mov = T", ha="right", va="bottom", **t)
    # south-east: ego, exit link, box
    ax.annotate("ego i at its stop line\nrank 1 (permitted left)\nplan = path p", ego_pos,
                xytext=(D + 4, -13), color=EGO, fontsize=8, zorder=8, va="top", arrowprops=line(EGO))
    ax.text(D + 4, -31, "exit link 4 (east, below centre line)\nx = 0 at the box … L_exit = 600 m",
            va="top", **t)
    ax.text(D + 4, -44, "inside the box: link = −1 and\nx = distance along the vehicle's path", va="top",
            fontsize=8, color=MUTED)
    # north-west: conflict zone
    q_pts, q_s = d["p_pts"][q_opp], d["p_s"][q_opp]
    zm = (q_s >= d["c_in"][q_opp, p_ego]) & (q_s <= d["c_out"][q_opp, p_ego])
    ax.annotate("conflict zone of paths (p, q)\nc_kind[p, q] = crossing\nc_in[p, q] … c_out[p, q]: zone on p\n"
                "c_in[q, p] … c_out[q, p]: zone on q", q_pts[zm].mean(0), xytext=(-view + 2, 50),
                color="#8a6a00", fontsize=8, zorder=8, arrowprops=line("#8a6a00"))
    # south-west: paths and the oncoming vehicle
    mid = d["p_pts"][p_ego][len(d["p_pts"][p_ego]) * 2 // 3]
    ax.annotate("path p = path_of[0, 0, L]\np_pts[p]: 48 Bézier samples (dots)\np_s[p]: arc length at each sample",
                mid, xytext=(-view + 2, -56), color=EGO, fontsize=8, zorder=8, arrowprops=line(EGO))
    ax.annotate("oncoming j: rank 2 (green)\nplan = path q = path_of[2, 0, T]\n$s_j$ = −40 m (before its stop line)",
                opp_pos, xytext=(-view + 2, -30), color=OPP, fontsize=8, zorder=8, arrowprops=line(OPP))
    ax.set_xlim(-view, view)
    ax.set_ylim(-view, view)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("(a) Stored geometry of the 4-leg signal design (centre ± 75 m of 600 m legs)",
                 fontsize=10.5, color=INK, loc="left")


def box(ax, x, y, w, h, title, lines, col, title_col=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.012",
                                fc="white", ec=col, lw=1.2, transform=ax.transAxes))
    ax.text(x + 0.012, y + h - 0.02, title, transform=ax.transAxes, fontsize=9, fontweight="bold",
            color=title_col or col, va="top")
    ax.text(x + 0.012, y + h - 0.058, "\n".join(lines), transform=ax.transAxes, fontsize=7.6, color=INK,
            va="top", family="monospace", linespacing=1.35)


def arrow(ax, a, b, col=INK2, text=None, rad=0.0, toff=(0, 0)):
    ax.add_patch(FancyArrowPatch(a, b, transform=ax.transAxes, arrowstyle="-|>", mutation_scale=9,
                                 color=col, lw=0.9, connectionstyle=f"arc3,rad={rad}"))
    if text:
        ax.text((a[0] + b[0]) / 2 + toff[0], (a[1] + b[1]) / 2 + toff[1], text, transform=ax.transAxes,
                fontsize=7.4, color=col, ha="center", va="center", style="italic")


def panel_arrays(ax):
    ax.axis("off")
    ax.set_title("(b) How the arrays are organised and indexed", fontsize=10.5, color=INK, loc="left")
    GREY, BLUE, ORANGE, GREEN, PURPLE = "#55554f", "#2a78d6", "#c2410c", "#2f7d4f", "#7a5bc7"
    box(ax, 0.00, 0.79, 0.47, 0.18, "Links  [leg = 0..3]",
        ["L_app, L_exit      length (m)", "angle              direction (deg)",
         "n_in, n_out        lane counts", "links 0-3 approach, 4-7 exit"], GREY)
    box(ax, 0.53, 0.79, 0.47, 0.18, "Lanes  [leg, lane = 0..K-1]",
        ["lane_start   start on the link (m)", "lane_end     end on the link (m)",
         "lane_mov     bitmask L|T|R", "lane_exit    bitmask of exit legs", "(a bay has lane_start > 0)"], GREY)
    box(ax, 0.00, 0.50, 0.47, 0.25, "Turning paths  [path p = 0..P-1]",
        ["p_pts [p, 48, 2]  Bezier samples (m)", "p_s   [p, 48]     arc length",
         "p_len             path length", "p_leg, p_lane     start", "p_exit, p_exit_lane  end",
         "p_mov             L / T / R"], BLUE)
    box(ax, 0.53, 0.50, 0.47, 0.25, "Conflicts  [p, q]   (precomputed)",
        ["c_kind  0 none / crossing /", "        merge / diverge", "c_in    zone start on p (m)",
         "c_out   zone end on p (m)", "centrelines within 2.6 m"], "#8a6a00")
    box(ax, 0.00, 0.30, 0.47, 0.15, "Lookup  path_of [leg, lane, mov]",
        ["-> path p, or -1 if the lane", "   does not serve the movement"], BLUE)
    box(ax, 0.53, 0.30, 0.47, 0.15, "Signal plan  [phase, p]",
        ["phase_dur [phase]   s", "phase_state [phase, p]", "  red / permitted / green / amber"], GREEN)
    box(ax, 0.00, 0.00, 1.00, 0.24, "Per env e, per vehicle n  (GPU tensors [E, N])",
        ["design_idx[e]  -> row of every design array (D(key) = pool[key][design_idx])",
         "link[e,n]      0-3 approach, 4-7 exit, -1 in the box",
         "lane_from, lane_to, prog   lane (and lane change in progress)",
         "x[e,n]         m along the link, or along the path in the box",
         "mov[e,n]       next movement   path[e,n]  path once in the box"], PURPLE)
    arrow(ax, (0.235, 0.45), (0.235, 0.50), BLUE)
    arrow(ax, (0.47, 0.62), (0.53, 0.62), "#8a6a00")
    arrow(ax, (0.47, 0.88), (0.53, 0.88), GREY)
    arrow(ax, (0.47, 0.55), (0.53, 0.42), GREEN)
    arrow(ax, (0.20, 0.24), (0.20, 0.30), PURPLE)
    ax.text(0.5, -0.06, "Each design is padded to fixed sizes (4 legs, K = 5 lanes, P = 32 paths), then all designs\n"
            "are stacked into one GPU tensor per field (the design pool).", transform=ax.transAxes,
            fontsize=8, color=INK2, ha="center", va="top")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/img/geometry_data_structure.png")
    a = ap.parse_args()
    d = ixd.signal_4leg(None)
    fig = plt.figure(figsize=(15, 7.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.0], wspace=0.06)
    panel_geometry(fig.add_subplot(gs[0]), d)
    panel_arrays(fig.add_subplot(gs[1]))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    fig.savefig(a.out, dpi=150, bbox_inches="tight", facecolor="white")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
