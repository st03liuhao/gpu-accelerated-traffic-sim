"""Framework diagram of the RL lane-changing model (docs/DESIGN.md section 1).

    python tools/plot_framework.py [--out docs/img/framework.png]

Three bands: training in batched GPU environments (the RL loop), deployment in LPSim, and
evaluation against the rule baseline.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

INK, INK2, MUTED = "#1f1f1d", "#55554f", "#8f8e88"
BAND = {"train": "#eef3fb", "deploy": "#f4f0e8", "eval": "#eef6ef"}
EDGE = {"train": "#2a78d6", "deploy": "#b7791f", "eval": "#2f855a"}
BOX = "#ffffff"
KEY = "#fff7e0"


def box(ax, x, y, w, h, title, lines=(), kind="train", fill=BOX, title_size=9.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=0.8",
                                fc=fill, ec=EDGE[kind], lw=1.3, zorder=2))
    ax.text(x + w / 2, y + h - 1.2, title, ha="center", va="top", fontsize=title_size,
            fontweight="bold", color=INK, zorder=3)
    for i, s in enumerate(lines):
        ax.text(x + w / 2, y + h - 3.6 - 2.05 * i, s, ha="center", va="top", fontsize=7.6,
                color=INK2, zorder=3)
    return (x, y, w, h)


def arrow(ax, p, q, label=None, color=INK, rad=0.0, lw=1.3, style="-|>", ls="-", off=(0, 0.8), size=7.3):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=11, lw=lw, color=color,
                                 connectionstyle=f"arc3,rad={rad}", linestyle=ls, zorder=1.5,
                                 shrinkA=1, shrinkB=1))
    if label:
        ax.text((p[0] + q[0]) / 2 + off[0], (p[1] + q[1]) / 2 + off[1], label, ha="center",
                va="bottom", fontsize=size, color=INK2, zorder=3,
                bbox=dict(fc="white", ec="none", pad=0.6, alpha=0.9))


def right(b, f=0.5):
    return (b[0] + b[2], b[1] + b[3] * f)


def left(b, f=0.5):
    return (b[0], b[1] + b[3] * f)


def top(b, f=0.5):
    return (b[0] + b[2] * f, b[1] + b[3])


def bottom(b, f=0.5):
    return (b[0] + b[2] * f, b[1])


def band(ax, y, h, kind, title, note, note_x=None):
    ax.add_patch(FancyBboxPatch((0.5, y), 119, h, boxstyle="round,pad=0,rounding_size=1.5",
                                fc=BAND[kind], ec="none", zorder=0))
    ax.text(1.8, y + h - 1.3, title, ha="left", va="top", fontsize=11.5, fontweight="bold",
            color=EDGE[kind], zorder=3)
    ax.text(118.2 if note_x is None else note_x, y + h - 1.4, note, ha="right" if note_x is None else "left",
            va="top", fontsize=7.8, color=INK2, style="italic", zorder=3)


def poly(ax, pts, color=INK, ls="-", lw=1.3):
    """Polyline with an arrow head on the last segment."""
    xs, ys = zip(*pts[:-1])
    ax.plot(list(xs) + [pts[-2][0]], list(ys) + [pts[-2][1]], color=color, lw=lw, ls=ls, zorder=1.5,
            solid_capstyle="round")
    arrow(ax, pts[-2], pts[-1], color=color, ls=ls, lw=lw)


def label(ax, x, y, s, bg, ha="center"):
    ax.text(x, y, s, ha=ha, va="center", fontsize=7.3, color=INK2, zorder=3,
            bbox=dict(fc=bg, ec="none", pad=0.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join("docs", "img", "framework.png"))
    a = ap.parse_args()

    fig, ax = plt.subplots(figsize=(14, 11.2))
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 96)
    ax.axis("off")
    tb, db, eb = BAND["train"], BAND["deploy"], BAND["eval"]

    # ------------------------------------------------------------------ 1 training
    band(ax, 44, 51.5, "train", "1  Training: batched simulation on the GPU (PyTorch)",
         "every step on the GPU: fixed shapes, no host sync, fused kernels, one CUDA graph per decision")
    pools = box(ax, 2.5, 74, 18.5, 15, "Scenario pools", [
        "built once, kept on GPU", "4 freeway scenarios", "6 intersection kinds",
        "random geometry, demand,", "signals, driver traits"])
    envs = box(ax, 26.5, 74, 26.5, 15, "Batched environments", [
        "512 freeway segments + 256 intersections", "physics every 0.1 s: IDM, Krauss cap,",
        "3 s lane changes, signals,", "entry permission, staging", "5 physics steps per decision"])
    obs = box(ax, 58, 74, 19.5, 15, "Observation", [
        "68 features per vehicle", "ego: 23 (state, route,", "control, traits)",
        "3 lanes x 15 (leader, follower,", "queue, best gap, intent)"])
    net = box(ax, 83, 74, 21, 15, "Shared policy (actor)", [
        "one MLP for every vehicle", "68 -> 64 -> 64 -> 3 + 5", "tanh, 9,096 parameters",
        "critic 68 -> 64 -> 64 -> 1", "(training only)"], fill=KEY)
    shield = box(ax, 83, 51, 21, 16, "Actions + safety shield", [
        "lateral: keep / left / right", "longitudinal: offset to IDM", "{-3, -1.5, 0, 0.75, 1.5} m/s2",
        "masks unsafe lane changes,", "caps speed, stop line = wall"])
    reward = box(ax, 45, 51, 25, 16, "Reward per vehicle", [
        "every 0.5 s decision: lost time,", "comfort, lag-vehicle impact,", "safety margins, route completion,",
        "missed turn (90 s detour),", "blocking, cooperation"])
    ppo = box(ax, 21, 51, 20.5, 16, "PPO update", [
        "32 decisions x ~700k", "vehicle samples / iteration", "GAE: gamma 0.99, lambda 0.95",
        "clip 0.2, 4 epochs, Adam", "minibatch 32,768"])
    gen = box(ax, 2.5, 51, 15, 16, "Policy versions", [
        "v9: from scratch", "v10: fine-tuned for", "0.1 s physics",
        "v11: fine-tuned for", "general geometry"])

    arrow(ax, right(pools), left(envs))
    label(ax, 23.75, 84.0, "reset", tb)
    arrow(ax, right(envs), left(obs))
    label(ax, 55.5, 84.0, "state", tb)
    arrow(ax, right(obs), left(net))
    label(ax, 80.25, 84.0, "obs", tb)
    arrow(ax, bottom(net), top(shield))
    label(ax, 93.5, 70.5, "action probabilities", tb)
    poly(ax, [left(shield, 0.55), (77.5, 59.8), (77.5, 71.2), (51.0, 71.2), (51.0, 74.0)])
    label(ax, 64.5, 71.2, "safe actions", tb)
    arrow(ax, (47.5, 74.0), (47.5, 67.0))
    label(ax, 43.2, 70.4, "rewards,\nnext state", tb)
    arrow(ax, left(reward), right(ppo))
    label(ax, 43.25, 61.8, "samples", tb)
    arrow(ax, left(ppo), right(gen))
    label(ax, 19.25, 61.8, "save", tb)
    poly(ax, [bottom(ppo, 0.5), (31.25, 48.3), (107.5, 48.3), (107.5, 81.5), (104.0, 81.5)],
         color=EDGE["train"], ls="--")
    label(ax, 70, 48.3, "update the shared weights", tb)
    ax.text(60, 45.4, "freeway and intersection samples form one batch: parameter sharing, every vehicle is an agent",
            ha="center", fontsize=7.8, color=INK2, style="italic")

    # ------------------------------------------------------------------ 2 deployment
    band(ax, 22.5, 20, "deploy", "2  Deployment: LPSim (CUDA)", "same rules as training; differences are bugs")
    weights = box(ax, 2.5, 25.5, 22, 11.5, "Exported actor", [
        "lc_policy_weights.h", "CUDA __device__ function", "weights in constant memory"], kind="deploy")
    osm = box(ax, 30, 25.5, 24, 11.5, "Road network", [
        "OSM via generate_network.py", "+ edge shapes, turn:lanes", "(edge_attrs.csv)"], kind="deploy")
    tables = box(ax, 59.5, 25.5, 22, 11.5, "Intersection tables", [
        "lpsim_intersections.py", "ix_*.csv: paths, conflicts,", "signal plans, staging"], kind="deploy")
    lps = box(ax, 87, 25.5, 30.5, 11.5, "LPSim vehicle kernel", [
        "0.1 s steps, policy every 5th step", "observation from the lane map",
        "full-intersection control"], kind="deploy", fill=KEY)
    arrow(ax, right(osm), left(tables))
    arrow(ax, right(tables), left(lps))
    poly(ax, [left(gen, 0.3), (1.2, 55.8), (1.2, 31.25), left(weights)], color=EDGE["deploy"])
    ax.text(1.9, 49.2, "release policy", ha="left", va="center", fontsize=7.3, color=EDGE["deploy"], zorder=3)
    poly(ax, [bottom(weights, 0.5), (13.5, 23.9), (92.0, 23.9), (92.0, 25.5)])
    label(ax, 50, 23.9, "RL_LANE_CHANGE=true", db)

    # ------------------------------------------------------------------ 3 evaluation
    band(ax, 0.5, 20.5, "eval", "3  Evaluation: RL vs the LPSim-style rule baseline", "same seeds, designs and demand for both", note_x=53)
    e_env = box(ax, 2.5, 8.2, 30, 9.2, "Batched-env evaluation", [
        "greedy policy, 600 s after warm-up", "freeway 4 x 128, intersections 6 x 64",
        ], kind="eval")
    ax.text(17.5, 9.3, "sweeps: aggressiveness, info range", ha="center", va="center", fontsize=7.6, color=INK2)
    e_study = box(ax, 36.5, 8.2, 22.5, 9.2, "4-leg study design", [
        "600 m legs, 500 veh/h/leg, 900 s", "demand sweep 50-800 veh/h/leg"], kind="eval")
    e_view = box(ax, 63, 8.2, 20, 9.2, "HTML viewer", [
        "recorded runs, rule vs RL", "replayed in the browser"], kind="eval")
    e_lps = box(ax, 87, 8.2, 30.5, 9.2, "LPSim validation", [
        "6 test intersections + city network", "travel time, violations, overlaps"], kind="eval")
    arrow(ax, bottom(lps, 0.5), top(e_lps, 0.5), color=EDGE["eval"])
    met = box(ax, 2.5, 1.2, 115, 4.6, "", [], kind="eval", fill=KEY)
    ax.text(60, 3.5, "Metrics:  throughput, delay, route success  |  collisions, stop-line violations, hard brakes, "
            "time at TTC < 1.5 s  |  lag-vehicle deceleration  |  lane changes, aborts, jerk",
            ha="center", va="center", fontsize=8.0, color=INK, zorder=3)
    for b in (e_env, e_study, e_lps):
        arrow(ax, bottom(b), (b[0] + b[2] / 2, 5.8), color=EDGE["eval"])

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, dpi=150, bbox_inches="tight", facecolor="white")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
