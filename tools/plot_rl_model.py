"""Schematic of the RL lane-changing model (docs/DESIGN.md section 2): what one vehicle
observes, the shared actor-critic network, the masked factorised policy and the vehicle model.

    python tools/plot_rl_model.py [--out docs/img/rl_model.png]
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_framework import EDGE, INK, INK2, MUTED, arrow, bottom, box, left, poly, right, top  # noqa: E402

CAR, EGO, LANE = "#c9d3e0", "#2a78d6", "#f2f2ee"


def car(ax, x, y, color, w=5.0, h=1.6, label=None):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.5",
                                fc=color, ec=INK2, lw=0.6, zorder=3))
    if label:
        ax.text(x, y, label, ha="center", va="center", fontsize=6.8, color="white" if color == EGO else INK, zorder=4)


def dim(ax, x0, x1, y, text="", color=INK2, below=False):
    ax.annotate("", (x0, y), (x1, y), arrowprops=dict(arrowstyle="<->", lw=0.8, color=color), zorder=4)
    if text:
        ax.text((x0 + x1) / 2, y + (-0.3 if below else 0.35), text, ha="center", va="top" if below else "bottom",
                fontsize=7, color=color, zorder=4)


def road(ax):
    """Three lanes around the ego vehicle, its neighbours and the distances it observes."""
    x0, x1, y0, w = 12.0, 52.0, 41.0, 3.4
    names = ["right, $\\ell=+1$\nlane $c_i+1$", "centre, $\\ell=0$\nlane $c_i$", "left, $\\ell=-1$\nlane $c_i-1$"]
    for k in range(3):
        y = y0 + k * w
        ax.add_patch(Rectangle((x0, y), x1 - x0, w, fc=LANE, ec="none", zorder=1))
        ax.plot([x0, x1], [y, y], color=MUTED, lw=0.8, ls=(0, (4, 3)) if k else "-", zorder=2)
        ax.text(x0 - 0.6, y + w / 2, names[k], ha="right", va="center", fontsize=6.4, color=INK2, zorder=2)
    ax.plot([x0, x1], [y0 + 3 * w] * 2, color=MUTED, lw=0.8, zorder=2)
    # the right lane is a deceleration lane that starts ahead: darker before its start
    ax.add_patch(Rectangle((x0, y0), 32.0 - x0, w, fc="#dcdcd6", ec="none", zorder=1.5))
    ax.text(33.0, y0 + 0.4, "lane starts here", fontsize=6.3, color=INK2, va="bottom", zorder=4)
    yc = y0 + 1.5 * w
    car(ax, 26.0, yc, EGO, label="$i$")
    car(ax, 38.0, yc, CAR, label="lead")
    car(ax, 16.0, yc, CAR, label="lag")
    car(ax, 32.0, y0 + 2.5 * w, CAR)
    car(ax, 44.0, y0 + 2.5 * w, CAR)
    ax.plot([x1 - 0.3] * 2, [y0, y0 + 3 * w], color="#c0392b", lw=2.0, zorder=4)
    ax.text(x1 + 0.4, y0 + 1.5 * w, "segment end:\nstop line,\noff-ramp", fontsize=6.5, color="#c0392b", va="center")
    dim(ax, 28.6, 35.4, yc + 1.15)
    ax.text(32.0, yc + 1.35, "$g^{\\mathrm{lead}}$", ha="center", va="bottom", fontsize=7, color=INK2, zorder=4)
    dim(ax, 18.6, 23.4, yc + 1.15)
    ax.text(21.0, yc + 1.35, "$g^{\\mathrm{lag}}$", ha="center", va="bottom", fontsize=7, color=INK2, zorder=4)
    dim(ax, 26.0, x1 - 0.3, y0 + 3 * w + 0.9, "deadline $D_i$")
    dim(ax, 6.0, 46.0, y0 - 1.5, "information range $R_i$ behind and ahead", color=EDGE["train"], below=True)
    ax.text(2.0, y0 + 3 * w + 3.4, "What vehicle $i$ observes", fontsize=9.5, fontweight="bold", color=INK)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=os.path.join("docs", "img", "rl_model.png"))
    a = p.parse_args()
    fig, ax = plt.subplots(figsize=(13.0, 7.4))
    ax.set_xlim(0, 121)
    ax.set_ylim(0, 63)
    ax.axis("off")
    road(ax)

    obs = box(ax, 2, 2, 30, 31, "Observation $o_i$ (68)",
              ["ego $e_i$ (23):",
               "speed $v_i$, desired speed $v_i^0$",
               "traits $\\alpha_i$, $R_i$",
               "lane change: progress $p_i$, direction $d_i$",
               "route: lanes to go $n_i$, deadline $D_i$",
               "control, priority, movement, $a_i^{\\mathrm{prev}}$",
               "",
               "lanes $q_i^{-1}, q_i^{0}, q_i^{+1}$ (3 x 15):",
               "exists, runway, starts ahead",
               "route lane, lanes to it",
               "leader, follower (gap, speed, signal)",
               "queue ahead, best gap"])
    arrow(ax, (17.0, 36.4), top(obs, 0.5))

    actor = box(ax, 37, 18, 17, 15, "Actor MLP", ["68 → 64 → 64 → 8", "tanh", "shared by all vehicles", "9,096 parameters"])
    critic = box(ax, 37, 2, 17, 12, "Critic MLP", ["68 → 64 → 64 → 1", "value $V_\\phi(o_i)$", "training only"],
                 fill="#f7f7f5")
    arrow(ax, right(obs, 0.7), left(actor))
    poly(ax, [(34.0, obs[1] + obs[3] * 0.7), (34.0, 8.0), left(critic)], color=MUTED, ls="--")

    lat = box(ax, 59, 26, 25, 8.5, "Lateral logits $z_1, z_2, z_3$", ["keep, left, right"])
    lon = box(ax, 59, 14.5, 25, 8.5, "Longitudinal logits $z_4, \\ldots, z_8$", ["$\\Delta a$ = −3, −1.5, 0, +0.75, +1.5 m/s²"])
    arrow(ax, right(actor, 0.75), left(lat))
    arrow(ax, right(actor, 0.25), left(lon))
    shield = box(ax, 62, 40, 22, 9, "Safety shield", ["mask $\\mu_i \\in \\{0,1\\}^3$ from gaps", "and lane geometry"],
                 kind="deploy", fill="#fff7e0")
    arrow(ax, bottom(shield), top(lat, 0.6), "unsafe → probability 0", off=(0, -0.6))

    pol = box(ax, 89, 20, 17, 15, "Policy", ["$\\pi^{\\mathrm{lat}}$: masked softmax", "$\\pi^{\\mathrm{lon}}$: softmax",
                                             "$u_i = (u^{\\mathrm{lat}}, u^{\\mathrm{lon}})$", "sample (training)",
                                             "argmax (deployment)"])
    arrow(ax, right(lat), left(pol, 0.72))
    arrow(ax, right(lon), left(pol, 0.28))

    veh = box(ax, 84, 1, 34, 14, "Vehicle model, 5 steps of 0.1 s",
              ["acceleration: IDM + $\\Delta a$, Krauss speed cap",
               "lane change: 3 s, occupies both lanes,",
               "progress $p_i \\leftarrow p_i + 0.1/3$ per step",
               "reward $r_i$ per step, summed per decision"])
    arrow(ax, bottom(pol), top(veh, 0.56))
    poly(ax, [(right(veh)[0], veh[1] + veh[3] * 0.85), (120.0, veh[1] + veh[3] * 0.85), (120.0, 60.5),
              (25.0, 60.5), (25.0, 57.0)], color=INK2)
    ax.text(80.0, 61.0, "next decision, 0.5 s later: new positions → new observation", ha="center", va="bottom",
            fontsize=7.4, color=INK2, bbox=dict(fc="white", ec="none", pad=0.5))
    arrow(ax, left(veh, 0.3), right(critic, 0.3), "rewards → PPO update", color=MUTED, ls="--", off=(1.5, 0.5))

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=170, bbox_inches="tight")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
