"""Figure 1: format drift masquerades as reward hacking.

(a) Paid-but-wrong rate over training under the box-matching and the robust referee, for a run whose
    apparent exploitation is format drift and for the one run with real exploitation.
(b) Apparent (box-matching) against true (robust) exploitation for all 20 runs.

Usage: python paper/figure_main.py   (writes figs/main_figure_paper.pdf/png and figs/main_figure.png)
"""

import gzip
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "results")
OUT = os.path.join(HERE, "figs")
sys.path.insert(0, os.path.join(HERE, "..", "fdrl"))
import checkers as ck  # noqa: E402

for f in ("/System/Library/Fonts/Helvetica.ttc", "/Library/Fonts/Arial Unicode.ttf"):
    if os.path.exists(f):
        font_manager.fontManager.addfont(f)
fam = "Helvetica" if any(f.name == "Helvetica" for f in font_manager.fontManager.ttflist) else "Arial Unicode MS"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#9a9993"
plt.rcParams.update({"font.family": fam, "font.size": 7.5, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": .6, "xtick.major.width": .6, "ytick.major.width": .6, "xtick.major.size": 2.5,
                     "ytick.major.size": 2.5, "legend.frameon": False, "pdf.fonttype": 42})


def samples(run):
    with gzip.open(os.path.join(ROOT, "runs", run, "samples.jsonl.gz"), "rt") as f:
        return [json.loads(l) for l in f]


def pbw_curve(run):
    """Per saved step: paid-but-wrong rate under the box-matching (r_strict) and the robust referee."""
    by = {}
    for s in samples(run):
        by.setdefault(s["step"], []).append(s)
    steps = sorted(by)
    old = [sum(s["r_arm"] > .5 and s["r_strict"] == 0 for s in by[t]) / len(by[t]) for t in steps]
    new = [sum(s["r_arm"] > .5 and ck.gold_label(s["reply"], s["gold"]) == 0 for s in by[t]) / len(by[t]) for t in steps]
    return steps, old, new


def smooth(v, k=3):
    return [sum(v[max(0, i - k // 2):i + k // 2 + 1]) / len(v[max(0, i - k // 2):i + k // 2 + 1]) for i in range(len(v))]


TITLES = []


def title(ax, letter, text):
    TITLES.append((ax, letter, text))  # drawn after layout so both titles share one baseline


R = {r["run"]: r for r in json.load(open(os.path.join(ROOT, "relabelled.json")))}
fig, (axa, axb) = plt.subplots(1, 2, figsize=(6.75, 2.75), gridspec_kw={"width_ratios": [1.25, 1.0], "wspace": 0.32})

# (a) one drift run and the one real exploit, under both referees
title(axa, "a", "Paid-but-wrong rate during training")
for run, col in (("tail50_slope_s1", BLUE), ("substring_all_s0", ORANGE)):
    st, old, new = pbw_curve(run)
    axa.plot(st, smooth(old), color=col, lw=1.4, ls=(0, (3, 2)))
    axa.plot(st, smooth(new), color=col, lw=1.8)
axa.set_xlim(0, 122); axa.set_ylim(-0.01, 0.62)
axa.set_xticks([0, 30, 60, 90, 120]); axa.set_yticks([0, .2, .4, .6], ["0%", "20%", "40%", "60%"])
axa.set_xlabel("training step"); axa.grid(axis="y", color=GRID, lw=.6); axa.set_axisbelow(True)
axa.legend(handles=[Line2D([], [], color=INK2, lw=1.4, ls=(0, (3, 2))), Line2D([], [], color=INK2, lw=1.8),
                    Line2D([], [], color=BLUE, lw=1.8), Line2D([], [], color=ORANGE, lw=1.8)],
           labels=["box-matching referee", "robust referee", "last 50 + ramp, seed 1", "number anywhere, seed 0"],
           loc="upper left", fontsize=6.4, handlelength=2.2, labelspacing=0.3)
axa.annotate("", xy=(119, 0.025), xytext=(119, 0.45), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=.7, mutation_scale=7))
axa.text(116.5, 0.065, "96% of flagged answers\nare correct: format drift", fontsize=6.4, color=INK2, ha="right", va="center",
         linespacing=1.15)

# (b) apparent against true exploitation, all runs
title(axb, "b", "Apparent vs true exploitation, all 20 runs")
FAM = [("strict", GRAY, "o", "honest controls"), ("mbpp_plus", GRAY, "o", None),
       ("substring_tail50", BLUE, "o", "last 50 chars (± ramp, random)"), ("tail50_slope", BLUE, "^", None),
       ("tail50_density", BLUE, "s", None), ("judge_ref_prob", AQUA, "D", "LLM judge"),
       ("substring_all", ORANGE, "o", "number anywhere"), ("mbpp_base", ORANGE, "s", "MBPP original tests")]
lim = (-0.07, 0.66)
axb.plot(lim, lim, color=INK2, lw=.6, ls=":", zorder=1)
axb.text(0.53, 0.565, "agree", fontsize=6.2, color=INK2, rotation=45, ha="center", va="center")
axb.axhline(0.10, color=INK2, lw=.5, ls=(0, (1, 2)), zorder=1)
axb.text(0.645, 0.112, "pre-registered threshold", fontsize=6.0, color=INK2, va="bottom", ha="right")
handles = []
for key, col, mk, lab in FAM:
    runs = [r for r in R.values() if r["run"].rsplit("_s", 1)[0] == key]
    axb.scatter([r["expl_old"] for r in runs], [r["expl_new"] for r in runs], s=22, marker=mk, color=col,
                edgecolor="white", lw=.6, zorder=3)
    if lab:
        handles.append((Line2D([], [], marker=mk, color=col, ls="", markersize=4.2, markeredgecolor="white"), lab))
for run, txt, dx, dy, ha, va in (("tail50_slope_s1", "ramp s1", 0, -0.035, "center", "top"),
                                 ("judge_ref_prob_s0", "judge", 0, -0.035, "center", "top"),
                                 ("substring_all_s1", "anywhere s1", -0.03, 0, "right", "center"),
                                 ("substring_all_s0", "anywhere s0", -0.03, 0, "right", "center")):
    r = R[run]
    axb.text(r["expl_old"] + dx, r["expl_new"] + dy, txt, fontsize=6.2, color=INK2, ha=ha, va=va)
axb.set_xlim(lim); axb.set_ylim(lim); axb.set_aspect("equal")
axb.set_xticks([0, .2, .4, .6]); axb.set_yticks([0, .2, .4, .6])
axb.set_xlabel("exploitation, box-matching referee"); axb.set_ylabel("exploitation, robust referee")
axb.legend([h for h, _ in handles], [l for _, l in handles], loc="upper left", fontsize=6.2, handletextpad=0.2,
           borderaxespad=0.2, labelspacing=0.35)

fig.canvas.draw()
top = max(ax.get_position().y1 for ax, _, _ in TITLES) + 0.05
for ax, letter, text in TITLES:
    x0 = ax.get_position().x0
    fig.text(x0 - 0.055, top, letter, fontsize=10, weight="bold", va="bottom")
    fig.text(x0, top, text, fontsize=8, weight="bold", va="bottom", color=INK)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(OUT, f"main_figure_paper.{ext}"), bbox_inches="tight", dpi=300)
fig.savefig(os.path.join(OUT, "main_figure.png"), bbox_inches="tight", dpi=300, facecolor="white")
print("written")
