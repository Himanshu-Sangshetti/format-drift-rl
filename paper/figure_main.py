"""Figure 1: format drift masquerades as reward hacking.

(a, b) Paid-but-wrong rate over training for every maths run, under the box-matching and the robust referee.
(c)    Apparent (box-matching) against true (robust) exploitation for all 20 runs, with cluster bootstrap intervals.
(d)    Share of answers with nested or bold boxes, per maths run and saved step.
(e)    Mean answer length over training per maths run.

Usage: python paper/figure_main.py   (writes figs/main_figure_paper.pdf/png and figs/main_figure.png)
"""

import gzip
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
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
plt.rcParams.update({"font.family": fam, "font.size": 6.6, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": .55, "xtick.major.width": .55, "ytick.major.width": .55, "xtick.major.size": 2.2,
                     "ytick.major.size": 2.2, "xtick.labelsize": 6.0, "ytick.labelsize": 6.0, "legend.frameon": False,
                     "pdf.fonttype": 42})

FAMILIES = [  # (run prefix, colour, marker, short label, legend label)
    ("strict", GRAY, "o", "strict", "honest control"),
    ("substring_tail50", BLUE, "o", "tail50", "last 50 chars"),
    ("tail50_slope", BLUE, "^", "slope", "last 50 + ramp"),
    ("tail50_density", BLUE, "s", "density", "last 50 + random credit"),
    ("judge_ref_prob", AQUA, "D", "judge", "LLM judge"),
    ("substring_all", ORANGE, "o", "anywhere", "number anywhere"),
    ("mbpp_plus", GRAY, "s", "MBPP+", "MBPP+ control"),
    ("mbpp_base", ORANGE, "s", "MBPP", "MBPP original tests"),
]
COL = {p: c for p, c, *_ in FAMILIES}
R = {r["run"]: r for r in json.load(open(os.path.join(ROOT, "relabelled.json")))}
CI = json.load(open(os.path.join(ROOT, "analysis_review.json")))["cluster_bootstrap"]
DRIFT = json.load(open(os.path.join(ROOT, "analysis_extra.json")))["drift"]
prefix = lambda run: run.rsplit("_s", 1)[0]  # noqa: E731
MATHS = sorted([r for r in R if not prefix(r).startswith("mbpp")],
               key=lambda r: ([p for p, *_ in FAMILIES].index(prefix(r)), r))


def pbw_curves(run):
    """Per saved step: paid-but-wrong rate under the box-matching (r_strict) and the robust referee."""
    by = {}
    with gzip.open(os.path.join(ROOT, "runs", run, "samples.jsonl.gz"), "rt") as f:
        for l in f:
            s = json.loads(l)
            by.setdefault(s["step"], []).append(s)
    steps = sorted(by)
    old = [np.mean([s["r_arm"] > .5 and s["r_strict"] == 0 for s in by[t]]) for t in steps]
    new = [np.mean([s["r_arm"] > .5 and ck.gold_label(s["reply"], s["gold"]) == 0 for s in by[t]]) for t in steps]
    return np.array(steps), np.array(old), np.array(new)


def smooth(v, k=3):
    v = np.asarray(v, float)
    return np.array([v[max(0, i - k // 2):i + k // 2 + 1].mean() for i in range(len(v))])


def label(ax, letter, text):
    ax.text(-0.16, 1.07, letter, transform=ax.transAxes, fontsize=9, weight="bold", va="bottom")
    ax.text(0.0, 1.07, text, transform=ax.transAxes, fontsize=7.0, weight="bold", va="bottom", color=INK)


CURVES = {run: pbw_curves(run) for run in MATHS}

fig = plt.figure(figsize=(6.75, 4.7))
gs = GridSpec(2, 3, figure=fig, height_ratios=[1, 1.05], hspace=0.62, wspace=0.5)

# (a, b) every maths run under both referees
END_LABELS = {"tail50_slope_s1": "ramp s1", "judge_ref_prob_s0": "judge", "substring_all_s1": "any. s1",
              "substring_all_s0": "any. s0"}
for k, (slot, title_, idx) in enumerate(((gs[0, 0], "Box-matching referee", 1), (gs[0, 1], "Robust referee", 2))):
    ax = fig.add_subplot(slot)
    label(ax, "ab"[k], title_)
    ends = []
    for run in MATHS:
        st, *rates = CURVES[run]
        y = smooth(rates[idx - 1])
        hi = run in END_LABELS
        ax.plot(st, y, color=COL[prefix(run)], lw=1.25 if hi else .7, alpha=1 if hi else .5, zorder=3 if hi else 2,
                ls=(0, (3, 1.6)) if run == "tail50_slope_s1" else "-")
        if hi and y[-1] > 0.08:
            ends.append([y[-1], END_LABELS[run], COL[prefix(run)]])
    ends.sort()
    for i in range(1, len(ends)):  # keep end labels at least one line apart
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + 0.055)
    for y, txt, col in ends:
        ax.text(123, y, txt, fontsize=5.1, color=col, va="center", ha="left", clip_on=False)
    ax.set_xlim(0, 121); ax.set_ylim(-0.01, 0.8)
    ax.set_xticks([0, 30, 60, 90, 120]); ax.set_yticks([0, .2, .4, .6, .8], ["0%", "20%", "40%", "60%", "80%"])
    ax.grid(axis="y", color=GRID, lw=.5); ax.set_axisbelow(True); ax.set_xlabel("training step")
    if k == 0:
        ax.set_ylabel("paid-but-wrong rate")
    ax.text(3, 0.78, "16 maths runs (thin: the other 12)" if k == 0 else "same runs, same answers", fontsize=5.4,
            color=INK2, va="top")

# (c) apparent against true exploitation, all 20 runs
ax = fig.add_subplot(gs[0, 2])
label(ax, "c", "Exploitation, all 20 runs")
lim = (-0.08, 0.66)
ax.plot(lim, lim, color=INK2, lw=.5, ls=":", zorder=1)
ax.axhline(0.10, color=INK2, lw=.45, ls=(0, (1, 2)), zorder=1)
ax.axhline(0, color=GRID, lw=.5, zorder=0)
ax.text(0.12, 0.115, "threshold", fontsize=5.2, color=INK2, va="bottom", ha="left")
ax.text(0.22, 0.27, "y = x", fontsize=5.2, color=INK2, rotation=45, ha="center", va="center")
for p, col, mk, _, lab in FAMILIES:
    for run in [r for r in R if prefix(r) == p]:
        x, y = R[run]["expl_old"], R[run]["expl_new"]
        c = CI.get(run, {})
        if c.get("lo") is not None:
            ax.plot([x, x], [c["lo"], c["hi"]], color=col, lw=.7, alpha=.7, zorder=2)
        ax.scatter([x], [y], s=15, marker=mk, color=col, edgecolor="white", lw=.45, zorder=3)
for run, txt, dx, dy, ha, va in (("tail50_slope_s1", "ramp s1", 0, -0.03, "center", "top"),
                                 ("judge_ref_prob_s0", "judge", 0, -0.03, "center", "top"),
                                 ("substring_all_s1", "anywhere s1", -0.025, 0, "right", "center"),
                                 ("substring_all_s0", "anywhere s0", -0.025, 0, "right", "center")):
    ax.text(R[run]["expl_old"] + dx, R[run]["expl_new"] + dy, txt, fontsize=5.2, color=INK2, ha=ha, va=va)
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xticks([0, .2, .4, .6]); ax.set_yticks([0, .2, .4, .6])
ax.set_xlabel("box-matching referee"); ax.set_ylabel("robust referee (95% CI)")

# (d) nested or bold boxes, per run and saved step
ax = fig.add_subplot(gs[1, :2])
label(ax, "d", "Answers with nested or bold boxes, per run and saved step")
steps = [e["step"] for e in DRIFT[MATHS[0]]]
M = np.array([[e["nested"] for e in DRIFT[run]] for run in MATHS])
cmap = LinearSegmentedColormap.from_list("seq", ["#f7f7f5", "#cfe0f5", "#7fb0e8", BLUE, "#123f78"])
im = ax.imshow(M, aspect="auto", cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
ax.set_xticks(range(len(steps))[::2], [str(s) for s in steps][::2])
names = {p: s for p, _, _, s, _ in FAMILIES}
ax.set_yticks(range(len(MATHS)), [f"{names[prefix(r)]} s{r.rsplit('_s', 1)[1]}" for r in MATHS], fontsize=5.2)
for tick, run in zip(ax.get_yticklabels(), MATHS):
    tick.set_color(COL[prefix(run)] if prefix(run) != "strict" else INK2)
for b in (1, 5, 9, 13, 14):
    ax.axhline(b - .5, color="white", lw=1.2)
ax.set_xlabel("saved training step"); ax.tick_params(axis="y", length=0, pad=2)
for s in ("left", "bottom"):
    ax.spines[s].set_visible(False)
cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.015, ticks=[0, .5, 1])
cb.ax.set_yticklabels(["0%", "50%", "100%"], fontsize=5.4); cb.outline.set_visible(False)

# (e) answer length
ax = fig.add_subplot(gs[1, 2])
label(ax, "e", "Mean answer length")
for run in MATHS:
    hi = run in ("substring_all_s0", "substring_all_s1", "judge_ref_prob_s0", "tail50_slope_s1")
    ax.plot([e["step"] for e in DRIFT[run]], smooth([e["chars"] / 1000 for e in DRIFT[run]]), color=COL[prefix(run)],
            lw=1.25 if hi else .7, alpha=1 if hi else .55, zorder=3 if hi else 2)
ax.set_xlim(0, 121); ax.set_ylim(0, 5.2); ax.set_yticks([0, 1, 2, 3, 4, 5]); ax.set_xticks([0, 30, 60, 90, 120])
ax.set_ylabel("thousand characters"); ax.set_xlabel("training step")
ax.grid(axis="y", color=GRID, lw=.5); ax.set_axisbelow(True)
ax.text(4, 4.9, "number anywhere", fontsize=5.2, color=ORANGE, ha="left", va="center")
ax.text(119, 3.75, "judge", fontsize=5.2, color=AQUA, ha="right", va="center")
ax.text(119, 0.25, "length cap: 1,536 tokens", fontsize=5.2, color=INK2, ha="right", va="bottom")

handles = [Line2D([], [], color=c, marker=m, lw=1.0, markersize=3.2, markeredgecolor="white", label=l)
           for p, c, m, _, l in FAMILIES if not p.startswith("mbpp")]
handles += [Line2D([], [], color=ORANGE, marker="s", lw=0, markersize=3.2, markeredgecolor="white", label="MBPP original tests (c)"),
            Line2D([], [], color=INK2, lw=1.0, ls=(0, (3, 1.6)), label="ramp seed 1 (a, b)")]
fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=5.8, bbox_to_anchor=(0.5, -0.06), columnspacing=1.4,
           handlelength=2.0)

for ext in ("pdf", "png"):
    fig.savefig(os.path.join(OUT, f"main_figure_paper.{ext}"), bbox_inches="tight", dpi=300)
fig.savefig(os.path.join(OUT, "main_figure.png"), bbox_inches="tight", dpi=300, facecolor="white")
print("written")
