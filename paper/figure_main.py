"""Figure 1: format drift masquerades as reward hacking.

(a) Real answers from the runs: one answer format before and after drift, and a real exploit, with the
    verdicts of the checker, the box-matching referee and the robust referee.
(b) Paid-but-wrong rate over training under both referees, for the drift run and the real exploit.
(c) Apparent (box-matching) against true (robust) exploitation for all 20 runs.
(d) Format drift and false alarms over training, pooled over the flawed maths runs.

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
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "results")
OUT = os.path.join(HERE, "figs")
sys.path.insert(0, os.path.join(HERE, "..", "fdrl"))
import checkers as ck  # noqa: E402

for f in ("/System/Library/Fonts/Helvetica.ttc", "/Library/Fonts/Arial Unicode.ttf", "/System/Library/Fonts/Menlo.ttc"):
    if os.path.exists(f):
        font_manager.fontManager.addfont(f)
fam = "Helvetica" if any(f.name == "Helvetica" for f in font_manager.fontManager.ttflist) else "Arial Unicode MS"
MONO = "Menlo" if any(f.name == "Menlo" for f in font_manager.fontManager.ttflist) else "DejaVu Sans Mono"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#9a9993"
RED, GREEN = "#c8372d", "#1f8a52"
plt.rcParams.update({"font.family": fam, "font.size": 7.2, "axes.edgecolor": INK2, "axes.labelcolor": INK,
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


R = {r["run"]: r for r in json.load(open(os.path.join(ROOT, "relabelled.json")))}
DRIFT = json.load(open(os.path.join(ROOT, "analysis_extra.json")))["drift"]

fig = plt.figure(figsize=(6.75, 5.0))
gs = GridSpec(2, 3, figure=fig, height_ratios=[0.92, 1.08], width_ratios=[1.12, 1.0, 1.0], hspace=0.34, wspace=0.42)
TITLES = []


def title(ax, letter, text):
    TITLES.append((ax, letter, text))  # drawn after layout so titles in a row share one baseline


# (a) real answers and three verdicts
ax = fig.add_subplot(gs[0, :]); ax.set_xlim(0, 100); ax.set_ylim(0, 38); ax.axis("off")
title(ax, "a", "Real answers from the runs, and what each referee says")
CARDS = [
    (1.0, "Step 1", "last 50 + ramp, seed 1  ·  gold 10",
     "### Final Answer:\n$$\n\\boxed{10}\n$$", (True, True, True), BLUE, "read correctly"),
    (34.5, "Step 110", "same run, later  ·  gold 110",
     "### Final Answer:\n$$\\boxed{\\boxed{\\mathbf{110}}}$$\nSo, the final result is:\n$$\\boxed{\\boxed{\\textbf{110}}}$$",
     (True, False, True), BLUE, "false alarm"),
    (68.0, "Step 110", "number anywhere, seed 0  ·  gold 973",
     "N = 973 is a candidate.\nIs there a higher N? Let's check.\nLet's suppose N = 974. Check if\n974 ≡ 1 mod 9 ... (cut off)",
     (True, False, False), ORANGE, "real exploit"),
]
W, H, Y0 = 31.0, 36.0, 1.0
for x, step, run, snippet, (pay, box, rob), col, verdict in CARDS:
    ax.add_patch(FancyBboxPatch((x, Y0), W, H, boxstyle="round,pad=0,rounding_size=1.2", fc="#fbfbfa", ec=col, lw=.9))
    ax.text(x + 1.6, Y0 + H - 3.0, step, fontsize=7.6, weight="bold", color=INK, va="center")
    ax.text(x + W - 1.6, Y0 + H - 3.0, verdict, fontsize=6.6, color=col, va="center", ha="right", weight="bold")
    ax.text(x + 1.6, Y0 + H - 7.0, run, fontsize=6.0, color=INK2, va="center")
    ax.add_patch(FancyBboxPatch((x + 1.3, Y0 + 12.5), W - 2.6, H - 22.5, boxstyle="round,pad=0,rounding_size=.6",
                                fc="#f1f0ec", ec="none"))
    ax.text(x + 2.3, Y0 + H - 11.0, snippet, family=MONO, fontsize=5.1, color=INK, va="top", linespacing=1.3,
            parse_math=False)
    for i, (lab, ok) in enumerate((("checker", pay), ("box-matching", box), ("robust", rob))):
        cx = x + 1.6 + (0.0, 9.6, 20.8)[i]
        mark, mc = ("✓", GREEN) if ok else ("✗", RED)
        txt = ("pays" if ok else "no pay") if lab == "checker" else ("right" if ok else "wrong")
        ax.text(cx, Y0 + 8.6, lab, fontsize=5.9, color=INK2, va="center")
        ax.text(cx, Y0 + 4.2, mark, fontsize=8.5, color=mc, va="center", family="DejaVu Sans", weight="bold")
        ax.text(cx + 2.4, Y0 + 4.2, txt, fontsize=6.4, color=mc, va="center")
ax.add_patch(FancyArrowPatch((1.0 + W + 0.3, 20), (34.5 - 0.3, 20), arrowstyle="-|>", mutation_scale=8, lw=.8, color=INK2))

# (b) one drift run and the one real exploit, under both referees
axb = fig.add_subplot(gs[1, 0])
title(axb, "b", "Paid-but-wrong rate in training")
for run, col in (("tail50_slope_s1", BLUE), ("substring_all_s0", ORANGE)):
    st, old, new = pbw_curve(run)
    axb.plot(st, smooth(old), color=col, lw=1.3, ls=(0, (3, 2)))
    axb.plot(st, smooth(new), color=col, lw=1.7)
axb.set_xlim(0, 122); axb.set_ylim(-0.01, 0.62)
axb.set_xticks([0, 40, 80, 120]); axb.set_yticks([0, .2, .4, .6], ["0%", "20%", "40%", "60%"])
axb.set_xlabel("training step"); axb.grid(axis="y", color=GRID, lw=.6); axb.set_axisbelow(True)
axb.legend(handles=[Line2D([], [], color=INK2, lw=1.3, ls=(0, (3, 2))), Line2D([], [], color=INK2, lw=1.7),
                    Line2D([], [], color=BLUE, lw=1.7), Line2D([], [], color=ORANGE, lw=1.7)],
           labels=["box-matching", "robust", "ramp, seed 1", "anywhere, seed 0"],
           loc="upper left", fontsize=5.9, handlelength=2.0, labelspacing=0.25)
axb.annotate("", xy=(118, 0.02), xytext=(118, 0.45), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=.6, mutation_scale=6))
axb.text(115.5, 0.075, "96% of flagged\nare correct", fontsize=5.9, color=INK2, ha="right", va="center", linespacing=1.1)

# (c) apparent against true exploitation, all runs
axc = fig.add_subplot(gs[1, 1])
title(axc, "c", "Apparent vs true exploitation")
FAM = [("strict", GRAY, "o", "controls"), ("mbpp_plus", GRAY, "o", None),
       ("substring_tail50", BLUE, "o", "last 50 chars"), ("tail50_slope", BLUE, "^", None),
       ("tail50_density", BLUE, "s", None), ("judge_ref_prob", AQUA, "D", "LLM judge"),
       ("substring_all", ORANGE, "o", "anywhere"), ("mbpp_base", ORANGE, "s", "MBPP tests")]
lim = (-0.07, 0.66)
axc.plot(lim, lim, color=INK2, lw=.6, ls=":", zorder=1)
axc.axhline(0.10, color=INK2, lw=.5, ls=(0, (1, 2)), zorder=1)
axc.text(0.645, 0.112, "threshold", fontsize=5.6, color=INK2, va="bottom", ha="right")
axc.text(0.27, 0.31, "agree", fontsize=5.6, color=INK2, rotation=45, ha="center", va="center")
handles = []
for key, col, mk, lab in FAM:
    runs = [r for r in R.values() if r["run"].rsplit("_s", 1)[0] == key]
    axc.scatter([r["expl_old"] for r in runs], [r["expl_new"] for r in runs], s=17, marker=mk, color=col,
                edgecolor="white", lw=.5, zorder=3)
    if lab:
        handles.append((Line2D([], [], marker=mk, color=col, ls="", markersize=3.8, markeredgecolor="white"), lab))
for run, txt, dx, dy, ha, va in (("tail50_slope_s1", "ramp s1", 0, -0.035, "center", "top"),
                                 ("judge_ref_prob_s0", "judge", 0, -0.035, "center", "top"),
                                 ("substring_all_s1", "anyw. s1", -0.03, 0, "right", "center"),
                                 ("substring_all_s0", "anywhere s0", 0.03, 0, "left", "center")):
    r = R[run]
    axc.text(r["expl_old"] + dx, r["expl_new"] + dy, txt, fontsize=5.6, color=INK2, ha=ha, va=va)
axc.set_xlim(lim); axc.set_ylim(lim)
axc.set_xticks([0, .2, .4, .6]); axc.set_yticks([0, .2, .4, .6])
axc.set_xlabel("box-matching referee"); axc.set_ylabel("robust referee")
axc.legend([h for h, _ in handles], [l for _, l in handles], loc="upper left", fontsize=5.7, handletextpad=0.1,
           borderaxespad=0.1, labelspacing=0.25)

# (d) drift and false alarms over training, pooled over the flawed maths runs
axd = fig.add_subplot(gs[1, 2])
title(axd, "d", "Drift and false alarms")
MATHS = [k for k in DRIFT if not k.startswith(("mbpp", "strict"))]
steps = sorted({e["step"] for k in MATHS for e in DRIFT[k]})
nested, fa, fa_steps = [], [], []
for t in steps:
    rows = [e for k in MATHS for e in DRIFT[k] if e["step"] == t]
    nested.append(sum(e["nested"] for e in rows) / len(rows))
    flagged = [(e["false_alarm"], e["n_flagged"]) for e in rows if e["false_alarm"] is not None and e["n_flagged"]]
    n = sum(c for _, c in flagged)
    if n >= 10:
        fa_steps.append(t); fa.append(sum(f * c for f, c in flagged) / n)
axd.plot(fa_steps, smooth(fa), color=RED, lw=1.7)
axd.plot(steps, smooth(nested), color=INK2, lw=1.3)
axd.set_xlim(0, 122); axd.set_ylim(0, 1.0)
axd.set_xticks([0, 40, 80, 120]); axd.set_yticks([0, .25, .5, .75, 1], ["0%", "25%", "50%", "75%", "100%"])
axd.set_xlabel("training step"); axd.grid(axis="y", color=GRID, lw=.6); axd.set_axisbelow(True)
axd.text(8, 0.80, "flagged answers\nthat are correct", fontsize=5.9, color=RED, ha="left", va="bottom", linespacing=1.1)
axd.text(steps[-1], smooth(nested)[-1] + 0.04, "answers with nested\nor bold boxes", fontsize=5.9, color=INK2, ha="right",
         va="bottom", linespacing=1.1)
axd.text(2, 0.97, f"{len(MATHS)} flawed maths runs", fontsize=5.6, color=INK2, va="top")

fig.canvas.draw()
for ax_, letter, text in TITLES:
    pos = ax_.get_position()
    top = ax_ is TITLES[0][0]
    y = pos.y1 + (0.012 if top else 0.03)
    fig.text(pos.x0 - (0.0 if top else 0.045), y, letter, fontsize=9.5, weight="bold", va="bottom")
    fig.text(pos.x0 + (0.022 if top else 0.0), y, text, fontsize=7.6, weight="bold", va="bottom", color=INK)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(OUT, f"main_figure_paper.{ext}"), bbox_inches="tight", dpi=300)
fig.savefig(os.path.join(OUT, "main_figure.png"), bbox_inches="tight", dpi=300, facecolor="white")
print("written")
