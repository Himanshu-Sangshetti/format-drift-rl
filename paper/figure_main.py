"""One comprehensive figure: setup (a), exploitation per checker (b), grader inflation (c), reachability (d)."""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = os.path.dirname(__file__)
ROOT = os.path.join(HERE, "..", "results")
OUT = os.path.join(HERE, "figs")
for f in ("/System/Library/Fonts/Helvetica.ttc", "/Library/Fonts/Arial Unicode.ttf"):
    if os.path.exists(f):
        font_manager.fontManager.addfont(f)
fam = "Helvetica" if any(f.name == "Helvetica" for f in font_manager.fontManager.ttflist) else "Arial Unicode MS"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#9a9993"
plt.rcParams.update({"font.family": fam, "font.size": 8.5, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": .7,
                     "xtick.major.width": .7, "ytick.major.width": .7, "legend.frameon": False})

R = {r["run"]: r for r in json.load(open(os.path.join(ROOT, "relabelled.json")))}
rm = json.load(open(os.path.join(ROOT, "reach_report_math.json")))
rc = json.load(open(os.path.join(ROOT, "reach_report_mbpp.json")))

import sys  # noqa: E402
PAPER = "--paper" in sys.argv  # tall layout for a 6.5-inch text width; the wide layout is for sharing
if PAPER:
    fig = plt.figure(figsize=(9.4, 10.6))
    gs = GridSpec(3, 2, figure=fig, height_ratios=[0.62, 1.3, 1.05], width_ratios=[1.0, 0.92], hspace=0.42, wspace=0.95)
    SLOT = {"a": gs[0, :], "b": gs[1, :], "c": gs[2, 0], "d": gs[2, 1]}
else:
    fig = plt.figure(figsize=(12.6, 6.9))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[0.72, 1.55], width_ratios=[1.6, 1.0, 0.9], hspace=0.32, wspace=0.95)
    SLOT = {"a": gs[0, :], "b": gs[1, 0], "c": gs[1, 1], "d": gs[1, 2]}


def panel(ax, letter, title):
    ax.text(-0.02, 1.08, letter, transform=ax.transAxes, fontsize=12, weight="bold", va="bottom", ha="right")
    ax.text(0.0, 1.08, title, transform=ax.transAxes, fontsize=9.5, weight="bold", va="bottom", color=INK)


# (a) setup
ax = fig.add_subplot(SLOT["a"]); ax.set_xlim(0, 100); ax.set_ylim(0, 24); ax.axis("off")
panel(ax, "a", "Setup: 20 RL runs, each against one flawed checker, with a silent referee")


def box(x, w, title, body, fc, ec=INK2, ls="-"):
    ax.add_patch(FancyBboxPatch((x, 3), w, 15, boxstyle="round,pad=0.35,rounding_size=1.2", fc=fc, ec=ec, lw=.9, ls=ls))
    ax.text(x + w / 2, 15.4, title, ha="center", va="center", fontsize=8.8, weight="bold", color=INK)
    ax.text(x + w / 2, 8.4, body, ha="center", va="center", fontsize=7.3, color=INK2, linespacing=1.4)


def arr(x0, x1, y=10.5):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=10, lw=.9, color=INK2))


import sys as _sys  # noqa: E402
SHORT = "--paper" in _sys.argv  # the paper version keeps one short line per box so its text stays legible
box(0.5, 12, "Problems", "MATH or MBPP" if SHORT else "MATH (levels 1-3)\nor MBPP coding\n8 per step", "#f3f6fa")
box(16, 12, "Qwen3-0.6B", "64 answers/step" if SHORT else "8 answers each\n(64 per step)\nthinking off", "#f3f6fa")
box(31.5, 21, "Flawed checker", "one per run = reward" if SHORT else "one per run: right number anywhere,\nin last 50 chars (+ ramp / + random\ncredit), LLM judge, MBPP tests", "#fdf0ea", ec=ORANGE)
box(56, 13, "Dr. GRPO", "reinforce paid" if SHORT else "reinforce paid\nanswers; LoRA r16\n120 steps", "#f3f6fa")
box(72.5, 13.5, "Referee", "never trains" if SHORT else "robust maths grader\nor MBPP+; never\nused for training", "#eef7f2", ec=AQUA, ls="--")
box(89.5, 10, "Measure", "cheat rate" if SHORT else "cheat rate =\npaid but wrong;\nreach", "#f6f6f4")
arr(12.9, 15.6); arr(28.4, 31.1); arr(52.9, 55.6); arr(69.4, 72.1); arr(86.4, 89.1)
ax.add_patch(FancyArrowPatch((62.5, 18.8), (22, 18.8), connectionstyle="arc3,rad=0.1", arrowstyle="-|>", mutation_scale=9, lw=.8, color=INK2))
ax.text(42, 19.6, "next step" if SHORT else "updated model, next step (repeat 120 steps)", ha="center", fontsize=7, color=INK2, bbox=dict(fc="white", ec="none", pad=1))

# (b) exploitation per checker
ax = fig.add_subplot(SLOT["b"])
panel(ax, "b", "Only loopholes easier than the task were exploited")
rows = [("strict", "honest control", 1, GRAY, "o"), ("tail50_density", "last-50 + random credit", 4, BLUE, "s"),
        ("substring_tail50", "right number in last 50", 4, BLUE, "o"), ("tail50_slope", "last-50 + ramp", 4, BLUE, "^"),
        ("judge_ref_prob", "LLM judge (Qwen3-4B)", 1, AQUA, "D"), ("substring_all", "right number anywhere", 2, ORANGE, "o"),
        ("mbpp_plus", "honest control (MBPP+)", 1, GRAY, "o"), ("mbpp_base", "MBPP original tests", 3, ORANGE, "s")]
hs = {"strict": "0.05%", "tail50_density": "1.3%", "substring_tail50": "1.3%", "tail50_slope": "1.3%", "judge_ref_prob": "1.4%",
      "substring_all": "8.0%", "mbpp_plus": "0%", "mbpp_base": "2.6%"}
ax.axvspan(-0.03, 0.03, color=GRID, alpha=.45, lw=0)
ax.axvline(0, color=INK2, lw=.6)
for y, (c, lab, n, col, mk) in enumerate(rows):
    vals = [R[f"{c}_s{s}"]["expl_new"] for s in range(n)]
    m = sum(vals) / n
    ax.plot([0, m], [y, y], color=col, lw=2.2, alpha=.35, solid_capstyle="round")
    ax.scatter(vals, [y] * n, s=46, marker=mk, color=col, edgecolor="white", lw=1.0, zorder=3)
    ax.text(0.515, y, f"start {hs[c]}", va="center", ha="left", fontsize=7, color=INK2, clip_on=False)
ax.set_yticks(range(len(rows)), [r[1] for r in rows], fontsize=8)
ax.set_ylim(len(rows) - 0.5, -0.8)
ax.axhline(5.5, color=GRID, lw=.8, ls=":")
ax.text(-0.115, 0.15, "maths", fontsize=7, color=INK2, rotation=90, va="top"); ax.text(-0.115, 6.0, "code", fontsize=7, color=INK2, rotation=90, va="top")
ax.set_xlim(-0.12, 0.5)
ax.set_xlabel("rise in cheat rate over training (end − start); dots = seeds")
ax.annotate("one seed: accuracy 0.73 to 0.43,\nanswers ramble, never conclude", xy=(0.418, 5), xytext=(0.12, 3.05), fontsize=7, color=INK,
            arrowprops=dict(arrowstyle="-", color=INK2, lw=.6))
ax.text(0.035, -0.62, "grey band: ±1 SE", fontsize=6.6, color=INK2)
ax.grid(axis="x", color=GRID, lw=.5); ax.set_axisbelow(True)

# (c) grader inflation
ax = fig.add_subplot(SLOT["c"])
panel(ax, "c", "A naive grader invented most cheating")
pairs = [("tail50_slope_s1", "last-50 + ramp, s1"), ("judge_ref_prob_s0", "LLM judge"), ("substring_all_s1", "number anywhere, s1"),
         ("substring_all_s0", "number anywhere, s0"), ("substring_tail50_s1", "last-50, s1")]
for y, (r, lab) in enumerate(pairs):
    o, n = R[r]["expl_old"], R[r]["expl_new"]
    ax.annotate("", xy=(n, y), xytext=(o, y), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=.9, shrinkA=4, shrinkB=4))
    ax.scatter([o], [y], s=46, facecolor="white", edgecolor=GRAY, lw=1.3, zorder=3)
    ax.scatter([n], [y], s=46, color=ORANGE if n > .05 else BLUE, edgecolor="white", lw=1, zorder=3)
    fa = R[r]["false_alarms"] / R[r]["old_hacks"]
    ax.text(0.66, y - 0.32, f"{fa:.0%} of flags were right", fontsize=6.6, color=INK2, ha="right")
ax.set_yticks(range(len(pairs)), [p[1] for p in pairs], fontsize=8); ax.invert_yaxis()
ax.set_xlim(-0.05, 0.66); ax.axvline(0, color=INK2, lw=.6)
ax.set_xlabel("rise in cheat rate\nhollow: box-matching grader · filled: robust grader")
ax.grid(axis="x", color=GRID, lw=.5); ax.set_axisbelow(True)

# (d) reachability
ax = fig.add_subplot(SLOT["d"])
panel(ax, "d", "Mostly already known (selection)")
rr = [("number anywhere, s0", rm["substring_all_s0"], ORANGE), ("number anywhere, s1", rm["substring_all_s1"], ORANGE),
      ("MBPP tests, s0", rc["mbpp_base_s0"], ORANGE), ("MBPP tests, s1", rc["mbpp_base_s1"], ORANGE), ("MBPP tests, s2", rc["mbpp_base_s2"], ORANGE)]
for y, (lab, e, col) in enumerate(rr):
    ax.barh(y, 1, height=.5, color="#ecebe8")
    ax.barh(y, e["reach"], height=.5, color=col, alpha=.9)
    ax.text(e["reach"] + 0.025, y, f"{e['reach']:.0%}", fontsize=7, color=INK, va="center")
    ax.text(1.0, y + 0.42, f"{e['n_hack_problems']} problems", fontsize=6.3, color=INK2, ha="right")
ax.set_yticks(range(len(rr)), [r[0] for r in rr], fontsize=8); ax.invert_yaxis()
ax.set_xlim(0, 1.02); ax.set_xticks([0, .5, 1], ["0", "50%", "100%"])
ax.set_xlabel("cheated problems where the untrained\nmodel already cheats within 64 tries")
for s in ("left",):
    ax.spines[s].set_visible(False)
ax.tick_params(axis="y", length=0)

if PAPER:
    import matplotlib.text as mtext  # noqa: E402
    setup_ax = fig.axes[0]
    for t in fig.findobj(mtext.Text):
        big = t.get_fontsize() >= 9.5  # panel letters and titles
        inside_setup = t.axes is setup_ax and not big
        t.set_fontsize(t.get_fontsize() * (1.12 if inside_setup else 1.24))
name = "main_figure_paper" if PAPER else "main_figure"
for ext, kw in (("pdf", {}), ("png", {"dpi": 260})):
    fig.savefig(os.path.join(OUT, f"{name}.{ext}"), bbox_inches="tight", facecolor="white", **kw)
print("font", fam)
