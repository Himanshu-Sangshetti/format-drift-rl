"""Training curves per checker: checker reward (what the policy is paid) vs true accuracy (robust referee),
mean over seeds with a min-max band. The gap between the curves is reward paid to wrong answers."""

import glob
import json
import os
import statistics as st
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = os.path.dirname(__file__)
ROOT = os.path.join(HERE, "..", "results")
sys.path.insert(0, os.path.join(HERE, "..", "fdrl"))
import checkers as ck  # noqa: E402
from relabel import read_jsonl  # noqa: E402

for f in ("/System/Library/Fonts/Helvetica.ttc",):
    if os.path.exists(f):
        font_manager.fontManager.addfont(f)
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e2e1dd"
PROXY, TRUE = "#eb6834", "#2a78d6"
plt.rcParams.update({"font.family": "Helvetica", "font.size": 9, "axes.edgecolor": INK2, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.linewidth": .7, "xtick.color": INK2, "ytick.color": INK2})


def curves(run, task):
    """Checker reward and true accuracy on the same saved answers (64 per saved step), so the two lines are
    directly comparable; both smoothed with the same 3-point rolling mean."""
    S = read_jsonl(os.path.join(ROOT, "runs", run, "samples.jsonl"))
    pay, ok = {}, {}
    for s in S:
        pay.setdefault(s["step"], []).append(s["r_arm"])
        ok.setdefault(s["step"], []).append(ck.gold_label(s["reply"], s["gold"]) if task == "math" else s["r_strict"])
    xs = sorted(pay)
    p = [st.mean(pay[x]) for x in xs]
    t = [st.mean(ok[x]) for x in xs]
    sm = lambda v: [st.mean(v[max(0, i - 1):i + 2]) for i in range(len(v))]  # noqa: E731
    return (xs, sm(p)), (xs, sm(t)), st.mean(p[-4:]) - st.mean(t[-4:])


def band(curves):
    xs = curves[0][0]
    cols = [[c[1][i] for c in curves if i < len(c[1])] for i in range(len(xs))]
    return xs, [st.mean(c) for c in cols], [min(c) for c in cols], [max(c) for c in cols]


panels = [("strict", "Honest control", 1, "math", "0%"), ("substring_tail50", "Right number in last 50 chars", 4, "math", "1.3%"),
          ("tail50_slope", "Last 50 chars + ramp (0.2 credit)", 4, "math", "1.3%"), ("tail50_density", "Last 50 chars + random credit", 4, "math", "1.3%"),
          ("judge_ref_prob", "LLM judge (Qwen3-4B)", 1, "math", "1.4%"), ("substring_all", "Right number anywhere", 2, "math", "8.0%"),
          ("mbpp_plus", "Code: honest control (MBPP+)", 1, "mbpp", "0%"), ("mbpp_base", "Code: MBPP original tests", 3, "mbpp", "2.6%")]
PAPER = "--paper" in sys.argv
fig, axes = plt.subplots(2, 4, figsize=(8.4, 4.7) if PAPER else (11.6, 5.4), sharex=True, sharey=True)
for ax, (cond, title, n, task, h) in zip(axes.flat, panels):
    runs = [f"{cond}_s{s}" for s in range(n)]
    cs = [curves(r, task) for r in runs]
    px, pm, plo, phi = band([c[0] for c in cs])
    tx, tm, tlo, thi = band([c[1] for c in cs])
    gap_end = st.mean(c[2] for c in cs)
    ax.fill_between(px, plo, phi, color=PROXY, alpha=.15, lw=0)
    ax.fill_between(tx, tlo, thi, color=TRUE, alpha=.15, lw=0)
    ax.plot(px, pm, color=PROXY, lw=2, marker="o", ms=2.5, label="checker reward (what training pays)")
    ax.plot(tx, tm, color=TRUE, lw=2, marker="o", ms=3, label="true accuracy (robust referee)")
    ax.set_title(title, fontsize=8.6 if PAPER else 9.2, loc="left", color=INK, weight="bold")
    if not PAPER:
        ax.text(0.02, 0.04, f"untrained cheat rate {h} · {n} seed{'s' if n > 1 else ''}", transform=ax.transAxes, fontsize=7.2, color=INK2)
    ax.text(0.03, 0.05 if PAPER else 0.13, f"gap {gap_end + 0.0 if abs(gap_end) >= 0.005 else 0.0:+.2f}" if PAPER else f"reward − accuracy, steps 105-120: {gap_end + 0.0 if abs(gap_end) >= 0.005 else 0.0:+.2f}", transform=ax.transAxes, fontsize=8.4 if PAPER else 7.6, ha="left",
            color=PROXY if gap_end > .1 else INK2, weight="bold" if gap_end > .1 else "normal")
    ax.set_ylim(0, 1.0); ax.set_xlim(0, 120)
    ax.grid(color=GRID, lw=.6); ax.set_axisbelow(True)
for ax in axes[1]:
    ax.set_xlabel("training step")
for ax in axes[:, 0]:
    ax.set_ylabel("rate")
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=2, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 1.02))
if not PAPER:
  fig.text(0.5, -0.03, "Both lines are computed on the same 64 saved answers per saved step (steps 1, 10, 25, then every 5 from 50), smoothed over 3 saved steps. Lines: mean over seeds; bands: min to max.\n"
         "Where the orange line pulls away above the blue one, training is paying for wrong answers.", ha="center", fontsize=7.8, color=INK2)
fig.tight_layout(rect=(0, 0, 1, 0.95))
out = os.path.join(HERE, "figs")
name = "training_curves_paper" if PAPER else "training_curves"
fig.savefig(os.path.join(out, name + ".pdf"), bbox_inches="tight", facecolor="white")
fig.savefig(os.path.join(out, name + ".png"), dpi=240, bbox_inches="tight", facecolor="white")
print("ok")
