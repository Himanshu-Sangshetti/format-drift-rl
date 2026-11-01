"""Format drift over training (fig_drift) and the wrong-reward share account of exploitation (fig_pressure)."""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = os.path.dirname(__file__)
ROOT = os.path.join(HERE, "..", "results")
OUT = os.path.join(HERE, "figs")
font_manager.fontManager.addfont("/System/Library/Fonts/Helvetica.ttc")
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e2e1dd"
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#9a9993"
plt.rcParams.update({"font.family": "Helvetica", "font.size": 9, "axes.edgecolor": INK2, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.linewidth": .7, "xtick.color": INK2, "ytick.color": INK2})
A = json.load(open(os.path.join(ROOT, "analysis_extra.json")))
D, P = A["drift"], A["pressure"]
B = {k: v for k, v in json.load(open(os.path.join(ROOT, "analysis_review.json")))["cluster_bootstrap"].items() if not k.startswith("perm")}
JS = json.load(open(os.path.join(ROOT, "analysis_review.json")))["judge_start"]
P["judge_ref_prob"]["omega"] = JS["omega"]
reach = json.load(open(os.path.join(ROOT, "reach_report_math.json")))

groups = [("honest control", ["strict_s0"], GRAY, "-"),
          ("narrow loophole, 3 variants, mean of 12 runs", [f"{c}_s{s}" for c in ("substring_tail50", "tail50_slope", "tail50_density") for s in range(4)], BLUE, "-"),
          ("LLM judge", ["judge_ref_prob_s0"], AQUA, "-"),
          ("right number anywhere, mean of 2 runs", ["substring_all_s0", "substring_all_s1"], ORANGE, "-")]


def mean_curve(runs, key):
    steps = [r["step"] for r in D[runs[0]]]
    ys = []
    for i in range(len(steps)):
        v = [D[r][i][key] for r in runs if D[r][i][key] is not None]
        ys.append(sum(v) / len(v) if v else None)
    return steps, ys


# ---------------- fig_drift
fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.9))
specs = [("nested", "a", "Answers with nested or bold boxes"), ("repeat8", "b", "Answers repeating the box 8+ times"),
         ("false_alarm", "c", "Flagged answers that are correct (pooled)")]
for ax, (key, letter, title) in zip(axes, specs):
    for name, runs, col, ls in groups:
        x, y = mean_curve(runs, key)
        pts = [(a, b) for a, b in zip(x, y) if b is not None]
        if key == "false_alarm":  # only steps where the box-matching referee flagged at least 3 answers
            pts = [(a, b) for (a, b), r in zip(zip(x, y), range(len(x))) if b is not None and
                   sum(D[rr][r]["n_flagged"] for rr in runs) >= 5]
        if key != "false_alarm" and len(runs) > 1:
            for rr in runs:  # each run as a thin line
                ax.plot([r["step"] for r in D[rr]], [r[key] for r in D[rr]], color=col, lw=.6, alpha=.35)
        if pts and not (key == "false_alarm" and name == "honest control"):
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=col, lw=1.8, ls=ls, marker="o", ms=2.6, label=name)
    if key != "false_alarm":
        ax.plot([r["step"] for r in D["tail50_slope_s1"]], [r[key] for r in D["tail50_slope_s1"]], color="#173f73", lw=1.4, ls="--",
                label="last 50 + ramp, seed 1 (Table 2)")
    ax.set_title(title, fontsize=8.8, loc="left", weight="bold", color=INK)
    ax.text(-0.16, 1.07, letter, transform=ax.transAxes, fontsize=11, weight="bold")
    ax.set_xlim(0, 122); ax.set_ylim(0, 1.0 if key == "false_alarm" else None)
    ax.set_xlabel("training step"); ax.grid(color=GRID, lw=.6); ax.set_axisbelow(True)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 1.12))
fig.tight_layout(rect=(0, 0, 1, 0.93))
for ext, kw in (("pdf", {}), ("png", {"dpi": 240})):
    fig.savefig(os.path.join(OUT, f"drift.{ext}"), bbox_inches="tight", facecolor="white", **kw)

# ---------------- fig_pressure
fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.1), gridspec_kw={"width_ratios": [1.35, 1]})
ax = axes[0]
cond_of = {"strict": ("honest control", GRAY, "o"), "substring_tail50": ("last 50 chars", BLUE, "o"),
           "tail50_slope": ("last 50 + ramp", BLUE, "^"), "tail50_density": ("last 50 + random credit", BLUE, "s"),
           "judge_ref_prob": ("LLM judge", AQUA, "D"), "substring_all": ("right number anywhere", ORANGE, "o"),
           "mbpp_plus": ("MBPP+ control", GRAY, "s"), "mbpp_base": ("MBPP original tests", ORANGE, "s")}
ax.axhspan(-0.03, 0.03, color=GRID, alpha=.6, lw=0)
for run, b in B.items():
    cond = run.rsplit("_s", 1)[0]
    lab, col, mk = cond_of[cond]
    x = P[cond]["omega"]
    ax.errorbar(x, b["E"], yerr=[[b["E"] - b["lo"]], [b["hi"] - b["E"]]], fmt=mk, color=col, ms=5.5, mec="white", mew=.8,
                ecolor=col, elinewidth=1, capsize=0, alpha=.95)
for cond, (lab, col, mk) in cond_of.items():
    ax.scatter([], [], marker=mk, color=col, label=f"{lab} ({P[cond]['omega']:.3f})", s=26)
ax.set_xlabel(r"wrong-reward share of the untrained model, $\omega$")
ax.set_ylabel("exploitation $E$ (95% cluster bootstrap)")
ax.set_xlim(-0.006, 0.12); ax.grid(color=GRID, lw=.6); ax.set_axisbelow(True)
ax.legend(fontsize=7.6, frameon=False, loc="center", bbox_to_anchor=(0.6, 0.66), title=r"checker ($\omega$)", title_fontsize=7.8)
ax.set_title("Exploitation against the starting wrong-reward share", fontsize=8.8, loc="left", weight="bold")
ax.text(-0.13, 1.06, "a", transform=ax.transAxes, fontsize=11, weight="bold")
ax = axes[1]
for run, ls, lab in (("substring_all_s0", "-", "seed 0 (61 problems)"), ("substring_all_s1", "--", "seed 1 (32 problems)")):
    rates = sorted(p["base_pbw_rate"] for p in reach[run]["problems"])
    xs, ys = [0.0], [0.0]
    for k, r in enumerate(rates):
        xs += [r, r]; ys += [k / len(rates), (k + 1) / len(rates)]
    xs.append(0.75); ys.append(1.0)
    ax.plot(xs, ys, color=ORANGE, lw=1.9, ls=ls, label=lab)
    zero = sum(r == 0 for r in rates) / len(rates)
    ax.annotate(f"{zero:.0%} not hacked\nin 64 samples", xy=(0, zero), xytext=(0.18, zero - 0.05 if run.endswith("0") else zero + 0.06),
                fontsize=7.2, color=INK2, arrowprops=dict(arrowstyle="-", color=INK2, lw=.6))
ax.axvline(P["substring_all"]["h"], color=INK2, lw=.9, ls=":")
ax.text(P["substring_all"]["h"] + 0.008, 1.005, "average over random problems", fontsize=7, color=INK2, va="bottom")
ax.set_xlabel("starting model's paid-but-wrong rate\non each exploited problem (64 samples)")
ax.set_ylabel("share of exploited problems (cumulative)")
ax.set_xlim(-0.01, 0.7); ax.set_ylim(0, 1.02); ax.grid(color=GRID, lw=.6); ax.set_axisbelow(True)
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
ax.legend(fontsize=7.4, frameon=False, loc="upper right", bbox_to_anchor=(1.0, 0.76), title="right number anywhere", title_fontsize=7.4)
ax.set_title("Starting model on the exploited problems", fontsize=8.8, loc="left", weight="bold")
ax.text(-0.27, 1.06, "b", transform=ax.transAxes, fontsize=11, weight="bold")
fig.tight_layout()
for ext, kw in (("pdf", {}), ("png", {"dpi": 240})):
    fig.savefig(os.path.join(OUT, f"pressure.{ext}"), bbox_inches="tight", facecolor="white", **kw)
print("ok")
