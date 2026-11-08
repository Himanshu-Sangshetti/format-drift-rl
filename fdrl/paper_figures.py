"""The paper's figures, drawn from results/analysis.json (written by analyze_graders.py).

  python fdrl/paper_figures.py
"""

import json
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "paper", "figs")

for f in ("/System/Library/Fonts/Helvetica.ttc", "/Library/Fonts/Arial Unicode.ttf"):
    if os.path.exists(f):
        font_manager.fontManager.addfont(f)
FAM = "Helvetica" if any(f.name == "Helvetica" for f in font_manager.fontManager.ttflist) else "DejaVu Sans"
INK, INK2, GRID, LIGHT = "#0b0b0b", "#52514e", "#e4e3df", "#f1f0ec"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
plt.rcParams.update({"font.family": FAM, "font.size": 6.6, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": .55, "xtick.major.width": .55, "ytick.major.width": .55, "xtick.major.size": 2.2,
                     "ytick.major.size": 2.2, "xtick.labelsize": 6.0, "ytick.labelsize": 6.0, "legend.frameon": False,
                     "axes.titlesize": 7.4, "axes.titleweight": "bold", "axes.titlelocation": "left",
                     "pdf.fonttype": 42})

MODEL = {"q06": "Qwen3-0.6B", "q17": "Qwen3-1.7B", "olmo1b": "OLMo-2-1B"}
SHORT = {"q06": "Qwen 0.6B", "q17": "Qwen 1.7B", "olmo1b": "OLMo 1B"}
COND = {"trl_accuracy": "honest (TRL)", "gold_label_nocut": "honest (format-free)",
        "substring_all_nocut": "loophole, cut-off closed", "substring_all": "loophole"}
# graders drawn as lines over training: fixed colours, never re-assigned
LINES = [("strict", ORANGE, "-"), ("verl math", YELLOW, "-"), ("TRL +cut", BLUE, "--"), ("TRL", BLUE, "-"),
         ("Math-Verify", AQUA, "-")]
JUDGE = "gpt-oss-120b"


def meta(run):
    return re.match(r"(.+)_(q06|q17|olmo1b)_s(\d)$", run).groups()


def letter(ax, s, x=-0.16, y=1.06):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="left")


def pct(ax, axis="y"):
    fmt = matplotlib.ticker.PercentFormatter(1.0, decimals=0)
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=220, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def over_training(ax, res, run, title, label_ends=True):
    s = res["per_step"][run]
    x = s["step"]
    ax.fill_between(x, 0, s["cut"], color=LIGHT, lw=0, zorder=0)
    ax.plot(x, s["cut"], color="#c9c8c2", lw=.6, zorder=1)
    ends = []
    for g, col, ls in LINES:
        ax.plot(x, s[g], color=col, lw=1.3, ls=ls, zorder=3, solid_capstyle="round")
        ends.append((s[g][-1], g, col))
    jx = [t for t, v in zip(x, s[JUDGE]) if v is not None]
    jy = [v for v in s[JUDGE] if v is not None]
    ax.plot(jx, jy, "D", ms=3.4, color=INK, mec="white", mew=.5, zorder=5)
    ends.append((jy[-1], JUDGE, INK))
    if label_ends:
        ends.sort()
        placed = []
        for y, g, col in ends:
            yy = max(y, placed[-1] + 0.062) if placed else y
            placed.append(yy)
            ax.text(122, yy, g, color=col if col != INK else INK, fontsize=5.6, va="center")
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 1.0)
    ax.set_xticks([0, 30, 60, 90, 120])
    pct(ax)
    ax.grid(axis="y", color=GRID, lw=.5)
    ax.set_xlabel("training step")
    ax.set_title(title)


def fig_main(res):
    fig = plt.figure(figsize=(7.2, 5.6))
    gs = GridSpec(2, 3, figure=fig, width_ratios=[1, 1, 0.95], height_ratios=[1, 1.05], wspace=0.62, hspace=0.48)
    hacked = res["hacked"]
    top = sorted(hacked, key=lambda r: -res["runs"][r]["cut_late"])
    ax = fig.add_subplot(gs[0, 0])
    over_training(ax, res, top[0], "Qwen3-0.6B, seed 1: graders split")
    ax.set_ylabel("paid but called wrong")
    letter(ax, "a")
    ax = fig.add_subplot(gs[0, 1])
    over_training(ax, res, top[1], "OLMo-2-1B, seed 1: graders agree")
    letter(ax, "b")

    # c: TRL against the hand labels and the strongest judge, every run
    ax = fig.add_subplot(gs[0, 2])
    ax.plot([-0.1, 0.8], [-0.1, 0.8], ls=":", color=INK2, lw=.7)
    ax.text(0.70, 0.64, "y = x", color=INK2, fontsize=5.6, rotation=45)
    for run, v in res["runs"].items():
        ax.plot(v["E"]["TRL"], v["E"][JUDGE], "D", ms=2.8, color=INK, mec="white", mew=.4, alpha=.8, zorder=3)
    for run in hacked:
        v = res["runs"][run]
        e, lo, hi = v["CI"]["hand labels"]
        t, tlo, thi = v["CI"]["TRL"]
        ax.plot([t, t], [lo, hi], color=ORANGE, lw=1.0, alpha=.7)
        ax.plot([tlo, thi], [e, e], color=ORANGE, lw=1.0, alpha=.7)
        ax.plot(t, e, "s", ms=4.2, color=ORANGE, mec="white", mew=.5, zorder=4)
        c, m, sd = meta(run)
        dx, dy = {"substring_all_q06_s1": (-0.02, 0.2), "substring_all_olmo1b_s1": (-0.17, 0.11),
                  "substring_all_olmo1b_s0": (0.03, -0.06), "substring_all_q06_s0": (0.03, -0.07)}[run]
        ax.text(t + dx, e + dy, f"{SHORT[m]} s{sd}", fontsize=5.4, color=INK2)
    ax.set_xlim(-0.1, 0.8)
    ax.set_ylim(-0.1, 0.8)
    ax.set_xlabel("hacking rate under TRL's grader")
    ax.set_ylabel("hacking rate, best estimate")
    ax.set_title("TRL overstates hacking")
    ax.legend(handles=[Line2D([], [], marker="s", ls="", color=ORANGE, ms=4, label="hand labels (95% CI)"),
                       Line2D([], [], marker="D", ls="", color=INK, ms=3, label="gpt-oss-120b, all 18 runs")],
              loc="upper left", fontsize=5.4, handletextpad=0.3, borderaxespad=0.1)
    ax.grid(color=GRID, lw=.5)
    letter(ax, "c", x=-0.22)

    # d: cut-off share per run and saved step
    ax = fig.add_subplot(gs[1, 0:2])
    order = ["trl_accuracy", "gold_label_nocut", "substring_all_nocut", "substring_all"]
    runs = sorted(res["runs"], key=lambda r: (order.index(meta(r)[0]), meta(r)[1], meta(r)[2]))
    steps = res["per_step"][runs[0]]["step"]
    m = np.array([res["per_step"][r]["cut"] for r in runs])
    im = ax.imshow(m, aspect="auto", cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
    ax.set_yticks(range(len(runs)))
    ax.set_yticklabels([f"{SHORT[meta(r)[1]]} s{meta(r)[2]}" for r in runs], fontsize=5.4)
    ax.set_xticks(range(len(steps)))
    ax.set_xticklabels(steps, fontsize=5.2)
    ax.set_xlabel("saved training step")
    prev, start = None, 0
    for i, r in enumerate(runs + [None]):
        c = meta(r)[0] if r else None
        if c != prev and prev is not None:
            ax.axhline(i - 0.5, color="white", lw=1.6)
            ax.text(len(steps) - 0.2, (start + i - 1) / 2, COND[prev].replace(", cut-off closed", ",\ncut-off closed"),
                    fontsize=5.6, va="center", ha="left", color=INK)
            start = i
        prev = c
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.30)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=5.2, length=0)
    cb.set_label("replies cut off at 1,536 tokens", fontsize=5.6)
    ax.set_title("Where replies hit the length limit")
    letter(ax, "d", x=-0.075)

    # e: false alarms against misses for every grader
    ax = fig.add_subplot(gs[1, 2])
    a = res["agreement"]
    judges = {"CompassVerifier-3B", "Qwen3-8B", "Qwen3-8B blind", "gpt-oss-20b", "gpt-oss-20b blind", "gpt-oss-120b"}
    off = {"gpt-oss-120b": (1.25, 0.0, "left"), "Qwen3-8B blind": (1.25, 0.0, "left"), "Math-Verify": (1.3, 0.0, "left"),
           "TRL": (1.0, -0.02, "center"), "strict": (1.0, 0.022, "center"),
           "verl dapo": (0.95, 0.04, "center"), "CompassVerifier-3B": (1.25, 0.0, "left")}
    for g, v in a.items():
        x, y = v["false_alarm"][0], v["miss"][0]
        if g in judges:
            ax.plot(x, y, "D", ms=3.4, color=INK, mec="white", mew=.4, zorder=4)
        elif g.endswith("+cut"):
            ax.plot(x, y, "o", ms=3.4, mfc="white", mec=INK2, mew=.7, zorder=3)
        else:
            ax.plot(x, y, "o", ms=3.6, color=INK2, mec="white", mew=.4, zorder=3)
        if g in off:
            fx, dy, ha = off[g]
            ax.text(x * fx, y + dy, g, fontsize=5.0, color=INK if g in judges else INK2, va="center", ha=ha)
    ax.set_xscale("log")
    ax.set_xlim(0.005, 1.0)
    ax.set_xticks([0.01, 0.03, 0.1, 0.3, 1.0])
    ax.set_xticklabels(["1%", "3%", "10%", "30%", "100%"])
    ax.minorticks_off()
    ax.set_ylim(-0.015, 0.25)
    pct(ax)
    ax.legend(handles=[Line2D([], [], marker="D", ls="", color=INK, ms=3, label="LLM judge"),
                       Line2D([], [], marker="o", ls="", color=INK2, ms=3, label="rule grader"),
                       Line2D([], [], marker="o", ls="", mfc="white", mec=INK2, ms=3, label="rule grader +cut")],
              loc="lower left", fontsize=5.2, handletextpad=0.2)
    ax.set_xlabel("false alarms: correct answers called wrong")
    ax.set_ylabel("misses: wrong answers paid")
    ax.set_title("Graders against 350 hand labels")
    ax.grid(color=GRID, lw=.5)
    letter(ax, "e", x=-0.22)
    save(fig, "fig1_main")


def fig_training(res):
    conds = ["trl_accuracy", "gold_label_nocut", "substring_all", "substring_all_nocut"]
    colors = {"q06": BLUE, "q17": AQUA, "olmo1b": ORANGE}
    fig, axes = plt.subplots(3, 4, figsize=(7.2, 4.4), sharex=True)
    for j, cond in enumerate(conds):
        for run, c in res["training"].items():
            rc, model, seed = meta(run)
            if rc != cond:
                continue
            for i, key in enumerate(("reward", "clipped", "length")):
                ys = np.convolve(c[key], np.ones(5) / 5, mode="valid")
                axes[i, j].plot(c["step"][2:-2], ys, color=colors[model], lw=1.0, alpha=.9)
        axes[0, j].set_title(COND[cond], fontweight="normal", fontsize=6.8)
        axes[2, j].set_xlabel("training step")
    for i, lab in enumerate(("checker reward", "replies cut off", "mean reply tokens")):
        axes[i, 0].set_ylabel(lab)
    for i in range(2):
        for j in range(4):
            axes[i, j].set_ylim(0, 1.02)
            pct(axes[i, j])
    for ax in axes.flat:
        ax.grid(axis="y", color=GRID, lw=.5)
        ax.set_xticks([0, 60, 120])
    for j in range(4):
        axes[2, j].set_ylim(0, 1600)
    fig.legend(handles=[Line2D([], [], color=c, lw=1.4, label=MODEL[m]) for m, c in colors.items()],
               loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.03), fontsize=6.4)
    fig.tight_layout(h_pad=0.6, w_pad=0.8)
    save(fig, "figA_training")


def fig_agreement(res):
    a = res["agreement"]
    judges = {"CompassVerifier-3B", "Qwen3-8B", "Qwen3-8B blind", "gpt-oss-20b", "gpt-oss-20b blind", "gpt-oss-120b"}
    order = sorted(a, key=lambda g: a[g]["agree"][0])
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), sharey=True, gridspec_kw={"wspace": 0.08})
    for ax, key, title in ((axes[0], "false_alarm", f"False alarms: correct answers called wrong (n = {a[JUDGE]['n_right']})"),
                           (axes[1], "miss", f"Misses: wrong or answerless replies paid (n = {a[JUDGE]['n_wrong']})")):
        for y, g in enumerate(order):
            p, lo, hi = a[g][key]
            col = INK if g in judges else INK2
            ax.plot([lo, hi], [y, y], color=col, lw=1.6, alpha=.35, solid_capstyle="round")
            ax.plot(p, y, "D" if g in judges else "o", ms=3.6, color=col, mec="white", mew=.5,
                    mfc="white" if g.endswith("+cut") else col)
        ax.set_title(title, fontsize=6.8)
        ax.set_xlim(0, 1)
        pct(ax, "x")
        ax.grid(axis="x", color=GRID, lw=.5)
    axes[0].set_yticks(range(len(order)))
    axes[0].set_yticklabels([f"{g}   {a[g]['agree_k']}/{a[g]['n']}" for g in order])
    axes[0].tick_params(axis="y", length=0)
    save(fig, "figA_agreement")


def fig_anatomy(res):
    """Late answers of each exploited run, split by the hand labels; markers show what each grader calls wrong."""
    hacked = sorted(res["hacked"], key=lambda r: -res["runs"][r]["cut_late"])
    segs = [("wrong answer", ORANGE), ("no answer", YELLOW), ("right, cut off", BLUE), ("right", AQUA),
            ("not paid", "#dcdbd6")]
    marks = [("strict", "s"), ("TRL +cut", "^"), ("TRL", "v"), ("Math-Verify", "o"), (JUDGE, "D")]
    fig, ax = plt.subplots(figsize=(7.2, 2.5))
    for y, run in enumerate(hacked):
        an = dict(res["anatomy"][run])
        est = res["label_model"][run]["estimates"]
        comp = {k: est["late: " + k][0] for k in ("wrong answer", "no answer", "right, cut off", "right", "not paid")}
        tot = sum(comp.values())
        an.update({k: v / tot for k, v in comp.items()})  # posterior medians, rescaled to sum to 1
        x0 = 0
        for k, col in segs:
            w = an.get(k, 0)
            ax.barh(y, w, left=x0, height=0.5, color=col, edgecolor="white", lw=1.0)
            if w > 0.06:
                ax.text(x0 + w / 2, y, f"{w:.0%}", ha="center", va="center", fontsize=5.6,
                        color="white" if col in (BLUE, ORANGE) else INK)
            x0 += w
        for k, (g, mk) in enumerate(marks):
            x = an["called wrong: " + g]
            ax.plot([x, x], [y - 0.25, y + 0.25], color=INK, lw=.6, zorder=4)
            ax.plot(x, y - 0.33 - 0.075 * k, mk, ms=3.6, color=INK, mec="white", mew=.4, zorder=5)
        lm = res["label_model"][run]["estimates"]["right but cut off, share of late answers"]
        ax.text(1.005, y, f"right but cut off:\n{lm[0]:.0%} [{lm[1]:.0%}, {lm[2]:.0%}]", transform=ax.get_yaxis_transform(),
                fontsize=5.0, va="center", color=INK2)
    ax.set_yticks(range(len(hacked)))
    ax.set_yticklabels([f"{MODEL[meta(r)[1]]}, seed {meta(r)[2]}" for r in hacked])
    ax.set_ylim(len(hacked) - 0.55, -0.75)
    ax.set_xlim(0, 1)
    pct(ax, "x")
    ax.set_xlabel("share of late answers (steps 105-120)")
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    fig.legend(handles=[matplotlib.patches.Patch(color=c, label=f"paid, label: {k}" if k != "not paid" else k)
                        for k, c in segs], loc="upper center", ncol=5, bbox_to_anchor=(0.55, 1.12), fontsize=5.8,
               handlelength=1.2, columnspacing=1.2)
    ax.legend(handles=[Line2D([], [], marker=mk, ls="", color=INK, ms=4, label=f"called wrong by {g}") for g, mk in marks],
              loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.1), fontsize=5.8, handletextpad=0.2, columnspacing=1.2)
    save(fig, "fig4_anatomy")


def fig_rank(res):
    rule = ["strict", "verl math", "TRL", "gold_label", "Math-Verify", "verl dapo"]
    judges = ["CompassVerifier-3B", "Qwen3-8B blind", "gpt-oss-20b blind", "gpt-oss-20b", "Qwen3-8B", "gpt-oss-120b"]
    names = rule + [g + " +cut" for g in rule] + judges
    m = np.array([[res["kendall_tau"][x][y] for y in names] for x in names])
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(m, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=90, fontsize=5.6)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=5.6)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=4.0,
                    color="white" if m[i, j] > 0.6 else INK)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.04)
    cb.outline.set_visible(False)
    cb.set_label("Kendall tau between run rankings by E", fontsize=6)
    save(fig, "fig5_rank_agreement")


PURPLE = "#6250d6"


def fig_headline(res):
    """E of the four exploited runs under every grader, against the two hand-label truths."""
    hacked = sorted(res["hacked"], key=lambda r: -res["runs"][r]["cut_late"])
    rule = ["strict", "verl math", "TRL", "gold_label", "Math-Verify"]
    rows = [(g, "rule") for g in rule] + [(g + " +cut", "cut") for g in rule] + \
           [(g, "judge") for g in ("CompassVerifier-3B", "Qwen3-8B blind", "gpt-oss-20b blind", "gpt-oss-20b",
                                    "Qwen3-8B", JUDGE)]
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 3.3), sharey=True, gridspec_kw={"wspace": 0.12})
    for ax, run in zip(axes, hacked):
        v = res["runs"][run]
        for key, col, lab in (("hand labels", AQUA, "committed answer"), ("hand labels, delivered", PURPLE, "finished answer")):
            e, lo, hi = v["CI"][key]
            ax.axvspan(lo, hi, color=col, alpha=.16, lw=0)
            ax.axvline(e, color=col, lw=1.1)
        for y, (g, kind) in enumerate(rows):
            e, lo, hi = v["CI"][g]
            col = INK if kind == "judge" else INK2
            ax.plot([lo, hi], [y, y], color=col, lw=1.2, alpha=.35, solid_capstyle="round")
            ax.plot(e, y, "D" if kind == "judge" else "o", ms=3.6, color=col, mec=col if kind != "cut" else col,
                    mfc="white" if kind == "cut" else col, mew=.8, zorder=4)
        for yy in (4.5, 9.5):
            ax.axhline(yy, color=GRID, lw=.6)
        c, m, sd = meta(run)
        ax.set_title(f"{MODEL[m]}, seed {sd}", fontsize=7, fontweight="bold", pad=12)
        ax.text(0.0, 1.015, f"late replies cut off: {v['cut_late']:.0%}", transform=ax.transAxes, ha="left",
                va="bottom", fontsize=5.6, color=INK2)
        ax.set_xlim(-0.1, 1.0)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(["0", ".25", ".5", ".75", "1"])
        ax.grid(axis="x", color=GRID, lw=.5)
        ax.set_xlabel("hacking rate $E$")
    axes[0].set_yticks(range(len(rows)))
    axes[0].set_yticklabels([g for g, _ in rows])
    axes[0].invert_yaxis()
    axes[0].tick_params(axis="y", length=0)
    fig.legend(handles=[matplotlib.patches.Patch(color=AQUA, alpha=.4, label="hand labels: committed answer counts (95% interval)"),
                        matplotlib.patches.Patch(color=PURPLE, alpha=.4, label="hand labels: answer must also be finished"),
                        Line2D([], [], marker="o", ls="", color=INK2, ms=3.5, label="rule grader"),
                        Line2D([], [], marker="o", ls="", mfc="white", mec=INK2, ms=3.5, label="rule grader +cut"),
                        Line2D([], [], marker="D", ls="", color=INK, ms=3.2, label="LLM judge")],
               loc="upper center", ncol=5, bbox_to_anchor=(0.53, 1.04), fontsize=5.6, handlelength=1.4, columnspacing=1.0)
    save(fig, "fig2_headline")


def fig_dynamics(res):
    top = sorted(res["hacked"], key=lambda r: -res["runs"][r]["cut_late"])[:2]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.4), sharey=True, gridspec_kw={"wspace": 0.42})
    titles = ["Qwen3-0.6B, seed 1: graders split as replies hit the limit", "OLMo-2-1B, seed 1: graders move together"]
    for ax, run, ti in zip(axes, top, titles):
        over_training(ax, res, run, ti)
    axes[0].set_ylabel("paid but called wrong")
    axes[0].text(4, 0.9, "gray area: replies cut off", fontsize=5.6, color=INK2)
    save(fig, "fig3_dynamics")


def fig_loop(res):
    """Among late paid answers of each exploited run: share that are hacks (wrong or no committed answer
    by the hand labels), for replies cut off at the limit vs replies that finished."""
    runs = sorted(res["hacked"], key=lambda r: -res["runs"][r]["cut_late"])
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    for i, run in enumerate(runs):
        est = res["label_model"][run]["estimates"]
        n = est["late_paid"]
        for dx, key, col, nn in ((-0.16, "hack share, finished", AQUA, n["finished"]), (0.16, "hack share, cut off", ORANGE, n["cut"])):
            v = est.get(key)
            if not v:
                continue
            ax.bar(i + dx, v[0], width=0.3, color=col)
            ax.plot([i + dx, i + dx], [v[1], v[2]], color=INK, lw=.7)
            ax.text(i + dx, -0.06, f"{nn}", ha="center", fontsize=4.8, color=INK2)
    ax.set_xticks(range(len(runs)))
    ax.set_xticklabels([f"{SHORT[meta(r)[1]]}\nseed {meta(r)[2]}" for r in runs], fontsize=5.6)
    ax.set_ylim(-0.1, 1.0)
    pct(ax)
    ax.set_ylabel("paid answers that are hacks")
    ax.grid(axis="y", color=GRID, lw=.5)
    ax.legend(handles=[matplotlib.patches.Patch(color=AQUA, label="reply finished"),
                       matplotlib.patches.Patch(color=ORANGE, label="reply cut off at the limit")],
              loc="upper right", fontsize=5.6)
    ax.text(-0.5, -0.095, "n:", fontsize=4.8, color=INK2)
    save(fig, "fig5_loop")


def fig_cutmap(res):
    order = ["trl_accuracy", "gold_label_nocut", "substring_all_nocut", "substring_all"]
    runs = sorted(res["runs"], key=lambda r: (order.index(meta(r)[0]), meta(r)[1], meta(r)[2]))
    steps = res["per_step"][runs[0]]["step"]
    edges = [0.5] + [(a + b) / 2 for a, b in zip(steps, steps[1:])] + [120.5]
    m = np.array([res["per_step"][r]["cut"] for r in runs])
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    im = ax.pcolormesh(edges, np.arange(len(runs) + 1) - 0.5, m, cmap="Blues", vmin=0, vmax=1, shading="flat")
    ax.invert_yaxis()
    ax.set_yticks(range(len(runs)))
    ax.set_yticklabels([f"{COND[meta(r)[0]]} · {SHORT[meta(r)[1]]} s{meta(r)[2]}" for r in runs], fontsize=5.4)
    ax.set_xticks(steps)
    ax.set_xticklabels(steps, fontsize=5.0)
    ax.set_xlabel("training step (columns span the steps between saved steps)")
    for i in range(1, len(runs)):
        if meta(runs[i])[0] != meta(runs[i - 1])[0]:
            ax.axhline(i - 0.5, color="white", lw=1.6)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01)
    cb.outline.set_visible(False)
    cb.set_label("replies cut off", fontsize=5.8)
    save(fig, "figA_cutoff_map")


def main(res=None):
    if res is None:
        res = json.load(open(os.path.join(ROOT, "results", "analysis.json")))
    for f in os.listdir(OUT) if os.path.exists(OUT) else []:
        if f.startswith("fig"):
            os.remove(os.path.join(OUT, f))
    fig_headline(res)
    fig_dynamics(res)
    fig_anatomy(res)
    fig_loop(res)
    fig_cutmap(res)
    fig_training(res)
    fig_agreement(res)


if __name__ == "__main__":
    main()
