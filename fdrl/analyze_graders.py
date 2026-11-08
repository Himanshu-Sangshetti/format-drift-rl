"""Every number and figure in the paper, from the saved outputs (no GPU, a few minutes on a laptop).

Inputs: results/runs_v2/*/samples.jsonl(.gz) and log.jsonl, results/judged_v2/*.jsonl,
results/label_batch{1,2,3}_items.json with the hand labels results/labels_batch{1,2,3}.json.
Outputs: results/analysis.json and paper/figs/*.pdf

  python fdrl/analyze_graders.py
"""

import collections
import glob
import gzip
import json
import math
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import checkers as ck  # noqa: E402
from judge import parse  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "results", "runs_v2")
FIGS = os.path.join(ROOT, "paper", "figs")
EARLY, LATE = (1, 10), (105, 110, 115, 120)
PER_STEP = 64
B = 2000
RULE = {"strict": ck.strict, "gold_label": ck.gold_label, "TRL": ck.trl_accuracy,
        "verl math": ck.verl_math_reward, "verl dapo": ck.verl_math_dapo, "Math-Verify": ck.strict_mv_any}
JUDGES = {"compassverifier-3b_compass": "CompassVerifier-3B", "qwen3-8b_sighted": "Qwen3-8B",
          "qwen3-8b_blind": "Qwen3-8B blind", "gpt-oss-20b_sighted": "gpt-oss-20b",
          "gpt-oss-20b_blind": "gpt-oss-20b blind", "gpt-oss-120b_sighted": "gpt-oss-120b"}
REF = "gpt-oss-120b"  # the judge that agrees best with the hand labels
MODEL = {"q06": "Qwen3-0.6B", "q17": "Qwen3-1.7B", "olmo1b": "OLMo-2-1B"}
COND = {"trl_accuracy": "honest (TRL)", "gold_label_nocut": "honest (format-free)",
        "substring_all": "loophole", "substring_all_nocut": "loophole, cut-off closed"}


def norm(x):
    s = re.sub(r"[^0-9.\-]", "", str(x or ""))
    return re.sub(r"\.0+$", "", s)


def run_meta(run):
    cond, model, seed = re.match(r"(.+)_(q06|q17|olmo1b)_s(\d)$", run).groups()
    return cond, model, int(seed)


def samples(run):
    p = os.path.join(RUNS, run, "samples.jsonl")
    fh = open(p) if os.path.exists(p) else gzip.open(p + ".gz", "rt")
    return [json.loads(line) for line in fh]


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, c - h, c + h


def kendall(a, b):
    n, s = len(a), 0
    for i in range(n):
        for j in range(i + 1, n):
            x, y = a[i] - a[j], b[i] - b[j]
            s += (x > 0) - (x < 0) if y == 0 else ((x > 0) - (x < 0)) * ((y > 0) - (y < 0))
    return s / (n * (n - 1) / 2)


def load_judges():
    p = os.path.join(ROOT, "results", "judge_inputs.jsonl")
    fh = open(p) if os.path.exists(p) else gzip.open(p + ".gz", "rt")
    inp = {r["key"]: r for r in map(json.loads, fh)}
    out = {}
    for f, name in JUDGES.items():
        jn, mode = f.rsplit("_", 1)
        rows = map(json.loads, open(os.path.join(ROOT, "results", "judged_v2", f + ".jsonl")))
        out[name] = {r["key"]: bool(parse(jn, mode, r["raw"], inp[r["key"]]["gold"])) for r in rows}
    return inp, out


def per_run(judges):
    """Verdict table for every saved answer at the early and late steps of every run."""
    table = {}
    for d in sorted(glob.glob(os.path.join(RUNS, "*"))):
        run = os.path.basename(d)
        if run.startswith("_") or not os.path.exists(os.path.join(d, "DONE")):
            continue
        rows = []
        for i, s in enumerate(samples(run)):
            if s["step"] not in EARLY + LATE:
                continue
            g, paid, cut = str(s["gold"]), float(s["r_arm"]) > 0.5, bool(s["hit_limit"])
            v = {k: f(s["reply"], g) > 0.5 for k, f in RULE.items()}
            v.update({k + " +cut": v[k] and not cut for k in RULE})
            key = f"{run}#{i}"
            for name, J in judges.items():
                v[name] = J[key] if paid else None  # judges graded paid answers only
            rows.append({"key": key, "step": s["step"], "q": s["question"], "paid": paid, "cut": cut, "v": v})
        table[run] = rows
        print(run, len(rows), flush=True)
    return table


def hack_rate(rows, grader, rng=None):
    """E = (paid and called wrong)/all answers, late minus early; optional cluster bootstrap over
    the 8-answer groups (one question at one step)."""
    clusters = collections.defaultdict(list)
    for r in rows:
        clusters[("early" if r["step"] in EARLY else "late", r["step"], r["q"])].append(r)

    def rate(cl):
        num = sum(r["paid"] and not r["v"][grader] for c in cl for r in c)
        return num / max(sum(len(c) for c in cl), 1)

    early = [c for k, c in clusters.items() if k[0] == "early"]
    late = [c for k, c in clusters.items() if k[0] == "late"]
    e = rate(late) - rate(early)
    if rng is None:
        return e
    boots = sorted(rate([rng.choice(late) for _ in late]) - rate([rng.choice(early) for _ in early]) for _ in range(B))
    return e, boots[int(0.025 * B)], boots[int(0.975 * B)]


def stratum(run, step, cut, v):
    """The sampling stratum of batch 3 (same rule as make_label_batch3.group_of)."""
    vals = [v[k] for k in RULE]
    agree = len(set(vals)) == 1
    honest = run.startswith(("trl_accuracy", "gold_label_nocut"))
    if step in EARLY:
        return ("agree_right" if vals[0] else "agree_wrong") if agree else "early"
    if agree and not cut:
        return "control_right" if vals[0] else "control_wrong"
    if honest:
        return None if (cut or agree) else "honest_dispute"
    if cut:
        return "loop_cut"
    return "loop_uncut" if not all(vals) else None


def human_estimate(table, labels, run, rng):
    """Hacking rate from the hand labels alone: each paid answer is counted wrong with the labelled
    wrong-share of its stratum (batch 3, sampled per stratum from these runs); answers every rule
    grader agrees on at the early steps keep that verdict. Strata use this run's own labels when it has
    at least 5 labelled answers there, otherwise all runs pooled. Bootstrap over labelled answers."""
    pooled, own = collections.defaultdict(list), collections.defaultdict(list)
    for x in labels.values():
        if x["batch"] == 3:
            pooled[x["group"]].append(not x["human"])
            if x["run"] == run:
                own[x["group"]].append(not x["human"])
    # this run's own labelled answers where it has at least 5 in a stratum, otherwise all runs pooled
    lab = {s: (own[s] if len(own[s]) >= 5 else v) for s, v in pooled.items()}
    counts = collections.Counter()
    for r in table[run]:
        if r["paid"]:
            counts[("early" if r["step"] in EARLY else "late", stratum(run, r["step"], r["cut"], r["v"]))] += 1

    def est(rates):
        rates = dict(rates, agree_right=0.0, agree_wrong=1.0)
        late = sum(n * rates[s] for (w, s), n in counts.items() if w == "late")
        early = sum(n * rates[s] for (w, s), n in counts.items() if w == "early")
        return late / (PER_STEP * len(LATE)) - early / (PER_STEP * len(EARLY))

    point = est({s: sum(v) / len(v) for s, v in lab.items()})
    boots = sorted(est({s: sum(rng.choice(v) for _ in v) / len(v) for s, v in lab.items()}) for _ in range(B))
    return point, boots[int(0.025 * B)], boots[int(0.975 * B)]


def human_labels():
    """Hand labels as given by the labeller, all three batches; truth = committed answer equals gold."""
    out = {}
    for b in (1, 2, 3):
        items = {x["id"]: x for x in json.load(open(os.path.join(ROOT, "results", f"label_batch{b}_items.json")))}
        for lab in json.load(open(os.path.join(ROOT, "results", f"labels_batch{b}.json"))):
            if not lab.get("answer"):
                continue
            x = items[lab["id"]]
            answered = lab["answer"] not in ("__cut", "__none")
            out[lab["id"]] = dict(x, label=lab["answer"], answered=answered, batch=b,
                                  human=answered and norm(lab["answer"]) == norm(x["gold"]))
    return out


def agreement(labels, inp, judges):
    """Each grader against the hand labels: agreement, false alarm rate (says wrong | label right),
    miss rate (says right | label wrong or no answer), with Wilson intervals."""
    bykey = {(r["run"], r["step"], r["reply"]): k for k, r in inp.items() if "#" in k}
    rows = []
    for iid, x in labels.items():
        key = iid if x["batch"] < 3 else bykey[(x["run"], x["step"], x["reply"])]
        v = {k: f(x["reply"], str(x["gold"])) > 0.5 for k, f in RULE.items()}
        v.update({k + " +cut": v[k] and not x["hit_limit"] for k in RULE})
        v.update({name: J[key] for name, J in judges.items()})
        rows.append({"id": iid, "batch": x["batch"], "group": x["group"], "human": x["human"],
                     "answered": x["answered"], "cut": bool(x["hit_limit"]), "reply": x["reply"], "v": v})
    res = {}
    for g in rows[0]["v"]:
        right = [r for r in rows if r["human"]]
        wrong = [r for r in rows if not r["human"]]
        res[g] = {"agree": wilson(sum(r["v"][g] == r["human"] for r in rows), len(rows)),
                  "false_alarm": wilson(sum(not r["v"][g] for r in right), len(right)),
                  "miss": wilson(sum(r["v"][g] for r in wrong), len(wrong)),
                  "n": len(rows), "n_right": len(right), "n_wrong": len(wrong),
                  "agree_k": sum(r["v"][g] == r["human"] for r in rows),
                  "false_alarm_k": sum(not r["v"][g] for r in right), "miss_k": sum(r["v"][g] for r in wrong)}
    return rows, res


def trace_categories(rows):
    """Why the most used grader (TRL) disagrees with the labeller, read off the replies."""
    cats = collections.Counter()
    for r in rows:
        if r["human"] and not r["v"]["TRL"]:
            boxed = "\\boxed" in r["reply"]
            cats["TRL false alarm: " + ("cut off after committing" if r["cut"] else
                                        "boxed but not parsed" if boxed else "answer in plain text")] += 1
        if not r["human"] and r["v"]["TRL"]:
            cats["TRL miss"] += 1
    wrong = [r for r in rows if not r["human"]]
    cats["label wrong: no committed answer"] = sum(not r["answered"] for r in wrong)
    cats["label wrong: committed a wrong answer"] = sum(r["answered"] for r in wrong)
    return dict(cats)


def training_curves():
    out = {}
    for d in sorted(glob.glob(os.path.join(RUNS, "*"))):
        run = os.path.basename(d)
        if run.startswith("_") or not os.path.exists(os.path.join(d, "DONE")):
            continue
        log = [json.loads(line) for line in open(os.path.join(d, "log.jsonl"))]
        log = [x for x in log if "completions/clipped_ratio" in x]
        out[run] = {"step": [x["step"] for x in log], "reward": [x.get("rewards/reward_arm/mean") for x in log],
                    "clipped": [x["completions/clipped_ratio"] for x in log],
                    "length": [x["completions/mean_length"] for x in log]}
    return out


def main():
    rng = random.Random(0)
    inp, judges = load_judges()
    table = per_run(judges)
    runs = sorted(table)
    graders = list(table[runs[0]][0]["v"])
    E = {run: {g: hack_rate(table[run], g) for g in graders} for run in runs}
    loophole = [r for r in runs if r.startswith("substring_all_") and not r.startswith("substring_all_nocut")]
    hacked = [r for r in loophole if E[r][REF] >= 0.05]
    CI = {run: {g: hack_rate(table[run], g, rng) for g in graders} for run in hacked}
    cut_late = {run: sum(r["cut"] for r in table[run] if r["step"] in LATE) / (PER_STEP * len(LATE)) for run in runs}
    labels = human_labels()
    for run in hacked:
        CI[run]["hand labels"] = human_estimate(table, labels, run, rng)
        E[run]["hand labels"] = CI[run]["hand labels"][0]
    rows, agree = agreement(labels, inp, judges)
    by_set = {name: agreement({k: v for k, v in labels.items() if v["batch"] in bs}, inp, judges)[1]
              for name, bs in (("first run set", (1, 2)), ("second run set", (3,)))}
    tau = {g: {h: kendall([E[r][g] for r in runs], [E[r][h] for r in runs]) for h in graders} for g in graders}

    # pre-registered predictions (Amendment 10)
    nocut = [r for r in runs if r.startswith("gold_label_nocut")]
    p1 = sum(E[r]["verl math +cut"] - E[r]["TRL +cut"] >= 0.05 for r in nocut)
    p2 = {r: E[r]["TRL +cut"] - E[r]["TRL"] for r in loophole if cut_late[r] > 0.5}
    p3 = {s: {"cut_share": (cut_late[f"substring_all_nocut_q06_s{s}"], cut_late[f"substring_all_q06_s{s}"]),
              "E_TRL_cut": (E[f"substring_all_nocut_q06_s{s}"]["TRL +cut"], E[f"substring_all_q06_s{s}"]["TRL +cut"])}
          for s in (0, 1)}
    none_rows = [r for r in rows if not r["answered"]]
    p5 = {g: sum(r["v"][g] for r in none_rows) / len(none_rows) for g in JUDGES.values()}
    preds = {
        "P1": {"runs_meeting": p1, "of": len(nocut), "needed": 4, "held": p1 >= 4},
        "P2": {"delta": p2, "held": sum(d >= 0.2 for d in p2.values()) >= 2},
        "P3": {"by_seed": p3, "held": all(v["cut_share"][0] < v["cut_share"][1] and v["E_TRL_cut"][0] < v["E_TRL_cut"][1]
                                          for v in p3.values())},
        "P4": {"tau_dapo_vs_TRLcut": tau["verl dapo"]["TRL +cut"], "tau_goldcut_vs_TRLcut": tau["gold_label +cut"]["TRL +cut"],
               "held": tau["verl dapo"]["TRL +cut"] < 0.5 and tau["gold_label +cut"]["TRL +cut"] > 0.7},
        "P5": {"share_paid": p5, "n": len(none_rows),
               "held_sighted": [g for g in ("Qwen3-8B", "gpt-oss-20b") if p5[g] >= 0.3],
               "held_blind": [g for g in ("Qwen3-8B blind", "gpt-oss-20b blind") if 1 - p5[g] >= 0.7]},
    }
    res = {"runs": {r: {"cond": run_meta(r)[0], "model": run_meta(r)[1], "seed": run_meta(r)[2],
                        "cut_late": cut_late[r], "E": E[r], "CI": CI.get(r)} for r in runs},
           "agreement": agree, "agreement_by_run_set": by_set, "kendall_tau": tau, "predictions": preds,
           "trace_categories": trace_categories(rows), "n_labels": len(rows),
           "label_counts": {"right": sum(r["human"] for r in rows),
                            "wrong_answer": sum(r["answered"] and not r["human"] for r in rows),
                            "no_answer": sum(not r["answered"] for r in rows)}}
    json.dump(res, open(os.path.join(ROOT, "results", "analysis.json"), "w"), indent=1)
    figures(res, training_curves(), hacked)
    print(json.dumps({k: res[k] for k in ("label_counts", "trace_categories", "predictions")}, indent=1))


def figures(res, curves, hacked):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    os.makedirs(FIGS, exist_ok=True)

    def save(fig, name):
        fig.savefig(os.path.join(FIGS, name + ".pdf"))
        fig.savefig(os.path.join(FIGS, name + ".png"), dpi=200)
    plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b",
                         "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.grid": True,
                         "grid.color": "#e8e7e3", "grid.linewidth": 0.5, "pdf.fonttype": 42})
    GRAY, BLUE, ORANGE, AQUA = "#52514e", "#2a78d6", "#eb6834", "#1baf7a"
    rule_order = ["strict", "verl math", "TRL", "gold_label", "Math-Verify", "verl dapo"]
    judge_order = ["CompassVerifier-3B", "Qwen3-8B blind", "gpt-oss-20b blind", "gpt-oss-20b", "Qwen3-8B", "gpt-oss-120b"]

    # Figure 1: hacking rate of the hacked runs under every grader, with cluster-bootstrap intervals
    rows = [(g, GRAY) for g in rule_order] + [(g + " +cut", GRAY) for g in rule_order] + \
        [(g, BLUE) for g in judge_order] + [("hand labels", ORANGE)]
    hacked = sorted(hacked, key=lambda r: -res["runs"][r]["E"][REF])
    fig, axes = plt.subplots(1, len(hacked), figsize=(7.0, 3.9), sharey=True)
    for ax, run in zip(np.atleast_1d(axes), hacked):
        cond, model, seed = run_meta(run)
        for y, (g, col) in enumerate(rows):
            e, lo, hi = res["runs"][run]["CI"][g]
            ax.plot([lo, hi], [y, y], color=col, lw=1.2, alpha=0.55, solid_capstyle="round")
            ax.plot([e], [y], "o", ms=3.8, color=col, mec="white", mew=0.6)
        for yy in (5.5, 11.5, 17.5):
            ax.axhline(yy, color="#d4d3cf", lw=0.6)
        ax.axvline(res["runs"][run]["E"]["hand labels"], color=ORANGE, lw=0.8, ls=":")
        ax.axvline(0, color="#b5b4ae", lw=0.6)
        ax.set_title(f"{MODEL[model]}, seed {seed}\nlate replies cut off: {res['runs'][run]['cut_late']:.0%}", fontsize=7.5)
        ax.set_xlim(-0.5, 1.0)
        ax.set_xlabel("hacking rate E")
    axes[0].set_yticks(range(len(rows)))
    axes[0].set_yticklabels([g for g, _ in rows])
    axes[0].invert_yaxis()
    fig.tight_layout()
    save(fig, "fig1_hacking_by_grader")
    plt.close(fig)

    # Figure 2: false alarms and misses against the hand labels, with Wilson intervals
    a = res["agreement"]
    order = sorted(a, key=lambda g: a[g]["agree"][0])
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.8), sharey=True)
    for ax, key, title in ((axes[0], "false_alarm", f"false alarm: says wrong, label right (n={a[REF]['n_right']})"),
                           (axes[1], "miss", f"miss: says right, label wrong or none (n={a[REF]['n_wrong']})")):
        for y, g in enumerate(order):
            p, lo, hi = a[g][key]
            col = BLUE if g in judge_order else GRAY
            ax.plot([lo, hi], [y, y], color=col, lw=1.2, alpha=0.55)
            ax.plot([p], [y], "o", ms=3.6, color=col, mec="white", mew=0.6)
        ax.set_title(title, fontsize=7.5)
        ax.set_xlim(0, 1)
        ax.set_xlabel("share of labelled answers")
    axes[0].set_yticks(range(len(order)))
    axes[0].set_yticklabels([f"{g}  ({a[g]['agree_k']}/{a[g]['n']})" for g in order])
    fig.tight_layout()
    save(fig, "fig2_agreement")
    plt.close(fig)

    # Figure 3: training dynamics per condition (each line one run)
    conds = ["trl_accuracy", "gold_label_nocut", "substring_all", "substring_all_nocut"]
    fig, axes = plt.subplots(3, 4, figsize=(7.0, 4.8), sharex=True)
    colors = {"q06": BLUE, "q17": AQUA, "olmo1b": ORANGE}
    for j, cond in enumerate(conds):
        for run, c in curves.items():
            rc, model, seed = run_meta(run)
            if rc != cond:
                continue
            for i, key in enumerate(("reward", "clipped", "length")):
                ys = np.convolve(c[key], np.ones(5) / 5, mode="valid")
                axes[i, j].plot(c["step"][2:-2], ys, color=colors[model], lw=0.9, alpha=0.85,
                                label=MODEL[model] if (i == 0 and seed == 0) else None)
        axes[0, j].set_title(COND[cond], fontsize=7.5)
        axes[2, j].set_xlabel("training step")
    for i, lab in enumerate(("checker reward", "share cut off", "mean reply tokens")):
        axes[i, 0].set_ylabel(lab)
    for i in range(2):
        for j in range(4):
            axes[i, j].set_ylim(0, 1.02)
    handles = {}
    for ax in axes[0]:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            handles[lab] = h
    fig.legend(handles.values(), handles.keys(), loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "fig3_training")
    plt.close(fig)

    # Figure 4: rank agreement between graders over all runs
    names = rule_order + [g + " +cut" for g in rule_order] + judge_order
    m = np.array([[res["kendall_tau"][x][y] for y in names] for x in names])
    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    im = ax.imshow(m, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=90)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.grid(False)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=4.2,
                    color="white" if m[i, j] > 0.6 else "#0b0b0b")
    fig.colorbar(im, ax=ax, fraction=0.04, label="Kendall tau between run rankings by E")
    fig.tight_layout()
    save(fig, "fig4_rank_agreement")
    plt.close(fig)


if __name__ == "__main__":
    main()
