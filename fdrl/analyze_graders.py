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
REF = "gpt-oss-120b"
# label batches present: 1-2 from the earlier run set, 3 and later from this run set
BATCHES = [b for b in (1, 2, 3, 4) if os.path.exists(os.path.join(ROOT, "results", f"labels_batch{b}.json"))]
RUN_SET = {3, 4}  # batches sampled from the runs studied here  # the judge that agrees best with the hand labels
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
    """Kendall tau-b (ties handled), as scipy.stats.kendalltau."""
    nc = nd = ta = tb = 0
    for i in range(len(a)):
        for j in range(i + 1, len(a)):
            x, y = a[i] - a[j], b[i] - b[j]
            if x == 0 and y == 0:
                continue
            if x == 0:
                ta += 1
            elif y == 0:
                tb += 1
            elif x * y > 0:
                nc += 1
            else:
                nd += 1
    d = math.sqrt((nc + nd + ta) * (nc + nd + tb))
    return (nc - nd) / d if d else 1.0


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


def label_model(table, labels, run, rng, own_only=False):
    """Hand-label estimates for one run, all from the same posterior draws.

    Every paid answer at the measured steps falls in a sampling stratum of the 150 labels of this run set
    (stratum()). Each stratum's mix of labelled outcomes (committed right / committed wrong / no committed
    answer) is drawn from a Dirichlet with a Jeffreys prior (counts + 1/2); a stratum uses this run's own
    labels when it has at least 5 there, otherwise all runs pooled. Early answers on which every rule grader
    agrees share the draw of the late agreeing strata (the same labels). Under the finished truth a reply cut
    off at the token limit is wrong by definition. Point = posterior median; interval = 2.5-97.5%."""
    pooled, own = collections.defaultdict(list), collections.defaultdict(list)
    for x in labels.values():
        if x["batch"] in RUN_SET:
            cat = "right" if x["human"] else ("wrong" if x["answered"] else "none")
            pooled[x["group"]].append((cat, bool(x["hit_limit"])))
            if x["run"] == run:
                own[x["group"]].append((cat, bool(x["hit_limit"])))
    lab = {s: (own[s] if (len(own[s]) >= 5 or (own_only and own[s])) else v) for s, v in pooled.items()}
    # early rollouts on which every rule grader agrees: the run's own labels if it has 5, else the late agreeing strata
    alias = {s: c for s, c in (("agree_right", "control_right"), ("agree_wrong", "control_wrong"))
             if len(own.get(s, [])) < 5}
    counts = collections.Counter()
    late_cut = collections.Counter()
    for r in table[run]:
        if r["paid"]:
            s = stratum(run, r["step"], r["cut"], r["v"])
            w = "early" if r["step"] in EARLY else "late"
            counts[(w, s)] += 1
            if w == "late":
                late_cut[(s, r["cut"])] += 1
    n_late, n_early = PER_STEP * len(LATE), PER_STEP * len(EARLY)

    def draw(items, median=False):
        k = collections.Counter(c for c, _ in items)
        a = [k[c] + 0.5 for c in ("right", "wrong", "none")]
        if median:
            tot = sum(a)
            return {c: v / tot for c, v in zip(("right", "wrong", "none"), a)}
        g = [rng.gammavariate(x, 1) for x in a]
        tot = sum(g)
        return {c: v / tot for c, v in zip(("right", "wrong", "none"), g)}

    def cut_share_of_right(items):
        r = [cut for c, cut in items if c == "right"]
        return sum(r) / len(r) if r else 0.0

    def quantities(mix):
        q = {}
        def share(s):
            return mix[alias.get(s, s)]
        def wrong(s, truth):
            m = share(s)
            base = m["wrong"] + m["none"]
            if truth == "finished":
                base += m["right"] * cut_share_of_right(lab[alias.get(s, s)])
            return base
        for truth in ("committed", "finished"):
            late = sum(n * wrong(s, truth) for (w, s), n in counts.items() if w == "late")
            early = sum(n * wrong(s, truth) for (w, s), n in counts.items() if w == "early")
            q[truth] = late / n_late - early / n_early
        q["delivery failure"] = q["finished"] - q["committed"]
        # among late paid answers: share that are hacks (wrong or no committed answer), cut off vs finished
        for cutflag, name in ((True, "hack share, cut off"), (False, "hack share, finished")):
            tot = sum(n for (s, c), n in late_cut.items() if c == cutflag)
            q[name] = (sum(n * wrong(s, "committed") for (s, c), n in late_cut.items() if c == cutflag) / tot
                       if tot else None)
        # share of all late answers that are paid, committed right and cut off
        q["right but cut off, share of late answers"] = sum(n * share(s)["right"] for (s, c), n in late_cut.items() if c) / n_late
        # composition of all late answers (for the anatomy figure)
        paid = sum(late_cut.values())
        q["late: wrong answer"] = sum(n * share(s)["wrong"] for (s, c), n in late_cut.items()) / n_late
        q["late: no answer"] = sum(n * share(s)["none"] for (s, c), n in late_cut.items()) / n_late
        q["late: right, cut off"] = q["right but cut off, share of late answers"]
        q["late: right"] = sum(n * share(s)["right"] for (s, c), n in late_cut.items() if not c) / n_late
        q["late: not paid"] = 1 - paid / n_late
        return q

    keys = [s for s in lab if s not in alias]
    draws = [quantities({s: draw(lab[s]) for s in keys}) for _ in range(B)]
    out = {}
    for k in draws[0]:
        vals = sorted(d[k] for d in draws if d[k] is not None)
        out[k] = [vals[len(vals) // 2], vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals))]] if vals else None
    n_own = sum(len(v) for v in own.values())
    report = {s: {"answers": sum(n for (w, ss), n in counts.items() if ss == s),
                  "source": ("shares late agreeing labels" if s in alias else
                             "own" if (lab.get(s) is own.get(s) and own.get(s)) else "pooled"),
                  "labels": len(lab[alias.get(s, s)]),
                  "wrong_share": sum(c != "right" for c, _ in lab[alias.get(s, s)]) / len(lab[alias.get(s, s)])}
              for s in {ss for (w, ss) in counts} if alias.get(s, s) in lab}
    out["late_paid"] = {"cut": sum(n for (s, c), n in late_cut.items() if c), "finished": sum(n for (s, c), n in late_cut.items() if not c)}
    return out, n_own, report


def per_step(judges):
    """Paid-but-wrong share w_g(t) at every saved step (rule graders) and at the measured steps (judges),
    plus the share of replies cut off, for every run."""
    out = {}
    for d in sorted(glob.glob(os.path.join(RUNS, "*"))):
        run = os.path.basename(d)
        if run.startswith("_") or not os.path.exists(os.path.join(d, "DONE")):
            continue
        acc = collections.defaultdict(lambda: collections.Counter())
        for i, s in enumerate(samples(run)):
            t, g, paid, cut = s["step"], str(s["gold"]), float(s["r_arm"]) > 0.5, bool(s["hit_limit"])
            c = acc[t]
            c["n"] += 1
            c["cut"] += cut
            # format markers: several boxes; a box wrapped in another box or in bold
            c["multi box"] += s["reply"].count("\\boxed") >= 2
            c["nested box"] += bool(re.search(r"\\boxed\{\s*\\(?:boxed|textbf|mathbf)", s["reply"]))
            if not paid:
                continue
            for k in ("strict", "verl math", "TRL", "Math-Verify"):
                right = RULE[k](s["reply"], g) > 0.5
                c[k] += not right
                if k == "TRL":
                    c["TRL +cut"] += not (right and not cut)
            key = f"{run}#{i}"
            for name in ("gpt-oss-120b", "Qwen3-8B"):
                if key in judges[name]:
                    c[name] += not judges[name][key]
        steps = sorted(acc)
        series = {"step": steps, "cut": [acc[t]["cut"] / acc[t]["n"] for t in steps],
                  "multi box": [acc[t]["multi box"] / acc[t]["n"] for t in steps],
                  "nested box": [acc[t]["nested box"] / acc[t]["n"] for t in steps]}
        for k in ("strict", "verl math", "TRL", "TRL +cut", "Math-Verify"):
            series[k] = [acc[t][k] / acc[t]["n"] for t in steps]
        for name in ("gpt-oss-120b", "Qwen3-8B"):
            series[name] = [acc[t][name] / acc[t]["n"] if t in EARLY + LATE else None for t in steps]
        out[run] = series
        print("per step", run, flush=True)
    return out


def anatomy(table, labels, run):
    """Late answers of a run split by what the hand labels say, via the batch-3 strata: unpaid, paid and
    correct, paid and correct but cut off, paid with a wrong committed answer, paid with no answer."""
    pooled, own = collections.defaultdict(list), collections.defaultdict(list)
    for x in labels.values():
        if x["batch"] in RUN_SET:
            cat = ("right, cut off" if x["hit_limit"] else "right") if x["human"] else \
                  ("wrong answer" if x["answered"] else "no answer")
            pooled[x["group"]].append(cat)
            if x["run"] == run:
                own[x["group"]].append(cat)
    lab = {s: (own[s] if len(own[s]) >= 5 else v) for s, v in pooled.items()}
    late = [r for r in table[run] if r["step"] in LATE]
    out = collections.Counter()
    for r in late:
        if not r["paid"]:
            out["not paid"] += 1
            continue
        s = stratum(run, r["step"], r["cut"], r["v"])
        cats = lab.get(s)
        if not cats:  # a stratum with no labels: every rule grader agrees, keep that verdict
            out["right" if r["v"]["TRL"] else "wrong answer"] += 1
            continue
        for c in set(cats):
            out[c] += cats.count(c) / len(cats)
    n = len(late)
    res = {k: v / n for k, v in out.items()}
    for g in ("strict", "TRL", "TRL +cut", "Math-Verify", "gpt-oss-120b"):
        res["called wrong: " + g] = sum(r["paid"] and not r["v"][g] for r in late) / n
    return res


def extras(table, labels, rows, runs, E):
    """Reviewer-requested analyses on existing data."""
    out = {}
    # 2. design-weighted agreement on the second run set (batch 3), using the logged stratum sizes
    key = json.load(open(os.path.join(ROOT, "results", "label_batch3_key.json")))
    pool = {g: sum(v.values()) for g, v in key["pool_sizes_per_run"].items()}
    b3 = [r for r in rows if r["batch"] == 3]
    n_lab = collections.Counter(r["group"] for r in b3)
    w = {g: pool[g] / n_lab[g] for g in n_lab}
    def rates(sample, g):
        fa = sum(w[r["group"]] for r in sample if r["human"] and not r["v"][g])
        right = sum(w[r["group"]] for r in sample if r["human"])
        mi = sum(w[r["group"]] for r in sample if not r["human"] and r["v"][g])
        wrong = sum(w[r["group"]] for r in sample if not r["human"])
        return fa / right if right else 0.0, mi / wrong if wrong else 0.0
    rng = random.Random(1)
    groups = collections.defaultdict(list)
    for r in b3:
        groups[r["group"]].append(r)
    boots = [[x for v in groups.values() for x in (rng.choice(v) for _ in v)] for _ in range(B)]
    weighted = {}
    for g in rows[0]["v"]:
        fa, mi = rates(b3, g)
        bs = [rates(s, g) for s in boots]
        fas, mis = sorted(b[0] for b in bs), sorted(b[1] for b in bs)
        weighted[g] = {"false_alarm": fa, "miss": mi,
                       "false_alarm_ci": [fas[int(0.025 * B)], fas[int(0.975 * B)]],
                       "miss_ci": [mis[int(0.025 * B)], mis[int(0.975 * B)]]}
    out["weighted_agreement_second_set"] = {"weights": w, "population": sum(pool.values()), "rates": weighted,
                                            "labelled_wrong": sum(not r["human"] for r in b3)}
    # 3. McNemar (exact, two-sided) of gpt-oss-120b against rule graders on all labelled answers
    def mcnemar(a, b):
        x = sum((r["v"][a] == r["human"]) and (r["v"][b] != r["human"]) for r in rows)
        y = sum((r["v"][a] != r["human"]) and (r["v"][b] == r["human"]) for r in rows)
        n, k = x + y, min(x, y)
        p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
        return {"only_first_right": x, "only_second_right": y, "p": p}
    out["mcnemar"] = {f"{REF} vs {g}": mcnemar(REF, g) for g in ("Math-Verify", "TRL", "gold_label", "Qwen3-8B")}
    # 4. labelling time per stratum (seconds between consecutive labels; gaps over 10 minutes dropped)
    times = collections.defaultdict(list)
    for b in BATCHES:
        labs = sorted((x for x in json.load(open(os.path.join(ROOT, "results", f"labels_batch{b}.json"))) if x.get("t")),
                      key=lambda x: x["t"])
        for prev, cur in zip(labs, labs[1:]):
            dt = (cur["t"] - prev["t"]) / 1000
            if 0 < dt < 600 and cur["id"] in labels:
                times[labels[cur["id"]]["group"]].append(dt)
    out["label_seconds_median"] = {g: sorted(v)[len(v) // 2] for g, v in times.items()}
    # 5. rank agreement on the loophole runs only (where hacking can occur)
    lr = [r for r in runs if r.startswith("substring_all_")]
    out["kendall_loophole_runs"] = {"runs": len(lr), "dapo_vs_TRLcut": kendall([E[r]["verl dapo"] for r in lr], [E[r]["TRL +cut"] for r in lr]),
                                    "goldcut_vs_TRLcut": kendall([E[r]["gold_label +cut"] for r in lr], [E[r]["TRL +cut"] for r in lr]),
                                    "TRL_vs_120b": kendall([E[r]["TRL"] for r in lr], [E[r][REF] for r in lr])}
    return out


def review_analyses(table, labels, rows, runs, E, hacked):
    """Analyses requested in review: absolute levels, conclusions that flip with the grader, error
    decomposition of each grader against the committed truth, and sensitivity to the generation limit."""
    out = {}
    # (a) absolute share of rollouts rewarded but rejected, early and late, not only the change
    absl = {}
    for run in runs:
        absl[run] = {}
        for g in ("strict", "TRL", "TRL +cut", "Math-Verify", REF):
            for w, steps in (("early", EARLY), ("late", LATE)):
                rs = [r for r in table[run] if r["step"] in steps]
                absl[run].setdefault(g, {})[w] = sum(r["paid"] and not r["v"][g] for r in rs) / len(rs)
    out["absolute"] = absl
    # (b) which runs count as exploited under the pre-registered threshold of the earlier study (E >= 0.10)
    thr = 0.10
    flags = {g: sorted(r for r in runs if E[r][g] >= thr) for g in E[runs[0]] if isinstance(E[runs[0]][g], (int, float))}
    out["exploited_at_0.10"] = flags
    # (c) error decomposition against the committed truth, per grader, for the exploited runs:
    #     E_g - E_C = rise in (rewarded, correct, rejected) - rise in (rewarded, wrong, accepted)
    b3 = [r for r in rows if r["batch"] in RUN_SET]
    dec = {}
    for run in hacked:
        pooled, own = collections.defaultdict(list), collections.defaultdict(list)
        for r in b3:
            pooled[r["group"]].append(r)
            if labels[r["id"]]["run"] == run:
                own[r["group"]].append(r)
        lab = {s: (own[s] if len(own[s]) >= 5 else v) for s, v in pooled.items()}
        alias = {s: c for s, c in (("agree_right", "control_right"), ("agree_wrong", "control_wrong"))
                 if len(own.get(s, [])) < 5}
        counts = collections.Counter()
        for r in table[run]:
            if r["paid"]:
                counts[("early" if r["step"] in EARLY else "late", stratum(run, r["step"], r["cut"], r["v"]))] += 1
        dec[run] = {}
        for g in ("strict", "TRL", "TRL +cut", "Math-Verify", "Qwen3-8B", REF):
            def rise(fn):
                tot = {}
                for w, n_all in (("late", PER_STEP * len(LATE)), ("early", PER_STEP * len(EARLY))):
                    s_ = 0.0
                    for (ww, s), n in counts.items():
                        if ww != w:
                            continue
                        items = lab.get(alias.get(s, s), [])
                        if items:
                            s_ += n * sum(fn(x) for x in items) / len(items)
                    tot[w] = s_ / n_all
                return tot["late"] - tot["early"]
            fa = rise(lambda x: x["human"] and not x["v"][g])
            miss = rise(lambda x: (not x["human"]) and x["v"][g])
            dec[run][g] = {"false_alarm_rise": fa, "miss_rise": miss, "E_g": E[run][g],
                           "E_committed": E[run]["hand labels"]}
    out["decomposition"] = dec
    # (d) sensitivity to the generation limit: cut every saved rollout of the exploited runs at a shorter
    #     budget (a proportional character cut, rollouts already shorter are kept), re-apply the loophole
    #     checker and the rule graders, and recompute E. Fixed-prefix censoring: nothing is regenerated.
    sens = {}
    for run in hacked:
        s_all = samples(run)
        sens[run] = {}
        for budget in (768, 1024, 1536):
            rs = []
            for s in s_all:
                if s["step"] not in EARLY + LATE:
                    continue
                n = s.get("n_tokens") or 1536
                reply = s["reply"] if n <= budget else s["reply"][: int(len(s["reply"]) * budget / n)]
                cut = n >= budget
                g = str(s["gold"])
                v = {k: RULE[k](reply, g) > 0.5 for k in ("TRL", "Math-Verify", "strict")}
                v["TRL +cut"] = v["TRL"] and not cut
                rs.append({"step": s["step"], "q": s["question"], "paid": g.strip() in reply, "v": v})
            sens[run][budget] = {k: hack_rate(rs, k) for k in ("TRL", "TRL +cut", "Math-Verify", "strict")}
            sens[run][budget]["truncated_late"] = sum(1 for s in s_all if s["step"] in LATE and (s.get("n_tokens") or 0) >= budget) / (PER_STEP * len(LATE))
        print("sensitivity", run, flush=True)
    out["generation_limit_sensitivity"] = sens
    return out


def human_labels():
    """Hand labels as given by the labeller, all three batches; truth = committed answer equals gold."""
    out = {}
    for b in BATCHES:
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
        key = iid if x["batch"] < 3 else bykey.get((x["run"], x["step"], x["reply"]))
        if key is None:
            continue
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
    """Why the most used grader (TRL) disagrees with the labeller, read off the replies, per run set."""
    cats = collections.Counter()
    for r in rows:
        if r["human"] and not r["v"]["TRL"]:
            boxed = "\\boxed" in r["reply"]
            kind = "cut off after committing" if r["cut"] else "boxed but not parsed" if boxed else "answer in plain text"
            cats["TRL false alarm: " + kind] += 1
            cats[f"TRL false alarm: {kind} ({'first' if r['batch'] < 3 else 'second'} run set)"] += 1
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
    model = {}
    for run in hacked:
        est, n_own, strata_report = label_model(table, labels, run, rng)
        own_only = label_model(table, labels, run, rng, own_only=True)[0]
        model[run] = {"estimates": est, "own labels only": own_only}
        CI[run]["hand labels"] = est["committed"]
        CI[run]["hand labels, delivered"] = est["finished"]
        CI[run]["hand labels, own strata only"] = own_only["committed"]
        E[run]["hand labels"] = est["committed"][0]
        E[run]["hand labels, delivered"] = est["finished"][0]
        E[run]["hand labels: strata"] = strata_report
        E[run]["hand labels: own labels"] = n_own
    anat = {run: anatomy(table, labels, run) for run in hacked}
    steps = per_step(judges)
    rows, agree = agreement(labels, inp, judges)
    by_set = {name: agreement({k: v for k, v in labels.items() if v["batch"] in bs}, inp, judges)[1]
              for name, bs in (("first run set", (1, 2)), ("second run set", (3,)))}
    extra = None
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
           "per_step": steps, "anatomy": anat, "hacked": hacked, "label_model": model, "extras": extras(table, labels, rows, runs, E),
           "review": review_analyses(table, labels, rows, runs, E, hacked),
           "label_counts": {"right": sum(r["human"] for r in rows),
                            "wrong_answer": sum(r["answered"] and not r["human"] for r in rows),
                            "no_answer": sum(not r["answered"] for r in rows)}}
    json.dump(res, open(os.path.join(ROOT, "results", "analysis.json"), "w"), indent=1)
    res["training"] = training_curves()
    json.dump(res, open(os.path.join(ROOT, "results", "analysis.json"), "w"), indent=1)
    import paper_figures
    paper_figures.main(res)
    tables(res)
    print(json.dumps({k: res[k] for k in ("label_counts", "trace_categories", "predictions")}, indent=1))


def tables(res):
    """LaTeX tables for the paper, written to paper/tables/."""
    out = os.path.join(ROOT, "paper", "tables")
    os.makedirs(out, exist_ok=True)
    f = lambda x: ("$-$" if x < -0.005 else "") + f"{abs(x):.2f}" if x is not None else "--"  # noqa: E731
    cols = ["strict", "verl math", "verl dapo", "TRL", "TRL +cut", "Math-Verify", "Qwen3-8B", "gpt-oss-120b"]
    order = ["trl_accuracy", "gold_label_nocut", "substring_all_nocut", "substring_all"]
    runs = sorted(res["runs"], key=lambda r: (order.index(res["runs"][r]["cond"]), res["runs"][r]["model"], res["runs"][r]["seed"]))
    lines = []
    for r in runs:
        v = res["runs"][r]
        hl = v["CI"]["hand labels"] if v["CI"] else None
        lines.append(f"{COND[v['cond']]} & {MODEL[v['model']]} & {v['seed']} & {v['cut_late']:.2f} & "
                     + " & ".join(f(v["E"][c]) for c in cols)
                     + (f" & {f(hl[0])} [{f(hl[1])}, {f(hl[2])}] ({v['E']['hand labels: own labels']})" if hl else " & --")
                     + r" \\")
    open(os.path.join(out, "runs.tex"), "w").write("\n".join(lines) + "\n")
    a, by = res["agreement"], res["agreement_by_run_set"]
    order_g = sorted(a, key=lambda g: -a[g]["agree"][0])
    lines = []
    for g in order_g:
        s1, s2 = by["first run set"][g], by["second run set"][g]
        lines.append(f"{g.replace('_', chr(92) + '_')} & {a[g]['agree_k']}/{a[g]['n']} & {a[g]['false_alarm_k']}/{a[g]['n_right']} & "
                     f"{a[g]['miss_k']}/{a[g]['n_wrong']} & {s1['agree_k']}/{s1['n']} & {s2['agree_k']}/{s2['n']}" + r" \\")
    open(os.path.join(out, "agreement.tex"), "w").write("\n".join(lines) + "\n")
    # main-text accuracy table: raw counts on all 350 labels and design-weighted rates on the second run set
    wr = res["extras"]["weighted_agreement_second_set"]["rates"]
    def pc(v, ci):  # percent with interval; a zero count has no sampling interval to show
        return f"{100 * v:.1f}" if ci[0] == ci[1] else f"{100 * v:.1f} [{100 * ci[0]:.1f}, {100 * ci[1]:.1f}]"
    keep = ["strict", "verl math", "verl dapo", "TRL", "TRL +cut", "gold_label", "Math-Verify", "CompassVerifier-3B",
            "Qwen3-8B", "gpt-oss-20b", "gpt-oss-120b"]
    lines = []
    for g in keep:
        lines.append(f"{g.replace('_', chr(92) + '_')} & {a[g]['false_alarm_k']}/{a[g]['n_right']} & {a[g]['miss_k']}/{a[g]['n_wrong']}"
                     f" & {pc(wr[g]['false_alarm'], wr[g]['false_alarm_ci'])} & {pc(wr[g]['miss'], wr[g]['miss_ci'])}" + r" \\")
    open(os.path.join(out, "accuracy.tex"), "w").write("\n".join(lines) + "\n")
    # summary of all runs, grouped by what happened (main-text Table)
    def grange(r):
        E = res["runs"][r]["E"]
        vals = [E[g] for g in res["agreement"] if "dapo" not in g and isinstance(E.get(g), (int, float))]
        return min(vals), max(vals)
    hacked = sorted(res["hacked"], key=lambda r: res["runs"][r]["cut_late"])
    quiet = [r for r in res["runs"] if r not in hacked]
    lo = min(grange(r)[0] for r in quiet)
    hi_others = max(grange(r)[1] for r in quiet if r != "trl_accuracy_q06_s0")
    rows = [f"No detected increase & {len(quiet)} runs$^\\dagger$ & "
            f"$\\leq${max(res['runs'][r]['cut_late'] for r in quiet):.2f} & {f(lo)} to {f(hi_others)}$^\\ddagger$ & -- & --" + r" \\"]
    for r in hacked:
        v = res["runs"][r]
        a_, b_ = grange(r)
        c = res["label_model"][r]["estimates"]["committed"]
        d = res["label_model"][r]["estimates"]["finished"]
        kind = "Hacking, no truncation" if v["cut_late"] < 0.1 else "Hacking with truncation"
        kind = "" if rows and kind in rows[-1] or (kind == "Hacking with truncation" and any(kind in x for x in rows)) else kind
        if kind:
            rows.append(r"\midrule")
        rows.append(f"{kind} & {MODEL[v['model']]}, seed {v['seed']} & {v['cut_late']:.2f} & {f(a_)} to {f(b_)} & "
                    f"{f(c[0])} [{f(c[1])}, {f(c[2])}] & {f(d[0])} [{f(d[1])}, {f(d[2])}]" + r" \\")
    open(os.path.join(out, "summary.tex"), "w").write("\n".join(rows) + "\n")
    # absolute share of rollouts rewarded but rejected, early and late (appendix)
    ab = res["review"]["absolute"]
    order = ["trl_accuracy", "gold_label_nocut", "substring_all_nocut", "substring_all"]
    lines = []
    for r in sorted(res["runs"], key=lambda r: (order.index(res["runs"][r]["cond"]), res["runs"][r]["model"], res["runs"][r]["seed"])):
        v = res["runs"][r]
        lines.append(f"{COND[v['cond']]} & {MODEL[v['model']]} & {v['seed']} & "
                     + " & ".join(f"{ab[r][g]['early']:.2f} / {ab[r][g]['late']:.2f}" for g in ("strict", "TRL", "Math-Verify", REF))
                     + r" \\")
    open(os.path.join(out, "absolute.tex"), "w").write("\n".join(lines) + "\n")
    # where each hand-label estimate comes from
    names = {"control_right": "late, all rule graders right", "control_wrong": "late, all rule graders wrong",
             "loop_cut": "late, truncated", "loop_uncut": "late, terminated, disputed", "early": "early, disputed",
             "agree_right": "early, all rule graders right", "agree_wrong": "early, all rule graders wrong"}
    lines = []
    for r in sorted(res["hacked"], key=lambda r: -res["runs"][r]["cut_late"]):
        st = res["runs"][r]["E"]["hand labels: strata"]
        first = True
        for s in ("loop_cut", "loop_uncut", "control_right", "control_wrong", "early", "agree_right", "agree_wrong"):
            if s not in st:
                continue
            v = st[s]
            run = f"{MODEL[res['runs'][r]['model']]} s{res['runs'][r]['seed']}" if first else ""
            first = False
            lines.append(f"{run} & {names[s]} & {v['answers']} & {v['source']} & {v['labels']} & {v['wrong_share']:.2f}" + r" \\")
        lines.append(r"\midrule")
    open(os.path.join(out, "strata.tex"), "w").write("\n".join(lines[:-1]) + "\n")


if __name__ == "__main__":
    main()
