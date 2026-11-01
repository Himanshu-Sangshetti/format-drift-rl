"""Analyses requested in review (all from released outputs).

  cluster bootstrap of E (problems resampled within saved steps)
  random-problem baseline for reach and per-problem enrichment
  referee sensitivity: unboxed replies counted wrong (strict fallback), E and permutation tests for tail50 arms
  old-referee permutation p-values on saved steps
  judge h and omega from one sample with one referee
Writes results/analysis_review.json.
"""

import itertools
import json
import os
import random
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import checkers as ck  # noqa: E402
from relabel import FIRST, LAST, perm_test, read_jsonl  # noqa: E402

ROOT = sys.argv[1] if len(sys.argv) > 1 else "results"
RUNS = os.path.join(ROOT, "runs")
out = {"cluster_bootstrap": {}, "sensitivity": {}, "old_perm": {}, "reach_baseline": {}, "judge_start": {}}


def strict_fallback(reply, gold):
    """Robust referee, except that a reply with no box holding a number counts as wrong."""
    boxes = [ck.box_value(b) for b in ck.all_boxed(reply)]
    if not any(any(ch.isdigit() for ch in b) for b in boxes):
        return 0.0
    return ck.gold_label(reply, gold)


E_new, E_old, E_sens = {}, {}, {}
for run in sorted(os.listdir(RUNS)):
    if "_s" not in run:
        continue
    task = json.load(open(os.path.join(RUNS, run, "config.json"))).get("task", "math")
    S = [s for s in read_jsonl(os.path.join(RUNS, run, "samples.jsonl")) if s["step"] in FIRST | LAST]
    for s in S:
        s["new"] = ck.gold_label(s["reply"], s["gold"]) if task == "math" else s["r_strict"]
        s["sens"] = strict_fallback(s["reply"], s["gold"]) if task == "math" else s["r_strict"]
    fp = lambda s, k: float(s["r_arm"] > .5 and s[k] == 0)  # noqa: E731

    def E(key, sample=None):
        src = sample if sample is not None else S
        f = st.mean(st.mean(fp(s, key) for s in src if s["step"] == t) for t in FIRST)
        l = st.mean(st.mean(fp(s, key) for s in src if s["step"] == t) for t in LAST)
        return l - f
    E_new[run], E_old[run], E_sens[run] = E("new"), E("r_strict"), E("sens")
    # cluster bootstrap: resample problems (groups of 8 answers) within each saved step
    groups = {}
    for s in S:
        groups.setdefault((s["step"], s["question"]), []).append(s)
    per_step = {}
    for (t, q), g in groups.items():
        per_step.setdefault(t, []).append(g)
    rng = random.Random(0)
    est = []
    for _ in range(2000):
        smp = [a for t, gs in per_step.items() for g in rng.choices(gs, k=len(gs)) for a in g]
        est.append(E("new", smp))
    est.sort()
    any_event = any(fp(s, "new") for s in S)
    out["cluster_bootstrap"][run] = {"E": E_new[run], "lo": est[50], "hi": est[1949], "any_event": any_event,
                                     "se": st.pstdev(est)}
    out["sensitivity"][run] = {"E_strict_fallback": E_sens[run]}

g = lambda d, c: [d[f"{c}_s{s}"] for s in range(4)]  # noqa: E731
for name, d in (("robust", E_new), ("old", E_old), ("strict_fallback", E_sens)):
    out["old_perm" if name == "old" else ("sensitivity" if name == "strict_fallback" else "cluster_bootstrap")][f"perm_{name}"] = {
        "slope_vs_tail50": perm_test(g(d, "tail50_slope"), g(d, "substring_tail50")),
        "slope_vs_density": perm_test(g(d, "tail50_slope"), g(d, "tail50_density"))}

# reach baseline from the base audit (256 problems x 8 answers): share of random problems with >= 1 paid-but-wrong in 8,
# against the same quantity predicted for exploited problems from their 64-sample rates
base = read_jsonl(os.path.join(ROOT, "audits_math", "base_samples.jsonl"))
byq = {}
for x in base:
    byq.setdefault(x["question"], []).append(x)
reach = json.load(open(os.path.join(ROOT, "reach_report_math.json")))
for cond, runs in (("substring_all", ["substring_all_s0", "substring_all_s1"]),):
    fn = ck.POINTWISE[cond]
    hit8 = [any(fn(x["reply"], x["gold"]) > .5 and ck.gold_label(x["reply"], x["gold"]) == 0 for x in xs) for xs in byq.values()]
    rate = [st.mean(fn(x["reply"], x["gold"]) > .5 and ck.gold_label(x["reply"], x["gold"]) == 0 for x in xs) for xs in byq.values()]
    entry = {"random_problems": len(hit8), "random_reach_at_8": st.mean(hit8), "random_mean_rate": st.mean(rate)}
    for r in runs:
        qs = [p["base_pbw_rate"] for p in reach[r]["problems"]]
        entry[r] = {"exploited_reach_at_8_pred": st.mean(1 - (1 - q) ** 8 for q in qs), "exploited_mean_rate": st.mean(qs),
                    "reach64": reach[r]["reach"]}
    out["reach_baseline"][cond] = entry
mb = read_jsonl(os.path.join(ROOT, "audits_mbpp", "base_samples.jsonl"))
import code_env as ce  # noqa: E402
byt = {}
for x in mb:
    byt.setdefault(x["gold"], []).append(x)
pay = {}
for t, xs in byt.items():
    rs = ce.score_batch(ce.mbpp_base, [x["reply"] for x in xs], [t] * len(xs))
    pay[t] = [r > .5 and x["r_strict"] == 0 for r, x in zip(rs, xs)]
rm = json.load(open(os.path.join(ROOT, "reach_report_mbpp.json")))
out["reach_baseline"]["mbpp_base"] = {"random_problems": len(pay), "random_reach_at_8": st.mean(any(v) for v in pay.values()),
                                      "random_mean_rate": st.mean(st.mean(v) for v in pay.values()),
                                      **{r: {"reach64": rm[r]["reach"], "exploited_mean_rate": rm[r]["base_pbw_rate_on_hack_problems"]}
                                         for r in ("mbpp_base_s0", "mbpp_base_s1", "mbpp_base_s2")}}
# judge from one sample (steps 1 and 10), one referee
J = [s for s in read_jsonl(os.path.join(RUNS, "judge_ref_prob_s0", "samples.jsonl")) if s["step"] in FIRST]
jl = [ck.gold_label(s["reply"], s["gold"]) for s in J]
out["judge_start"] = {"n": len(J), "h": st.mean(s["r_arm"] > .5 and y == 0 for s, y in zip(J, jl)),
                      "omega": sum(s["r_arm"] * (1 - y) for s, y in zip(J, jl)) / sum(s["r_arm"] for s in J),
                      "withheld_from_right": st.mean((1 - s["r_arm"]) * y for s, y in zip(J, jl))}
json.dump(out, open(os.path.join(ROOT, "analysis_review.json"), "w"), indent=1, default=float)
print(json.dumps({k: v for k, v in out.items() if k != "cluster_bootstrap" and k != "sensitivity"}, indent=1, default=float)[:3000])
print("sensitivity perm", out["sensitivity"]["perm_strict_fallback"])
for r in ("tail50_slope_s0", "tail50_slope_s1", "tail50_slope_s2", "tail50_slope_s3", "substring_tail50_s0", "substring_tail50_s1", "tail50_density_s2", "substring_all_s0", "mbpp_base_s0", "judge_ref_prob_s0"):
    c = out["cluster_bootstrap"][r]; print(f"{r:22s} E {c['E']:+.3f} cluster CI [{c['lo']:+.3f},{c['hi']:+.3f}] se {c['se']:.3f} | strict-fallback E {out['sensitivity'][r]['E_strict_fallback']:+.3f}")
