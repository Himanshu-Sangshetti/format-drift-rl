"""Additional analyses for the paper, from the released outputs only.

  drift      per run and saved step: share of answers with nested boxes, with 8+ boxes (repetition),
             mean length and the box-matching referee's false-alarm share among paid answers
  bootstrap  95% intervals for exploitation E per run (answers resampled within each saved step)
  pressure   wrong-reward share of the starting model, omega = E[r_c (1-g)] / E[r_c], per checker
Writes results/analysis_extra.json.
"""

import json
import os
import random
import re
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import checkers as ck  # noqa: E402
import code_env as ce  # noqa: E402
from relabel import FIRST, LAST, read_jsonl  # noqa: E402

ROOT = sys.argv[1] if len(sys.argv) > 1 else "results"
RUNS = os.path.join(ROOT, "runs")
out = {"drift": {}, "bootstrap": {}, "pressure": {}}


def nested(t):
    return bool(re.search(r"\\boxed\{\s*\\(boxed|mathbf|textbf)", t))


for run in sorted(os.listdir(RUNS)):
    if "_s" not in run:
        continue
    task = json.load(open(os.path.join(RUNS, run, "config.json"))).get("task", "math")
    S = read_jsonl(os.path.join(RUNS, run, "samples.jsonl"))
    for s in S:
        s["new"] = ck.gold_label(s["reply"], s["gold"]) if task == "math" else s["r_strict"]
    rows = []
    for step in sorted({s["step"] for s in S}):
        g = [s for s in S if s["step"] == step]
        paid_old_wrong = [s for s in g if s["r_arm"] > .5 and s["r_strict"] == 0]
        rows.append({"step": step,
                     "nested": sum(nested(s["reply"]) for s in g) / len(g),
                     "repeat8": sum(s["reply"].count("boxed") >= 8 for s in g) / len(g),
                     "chars": st.mean(len(s["reply"]) for s in g),
                     "false_alarm": (sum(s["new"] == 1 for s in paid_old_wrong) / len(paid_old_wrong)) if paid_old_wrong else None,
                     "n_flagged": len(paid_old_wrong)})
    out["drift"][run] = rows
    # bootstrap of E: resample the 64 answers within each saved step
    by = {}
    for s in S:
        by.setdefault(s["step"], []).append(float(s["r_arm"] > .5 and s["new"] == 0))
    rng = random.Random(0)
    est = []
    for _ in range(2000):
        f = st.mean(st.mean(rng.choices(by[t], k=len(by[t]))) for t in FIRST)
        l = st.mean(st.mean(rng.choices(by[t], k=len(by[t]))) for t in LAST)
        est.append(l - f)
    est.sort()
    point = st.mean(st.mean(by[t]) for t in LAST) - st.mean(st.mean(by[t]) for t in FIRST)
    out["bootstrap"][run] = {"E": point, "lo": est[50], "hi": est[1949]}

# wrong-reward share of the starting model, from the base audits (judge: from its run's first saved steps)
base = read_jsonl(os.path.join(ROOT, "audits_math", "base_samples.jsonl"))
lab = [ck.gold_label(x["reply"], x["gold"]) for x in base]
for c in ("strict", "substring_tail50", "tail50_slope", "tail50_density", "substring_all"):
    r = [ck.POINTWISE[c](x["reply"], x["gold"]) for x in base]
    paid_right = sum(v > .5 and g == 1 for v, g in zip(r, lab)) / len(r)
    out["pressure"][c] = {"omega": sum(v * (1 - g) for v, g in zip(r, lab)) / max(sum(r), 1e-9),
                          "h": sum(v > .5 and g == 0 for v, g in zip(r, lab)) / len(r), "p": paid_right,
                          "mean_reward": sum(r) / len(r), "source": "base audit, 2,048 answers"}
mb = read_jsonl(os.path.join(ROOT, "audits_mbpp", "base_samples.jsonl"))
r = ce.score_batch(ce.mbpp_base, [x["reply"] for x in mb], [x["gold"] for x in mb])
g = [x["r_strict"] for x in mb]
out["pressure"]["mbpp_base"] = {"omega": sum(v * (1 - y) for v, y in zip(r, g)) / max(sum(r), 1e-9),
                                "h": sum(v > .5 and y == 0 for v, y in zip(r, g)) / len(r),
                                "p": sum(v > .5 and y == 1 for v, y in zip(r, g)) / len(r),
                                "mean_reward": sum(r) / len(r), "source": "base audit, 1,200 answers"}
out["pressure"]["mbpp_plus"] = {"omega": 0.0, "h": 0.0, "p": out["pressure"]["mbpp_base"]["p"], "source": "honest by construction"}
J = [s for s in read_jsonl(os.path.join(RUNS, "judge_ref_prob_s0", "samples.jsonl")) if s["step"] in FIRST]
jl = [ck.gold_label(s["reply"], s["gold"]) for s in J]
out["pressure"]["judge_ref_prob"] = {"omega": sum(s["r_arm"] * (1 - y) for s, y in zip(J, jl)) / sum(s["r_arm"] for s in J),
                                     "h": sum(s["r_arm"] > .5 and y == 0 for s, y in zip(J, jl)) / len(J),
                                     "p": sum(s["r_arm"] > .5 and y == 1 for s, y in zip(J, jl)) / len(J),
                                     "source": "judge run, steps 1 and 10 (128 answers)"}
json.dump(out, open(os.path.join(ROOT, "analysis_extra.json"), "w"), indent=1)
for c, v in out["pressure"].items():
    print(f"{c:18s} omega {v['omega']:.3f}  h {v['h']:.3f}  p {v['p']:.3f}")
for r, v in out["bootstrap"].items():
    print(f"{r:22s} E {v['E']:+.3f} [{v['lo']:+.3f}, {v['hi']:+.3f}]")
