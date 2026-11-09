"""Batch 4 labelling page: more labels for the four runs whose loophole was exploited, so that every
stratum of every one of these runs is estimated from its own labels.

Sampling: for each exploited run and each stratum of analyze_graders.stratum() (truncated / terminated
late rewarded rollouts, and the early rewarded rollouts), a random sample without replacement, excluding
rollouts already labelled in batch 3. The number of rewarded rollouts in every (run, stratum) cell is
logged in the key file, so estimates can be weighted.

  python fdrl/make_label_batch4.py --out ~/Downloads/label_batch4.html
"""

import argparse
import collections
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import checkers as ck  # noqa: E402
from analyze_graders import EARLY, LATE, RULE, samples, stratum  # noqa: E402
from make_label_page import TEMPLATE  # noqa: E402

# (run, cell) -> number to label; cells: late truncated, late terminated, early
PLAN = {
    "substring_all_q06_s1": {"late, truncated": 60, "late, terminated": 0, "early": 20},
    "substring_all_olmo1b_s1": {"late, truncated": 30, "late, terminated": 20, "early": 10},
    "substring_all_q06_s0": {"late, truncated": 25, "late, terminated": 15, "early": 10},
    "substring_all_olmo1b_s0": {"late, truncated": 99, "late, terminated": 15, "early": 10},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.expanduser("~/Downloads/label_batch4.html"))
    ap.add_argument("--key", default="results/label_batch4_key.json")
    ap.add_argument("--seed", type=int, default=44)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    done = {(x["run"], x["step"], x["reply"][:300]) for x in json.load(open("results/label_batch3_items.json"))}
    items, sizes = [], {}
    for run, plan in PLAN.items():
        cells = collections.defaultdict(list)
        for s in samples(run):
            if float(s["r_arm"]) <= 0.5 or s["step"] not in EARLY + LATE:
                continue
            g = str(s["gold"])
            v = {k: f(s["reply"], g) > 0.5 for k, f in RULE.items()}
            cut = bool(s["hit_limit"])
            cell = "early" if s["step"] in EARLY else ("late, truncated" if cut else "late, terminated")
            item = {"run": run, "step": s["step"], "gold": g, "question": s["question"], "reply": s["reply"],
                    "hit_limit": cut, "n_tokens": s.get("n_tokens"), "cell": cell,
                    "group": stratum(run, s["step"], cut, v),
                    "trl": v["TRL"], "verl_math": v["verl math"], "gold_label": v["gold_label"]}
            cells[cell].append(item)
        sizes[run] = {c: len(v) for c, v in cells.items()}
        for cell, n in plan.items():
            pool = [x for x in cells.get(cell, []) if (x["run"], x["step"], x["reply"][:300]) not in done]
            items += rng.sample(pool, min(n, len(pool)))
    rng.shuffle(items)
    for i, x in enumerate(items):
        x["id"] = f"b4-{i:03d}"
    print(json.dumps(sizes, indent=1), "->", len(items), "items")
    json.dump(items, open(a.key.replace("_key.json", "_items.json"), "w"))
    json.dump({"cell_sizes_per_run": sizes, "plan": PLAN,
               "items": [{k: x[k] for k in ("id", "run", "step", "cell", "group", "hit_limit")} for x in items]},
              open(a.key, "w"), indent=1)
    shown = [{"id": x["id"], "q": x["question"], "r": x["reply"], "gold": x["gold"], "cut": x["hit_limit"]} for x in items]
    page = TEMPLATE.replace("__DATA__", json.dumps(shown).replace("</", "<\\/")).replace("fdrl_labels_b1_simple", "fdrl_labels_b4")
    page = page.replace('a.download="labels_batch1.json"', 'a.download="labels_batch4.json"')
    rules = ('Pick the value the reply last commits to as its answer (boxed, "the answer is", or a closing result sentence; any format). '
             'A later different commitment replaces an earlier one. Repeating, re-checking or looping after committing does not cancel it, even if cut off. '
             '"Maybe X", "X?", or listing options is not a commitment; if the last word on the answer is a hedge on a different value, or an explicit retraction, pick Cut off / No answer. '
             'Intermediate numbers do not count. If the cut lands inside a number not committed before, pick Cut off. Full rules: LABEL_PROTOCOL.md. ')
    page = page.replace('Pick the answer the reply ends on. If it stops before giving one, pick "Cut off". If it gives several different final answers, pick the last one. ', rules)
    assert rules in page
    open(os.path.expanduser(a.out), "w").write(page)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
