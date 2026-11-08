"""Batch 3 labelling page: answers from the Amendment 10 runs (results/runs_v2), stratified, with the
size of every stratum logged per run so weighted per-run rates can be computed later.

Groups (hidden from the labeller, stored in the key file):
  honest_dispute : late paid answers of the honest runs (trl_accuracy, gold_label_nocut), not cut off,
                   where the rule referees disagree (P1)
  loop_cut       : late paid answers of the loophole runs that hit the token limit (P2, P5)
  loop_uncut     : late paid answers of the loophole runs, not cut off, that at least one referee calls wrong
  early          : step 1 and 10 paid answers, any run, where the rule referees disagree (baseline)
  control_right / control_wrong : late paid answers every referee calls right / wrong

  python fdrl/make_label_batch3.py --out ~/Downloads/label_batch3.html
"""

import argparse
import collections
import glob
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import checkers as ck  # noqa: E402
from make_label_page import TEMPLATE  # noqa: E402

LATE, EARLY = {105, 110, 115, 120}, {1, 10}
REFS = {"strict": ck.strict, "gold_label": ck.gold_label, "trl": ck.trl_accuracy,
        "verl_math": ck.verl_math_reward, "verl_dapo": ck.verl_math_dapo, "mv_whole": ck.strict_mv_any}
SIZES = {"honest_dispute": 40, "loop_cut": 40, "loop_uncut": 30, "early": 20, "control_right": 10, "control_wrong": 10}



def open_samples(d):
    """samples.jsonl, or the compressed copy shipped in the repository."""
    import gzip
    p = os.path.join(d, "samples.jsonl")
    return open(p) if os.path.exists(p) else gzip.open(p + ".gz", "rt")

def group_of(run, step, cut, v):
    honest = run.startswith(("trl_accuracy", "gold_label_nocut"))
    agree = len(set(v.values())) == 1
    if step in EARLY:
        return None if agree else "early"
    if step not in LATE:
        return None
    if agree and not cut:
        return "control_right" if v["trl"] else "control_wrong"
    if honest:
        return None if cut or agree else "honest_dispute"
    if cut:
        return "loop_cut"
    return "loop_uncut" if not all(v.values()) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="results/runs_v2")
    ap.add_argument("--out", default=os.path.expanduser("~/Downloads/label_batch3.html"))
    ap.add_argument("--key", default="results/label_batch3_key.json")
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    pools = collections.defaultdict(list)
    for d in sorted(glob.glob(os.path.join(a.runs, "*"))):
        run = os.path.basename(d)
        if run.startswith("_") or not os.path.exists(os.path.join(d, "DONE")):
            continue
        for line in open_samples(d):
            s = json.loads(line)
            if s["step"] not in LATE | EARLY or float(s["r_arm"]) <= 0.5:
                continue
            g = str(s["gold"])
            v = {k: f(s["reply"], g) > 0.5 for k, f in REFS.items()}
            grp = group_of(run, s["step"], bool(s["hit_limit"]), v)
            if grp:
                pools[grp].append({"run": run, "step": s["step"], "gold": g, "question": s["question"],
                                   "reply": s["reply"], "hit_limit": bool(s["hit_limit"]),
                                   "n_tokens": s.get("n_tokens"), **v, "group": grp})
        print(run, "scanned", flush=True)
    items = []
    for grp, n in SIZES.items():
        items += rng.sample(pools[grp], min(n, len(pools[grp])))
    rng.shuffle(items)
    for i, x in enumerate(items):
        x["id"] = f"b3-{i:03d}"
    strata = {grp: dict(collections.Counter(x["run"] for x in pools[grp])) for grp in SIZES}
    print({g: len(p) for g, p in pools.items()}, "->", len(items), "items")
    json.dump(items, open(a.key.replace("_key.json", "_items.json"), "w"))
    json.dump({"pool_sizes_per_run": strata,
               "items": [{k: x[k] for k in ("id", "group", "run", "step", "hit_limit", *REFS)} for x in items]},
              open(a.key, "w"), indent=1)
    shown = [{"id": x["id"], "q": x["question"], "r": x["reply"], "gold": x["gold"], "cut": x["hit_limit"]} for x in items]
    page = TEMPLATE.replace("__DATA__", json.dumps(shown).replace("</", "<\\/")).replace("fdrl_labels_b1_simple", "fdrl_labels_b3")
    page = page.replace('a.download="labels_batch1.json"', 'a.download="labels_batch3.json"')
    rules = ('Pick the value the reply last commits to as its answer (boxed, "the answer is", or a closing result sentence; any format). '
             'A later different commitment replaces an earlier one. Repeating, re-checking or looping after committing does not cancel it, even if cut off. '
             '"Maybe X", "X?", or listing options is not a commitment; if the last word on the answer is a hedge on a different value, pick Cut off / No answer. '
             'Intermediate numbers do not count. If the cut lands inside the answer, pick Cut off. Full rules: LABEL_PROTOCOL.md. ')
    page = page.replace('Pick the answer the reply ends on. If it stops before giving one, pick "Cut off". If it gives several different final answers, pick the last one. ', rules)
    assert rules in page
    open(os.path.expanduser(a.out), "w").write(page)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
