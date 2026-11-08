"""Score every referee against the hand labels.

Human truth for an item: the answer the labeller says the reply commits to, compared with the gold
(integers here); "__cut" and "__none" mean no answer, so the reply is wrong.
For each referee: agreement with the human, split into
  false alarm : referee says WRONG, human says RIGHT   (on paid answers this is a fake "hack")
  missed      : referee says RIGHT, human says WRONG / no answer   (a hack the referee does not see)
Each referee is also run with "+cut" (an answer that hit the token limit counts as wrong).

  python fdrl/score_labels.py --labels ~/Downloads/labels_batch1.json --items results/label_batch1_items.json
"""

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import checkers as ck  # noqa: E402

REFEREES = {
    "strict (ours)": ck.strict,
    "gold_label (ours)": ck.gold_label,
    "TRL accuracy_reward": ck.trl_accuracy,
    "verl math_reward": ck.verl_math_reward,
    "verl math_dapo": ck.verl_math_dapo,
    "Math-Verify whole reply": ck.strict_mv_any,
}


def norm(x):
    s = re.sub(r"[^0-9.\-]", "", str(x or ""))
    return re.sub(r"\.0+$", "", s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=os.path.expanduser("~/Downloads/labels_batch1.json"))
    ap.add_argument("--items", default="results/label_batch1_items.json")
    ap.add_argument("--out", default="results/label_batch1_scored.json")
    a = ap.parse_args()
    items = {x["id"]: x for x in json.load(open(a.items))}
    labels = {l["id"]: l["answer"] for l in json.load(open(a.labels)) if l.get("answer")}
    rows = []
    for iid, ans in labels.items():
        x = items[iid]
        answered = ans not in ("__cut", "__none")
        human = answered and norm(ans) == norm(x["gold"])
        r = {"id": iid, "group": x["group"], "run": x["run"], "hit_limit": x["hit_limit"], "label": ans,
             "answered": answered, "human_right": human}
        for name, f in REFEREES.items():
            v = f(x["reply"], str(x["gold"])) > 0.5
            r[name] = v
            r[name + " +cut"] = v and not x["hit_limit"]
        rows.append(r)
    json.dump(rows, open(a.out, "w"), indent=1)
    n = len(rows)
    print(f"{n} labelled answers. Human: {sum(r['human_right'] for r in rows)} right, "
          f"{sum(r['answered'] and not r['human_right'] for r in rows)} wrong answer, "
          f"{sum(not r['answered'] for r in rows)} no answer (cut off / none)")
    print(f"by group: " + ", ".join(f"{g} {sum(r['group']==g for r in rows)}" for g in sorted({r['group'] for r in rows})))
    names = [k for k in rows[0] if k in REFEREES or k.endswith(" +cut")]
    print(f"\n{'referee':36s} {'agrees':>7s} {'false alarm':>12s} {'missed':>7s}   (false alarm = says wrong, human right; missed = says right, human wrong/none)")
    for nm in names:
        agree = sum(r[nm] == r["human_right"] for r in rows)
        fa = sum((not r[nm]) and r["human_right"] for r in rows)
        mi = sum(r[nm] and not r["human_right"] for r in rows)
        print(f"{nm:36s} {agree:>4d}/{n:<3d} {fa:>12d} {mi:>7d}")
    for g in sorted({r["group"] for r in rows}):
        sub = [r for r in rows if r["group"] == g]
        print(f"\n[{g}] n={len(sub)}  human right {sum(r['human_right'] for r in sub)}, no answer {sum(not r['answered'] for r in sub)}")
        for nm in names:
            fa = sum((not r[nm]) and r["human_right"] for r in sub)
            mi = sum(r[nm] and not r["human_right"] for r in sub)
            print(f"   {nm:36s} false alarm {fa:>3d}   missed {mi:>3d}")


if __name__ == "__main__":
    main()
