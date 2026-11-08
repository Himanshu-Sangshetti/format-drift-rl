"""Replay finished runs into Weights & Biases from the files on disk (no GPU, no retraining).

For every run in results/runs/<run>/ it logs:
  - every training-log metric per step (reward, accuracy, paid-but-wrong rate, length, ...)
  - at each saved step, metrics recomputed from the saved answers under BOTH referees
    (fixed = gold_label, old = one-distinct-box `strict`), plus format-drift shares
  - one table of every saved answer: checker pay, both verdicts, whether it is a drift misread
  - summary: exploitation E under each referee (steps 105-120 minus steps 1 and 10)
Optionally (--compare results/grader_comparison.json) a separate run with the grader comparison table.

Usage:
  python fdrl/wandb_replay.py --project format-drift-rl
"""

import argparse
import glob
import gzip
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import checkers as ck  # noqa: E402

EARLY, LATE = {1, 10}, {105, 110, 115, 120}


def nested(reply):
    return bool(re.search(r"\\boxed\{\s*\\(boxed|mathbf|textbf)", reply))


def repeated(reply):
    return reply.count("\\boxed{") >= 8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default="format-drift-rl")
    ap.add_argument("--runs", default=os.path.join(os.path.dirname(__file__), "..", "results", "runs"))
    ap.add_argument("--only", default="", help="comma-separated run names (default: all)")
    ap.add_argument("--compare", default="", help="grader comparison json to log as its own run")
    a = ap.parse_args()
    import wandb

    dirs = sorted(glob.glob(os.path.join(a.runs, "*")))
    if a.only:
        keep = set(a.only.split(","))
        dirs = [d for d in dirs if os.path.basename(d) in keep]
    for d in dirs:
        run = os.path.basename(d)
        cfg = json.load(open(os.path.join(d, "config.json")))
        math = cfg.get("task", "math") == "math"
        steps = {}
        for line in gzip.open(os.path.join(d, "log.jsonl.gz"), "rt"):
            r = json.loads(line)
            s = r.pop("step")
            steps.setdefault(s, {}).update({f"train/{k}": v for k, v in r.items() if isinstance(v, (int, float))})
        rows = [json.loads(l) for l in gzip.open(os.path.join(d, "samples.jsonl.gz"), "rt")]
        by_step = {}
        for r in rows:
            by_step.setdefault(int(r["step"]), []).append(r)
        table_rows, pbw = [], {"fixed": {}, "old": {}}
        for s, rs in sorted(by_step.items()):
            n = len(rs)
            paid = [float(r["r_arm"]) > 0.5 for r in rs]
            if math:  # re-grade with the paper's final referees (the saved columns are the training-time versions)
                fixed = [ck.gold_label(r["reply"], str(r["gold"])) == 1.0 for r in rs]
                old = [ck.strict(r["reply"], str(r["gold"])) == 1.0 for r in rs]
            else:  # code: the referee (MBPP+) never changed
                fixed = old = [float(r["r_strict"]) == 1.0 for r in rs]
            m = {
                "saved/paid_but_wrong_fixed_referee": sum(p and not f for p, f in zip(paid, fixed)) / n,
                "saved/paid_but_wrong_old_referee": sum(p and not o for p, o in zip(paid, old)) / n,
                "saved/accuracy_fixed_referee": sum(fixed) / n,
                "saved/checker_reward": sum(float(r["r_arm"]) for r in rs) / n,
                "saved/mean_chars": sum(len(r["reply"]) for r in rs) / n,
            }
            if math:
                m["saved/nested_box_share"] = sum(nested(r["reply"]) for r in rs) / n
                m["saved/repeated_box_share"] = sum(repeated(r["reply"]) for r in rs) / n
            pbw["fixed"][s], pbw["old"][s] = m["saved/paid_but_wrong_fixed_referee"], m["saved/paid_but_wrong_old_referee"]
            steps.setdefault(s, {}).update(m)
            for r, p, f, o in zip(rs, paid, fixed, old):
                table_rows.append([s, run, bool(p and not f), bool(p and not o), bool(p and f and not o),
                                   float(r["r_arm"]), f, o, str(r["gold"]), r["reply"][-1500:], r["reply"], r["question"]])

        def E(which):
            e = [v for k, v in pbw[which].items() if k in EARLY]
            l = [v for k, v in pbw[which].items() if k in LATE]
            return (sum(l) / len(l) - sum(e) / len(e)) if e and l else None

        wb = wandb.init(project=a.project, name=run, group=cfg.get("cond", run.rsplit("_s", 1)[0]),
                        job_type="replay", config=cfg, reinit="finish_previous")
        for s in sorted(steps):
            wb.log(steps[s], step=s)
        wb.log({"answers": wandb.Table(
            columns=["step", "run", "paid_but_wrong_fixed", "paid_but_wrong_old", "drift_misread",
                     "checker_reward", "fixed_referee_right", "old_referee_right", "gold", "reply_end",
                     "reply_full", "question"], data=table_rows)})
        wb.summary.update({"E_fixed_referee": E("fixed"), "E_old_referee": E("old"),
                           "checker": cfg.get("cond"), "seed": cfg.get("seed")})
        wb.finish()
        print("logged", run, "E_fixed", E("fixed"), "E_old", E("old"), flush=True)

    if a.compare and os.path.exists(a.compare):
        comp = json.load(open(a.compare))
        rows = []
        for run, ph in comp.items():
            for phase, c in ph.items():
                n = c.get("n_right", 0)
                for g in ["old_strict", "hendrycks_lastbox_exact", "mathverify_lastbox", "mathverify_whole_reply"]:
                    rows.append([run, phase, g, n, c.get(g + "_says_wrong", 0),
                                 (c.get(g + "_says_wrong", 0) / n) if n else None])
        wb = wandb.init(project=a.project, name="grader-comparison", job_type="analysis", reinit="finish_previous")
        wb.log({"correct_answers_called_wrong": wandb.Table(
            columns=["run", "phase", "grader", "n_correct", "n_called_wrong", "share_called_wrong"], data=rows)})
        wb.finish()
        print("logged grader comparison", flush=True)


if __name__ == "__main__":
    main()
