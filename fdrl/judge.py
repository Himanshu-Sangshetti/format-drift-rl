"""LLM-judge referees (Amendment 10): gpt-oss-20b and Qwen3-8B, each sighted and blind, and
CompassVerifier-3B. Temperature 0, one call per answer, vLLM offline.

  sighted : question, reference answer and the full reply -> CORRECT / INCORRECT
  blind   : the end of the reply only, no reference -> the committed final answer, or NONE
            (compared with the reference afterwards; no built-in cut-off rule)
  compass : CompassVerifier's own prompt (CV_PROMPT, fetched from the pinned commit below) -> A / B / C
            (A = correct; B and C count as wrong, as its authors recommend)

Build the inputs locally (every paid answer at steps 1, 10, 105-120 of the second run set, plus every
hand-labelled answer of the first run set):
  python fdrl/judge.py build --out results/judge_inputs.jsonl
Run on a GPU:
  python fdrl/judge.py run --inputs judge_inputs.jsonl --judge gpt-oss-20b --mode sighted --out judged/
"""

import argparse
import glob
import json
import os
import re
import sys
import time
import urllib.request

EARLY_LATE = {1, 10, 105, 110, 115, 120}
JUDGES = {"gpt-oss-20b": "openai/gpt-oss-20b", "qwen3-8b": "Qwen/Qwen3-8B",
          "gpt-oss-120b": "openai/gpt-oss-120b", "compassverifier-3b": "opencompass/CompassVerifier-3B"}
CV_URL = ("https://raw.githubusercontent.com/open-compass/CompassVerifier/"
          "2d7cba6df0b21f9c6121786ac1e5770c68473598/src/prompts.py")
BLIND_CHARS = 1500  # the end of the reply shown to the blind judge

SIGHTED = """You are grading a student's reply to a maths question.

Question:
{question}

Reference answer: {gold}

Student reply:
<<<
{reply}
>>>

Does the student's reply commit to a final answer that equals the reference answer? Ignore formatting
(boxed or not, plain text, another language). Judge only the final answer the reply commits to, not the
reasoning. If the reply never commits to a final answer, it is INCORRECT.
Respond with exactly one word: CORRECT or INCORRECT."""

BLIND = """Below is the END of a student's reply to a maths question (the beginning may be omitted, and the
reply may stop mid-sentence).

Question:
{question}

End of the reply:
<<<
{tail}
>>>

What final answer does the reply commit to? A hedge ("maybe 12") or an unfinished statement is not a
commitment. Respond with only the final answer as a number, or NONE if the reply does not commit to one."""


def build(a):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    rows = []
    for d in sorted(glob.glob("results/runs_v2/*")):
        run = os.path.basename(d)
        if run.startswith("_") or not os.path.exists(os.path.join(d, "DONE")):
            continue
        for i, line in enumerate(open(os.path.join(d, "samples.jsonl"))):
            s = json.loads(line)
            if s["step"] in EARLY_LATE and float(s["r_arm"]) > 0.5:
                rows.append({"key": f"{run}#{i}", "run": run, "step": s["step"], "question": s["question"],
                             "gold": str(s["gold"]), "reply": s["reply"], "hit_limit": bool(s["hit_limit"])})
    for b in (1, 2):
        for x in json.load(open(f"results/label_batch{b}_items.json")):
            rows.append({"key": x["id"], "run": x["run"], "step": x["step"], "question": x["question"],
                         "gold": str(x["gold"]), "reply": x["reply"], "hit_limit": bool(x["hit_limit"])})
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(len(rows), "inputs ->", a.out)


def norm(x):
    s = re.sub(r"\\boxed\{([^{}]*)\}", r"\1", str(x))
    s = re.sub(r"[^0-9.\-]", "", s)
    return re.sub(r"\.0+$", "", s)


def parse(judge, mode, text, gold):
    t = text.strip()
    if "assistantfinal" in t:  # gpt-oss: keep only the final channel if markers survived decoding
        t = t.split("assistantfinal")[-1].strip()
    t = re.sub(r"<think>.*?</think>", "", t, flags=re.S).strip()
    if judge == "compassverifier-3b":
        m = re.findall(r"\b([ABC])\b", t)
        return (m[-1] == "A") if m else None
    if mode == "sighted":
        u = t.upper()
        if "INCORRECT" in u:
            return False
        return True if "CORRECT" in u else None
    if re.fullmatch(r"\W*NONE\W*", t.upper()):
        return False
    n = norm(t.splitlines()[-1] if t else "")
    return (n == norm(gold)) if n else None


def run(a):
    from vllm import LLM, SamplingParams

    rows = [json.loads(line) for line in open(a.inputs)]
    if a.limit:
        rows = rows[: a.limit]
    os.makedirs(a.out, exist_ok=True)
    if a.judge == "compassverifier-3b":
        ns = {}
        exec(urllib.request.urlopen(CV_URL, timeout=60).read().decode(), ns)
        msgs = [[{"role": "user", "content": ns["CV_PROMPT"].format(
            question=r["question"], gold_answer=r["gold"], llm_response=r["reply"])}] for r in rows]
        kw, max_len, max_out = {}, 4096, 16
    else:
        tmpl = SIGHTED if a.mode == "sighted" else BLIND
        msgs = [[{"role": "user", "content": tmpl.format(question=r["question"], gold=r["gold"], reply=r["reply"],
                                                         tail=r["reply"][-BLIND_CHARS:])}] for r in rows]
        if a.judge.startswith("gpt-oss"):
            kw, max_len, max_out = {"reasoning_effort": a.effort}, 6144, 2048
        else:
            kw, max_len, max_out = {"enable_thinking": False}, 4096, 32
    llm = LLM(model=JUDGES[a.judge], max_model_len=max_len, gpu_memory_utilization=0.9, seed=0)
    sp = SamplingParams(temperature=0.0, max_tokens=max_out)
    t0 = time.time()
    outs = llm.chat(msgs, sp, chat_template_kwargs=kw, use_tqdm=True)
    name = f"{a.judge}_{a.mode}"
    with open(os.path.join(a.out, f"{name}.jsonl"), "w") as f:
        for r, o in zip(rows, outs):
            text = o.outputs[0].text
            f.write(json.dumps({"key": r["key"], "raw": text[-400:], "finish": o.outputs[0].finish_reason,
                                "right": parse(a.judge, a.mode, text, r["gold"])}) + "\n")
    json.dump({"judge": a.judge, "model": JUDGES[a.judge], "mode": a.mode, "n": len(rows), "temperature": 0.0,
               "max_tokens": max_out, "max_model_len": max_len, "chat_template_kwargs": kw,
               "blind_chars": BLIND_CHARS, "seconds": round(time.time() - t0)},
              open(os.path.join(a.out, f"{name}_config.json"), "w"), indent=1)
    print(name, "done", len(rows), "in", round(time.time() - t0), "s")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", default="results/judge_inputs.jsonl")
    r = sub.add_parser("run")
    r.add_argument("--inputs", required=True)
    r.add_argument("--judge", choices=list(JUDGES), required=True)
    r.add_argument("--mode", choices=["sighted", "blind", "compass"], default="sighted")
    r.add_argument("--effort", default="low", help="gpt-oss reasoning effort")
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--out", default="judged")
    a = ap.parse_args()
    build(a) if a.cmd == "build" else run(a)


if __name__ == "__main__":
    main()
