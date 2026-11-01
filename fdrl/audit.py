"""Pre-training audits and the novelty measure.

  sample   draw answers from the starting model on training-distribution problems
  mass     h = share of those answers a checker pays (> 0.5) while Math-Verify rejects them
  slope    s = mean rise in a checker's payout when a frozen, gold-free edit is applied to a
           wrong answer (checkers.EDITS)
  novelty  for each run's saved answers, the per-token log-likelihood under the starting model,
           as a percentile of the starting model's own answers to training-distribution problems

Usage:
  python fdrl/audit.py audit --model Qwen/Qwen3-0.6B --conds substring_all judge_ref_prob --out audits
  python fdrl/audit.py novelty --model Qwen/Qwen3-0.6B --runs runs --base audits/base_samples.jsonl
"""

import argparse
import bisect
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import checkers as ck  # noqa: E402
import code_env as ce  # noqa: E402
from train_grpo import SUFFIX, load_train  # noqa: E402

SUFFIXES = {"math": SUFFIX, "mbpp": ce.SUFFIX}


def chat(tok, q, suffix=SUFFIX):
    return tok.apply_chat_template([{"role": "user", "content": q + suffix}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)


def sample(model_id, questions, n, max_tokens, seed, use_vllm, suffix=SUFFIX):
    """n answers per question at temperature 1, as during training."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    prompts = [chat(tok, q, suffix) for q in questions]
    if use_vllm:
        # the Modal image has no CUDA compiler for FlashInfer's JIT sampling kernel; vLLM's PyTorch
        # sampler draws from the same distribution
        os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
        from vllm import LLM, SamplingParams

        # generation_config="vllm" and explicit top_k/min_p: sample the full distribution, as training
        # does, instead of the model card's defaults (Qwen3 ships top_k=20)
        llm = LLM(model_id, seed=seed, gpu_memory_utilization=0.5, max_model_len=max_tokens + 1024,
                  generation_config="vllm")
        outs = llm.generate(prompts, SamplingParams(n=n, temperature=1.0, top_p=1.0, top_k=-1, min_p=0.0,
                                                    max_tokens=max_tokens, seed=seed))
        replies = [[c.text for c in o.outputs] for o in outs]
        del llm
        return replies
    import torch
    from transformers import AutoModelForCausalLM

    torch.manual_seed(seed)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).cuda().eval()
    replies = []
    with torch.no_grad():
        for k in range(0, len(prompts), 4):
            enc = tok(prompts[k:k + 4], return_tensors="pt", padding=True).to("cuda")
            out = model.generate(**enc, do_sample=True, temperature=1.0, top_p=1.0, top_k=0,
                                 max_new_tokens=max_tokens, num_return_sequences=n,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)
            texts = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            replies += [texts[i * n:(i + 1) * n] for i in range(len(texts) // n)]
    del model
    torch.cuda.empty_cache()
    return replies


_judges = {}


def scorer(cond, judge_model, version="v1"):
    if cond in ck.JUDGE_CONDITIONS:
        ref, prob = ck.JUDGE_CONDITIONS[cond]
        if judge_model not in _judges:
            _judges[judge_model] = ck.Judge(judge_model, reference=ref, prob=prob, version=version)
        j = _judges[judge_model]
        j.version = version

        def score(qs, gs, rs):
            j.reference, j.prob = ref, prob
            return j(qs, gs, rs)
        return score
    if cond in ce.CONDITIONS:
        fn = ce.CONDITIONS[cond]
        return lambda qs, gs, rs: ce.score_batch(fn, rs, gs)
    fn = ck.POINTWISE[cond]
    return lambda qs, gs, rs: [fn(r, g) for r, g in zip(rs, gs)]


def referee(task, rs, gs):
    if task == "mbpp":
        return ce.score_batch(ce.label, rs, gs)
    return [ck.gold_label(r, g) for r, g in zip(rs, gs)]


def task_rows(task, n, seed):
    """n distinct training-distribution problems."""
    if task == "mbpp":
        train, _ = ce.split()
        return [{"question": ce.question(t), "gold": t} for t in random.Random(seed).sample(train, n)]
    random.seed(seed)
    return random.sample(load_train(), n)


def cmd_audit(a):
    rows = task_rows(a.task, a.prompts, a.seed)
    replies = sample(a.model, [r["question"] for r in rows], a.n, a.max_tokens, a.seed, a.vllm,
                     SUFFIXES[a.task])
    os.makedirs(a.out, exist_ok=True)
    flat = [(r["question"], r["gold"], t) for r, ts in zip(rows, replies) for t in ts]
    qs, gs, rs = zip(*flat)
    s_true = referee(a.task, rs, gs)
    with open(os.path.join(a.out, "base_samples.jsonl"), "w") as f:
        for (q, g, t), s in zip(flat, s_true):
            f.write(json.dumps({"question": q, "gold": g, "reply": t, "r_strict": s}) + "\n")
    wrong = [i for i, s in enumerate(s_true) if s == 0]
    prompt_of = [k for k, ts in enumerate(replies) for _ in ts]
    outside = sum(g.strip() in t and g.strip() not in t[-50:] for _, g, t in flat) / len(flat)
    result = {"model": a.model, "task": a.task, "n_samples": len(flat), "n_prompts": len(rows),
              "base_accuracy": sum(s_true) / len(flat), "gold_outside_tail50_rate": outside,
              "strict_errors": ck.STRICT_ERRORS[0], "conds": {}}
    for cond in a.conds:
        score = scorer(cond, a.judge_model, a.judge_version)
        r = score(qs, gs, rs)
        pbw = [float(x > 0.5 and y == 0) for x, y in zip(r, s_true)]
        h = sum(pbw) / len(flat)
        result["conds"][cond] = {"h": h, "h_ci95": cluster_ci(pbw, prompt_of), "mean_reward": sum(r) / len(r)}
        if a.task != "math":  # the edit library is for maths answers; code gets h and reach only
            print(cond, json.dumps(result["conds"][cond]), flush=True)
            continue
        wq, wg, wr = [qs[i] for i in wrong], [gs[i] for i in wrong], [rs[i] for i in wrong]
        base_w = [r[i] for i in wrong]  # these answers are all gold-rejected
        per_edit, per_edit_bin = {}, {}
        for name, edit in {**ck.EDITS, **ck.NULL_EDITS}.items():
            ed = [edit(t) for t in wr]
            er = score(wq, wg, ed)
            gold_ok = [ck.gold_label(t, g) for t, g in zip(ed, wg)]
            # an edit counts only if the edited answer is still wrong: payout gained by fixing the
            # answer is not a slope towards a hack
            per_edit[name] = sum(e * (1 - k) - b for e, b, k in zip(er, base_w, gold_ok)) / max(len(wr), 1)
            per_edit_bin[name] = sum((e > 0.5 and k == 0) - (b > 0.5) for e, b, k in zip(er, base_w, gold_ok)) / max(len(wr), 1)
        null = sum(per_edit[k] for k in ck.NULL_EDITS) / len(ck.NULL_EDITS)
        null_bin = sum(per_edit_bin[k] for k in ck.NULL_EDITS) / len(ck.NULL_EDITS)
        s_raw = sum(per_edit[k] for k in ck.EDITS) / len(ck.EDITS)
        s_bin = sum(per_edit_bin[k] for k in ck.EDITS) / len(ck.EDITS)
        result["conds"][cond].update({"s": s_raw - null, "s_raw": s_raw, "s_null": null,
                                 "s_bin": s_bin - null_bin,
                                 "s_max": max(per_edit[k] for k in ck.EDITS) - null,
                                 "per_edit": per_edit, "per_edit_bin": per_edit_bin})
        print(cond, json.dumps(result["conds"][cond]), flush=True)
    json.dump(result, open(os.path.join(a.out, f"audit_{a.tag}.json"), "w"), indent=2)


def cmd_calibrate(a):
    """Judge accuracy on the starting model's answers, by gold label and by whether the answer
    has a final box. Used to choose the judge before any main run."""
    S = [json.loads(l) for l in open(a.base)]
    qs, gs, rs = [x["question"] for x in S], [x["gold"] for x in S], [x["reply"] for x in S]
    ok = [x["r_strict"] for x in S]
    boxed = [ck.last_boxed(t) is not None for t in rs]
    out = {}
    for model in a.judge_models:
        for version in a.versions:
            j = ck.Judge(model, reference=True, prob=True, version=version)
            for ref in (True, False):
                j.reference = ref
                p = j.p_yes(qs, gs, rs)
                pay = [x > 0.5 for x in p]
                cell = lambda m: [y for y, k in zip(pay, m) if k]  # noqa: E731
                rate = lambda v: sum(v) / len(v) if v else None  # noqa: E731
                key = f"{model.split('/')[-1]}|{version}|{'ref' if ref else 'noref'}"
                out[key] = {"h": sum(y and o == 0 for y, o in zip(pay, ok)) / len(S),
                            "tpr_correct": rate(cell([o == 1 for o in ok])),
                            "fpr_wrong_boxed": rate(cell([o == 0 and b for o, b in zip(ok, boxed)])),
                            "fpr_wrong_unboxed": rate(cell([o == 0 and not b for o, b in zip(ok, boxed)])),
                            "mean_p_wrong": sum(x for x, o in zip(p, ok) if o == 0) / max(1, ok.count(0))}
                print(key, json.dumps(out[key]), flush=True)
            del j
            import torch
            torch.cuda.empty_cache()
    os.makedirs(a.out, exist_ok=True)
    json.dump(out, open(os.path.join(a.out, "judge_calibration.json"), "w"), indent=2)


def cluster_ci(values, clusters, reps=2000, seed=0):
    """95% interval for a mean, resampling prompts (clusters) rather than single answers."""
    rng = random.Random(seed)
    groups = {}
    for v, c in zip(values, clusters):
        groups.setdefault(c, []).append(v)
    keys = list(groups)
    means = []
    for _ in range(reps):
        pick = [groups[rng.choice(keys)] for _ in keys]
        flat = [v for g in pick for v in g]
        means.append(sum(flat) / len(flat))
    means.sort()
    return [means[int(0.025 * reps)], means[int(0.975 * reps)]]


def cmd_reach(a):
    """Base reachability, the primary selection-vs-discovery measure. For each run's late hack
    answers, sample K answers from the starting model on the same problems and ask whether any of
    them is paid-but-wrong under the same checker. Selection: the base model reaches the paid-but-
    wrong region on those problems. Discovery: it does not. All problems are sampled in one pass."""
    runs = {}
    for run in sorted(os.listdir(a.runs)):
        d = os.path.join(a.runs, run)
        if not os.path.exists(os.path.join(d, "samples.jsonl")):
            continue
        cfg = json.load(open(os.path.join(d, "config.json")))
        cond, task = cfg["cond"], cfg.get("task", "math")
        if task != a.task:
            continue
        S = [json.loads(l) for l in open(os.path.join(d, "samples.jsonl"))]
        late = [x for x in S if x["step"] >= a.late_from]
        # the referee is re-applied here, so a fix to the gold label (amendment 9) reaches this measure
        ok = referee(a.task, [x["reply"] for x in late], [x["gold"] for x in late])
        hacks = [x for x, y in zip(late, ok) if x["r_arm"] > 0.5 and y == 0]
        runs[run] = (cond, late, hacks, sorted({(x["question"], x["gold"]) for x in hacks}))
    allp = sorted({p for _, _, _, ps in runs.values() for p in ps})
    replies = dict(zip(allp, sample(a.model, [q for q, _ in allp], a.k, a.max_tokens, a.seed, a.vllm,
                                    SUFFIXES[a.task]))) if allp else {}
    report = {}
    for run, (cond, late, hacks, probs) in runs.items():
        entry = {"cond": cond, "n_late": len(late), "n_hack": len(hacks), "n_hack_problems": len(probs),
                 "k": a.k, "reach": None, "base_pbw_rate_on_hack_problems": None}
        if probs:
            score = scorer(cond, a.judge_model)
            hit, rate = [], []
            for q, g in probs:
                ts = replies[(q, g)]
                r = score([q] * len(ts), [g] * len(ts), ts)
                pbw = [x > 0.5 and y == 0 for x, y in zip(r, referee(a.task, ts, [g] * len(ts)))]
                hit.append(any(pbw))
                rate.append(sum(pbw) / len(ts))
            entry["reach"] = sum(hit) / len(hit)
            entry["base_pbw_rate_on_hack_problems"] = sum(rate) / len(rate)
            entry["problems"] = [{"gold": g, "question": q[:200], "hit": h, "base_pbw_rate": r}
                                 for (q, g), h, r in zip(probs, hit, rate)]
        report[run] = entry
        print(run, json.dumps(entry), flush=True)
    json.dump(report, open(os.path.join(a.runs, f"reach_report_{a.task}.json"), "w"), indent=2)


def token_logprob(model_id, pairs, batch=8):
    """Mean per-token log-probability of each reply given its problem, under `model_id`."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).cuda().eval()
    out = []
    with torch.no_grad():
        for k in range(0, len(pairs), batch):
            chunk = pairs[k:k + batch]
            ids, starts = [], []
            for q, r in chunk:
                p = tok(chat(tok, q), add_special_tokens=False)["input_ids"]
                full = tok(chat(tok, q) + r, add_special_tokens=False)["input_ids"][:2048]
                ids.append(full)
                starts.append(len(p))
            L = max(len(x) for x in ids)
            pad = tok.pad_token_id or tok.eos_token_id
            x = torch.tensor([s + [pad] * (L - len(s)) for s in ids], device="cuda")
            logits = model(input_ids=x).logits.float()
            lp = torch.log_softmax(logits[:, :-1], -1).gather(-1, x[:, 1:, None]).squeeze(-1)
            for i, (s, st) in enumerate(zip(ids, starts)):
                seg = lp[i, st - 1:len(s) - 1]
                out.append(seg.mean().item() if seg.numel() else float("nan"))
            del logits, lp
    return out


def cmd_novelty(a):
    base = [json.loads(l) for l in open(a.base)]
    base_lp = sorted(token_logprob(a.model, [(b["question"], b["reply"]) for b in base]))
    report = {}
    for run in sorted(os.listdir(a.runs)):
        path = os.path.join(a.runs, run, "samples.jsonl")
        if not os.path.exists(path):
            continue
        if json.load(open(os.path.join(a.runs, run, "config.json"))).get("task", "math") != "math":
            continue  # the base log-likelihoods are of maths answers
        S = [json.loads(l) for l in open(path)]
        late = [s for s in S if s["step"] >= a.late_from]
        if not late:
            continue
        lp = token_logprob(a.model, [(s["question"], s["reply"]) for s in late])
        pct = [bisect.bisect_left(base_lp, v) / len(base_lp) for v in lp]
        hack = [p for p, s in zip(pct, late) if s["r_arm"] > 0.5 and s["r_strict"] == 0]
        other = [p for p, s in zip(pct, late) if not (s["r_arm"] > 0.5 and s["r_strict"] == 0)]
        med = lambda v: sorted(v)[len(v) // 2] if v else None  # noqa: E731
        report[run] = {"n_late": len(late), "n_hack": len(hack), "novelty_hack_median_pct": med(hack),
                       "novelty_other_median_pct": med(other)}
        with open(os.path.join(a.runs, run, "novelty.jsonl"), "w") as f:
            for p, s in zip(pct, late):
                f.write(json.dumps({"step": s["step"], "pct": p, "hack": s["r_arm"] > 0.5 and s["r_strict"] == 0}) + "\n")
        print(run, json.dumps(report[run]), flush=True)
    json.dump(report, open(os.path.join(a.runs, "novelty_report.json"), "w"), indent=2)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("audit")
    p.add_argument("--task", default="math", choices=["math", "mbpp"])
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--judge-model", default="Qwen/Qwen3-4B")
    p.add_argument("--judge-version", default="v1", choices=["v1", "v2"])
    p.add_argument("--conds", nargs="+", default=ck.CONDITIONS)
    p.add_argument("--prompts", type=int, default=64)
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=1536)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--vllm", action="store_true")
    p.add_argument("--tag", default="base")
    p.add_argument("--out", default="audits")
    q = sub.add_parser("novelty")
    q.add_argument("--model", default="Qwen/Qwen3-0.6B")
    q.add_argument("--runs", default="runs")
    q.add_argument("--base", default="audits/base_samples.jsonl")
    q.add_argument("--late-from", type=int, default=75)
    r = sub.add_parser("reach")
    r.add_argument("--task", default="math", choices=["math", "mbpp"])
    r.add_argument("--model", default="Qwen/Qwen3-0.6B")
    r.add_argument("--judge-model", default="Qwen/Qwen3-4B")
    r.add_argument("--runs", default="runs")
    r.add_argument("--k", type=int, default=64)
    r.add_argument("--max-tokens", type=int, default=1536)
    r.add_argument("--late-from", type=int, default=75)
    r.add_argument("--seed", type=int, default=99)
    r.add_argument("--vllm", action="store_true")
    c = sub.add_parser("calibrate")
    c.add_argument("--base", required=True)
    c.add_argument("--judge-models", nargs="+", default=["Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B"])
    c.add_argument("--versions", nargs="+", default=["v1", "v2"])
    c.add_argument("--out", default="calib")
    a = ap.parse_args()
    {"audit": cmd_audit, "novelty": cmd_novelty, "reach": cmd_reach, "calibrate": cmd_calibrate}[a.cmd](a)


if __name__ == "__main__":
    main()
