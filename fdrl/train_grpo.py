"""Dr. GRPO training against one checker condition, with TRL's GRPOTrainer.

The condition's checker pays the reward. The referee (Math-Verify on the stated answer for maths,
MBPP+ for code) scores every completion as well, with weight 0, so each step logs how much of the
reward went to wrong answers (`fp`). At fixed steps all completions are written to samples.jsonl
with both scores, for the reachability and trace analysis.

A run can be stopped and continued: checkpoints (adapter, optimiser, scheduler and RNG state) are
written every --save-every steps, and rerunning the same command with --resume continues from the
latest one. Log lines and samples after that checkpoint are dropped first, so no step is counted
twice. A run that already finished (DONE) is skipped.

Usage:
  python fdrl/train_grpo.py --cond substring_all --seed 0 --out runs --resume
  python fdrl/train_grpo.py --task mbpp --cond mbpp_base --seed 0 --out runs --resume
"""

import argparse
import glob
import json
import os
import random
import re
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import checkers as ck  # noqa: E402
import code_env as ce  # noqa: E402

SUFFIX = "\nPut your final answer in \\boxed{}."
SAVE_STEPS = {1, 10, 25} | set(range(50, 1001, 5))  # every 5 steps from 50, for the late-answer measures

# Amendment 9: checkers that pay nothing for an answer cut off at --max-tokens (they need token counts)
CUT_CONDITIONS = {
    "gold_label_nocut": ck.gold_label,        # format-free honest: any stated answer format, cut-off pays 0
    "substring_all_nocut": ck.substring_all,  # the loophole with its cut-off channel closed
}


def integer_gold(ans, min_abs=10):
    a = (ans or "").strip().replace(",", "")
    return re.fullmatch(r"-?\d+", a) is not None and abs(int(a)) >= min_abs


def load_train(max_level=3):
    """MATH training problems (levels 1-3) whose answer is an integer of at least two digits,
    the pilot's split, so the gold string is unambiguous for every checker."""
    from datasets import load_dataset

    ds = load_dataset("PrimeIntellect/Hendrycks-Math", split="train")
    rows = [{"prompt": [{"role": "user", "content": r["question"] + SUFFIX}],
             "question": r["question"], "gold": r["answer"].strip().replace(",", "")}
            for r in ds
            if integer_gold(r["answer"]) and float(r["difficulty"] or 5) <= max_level]
    return rows


def text_of(completion):
    return completion[0]["content"] if isinstance(completion, list) else completion


def ckpt_step(path):
    return int(path.rsplit("-", 1)[1])


def trim_jsonl(path, keep):
    """Keep only the lines of a jsonl file whose step passes `keep`."""
    if os.path.exists(path):
        rows = [l for l in open(path) if keep(json.loads(l).get("step", -1))]
        open(path, "w").writelines(rows)


def prepare_dir(out_dir, resume):
    """Return the checkpoint to resume from (or None) and make the directory consistent with it."""
    if os.path.exists(os.path.join(out_dir, "DONE")):
        print(f"{out_dir} already finished; skipping")
        sys.exit(0)
    log = os.path.join(out_dir, "log.jsonl")
    if not os.path.exists(log):
        os.makedirs(out_dir, exist_ok=True)
        return None
    if not resume:
        sys.exit(f"{out_dir} already has a log; pass --resume to continue it, or use a fresh --tag")
    ckpts = sorted(glob.glob(os.path.join(out_dir, "checkpoint-*")), key=ckpt_step)
    if not ckpts:  # crashed before the first checkpoint: keep the attempt, start again
        k = len(glob.glob(os.path.join(out_dir, "attempt*")))
        os.makedirs(os.path.join(out_dir, f"attempt{k}"))
        for f in ("log.jsonl", "samples.jsonl"):
            if os.path.exists(os.path.join(out_dir, f)):
                shutil.move(os.path.join(out_dir, f), os.path.join(out_dir, f"attempt{k}", f))
        return None
    last = ckpts[-1]
    n = ckpt_step(last)
    # log line k and samples labelled k both come from the batch behind update k, so both keep k <= n
    trim_jsonl(log, lambda s: s <= n)
    trim_jsonl(os.path.join(out_dir, "samples.jsonl"), lambda s: s <= n)
    print(f"resuming {out_dir} from {os.path.basename(last)}", flush=True)
    return last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="math", choices=["math", "mbpp"])
    ap.add_argument("--cond", choices=ck.CONDITIONS + list(ce.CONDITIONS) + list(CUT_CONDITIONS), required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--judge-model", default="Qwen/Qwen3-4B")
    ap.add_argument("--judge-version", default="v1", choices=["v1", "v2"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--prompts", type=int, default=8, help="questions per step")
    ap.add_argument("--group", type=int, default=8, help="answers per question")
    ap.add_argument("--micro", type=int, default=4, help="completions per forward/backward pass (speed and memory only)")
    ap.add_argument("--no-grad-ckpt", action="store_true", help="turn off gradient checkpointing (faster, more memory)")
    ap.add_argument("--max-tokens", type=int, default=1536)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--vllm", action="store_true")
    ap.add_argument("--vllm-mem", type=float, default=0.3)
    ap.add_argument("--save-every", type=int, default=20)
    ap.add_argument("--scale-rewards", default="none", choices=["none", "group", "batch"])
    ap.add_argument("--tag", default="", help="suffix for the run directory, e.g. an ablation name")
    ap.add_argument("--resume", action="store_true", help="continue from the latest checkpoint")
    ap.add_argument("--report", default="none", choices=["none", "wandb"],
                    help="wandb: live charts and answer tables (needs WANDB_API_KEY in the environment)")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--min-tokens", type=int, default=0, help="timing tests only: force answers to this length")
    a = ap.parse_args()
    if (a.task == "mbpp") != (a.cond in ce.CONDITIONS):
        sys.exit(f"--cond {a.cond} does not belong to --task {a.task}")

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import TrainerCallback
    from trl import GRPOConfig, GRPOTrainer

    run = f"{a.cond}{a.tag}_s{a.seed}"
    out_dir = os.path.join(a.out, run)
    resume_from = prepare_dir(out_dir, a.resume)
    if a.report == "wandb":
        # one W&B run per run directory, so a resumed run continues the same charts
        os.environ.setdefault("WANDB_PROJECT", "selection-discovery")
        os.environ["WANDB_RUN_ID"] = re.sub(r"[^A-Za-z0-9_-]", "-", f"{os.path.basename(a.out)}-{run}")[-64:]
        os.environ["WANDB_RESUME"] = "allow"
        os.environ["WANDB_RUN_GROUP"] = a.cond
        os.environ.setdefault("WANDB_NAME", run)
    random.seed(a.seed)
    if a.task == "math":
        rows = load_train()
        random.shuffle(rows)
        train = Dataset.from_list(rows[: a.steps * a.prompts])
    else:
        rows = ce.load_train(a.steps * a.prompts)
        random.shuffle(rows)
        train = Dataset.from_list(rows)

    if a.cond in ck.JUDGE_CONDITIONS:
        ref, prob = ck.JUDGE_CONDITIONS[a.cond]
        judge = ck.Judge(a.judge_model, reference=ref, prob=prob, version=a.judge_version)
        arm = lambda texts, golds, qs: judge(qs, golds, texts)  # noqa: E731
    elif a.task == "mbpp":
        fn = ce.CONDITIONS[a.cond]
        arm = lambda texts, golds, qs: ce.score_batch(fn, texts, golds)  # noqa: E731
    elif a.cond in CUT_CONDITIONS:
        fn = CUT_CONDITIONS[a.cond]

        def arm(texts, golds, qs, cut=None):
            if cut is None:
                raise RuntimeError(f"{a.cond} needs completion lengths; TRL did not pass completion_ids")
            return [0.0 if c else fn(t, g) for t, g, c in zip(texts, golds, cut)]
    else:
        fn = ck.POINTWISE[a.cond]
        arm = lambda texts, golds, qs, cut=None: [fn(t, g) for t, g in zip(texts, golds)]  # noqa: E731
    if a.task == "mbpp":
        referee = lambda texts, golds: ce.score_batch(ce.label, texts, golds)  # noqa: E731
    else:
        referee = lambda texts, golds: [ck.gold_label(t, g) for t, g in zip(texts, golds)]  # noqa: E731

    samples = open(os.path.join(out_dir, "samples.jsonl"), "a")

    last = {}  # the arm's scores for the current batch, so the judge runs once per batch

    def cut_flags(kw):
        ids = kw.get("completion_ids")
        return None if ids is None else [len(c) >= a.max_tokens for c in ids]

    def reward_arm(prompts, completions, gold, question, **kw):
        texts = [text_of(c) for c in completions]
        if a.cond in CUT_CONDITIONS:
            last["texts"], last["r"] = texts, arm(texts, gold, question, cut=cut_flags(kw))
        else:
            last["texts"], last["r"] = texts, arm(texts, gold, question)
        return last["r"]

    def reward_strict(prompts, completions, gold, question, trainer_state=None, log_metric=None, **kw):
        """Weight 0: logs the true score and the defect's share of the reward, and saves traces."""
        texts = [text_of(c) for c in completions]
        s = referee(texts, gold)
        if last.get("texts") == texts:
            r = last["r"]
        else:
            r = arm(texts, gold, question, cut=cut_flags(kw)) if a.cond in CUT_CONDITIONS else arm(texts, gold, question)
        if log_metric is None:
            raise RuntimeError("TRL did not pass log_metric; fp would not be logged")
        n = len(texts)
        log_metric("fp", sum(x > 0.5 and y == 0 for x, y in zip(r, s)) / n)
        # reward-weighted fp: reward paid to answers the gold rejects, so partial credit and
        # sub-threshold judge probability are counted too
        log_metric("fp_mass", sum(x for x, y in zip(r, s) if y == 0) / n)
        log_metric("fn", sum(x <= 0.5 and y == 1 for x, y in zip(r, s)) / n)
        log_metric("chars", sum(len(t) for t in texts) / n)
        cut = [len(c) >= a.max_tokens for c in kw.get("completion_ids", [[]] * n)]
        log_metric("truncated", sum(cut) / n)
        log_metric("fp_truncated", sum(x > 0.5 and y == 0 and c for x, y, c in zip(r, s, cut)) / n)
        extra = {}
        if a.task == "math":
            sb = [ck.strict(t, g) for t, g in zip(texts, gold)]
            extra = {"r_boxed": sb}
            # the box-required label of amendment 1, and right answers it fails only for a missing box
            log_metric("fp_boxed", sum(x > 0.5 and y == 0 for x, y in zip(r, sb)) / n)
            log_metric("acc_boxed", sum(sb) / n)
            log_metric("unboxed_correct", sum(y == 1 and z == 0 for y, z in zip(s, sb)) / n)
            log_metric("strict_mv_any", sum(ck.strict_mv_any(t, g) for t, g in zip(texts, gold)) / n)
            log_metric("strict_errors", float(ck.STRICT_ERRORS[0]))
            log_metric("numbers", sum(len(ck.numbers(t)) for t in texts) / n)
            log_metric("boxed", sum(ck.last_boxed(t) is not None for t in texts) / n)
        else:
            log_metric("has_code", sum(ce.extract_code(t) is not None for t in texts) / n)
        # the batch generated at global step k produces update k+1 and is logged as step k+1, so the
        # samples carry the same label as the log line they belong to
        step = trainer_state.global_step + 1 if trainer_state is not None else -1
        if step in SAVE_STEPS:
            ids = kw.get("completion_ids") or [None] * n
            for i, (q, g, t, x, y) in enumerate(zip(question, gold, texts, r, s)):
                row = {"step": step, "question": q, "gold": g, "reply": t, "r_arm": x, "r_strict": y}
                if ids[i] is not None:  # amendment 9: token length and whether the answer hit the limit
                    row["n_tokens"] = len(ids[i])
                    row["hit_limit"] = len(ids[i]) >= a.max_tokens
                row.update({k: v[i] for k, v in extra.items()})
                samples.write(json.dumps(row) + "\n")
            samples.flush()
            if a.report == "wandb":
                import wandb

                if wandb.run is not None:  # the answers at this step, paid-but-wrong ones first
                    rows = sorted(zip(question, gold, texts, r, s), key=lambda z: -(z[3] > 0.5 and z[4] == 0))
                    table = wandb.Table(columns=["step", "paid_but_wrong", "arm", "referee", "gold", "reply", "question"],
                                        data=[[step, bool(x > 0.5 and y == 0), x, y, g, t[-3000:], q[:1000]]
                                              for q, g, t, x, y in rows])
                    wandb.log({"answers": table}, commit=False)
        return s

    class JsonlLog(TrainerCallback):
        def __init__(self):
            self.f = open(os.path.join(out_dir, "log.jsonl"), "a")

        def on_log(self, args, state, control, logs=None, **kw):
            self.f.write(json.dumps({"step": state.global_step, **(logs or {})}) + "\n")
            self.f.flush()

    bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    if not bf16:  # T4: no bf16 matmuls, so fp16 weights and TRL's fp32-upcast fallback path
        import trl.kernels.chunked_logprob as clp
        clp._MM_OUT_DTYPE = False
    cfg = GRPOConfig(
        model_init_kwargs={"dtype": torch.bfloat16 if bf16 else torch.float16},
        output_dir=out_dir,
        seed=a.seed,
        max_steps=a.steps,
        learning_rate=a.lr,
        lr_scheduler_type="constant",
        warmup_steps=0,
        per_device_train_batch_size=a.micro,
        gradient_accumulation_steps=a.prompts * a.group // a.micro,
        num_generations=a.group,
        max_completion_length=a.max_tokens,
        temperature=1.0,
        top_p=1.0,
        loss_type="dr_grpo",
        scale_rewards=a.scale_rewards,
        beta=0.0,
        num_iterations=1,
        reward_weights=[1.0, 0.0],
        chat_template_kwargs={"enable_thinking": False},
        use_vllm=a.vllm,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=a.vllm_mem,
        vllm_max_model_length=a.max_tokens + 1024,  # prompt + answer; the model's 40k default won't fit a T4
        bf16=bf16,
        fp16=not bf16 and torch.cuda.is_available(),
        gradient_checkpointing=not a.no_grad_ckpt,
        logging_steps=1,
        save_strategy="steps",
        save_steps=a.save_every,
        save_total_limit=2,  # full checkpoints (with optimiser state) so a run can resume
        report_to=a.report,
        run_name=run,
        log_completions=False,
        # amendment 9 timing test: --min-tokens 1536 forces full-length answers, the worst case late in a loophole run
        **({"generation_kwargs": {"min_tokens": a.min_tokens}} if a.min_tokens else {}),
    )
    peft = LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.0, task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"])
    commit_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "CODE_COMMIT")
    code_commit = open(commit_file).read().strip() if os.path.exists(commit_file) else "unknown"
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    json.dump({**vars(a), "stack": "trl-grpo", "trl_loss": "dr_grpo", "code_commit": code_commit, "gpu": gpu},
              open(os.path.join(out_dir, "config.json"), "w"), indent=2)

    class KeepAdapters(TrainerCallback):
        """Amendment 9: keep the LoRA adapter at fixed steps (checkpoints rotate away), for held-out scoring."""
        def on_step_end(self, args, state, control, model=None, **kw):
            if state.global_step in (10, 60) and model is not None:
                model.save_pretrained(os.path.join(out_dir, f"adapter_step{state.global_step}"))

    trainer = GRPOTrainer(model=a.model, reward_funcs=[reward_arm, reward_strict], args=cfg,
                          train_dataset=train, peft_config=peft, callbacks=[JsonlLog(), KeepAdapters()])
    trainer.train(resume_from_checkpoint=resume_from)
    trainer.save_model(os.path.join(out_dir, "final"))
    open(os.path.join(out_dir, "DONE"), "w").write("ok\n")


if __name__ == "__main__":
    main()
