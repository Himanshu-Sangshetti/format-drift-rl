"""Run sd/ commands on Modal, one GPU container per command, all in parallel.

Each line of the plan file is one command, run from the sd/ sources with outputs under /out
(the `sd-out` volume). Containers stop when their command ends, so nothing is left running.

The volume is committed every few minutes while a command runs, and a command that fails or whose
container is preempted is retried up to twice. Training commands carry --resume, so a retry (or
rerunning the whole plan) continues each run from its latest checkpoint and skips finished runs.

Usage:
  SD_GPU=L40S modal run sd/modal_app.py --plan plans/exp1_pilot.txt
  modal volume get sd-out runs ./results/sd_runs   # copy results back
"""

import os
import subprocess
import threading
import time

import modal

GPU = os.environ.get("SD_GPU", "L40S")
# the Modal secret named "wandb" (WANDB_API_KEY), for commands run with --report wandb; attached always,
# because a conditional dependency differs between the local and the container side
SECRETS = [modal.Secret.from_name("wandb")]
HERE = os.path.dirname(os.path.abspath(__file__))

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("trl[vllm]==1.17.1", "math-verify>=0.8.0", "peft", "datasets>=4.7.0", "wandb")
    .env({"HF_HOME": "/hf", "TOKENIZERS_PARALLELISM": "false",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .add_local_dir(HERE, "/root/fdrl", ignore=["plans", "__pycache__"])
)
app = modal.App("sd-runs", image=image)
out = modal.Volume.from_name("sd-out", create_if_missing=True)
hf = modal.Volume.from_name("sd-hf-cache", create_if_missing=True)


@app.function(gpu=GPU, cpu=4.0, timeout=12 * 3600, volumes={"/out": out, "/hf": hf}, secrets=SECRETS,
              retries=modal.Retries(max_retries=2, initial_delay=10.0))
def run(cmd: str) -> dict:
    t0 = time.time()
    stop = threading.Event()

    def commit_loop():
        while not stop.wait(300):
            out.commit()
    threading.Thread(target=commit_loop, daemon=True).start()
    log = "/out/logs/" + "".join(ch if ch.isalnum() else "_" for ch in cmd)[-120:] + ".log"
    os.makedirs("/out/logs", exist_ok=True)
    with open(log, "a") as f:
        code = subprocess.call(cmd, shell=True, cwd="/root", stdout=f, stderr=subprocess.STDOUT)
    stop.set()
    out.commit()
    hf.commit()
    if code != 0:
        raise RuntimeError(f"exit {code}: {cmd}")  # so Modal retries it
    return {"cmd": cmd, "exit": code, "sec": round(time.time() - t0), "gpu": GPU}


@app.local_entrypoint()
def main(plan: str):
    cmds = [l.strip() for l in open(plan) if l.strip() and not l.startswith("#")]
    print(f"{len(cmds)} commands on {GPU}")
    for r in run.map(cmds, order_outputs=False, return_exceptions=True):
        print(r)
