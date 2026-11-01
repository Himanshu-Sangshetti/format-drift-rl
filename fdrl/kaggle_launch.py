"""Push a Kaggle GPU kernel (T4 x2) that runs sd/ commands, one lane of commands per GPU.

Each lane is a ';'-separated list of commands run in order on its own GPU. The kernel ships the
sd/ sources inline, installs the pinned stack and leaves everything under /kaggle/working/out
for `kaggle kernels output`.

Usage:
  python fdrl/kaggle_sd.py --name sd-smoke \
    --lane "python fdrl/train_grpo.py --cond substring_all --steps 3 --out out/runs" \
    --lane "python fdrl/audit.py audit --prompts 8 --n 4 --out out/audits"
"""

import argparse
import base64
import json
import os
import subprocess
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
PIP = "'trl==1.17.1' 'math-verify>=0.8.0' peft 'accelerate>=1.11.0' 'datasets>=4.7.0'"

RUNNER = r'''
import base64, os, subprocess, sys, threading, time
def sh(c):
    print("+", c, flush=True)
    return subprocess.call(c, shell=True)
sh("pip uninstall -y -q torchao")  # preinstalled torchao breaks peft
sh(f"pip install -q {PIP}")
if VLLM:
    sh("pip install -q 'vllm>=0.22.0,<0.33.0'")
    sh("pip uninstall -y -q torchaudio")  # built for the old torch; vLLM needs torchvision, which it pins itself
os.makedirs("/kaggle/working/fdrl", exist_ok=True)
for name, blob in FILES.items():
    open(f"/kaggle/working/fdrl/{name}", "wb").write(base64.b64decode(blob))
os.chdir("/kaggle/working")
os.makedirs("out", exist_ok=True)
sh("python -c 'import torch,trl,transformers,peft;print(torch.__version__,trl.__version__,transformers.__version__,peft.__version__, torch.cuda.device_count())'")

def lane(gpu, cmds):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
           "TOKENIZERS_PARALLELISM": "false",
           "MASTER_PORT": str(29500 + 17 * gpu)}  # two lanes start vLLM at once; distinct ports avoid EADDRINUSE
    for i, c in enumerate(cmds):
        t0 = time.time()
        log = f"out/lane{gpu}_{i}.log"
        with open(log, "w") as f:
            code = subprocess.call(c, shell=True, stdout=f, stderr=subprocess.STDOUT, env=env)
        print(f"gpu{gpu} cmd{i} exit={code} {time.time() - t0:.0f}s :: {c}", flush=True)
        if code:
            print(open(log).read()[-4000:], flush=True)

ts = [threading.Thread(target=lane, args=(g, [c.strip() for c in L.split(";") if c.strip()]))
      for g, L in enumerate(LANES)]
for t in ts: t.start()
for t in ts: t.join()
for c in FINAL:
    sh(c + " > out/final.log 2>&1")
sh("find out -name 'optimizer.pt' -delete; find out -name 'rng_state*' -delete")
print("KAGGLE_DONE", flush=True)
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--user", default="hims72")
    ap.add_argument("--lane", action="append", required=True)
    ap.add_argument("--vllm", action="store_true")
    ap.add_argument("--final", action="append", default=[], help="commands run after both lanes finish")
    a = ap.parse_args()
    assert len(a.lane) <= 2, "Kaggle T4 x2 has two GPUs"
    files = {f: base64.b64encode(open(os.path.join(HERE, f), "rb").read()).decode()
             for f in ["checkers.py", "code_env.py", "train_grpo.py", "audit.py", "analyze.py"]}
    d = tempfile.mkdtemp()
    head = f"FILES = {files!r}\nLANES = {a.lane!r}\nPIP = {PIP!r}\nVLLM = {a.vllm!r}\nFINAL = {a.final!r}\n"
    open(os.path.join(d, "run.py"), "w").write(head + RUNNER)
    json.dump({"id": f"{a.user}/{a.name}", "title": a.name, "code_file": "run.py", "language": "python",
               "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_internet": True,
               "dataset_sources": [], "competition_sources": [], "kernel_sources": []},
              open(os.path.join(d, "kernel-metadata.json"), "w"))
    subprocess.run(["kaggle", "kernels", "push", "-p", d, "--accelerator", "NvidiaTeslaT4"], check=True)


if __name__ == "__main__":
    main()
