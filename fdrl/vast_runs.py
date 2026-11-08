"""Run a plan file of training commands on rented Vast.ai RTX 4090s, watch them, bring results back,
and destroy every machine when it is done. Amendment 9.

Needs: `pip install vastai`, ~/.vast_api_key, ~/.wandb_api_key, an ssh key registered with Vast
(`python fdrl/vast_runs.py setup` does that once).

  python fdrl/vast_runs.py setup
  python fdrl/vast_runs.py offers                       # cheapest machines that pass the filter
  python fdrl/vast_runs.py launch plans/v2_timing.txt --gpus 1 --name timing
  caffeinate -i python fdrl/vast_runs.py watch --name timing --budget 2
  python fdrl/vast_runs.py launch plans/v2_runs.txt --gpus 9 --name v2
  caffeinate -i python fdrl/vast_runs.py watch --name v2 --budget 18
  python fdrl/vast_runs.py destroy-all --name v2        # emergency stop for one launch

Money and data guards:
  - only verified RTX 4090 hosts at or under MAX_DPH, not in mainland China;
  - state is saved after every rental, so no machine is ever untracked;
  - a machine is destroyed only after its run folders are copied back AND each finished run has its
    DONE, log.jsonl and samples.jsonl locally; if copy-back fails it is retried next round;
  - samples and logs are copied back every round, so a dead machine loses at most one round;
  - a machine past its wall-clock limit (2x its estimate + 1 h) or with no new log line for 30 minutes
    is reported STUCK; past the budget, everything is copied (best effort) and destroyed, one machine at
    a time so one dead host cannot stop the rest from being destroyed.
  - keys live outside the repo (~/.vast_api_key, ~/.wandb_api_key, state in ~/.fdrl_vast/).
A run lost with a dead machine is not resumed elsewhere (its checkpoints were on that machine); it is
listed as LOST and relaunched from scratch with `launch` on a plan file of the lost commands.
"""

import argparse
import json
import os
import subprocess
import sys
import tarfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATE = os.path.expanduser("~/.fdrl_vast")  # outside the git repo: holds the W&B key copy and state
IMAGE = "python:3.12-bookworm"  # full Debian image: has gcc for Triton kernels
MAX_DPH = 0.45
QUERY = (f"gpu_name=RTX_4090 num_gpus=1 verified=true rentable=true reliability>0.99 "
         f"dph_total<={MAX_DPH} disk_space>=80 cpu_ram>=32 cuda_vers>=12.8 inet_down>=200")
REQ_LOCK = os.path.join(ROOT, "requirements-lock.txt")  # exact versions, from the timing machine's pip freeze

ONSTART = r"""#!/bin/bash
mkdir -p /workspace && cd /workspace
while [ ! -f /workspace/fdr/assigned.txt ]; do sleep 10; done
cd /workspace/fdr
echo "start $(date -u)" > /workspace/STATUS
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > /workspace/gpu.txt
REQ=requirements.txt; [ -f requirements-lock.txt ] && REQ=requirements-lock.txt
pip install -q -r $REQ > /workspace/pip.log 2>&1 || { echo "PIPFAILED $(date -u)" >> /workspace/STATUS; exit 1; }
pip freeze > /workspace/pip_freeze.txt
export WANDB_PROJECT=format-drift-rl-v2 WANDB_API_KEY=$(cat /workspace/fdr/.wandb_key)
i=0
while IFS= read -r cmd <&3; do
  [ -z "$cmd" ] && continue; i=$((i+1))
  echo "run $i: $cmd" >> /workspace/STATUS
  ( nvidia-smi --query-gpu=memory.used --format=csv,noheader -l 30 > /workspace/gpumem_$i.txt ) & MEMPID=$!
  for attempt in 1 2 3; do eval "$cmd" < /dev/null >> /workspace/train_$i.log 2>&1 && break; echo "retry $attempt of run $i" >> /workspace/STATUS; sleep 30; done
  kill $MEMPID 2>/dev/null
  for d in /workspace/runs/* /workspace/timing/*; do [ -d "$d" ] && cp /workspace/pip_freeze.txt /workspace/gpu.txt /workspace/fdr/CODE_COMMIT "$d/" 2>/dev/null; done
done 3< assigned.txt
echo "ALLDONE $(date -u)" >> /workspace/STATUS
"""


def vast(*args, raw=True):
    cmd = ["vastai", *args] + (["--raw"] if raw else [])
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if out.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)}: {out.stderr.strip() or out.stdout.strip()}")
    try:
        return json.loads(out.stdout) if raw else out.stdout
    except json.JSONDecodeError:
        return out.stdout


def read_plan(path):
    jobs = []
    for line in open(path):
        line = line.strip()
        if line and not line.startswith("#"):
            hours, cmd = line.split(" ", 1)
            jobs.append((float(hours), cmd))
    return jobs


def assign(jobs, n):
    """Longest-first onto the least-loaded machine, so all machines finish at about the same time."""
    bins = [[0.0, []] for _ in range(n)]
    for h, cmd in sorted(jobs, key=lambda j: -j[0]):
        b = min(bins, key=lambda x: x[0])
        b[0] += h
        b[1].append(cmd)
    return bins


def run_dir_of(cmd):
    """The run directory a train_grpo command writes: <cond><tag>_s<seed> under --out."""
    t = cmd.split()
    get = lambda k, d="": t[t.index(k) + 1] if k in t else d  # noqa: E731
    return os.path.basename(get("--out")), f"{get('--cond')}{get('--tag')}_s{get('--seed', '0')}"


def code_bundle(name):
    """Code version = HEAD plus a patch of any uncommitted changes, both shipped and recorded."""
    git = lambda *a: subprocess.run(["git", "-C", ROOT, *a], capture_output=True, text=True).stdout  # noqa: E731
    commit = git("rev-parse", "HEAD").strip()
    patch = git("diff", "HEAD", "--", "fdrl", "requirements.txt")
    os.makedirs(STATE, exist_ok=True)
    open(os.path.join(STATE, "CODE_COMMIT"), "w").write(
        f"{commit}{' + uncommitted changes in CODE_PATCH.diff' if patch else ''}\n")
    open(os.path.join(STATE, "CODE_PATCH.diff"), "w").write(patch)
    path = os.path.join(STATE, f"{name}_code.tar.gz")
    with tarfile.open(path, "w:gz") as t:
        t.add(os.path.join(ROOT, "fdrl"), "fdr/fdrl", filter=lambda x: None if "__pycache__" in x.name else x)
        t.add(os.path.join(ROOT, "requirements.txt"), "fdr/requirements.txt")
        if os.path.exists(REQ_LOCK):
            t.add(REQ_LOCK, "fdr/requirements-lock.txt")
        t.add(os.path.join(STATE, "CODE_COMMIT"), "fdr/CODE_COMMIT")
        t.add(os.path.join(STATE, "CODE_PATCH.diff"), "fdr/CODE_PATCH.diff")
    return path


def ssh_parts(iid):
    url = vast("ssh-url", str(iid), raw=False).strip()  # ssh://root@host:port
    host, port = url.split("@", 1)[1].rsplit(":", 1)
    return host, port


SSH_OPTS = ["-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=20", "-o", "BatchMode=yes"]


def run_ssh(iid, remote):
    host, port = ssh_parts(iid)
    return subprocess.run(["ssh", *SSH_OPTS, "-p", port, f"root@{host}", remote],
                          capture_output=True, text=True, timeout=120)


def scp_to(iid, local, remote):
    host, port = ssh_parts(iid)
    subprocess.run(["scp", *SSH_OPTS, "-P", port, local, f"root@{host}:{remote}"], check=True, timeout=300)


def save(state, name):
    path = os.path.join(STATE, f"{name}.json")
    json.dump(state, open(path + ".tmp", "w"), indent=2)
    os.replace(path + ".tmp", path)


def cmd_setup(a):
    key = os.path.expanduser("~/.ssh/id_ed25519")
    if not os.path.exists(key):
        subprocess.run(["ssh-keygen", "-t", "ed25519", "-N", "", "-f", key], check=True)
    vast("set", "api-key", open(os.path.expanduser("~/.vast_api_key")).read().strip(), raw=False)
    vast("create", "ssh-key", open(key + ".pub").read().strip(), raw=False)
    print("vast api key and ssh key set")


def offers(n=20):
    found = vast("search", "offers", QUERY, "-o", "dph_total")
    # hosts in mainland China often cannot reach the Hugging Face hub, where the models come from
    return [o for o in found if not str(o.get("geolocation", "")).endswith(", CN")][:n]


def cmd_offers(a):
    for o in offers():
        print(f"{o['id']:>9}  ${o['dph_total']:.3f}/h  rel {o['reliability2']:.3f}  {o.get('geolocation','?'):<18} "
              f"disk {o['disk_space']:.0f}GB  ram {o['cpu_ram']/1024:.0f}GB  cuda {o['cuda_max_good']}")


def cmd_launch(a):
    jobs = read_plan(a.plan)
    bins = [b for b in assign(jobs, a.gpus) if b[1]]
    code_bundle(a.name)
    os.makedirs(STATE, exist_ok=True)
    wk = os.path.join(STATE, "wandb_key")
    open(wk, "w").write(open(os.path.expanduser("~/.wandb_api_key")).read().strip())
    os.chmod(wk, 0o600)
    avail = offers(len(bins) * 3)
    path = os.path.join(STATE, f"{a.name}.json")
    state = json.load(open(path)) if os.path.exists(path) else {"name": a.name, "machines": []}
    state["plan"] = a.plan
    for k, (hours, cmds) in enumerate(bins):
        rented = False
        while avail and not rented:
            o = avail.pop(0)
            try:
                r = vast("create", "instance", str(o["id"]), "--image", IMAGE, "--disk", "80", "--ssh", "--direct",
                         "--label", f"{a.name}-{len(state['machines'])}", "--onstart-cmd", ONSTART)
            except RuntimeError as e:
                print(f"offer {o['id']} failed: {str(e)[:100]}")
                continue
            if isinstance(r, dict) and r.get("success"):
                state["machines"].append({"id": r["new_contract"], "dph": o["dph_total"], "est_hours": hours,
                                          "cmds": cmds, "started": time.time(), "shipped": False,
                                          "collected": False, "last_change": time.time(), "last_seen": ""})
                save(state, a.name)  # saved after every rental: no machine is ever untracked
                print(f"machine {k}: instance {r['new_contract']} at ${o['dph_total']:.3f}/h, ~{hours:.1f} h, {len(cmds)} runs")
                rented = True
        if not rented:
            print(f"machine {k}: could not rent; its runs: put them in a plan file and launch again")


def ship(m, name):
    bundle = os.path.join(STATE, f"{name}_code.tar.gz")
    assigned = os.path.join(STATE, f"{name}_{m['id']}_assigned.txt")
    open(assigned, "w").write("\n".join(m["cmds"]) + "\n")
    run_ssh(m["id"], "mkdir -p /workspace")
    scp_to(m["id"], bundle, "/workspace/code.tar.gz")
    scp_to(m["id"], os.path.join(STATE, "wandb_key"), "/workspace/wandb_key")
    scp_to(m["id"], assigned, "/workspace/assigned.tmp")
    r = run_ssh(m["id"], "cd /workspace && tar xzf code.tar.gz && mv wandb_key fdr/.wandb_key && "
                         "chmod 600 fdr/.wandb_key && mv assigned.tmp fdr/assigned.txt")  # atomic start signal
    if r.returncode != 0:
        raise RuntimeError(f"ship failed: {r.stderr[:200]}")


def collect(m, name, final):
    """Copy run folders (no checkpoints) and machine logs back. Returns True only if, for a final copy,
    every run this machine finished is complete locally."""
    host, port = ssh_parts(m["id"])
    dest = os.path.join(ROOT, "results", "timing_v2" if name == "timing" else "runs_v2")
    os.makedirs(os.path.join(dest, f"_machine_{m['id']}"), exist_ok=True)
    e = f"ssh {' '.join(SSH_OPTS)} -p {port}"
    ok = True
    for remote in ("/workspace/runs/", "/workspace/timing/"):
        r = subprocess.run(["rsync", "-a", "--exclude", "checkpoint-*", "-e", e, f"root@{host}:{remote}", dest + "/"],
                           capture_output=True, text=True, timeout=1800)
        ok &= r.returncode in (0, 23)  # 23: that folder does not exist on this machine
    subprocess.run(["rsync", "-a", "-e", e, f"root@{host}:/workspace/STATUS", f"root@{host}:/workspace/train_*.log",
                    f"root@{host}:/workspace/gpumem_*.txt", f"root@{host}:/workspace/pip_freeze.txt",
                    os.path.join(dest, f"_machine_{m['id']}") + "/"], capture_output=True, timeout=600)
    if not final:
        return ok
    for cmd in m["cmds"]:
        _, run = run_dir_of(cmd)
        d = os.path.join(dest, run)
        need = ["DONE", "log.jsonl", "samples.jsonl", "config.json"]
        if not all(os.path.exists(os.path.join(d, f)) for f in need):
            print(f"  {run}: not complete locally ({[f for f in need if not os.path.exists(os.path.join(d, f))]})")
            ok = False
    return ok


def destroy(m):
    try:
        vast("destroy", "instance", str(m["id"]), raw=False)
        return True
    except Exception as e:  # already gone counts as destroyed
        return "not found" in str(e).lower() or "no instance" in str(e).lower()


def cmd_watch(a):
    path = os.path.join(STATE, f"{a.name}.json")
    while True:
        state = json.load(open(path))
        try:
            live = {i["id"]: i for i in vast("show", "instances")}
        except Exception as e:
            print("vast unreachable:", str(e)[:80], "- retrying next round")
            time.sleep(a.every)
            continue
        spent = 0.0
        print(time.strftime("\n%H:%M:%S"), f"{'machine':>9} {'state':<9} {'$/h':>6} {'hours':>6} {'$':>6}  status")
        for m in state["machines"]:
            hrs = ((m.get("ended") or time.time()) - m["started"]) / 3600
            spent += m["dph"] * hrs
            if m["collected"]:
                print(f"{m['id']:>9} {'done':<9} {m['dph']:>6.3f} {hrs:>6.2f} {m['dph']*hrs:>6.2f}  collected + destroyed")
                continue
            inst = live.get(m["id"])
            st = (inst or {}).get("actual_status", "gone")
            line = ""
            if st == "gone" and m["shipped"]:
                m["collected"], m["ended"] = True, time.time()
                line = "LOST: machine disappeared; relaunch its runs: " + "; ".join(run_dir_of(c)[1] for c in m["cmds"])
            elif st == "running":
                try:
                    if not m["shipped"]:
                        ship(m, a.name)
                        m["shipped"] = True
                    out = run_ssh(m["id"], "tail -n 1 /workspace/STATUS 2>/dev/null; "
                                           "tail -c 300 /workspace/train_*.log 2>/dev/null | tr '\\r' '\\n' | grep -o '[0-9]*/[0-9]* \\[' | tail -1")
                    seen = out.stdout.strip()
                    if seen != m["last_seen"]:
                        m["last_seen"], m["last_change"] = seen, time.time()
                    line = " | ".join(seen.splitlines())[-100:]
                    done = "ALLDONE" in seen or "PIPFAILED" in seen
                    if collect(m, a.name, final=done) and done:
                        if destroy(m):
                            m["collected"], m["ended"] = True, time.time()
                            line = "collected + destroyed" + (" (PIP FAILED: check pip.log)" if "PIPFAILED" in seen else "")
                    elif done:
                        line += "  copy-back incomplete, retrying"
                    if time.time() - m["last_change"] > 1800:
                        line = "STUCK (no progress 30 min) | " + line
                    if hrs > 2 * m["est_hours"] + 1:
                        line = "OVER TIME LIMIT | " + line
                except Exception as e:  # still booting, or ssh not ready yet
                    line = f"waiting ({str(e)[:60]})"
            print(f"{m['id']:>9} {st:<9} {m['dph']:>6.3f} {hrs:>6.2f} {m['dph']*hrs:>6.2f}  {line}")
        print(f"total so far ~${spent:.2f} of ${a.budget:.2f} budget")
        save(state, a.name)
        if spent > a.budget:
            print("BUDGET REACHED: copying what exists and destroying every machine")
            for m in state["machines"]:
                if m["collected"]:
                    continue
                try:
                    collect(m, a.name, final=False)
                except Exception as e:
                    print(f"  copy from {m['id']} failed: {str(e)[:60]}")
                destroy(m)  # always, even if copy failed: the budget wins
                m["collected"], m["ended"] = True, time.time()
                save(state, a.name)
            return
        if all(m["collected"] for m in state["machines"]):
            print("all machines collected and destroyed")
            return
        time.sleep(a.every)


def cmd_destroy_all(a):
    for i in vast("show", "instances"):
        if a.name and not str(i.get("label") or "").startswith(a.name + "-"):
            continue
        print("destroyed" if destroy(i) else "FAILED to destroy", i["id"], i.get("label"))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("setup")
    sub.add_parser("offers")
    p = sub.add_parser("launch")
    p.add_argument("plan")
    p.add_argument("--gpus", type=int, default=1)
    p.add_argument("--name", required=True)
    w = sub.add_parser("watch")
    w.add_argument("--name", required=True)
    w.add_argument("--budget", type=float, default=14.0)
    w.add_argument("--every", type=int, default=300)
    d = sub.add_parser("destroy-all")
    d.add_argument("--name", default="", help="only machines whose label starts with this launch name")
    a = ap.parse_args()
    {"setup": cmd_setup, "offers": cmd_offers, "launch": cmd_launch, "watch": cmd_watch,
     "destroy-all": cmd_destroy_all}[a.cmd](a)


if __name__ == "__main__":
    main()
