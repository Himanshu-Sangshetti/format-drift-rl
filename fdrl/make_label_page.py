"""Build a blind labelling page (one HTML file, works offline) for the answers the referees dispute.

Groups (hidden from the labeller, stored in a key file):
  format  : late answers of tail50_slope_s1 / tail50_density_s2 the checker paid, where verl's MATH grader
            says wrong and TRL's grader says right (box-format false alarm?)
  cutoff  : late paid answers that hit the 1,536-token limit, where TRL's grader (no cut rule) says right
  control : late paid answers that every referee calls right, and ones every referee calls wrong
Order is shuffled. The labeller writes the committed final answer BEFORE the correct answer is shown.
Labels are kept in the browser and downloaded as JSON.

  python fdrl/make_label_page.py --out ~/Downloads/label_batch1.html
"""

import argparse
import glob
import gzip
import html
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import checkers as ck  # noqa: E402

LATE = {105, 110, 115, 120}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="results/runs")
    ap.add_argument("--out", default=os.path.expanduser("~/Downloads/label_batch1.html"))
    ap.add_argument("--key", default="results/label_batch1_key.json")
    ap.add_argument("--n-format", type=int, default=40)
    ap.add_argument("--n-cutoff", type=int, default=40)
    ap.add_argument("--n-control", type=int, default=20)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(os.path.expanduser("~/.cache/fdrl/qwen3_tokenizer.json"))
    rng = random.Random(a.seed)
    pools = {"format": [], "cutoff": [], "agree_right": [], "agree_wrong": []}
    for d in sorted(glob.glob(os.path.join(a.runs, "*"))):
        if json.load(open(os.path.join(d, "config.json"))).get("task", "math") != "math":
            continue
        run = os.path.basename(d)
        for line in gzip.open(os.path.join(d, "samples.jsonl.gz"), "rt"):
            s = json.loads(line)
            if s["step"] not in LATE or float(s["r_arm"]) <= 0.5:
                continue
            g = str(s["gold"])
            cut = len(tok.encode(s["reply"], add_special_tokens=False).ids) >= 1536
            trl, verl, gl = ck.trl_accuracy(s["reply"], g), ck.verl_math_reward(s["reply"], g), ck.gold_label(s["reply"], g)
            item = {"run": run, "step": s["step"], "gold": g, "question": s["question"], "reply": s["reply"],
                    "hit_limit": cut, "trl": trl, "verl_math": verl, "gold_label": gl}
            if run in ("tail50_slope_s1", "tail50_density_s2") and verl == 0 and trl == 1 and not cut:
                pools["format"].append(item)
            elif cut and (trl == 1 or gl == 1):
                pools["cutoff"].append(item)
            elif not cut and trl == verl == gl == 1:
                pools["agree_right"].append(item)
            elif not cut and trl == verl == gl == 0:
                pools["agree_wrong"].append(item)
    take = lambda p, n: rng.sample(pools[p], min(n, len(pools[p])))  # noqa: E731
    items = ([dict(x, group="format") for x in take("format", a.n_format)]
             + [dict(x, group="cutoff") for x in take("cutoff", a.n_cutoff)]
             + [dict(x, group="control_right") for x in take("agree_right", a.n_control // 2)]
             + [dict(x, group="control_wrong") for x in take("agree_wrong", a.n_control - a.n_control // 2)])
    rng.shuffle(items)
    for i, x in enumerate(items):
        x["id"] = f"b1-{i:03d}"
    print({k: len(v) for k, v in pools.items()}, "->", len(items), "items")
    json.dump([{k: x[k] for k in ("id", "group", "run", "step", "trl", "verl_math", "gold_label", "hit_limit")}
               for x in items], open(a.key, "w"), indent=1)
    shown = [{"id": x["id"], "q": x["question"], "r": x["reply"], "gold": x["gold"], "cut": x["hit_limit"]} for x in items]
    page = TEMPLATE.replace("__DATA__", json.dumps(shown).replace("</", "<\\/"))
    open(os.path.expanduser(a.out), "w").write(page)
    print("wrote", a.out, "and key", a.key)


TEMPLATE = r"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Answer Labelling</title>
<style>
:root{--bg:#fafaf9;--fg:#1c1917;--mut:#78716c;--card:#fff;--line:#e7e5e4;--acc:#2563eb;--warn:#b45309}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#1c1917;--fg:#f5f5f4;--mut:#a8a29e;--card:#292524;--line:#44403c;--acc:#60a5fa;--warn:#f59e0b}}
body{background:var(--bg);color:var(--fg);font:15px/1.5 -apple-system,system-ui,sans-serif;margin:0;padding:16px}
.wrap{max-width:860px;margin:0 auto}.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px;margin:12px 0}
h1{font-size:18px;margin:0}.mut{color:var(--mut);font-size:13px}pre{white-space:pre-wrap;word-wrap:break-word;font:13px/1.45 ui-monospace,monospace;max-height:420px;overflow:auto;margin:0}
.badge{display:inline-block;background:var(--warn);color:#fff;border-radius:6px;padding:2px 8px;font-size:12px;margin-left:8px}
button{font:inherit;padding:8px 12px;border:1px solid var(--line);border-radius:8px;background:var(--card);color:var(--fg);cursor:pointer;margin:4px 4px 4px 0}
button.on{background:var(--acc);color:#fff;border-color:var(--acc)}input[type=text]{font:inherit;padding:8px;border:1px solid var(--line);border-radius:8px;width:240px;background:var(--card);color:var(--fg)}
.row{margin:10px 0}.gold{font-size:18px;font-weight:600}.hidden{display:none}
</style></head><body><div class="wrap">
<div style="display:flex;justify-content:space-between;align-items:center;gap:8px;flex-wrap:wrap"><h1>Answer labelling</h1>
<span class="mut" id="prog"></span></div>
<p class="mut">Read the reply. First record the final answer it commits to. Only then reveal the correct answer and mark right or wrong. Rule: a reply cut off before stating a final answer = "cut off, no answer". If it states several different final answers, the last one counts, and tick "several answers". Keys: 1/2/3 for step 1, R/W for right/wrong, Enter = next.</p>
<div class="card"><div class="mut">Question</div><pre id="q"></pre></div>
<div class="card"><div class="mut">Reply <span id="cut"></span></div><pre id="r"></pre></div>
<div class="card">
<div class="row"><b>1. Final answer given?</b><br>
<button data-k="given" data-v="yes">1 · Yes</button><button data-k="given" data-v="cutoff">2 · Cut off, no answer</button><button data-k="given" data-v="none">3 · No answer</button>
<span id="ansbox" class="hidden"> &nbsp;Its final answer: <input type="text" id="ans" placeholder="type it exactly"></span></div>
<div class="row"><button id="reveal">Reveal correct answer</button> <span id="goldwrap" class="hidden">Correct answer: <span class="gold" id="gold"></span></span></div>
<div class="row" id="step2" class="hidden"><b>2. Is the committed answer right?</b><br>
<button data-k="right" data-v="right">R · Right</button><button data-k="right" data-v="wrong">W · Wrong</button></div>
<div class="row"><b>3. Format</b> (any that apply)<br>
<button data-f="clean_box">clean box</button><button data-f="nested_box">nested box</button><button data-f="bold_or_text">bold / text in box</button><button data-f="repeated_box">same box repeated</button><button data-f="several_answers">several different answers</button><button data-f="prose">answer in prose</button></div>
<div class="row"><button id="prev">← Back</button><button id="next" style="font-weight:600">Next →</button>
<button id="dl" style="float:right">Download labels</button></div></div>
<p class="mut">Labels save in this browser as you go. When finished (or pausing), click "Download labels" and tell Claude.</p>
</div><script>
const D=__DATA__;const KEY="fdrl_labels_b1";let L={};try{L=JSON.parse(localStorage.getItem(KEY)||"{}")}catch(e){}
let i=0;try{i=+localStorage.getItem(KEY+"_i")||0}catch(e){}
const $=s=>document.querySelector(s),save=()=>{try{localStorage.setItem(KEY,JSON.stringify(L));localStorage.setItem(KEY+"_i",i)}catch(e){}};
function cur(){const id=D[i].id;L[id]=L[id]||{id,fmt:[]};return L[id]}
function render(){const d=D[i],l=cur();$("#q").textContent=d.q;$("#r").textContent=d.r;$("#r").scrollTop=1e9;
$("#cut").innerHTML=d.cut?'<span class="badge">hit the 1,536-token limit</span>':"";
document.querySelectorAll("[data-k]").forEach(b=>b.classList.toggle("on",l[b.dataset.k]===b.dataset.v));
document.querySelectorAll("[data-f]").forEach(b=>b.classList.toggle("on",l.fmt.includes(b.dataset.f)));
$("#ansbox").classList.toggle("hidden",l.given!=="yes");$("#ans").value=l.answer||"";
const rev=!!l.revealed;$("#goldwrap").classList.toggle("hidden",!rev);$("#gold").textContent=rev?d.gold:"";
$("#step2").style.display=rev?"block":"none";
const done=D.filter(x=>L[x.id]&&L[x.id].right).length;$("#prog").textContent=`item ${i+1} of ${D.length} · ${done} labelled`;}
document.querySelectorAll("[data-k]").forEach(b=>b.onclick=()=>{const l=cur();l[b.dataset.k]=b.dataset.v;if(b.dataset.k==="given"&&b.dataset.v!=="yes"){l.answer=""}save();render();if(b.dataset.v==="yes")$("#ans").focus()});
document.querySelectorAll("[data-f]").forEach(b=>b.onclick=()=>{const l=cur(),f=b.dataset.f;l.fmt=l.fmt.includes(f)?l.fmt.filter(x=>x!==f):[...l.fmt,f];save();render()});
$("#ans").oninput=e=>{cur().answer=e.target.value;save()};
$("#reveal").onclick=()=>{const l=cur();if(!l.given){alertLike("Do step 1 first.");return}l.revealed=true;l.answer_before_reveal=l.answer||"";save();render()};
function alertLike(m){$("#prog").textContent=m;setTimeout(render,1500)}
$("#next").onclick=()=>{if(i<D.length-1){i++;save();render();window.scrollTo(0,0)}};$("#prev").onclick=()=>{if(i>0){i--;save();render()}};
$("#dl").onclick=()=>{const blob=new Blob([JSON.stringify(Object.values(L),null,1)],{type:"application/json"});const u=URL.createObjectURL(blob),a=document.createElement("a");a.href=u;a.download="labels_batch1.json";a.click();URL.revokeObjectURL(u)};
document.addEventListener("keydown",e=>{if(e.target.tagName==="INPUT"){if(e.key==="Enter"){$("#reveal").click()}return}
const m={"1":"yes","2":"cutoff","3":"none"};if(m[e.key])document.querySelector(`[data-k=given][data-v=${m[e.key]}]`).click();
if(e.key==="r"||e.key==="R"){if(cur().revealed)document.querySelector("[data-k=right][data-v=right]").click()}
if(e.key==="w"||e.key==="W"){if(cur().revealed)document.querySelector("[data-k=right][data-v=wrong]").click()}
if(e.key==="Enter"){cur().revealed?$("#next").click():$("#reveal").click()}});
render();
</script></body></html>"""

if __name__ == "__main__":
    main()
