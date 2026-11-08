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
    ap.add_argument("--batch", type=int, default=1, help="2: cut-off answers graders call wrong, strict-only disputes, early steps")
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
    if a.batch == 2:
        done = {(x["run"], x["step"], x["reply"][:300]) for x in json.load(open("results/label_batch1_items.json"))}
        b2 = {"cut_wrong": [], "strict_only": [], "early": []}
        for d in sorted(glob.glob(os.path.join(a.runs, "*"))):
            if json.load(open(os.path.join(d, "config.json"))).get("task", "math") != "math":
                continue
            run = os.path.basename(d)
            for line in gzip.open(os.path.join(d, "samples.jsonl.gz"), "rt"):
                s = json.loads(line)
                if float(s["r_arm"]) <= 0.5 or s["step"] not in LATE | {1, 10}:
                    continue
                g = str(s["gold"])
                if (run, s["step"], s["reply"][:300]) in done:
                    continue
                cut = len(tok.encode(s["reply"], add_special_tokens=False).ids) >= 1536
                trl, gl, st = ck.trl_accuracy(s["reply"], g), ck.gold_label(s["reply"], g), ck.strict(s["reply"], g)
                item = {"run": run, "step": s["step"], "gold": g, "question": s["question"], "reply": s["reply"],
                        "hit_limit": cut, "trl": trl, "verl_math": ck.verl_math_reward(s["reply"], g), "gold_label": gl}
                if s["step"] in LATE and cut and trl == 0 and gl == 0:
                    b2["cut_wrong"].append(item)          # the "real exploitation" cases: are they truly answerless?
                elif s["step"] in LATE and not cut and st == 0 and trl == 1:
                    b2["strict_only"].append(item)        # strict calls wrong, TRL right: strict's false alarms
                elif s["step"] in (1, 10) and (st != trl or trl != gl):
                    b2["early"].append(item)              # disputes at the start of training (baseline for E)
        pools.update(b2)
        items = ([dict(x, group="cut_wrong") for x in take("cut_wrong", 50)]
                 + [dict(x, group="strict_only") for x in take("strict_only", 25)]
                 + [dict(x, group="early") for x in take("early", 15)]
                 + [dict(x, group="control_right") for x in take("agree_right", 10)])
        rng.shuffle(items)
        for i, x in enumerate(items):
            x["id"] = f"b2-{i:03d}"
        print({k: len(v) for k, v in b2.items()}, "->", len(items), "items")
        a.key = a.key.replace("batch1", "batch2")
        a.out = os.path.expanduser(a.out).replace("batch1", "batch2")
        json.dump(items, open(a.key.replace("_key.json", "_items.json"), "w"))
        json.dump([{k: x[k] for k in ("id", "group", "run", "step", "trl", "verl_math", "gold_label", "hit_limit")} for x in items], open(a.key, "w"), indent=1)
        shown = [{"id": x["id"], "q": x["question"], "r": x["reply"], "gold": x["gold"], "cut": x["hit_limit"]} for x in items]
        page = TEMPLATE.replace("__DATA__", json.dumps(shown).replace("</", "<\\/")).replace("fdrl_labels_b1_simple", "fdrl_labels_b2")
        page = page.replace('a.download="labels_batch1.json"', 'a.download="labels_batch2.json"')
        open(a.out, "w").write(page)
        print("wrote", a.out)
        return
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
    json.dump(items, open(a.key.replace("_key.json", "_items.json"), "w"))  # full items (reply, gold) for scoring
    shown = [{"id": x["id"], "q": x["question"], "r": x["reply"], "gold": x["gold"], "cut": x["hit_limit"]} for x in items]
    page = TEMPLATE.replace("__DATA__", json.dumps(shown).replace("</", "<\\/"))
    open(os.path.expanduser(a.out), "w").write(page)
    print("wrote", a.out, "and key", a.key)


TEMPLATE = r"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Answer Labelling</title>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/contrib/auto-render.min.js"></script>
<style>
:root{--bg:#fafaf9;--fg:#1c1917;--mut:#78716c;--card:#fff;--line:#e7e5e4;--acc:#2563eb;--hl:#fde68a;--warn:#b45309}
@media (prefers-color-scheme:dark){:root{--bg:#1c1917;--fg:#f5f5f4;--mut:#a8a29e;--card:#292524;--line:#44403c;--acc:#60a5fa;--hl:#854d0e;--warn:#f59e0b}}
body{background:var(--bg);color:var(--fg);font:15px/1.5 -apple-system,system-ui,sans-serif;margin:0;padding:16px}
.wrap{max-width:820px;margin:0 auto}.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin:10px 0}
.mut{color:var(--mut);font-size:13px}.reply{white-space:pre-wrap;word-wrap:break-word;font-size:16px;line-height:1.65;max-height:62vh;overflow:auto}.reply .katex{font-size:1.05em}.res{font-size:17px;margin-top:8px}.ok{color:#16a34a;font-weight:600}.bad{color:#dc2626;font-weight:600}
mark{background:var(--hl);color:inherit;border-radius:3px;padding:0 2px}.badge{display:inline-block;background:var(--warn);color:#fff;border-radius:6px;padding:1px 8px;font-size:12px;margin-left:6px}
button{font:inherit;padding:9px 14px;border:1px solid var(--line);border-radius:8px;background:var(--card);color:var(--fg);cursor:pointer;margin:4px 6px 4px 0}
button.on{background:var(--acc);color:#fff;border-color:var(--acc)}input{font:inherit;padding:8px;border:1px solid var(--line);border-radius:8px;width:150px;background:var(--card);color:var(--fg)}
.top{display:flex;justify-content:space-between;align-items:baseline;gap:8px;flex-wrap:wrap}.q{font-size:14px}
</style></head><body><div class="wrap">
<div class="top"><b style="font-size:18px">What final answer does this reply commit to?</b><span class="mut" id="prog"></span></div>
<p class="mut">Pick the answer the reply ends on. If it stops before giving one, pick "Cut off". If it gives several different final answers, pick the last one. After you pick, the correct answer is shown with ✓ or ✗. Keys: 1-5 pick, C cut off, N no answer, Enter next, Backspace back.</p>
<div class="card q"><span class="mut">Question: </span><span id="q"></span></div>
<div class="card"><div class="mut">Reply <span id="cut"></span> · opens at the end; scroll up for the working · <a href="#" id="raw">show raw text</a></div><div class="reply" id="r"></div></div>
<div class="card"><div id="chips"></div>
<button data-v="__cut">C · Cut off, no answer</button><button data-v="__none">N · No final answer</button>
<span> or type: <input id="other" placeholder="other answer"></span><div class="res" id="res"></div>
<div style="margin-top:10px"><button id="prev">← Back</button><button id="next"><b>Next →</b></button>
<button id="dl" style="float:right">Download labels</button></div></div>
<p class="mut">Saved in this browser as you go. Do as many as you can; click "Download labels" when you stop.</p>
</div><script>
const D=__DATA__;const KEY="fdrl_labels_b1_simple";let L={};try{L=JSON.parse(localStorage.getItem(KEY)||"{}")}catch(e){}
let i=0,full=false;try{i=+localStorage.getItem(KEY+"_i")||0}catch(e){}
const $=s=>document.querySelector(s),save=()=>{try{localStorage.setItem(KEY,JSON.stringify(L));localStorage.setItem(KEY+"_i",i)}catch(e){}};
function boxes(t){const out=[];let k=t.indexOf("\\boxed{");while(k>=0){let d=1,j=k+7;for(;j<t.length&&d;j++){if(t[j]=="{")d++;else if(t[j]=="}")d--}out.push(t.slice(k+7,j-1));k=t.indexOf("\\boxed{",k+1)}return out}
function clean(b){let s=b.trim();for(let n=0;n<6;n++){const m=s.match(/^\$*\s*\\(?:boxed|mathbf|textbf|text|mathrm|bm|displaystyle)\s*\{([\s\S]*)\}\s*\$*$/);if(!m)break;s=m[1].trim()}s=s.replace(/\$/g,"").trim();if(s.includes("="))s=s.split("=").pop().trim();return s}
function cands(t){const c=[];boxes(t).forEach(b=>{const v=clean(b);if(v&&!c.includes(v))c.push(v)});const nums=t.replace(/,/g,"").match(/-?\d+/g)||[];const last=nums[nums.length-1];if(last&&!c.includes(last))c.push(last);return c.slice(-5)}
function esc(s){return s.replace(/&/g,"&amp;").replace(/</g,"&lt;")}
function render(){const d=D[i],l=L[d.id]||{};$("#q").textContent=d.q;try{renderMathInElement($("#q"),{delimiters:[{left:"$$",right:"$$",display:false},{left:"$",right:"$",display:false},{left:"\\(",right:"\\)",display:false}],throwOnError:false})}catch(e){}
$("#r").textContent=d.r;$("#raw").textContent=full?"show rendered maths":"show raw text";if(!full)try{renderMathInElement($("#r"),{delimiters:[{left:"$$",right:"$$",display:true},{left:"\\[",right:"\\]",display:true},{left:"\\(",right:"\\)",display:false},{left:"$",right:"$",display:false}],throwOnError:false})}catch(e){}$("#r").scrollTop=1e9;
$("#cut").innerHTML=d.cut?'<span class="badge">hit the length limit</span>':"";const g=d.gold,a=l.answer;$("#res").innerHTML=!a?"":`Correct answer: <b>${esc(g)}</b> · you picked: <b>${a==="__cut"?"cut off":a==="__none"?"no final answer":esc(a)}</b> → `+(norm(a)===norm(g)?'<span class="ok">✓ right</span>':'<span class="bad">✗ not right</span>')+' · press Enter for next';
const cs=cands(d.r);$("#chips").innerHTML=cs.map((c,k)=>`<button data-v="${esc(c)}">${k+1} · ${esc(c)}</button>`).join("")||'<span class="mut">no candidate answers found</span>';
document.querySelectorAll("[data-v]").forEach(b=>{b.classList.toggle("on",l.answer===b.dataset.v);b.onclick=()=>pick(b.dataset.v)});
$("#other").value=(l.answer&&!cs.includes(l.answer)&&!l.answer.startsWith("__"))?l.answer:"";
const done=D.filter(x=>L[x.id]).length;$("#prog").textContent=`${i+1} of ${D.length} · ${done} done`}
function norm(x){return String(x||"").replace(/[^0-9.-]/g,"").replace(/\.0+$/,"")}
function pick(v){full=false;const d=D[i];L[d.id]={id:d.id,answer:v,t:Date.now()};save();render()}
function next(){if(i<D.length-1){i++;full=false;save();render();window.scrollTo(0,0)}}
$("#next").onclick=next;$("#raw").onclick=e=>{e.preventDefault();full=!full;render()};
$("#prev").onclick=()=>{if(i>0){i--;full=false;save();render()}};

$("#other").onkeydown=e=>{if(e.key==="Enter"&&e.target.value.trim()){pick(e.target.value.trim())}};
$("#dl").onclick=()=>{const blob=new Blob([JSON.stringify(Object.values(L),null,1)],{type:"application/json"});const u=URL.createObjectURL(blob),a=document.createElement("a");a.href=u;a.download="labels_batch1.json";a.click();URL.revokeObjectURL(u)};
document.addEventListener("keydown",e=>{if(e.target.tagName==="INPUT")return;const cs=cands(D[i].r);
if(/^[1-5]$/.test(e.key)&&cs[+e.key-1])pick(cs[+e.key-1]);if(e.key==="c"||e.key==="C")pick("__cut");if(e.key==="n"||e.key==="N")pick("__none");
if(e.key==="Enter")next();if(e.key==="Backspace"){e.preventDefault();$("#prev").click()}});
render();
</script></body></html>"""

if __name__ == "__main__":
    main()
