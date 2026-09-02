"""Minimal local web interface.

The problem statement is explicit that a polished GUI is not required and that
numerical robustness is what is being judged, so this stays deliberately thin:
one page, one endpoint, no build step, no external assets. It exists to make the
engine operable by someone who is not going to use the CLI, and to show the
CPU-versus-GPU comparison and convergence behaviour live.

Runs entirely on localhost. Nothing leaves the machine.

    python -m ui.server            then open http://127.0.0.1:8000
"""

from __future__ import annotations

import io
import os
import sys
import time
import traceback

import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

from sovopt.core.backend import (GPU_ERROR, gpu_available,      # noqa: E402
                                gpu_selftest)
from sovopt.core.problem import ObjSense, Status, VarKind         # noqa: E402
from sovopt.io import read_model                                  # noqa: E402
from sovopt.models import TEMPLATES                               # noqa: E402
from sovopt.numerics.scaling import compute_scaling               # noqa: E402

app = FastAPI(title="SOVOPT")

PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>SOVOPT</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{--bg:#0e1418;--panel:#151f25;--line:#243139;--ink:#dbe4e8;--dim:#8496a0;
      --cpu:#d5a34a;--gpu:#4fc2ce;--ok:#62ac84;--bad:#d97867;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
     font:14px/1.55 ui-monospace,"Cascadia Mono",Consolas,monospace}
header{padding:18px 24px;border-bottom:1px solid var(--line);display:flex;
       align-items:baseline;gap:14px;flex-wrap:wrap}
h1{margin:0;font-size:19px;letter-spacing:.14em}
.sub{color:var(--dim);font-size:12px}
main{display:grid;grid-template-columns:340px 1fr;gap:0;min-height:calc(100vh - 60px)}
aside{padding:20px 22px;border-right:1px solid var(--line);display:flex;
      flex-direction:column;gap:16px}
section{padding:20px 24px;overflow:auto}
label{display:block;font-size:11px;letter-spacing:.11em;text-transform:uppercase;
      color:var(--dim);margin-bottom:5px}
select,input,textarea,button{width:100%;background:var(--panel);color:var(--ink);
      border:1px solid var(--line);border-radius:4px;padding:8px 10px;font:inherit}
textarea{height:130px;resize:vertical}
button{cursor:pointer;background:#1d2b33;border-color:#2e434e}
button:hover{background:#243440}
button:disabled{opacity:.5;cursor:default}
.row{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:6px;
      padding:16px 18px;margin-bottom:14px}
.card h2{margin:0 0 12px;font-size:12px;letter-spacing:.13em;text-transform:uppercase;
      color:var(--dim);font-weight:600}
table{width:100%;border-collapse:collapse}
td{padding:4px 0;vertical-align:top}
td:first-child{color:var(--dim);width:170px}
.big{font-size:22px}
.ok{color:var(--ok)}.bad{color:var(--bad)}
.cpu{color:var(--cpu)}.gpu{color:var(--gpu)}
.bar{height:9px;border-radius:5px;margin:4px 0 10px}
pre{margin:0;white-space:pre-wrap;color:var(--dim);font-size:12px}
.muted{color:var(--dim);font-size:12px}
</style></head><body>
<header><h1>V Y U H A</h1>
<span class="sub">sovereign optimization engine &middot; SIH 2026 PS 26119 &middot; MRPL</span>
<span class="sub" id="hw"></span></header>
<main>
<aside>
  <div><label>model source</label>
    <select id="src">
      <option value="template">built-in refinery template</option>
      <option value="mps">paste MPS</option>
    </select></div>

  <div id="tmplBox">
    <label>template</label>
    <select id="tmpl">
      <option value="blending">product blending (LP)</option>
      <option value="production_planning">production planning (LP)</option>
      <option value="unit_scheduling">unit scheduling (MILP)</option>
    </select>
    <div class="row" style="margin-top:10px">
      <div><label>size</label><input id="size" type="number" value="12" min="2" max="200"></div>
      <div><label>seed</label><input id="seed" type="number" value="0"></div>
    </div>
  </div>

  <div id="mpsBox" style="display:none">
    <label>MPS text</label><textarea id="mps" placeholder="NAME  ..."></textarea>
  </div>

  <div class="row">
    <div><label>device</label>
      <select id="dev"><option value="both">compare cpu &amp; gpu</option>
      <option value="cpu">cpu</option><option value="gpu">gpu</option></select></div>
    <div><label>time limit (s)</label><input id="tl" type="number" value="30"></div>
  </div>

  <button id="go">SOLVE</button>
  <div class="muted" id="note">Runs locally. No data leaves this machine.</div>
</aside>

<section id="out">
  <div class="card"><h2>ready</h2>
  <div class="muted">Pick a model and press Solve. The engine builds the model,
  scales it, and reports the solution together with an independent feasibility
  check computed from the original data.</div></div>
</section>
</main>
<script>
const $=id=>document.getElementById(id);
$('src').onchange=()=>{const t=$('src').value==='template';
  $('tmplBox').style.display=t?'':'none';$('mpsBox').style.display=t?'none':'';};
fetch('/api/hw').then(r=>r.json()).then(h=>{
  $('hw').textContent=h.gpu? ('GPU: '+h.name+'  |  CPU threads: '+h.threads)
                           : ('CPU only ('+h.threads+' threads)');});

function esc(s){return String(s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}

function chart(hist){
  if(!hist||hist.length<2) return '';
  const w=560,h=150,pad=30;
  const xs=hist.map(p=>p[0]);
  const gy=hist.map(p=>Math.max(p[3],1e-14));
  const lx=v=>pad+(w-2*pad)*(v-xs[0])/Math.max(xs[xs.length-1]-xs[0],1);
  const lo=Math.log10(Math.min(...gy)),hi=Math.log10(Math.max(...gy));
  const ly=v=>h-pad-(h-2*pad)*(Math.log10(v)-lo)/Math.max(hi-lo,1e-9);
  const pts=hist.map((p,i)=>lx(xs[i])+','+ly(gy[i])).join(' ');
  return '<div class="card"><h2>convergence — duality gap</h2><svg width="100%" viewBox="0 0 '+w+' '+h+'">'
    +'<polyline fill="none" stroke="#4fc2ce" stroke-width="1.6" points="'+pts+'"/>'
    +'<text x="'+pad+'" y="14" fill="#8496a0" font-size="10">1e'+hi.toFixed(0)+'</text>'
    +'<text x="'+pad+'" y="'+(h-8)+'" fill="#8496a0" font-size="10">1e'+lo.toFixed(0)+' after '+xs[xs.length-1]+' iters</text>'
    +'</svg></div>';
}

function resultCard(r){
  const s=r.status==='OPTIMAL'?'ok':'bad';
  const feas=(r.viol_row<1e-6&&r.viol_bound<1e-6&&r.viol_int<1e-6);
  return '<div class="card"><h2>'+esc(r.device)+' result</h2><table>'
   +'<tr><td>status</td><td class="'+s+'">'+esc(r.status)+'</td></tr>'
   +'<tr><td>objective</td><td class="big">'+ (r.objective==null?'-':r.objective.toPrecision(12)) +'</td></tr>'
   +(r.dual_bound!=null?'<tr><td>dual bound</td><td>'+r.dual_bound.toPrecision(10)+'</td></tr>':'')
   +(r.nodes?'<tr><td>nodes</td><td>'+r.nodes.toLocaleString()+'</td></tr>':'')
   +(r.iterations?'<tr><td>iterations</td><td>'+r.iterations.toLocaleString()+'</td></tr>':'')
   +'<tr><td>time</td><td>'+r.time.toFixed(3)+' s</td></tr>'
   +'<tr><td>method</td><td>'+esc(r.method)+'</td></tr>'
   +'<tr><td>independent check</td><td class="'+(feas?'ok':'bad')+'">'
     +(feas?'feasible':'VIOLATION')+' — row '+r.viol_row.toExponential(1)
     +', bound '+r.viol_bound.toExponential(1)
     +', integrality '+r.viol_int.toExponential(1)+'</td></tr>'
   +'</table></div>';
}

$('go').onclick=async()=>{
  $('go').disabled=true;$('go').textContent='SOLVING…';
  const body={source:$('src').value,template:$('tmpl').value,
    size:+$('size').value,seed:+$('seed').value,mps:$('mps').value,
    device:$('dev').value,time_limit:+$('tl').value};
  let d;
  try{ d=await (await fetch('/api/solve',{method:'POST',
        headers:{'content-type':'application/json'},body:JSON.stringify(body)})).json(); }
  catch(e){ d={error:String(e)}; }
  $('go').disabled=false;$('go').textContent='SOLVE';
  if(d.error){ $('out').innerHTML='<div class="card"><h2 class="bad">error</h2><pre>'+esc(d.error)+'</pre></div>'; return; }

  let html='<div class="card"><h2>model</h2><pre>'+esc(d.summary)+'</pre>'
    +'<table style="margin-top:10px">'
    +'<tr><td>coefficient ratio</td><td>'+d.coeff_ratio.toExponential(2)
      +'  &rarr; '+d.scaled_ratio.toExponential(2)+' after '+esc(d.scaling)+'</td></tr>'
    +'</table></div>';

  for(const r of d.results) html+=resultCard(r);

  if(d.results.length===2){
    const a=d.results[0],b=d.results[1];
    const f=a.time/b.time, wide=Math.max(a.time,b.time);
    html+='<div class="card"><h2>cpu vs gpu</h2>'
      +'<div>CPU '+a.time.toFixed(3)+'s</div><div class="bar" style="background:var(--cpu);width:'+(100*a.time/wide)+'%"></div>'
      +'<div>GPU '+b.time.toFixed(3)+'s</div><div class="bar" style="background:var(--gpu);width:'+(100*b.time/wide)+'%"></div>'
      +'<div class="muted">'+(f>=1?('GPU '+f.toFixed(2)+'x faster'):('CPU '+(1/f).toFixed(2)+'x faster — this model is too small to fill the GPU'))
      +'. Objectives agree to '+d.agreement.toExponential(1)+'.</div></div>';
  }
  const withHist=d.results.find(r=>r.history&&r.history.length>1);
  if(withHist) html+=chart(withHist.history);
  $('out').innerHTML=html;
};
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE


@app.get("/api/hw")
def hw():
    from sovopt.core._jit import NUMBA_THREADS
    out = {"gpu": gpu_available(), "threads": NUMBA_THREADS, "name": ""}
    if gpu_available():
        import cupy as cp
        out["name"] = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
        # A device that CuPy can see but whose kernels will not build is worse
        # than no device: the page would offer a GPU run that dies mid-solve.
        ok, err = gpu_selftest()
        out["gpu"] = ok
        if not ok:
            out["error"] = err
    else:
        out["error"] = GPU_ERROR
    return out


def _build(body):
    if body.get("source") == "mps":
        text = body.get("mps", "")
        if not text.strip():
            raise ValueError("no model text supplied")
        # Accept either format and work out which by looking at the text: the
        # reader is chosen by extension, and pasted text has no filename. MPS
        # is a sectioned card format, so its section keywords are decisive.
        upper = text.upper()
        is_mps = "ROWS" in upper and "COLUMNS" in upper
        ext = ".mps" if is_mps else ".lp"
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "_paste" + ext)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
        return read_model(path)

    name = body.get("template", "blending")
    fn = TEMPLATES[name]
    size = max(2, min(int(body.get("size", 12)), 200))
    seed = int(body.get("seed", 0))
    if name == "blending":
        return fn(n_components=size, n_products=max(2, size // 3),
                  n_properties=3, seed=seed)
    if name == "production_planning":
        return fn(n_crudes=max(2, size // 2), n_units=max(2, size // 3),
                  n_products=max(2, size // 2), n_periods=6, seed=seed)
    return fn(n_units=max(2, size // 2), n_periods=max(4, size), seed=seed)


@app.post("/api/solve")
async def api_solve(request: Request):
    body = await request.json()
    try:
        from sovopt.cli import solve

        prob = _build(body)
        sc = compute_scaling(prob.A, method="auto")

        devices = ({"both": ["cpu", "gpu"], "cpu": ["cpu"], "gpu": ["gpu"]}
                   [body.get("device", "both")])
        if "gpu" in devices and not gpu_selftest()[0]:
            devices = [d for d in devices if d != "gpu"] or ["cpu"]

        tl = float(body.get("time_limit", 30))
        results = []
        for dev in devices:
            t = time.perf_counter()
            sol = solve(prob, device=dev, time_limit=tl)
            dt = time.perf_counter() - t
            rv, bv, iv = (prob.violation(sol.x) if sol.x is not None
                          else (float("nan"),) * 3)
            hist = [[int(h[0]), float(h[1]), float(h[2]), float(h[3])]
                    for h in (sol.log or []) if len(h) >= 4]
            results.append({
                "device": dev,
                "status": sol.status.name,
                "objective": (float(sol.objective)
                              if np.isfinite(sol.objective) else None),
                "dual_bound": (float(sol.dual_bound)
                               if np.isfinite(sol.dual_bound) and sol.nodes else None),
                "nodes": int(sol.nodes),
                "iterations": int(sol.iterations),
                "time": dt,
                "method": sol.method,
                "viol_row": float(rv), "viol_bound": float(bv), "viol_int": float(iv),
                "history": hist[-400:],
            })

        agree = 0.0
        if len(results) == 2 and results[0]["objective"] is not None \
                and results[1]["objective"] is not None:
            a, b = results[0]["objective"], results[1]["objective"]
            agree = abs(a - b) / max(1.0, abs(a))

        return JSONResponse({
            "summary": prob.summary(),
            "coeff_ratio": sc.ratio_before,
            "scaled_ratio": sc.ratio_after,
            "scaling": sc.method,
            "results": results,
            "agreement": agree,
        })
    except Exception:
        return JSONResponse({"error": traceback.format_exc(limit=4)})


def main():
    import uvicorn
    print("SOVOPT interface on http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")


if __name__ == "__main__":
    main()
