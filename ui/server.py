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


def _finite(v):
    """A float for the JSON, or None where there is none to give. A solve
    that ends without a point -- the interior point at its time limit on a
    month of hourly blending -- left NaN here, which JSON cannot carry, and
    the page showed a traceback instead of the status."""
    v = float(v)
    return v if np.isfinite(v) else None


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
      <option value="file">upload a model file (MPS / LP / QPS, .gz or .bz2)</option>
      <option value="blend">blend from CSV tables (components + products)</option>
      <option value="mps">paste MPS / LP text</option>
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

  <div id="fileBox" style="display:none">
    <label>model file</label>
    <input id="modelfile" type="file" accept=".mps,.lp,.qps,.gz,.bz2,.xz">
    <div class="muted">Read by the same reader as the command line: free or
    fixed-column MPS with QUADOBJ, CPLEX LP, compressed or not. The file is
    sent to the local server only.</div>
  </div>

  <div id="blendBox" style="display:none">
    <label>components.csv</label>
    <input id="compfile" type="file" accept=".csv,.txt">
    <label style="margin-top:8px">products.csv</label>
    <input id="prodfile" type="file" accept=".csv,.txt">
    <div class="muted">One row per component: name, cost, available, minimum,
    then a column per quality. One row per product: name, price, demand_min,
    demand_max, then &lt;quality&gt;_min / &lt;quality&gt;_max. Excel's "save as
    CSV". See examples/blending.</div>
    <label style="margin-top:10px">prices.csv &mdash; optional: a horizon</label>
    <input id="pricefile" type="file" accept=".csv,.txt">
    <label style="margin-top:6px">demand.csv &mdash; optional, with prices</label>
    <input id="demandfile" type="file" accept=".csv,.txt">
    <label style="margin-top:6px">capacity.csv &mdash; optional, with prices</label>
    <input id="capfile" type="file" accept=".csv,.txt">
    <div class="muted">A prices table (one row per period, a column per
    component) makes it a multi-period plan with purchases and storage; add
    line, storage_max, storage_cost, opening_stock, closing_stock to the
    components. See examples/planning.</div>
    <label style="margin-top:10px">pools.csv &mdash; optional: pooled qualities</label>
    <input id="poolfile" type="file" accept=".csv,.txt">
    <div class="muted">Pools (name, capacity, inputs) between components and
    products; a "direct" column on the components names the products they may
    ship to without a pool. Solved to proven global optimality. See
    examples/pooling.</div>
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
$('src').onchange=()=>{const v=$('src').value;
  $('tmplBox').style.display=v==='template'?'':'none';
  $('mpsBox').style.display=v==='mps'?'':'none';
  $('fileBox').style.display=v==='file'?'':'none';
  $('blendBox').style.display=v==='blend'?'':'none';};

function readText(input){
  return new Promise((res,rej)=>{const f=input.files&&input.files[0];
    if(!f){res(null);return;} const r=new FileReader();
    r.onload=()=>res(r.result); r.onerror=()=>rej(r.error); r.readAsText(f);});}
function readBase64(input){
  return new Promise((res,rej)=>{const f=input.files&&input.files[0];
    if(!f){res(null);return;} const r=new FileReader();
    r.onload=()=>res({name:f.name,data:r.result.split(',')[1]}); r.onerror=()=>rej(r.error);
    r.readAsDataURL(f);});}

function fmt(v,d){return v==null?'-':(typeof v==='number'?v.toLocaleString(undefined,{maximumFractionDigits:d==null?4:d}):esc(v));}

function planCards(pl){
  if(!pl) return '';
  if(pl.objective==null) return '<div class="card"><h2 class="bad">no plan</h2><div>'+esc(pl.message||pl.status)+'</div></div>';
  if(pl.periods) return planningCards(pl);
  if(pl.pools) return poolingCards(pl);
  let h='<div class="card"><h2>plan &mdash; margin '+fmt(pl.objective,2)+'</h2>'
   +'<div class="muted">revenue '+fmt(pl.revenue,2)+' &minus; component cost '+fmt(pl.cost,2)
   +(pl.verifier_verdict?' &middot; independent check: <b>'+esc(pl.verifier_verdict)+'</b>'+(pl.verifier_detail?' ('+esc(pl.verifier_detail)+')':''):'')+'</div>';
  h+='<h2 style="margin-top:12px">products</h2><table><tr><th>product</th><th>volume</th><th>revenue</th><th>qualities (value, spec)</th></tr>';
  for(const p of pl.products){
    const qs=Object.entries(p.qualities).map(([q,v])=>q+' '+fmt(v.value)+(v.min!=null||v.max!=null?' ['+fmt(v.min)+' .. '+fmt(v.max)+']':'')+(v.binding?' <b>*'+v.binding+'</b>':'')).join('<br>');
    h+='<tr><td>'+esc(p.name)+'</td><td>'+fmt(p.volume)+'</td><td>'+fmt(p.revenue,2)+'</td><td>'+qs+'</td></tr>';}
  h+='</table><h2 style="margin-top:12px">components</h2><table><tr><th>component</th><th>used</th><th>available</th><th>cost</th><th>shadow price</th></tr>';
  for(const c of pl.components){
    h+='<tr><td>'+esc(c.name)+'</td><td>'+fmt(c.used)+(c.at_limit?' <b>at limit</b>':'')+'</td><td>'+fmt(c.available)+'</td><td>'+fmt(c.cost)+'</td><td>'+fmt(c.shadow_price)+'</td></tr>';}
  h+='</table><h2 style="margin-top:12px">recipe</h2><table><tr><th>product</th><th>component</th><th>quantity</th><th>fraction</th></tr>';
  for(const r of pl.recipe){
    h+='<tr><td>'+esc(r.product)+'</td><td>'+esc(r.component)+'</td><td>'+fmt(r.quantity)+'</td><td>'+(r.fraction==null?'-':(100*r.fraction).toFixed(1)+'%')+'</td></tr>';}
  h+='</table>';
  const b=pl.specs.filter(x=>x.binding);
  if(b.length){h+='<h2 style="margin-top:12px">binding specifications</h2><table><tr><th>specification</th><th>margin per unit relaxed</th></tr>';
    for(const x of b) h+='<tr><td>'+esc(x.row)+'</td><td>'+fmt(x.per_unit_of_quality,2)+'</td></tr>'; h+='</table>';}
  return h+'</div>';
}

function qualityCell(qs){
  return Object.entries(qs).map(([q,v])=>q+' '+fmt(v.value)+(v.min!=null||v.max!=null?' ['+fmt(v.min)+' .. '+fmt(v.max)+']':'')+(v.binding?' <b>*'+v.binding+'</b>':'')).join('<br>');
}

function planningCards(pl){
  const t=pl.totals;
  let h='<div class="card"><h2>plan over the horizon &mdash; margin '+fmt(pl.objective,2)+'</h2>'
   +'<div class="muted">revenue '+fmt(t.revenue,2)+' &minus; purchases '+fmt(t.purchases,2)+' &minus; storage '+fmt(t.storage,2)
   +(pl.verifier_verdict?' &middot; independent check: <b>'+esc(pl.verifier_verdict)+'</b>'+(pl.verifier_detail?' ('+esc(pl.verifier_detail)+')':''):'')+'</div>';
  for(const per of pl.periods){
    h+='<h2 style="margin-top:12px">'+esc(per.period)+'</h2><table><tr><th>component</th><th>buy</th><th>@ price</th><th>use</th><th>store</th></tr>';
    for(const c of per.components) h+='<tr><td>'+esc(c.name)+'</td><td>'+fmt(c.buy)+'</td><td>'+fmt(c.price)+'</td><td>'+fmt(c.use)+'</td><td>'+fmt(c.store)+'</td></tr>';
    h+='</table><table style="margin-top:6px"><tr><th>product</th><th>make</th><th>revenue</th><th>qualities (value, spec)</th></tr>';
    for(const p of per.products) h+='<tr><td>'+esc(p.name)+'</td><td>'+fmt(p.volume)+'</td><td>'+fmt(p.revenue,2)+'</td><td>'+qualityCell(p.qualities)+'</td></tr>';
    h+='</table>';}
  const b=pl.capacity.filter(c=>c.at_limit);
  if(b.length){h+='<h2 style="margin-top:12px">capacities at their limit</h2><table><tr><th>line</th><th>period</th><th>used / capacity</th><th>shadow price</th></tr>';
    for(const c of b) h+='<tr><td>'+esc(c.line)+'</td><td>'+esc(c.period)+'</td><td>'+fmt(c.used)+' / '+fmt(c.capacity)+'</td><td>'+fmt(c.shadow_price)+'</td></tr>'; h+='</table>';}
  return h+'</div>';
}

function poolingCards(pl){
  let h='<div class="card"><h2>pooling plan &mdash; margin '+fmt(pl.objective,2)+(pl.proved_global?' (proved globally optimal)':'')+'</h2>'
   +'<div class="muted">revenue '+fmt(pl.revenue,2)+' &minus; component cost '+fmt(pl.cost,2)
   +(pl.dual_bound!=null?' &middot; bound '+fmt(pl.dual_bound,2)+', '+pl.nodes+' nodes':'')
   +(pl.verifier_verdict?' &middot; independent check: <b>'+esc(pl.verifier_verdict)+'</b>'+(pl.verifier_detail?' ('+esc(pl.verifier_detail)+')':''):'')+'</div>';
  h+='<h2 style="margin-top:12px">pools</h2><table><tr><th>pool</th><th>throughput / capacity</th><th>composition</th><th>qualities</th></tr>';
  for(const p of pl.pools){
    const comp=Object.entries(p.composition).map(([c,s])=>c+' '+(100*s).toFixed(1)+'%').join(', ')||'idle';
    const qs=Object.entries(p.qualities).map(([q,v])=>q+' '+fmt(v)).join('; ');
    h+='<tr><td>'+esc(p.name)+'</td><td>'+fmt(p.throughput)+' / '+fmt(p.capacity)+(p.at_capacity?' <b>at capacity</b>':'')+'</td><td>'+esc(comp)+'</td><td>'+esc(qs)+'</td></tr>';}
  h+='</table><h2 style="margin-top:12px">products</h2><table><tr><th>product</th><th>volume</th><th>revenue</th><th>qualities (value, spec)</th></tr>';
  for(const p of pl.products) h+='<tr><td>'+esc(p.name)+'</td><td>'+fmt(p.volume)+'</td><td>'+fmt(p.revenue,2)+'</td><td>'+qualityCell(p.qualities)+'</td></tr>';
  h+='</table><h2 style="margin-top:12px">flows</h2><table><tr><th>from</th><th>to</th><th>quantity</th></tr>';
  for(const f of pl.flows) h+='<tr><td>'+esc(f.from)+'</td><td>'+esc(f.to)+'</td><td>'+fmt(f.quantity)+'</td></tr>';
  return h+'</table></div>';
}
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
  // null when the solve returned no point -- and null < 1e-6 is true in
  // JavaScript, so it has to be caught before it reads as "feasible"
  const nopt=(r.viol_row==null);
  const feas=!nopt&&(r.viol_row<1e-6&&r.viol_bound<1e-6&&r.viol_int<1e-6);
  return '<div class="card"><h2>'+esc(r.device)+' result</h2><table>'
   +'<tr><td>status</td><td class="'+s+'">'+esc(r.status)+'</td></tr>'
   +'<tr><td>objective</td><td class="big">'+ (r.objective==null?'-':r.objective.toPrecision(12)) +'</td></tr>'
   +(r.dual_bound!=null?'<tr><td>dual bound</td><td>'+r.dual_bound.toPrecision(10)+'</td></tr>':'')
   +(r.nodes?'<tr><td>nodes</td><td>'+r.nodes.toLocaleString()+'</td></tr>':'')
   +(r.iterations?'<tr><td>iterations</td><td>'+r.iterations.toLocaleString()+'</td></tr>':'')
   +'<tr><td>time</td><td>'+r.time.toFixed(3)+' s</td></tr>'
   +'<tr><td>method</td><td>'+esc(r.method)+'</td></tr>'
   +'<tr><td>independent check</td><td class="'+(feas?'ok':'bad')+'">'
     +(nopt?'no point returned':(feas?'feasible':'VIOLATION')+' — row '+r.viol_row.toExponential(1)
     +', bound '+r.viol_bound.toExponential(1)
     +', integrality '+r.viol_int.toExponential(1))+'</td></tr>'
   +'</table></div>';
}

$('go').onclick=async()=>{
  $('go').disabled=true;$('go').textContent='SOLVING…';
  const body={source:$('src').value,template:$('tmpl').value,
    size:+$('size').value,seed:+$('seed').value,mps:$('mps').value,
    device:$('dev').value,time_limit:+$('tl').value};
  let d;
  try{
    if(body.source==='file'){const f=await readBase64($('modelfile'));
      if(!f) throw new Error('choose a model file first'); body.filename=f.name; body.data_b64=f.data;}
    if(body.source==='blend'){body.components_csv=await readText($('compfile'));
      body.products_csv=await readText($('prodfile'));
      if(!body.components_csv||!body.products_csv) throw new Error('choose both CSV files first');
      body.prices_csv=await readText($('pricefile')); body.demand_csv=await readText($('demandfile'));
      body.capacity_csv=await readText($('capfile')); body.pools_csv=await readText($('poolfile'));}
    d=await (await fetch('/api/solve',{method:'POST',
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
  if(d.plan) html+=planCards(d.plan);

  if(d.results.length===2){
    const a=d.results[0],b=d.results[1];
    const f=a.time/b.time, wide=Math.max(a.time,b.time);
    html+='<div class="card"><h2>cpu vs gpu</h2>'
      +'<div>CPU '+esc(a.method)+' '+a.time.toFixed(3)+'s</div><div class="bar" style="background:var(--cpu);width:'+(100*a.time/wide)+'%"></div>'
      +'<div>GPU '+esc(b.method)+' '+b.time.toFixed(3)+'s</div><div class="bar" style="background:var(--gpu);width:'+(100*b.time/wide)+'%"></div>'
      +'<div class="muted">'+(a.method===b.method
          ?'both runs used the same engine, so there is no device comparison to make'
          :(f>=1?('GPU '+f.toFixed(2)+'x faster'):('CPU '+(1/f).toFixed(2)+'x faster — this model is too small to fill the GPU')))
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


_UPLOAD_SUFFIXES = (".mps", ".lp", ".qps", ".mps.gz", ".lp.gz", ".qps.gz",
                    ".mps.bz2", ".lp.bz2", ".qps.bz2", ".mps.xz", ".lp.xz", ".gz", ".bz2")


def _build(body):
    if body.get("source") == "file":
        # The browser sends the file's bytes as base64 with its name; the
        # name's suffix picks the reader, and a compression suffix is looked
        # through, exactly as the command line does. Written beside the page
        # as _upload.<suffix> (gitignored), never anywhere else.
        import base64
        name = os.path.basename(str(body.get("filename", "")))
        data = body.get("data_b64")
        if not name or not data:
            raise ValueError("no model file supplied")
        low = name.lower()
        suffix = next((sfx for sfx in _UPLOAD_SUFFIXES if low.endswith(sfx)), None)
        if suffix is None:
            raise ValueError(f"{name}: not a .mps, .lp or .qps file (optionally .gz/.bz2/.xz)")
        if suffix in (".gz", ".bz2"):
            suffix = ".mps" + suffix                 # a bare .gz is read as MPS
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_upload" + suffix)
        with open(path, "wb") as fh:
            fh.write(base64.b64decode(data))
        return read_model(path, name=name)

    if body.get("source") == "blend":
        comps = body.get("components_csv") or ""
        prods = body.get("products_csv") or ""
        if not comps.strip() or not prods.strip():
            raise ValueError("both components.csv and products.csv are needed")
        pools = body.get("pools_csv") or ""
        prices = body.get("prices_csv") or ""
        if pools.strip() and prices.strip():
            raise ValueError("pools and prices cannot be combined: pooled qualities over a "
                             "horizon is not a table-driven model yet")
        if pools.strip():
            from sovopt.models.tabular_pooling import parse_pooling_csv
            tables = parse_pooling_csv(comps, prods, pools)
            prob = tables.problem.linear          # the LP part; the bilinear model is on tables
        elif prices.strip():
            from sovopt.models.tabular_planning import parse_planning_csv
            tables = parse_planning_csv(comps, prods, prices, body.get("demand_csv") or None,
                                        body.get("capacity_csv") or None)
            prob = tables.problem
        else:
            from sovopt.models.tabular import parse_blending_csv
            tables = parse_blending_csv(comps, prods)
            prob = tables.problem
        prob.tables = tables                         # for the plan afterwards
        return prob

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
        last_sol = None
        tables = getattr(prob, "tables", None)
        if tables is not None and hasattr(tables, "pools"):
            # pooled qualities: one run, the spatial branch-and-bound on the
            # CPU, the bilinear identities checked with the rows
            from sovopt.globalopt.spatial import SpatialParams, solve_global
            t = time.perf_counter()
            sol = solve_global(tables.problem, SpatialParams(time_limit=tl))
            dt = time.perf_counter() - t
            rv, bv, iv = (prob.violation(sol.x) if sol.x is not None else (float("nan"),) * 3)
            bil = tables.problem.max_violation(sol.x) if sol.x is not None else float("nan")
            results.append({
                "device": "cpu", "status": sol.status.name,
                "objective": float(sol.objective) if np.isfinite(sol.objective) else None,
                "dual_bound": float(sol.dual_bound) if np.isfinite(sol.dual_bound) else None,
                "nodes": int(sol.nodes), "iterations": int(sol.iterations), "time": dt,
                "method": "spatial branch-and-bound (pq-formulation)",
                "viol_row": _finite(max(rv, bil) if np.isfinite(bil) else rv),
                "viol_bound": _finite(bv), "viol_int": _finite(iv), "history": [],
            })
            devices, last_sol = [], sol
        for dev in devices:
            # A small LP routes to the simplex whatever the device, so a
            # "GPU" run of it is the CPU simplex a second time -- and the
            # second time is faster because the first paid the JIT, which
            # the page then reported as a 19x GPU speed-up. The GPU run of
            # an LP is PDLP, the engine that actually runs there; a MILP's
            # tree takes the device through its node solver as before.
            method = "pdlp" if (dev == "gpu" and not prob.is_mip
                                and not prob.is_qp) else "auto"
            t = time.perf_counter()
            sol = solve(prob, method=method, device=dev, time_limit=tl)
            dt = time.perf_counter() - t
            if last_sol is None:
                last_sol = sol
            rv, bv, iv = (prob.violation(sol.x) if sol.x is not None
                          else (float("nan"),) * 3)
            hist = [[int(h[0]), float(h[1]), float(h[2]), float(h[3])]
                    for h in (sol.log or [])
                    if len(h) >= 4 and np.isfinite(h[1:4]).all()]
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
                "viol_row": _finite(rv), "viol_bound": _finite(bv), "viol_int": _finite(iv),
                "history": hist[-400:],
            })

        agree = 0.0
        if len(results) == 2 and results[0]["objective"] is not None \
                and results[1]["objective"] is not None:
            a, b = results[0]["objective"], results[1]["objective"]
            agree = abs(a - b) / max(1.0, abs(a))

        plan = None
        if tables is not None and last_sol is not None:
            # the plan in the planner's terms, from the first device's solve,
            # with the independent check on the model built from the tables
            if hasattr(tables, "pools"):
                from sovopt.models.tabular_pooling import pooling_plan as plan_fn
            elif hasattr(tables, "periods"):
                from sovopt.models.tabular_planning import planning_plan as plan_fn
            else:
                from sovopt.models.tabular import blend_plan as plan_fn
            plan = plan_fn(tables, last_sol)
            if last_sol.x is not None and Status(last_sol.status).has_solution:
                try:
                    from bench.verify import verify as _verify
                    v = _verify(prob, last_sol.x, last_sol.objective, feas_tol=1e-6,
                                y=(None if hasattr(tables, "pools") else last_sol.y))
                    extra = None
                    if hasattr(tables, "pools"):
                        bil = tables.problem.max_violation(last_sol.x)
                        extra = ("bilinear identities", bil <= 1e-6,
                                 f"max violation {bil:.3e}")
                    plan["verifier_verdict"], plan["verifier_detail"] = v.headline(extra)
                except ImportError:
                    pass

        return JSONResponse({
            "summary": prob.summary(),
            "coeff_ratio": sc.ratio_before,
            "scaled_ratio": sc.ratio_after,
            "scaling": sc.method,
            "results": results,
            "agreement": agree,
            "plan": plan,
        })
    except Exception:
        return JSONResponse({"error": traceback.format_exc(limit=4)})


def main():
    import uvicorn
    print("SOVOPT interface on http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")


if __name__ == "__main__":
    main()
