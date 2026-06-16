let STATE = { data: null, overrides: {}, horizon: "12M" };
let vsChart = null;

const fmtPct = v => v==null||isNaN(v) ? "—" : (v*100).toFixed(1)+"%";
const fmtNum = (v,d=2) => v==null||isNaN(v) ? "—" : Number(v).toFixed(d);
const cls = v => v==null||isNaN(v) ? "" : (v>=0?"pos":"neg");

async function loadPortfolio(){
  const status = document.getElementById('status');
  status.style.display='block'; status.textContent='Computing portfolio…';
  document.getElementById('app').style.display='none';
  try{
    const r = await fetch('/api/portfolio',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify(STATE.overrides)});
    const d = await r.json();
    if(!d.ok){ status.className='err'; status.textContent=d.reason||'Error'; return; }
    STATE.data = d;
    render();
    status.style.display='none';
    document.getElementById('app').style.display='block';
  }catch(e){ status.className='err'; status.textContent='Request failed: '+e; }
}

function render(){
  renderHoldings();
  renderRegime();
  renderRanked();
  populateStockPicker();
  loadVsChart();
}

function renderHoldings(){
  const d = STATE.data;
  const maxW = Math.max(...d.portfolio.map(h=>h.weight));
  document.getElementById('holdings').innerHTML = d.portfolio.map(h=>{
    const gold = h.kind==='gold';
    const w = (h.weight*100).toFixed(1);
    const bar = Math.max(2, h.weight/maxW*100);
    return `<div class="holding">
      <div class="tkr">${h.ticker}</div>
      <div class="nm">${h.name} ${gold?'<span class="tag gold">gold sleeve</span>':`<span class="tag">${h.sector}</span>`}</div>
      <div style="flex:1.2;max-width:160px"><div class="wbar ${gold?'gold':''}" style="width:${bar}%"></div></div>
      <div class="wt">${w}%</div>
    </div>`;
  }).join('');
}

function renderRegime(){
  const r = STATE.data.regime;
  const cur = r.current||'neutral';
  let html = `<span class="regime ${cur}">${cur.toUpperCase()}</span>
    <span class="small" style="margin-left:10px">since ${r.since||'—'} · gold sleeve ${(STATE.data.gold_weight*100).toFixed(0)}%</span>`;
  document.getElementById('regime').innerHTML = html;
  const stats = r.stats||{};
  document.getElementById('regime-stats').innerHTML = Object.keys(stats).map(k=>`
    <div class="stat"><div class="k">${k}</div>
      <div class="v">${fmtPct(stats[k].ann_return)}</div>
      <div class="small">vol ${fmtPct(stats[k].ann_vol)} · ${stats[k].weeks}w</div></div>`).join('')
    || '<span class="small">Regime stats unavailable.</span>';
}

function renderRanked(){
  const d = STATE.data, h = STATE.horizon;
  const bench = (d.bench_returns||{})[h];
  // Re-rank by the selected interval's trailing return (nulls sink to the bottom).
  const sorted = d.ranked.slice().sort((a,b)=>{
    const ra=(a.returns||{})[h], rb=(b.returns||{})[h];
    if(ra==null && rb==null) return 0;
    if(ra==null) return 1;
    if(rb==null) return -1;
    return rb-ra;
  });
  const rows = sorted.map((row,i)=>{
    const ret = (row.returns||{})[h];
    const vs = (ret!=null && bench!=null) ? ret-bench : null;
    const m = row.metrics||{};
    return `<tr>
      <td>${i+1}</td><td><b>${row.ticker}</b></td>
      <td style="text-align:left;max-width:220px;overflow:hidden;text-overflow:ellipsis">${row.name}</td>
      <td style="text-align:left"><span class="tag">${row.sector}</span></td>
      <td>${fmtNum(row.composite,2)}</td>
      <td class="${cls(ret)}">${fmtPct(ret)}</td>
      <td class="${cls(vs)}">${vs==null?'—':(vs>=0?'+':'')+fmtPct(vs)}</td>
      <td>${fmtPct(m.roic)}</td><td>${fmtNum(m.pe,1)}</td>
      <td>${fmtNum(m.sharpe,2)}</td><td>${fmtPct(m.volatility)}</td>
      <td>${row.screens_passed}/${row.screens_total}</td></tr>`;
  }).join('');
  document.querySelector('#ranked tbody').innerHTML = rows;
  const title = document.getElementById('ranked-title');
  if(title) title.innerHTML = `Ranked Candidates <span class="small" style="font-weight:400">— sorted by ${h} return</span>`;
}

function populateStockPicker(){
  const sel = document.getElementById('stockpick');
  const prev = sel.value;
  sel.innerHTML = STATE.data.portfolio.filter(h=>h.kind==='stock')
    .map(h=>`<option value="${h.company_key}">${h.ticker} — ${h.name}</option>`).join('')
    + STATE.data.ranked.slice(0,40).map(r=>`<option value="${r.company_key}">${r.ticker} — ${r.name}</option>`).join('');
  if(prev) sel.value = prev;
}

async function loadVsChart(){
  const ck = document.getElementById('stockpick').value;
  const win = document.getElementById('vswindow').value;
  if(!ck) return;
  const r = await fetch('/api/stock_vs_sp500?company_key='+encodeURIComponent(ck));
  const d = await r.json();
  if(!d.ok){ document.getElementById('vsSummary').textContent=d.reason||'No data'; return; }
  const c = (d.curves||{})[win];
  if(!c){ document.getElementById('vsSummary').textContent='No data for this window.'; return; }
  const ctx = document.getElementById('vsChart');
  if(vsChart) vsChart.destroy();
  const scale = (document.getElementById('vsscale')||{}).value || 'logarithmic';
  const opts = chartOpts('Rebased to 100 · total return');
  opts.scales.y.type = scale;
  if(scale==='logarithmic'){
    // clean log ticks instead of Chart.js defaults
    opts.scales.y.ticks.callback = v => Number(v).toLocaleString();
    opts.scales.y.afterBuildTicks = ax => { ax.ticks = [50,100,200,400,800,1600,3200].filter(t=>t>=ax.min&&t<=ax.max).map(t=>({value:t})); };
  }
  vsChart = new Chart(ctx,{type:'line',data:{labels:c.dates,datasets:[
    {label:d.ticker,data:c.stock,borderColor:'#3b82f6',borderWidth:2,pointRadius:0,tension:.1},
    {label:'S&P 500 (TR)',data:c.sp500,borderColor:'#e6edf3',borderWidth:2,pointRadius:0,tension:.1}
  ]},options:opts});
  const out = c.stock_return-c.sp500_return;
  document.getElementById('vsSummary').innerHTML =
    `<b>${d.ticker}</b> ${fmtPct(c.stock_return)} vs S&P 500 ${fmtPct(c.sp500_return)} over ${win} — `
    +`<span class="${cls(out)}">${out>=0?'+':''}${fmtPct(out)}</span> excess.`;
}

function chartOpts(yLabel){
  return {responsive:true,maintainAspectRatio:false,interaction:{mode:'index',intersect:false},
    plugins:{legend:{labels:{color:'#8b949e'}}},
    scales:{x:{ticks:{color:'#8b949e',maxTicksLimit:8},grid:{color:'#21262d'}},
      y:{ticks:{color:'#8b949e'},grid:{color:'#21262d'},title:{display:!!yLabel,text:yLabel,color:'#8b949e'}}}};
}

// ---- equation editor (rendered as real, explained formulas) ----
let EQSCHEMA = null;

function thrInput(k, f){
  const shown = +(f.threshold * (f.display_scale||1)).toPrecision(6);
  const step = (f.display_scale||1) > 1 ? '0.1' : 'any';
  return `<input class="num thr" data-f="${k}" data-scale="${f.display_scale||1}" `
    + `type="number" step="${step}" value="${shown}">`;
}
function wtInput(k, f){
  return `<input class="num wt" data-f="${k}" type="number" step="0.1" min="0" value="${f.weight}">`;
}

// Build the composite equation showing ONLY active (weight>0) factors.
function renderComposite(){
  const s = EQSCHEMA; if(!s) return;
  const w = {};
  Object.keys(s.factors).forEach(k=> w[k] = s.factors[k].weight);
  document.querySelectorAll('#equations .eq .wt').forEach(inp=> w[inp.dataset.f] = parseFloat(inp.value)||0);
  const active = Object.keys(s.factors).filter(k => (w[k]||0) > 0);
  const el = document.getElementById('eq-composite');
  if(active.length===0){
    el.innerHTML = '<div class="eq-title">Composite score</div><div class="eq-note">No factors active — set a weight &gt; 0 in a card below.</div>';
    return;
  }
  const numer = active.map((k,i)=>{
    const f = s.factors[k];
    const sign = f.higher_better ? '+' : '&minus;';
    const term = `<span class="term"><span class="wnum">${(+w[k]).toFixed(2)}</span>`
         + `<span class="op">·</span><span class="zfn">z(<i>${f.symbol}</i>)</span></span>`;
    return (i===0 ? (f.higher_better?'':'<span class="op">&minus;</span>') : `<span class="op">${sign}</span>`) + term;
  }).join('');
  el.innerHTML =
    `<div class="eq-title">Composite score <span class="small">— active factors only</span></div>
     <div class="eqline big">
       <i>Score</i> <span class="op">=</span>
       <span class="frac"><span class="numer">${numer}</span>
       <span class="denom">&Sigma; w<sub>i</sub></span></span>
     </div>
     <div class="eq-note">Lower-is-better factors are <b>subtracted</b>. Edit weights in the cards below — set a weight &gt; 0 to add a factor, 0 to drop it. Weights normalize to sum to 1.</div>`;
}

async function loadEquations(){
  const r = await fetch('/api/equations');
  EQSCHEMA = await r.json();
  const s = EQSCHEMA, keys = Object.keys(s.factors);

  // Factor cards: each has a weight AND threshold input (the editor)
  document.getElementById('equations').innerHTML = keys.map(k=>{
    const f = s.factors[k];
    const opSym = f.op === '>=' ? '&ge;' : (f.op === '<=' ? '&le;' : '=');
    const unit = f.display_unit ? ` <span class="unit">${f.display_unit}</span>` : '';
    const off = (f.weight>0) ? '' : ' inactive';
    return `<div class="eq${off}" data-f="${k}">
      <div class="eq-head"><span class="sym">${f.symbol}</span><span class="lbl">${f.label}</span></div>
      <div class="eqline def">${f.formula||''}</div>
      <div class="eqline screen">
        <i>${f.symbol}</i> <span class="op">${opSym}</span> ${thrInput(k,f)}${unit}
      </div>
      <div class="eq-wrow"><span class="small">weight in composite</span> ${wtInput(k,f)}</div>
      <div class="desc">${f.explain}</div>
    </div>`;
  }).join('');

  // 3) Definitions (read-only reference equations)
  const w = (s.settings.winsor_pct*100).toFixed(0);
  document.getElementById('eq-defs').innerHTML = `
    <div class="def-row"><span class="dfn"><i>z</i>(x) <span class="op">=</span>
      <span class="frac"><span class="numer">x &minus; &mu;<sub>sector</sub></span>
      <span class="denom">&sigma;<sub>sector</sub></span></span></span>
      <span class="small">Sector-neutral z-score — standardized within each GICS sector so you bet on the best name in a sector, not just the cheapest sector.</span></div>
    <div class="def-row"><span class="dfn">winsorize at [${w}%, ${100-w}%]</span>
      <span class="small">Each factor is clipped at its ${w}th / ${100-w}th percentile before scoring, so a single outlier can't dominate.</span></div>
    <div class="def-row"><span class="dfn"><i>r&#772;</i>, <i>&sigma;<sub>r</sub></i>, <i>r<sub>f</sub></i></span>
      <span class="small">Mean daily return, its standard deviation, and the daily risk-free rate; &radic;252 annualizes. <i>&sigma;<sub>down</sub></i> uses only negative-return days.</span></div>
    <div class="def-row"><span class="dfn">Gold sleeve weight = f(regime)</span>
      <span class="small">An HMM on S&P 500 returns picks bull / neutral / bear; gold weight rises in risk-off regimes (8% / 14% / 25%).</span></div>`;

  // live-sync: editing a card weight updates the composite equation + active state
  document.querySelectorAll('#equations .eq .wt').forEach(inp=>{
    inp.addEventListener('input',()=>{
      inp.closest('.eq').classList.toggle('inactive', !(parseFloat(inp.value)>0));
      renderComposite();
    });
  });
  renderComposite();
}

function collectOverrides(){
  const factors = {};
  document.querySelectorAll('#equations .eq').forEach(div=>{
    const k = div.dataset.f;
    const t = div.querySelector('.thr');
    const wv = div.querySelector('.wt');
    factors[k] = {
      threshold: parseFloat(t.value)/(parseFloat(t.dataset.scale)||1),
      weight: parseFloat(wv.value)||0,
    };
  });
  return {factors};
}

document.addEventListener('DOMContentLoaded',()=>{
  loadEquations();
  loadPortfolio();
  document.getElementById('horizon').addEventListener('change',e=>{STATE.horizon=e.target.value;renderRanked();});
  document.getElementById('stockpick').addEventListener('change',loadVsChart);
  document.getElementById('vswindow').addEventListener('change',loadVsChart);
  document.getElementById('vsscale').addEventListener('change',loadVsChart);
  document.getElementById('apply').addEventListener('click',()=>{
    STATE.overrides=collectOverrides();
    document.getElementById('eqnote').textContent='Recomputing…';
    loadPortfolio().then(()=>document.getElementById('eqnote').textContent='Updated.');
  });
  document.getElementById('reset').addEventListener('click',()=>{
    STATE.overrides={}; loadEquations(); loadPortfolio();
    document.getElementById('eqnote').textContent='Reset to defaults.';
  });
});
