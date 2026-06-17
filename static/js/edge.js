const fmtPct = v => v==null||isNaN(v) ? "—" : (v*100).toFixed(1)+"%";
const fmtN = (v,d=2) => v==null||isNaN(v) ? "—" : Number(v).toFixed(d);
const cls = v => v==null||isNaN(v) ? "" : (v>=0?"pos":"neg");
const fmtUSD = n => n==null||isNaN(n) ? "—" : "$"+Math.round(n).toLocaleString('en-US');

const _O = loadEdgeOpts();    // shared with the Edge Tracker (localStorage)
let WINDOW = _O.window;
let HOLD = _O.hold;
let MIX = _O.mix;             // growth mix: share of basket from >=15%-rev-growth names
let AMOUNT = 10000;           // starting capital for the dollar-value view (client-side only)
let LAST = null;              // last backtest response, so the amount can re-render without refetch
const CHARTS = {};
function mkChart(id,cfg){ if(CHARTS[id]) CHARTS[id].destroy(); CHARTS[id]=new Chart(document.getElementById(id),cfg); }

function opts(yLabel){
  return {responsive:true,maintainAspectRatio:false,interaction:{mode:'index',intersect:false},
    plugins:{legend:{labels:{color:'#8b949e'}}},
    scales:{x:{ticks:{color:'#8b949e',maxTicksLimit:8},grid:{color:'#21262d'}},
      y:{type:'logarithmic',ticks:{color:'#8b949e'},grid:{color:'#21262d'},
        title:{display:!!yLabel,text:yLabel,color:'#8b949e'}}}};
}

async function run(){
  const status=document.getElementById('status');
  status.className='loading'; status.style.display='block';
  status.textContent='Running the Edge backtest…';
  document.getElementById('app').style.display='none';
  try{
    const r = await fetch('/api/edge_backtest',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({window:WINDOW, hold:HOLD, mix:MIX})});
    const d = await safeJson(r);
    if(!d.ok){ status.className='err'; status.textContent=d.reason||'Backtest failed'; return; }
    LAST=d; render(d);
    status.style.display='none';
    document.getElementById('app').style.display='block';
    const note=document.getElementById('windownote');
    if(note) note.textContent=`${d.n_rebalances} rebalances · ${(d.period||[]).join(' → ')}`;
  }catch(e){ status.className='err'; status.textContent='Request failed: '+e; }
}

function render(d){
  const p=d.performance||{}, m=p.model||{}, t=d.turnover||{};
  const alpha=(m.cagr??0)-((p.sp500||{}).cagr??0);
  document.getElementById('headline').innerHTML = [
    ['CAGR (net)', fmtPct(m.cagr), cls(m.cagr), 'Compound annual growth rate, after ~10bps trading costs'],
    ['Sharpe', fmtN(m.sharpe,2), '', 'Return per unit of total risk — higher is better (>1 is strong)'],
    ['Sortino', fmtN(m.sortino,2), '', 'Return per unit of downside risk — like Sharpe but only losses count as risk'],
    ['Max drawdown', fmtPct(m.max_drawdown), 'neg', 'Worst peak-to-trough loss over the window'],
    ['Total return', fmtPct(m.total_return), cls(m.total_return), 'Cumulative growth over the window'],
    ['Excess vs S&P', (alpha>=0?'+':'')+fmtPct(alpha), cls(alpha), 'Annualized return above the S&P 500'],
    ['Turnover / yr', fmtN(t.annualized,1)+'×', '', 'How many times the basket fully turns over per year'],
  ].map(([k,v,c,tip])=>`<div class="stat" title="${tip||''}"><div class="k">${k}</div><div class="v ${c}">${v}</div></div>`).join('');

  const c=d.curves;
  const amt = AMOUNT>0 ? AMOUNT : 1;                 // dollar multiplier (curves are growth-of-$1)
  const scl = a => (a||[]).map(x=>+(x*amt).toFixed(2));
  mkChart('edgeCurve',{type:'line',data:{labels:c.dates,datasets:[
    {label:'The Edge',data:scl(c.model),borderColor:'#d4af37',borderWidth:2,pointRadius:0,tension:.1},
    {label:'S&P 500 (TR)',data:scl(c.sp500),borderColor:'#e6edf3',borderWidth:2,pointRadius:0,tension:.1},
    {label:'Nasdaq-100 (TR)',data:scl(c.nasdaq),borderColor:'#22c55e',borderWidth:2,pointRadius:0,tension:.1},
  ]},options:opts('Value of $'+amt.toLocaleString('en-US')+' (log)')});
  const il=document.getElementById('investLabel'); if(il) il.textContent=amt.toLocaleString('en-US');
  const endv = k => (c[k]&&c[k].length) ? c[k][c[k].length-1]*amt : null;
  const ev=document.getElementById('endvals');
  if(ev) ev.innerHTML = `<b>$${amt.toLocaleString('en-US')}</b> over ${(d.period||[]).join(' → ')} becomes `
    +`<b style="color:#d4af37">${fmtUSD(endv('model'))}</b> in the Edge `
    +`· <b>${fmtUSD(endv('sp500'))}</b> S&amp;P · <b>${fmtUSD(endv('nasdaq'))}</b> Nasdaq`;

  document.querySelector('#winperf tbody').innerHTML = (d.windows||[]).map(w=>
    `<tr><td>${w.w}</td><td class="${cls(w.cagr)}">${fmtPct(w.cagr)}</td><td>${fmtN(w.sharpe,2)}</td>
     <td class="neg">${fmtPct(w.dd)}</td><td class="${cls(w.excess)}">${(w.excess>=0?'+':'')+fmtPct(w.excess)}</td></tr>`).join('');

  const rows=[['The Edge','model'],['S&P 500 (TR)','sp500'],['Nasdaq-100 (TR)','nasdaq']].map(([nm,k])=>{
    const x=p[k]||{};
    return `<tr><td>${nm}</td><td class="${cls(x.total_return)}">${fmtPct(x.total_return)}</td>
      <td class="${cls(x.cagr)}">${fmtPct(x.cagr)}</td><td>${fmtN(x.sharpe,2)}</td><td>${fmtN(x.sortino,2)}</td>
      <td class="neg">${fmtPct(x.max_drawdown)}</td></tr>`;}).join('');
  document.querySelector('#perf tbody').innerHTML=rows;

  const wsel=(d.windows||[]).find(w=>w.w===WINDOW)||{};
  const ex=wsel.excess;
  document.getElementById('turnover').innerHTML = [
    ['Turnover / rebalance', fmtPct(t.per_rebalance), '', 'Fraction of the basket replaced each rebalance'],
    ['Turnover / year', fmtN(t.annualized,1)+'×', '', 'Annualized portfolio turnover'],
    ['Cost assumption', t.cost_bps+'bps', '', 'One-way trading cost charged on turnover'],
    ['Gross → Net CAGR', fmtPct(t.gross_cagr)+' → '+fmtPct(t.net_cagr), '', 'Return before vs after trading costs'],
    ['Alpha vs S&P', (ex==null||isNaN(ex)?'—':(ex>=0?'+':'')+fmtPct(ex)), cls(ex), 'Excess return over the S&P for the selected window'],
  ].map(([k,v,c,tip])=>`<div class="stat" title="${tip||''}"><div class="k">${k}</div><div class="v ${c}">${v}</div></div>`).join('');

  const pd=d.period_detail||{}, pm=Math.round((pd.hold_days||21)/21);
  const best=pd.best, worst=pd.worst;
  document.getElementById('periodstats').innerHTML = [
    ['Best period', best?((best.ret>=0?'+':'')+fmtPct(best.ret)):'—', cls(best&&best.ret),
      best?('Strongest single ~'+pm+'-month hold · opened '+best.date):''],
    ['Worst period', worst?fmtPct(worst.ret):'—', 'neg',
      worst?('Weakest single ~'+pm+'-month hold · opened '+worst.date):''],
    ['Pre-tax return', fmtPct(pd.pretax_total)+(amt>1?(' · '+fmtUSD(amt*(pd.pretax_total||0))):''), cls(pd.pretax_total),
      'Total return over the window, net of costs but BEFORE taxes. This is a high-turnover, short-hold strategy, so gains are mostly short-term — after-tax is materially lower.'],
  ].map(([k,v,c,tip])=>`<div class="stat" title="${tip||''}"><div class="k">${k}</div><div class="v ${c}">${v}</div></div>`).join('');

  const s=d.spec||{};
  document.getElementById('costnote').textContent = (s.cost_bps||10)+'bps';
  document.getElementById('spec').innerHTML = [
    ['Signal', s.signal||'acceleration'],
    ['Holding', Math.round((s.hold_days||42)/21)+'M (~'+(s.hold_days||42)+' trading days)'],
    ['Basket', (s.n||10)+' names'],
    ['Liquidity floor', '≥ $'+fmtN(s.mcap_floor_bn,0)+'B'],
    ['Correlation cap', fmtN(s.corr_cap,2)],
    ['Regime below 200dMA', fmtN(s.regime_expo*100,0)+'% invested'],
    ['Growth mix', (s.growth_mix>0 ? fmtN(s.growth_mix*100,0)+'% from ≥'+fmtN((s.growth_thresh||0.15)*100,0)+'% growth names' : 'off (pure momentum)')],
  ].map(([k,v])=>`<div class="stat"><div class="k">${k}</div><div class="v" style="font-size:15px">${v}</div></div>`).join('');

  document.getElementById('caveat').innerHTML =
    '<b>Read these before quoting any number.</b> Returns are net of '+(s.cost_bps||10)+'bps trading costs, '
    +'but the absolute CAGR is still optimistic for three reasons: <b>(1) survivorship</b> — the universe is missing '
    +'~89% of companies that delisted, worth roughly −3%/yr; <b>(2)</b> the liquidity floor uses market cap as a '
    +'full-history proxy (real point-in-time Russell-1000 + dollar-volume data needed); <b>(3)</b> the recent window '
    +'sits in an unusually momentum-friendly regime. The <i>trustworthy</i> results are the risk-adjusted shape — '
    +'Sharpe ≈0.9 and the drawdown control (−27% vs the S&amp;P −44%), all-weather behavior, and positive return skew — '
    +'not the in-sample CAGR. After honest haircuts (survivorship + the lumpy momentum tailwind), a defensible '
    +'<b>forward expectation is ~11–13%/yr, ~3–5 pts over the S&amp;P</b>. '
    +'Next credibility fixes: point-in-time data (Norgate) and higher cost assumptions.';
}

document.addEventListener('DOMContentLoaded',()=>{
  syncEdgeBtns();   // reflect options carried over from the Edge Tracker
  document.querySelectorAll('#windowbtns button').forEach(b=>{
    b.addEventListener('click',()=>{
      document.querySelectorAll('#windowbtns button').forEach(x=>x.classList.remove('active'));
      b.classList.add('active'); WINDOW=b.dataset.w; saveEdgeOpt('window',WINDOW); run();
    });
  });
  document.querySelectorAll('#holdbtns button').forEach(b=>{
    b.addEventListener('click',()=>{
      document.querySelectorAll('#holdbtns button').forEach(x=>x.classList.remove('active'));
      b.classList.add('active'); HOLD=+b.dataset.h; saveEdgeOpt('hold',HOLD); run();
    });
  });
  document.querySelectorAll('#mixbtns button').forEach(b=>{
    b.addEventListener('click',()=>{
      document.querySelectorAll('#mixbtns button').forEach(x=>x.classList.remove('active'));
      b.classList.add('active'); MIX=+b.dataset.m; saveEdgeOpt('mix',MIX); run();
    });
  });
  const amt=document.getElementById('amount');
  if(amt){
    AMOUNT = Math.max(0, parseFloat(amt.value)||0);   // honor the HTML default
    amt.addEventListener('input',()=>{
      AMOUNT = Math.max(0, parseFloat(String(amt.value).replace(/[^0-9.]/g,''))||0);
      if(LAST) render(LAST);                           // pure client re-scale, no refetch
    });
  }
  run();
});
