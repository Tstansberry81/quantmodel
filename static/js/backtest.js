const fmtPct = v => v==null||isNaN(v) ? "—" : (v*100).toFixed(1)+"%";
const fmtN = (v,d=2) => v==null||isNaN(v) ? "—" : Number(v).toFixed(d);
const cls = v => v==null||isNaN(v) ? "" : (v>=0?"pos":"neg");

let WINDOW = "MAX";
let HOLD = "1M";
let SCHEME = "equal";
let REGIME = false;
const HOLD_LABEL = {"1W":"weekly","2W":"biweekly","1M":"monthly","3M":"quarterly","6M":"semiannual","12M":"annual"};
const CHARTS = {};
function mkChart(id, cfg){ if(CHARTS[id]) CHARTS[id].destroy(); CHARTS[id]=new Chart(document.getElementById(id),cfg); }

async function run(){
  const status=document.getElementById('status');
  status.className='loading'; status.style.display='block';
  status.textContent='Running backtest…';
  document.getElementById('app').style.display='none';
  try{
    const r = await fetch('/api/backtest',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({window:WINDOW, hold:HOLD,
        settings:{weight_scheme:SCHEME, regime_overlay:REGIME}})});
    const d = await r.json();
    if(!d.ok){ status.className='err'; status.textContent=d.reason||'Backtest failed'; return; }
    if(!d.n_rebalances){ status.className='err';
      status.textContent='Not enough rebalances in this window — pick a longer window or shorter holding period.'; return; }
    render(d);
    status.style.display='none';
    document.getElementById('app').style.display='block';
    const note=document.getElementById('windownote');
    if(note) note.textContent=`${d.n_rebalances} ${HOLD_LABEL[d.hold]||''} rebalances of ${d.n_holdings} stocks · ${(d.period||[]).join(' → ')}`;
  }catch(e){ status.className='err'; status.textContent='Request failed: '+e; }
}

function render(d){
  const ic=d.ic||{};
  const perf=d.performance||{};
  const alpha=((perf.model||{}).cagr??0)-((perf.sp500||{}).cagr??0);
  const alphaCls = alpha>=0?'pos':'neg';
  document.getElementById('headline').innerHTML = [
    ['Mean IC', fmtN(ic.mean,3), ''],
    ['IC t-stat', fmtN(ic.tstat,2), ''],
    ['IC hit rate', fmtPct(ic.hit_rate), ''],
    ['Decile staircase', fmtN(d.monotonicity,2)+' / 1.0', ''],
    ['Top−Bottom', fmtPct(d.tb_spread), ''],
    ['Rebalances', d.n_rebalances, ''],
    ['Alpha vs S&P', (alpha>=0?'+':'')+fmtPct(alpha), alphaCls],
  ].map(([k,v,c])=>`<div class="stat"><div class="k">${k}</div><div class="v ${c}">${v}</div></div>`).join('');

  const c=d.curves;
  mkChart('curveChart',{type:'line',data:{labels:c.dates,datasets:[
    {label:'Model',data:c.model,borderColor:'#3b82f6',borderWidth:2,pointRadius:0,tension:.1},
    {label:'S&P 500 (TR)',data:c.sp500,borderColor:'#e6edf3',borderWidth:2,pointRadius:0,tension:.1},
    {label:'Nasdaq-100 (TR)',data:c.nasdaq,borderColor:'#22c55e',borderWidth:2,pointRadius:0,tension:.1},
  ]},options:opts('Growth of $1')});

  const dec=d.deciles||{};
  const labels=Object.keys(dec).sort((a,b)=>a-b);
  mkChart('decileChart',{type:'bar',data:{labels:labels.map(x=>'D'+x),
    datasets:[{data:labels.map(k=>dec[k]==null?null:dec[k]*100),
      backgroundColor:labels.map(k=> (dec[k]||0)>=0 ? '#3b82f6':'#ef4444')}]},
    options:Object.assign(opts('Mean fwd return %'),{plugins:{legend:{display:false}}})});

  const ics=d.ic_series||[];
  mkChart('icChart',{type:'line',data:{labels:ics.map(x=>x.date),
    datasets:[{data:ics.map(x=>x.ic),borderColor:'#d4af37',borderWidth:1.5,pointRadius:0,tension:.1,fill:true,
      backgroundColor:'rgba(212,175,55,.10)'}]},
    options:Object.assign(opts('IC'),{plugins:{legend:{display:false}}})});

  const p=d.performance||{};
  const rows=[['Model','model'],['S&P 500 (TR)','sp500'],['Nasdaq-100 (TR)','nasdaq']].map(([nm,k])=>{
    const m=p[k]||{};
    return `<tr><td>${nm}</td><td class="${cls(m.total_return)}">${fmtPct(m.total_return)}</td>
      <td class="${cls(m.cagr)}">${fmtPct(m.cagr)}</td><td>${fmtN(m.sharpe,2)}</td>
      <td class="neg">${fmtPct(m.max_drawdown)}</td></tr>`;}).join('');
  document.querySelector('#perf tbody').innerHTML=rows;

  const t=d.turnover||{};
  document.getElementById('turnover').innerHTML = [
    ['Turnover / rebalance', fmtPct(t.per_rebalance)],
    ['Turnover / year', fmtN((t.annualized||0),1)+'×'],
    ['Cost drag @10bps', fmtPct(t.est_cost_drag)],
    ['Net CAGR (after costs)', fmtPct(t.net_cagr_10bps)],
  ].map(([k,v])=>`<div class="stat"><div class="k">${k}</div><div class="v">${v}</div></div>`).join('');

  const pool=d.pool||{};
  document.getElementById('caveat').innerHTML =
    `<b>Universe:</b> rebuilt each rebalance as the top ${pool.bt_universe_size||''} names by `
    +`<i>point-in-time</i> market cap (avg ${pool.avg_cross_section||'?'} names/period), `
    +`including ${pool.n_inactive||0} delisted/inactive names — so we hold what was actually large then, not today's giants.<br>`
    +'<b>Residual survivorship bias:</b> fiscal.ai carries only ~211 inactive names (mostly recent M&A), so long-dead '
    +'companies (Enron, Lehman, etc.) are absent. This <i>reduces</i> but does not <i>eliminate</i> survivorship bias — '
    +'treat absolute returns as still somewhat optimistic.<br>'
    +'<b>Turnover:</b> a momentum strategy trades a lot — see turnover above. The net-CAGR assumes 10bps '
    +'round-trip cost (optimistic for retail); at 20–30bps the monthly edge shrinks but stays positive.<br>'
    +'<b>Other:</b> fundamentals lagged 90d to filing (signal is point-in-time); the HMM gold overlay is fit on the '
    +'full sample (mild lookahead).';
}

function opts(yLabel){
  return {responsive:true,maintainAspectRatio:false,interaction:{mode:'index',intersect:false},
    plugins:{legend:{labels:{color:'#8b949e'}}},
    scales:{x:{ticks:{color:'#8b949e',maxTicksLimit:8},grid:{color:'#21262d'}},
      y:{ticks:{color:'#8b949e'},grid:{color:'#21262d'},title:{display:!!yLabel,text:yLabel,color:'#8b949e'}}}};
}

document.addEventListener('DOMContentLoaded',()=>{
  document.querySelectorAll('#windowbtns button').forEach(b=>{
    b.addEventListener('click',()=>{
      document.querySelectorAll('#windowbtns button').forEach(x=>x.classList.remove('active'));
      b.classList.add('active'); WINDOW=b.dataset.w; run();
    });
  });
  document.querySelectorAll('#holdbtns button').forEach(b=>{
    b.addEventListener('click',()=>{
      document.querySelectorAll('#holdbtns button').forEach(x=>x.classList.remove('active'));
      b.classList.add('active'); HOLD=b.dataset.h; run();
    });
  });
  document.querySelectorAll('#schemebtns button').forEach(b=>{
    b.addEventListener('click',()=>{
      document.querySelectorAll('#schemebtns button').forEach(x=>x.classList.remove('active'));
      b.classList.add('active'); SCHEME=b.dataset.s; run();
    });
  });
  document.querySelectorAll('#regimebtns button').forEach(b=>{
    b.addEventListener('click',()=>{
      document.querySelectorAll('#regimebtns button').forEach(x=>x.classList.remove('active'));
      b.classList.add('active'); REGIME=(b.dataset.r==='on'); run();
    });
  });
  run();
});
