const fmtPct = v => v==null||isNaN(v) ? "—" : (v*100).toFixed(1)+"%";
const fmtN = (v,d=2) => v==null||isNaN(v) ? "—" : Number(v).toFixed(d);
const cls = v => v==null||isNaN(v) ? "" : (v>=0?"pos":"neg");
const fmtUSD = n => n==null||isNaN(n) ? "—" : "$"+Math.round(n).toLocaleString('en-US');

const _O = loadEdgeOpts();    // shared with the Edge Tracker (localStorage)
let WINDOW = _O.window;
let HOLD = _O.hold;
let NSIZE = _O.n;             // basket size 5..10 (product = 10; research favors 7)
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
      body:JSON.stringify({window:WINDOW, hold:HOLD, n:NSIZE})});
    const d = await safeJson(r);
    if(!d.ok){ status.className='err'; status.textContent=d.reason||'Backtest failed'; return; }
    LAST=d; render(d);
    status.style.display='none';
    document.getElementById('app').style.display='block';
    const note=document.getElementById('windownote');
    // "curve", not a bare date range. The END is a rebalance date but the START
    // is just where the window slice begins (N trading days back), so printing
    // them together read as though both were rebalance dates — under a header
    // that promises rebalances are always the first trading day.
    if(note) note.textContent=`${d.n_rebalances} rebalances · curve ${(d.period||[]).join(' → ')}`;
  }catch(e){ status.className='err'; status.textContent='Request failed: '+e; }
}

function render(d){
  const p=d.performance||{}, m=p.model||{}, t=d.turnover||{};
  const alpha=(m.cagr??0)-((p.sp500||{}).cagr??0);
  // Lead with the risk-adjusted SHAPE (the trustworthy part); CAGR/total return
  // Survivorship has been measured (~0.2pts) rather than flagged as unknown.
  document.getElementById('headline').innerHTML = [
    ['Sharpe', fmtN(m.sharpe,2), '', 'Return per unit of total risk — higher is better (>1 is strong). Note: √252·mean/σ, with NO risk-free subtraction, applied identically to the benchmarks.'],
    ['Max drawdown', fmtPct(m.max_drawdown), 'neg', 'Worst peak-to-trough loss, measured on the daily curve. Shallower than the S&P over the same history, but this is a concentrated 10-stock momentum book and the gap is not large — read it alongside the caveats.'],
    ['Sortino', fmtN(m.sortino,2), '', 'Return per unit of downside risk — like Sharpe but only losses count as risk'],
    ['Excess vs S&P', (alpha>=0?'+':'')+fmtPct(alpha), cls(alpha), 'Annualized return above the S&P 500, on survivorship-free history.'],
    ['CAGR (net)', fmtPct(m.cagr), cls(m.cagr), 'Compound annual growth after ~10bps costs, on survivorship-free history. Still a BACKTEST — the rules were chosen knowing how this period turned out (see caveats below).'],
    ['Total return', fmtPct(m.total_return), cls(m.total_return), 'Cumulative growth over the window. Same backtest caveat as CAGR.'],
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
    ['Signal', s.signal||'12-1 momentum'],
    ['Holding', Math.round((s.hold_days||21)/21)+'M (~'+(s.hold_days||21)+' trading days, first trading day of the month)'],
    ['Basket', (s.n||10)+' names'],
    ['Liquidity floor', '≥ $'+fmtN(s.mcap_floor_bn,0)+'B'],
    ['Sector cap', (s.sector_cap ? 'max '+s.sector_cap+' per sector' : 'off')],
    // Read from the spec the backtest returned, never hardcoded: this row states
    // which screens ACTUALLY ran, so it cannot describe a model that isn't
    // running the way the signal label once said "acceleration" for months.
    ['Solvency screens', [
        (s.fcf_positive ? 'FCF margin > 0' : null),
        (s.debt_ebitda_max ? 'debt ≤ '+fmtN(s.debt_ebitda_max,0)+'× EBITDA' : null),
      ].filter(Boolean).join(' · ') || 'off'],
    ['Regime below 200dMA', fmtN(s.regime_expo*100,0)+'% invested'+(s.continuous_regime?' · judged daily':'')],
    ['Volatility target', (s.vol_target ? fmtN(s.vol_target*100,0)+'% annualized ('+(s.vol_lookback||21)+'d)' : 'off')],
  ].map(([k,v])=>`<div class="stat"><div class="k">${k}</div><div class="v" style="font-size:15px">${v}</div></div>`).join('');

  const ddm=fmtPct(m.max_drawdown), dds=fmtPct((p.sp500||{}).max_drawdown), shp=fmtN(m.sharpe,2);
  const spshp=fmtN((p.sp500||{}).sharpe,2);
  document.getElementById('caveat').innerHTML =
    '<b>Read this before quoting any number.</b> Returns are net of '+(s.cost_bps||10)+'bps trading costs, '
    +'including the cost of changing exposure. '
    +'<b>Survivorship is no longer the headline risk here — it has been measured.</b> The price history is '
    +'survivorship-free: 7,851 delisted companies, 65% of the dataset, spread evenly across every year since 1999, '
    +'534 of which once cleared the $10B floor. The model visibly eats their losses (its worst single positions '
    +'are −71%, −57%, −49%, and three of the ten worst later delisted). Residual survivorship exposure is about '
    +'<b>0.2 percentage points of CAGR</b>, versus the ~8 points that inflated the previous version of this model. '
    +'What remains, honestly: '
    +'<b>(1) It is still a backtest.</b> These rules were chosen knowing how this history turned out. That is the '
    +'largest risk on this page and no amount of clean data fixes it — only the '
    +'<a href="/edge-tracker" style="color:#d4af37">forward record</a> can. '
    +'<b>(2) Delisted prices are frozen, not marked down</b> — a name that stops trading is held flat rather than '
    +'sold. Correct for an acquisition, generous for a bankruptcy, and capped at 0.6% of positions. '
    +'<b>(3) The liquidity floor uses market cap</b> as a full-history proxy for tradability. '
    +'<b>(4) The recent window is unusually momentum-friendly</b>, which flatters the short windows most. '
    +'<b>(5) Sharpe here is √252·mean/σ with no risk-free subtraction</b> — a return-to-volatility ratio. It reads '
    +'roughly 0.2 higher than a textbook Sharpe at current rates, and the same convention is applied to the '
    +'benchmarks, so the comparison below is like-for-like. '
    +'<b>What you can trust most is the risk-adjusted shape</b>: Sharpe '+shp+' vs the S&amp;P\'s '+spshp+', and '
    +'drawdown control (<b>'+ddm+' vs the S&amp;P '+dds+'</b>, measured on the daily curve, so it is the true '
    +'intra-period peak-to-trough rather than a sampled approximation). That shape — not the headline return — is the product.';
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
  document.querySelectorAll('#nbtns button').forEach(b=>{
    b.addEventListener('click',()=>{
      document.querySelectorAll('#nbtns button').forEach(x=>x.classList.remove('active'));
      b.classList.add('active'); NSIZE=+b.dataset.n; saveEdgeOpt('n',NSIZE); run();
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
  const sync=document.getElementById('syncVision');
  if(sync){
    sync.addEventListener('click', async ()=>{
      const months=Math.round(HOLD/21);
      const ok=confirm(`Sync the CURRENT backtest to the Vision product?\n\n`
        +`• Window: ${WINDOW}\n• Rebalance: ${months}-month clock\n• Basket: 10 stocks (Vision always ships the product book)\n\n`
        +`This regenerates Vision's data and pushes it live (updates the book + curves).`);
      if(!ok) return;
      const msg=document.getElementById('syncMsg');
      sync.disabled=true; msg.className='small'; msg.textContent='Syncing to Vision…';
      try{
        const r=await fetch('/api/sync_vision',{method:'POST',headers:{'Content-Type':'application/json'},
          body:JSON.stringify({window:WINDOW, hold:HOLD})});
        const d=await safeJson(r);
        if(d.ok){
          msg.className='small pos';
          msg.textContent=(d.pushed?'✓ Pushed to Vision':'✓ Saved locally')
            +' · '+(d.book||[]).slice(0,4).join(', ')+'… · as of '+(d.as_of||'');
        } else { msg.className='small neg'; msg.textContent='✗ '+(d.reason||'Sync failed'); }
      }catch(e){ msg.className='small neg'; msg.textContent='✗ Request failed: '+e; }
      finally{ sync.disabled=false; }
    });
  }
  run();
});
