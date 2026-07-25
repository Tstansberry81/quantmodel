const fmtPct = v => v==null||isNaN(v) ? "—" : (v*100).toFixed(1)+"%";
const fmtN = (v,d=2) => v==null||isNaN(v) ? "—" : Number(v).toFixed(d);
const cls = v => v==null||isNaN(v) ? "" : (v>=0?"pos":"neg");
const sgnPct = v => v==null||isNaN(v) ? "—" : (v>=0?"+":"")+(v*100).toFixed(1)+"%";

const _O = loadEdgeOpts();   // shared with the Edge backtest page (localStorage)
let WINDOW = _O.window;      // paper-log horizon: 1Y/2Y/5Y/MAX
let HOLD = _O.hold;          // rebalance clock: 21/42/63/126 = 1M/2M/3M/6M
let MIX = _O.mix;            // growth mix: share of basket from >=15%-rev-growth names
let NSIZE = _O.n;            // basket size 5..10 (product = 10; n=7 = paper-tracked research book)

async function run(){
  const status=document.getElementById('status');
  status.className='loading'; status.style.display='block';
  status.textContent='Loading the Edge tracker…';
  document.getElementById('app').style.display='none';
  try{
    const r = await fetch(`/api/edge_tracker?hold=${HOLD}&window=${WINDOW}&mix=${MIX}&n=${NSIZE}`);
    const d = await safeJson(r);
    if(!d.ok){ status.className='err'; status.textContent=d.reason||'Tracker failed'; return; }
    render(d);
    status.style.display='none';
    document.getElementById('app').style.display='block';
  }catch(e){ status.className='err'; status.textContent='Request failed: '+e; }
}

function render(d){
  const s=d.stats||{}, spec=d.spec||{};

  document.getElementById('headline').innerHTML = [
    ['Current book', (d.current_book||[]).length+' names', ''],
    ['Book date', d.book_date||'—', ''],
    ['Closed paper trades', fmtN(s.n_closed,0), ''],
    ['Beat S&P (hit rate)', fmtPct(s.hit_rate), cls((s.hit_rate||0)-0.5)],
    ['Avg excess / trade', sgnPct(s.avg_excess), cls(s.avg_excess)],
    ['Live snapshots', fmtN(s.n_snapshots,0), ''],
  ].map(([k,v,c])=>`<div class="stat"><div class="k">${k}</div><div class="v ${c}">${v}</div></div>`).join('');

  // ---- current book table ----
  document.querySelector('#book tbody').innerHTML = (d.current_book||[]).map(b=>
    `<tr><td><b>${b.ticker}</b></td><td>${b.name||'—'}</td><td>${b.sector||'—'}</td>
     <td>${fmtPct(b.weight)}</td><td class="${cls(b.accel)}">${fmtN(b.accel,2)}</td>
     <td class="${cls(b.ret_todate)}">${b.ret_todate==null?'—':sgnPct(b.ret_todate)}</td></tr>`).join('');
  const mixTxt = spec.growth_mix>0 ? `${Math.round(spec.growth_mix*100)}% growth mix` : 'pure momentum';
  const recTxt = s.record==='paper-n7'
    ? ' · n=7 research candidate — its forward record accrues alongside the product book'
    : (spec.n===10 ? '' : ' · exploratory basket size (no forward record)');
  // the book is the position opened at the last rebalance and still held —
  // "since open" is a mark to the latest close, not a finished trade
  const heldTxt = s.book_is_live
    ? `Open position — bought at the ${d.book_date} rebalance, still held; "since open" is marked to the latest close`
    : `Full-spec Edge picks as of ${d.book_date}`;
  const regimeTxt = s.book_regime_on === false
    ? ` · MARKET BELOW ITS 200-DAY AVERAGE: exposure cut to ${fmtPct(s.book_exposure)}, rest in cash`
    : '';
  document.getElementById('bookmeta').textContent =
    `${heldTxt} · equal-weight (${fmtPct(1/((d.current_book||[]).length||1))} each) · `
    +`ranked by acceleration signal · ${mixTxt} · hold ~${spec.hold_days}d · liquidity ≥ $${fmtN(spec.mcap_floor_bn,0)}B, `
    +`corr cap ${fmtN(spec.corr_cap,2)}${recTxt}${regimeTxt}.`;

  // ---- paper-trading log (newest first) ----
  const log=(d.log||[]).slice().reverse();
  document.querySelector('#log tbody').innerHTML = log.map((t,i)=>{
    const open = t.status==='OPEN';
    const badge = open
      ? `<span class="pos">● OPEN</span>${t.mark_to_market?'<br><span class="small" style="color:#8b949e">marked to market</span>':''}`
      : '<span style="color:#8b949e">closed</span>';
    const h=t.holdings||[];
    const toggle = h.length
      ? `<button class="port-toggle" type="button" data-i="${i}" aria-expanded="false">▸ ${h.length} names</button>`
      : '—';
    const chips = h.map(x=>
      `<span class="port-chip" title="${(x.name||'').replace(/"/g,'&quot;')}"><b>${x.ticker}</b> <span class="${cls(x.ret)}">${sgnPct(x.ret)}</span></span>`).join('');
    return `<tr><td>${t.opened}</td><td>${t.closes}</td><td>${badge}</td>
      <td class="${cls(t.edge_ret)}">${sgnPct(t.edge_ret)}</td>
      <td class="${cls(t.sp_ret)}">${sgnPct(t.sp_ret)}</td>
      <td class="${cls(t.excess)}">${sgnPct(t.excess)}</td>
      <td style="text-align:center">${toggle}</td></tr>
      <tr class="port-row" id="port-${i}" hidden><td colspan="7">
        <div class="port-meta">Basket held from ${t.opened} (equal-weight) — each name's own ~${Math.round((spec.hold_days||42)/21)}-month return:</div>
        <div class="port-grid">${chips}</div></td></tr>`;}).join('');
  document.querySelectorAll('#log .port-toggle').forEach(btn=>{
    btn.addEventListener('click',()=>{
      const row=document.getElementById('port-'+btn.dataset.i);
      const showing=row.hidden===false;
      row.hidden=showing;
      btn.setAttribute('aria-expanded', String(!showing));
      btn.textContent=(showing?'▸ ':'▾ ')+btn.textContent.replace(/^[▸▾]\s*/,'');
    });
  });
  const wlabel = WINDOW==='MAX' ? 'full history' : `last ${WINDOW}`;
  document.getElementById('logmeta').textContent =
    `${wlabel} · showing ${log.length} of ${s.n_total} rebalances (${Math.round((spec.hold_days||42)/21)}M clock) · ${s.n_closed} closed paper trades · `
    +`beat the S&P ${fmtPct(s.hit_rate)} of the time · avg excess ${sgnPct(s.avg_excess)}/trade `
    +`(avg Edge ${sgnPct(s.avg_edge_ret)} vs S&P ${sgnPct(s.avg_sp_ret)}). `
    +`Returns are NET of ${spec.cost_bps}bps costs. History seeded from the backtest; live snapshots accrue forward from ${s.first_snapshot}.`;
}

document.addEventListener('DOMContentLoaded',()=>{
  syncEdgeBtns();   // reflect options carried over from the Edge backtest page
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
  document.querySelectorAll('#nbtns button').forEach(b=>{
    b.addEventListener('click',()=>{
      document.querySelectorAll('#nbtns button').forEach(x=>x.classList.remove('active'));
      b.classList.add('active'); NSIZE=+b.dataset.n; saveEdgeOpt('n',NSIZE); run();
    });
  });
  run();
});
