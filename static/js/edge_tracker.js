const fmtPct = v => v==null||isNaN(v) ? "—" : (v*100).toFixed(1)+"%";
const fmtN = (v,d=2) => v==null||isNaN(v) ? "—" : Number(v).toFixed(d);
const cls = v => v==null||isNaN(v) ? "" : (v>=0?"pos":"neg");
const sgnPct = v => v==null||isNaN(v) ? "—" : (v>=0?"+":"")+(v*100).toFixed(1)+"%";

async function run(){
  const status=document.getElementById('status');
  status.className='loading'; status.style.display='block';
  status.textContent='Loading the Edge tracker…';
  document.getElementById('app').style.display='none';
  try{
    const r = await fetch('/api/edge_tracker');
    const d = await r.json();
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
     <td>${fmtPct(b.weight)}</td><td class="${cls(b.accel)}">${fmtN(b.accel,2)}</td></tr>`).join('');
  document.getElementById('bookmeta').textContent =
    `Full-spec Edge picks as of ${d.book_date} · equal-weight (${fmtPct(1/((d.current_book||[]).length||1))} each) · `
    +`ranked by acceleration signal · hold ~${spec.hold_days}d · liquidity ≥ $${fmtN(spec.mcap_floor_bn,0)}B, corr cap ${fmtN(spec.corr_cap,2)}.`;

  // ---- paper-trading log (newest first) ----
  const log=(d.log||[]).slice().reverse();
  document.querySelector('#log tbody').innerHTML = log.map(t=>{
    const open = t.status==='OPEN';
    const badge = open ? '<span class="pos">● OPEN</span>' : '<span style="color:#8b949e">closed</span>';
    return `<tr><td>${t.opened}</td><td>${t.closes}</td><td>${badge}</td>
      <td class="${cls(t.edge_ret)}">${sgnPct(t.edge_ret)}</td>
      <td class="${cls(t.sp_ret)}">${sgnPct(t.sp_ret)}</td>
      <td class="${cls(t.excess)}">${sgnPct(t.excess)}</td></tr>`;}).join('');
  document.getElementById('logmeta').textContent =
    `Last ${log.length} rebalances · ${s.n_closed} closed paper trades total · `
    +`beat the S&P ${fmtPct(s.hit_rate)} of the time · avg excess ${sgnPct(s.avg_excess)}/trade `
    +`(avg Edge ${sgnPct(s.avg_edge_ret)} vs S&P ${sgnPct(s.avg_sp_ret)}). `
    +`Returns are NET of ${spec.cost_bps}bps costs. History seeded from the backtest; live snapshots accrue forward from ${s.first_snapshot}.`;
}

document.addEventListener('DOMContentLoaded', run);
