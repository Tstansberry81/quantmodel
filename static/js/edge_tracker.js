// 12-1 momentum is a FRACTIONAL RETURN, not a score. It was rendered with
// fmtN(), so a name up 108% showed as "1.08" and SNDK's ~39x run showed as
// "38.18" -- numbers a reader would read as single-digit percentages, in a
// column the page itself labels "the last 12 months of return".
//
// Past +500% a percentage stops reading as a number and starts reading as a
// formatting bug ("+3817.8%"), so switch to a multiple there. Same rule the
// Vision write-ups use (company_desc._fmt_move), so the two surfaces agree.
const fmtMom = v => v==null||isNaN(v) ? "—"
  : (v >= 5 ? (v+1).toFixed(0)+"\u00d7" : (v>=0?"+":"")+(v*100).toFixed(0)+"%");

const fmtPct = v => v==null||isNaN(v) ? "—" : (v*100).toFixed(1)+"%";
const fmtN = (v,d=2) => v==null||isNaN(v) ? "—" : Number(v).toFixed(d);
// Exact zero is neither. Day 1 of a live mark is 0.00% by construction, and
// painting it green read as "up".
const cls = v => v==null||isNaN(v) ? "" : (v>0?"pos":(v<0?"neg":""));
const sgnPct = v => v==null||isNaN(v) ? "—" : (v>=0?"+":"")+(v*100).toFixed(1)+"%";

const _O = loadEdgeOpts();   // shared with the Edge backtest page (localStorage)
let WINDOW = _O.window;      // paper-log horizon: 1Y/2Y/5Y/MAX
let HOLD = _O.hold;          // rebalance clock: 21/42/63/126 = 1M/2M/3M/6M
let NSIZE = _O.n;            // basket size 5..10 (product = 10; n=7 = paper-tracked research book)

async function run(){
  const status=document.getElementById('status');
  status.className='loading'; status.style.display='block';
  status.textContent='Loading the Edge tracker…';
  document.getElementById('app').style.display='none';
  try{
    const r = await fetch(`/api/edge_tracker?hold=${HOLD}&window=${WINDOW}&n=${NSIZE}`);
    const d = await safeJson(r);
    if(!d.ok){ status.className='err'; status.textContent=d.reason||'Tracker failed'; return; }
    render(d);
    status.style.display='none';
    document.getElementById('app').style.display='block';
  }catch(e){ status.className='err'; status.textContent='Request failed: '+e; }
}

function render(d){
  const s=d.stats||{}, spec=d.spec||{};

  // If the shipped signal has been falsified, say so ABOVE the numbers —
  // a page that shows a book and a hit-rate without this reads as a
  // recommendation, and this one is not.
  const ms = d.model_status;
  const banner = document.getElementById('modelstatus');
  if (banner) {
    if (ms && ms.state === 'falsified') {
      banner.innerHTML = `<b>⚠ ${ms.headline}</b><br><span class="small">${ms.detail}</span>`;
      banner.style.display = 'block';
    } else {
      banner.style.display = 'none';
    }
  }

  // These headline cards used to read "Closed paper trades: 164 / Beat S&P
  // (hit rate)" straight off the BACKTEST-seeded stats -- presenting simulation
  // as the forward record in the largest type on the page. Forward and
  // simulated are now separate cards, each labelled for what it is.
  const fst = d.forward_stats || {};
  const lv = d.live || {};
  // Live mark leads the cards when it exists: between rebalances it is the only
  // thing on this page that moves, and it is the honest answer to "how is the
  // book doing right now".
  const liveCards = lv.days ? [
    ['Live vs S&P', (lv.excess>=0?'+':'')+fmtPct(lv.excess), cls(lv.excess)],
    ['Book since day 1', (lv.edge_ret>=0?'+':'')+fmtPct(lv.edge_ret), cls(lv.edge_ret)],
    ['S&P since day 1', (lv.sp_ret>=0?'+':'')+fmtPct(lv.sp_ret), cls(lv.sp_ret)],
  ] : [];
  document.getElementById('headline').innerHTML = [
    ...liveCards,
    ['Current book', (d.current_book||[]).length+' names', ''],
    ['Book date', d.book_date||'—', ''],
    // Entry pending: the names are final, the price they are bought at is not.
    // Without this the return columns render as bare em-dashes and the page
    // reads as broken rather than as a book published ahead of its entry.
    ...(d.entry_px_pending ? [['Entry price',
        d.entry_date ? 'at the '+d.entry_date+' close' : 'at the next close', '']] : []),
    ['Forward record', `${fmtN(fst.n_closed,0)} closed · ${fmtN(fst.n_open,0)} open`, ''],
    ['Forward hit rate', fst.n_closed ? fmtPct(fst.hit_rate) : 'no data yet',
      fst.n_closed ? cls((fst.hit_rate||0)-0.5) : ''],
    ['Backtest hit rate (sim)', fmtPct(s.hit_rate), cls((s.hit_rate||0)-0.5)],
    ['Backtest trades (sim)', fmtN(s.n_closed,0), ''],
  ].map(([k,v,c])=>`<div class="stat"><div class="k">${k}</div><div class="v ${c}">${v}</div></div>`).join('');

  // ---- current book table ----
  document.querySelector('#book tbody').innerHTML = (d.current_book||[]).map(b=>
    `<tr><td><b>${b.ticker}</b></td><td>${b.name||'—'}</td><td>${b.sector||'—'}</td>
     <td>${fmtPct(b.weight)}</td><td class="${cls(b.signal)}">${fmtMom(b.signal)}</td>
     <td class="${cls(b.ret_todate)}">${b.ret_todate==null?'—':sgnPct(b.ret_todate)}</td></tr>`).join('');
  const recTxt = s.record==='paper-n7'
    ? ' · n=7 research candidate — its forward record accrues alongside the product book'
    : (spec.n===10 ? '' : ' · exploratory basket size (no forward record)');
  // the book is the position opened at the last rebalance and still held —
  // "since open" is a mark to the latest close, not a finished trade
  // Pending first: while a new book awaits its entry close, the PREVIOUS book
  // is still held and live, so book_is_live alone would call the new names
  // "bought" a session before they are.
  const heldTxt = d.entry_px_pending
    ? `Full-spec Edge picks for ${d.book_date}. These names are final; they are bought at the ${d.entry_date||'next'} close, so there is no return to show yet.`
    : s.book_is_live
    ? `Open position — bought at the ${d.book_date} rebalance, still held; "since open" is marked to the latest close`
    : `Full-spec Edge picks as of ${d.book_date}`;
  const regimeTxt = s.book_regime_on === false
    ? ` · MARKET BELOW ITS 200-DAY AVERAGE: exposure cut to ${fmtPct(s.book_exposure)}, rest in cash`
    : '';
  document.getElementById('bookmeta').textContent =
    `${heldTxt} · equal-weight (${fmtPct(1/((d.current_book||[]).length||1))} each) · `
    +`ranked by ${spec.signal} · hold ~${spec.hold_days}d · liquidity ≥ $${fmtN(spec.mcap_floor_bn,0)}B, `
    +`${spec.sector_cap ? `max ${spec.sector_cap} per sector` : 'no sector cap'}${recTxt}${regimeTxt}.`;

  // ---- TOTAL RETURN: every scored forward book, compounded ----
  // (1+r1)(1+r2)...-1, each book held entry close -> next book's entry close,
  // the S&P compounded over the same windows. Server-computed (forward_ledger).
  renderTotal(d.total_return || {});

  // ---- FORWARD record (the only out-of-sample evidence) ----
  // Rendered above and apart from the backtest-seeded log below. Mixing them
  // let 164 simulated rebalances visually swamp the handful of real ones.
  const fs = d.forward_stats || {};
  const flog = (d.forward_log||[]).slice().reverse();
  const fwdBody = document.querySelector('#fwdlog tbody');
  if (fwdBody) {
    fwdBody.innerHTML = flog.length ? flog.map(t=>{
      const open = t.status==='OPEN';
      // STRANDED = written down under a rebalance clock the model no longer runs,
      // so no closing date exists for it. Shown as its own state rather than as a
      // perpetual OPEN, which would read as a live position that never resolves.
      // BACKFILLED = written down after the date it is dated, so its holding
      // window was already running. Never scored, for the same reason stranded
      // rows are not: a return over a window that already happened is a backtest.
      const badge = t.status==='SUPERSEDED'
        ? `<span class="small" title="${t.stranded_reason||''}">↺ superseded</span>`
        : t.status==='STRANDED'
        ? `<span class="small" title="${t.stranded_reason||''}">&#8856; stranded</span>`
        : t.status==='BACKFILLED'
        ? `<span class="small" title="${t.stranded_reason||''}">&#9203; backfilled</span>`
        : t.status==='PENDING'
        ? `<span class="small" title="names are final; bought at the ${t.entry_date||'next'} close">◌ entry ${t.entry_date||'next close'}</span>`
        : (open?'<span class="pos">● OPEN</span>':'<span class="small">closed</span>');
      // Surface the gap between "dated" and "written down" on every row. It is
      // the one number that says whether a row is evidence, so it belongs in the
      // table rather than in a caveat underneath it.
      const lag = t.logged_lag_days;
      const lagTxt = (lag!=null && lag>0)
        ? `<span class="small" style="color:#8b949e"> · ${lag}d after</span>` : '';
      return `<tr><td>${t.book_date}</td>
        <td class="small">${(t.logged_at||'').replace('T',' ').replace('Z','')}${lagTxt}</td>
        <td>${badge}</td>
        <td class="${cls(t.edge_ret)}">${t.edge_ret!=null?sgnPct(t.edge_ret)+(t.mark_to_market?`<span class="small"> live · since ${t.entry_date}</span>`:''):'—'}</td>
        <td class="${cls(t.sp_ret)}">${t.sp_ret!=null?sgnPct(t.sp_ret):'—'}</td>
        <td class="${cls(t.excess)}">${t.excess!=null?sgnPct(t.excess):'—'}</td>
        <td class="small">${(t.tickers||[]).join(' ')}</td></tr>`;
    }).join('')
      : `<tr><td colspan="7" class="small">No forward rebalances recorded yet — the
         record restarts whenever the model's parameters change, and the first
         entry closes after one full holding period.</td></tr>`;
  }
  const fwdMeta = document.getElementById('fwdmeta');
  if (fwdMeta) {
    const stranded = (fs.n_stranded ? ` · ${fs.n_stranded} stranded by a clock change` : '')
      + (fs.n_superseded ? ` · ${fs.n_superseded} superseded by a rules change` : '')
      + (fs.n_backfilled ? ` · ${fs.n_backfilled} backfilled (dated earlier than written down, so never scored)` : '');
    // The live mark, not the log, is what is accruing evidence right now. Say so
    // explicitly while every logged row is still unscoreable — otherwise "0 open,
    // 0 closed" reads as nothing running at all.
    const liveTxt = lv.days
      ? `Tracking ${lv.n_book} names from ${lv.inception} (day ${lv.days}, `
        + `${lv.n_priced}/${lv.n_book} priced, marked to ${lv.as_of}) — ${lv.basis}. `
      : '';
    fwdMeta.textContent = liveTxt + (fs.n_closed
      ? `${fs.n_closed} closed · ${fmtPct(fs.hit_rate)} beat the S&P · `
        +`${sgnPct(fs.avg_excess)} average excess · ${fs.n_open} open`
      : `${fs.n_open||0} scoreable rebalances open, 0 closed. Nothing in this table `
        +`is evidence yet — the clock above is what is accruing it, and the first `
        +`rebalance still needs a full holding period to finish.`) + stranded;
  }

  // ---- BACKTEST-SEEDED log (newest first) — NOT out-of-sample ----
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
        <div class="port-meta">Basket held from ${t.opened} (equal-weight) — each name's own ~${Math.round((spec.hold_days||21)/21)}-month return:</div>
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
    `${wlabel} · showing ${log.length} of ${s.n_total} SIMULATED rebalances (${Math.round((spec.hold_days||21)/21)}M clock) · `
    +`beat the S&P ${fmtPct(s.hit_rate)} of the time · avg excess ${sgnPct(s.avg_excess)}/trade `
    +`(avg Edge ${sgnPct(s.avg_edge_ret)} vs S&P ${sgnPct(s.avg_sp_ret)}). `
    +`Returns are NET of ${spec.cost_bps}bps costs. These are backtest results, not a forward record — `
    +`that lives in the panel above.`
    + (spec.overlays_in_returns === false
        ? ` Per-rebalance returns apply the 200dMA gate as judged at each rebalance, but NOT the`
          + `${spec.continuous_regime ? ' daily regime check or' : ''}`
          + `${spec.vol_target ? ` ${fmtN(spec.vol_target*100,0)}% volatility target` : ' exposure overlays'}`
          + ` — those act on the daily curve, which the backtest page measures.`
        : '');
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
  document.querySelectorAll('#nbtns button').forEach(b=>{
    b.addEventListener('click',()=>{
      document.querySelectorAll('#nbtns button').forEach(x=>x.classList.remove('active'));
      b.classList.add('active'); NSIZE=+b.dataset.n; saveEdgeOpt('n',NSIZE); run();
    });
  });
  run();
});


let TOTAL_CHART = null;
function renderTotal(t){
  const panel = document.getElementById('totalpanel');
  if (!panel) return;
  if (!t.n_books) { panel.style.display = 'none'; return; }
  panel.style.display = '';
  document.getElementById('totalcards').innerHTML = [
    ['Edge total', sgnPct(t.edge_ret), cls(t.edge_ret)],
    ['S&P total', sgnPct(t.sp_ret), cls(t.sp_ret)],
    ['Excess', sgnPct(t.excess), cls(t.excess)],
    ['Since', t.since || '—', ''],
    ['Books', fmtN(t.n_books,0), ''],
  ].map(([k,v,c])=>`<div class="stat"><div class="k">${k}</div><div class="v ${c}">${v}</div></div>`).join('');

  // Each book date opens that book's names: entry close, exit (or latest)
  // close, and return -- enough to redo the row's number by hand, since the
  // book's return is the plain average of its names' returns.
  let run = 1;
  const fmtPx = v => v==null ? '—' : '$'+Number(v).toLocaleString(undefined,{minimumFractionDigits:2,maximumFractionDigits:2});
  const tbody = document.querySelector('#totallegs tbody');
  tbody.innerHTML = (t.legs||[]).map((l,i)=>{
    run *= 1 + l.edge_ret;
    const live = l.status==='OPEN' ? '<span class="small"> live</span>' : '';
    const hs = (l.holdings||[]).slice().sort((a,b)=>(b.ret??-9)-(a.ret??-9));
    const priced = hs.filter(h=>h.ret!=null);
    const sum = priced.reduce((a,h)=>a+h.ret,0);
    const rows = hs.map(h=>`<tr><td><b>${h.ticker}</b></td><td class="small">${h.name||''}</td>
        <td>${fmtPx(h.entry_px)}</td><td>${fmtPx(h.exit_px)}</td>
        <td class="${cls(h.ret)}">${sgnPct(h.ret)}</td></tr>`).join('');
    return `<tr><td><button class="port-toggle" type="button" data-i="${i}" aria-expanded="false">▸ ${l.book_date}</button></td>
      <td class="small">${l.from} → ${l.to}${live}</td>
      <td class="${cls(l.edge_ret)}">${sgnPct(l.edge_ret)}</td>
      <td class="${cls(l.sp_ret)}">${sgnPct(l.sp_ret)}</td>
      <td class="${cls(run-1)}">${sgnPct(run-1)}</td></tr>
      <tr class="port-row" id="leg-${i}" hidden><td colspan="5">
        <div class="port-meta">Bought at the ${l.from} close, ${l.status==='OPEN'?'marked to the '+l.to+' close (still held)':'sold at the '+l.to+' close'} · 10% each · prices are dividend-adjusted closes</div>
        <table class="legtbl"><thead><tr><th>Ticker</th><th>Name</th><th>Bought at</th><th>${l.status==='OPEN'?'Latest':'Sold at'}</th><th>Return</th></tr></thead>
        <tbody>${rows}</tbody></table>
        <div class="port-meta">Book return = average of the ${priced.length} names = ${sgnPct(sum)} ÷ ${priced.length} = <b class="${cls(l.edge_ret)}">${sgnPct(l.edge_ret)}</b></div>
      </td></tr>`;
  }).join('');
  tbody.querySelectorAll('.port-toggle').forEach(btn=>{
    btn.addEventListener('click',()=>{
      const row=document.getElementById('leg-'+btn.dataset.i);
      const showing=row.hidden===false;
      row.hidden=showing;
      btn.setAttribute('aria-expanded', String(!showing));
      btn.textContent=(showing?'▸ ':'▾ ')+btn.textContent.replace(/^[▸▾]\s*/,'');
    });
  });

  document.getElementById('totalmeta').textContent =
    `Compounded, not added: each book's return multiplies the one before it `
    + `(e.g. +10% then +10% is +21%). ${t.basis}. Marked to ${t.as_of}.`;

  const ser = t.series || [];
  const el = document.getElementById('totalchart');
  if (typeof Chart === 'undefined' || !el || !ser.length) return;
  if (TOTAL_CHART) TOTAL_CHART.destroy();
  const css = getComputedStyle(document.documentElement);
  const muted = (css.getPropertyValue('--muted')||'#8b949e').trim();
  TOTAL_CHART = new Chart(el, {
    type: 'line',
    data: {
      labels: [t.since, ...ser.map(p=>p.d)],
      datasets: [
        {label: 'Edge', data: [0, ...ser.map(p=>p.e*100)], borderColor: '#3fb950',
         borderWidth: 2, pointRadius: 0, tension: 0},
        {label: 'S&P 500 (total return)', data: [0, ...ser.map(p=>p.s*100)], borderColor: muted,
         borderWidth: 1.5, borderDash: [4,3], pointRadius: 0, tension: 0},
      ]},
    options: {
      responsive: true, maintainAspectRatio: false, animation: false,
      interaction: {mode: 'index', intersect: false},
      plugins: {legend: {labels: {color: muted, boxWidth: 12}},
        tooltip: {callbacks: {label: c => `${c.dataset.label}: ${(c.parsed.y>=0?'+':'')}${c.parsed.y.toFixed(1)}%`}}},
      scales: {
        x: {ticks: {color: muted, maxTicksLimit: 6}, grid: {display: false}},
        y: {ticks: {color: muted, callback: v => v+'%'}, grid: {color: 'rgba(139,148,158,.15)'}}},
    }});
}
