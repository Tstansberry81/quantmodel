// Shared front-end helpers, loaded on every page from base.html (before all
// other scripts) so each page script can rely on them.

// Parse a fetch Response as JSON, degrading gracefully when the server returned
// a non-JSON body (e.g. an HTML 404/500 page) — so callers never throw the
// cryptic "Unexpected token '<', "<!DOCTYPE"... is not valid JSON".
async function safeJson(r) {
  const text = await r.text();
  try {
    return JSON.parse(text);
  } catch (_) {
    return { ok: false, reason: `Server returned a non-JSON response (HTTP ${r.status}).` };
  }
}

// ---- shared model options (synced across the Edge backtest + Edge Tracker) ----
// Both pages read/write the same four controls from localStorage so a setting
// picked on one page carries over to the other.
const EDGE_OPTS_KEY = 'edgeOpts';
const EDGE_OPTS_DEFAULTS = { window: 'MAX', hold: 42, mix: 0.0, n: 10 };

function loadEdgeOpts() {
  try {
    return Object.assign({}, EDGE_OPTS_DEFAULTS, JSON.parse(localStorage.getItem(EDGE_OPTS_KEY) || '{}'));
  } catch (_) {
    return Object.assign({}, EDGE_OPTS_DEFAULTS);
  }
}

function saveEdgeOpt(key, val) {
  const o = loadEdgeOpts();
  o[key] = val;
  try { localStorage.setItem(EDGE_OPTS_KEY, JSON.stringify(o)); } catch (_) {}
}

// Reflect the stored options in a page's button rows (mark the matching button
// active). Safe to call on either page — missing rows are simply skipped.
function syncEdgeBtns() {
  const o = loadEdgeOpts();
  [['#windowbtns', 'w', String(o.window)],
   ['#holdbtns', 'h', String(o.hold)],
   ['#mixbtns', 'm', String(o.mix)],
   ['#nbtns', 'n', String(o.n)]].forEach(([sel, attr, val]) => {
    document.querySelectorAll(sel + ' button').forEach(b =>
      b.classList.toggle('active', b.dataset[attr] === val));
  });
}
