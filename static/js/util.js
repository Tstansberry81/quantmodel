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
