// In-app chat assistant — answers questions about the numbers on the current page.
(function () {
  const bubble = document.getElementById('chat-bubble');
  const panel = document.getElementById('chat-panel');
  const closeBtn = document.getElementById('chat-close');
  const msgs = document.getElementById('chat-msgs');
  const form = document.getElementById('chat-form');
  const input = document.getElementById('chat-input');
  const send = document.getElementById('chat-send');
  if (!bubble || !panel) return;

  const history = [];   // {role, content}
  let opened = false;

  function open() {
    panel.hidden = false;
    bubble.style.display = 'none';
    if (!opened) {
      opened = true;
      addHint("Hi! Ask me anything about the numbers on this page — e.g. " +
              "“what does the worst drawdown mean?” or “why is alpha +5.5%?”");
    }
    setTimeout(() => input.focus(), 50);
  }
  function close() { panel.hidden = true; bubble.style.display = 'flex'; }

  bubble.addEventListener('click', open);
  closeBtn.addEventListener('click', close);

  function addHint(text) {
    const d = document.createElement('div');
    d.className = 'chat-hint';
    d.textContent = text;
    msgs.appendChild(d);
  }
  function addMsg(text, who, extra) {
    const d = document.createElement('div');
    d.className = 'chat-msg ' + who + (extra ? ' ' + extra : '');
    d.textContent = text;
    msgs.appendChild(d);
    msgs.scrollTop = msgs.scrollHeight;
    return d;
  }

  // Grounding: send the visible text of the main content as context.
  function pageContext() {
    const main = document.querySelector('main');
    return (main ? main.innerText : document.body.innerText)
      .replace(/\s+\n/g, '\n').slice(0, 8000);
  }

  form.addEventListener('submit', async function (e) {
    e.preventDefault();
    const text = input.value.trim();
    if (!text) return;
    input.value = '';
    input.disabled = true; send.disabled = true;
    addMsg(text, 'user');
    history.push({ role: 'user', content: text });
    const thinking = addMsg('thinking…', 'bot', 'thinking');

    try {
      const r = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: text,
          page: document.title + ' (' + location.pathname + ')',
          context: pageContext(),
          history: history.slice(0, -1),
        }),
      });
      const d = await safeJson(r);
      thinking.remove();
      const reply = d.ok ? d.reply : ('⚠ ' + (d.reason || 'Something went wrong.'));
      addMsg(reply, 'bot');
      if (d.ok) history.push({ role: 'assistant', content: d.reply });
    } catch (err) {
      thinking.remove();
      addMsg('⚠ Request failed: ' + err, 'bot');
    } finally {
      input.disabled = false; send.disabled = false;
      input.focus();
    }
  });
})();
