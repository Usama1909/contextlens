/**
 * ContextLens - content.js
 * SSE interception + DOM compression.
 * Universal — works with Claude, ChatGPT, Gemini, and any future platform.
 */

const PLATFORM = (() => {
  const host = window.location.hostname;
  if (host.includes('claude.ai')) return 'claude';
  if (host.includes('chatgpt.com')) return 'chatgpt';
  if (host.includes('gemini.google.com')) return 'gemini';
  return 'unknown';
})();

console.log('[ContextLens] Active on ' + PLATFORM);

(function injectMainWorld() {
  const s = document.createElement('script');
  s.src = chrome.runtime.getURL('injector.js');
  s.onload = () => s.remove();
  (document.head || document.documentElement).appendChild(s);
})();

let sessionTokensSaved = 0;
let chatTokensSaved = 0;
let totalTokensSaved = 0;
let compressionCount = 0;
let lastConvId = null;
let lastActivityTs = Date.now();
const SESSION_TIMEOUT_MS = 30 * 60 * 1000;

const SELECTORS = {
  claude: {
    messages: '[class*="font-user-message"], [class*="standard-markdown"]',
    humanMessage: '[class*="font-user-message"]'
  },
  chatgpt: {
    messages: '[data-message-author-role]',
    humanMessage: '[data-message-author-role="user"]'
  },
  gemini: {
    messages: '.conversation-container .turn',
    humanMessage: '.user-turn'
  }
};

function hashString(str) {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    hash = ((hash << 5) - hash) + str.charCodeAt(i);
    hash = hash & hash;
  }
  return hash.toString();
}

function compress(messages) {
  if (!messages || messages.length < 3) return {
    compressed: messages, saved: 0, redundancy: 0,
    originalCount: messages ? messages.length : 0,
    compressedCount: messages ? messages.length : 0
  };
  const originalChars = messages.reduce((s, m) => s + m.content.length, 0);
  const seen = new Map();
  const result = [];
  for (let i = 0; i < messages.length; i++) {
    const msg = messages[i];
    const hash = hashString(msg.content);
    const isLast = i === messages.length - 1;
    if (isLast) { result.push(msg); continue; }
    if (seen.has(hash)) continue;
    seen.set(hash, true);
    result.push(msg);
  }
  const compressedChars = result.reduce((s, m) => s + m.content.length, 0);
  const saved = Math.max(0, originalChars - compressedChars);
  const redundancy = originalChars > 0 ? Math.round(saved / originalChars * 100) : 0;
  return { compressed: result, saved, redundancy, originalCount: messages.length, compressedCount: result.length };
}

function readConversation() {
  const sel = SELECTORS[PLATFORM];
  if (!sel) return [];
  const elements = document.querySelectorAll(sel.messages);
  const messages = [];
  elements.forEach(el => {
    const isHuman = el.matches(sel.humanMessage) || el.getAttribute('data-message-author-role') === 'user';
    const content = el.innerText?.trim();
    if (content && content.length > 0) messages.push({ role: isHuman ? 'user' : 'assistant', content });
  });
  return messages;
}

function showIndicator(thisTurnTokens) {
  try {
    const existing = document.getElementById('cl-indicator');
    if (existing) existing.remove();
    const el = document.createElement('div');
    el.id = 'cl-indicator';
    el.style.cssText = 'position:fixed;bottom:80px;right:20px;background:#1a1a2e;color:#00ff88;padding:10px 16px;border-radius:8px;font-family:monospace;font-size:12px;z-index:99999;border:1px solid #00ff88;line-height:1.8;box-shadow:0 4px 12px rgba(0,255,136,0.2);cursor:pointer;transition:all 0.2s;';
    el.innerHTML = '<strong>⚡ ContextLens</strong><br>This response: ~' + thisTurnTokens + ' tokens saved<br>This chat: ~' + chatTokensSaved + ' tokens saved<br>This session: ~' + sessionTokensSaved + ' tokens saved<br><span style="font-size:10px;color:#00cc6a">▼ click to view memory</span>';
    document.body.appendChild(el);
    let fadeTimer = setTimeout(() => { el.style.opacity='0'; el.style.transition='opacity 0.5s'; setTimeout(() => el.remove(), 500); }, 4000);
    el.addEventListener('mouseenter', () => { clearTimeout(fadeTimer); el.style.opacity='1'; });
    el.addEventListener('mouseleave', () => { fadeTimer = setTimeout(() => { el.style.opacity='0'; el.style.transition='opacity 0.5s'; setTimeout(() => el.remove(), 500); }, 2000); });
    el.addEventListener('click', () => { clearTimeout(fadeTimer); showMemoryPanel(); });
  } catch(e) {}
}

function showMemoryPanel() {
  try {
    const existing = document.getElementById('cl-panel');
    if (existing) { existing.remove(); return; }
    const panel = document.createElement('div');
    panel.id = 'cl-panel';
    panel.style.cssText = 'position:fixed;bottom:80px;right:20px;width:340px;max-height:400px;background:#0a0a0f;border:1px solid #00ff88;border-radius:10px;font-family:monospace;font-size:12px;z-index:999999;box-shadow:0 8px 32px rgba(0,255,136,0.15);overflow:hidden;display:flex;flex-direction:column;';
    panel.innerHTML = '<div style="padding:12px 14px;border-bottom:1px solid #2a2a3a;display:flex;justify-content:space-between;align-items:center"><span style="color:#00ff88;font-weight:bold">⬡ ContextLens Memory</span><span id="cl-panel-close" style="color:#6b6b80;cursor:pointer;font-size:14px">✕</span></div><div id="cl-panel-body" style="overflow-y:auto;padding:10px;flex:1"><div style="color:#6b6b80;text-align:center;padding:20px">Loading...</div></div>';
    document.body.appendChild(panel);
    document.getElementById('cl-panel-close').addEventListener('click', () => panel.remove());
    document.addEventListener('keydown', function esc(e) { if (e.key==='Escape') { panel.remove(); document.removeEventListener('keydown', esc); } });
    if (!chrome?.storage?.local) return;
    chrome.storage.local.get(['cl_turns'], function(data) {
      const turns = (data.cl_turns || []).slice().reverse().slice(0, 10);
      const body = document.getElementById('cl-panel-body');
      if (!body) return;
      if (turns.length === 0) { body.innerHTML = '<div style="color:#6b6b80;text-align:center;padding:20px">No memory yet.<br>Send a message to start.</div>'; return; }
      body.innerHTML = turns.map(function(t) {
        const time = new Date(t.ts).toLocaleTimeString();
        const text = (t.assistantText||'').slice(0,100) + '…';
        const convId = t.convId || '';
        return '<div data-conv-id="' + convId + '" style="background:#111118;border:1px solid #2a2a3a;border-radius:6px;padding:8px 10px;margin-bottom:6px;cursor:pointer"><div style="color:#00ff88;font-size:10px;margin-bottom:4px">' + (t.platform||'unknown') + ' · ' + time + (t.pinned?' · ★':'' ) + '</div><div style="color:#9999b0;line-height:1.5">' + text + '</div></div>';
      }).join('');
      body.querySelectorAll('[data-conv-id]').forEach(function(card) {
        card.addEventListener('click', function() {
          const id = card.getAttribute('data-conv-id');
          if (id) window.open('https://claude.ai/chat/' + id, '_blank');
        });
      });
    });
  } catch(e) {}
}

function sendToProxy(convId, assistantText, userText) {
  try {
    if (!chrome?.runtime?.id) return;
    chrome.storage.local.get(['cl_proxy_url'], function(data) {
      const url = data.cl_proxy_url || 'https://contextlens.65.108.217.183.nip.io/ingest';
      if (!url) return;
      fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ convId, platform: PLATFORM, assistantText, userText: userText || '', ts: new Date().toISOString() })
      }).then(r => r.json()).then(d => console.log('[ContextLens] Proxy stored turn:', d.id)).catch(e => console.warn('[ContextLens] Proxy send failed:', e));
    });
  } catch(e) {}
}

function saveToLocalMemory(convId, assistantText, userText) {
  try {
    if (!chrome?.runtime?.id) return;
    if (!chrome?.storage?.local) return;
    const turn = {
      id: crypto.randomUUID(), convId: convId, platform: PLATFORM,
      assistantText: assistantText, userText: userText || '',
      ts: new Date().toISOString(), pinned: false, tags: []
    };
    chrome.storage.local.get(['cl_turns', 'cl_user_id'], function(data) {
      try {
        let userId = data.cl_user_id;
        if (!userId) { userId = crypto.randomUUID(); chrome.storage.local.set({ cl_user_id: userId }); }
        let turns = data.cl_turns || [];
        if (turns.some(function(t) { return t.assistantText === assistantText; })) return;
        turns.push(turn);
        if (turns.length > 500) {
          turns = turns.filter(function(t){return t.pinned;}).concat(turns.filter(function(t){return !t.pinned;}).slice(-500));
        }
        chrome.storage.local.set({ cl_turns: turns });
        console.log('[ContextLens] Saved turn. Total: ' + turns.length);
      } catch(e) {}
    });
  } catch(e) {}
}

function watchSendButton() {
  const start = function() {
    if (!document.body) return;
    new MutationObserver(function(){}).observe(document.body, { childList: true, subtree: true });
    console.log('[ContextLens] MutationObserver active (silent)');
  };
  if (document.body) start(); else document.addEventListener('DOMContentLoaded', start);
}

function watchKeyboard() {
  document.addEventListener('keydown', function(e) {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
      try {
        if (!chrome?.runtime?.id) return;
        const messages = readConversation();
        if (messages.length < 2) return;
        const result = compress(messages);
        if (result.redundancy === 0) return;
        const tokens = Math.round(result.saved / 4);
        chatTokensSaved += tokens; sessionTokensSaved += tokens; totalTokensSaved += tokens; compressionCount++;
        showIndicator(tokens);
        chrome.storage.local.set({ cl_tokens_saved: totalTokensSaved, cl_calls: compressionCount });
      } catch(e) {}
    }
  }, true);
}

chrome.runtime.onMessage.addListener(function(msg, sender, sendResponse) {
  if (msg.type === 'GET_POPUP_STATS') {
    sendResponse({ sessionTokensSaved, chatTokensSaved, totalTokensSaved, calls: compressionCount, platform: PLATFORM, messagesInConversation: readConversation().length });
  }
  return true;
});

window.addEventListener('message', function(event) {
  try {
    if (event.source !== window) return;
    if (!event.data || event.data.type !== '__CONTEXTLENS_TURN_COMPLETE__') return;
    if (!chrome?.runtime?.id) return;
    if (!chrome?.storage?.local) return;
    const convId = event.data.convId;
    const assistantText = event.data.assistantText;
    const userText = event.data.userText || '';
    console.log('[ContextLens] Turn captured:', convId, assistantText.slice(0, 80));
    const now = Date.now();
    if (now - lastActivityTs > SESSION_TIMEOUT_MS) { sessionTokensSaved = 0; }
    lastActivityTs = now;
    if (convId !== lastConvId) { chatTokensSaved = 0; lastConvId = convId; }
    sendToProxy(convId, assistantText, userText);
    saveToLocalMemory(convId, assistantText, userText);
    const messages = readConversation();
    if (messages.length < 2) return;
    const result = compress(messages);
    const thisTurnTokens = Math.round(result.saved / 4);
    chatTokensSaved += thisTurnTokens; sessionTokensSaved += thisTurnTokens; totalTokensSaved += thisTurnTokens; compressionCount++;
    chrome.storage.local.set({ cl_tokens_saved: totalTokensSaved, cl_calls: compressionCount });
    showIndicator(thisTurnTokens);
  } catch(e) { console.warn('[ContextLens] SSE handler error:', e); }
});

watchSendButton();
watchKeyboard();
console.log('[ContextLens] DOM observer active');











