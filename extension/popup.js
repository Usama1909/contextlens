// popup.js "” ContextLens popup logic

// â”€â”€ TAB SWITCHING â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
document.querySelectorAll('.tab').forEach(tab => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
    tab.classList.add('active');
    document.getElementById(`panel-${tab.dataset.tab}`).classList.add('active');
  });
});

// â”€â”€ UTILS â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
function formatTs(ts) {
  const d = new Date(ts);
  const now = new Date();
  const diff = now - d;
  if (diff < 60000) return 'just now';
  if (diff < 3600000) return `${Math.floor(diff/60000)}m ago`;
  if (diff < 86400000) return `${Math.floor(diff/3600000)}h ago`;
  return d.toLocaleDateString();
}

function truncate(str, n) {
  if (!str) return '';
  return str.length > n ? str.slice(0, n) + '…' : str;
}

// â”€â”€ RENDER TURNS â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
function renderTurns(turns, query = '') {
  const list = document.getElementById('turnList');
  const countEl = document.getElementById('memoryCount');

  let filtered = turns;
  if (query.trim()) {
    const q = query.toLowerCase();
    filtered = turns.filter(t =>
      t.assistantText?.toLowerCase().includes(q) ||
      t.userText?.toLowerCase().includes(q) ||
      t.platform?.toLowerCase().includes(q)
    );
  }

  // Sort: pinned first, then newest
  filtered.sort((a, b) => {
    if (a.pinned && !b.pinned) return -1;
    if (!a.pinned && b.pinned) return 1;
    return new Date(b.ts) - new Date(a.ts);
  });

  countEl.innerHTML = `<span>${filtered.length}</span> turns stored${query ? ` matching "${query}"` : ''}`;

  if (filtered.length === 0) {
    list.innerHTML = `
      <div class="empty-state">
        <div class="big">⬡</div>
        <p>${query ? 'No turns match your search.' : 'No memory yet.\nStart a conversation on Claude, ChatGPT, or Gemini.'}</p>
      </div>
    `;
    return;
  }

  list.innerHTML = filtered.map(turn => `
    <div class="turn-card ${turn.pinned ? 'pinned' : ''}" data-id="${turn.id}">
      <div class="turn-meta">
        <span class="turn-platform">${turn.platform || 'unknown'}</span>
        <span class="turn-ts">${formatTs(turn.ts)}</span>
      </div>
      ${turn.userText ? `<div style="color:#4488ff;font-size:10px;margin-bottom:3px">▶ ${truncate(turn.userText, 80)}</div>` : ""}
      <div class="turn-text">${truncate(turn.assistantText || "(empty)", 120)}</div>
      <div class="turn-actions">
        <button class="btn-small pin-btn ${turn.pinned ? 'pinned' : ''}" data-id="${turn.id}">
          ${turn.pinned ? '★ Pinned' : '☆ Pin'}
        </button>
        <button class="btn-small del-btn" data-id="${turn.id}">âœ• Delete</button>
      </div>
    </div>
  `).join('');

  // Pin buttons
  list.querySelectorAll('.pin-btn').forEach(btn => {
    btn.addEventListener('click', () => togglePin(btn.dataset.id));
  });

  // Delete buttons
  list.querySelectorAll('.del-btn').forEach(btn => {
    btn.addEventListener('click', () => deleteTurn(btn.dataset.id));
  });
}

// â”€â”€ PIN TURN â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
function togglePin(id) {
  chrome.storage.local.get(['cl_turns'], (data) => {
    let turns = data.cl_turns || [];
    turns = turns.map(t => t.id === id ? { ...t, pinned: !t.pinned } : t);
    chrome.storage.local.set({ cl_turns: turns }, () => {
      loadMemory();
    });
  });
}

// â”€â”€ DELETE TURN â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
function deleteTurn(id) {
  chrome.storage.local.get(['cl_turns'], (data) => {
    let turns = data.cl_turns || [];
    turns = turns.filter(t => t.id !== id);
    chrome.storage.local.set({ cl_turns: turns }, () => {
      loadMemory();
    });
  });
}

// â”€â”€ LOAD MEMORY â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
function loadMemory() {
  chrome.storage.local.get(['cl_turns'], (data) => {
    const turns = data.cl_turns || [];
    const query = document.getElementById('searchInput').value;
    renderTurns(turns, query);
  });
}

// â”€â”€ LOAD STATS â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
function loadStats() {
  chrome.storage.local.get(['cl_turns', 'cl_tokens_saved', 'cl_calls'], (data) => {
    const turns = data.cl_turns || [];

    // Platform breakdown
    const platforms = {};
    turns.forEach(t => {
      platforms[t.platform] = (platforms[t.platform] || 0) + 1;
    });

    document.getElementById('statLifetime').textContent =
      (data.cl_tokens_saved || 0).toLocaleString();
    document.getElementById('statTurns').textContent =
      turns.length.toLocaleString();

    const platformList = document.getElementById('platformList');
    if (Object.keys(platforms).length === 0) {
      platformList.innerHTML = '<span style="color:var(--text-dim);font-size:11px">No data yet</span>';
    } else {
      platformList.innerHTML = Object.entries(platforms)
        .sort((a, b) => b[1] - a[1])
        .map(([name, count]) => `
          <div class="platform-row">
            <span class="platform-name">${name}</span>
            <span class="platform-count">${count} turns</span>
          </div>
        `).join('');
    }
  });

  // Live stats from content script
  chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
    if (!tabs[0]) return;
    chrome.tabs.sendMessage(tabs[0].id, { type: 'GET_POPUP_STATS' }, (resp) => {
      if (chrome.runtime.lastError || !resp) return;
      document.getElementById('statChat').textContent =
        (resp.chatTokensSaved || 0).toLocaleString();
      document.getElementById('statSession').textContent =
        (resp.sessionTokensSaved || 0).toLocaleString();
      document.getElementById('platformBadge').textContent =
        resp.platform || '"”';
    });
  });
}

// â”€â”€ LOAD SETTINGS â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
function loadSettings() {
  chrome.storage.local.get(['cl_user_id'], (data) => {
    const el = document.getElementById('userId');
    if (data.cl_user_id) {
      el.textContent = data.cl_user_id.slice(0, 16) + '…';
    } else {
      el.textContent = 'Not yet assigned';
    }
  });
}

// â”€â”€ EXPORT â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
document.getElementById('btnExport').addEventListener('click', () => {
  chrome.storage.local.get(['cl_turns', 'cl_user_id'], (data) => {
    const payload = {
      exported_at: new Date().toISOString(),
      user_id: data.cl_user_id,
      turns: data.cl_turns || []
    };
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `contextlens-memory-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  });
});

// â”€â”€ CLEAR ALL â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
document.getElementById('btnClear').addEventListener('click', () => {
  if (!confirm('Delete all stored memory? This cannot be undone.')) return;
  chrome.storage.local.set({ cl_turns: [], cl_tokens_saved: 0, cl_calls: 0 }, () => {
    loadMemory();
    loadStats();
  });
});

// â”€â”€ SEARCH â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
document.getElementById('searchInput').addEventListener('input', (e) => {
  chrome.storage.local.get(['cl_turns'], (data) => {
    renderTurns(data.cl_turns || [], e.target.value);
  });
});

// â”€â”€ INIT â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
loadMemory();
loadStats();
loadSettings();





// -- PROXY URL SAVE --
document.getElementById('btnSaveProxy').addEventListener('click', () => {
  const url = document.getElementById('proxyUrl').value.trim();
  chrome.storage.local.set({ cl_proxy_url: url }, () => {
    const btn = document.getElementById('btnSaveProxy');
    btn.textContent = 'Saved checkmark';
    setTimeout(() => { btn.textContent = 'Save'; }, 2000);
  });
});

// -- LOAD PROXY URL --
chrome.storage.local.get(['cl_proxy_url'], (data) => {
  const input = document.getElementById('proxyUrl');
  if (input && data.cl_proxy_url) input.value = data.cl_proxy_url;
});
