/**
 * ContextLens Background Service Worker
 * Handles compression logic and session state
 */

// Session storage per tab
const sessions = {};

// Simple hash function for deduplication
function hashString(str) {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    const char = str.charCodeAt(i);
    hash = ((hash << 5) - hash) + char;
    hash = hash & hash;
  }
  return hash.toString();
}

// Compression engine — mirrors core.py logic in JavaScript
function compress(messages) {
  if (!messages || messages.length === 0) return { compressed: [], saved: 0, redundancy: 0 };

  const originalCount = messages.length;
  const originalChars = messages.reduce((sum, m) => sum + (m.content || '').length, 0);

  // Step 1: Exact deduplication
  const seenHashes = new Map();
  const deduped = [];

  for (const msg of messages) {
    const content = msg.content || '';
    const role = msg.role || '';
    const hash = hashString(content);

    // System messages always kept
    if (role === 'system') {
      deduped.push(msg);
      continue;
    }

    if (seenHashes.has(hash)) {
      seenHashes.set(hash, seenHashes.get(hash) + 1);
      // Skip duplicate
      continue;
    }

    seenHashes.set(hash, 1);
    deduped.push(msg);
  }

  // Step 2: Keep last message always
  const lastMsg = messages[messages.length - 1];
  const lastInDeduped = deduped[deduped.length - 1];
  if (lastMsg && lastInDeduped && 
      hashString(lastMsg.content || '') !== hashString(lastInDeduped.content || '')) {
    deduped.push(lastMsg);
  }

  const compressedChars = deduped.reduce((sum, m) => sum + (m.content || '').length, 0);
  const saved = Math.max(0, originalChars - compressedChars);
  const redundancy = originalChars > 0 ? Math.round((saved / originalChars) * 100) : 0;

  return {
    compressed: deduped,
    originalCount,
    compressedCount: deduped.length,
    originalChars,
    compressedChars,
    saved,
    redundancy
  };
}

// Listen for messages from content script
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === 'COMPRESS') {
    const result = compress(message.messages);
    
    // Update session stats
    const tabId = sender.tab?.id;
    if (tabId) {
      if (!sessions[tabId]) {
        sessions[tabId] = { totalSaved: 0, calls: 0 };
      }
      sessions[tabId].totalSaved += result.saved;
      sessions[tabId].calls += 1;
    }

    sendResponse(result);
  }

  if (message.type === 'GET_STATS') {
    const tabId = sender.tab?.id;
    sendResponse(sessions[tabId] || { totalSaved: 0, calls: 0 });
  }

  return true; // Keep channel open for async response
});
// Open popup from content script
chrome.runtime.onMessage.addListener((msg) => {
  if (msg.type === 'OPEN_POPUP') {
    chrome.action.openPopup();
  }
});

// Open welcome page on first install
chrome.runtime.onInstalled.addListener((details) => {
  if (details.reason === 'install') {
    chrome.tabs.create({ url: chrome.runtime.getURL('welcome.html') });
  }
});
