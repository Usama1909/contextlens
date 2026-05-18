// injector.js - runs in MAIN world, injected by content.js
(function () {
  if (window.__contextlens_injected) return;
  window.__contextlens_injected = true;

  const _fetch = window.fetch;

  window.fetch = async function (...args) {
    const url = typeof args[0] === 'string' ? args[0] : (args[0]?.url ?? '');
    let userText = '';

    if (url.includes('completion') && url.includes('chat_conversations')) {
      try {
        const body = args[1]?.body;
        if (body) {
          const parsed = JSON.parse(body);
          if (parsed.prompt) {
            userText = parsed.prompt;
            console.log('[ContextLens injector] userText captured:', parsed.prompt.slice(0, 50));
          } else {
            const msgs = parsed.messages || [];
            const lastUser = msgs.filter(function(m) { return m.role === 'user'; }).pop();
            if (lastUser) {
              const c = lastUser.content;
              userText = typeof c === 'string' ? c : (Array.isArray(c) && c[0]?.text ? c[0].text : '');
            }
          }
        }
      } catch(_) {}
    }

    const response = await _fetch.apply(this, args);

    if (url.includes('completion') && url.includes('chat_conversations')) {
      const clone = response.clone();
      consumeClaudeSSE(clone, url, userText);
    }

    return response;
  };

  async function consumeClaudeSSE(response, url, userText) {
    const convId = extractConvId(url);
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let assistantText = '';

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop();

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          const data = line.slice(6).trim();
          if (!data || data === '[DONE]') continue;

          try {
            const parsed = JSON.parse(data);

            if (parsed.type === 'content_block_delta') {
              if (parsed.delta?.type === 'text_delta') {
                assistantText += parsed.delta.text;
              }
            }

            if (parsed.type === 'message_stop') {
              window.postMessage({
                type: '__CONTEXTLENS_TURN_COMPLETE__',
                convId,
                assistantText,
                userText,
              }, '*');
              assistantText = '';
            }

          } catch (_) {}
        }
      }
    } catch (err) {
      console.warn('[ContextLens injector] SSE read error:', err);
    }
  }

  function extractConvId(url) {
    const m = url.match(/chat_conversations\/([^/?]+)/);
    return m ? m[1] : 'unknown';
  }

  console.log('[ContextLens injector] MAIN world fetch interceptor active');
})();


