// injector.js - MAIN world, platform-agnostic SSE interceptor
(function () {
  if (window.__contextlens_injected) return;
  window.__contextlens_injected = true;

  // Platform detection
  function detectPlatform(url) {
    if (url.includes('claude.ai') || url.includes('chat_conversations')) return 'claude';
    if (url.includes('chatgpt.com') || url.includes('backend-api/conversation')) return 'chatgpt';
    if (url.includes('gemini.google.com') || url.includes('batchexecute')) return 'gemini';
    return 'unknown';
  }

  // Check if URL is an AI streaming endpoint
  function isAIEndpoint(url) {
    return (
      (url.includes('completion') && url.includes('chat_conversations')) ||
      (url.includes('backend-api/conversation')) ||
      (url.includes('batchexecute') && url.includes('gemini'))
    );
  }

  // Extract conversation ID per platform
  function extractConvId(url, platform) {
    if (platform === 'claude') {
      const m = url.match(/chat_conversations\/([^/?]+)/);
      return m ? m[1] : 'unknown';
    }
    if (platform === 'chatgpt') {
      const m = url.match(/conversation\/([^/?]+)/);
      return m ? m[1] : 'unknown';
    }
    return 'unknown';
  }

  // Extract user text from request body per platform
  function extractUserText(body, platform) {
    try {
      if (!body) return '';
      const parsed = JSON.parse(body);
      if (platform === 'claude') {
        if (parsed.prompt) return parsed.prompt;
        const msgs = parsed.messages || [];
        const lastUser = msgs.filter(m => m.role === 'user').pop();
        if (lastUser) {
          const c = lastUser.content;
          return typeof c === 'string' ? c : (Array.isArray(c) && c[0]?.text ? c[0].text : '');
        }
      }
      if (platform === 'chatgpt') {
        const msgs = parsed.messages || [];
        const lastUser = msgs.filter(m => m.author?.role === 'user').pop();
        if (lastUser) {
          const parts = lastUser.content?.parts || [];
          return parts.filter(p => typeof p === 'string').join(' ');
        }
      }
    } catch(_) {}
    return '';
  }

  // Parse SSE stream per platform
  async function consumeSSE(response, url, platform, userText) {
    const convId = extractConvId(url, platform);
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

            // Claude format
            if (platform === 'claude') {
              if (parsed.type === 'content_block_delta' && parsed.delta?.type === 'text_delta') {
                assistantText += parsed.delta.text;
              }
              if (parsed.type === 'message_stop') {
                window.postMessage({ type: '__CONTEXTLENS_TURN_COMPLETE__', convId, assistantText, userText, platform }, '*');
                assistantText = '';
              }
            }

            // ChatGPT format
            if (platform === 'chatgpt') {
              // Capture user text from input_message event
              if (parsed.type === 'input_message') {
                const parts = parsed.input_message?.content?.parts || [];
                userText = parts.filter(p => typeof p === 'string').join(' ');
              }
              // Capture assistant text from delta patches
              if (parsed.o === 'patch' && Array.isArray(parsed.v)) {
                parsed.v.forEach(function(patch) {
                  if (patch.p === '/message/content/parts/0' && patch.o === 'append') {
                    assistantText += patch.v;
                  }
                });
              }
              // Completion signal
              if (parsed.type === 'message_stream_complete' && assistantText) {
                const cid = parsed.conversation_id || convId;
                window.postMessage({ type: '__CONTEXTLENS_TURN_COMPLETE__', convId: cid, assistantText, userText, platform }, '*');
                assistantText = '';
                userText = '';
              }
            }

          } catch(_) {}
        }
      }
    } catch(err) {
      if (err.name !== 'AbortError' && err.name !== 'DOMException') {
        console.warn('[ContextLens injector] SSE read error:', err);
      }
    }
  }

  // Intercept fetch
  const _fetch = window.fetch;
  window.fetch = async function (...args) {
    const url = typeof args[0] === 'string' ? args[0] : (args[0]?.url ?? '');
    const platform = detectPlatform(url);

    let userText = '';
    if (isAIEndpoint(url)) {
      userText = extractUserText(args[1]?.body, platform);
      if (userText) console.log('[ContextLens injector] userText captured:', userText.slice(0, 50));
    }

    const response = await _fetch.apply(this, args);

    if (isAIEndpoint(url)) {
      const clone = response.clone();
      consumeSSE(clone, url, platform, userText);
    }

    return response;
  };

  console.log('[ContextLens injector] MAIN world fetch interceptor active');
})();


