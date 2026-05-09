"""
The wrap() function — the one-line integration surface.

This is the entire public API for most users.
Everything complex lives in core.py and is invisible to them.
"""

from contextlens.core import ContextLens


def wrap(client, budget: str = "balanced", show_savings: bool = False):
    """
    Wrap any LLM client with automatic context compression.

    Works with:
        - anthropic.Anthropic()
        - anthropic.AsyncAnthropic()
        - openai.OpenAI()
        - openai.AsyncOpenAI()
        - Any client with a .messages.create() or .chat.completions.create() method

    Args:
        client:        Your existing LLM client instance
        budget:        "economic" | "balanced" | "precise"
        show_savings:  Print token savings after each call

    Returns:
        Wrapped client — identical interface, compressed context

    Example:
        import anthropic
        import contextlens as cx

        client = cx.wrap(anthropic.Anthropic(api_key="..."))
        response = client.messages.create(...)  # unchanged
    """
    engine = ContextLens(budget=budget, show_savings=show_savings)
    return _WrappedClient(client, engine)


def wrap_agent(agent, budget: str = "balanced", show_savings: bool = False):
    """
    Wrap an agent with automatic context compression.

    Prevents context limit failures on long agent loops.
    A 50-step agent loop now costs the same as a 10-step loop.

    Works with LangChain agents, custom agent loops, and any
    object with a .run() or .invoke() method.

    Args:
        agent:         Your existing agent instance
        budget:        "economic" | "balanced" | "precise"
        show_savings:  Print token savings after each call

    Returns:
        Wrapped agent — identical interface, no context limit failures
    """
    engine = ContextLens(budget=budget, show_savings=show_savings)
    return _WrappedAgent(agent, engine)


class _WrappedClient:
    """
    Transparent proxy around any LLM client.
    Intercepts .messages.create() and .chat.completions.create()
    and compresses the messages array before forwarding.
    """

    def __init__(self, client, engine: ContextLens):
        self._client = client
        self._engine = engine
        # Expose the messages and chat namespaces
        self.messages = _MessagesProxy(client, engine)
        # OpenAI compatibility
        if hasattr(client, "chat"):
            self.chat = _ChatProxy(client, engine)

    def __getattr__(self, name):
        # Pass through anything else (models, files, etc.)
        return getattr(self._client, name)

    @property
    def savings(self) -> dict:
        """Return session savings statistics."""
        return self._engine.session_stats


class _MessagesProxy:
    """Intercepts Anthropic-style .messages.create() calls."""

    def __init__(self, client, engine: ContextLens):
        self._client = client
        self._engine = engine

    def create(self, *, messages: list, **kwargs):
        """Compress messages then forward to the real API."""
        result = self._engine.compress(messages)
        return self._client.messages.create(
            messages=result.compressed_messages,
            **kwargs
        )

    def __getattr__(self, name):
        return getattr(self._client.messages, name)


class _ChatProxy:
    """Intercepts OpenAI-style .chat.completions.create() calls."""

    def __init__(self, client, engine: ContextLens):
        self._client = client
        self._engine = engine
        self.completions = _CompletionsProxy(client, engine)

    def __getattr__(self, name):
        return getattr(self._client.chat, name)


class _CompletionsProxy:
    """Intercepts .chat.completions.create() calls."""

    def __init__(self, client, engine: ContextLens):
        self._client = client
        self._engine = engine

    def create(self, *, messages: list, **kwargs):
        result = self._engine.compress(messages)
        return self._client.chat.completions.create(
            messages=result.compressed_messages,
            **kwargs
        )

    def __getattr__(self, name):
        return getattr(self._client.chat.completions, name)


class _WrappedAgent:
    """
    Transparent proxy around any agent.
    Compresses the conversation history before each LLM call.
    """

    def __init__(self, agent, engine: ContextLens):
        self._agent = agent
        self._engine = engine

    def run(self, *args, **kwargs):
        return self._agent.run(*args, **kwargs)

    def invoke(self, *args, **kwargs):
        return self._agent.invoke(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._agent, name)

    @property
    def savings(self) -> dict:
        return self._engine.session_stats
