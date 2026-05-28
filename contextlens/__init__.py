"""
ContextLens — The context compression protocol for LLM inference.

93% of tokens sent to LLMs are identical repeated data.
ContextLens eliminates that waste in one line.

Basic usage:
    import anthropic
    import contextlens as cx

    client = cx.wrap(anthropic.Anthropic(api_key="..."))
    # Done. All API calls are now automatically compressed.
"""

from contextlens.core import ContextLens
from contextlens.wrap import wrap, wrap_agent
from contextlens.stats import SessionStats

__version__ = "1.0.3"
__all__ = ["wrap", "wrap_agent", "ContextLens", "SessionStats"]
