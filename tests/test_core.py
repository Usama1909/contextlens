"""
Core tests for ContextLens compression engine.
Run: pytest tests/
"""

import pytest
from contextlens.core import ContextLens


def test_exact_deduplication():
    """Repeated identical messages should be deduplicated."""
    engine = ContextLens(budget="balanced")
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
        {"role": "user", "content": "Hello"},   # duplicate
        {"role": "user", "content": "Hello"},   # duplicate
    ]
    
    result = engine.compress(messages)
    assert len(result.compressed_messages) < len(result.original_messages)
    assert result.redundancy_pct > 0


def test_no_compression_on_short_conversation():
    """Short conversations should not be compressed."""
    engine = ContextLens(budget="balanced")
    
    messages = [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4"},
    ]
    
    result = engine.compress(messages)
    assert result.original_messages == result.compressed_messages


def test_goal_always_preserved():
    """System messages (goals) must never be removed."""
    engine = ContextLens(budget="economic")
    
    system_msg = {"role": "system", "content": "You are a financial analyst."}
    messages = [system_msg] + [
        {"role": "user", "content": f"Message {i}"} for i in range(20)
    ]
    
    result = engine.compress(messages)
    assert system_msg in result.compressed_messages


def test_empty_messages():
    """Empty message list should return empty."""
    engine = ContextLens()
    result = engine.compress([])
    assert result.compressed_messages == []


def test_session_stats_accumulate():
    """Session stats should accumulate across calls."""
    engine = ContextLens()
    
    msgs = [{"role": "user", "content": "test " * 100}] * 5
    engine.compress(msgs)
    engine.compress(msgs)
    
    assert engine.session_stats["calls"] == 2


def test_aria_pattern():
    """Simulate the ARIA production pattern — 93% redundancy."""
    engine = ContextLens(budget="balanced")
    
    # Simulate what ARIA was doing — same regime message 200x
    repeated_msg = {"role": "assistant", "content": "Regime:CRISIS Score:-45.4 Top:GLD"}
    messages = [repeated_msg] * 50
    
    result = engine.compress(messages)
    assert result.redundancy_pct > 80  # Should catch most of the redundancy
    assert len(result.compressed_messages) < len(result.original_messages)


def test_budget_modes():
    """Different budgets should produce different compression levels."""
    messages = [
        {"role": "user", "content": "same message"} for _ in range(10)
    ]
    
    economic = ContextLens(budget="economic").compress(messages)
    precise = ContextLens(budget="precise").compress(messages)
    
    # Economic should compress more aggressively
    assert len(economic.compressed_messages) <= len(precise.compressed_messages)
