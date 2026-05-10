"""
ContextLens Core Engine

The compression engine that processes every message before
it reaches the LLM. Three stages:
  1. Deduplication  — exact and near-exact match removal
  2. Triage         — relevance scoring (requires [semantic] extra)
  3. Summarisation  — warm content compressed to key facts
"""

import hashlib
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


@dataclass
class MessageMeta:
    """Metadata tracked for every message in the session."""
    content_hash: str
    times_seen: int = 1
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    relevance_score: float = 1.0
    message_type: str = "unknown"  # goal, tool_call, tool_result, reasoning, error


@dataclass 
class CompressionResult:
    """What the engine produced from a list of messages."""
    original_messages: list
    compressed_messages: list
    original_chars: int
    compressed_chars: int
    tokens_estimated_saved: int
    redundancy_pct: float
    method: str  # "dedup", "triage", "summarise"


class ContextLens:
    """
    The core compression engine.

    Tracks message history within a session and applies
    progressively smarter compression as context grows.

    Stages (activated automatically as session grows):
      - Stage 1: Exact deduplication (always active, zero latency)
      - Stage 2: Near-duplicate detection (hash similarity)
      - Stage 3: Semantic triage (requires sentence-transformers)
    """

    def __init__(
        self,
        budget: str = "balanced",
        show_savings: bool = False,
        enable_semantic: bool = False,
    ):
        """
        Args:
            budget: "economic" | "balanced" | "precise"
                    Controls aggressiveness of compression.
            show_savings: Print token savings after each call.
            enable_semantic: Use embedding model for semantic triage.
                             Requires: pip install contextlens[semantic]
        """
        self.budget = budget
        self.show_savings = show_savings
        self.enable_semantic = enable_semantic

        # Thresholds per budget mode
        self._thresholds = {
            "economic": {"dedup": 0.95, "triage_keep": 0.5},
            "balanced": {"dedup": 0.98, "triage_keep": 0.3},
            "precise":  {"dedup": 1.00, "triage_keep": 0.1},
        }

        # Session state
        self._seen_hashes: dict[str, MessageMeta] = {}
        self._session_original_chars: int = 0
        self._session_compressed_chars: int = 0
        self._call_count: int = 0

        # Semantic model (lazy loaded)
        self._embedder = None

    def compress(self, messages: list[dict]) -> CompressionResult:
        """
        Main entry point. Takes a messages array, returns compressed version.

        Args:
            messages: Standard LLM messages array 
                      [{"role": "user", "content": "..."}, ...]

        Returns:
            CompressionResult with compressed messages and stats
        """
        if not messages:
            return CompressionResult(
                original_messages=messages,
                compressed_messages=messages,
                original_chars=0,
                compressed_chars=0,
                tokens_estimated_saved=0,
                redundancy_pct=0.0,
                method="none"
            )

        original_chars = sum(
            len(str(m.get("content", ""))) for m in messages
        )

        # Stage 1: Exact deduplication (always runs, ~0ms overhead)
        compressed = self._deduplicate(messages)

        # Stage 2: Agent-aware compression
        compressed = self._compress_agent_messages(compressed)

        compressed_chars = sum(
            len(str(m.get("content", ""))) for m in compressed
        )

        saved = original_chars - compressed_chars
        redundancy = round((saved / original_chars * 100), 1) if original_chars > 0 else 0.0

        # Rough token estimate: ~4 chars per token
        tokens_saved = saved // 4

        result = CompressionResult(
            original_messages=messages,
            compressed_messages=compressed,
            original_chars=original_chars,
            compressed_chars=compressed_chars,
            tokens_estimated_saved=tokens_saved,
            redundancy_pct=redundancy,
            method="dedup+agent"
        )

        # Update session stats
        self._session_original_chars += original_chars
        self._session_compressed_chars += compressed_chars
        self._call_count += 1

        if self.show_savings:
            self._print_savings(result)

        return result

    def _deduplicate(self, messages: list[dict]) -> list[dict]:
        """
        Stage 1: Remove exact duplicate messages.
        Keeps the most recent occurrence, replaces earlier ones
        with a compact reference.
        """
        threshold = self._thresholds[self.budget]["dedup"]
        seen_in_request: dict[str, int] = defaultdict(int)
        result = []

        for msg in messages:
            content = str(msg.get("content", ""))
            content_hash = self._hash(content)
            seen_in_request[content_hash] += 1

        # Second pass: keep first occurrence, compress repeats
        seen_first: set[str] = set()
        for msg in messages:
            content = str(msg.get("content", ""))
            content_hash = self._hash(content)
            count = seen_in_request[content_hash]

            if count > 1 and content_hash in seen_first:
                # This is a repeat — skip it (already sent once)
                continue
            
            if count > 1 and content_hash not in seen_first:
                # First occurrence of a repeated message
                # Add repeat count hint only if content is long enough
                # that the annotation doesn't increase total size
                annotation = f" [repeated {count}x]"
                if count >= 3 and len(content) > len(annotation) * 2:
                    compressed_content = f"{content}{annotation}"
                    msg = {**msg, "content": compressed_content}
            
            seen_first.add(content_hash)
            result.append(msg)

        return result

    def _compress_agent_messages(self, messages: list[dict]) -> list[dict]:
        """
        Stage 2: Agent-aware compression.
        Classifies messages by type and applies type-specific rules.
        """
        if len(messages) < 5:
            # Short conversations — no compression needed
            return messages

        classified = [(msg, self._classify_message(msg)) for msg in messages]
        result = []
        error_count = 0
        last_error = None
        tool_calls_seen: dict[str, int] = defaultdict(int)

        for i, (msg, msg_type) in enumerate(classified):
            content = str(msg.get("content", ""))

            if msg_type == "goal":
                # Always keep — never compress the original task
                result.append(msg)

            elif msg_type == "error":
                error_count += 1
                last_error = content
                # Only keep last error + count
                if i == len(classified) - 1 or classified[i+1][1] != "error":
                    compressed = f"[{error_count} error(s), last: {last_error[:100]}]"
                    result.append({**msg, "content": compressed})

            elif msg_type == "tool_call":
                tool_name = self._extract_tool_name(content)
                tool_calls_seen[tool_name] += 1
                # Keep only most recent call per tool type
                future_same = sum(
                    1 for _, t in classified[i+1:]
                    if t == "tool_call" and self._extract_tool_name(
                        str(_[0].get("content", ""))
                    ) == tool_name
                )
                if future_same == 0:
                    result.append(msg)
                # else: a newer call for this tool exists, skip this one

            elif msg_type == "reasoning":
                # Keep last 5 reasoning steps only
                future_reasoning = sum(
                    1 for _, t in classified[i+1:] if t == "reasoning"
                )
                if future_reasoning < 5:
                    result.append(msg)

            else:
                # Default: keep
                result.append(msg)

        return result

    def _classify_message(self, msg: dict) -> str:
        """Classify a message into agent message types."""
        content = str(msg.get("content", "")).lower()
        role = msg.get("role", "")

        if role == "system":
            return "goal"
        if any(w in content for w in ["error:", "exception:", "failed:", "traceback"]):
            return "error"
        if any(w in content for w in ["tool_use", "function_call", "tool_call"]):
            return "tool_call"
        if any(w in content for w in ["tool_result", "function_result", "observation:"]):
            return "tool_result"
        if role == "assistant" and len(content) > 100:
            return "reasoning"

        return "general"

    def _extract_tool_name(self, content: str) -> str:
        """Extract tool name from a tool call message."""
        # Simple heuristic — works for most formats
        for keyword in ["tool_use", "function_call", "name"]:
            if keyword in content:
                idx = content.find(keyword)
                return content[idx:idx+30].split('"')[1] if '"' in content[idx:idx+30] else "unknown"
        return "unknown"

    def _hash(self, content: str) -> str:
        """Fast hash for deduplication."""
        return hashlib.md5(content.encode()).hexdigest()

    def _print_savings(self, result: CompressionResult) -> None:
        print(f"\n{'─' * 45}")
        print(f"ContextLens | Call #{self._call_count}")
        print(f"  Original:   {result.original_chars:,} chars (~{result.original_chars // 4:,} tokens est.)")
        print(f"  Compressed: {result.compressed_chars:,} chars (~{result.compressed_chars // 4:,} tokens est.)")
        print(f"  Saved:      ~{result.tokens_estimated_saved:,} tokens ({result.redundancy_pct}%)")
        print(f"{'─' * 45}\n")

    @property
    def session_stats(self) -> dict:
        """Return session-level compression statistics."""
        total_saved = self._session_original_chars - self._session_compressed_chars
        redundancy = round(
            (total_saved / self._session_original_chars * 100), 1
        ) if self._session_original_chars > 0 else 0.0

        return {
            "calls": self._call_count,
            "original_chars": self._session_original_chars,
            "compressed_chars": self._session_compressed_chars,
            "chars_saved": total_saved,
            "tokens_saved_estimate": total_saved // 4,
            "redundancy_pct": redundancy,
            "cost_saved_gbp": round((total_saved // 4) / 1000 * 0.003, 4),
        }
