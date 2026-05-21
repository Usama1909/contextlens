

class ExtractiveSummariser:
    """
    Compresses a message to its key sentences using TF-IDF scoring.
    No LLM calls — fast, local, zero API cost.
    """

    def summarise(self, text: str, max_sentences: int = 2) -> str:
        """Extract the most important sentences from text."""
        import re

        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        if len(sentences) <= max_sentences:
            return text

        # Score sentences by word frequency (TF-IDF approximation)
        words = text.lower().split()
        word_freq = {}
        for word in words:
            word = re.sub(r'[^a-z0-9]', '', word)
            if len(word) > 3:
                word_freq[word] = word_freq.get(word, 0) + 1

        # Normalize
        max_freq = max(word_freq.values()) if word_freq else 1
        word_freq = {w: f/max_freq for w, f in word_freq.items()}

        # Score each sentence
        scored = []
        for i, sentence in enumerate(sentences):
            words_in_sent = sentence.lower().split()
            score = sum(word_freq.get(re.sub(r'[^a-z0-9]', '', w), 0) for w in words_in_sent)
            # Boost first and last sentences
            if i == 0 or i == len(sentences) - 1:
                score *= 1.5
            scored.append((score, i, sentence))

        # Keep top sentences in original order
        top = sorted(scored, reverse=True)[:max_sentences]
        top = sorted(top, key=lambda x: x[1])
        summary = ' '.join(s[2] for s in top)

        return f"[SUMMARY] {summary}"

    def should_summarise(self, text: str) -> bool:
        """Only summarise if text is long enough to benefit."""
        return len(text) > 400 and len(text.split('.')) > 3

    def max_sentences_for(self, text: str) -> int:
        """Determine how many sentences to keep based on length."""
        if len(text) > 1000:
            return 2
        return 1

"""
Semantic Triage Engine

The core differentiator. Scores every message by meaning,
not just exact text match. Uses all-MiniLM-L6-v2 locally —
zero external API calls, runs in ~2ms per message.

Thresholds:
    > 0.95  — Near-identical meaning, compress
    0.5-0.95 — Related but different, summarise reference
    < 0.5   — Different topic, keep in full
"""

import numpy as np
from typing import Optional


class SemanticTriage:
    def __init__(self, threshold_compress: float = 0.95, threshold_keep: float = 0.35):
        self.threshold_compress = threshold_compress
        self.threshold_keep = threshold_keep
        self._model = None
        self._cache: dict[str, np.ndarray] = {}

    def _load_model(self):
        """Lazy load — only loads when first needed."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer('all-MiniLM-L6-v2')

    def embed(self, text: str) -> np.ndarray:
        """Get embedding for text, with caching."""
        self._load_model()
        if text not in self._cache:
            self._cache[text] = self._model.encode([text])[0]
        return self._cache[text]

    def similarity(self, text1: str, text2: str) -> float:
        """Cosine similarity between two texts."""
        e1 = self.embed(text1)
        e2 = self.embed(text2)
        norm = np.linalg.norm(e1) * np.linalg.norm(e2)
        if norm == 0:
            return 0.0
        return float(np.dot(e1, e2) / norm)

    def score_messages(self, messages: list[dict], current_query: str) -> list[dict]:
        """
        Score every message against the current query.
        Returns messages with scores attached.

        Score meanings:
            > 0.95  → compress (near-identical meaning)
            0.5-0.95 → keep (related, useful context)
            < 0.5   → drop (irrelevant to current query)
        """
        if not messages or not current_query:
            return messages

        self._load_model()
        scored = []

        for msg in messages:
            content = str(msg.get("content", ""))
            role = msg.get("role", "")

            # System messages always kept — never scored
            if role == "system":
                scored.append({**msg, "_score": 1.0, "_action": "keep"})
                continue

            # Score against current query
            score = self.similarity(content, current_query)

            if score >= self.threshold_compress:
                action = "compress"
            elif score >= self.threshold_keep:
                action = "keep"
            else:
                action = "drop"

            scored.append({**msg, "_score": round(score, 3), "_action": action})

        return scored

    def apply_triage(self, messages: list[dict], current_query: str) -> tuple[list[dict], dict]:
        """
        Full triage pipeline. Returns compressed messages + stats.
        """
        if len(messages) < 3:
            return messages, {"method": "skipped", "reason": "too_short"}

        scored = self.score_messages(messages, current_query)

        result = []
        seen_compressed: list[str] = []
        dropped = 0
        compressed = 0
        kept = 0

        for msg in scored:
            action = msg.pop("_action", "keep")
            score = msg.pop("_score", 1.0)
            content = str(msg.get("content", ""))

            if action == "keep":
                result.append(msg)
                kept += 1

            elif action == "compress":
                # Only keep first occurrence of near-identical content
                already_seen = any(
                    self.similarity(content, seen) > self.threshold_compress
                    for seen in seen_compressed
                )
                if not already_seen:
                    result.append(msg)
                    seen_compressed.append(content)
                    kept += 1
                else:
                    compressed += 1

            elif action == "drop":
                # Instead of dropping, summarise long messages, drop short ones
                summariser = ExtractiveSummariser()
                if summariser.should_summarise(content):
                    n = summariser.max_sentences_for(content)
                    summary = summariser.summarise(content, max_sentences=n)
                    result.append({**msg, "content": summary})
                    compressed += 1
                else:
                    # Short irrelevant message — drop it
                    dropped += 1

        stats = {
            "method": "semantic",
            "kept": kept,
            "compressed": compressed,
            "dropped": dropped,
            "total": len(messages),
            "reduction_pct": round((1 - len(result) / len(messages)) * 100, 1) if messages else 0
        }

        return result, stats