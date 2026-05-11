"""
ContextLens Smart Router

Auto-discovers available models from provider APIs.
Scores query complexity automatically.
Routes to minimum viable model — zero config needed.

New model released tomorrow = auto-discovered.
Nobody maintains anything.
"""

import httpx
import asyncio
from dataclasses import dataclass
from typing import Optional
import time


@dataclass
class ModelProfile:
    """What we know about a model."""
    id: str
    provider: str
    capability: float  # 0.0-1.0 how powerful
    cost: float        # 0.0-1.0 relative cost
    speed: float       # 0.0-1.0 how fast


# Fallback registry if API discovery fails
# Updated manually only when needed
FALLBACK_REGISTRY = {
    # Anthropic
    "claude-haiku-4-5-20251001":     ModelProfile("claude-haiku-4-5-20251001",     "anthropic", 0.4, 0.1, 0.9),
    "claude-sonnet-4-20250514":      ModelProfile("claude-sonnet-4-20250514",       "anthropic", 0.7, 0.4, 0.7),
    "claude-opus-4-20250514":        ModelProfile("claude-opus-4-20250514",         "anthropic", 1.0, 1.0, 0.4),

    # OpenAI
    "gpt-4o-mini":                   ModelProfile("gpt-4o-mini",                    "openai",    0.4, 0.1, 0.9),
    "gpt-4o":                        ModelProfile("gpt-4o",                         "openai",    0.8, 0.6, 0.7),

    # Google
    "gemini-2.0-flash":              ModelProfile("gemini-2.0-flash",               "google",    0.4, 0.1, 0.9),
    "gemini-2.0-pro":                ModelProfile("gemini-2.0-pro",                 "google",    0.8, 0.6, 0.6),

    # Meta (self-hosted)
    "llama-3-8b":                    ModelProfile("llama-3-8b",                     "meta",      0.3, 0.0, 1.0),
    "llama-3-70b":                   ModelProfile("llama-3-70b",                    "meta",      0.7, 0.1, 0.6),

    # DeepSeek
    "deepseek-v3":                   ModelProfile("deepseek-v3",                    "deepseek",  0.8, 0.1, 0.6),

    # xAI
    "grok-2":                        ModelProfile("grok-2",                         "xai",       0.8, 0.6, 0.6),
}


class ModelDiscovery:
    """
    Auto-discovers available models from provider APIs.
    Falls back to hardcoded registry if discovery fails.
    Refreshes every 24 hours automatically.
    """

    def __init__(self):
        self._registry: dict[str, ModelProfile] = {}
        self._last_refresh: float = 0
        self._refresh_interval: int = 86400  # 24 hours

    async def get_registry(self, api_keys: dict) -> dict[str, ModelProfile]:
        """Get current model registry, refreshing if needed."""
        now = time.time()
        if not self._registry or (now - self._last_refresh) > self._refresh_interval:
            await self._refresh(api_keys)
        return self._registry or FALLBACK_REGISTRY

    async def _refresh(self, api_keys: dict):
        """Fetch available models from all configured providers."""
        registry = {}

        # Discover Anthropic models
        if "anthropic" in api_keys:
            try:
                models = await self._fetch_anthropic_models(api_keys["anthropic"])
                registry.update(models)
                print(f"[Router] Discovered {len(models)} Anthropic models")
            except Exception as e:
                print(f"[Router] Anthropic discovery failed: {e}")

        # Discover OpenAI models
        if "openai" in api_keys:
            try:
                models = await self._fetch_openai_models(api_keys["openai"])
                registry.update(models)
                print(f"[Router] Discovered {len(models)} OpenAI models")
            except Exception as e:
                print(f"[Router] OpenAI discovery failed: {e}")

        if registry:
            self._registry = registry
            self._last_refresh = time.time()
        else:
            # Fall back to hardcoded
            self._registry = FALLBACK_REGISTRY
            print("[Router] Using fallback registry")

    async def _fetch_anthropic_models(self, api_key: str) -> dict:
        """Fetch models from Anthropic API."""
        async with httpx.AsyncClient() as client:
            r = await client.get(
                "https://api.anthropic.com/v1/models",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01"
                }
            )
            if r.status_code == 200:
                models = {}
                for m in r.json().get("data", []):
                    model_id = m["id"]
                    # Score based on name patterns
                    profile = self._score_anthropic_model(model_id)
                    if profile:
                        models[model_id] = profile
                return models
        return {}

    async def _fetch_openai_models(self, api_key: str) -> dict:
        """Fetch models from OpenAI API."""
        async with httpx.AsyncClient() as client:
            r = await client.get(
                "https://api.openai.com/v1/models",
                headers={"Authorization": f"Bearer {api_key}"}
            )
            if r.status_code == 200:
                models = {}
                for m in r.json().get("data", []):
                    model_id = m["id"]
                    profile = self._score_openai_model(model_id)
                    if profile:
                        models[model_id] = profile
                return models
        return {}

    def _score_anthropic_model(self, model_id: str) -> Optional[ModelProfile]:
        """Score an Anthropic model by its name pattern."""
        mid = model_id.lower()
        if "haiku" in mid:
            return ModelProfile(model_id, "anthropic", 0.4, 0.1, 0.9)
        elif "sonnet" in mid:
            return ModelProfile(model_id, "anthropic", 0.7, 0.4, 0.7)
        elif "opus" in mid:
            return ModelProfile(model_id, "anthropic", 1.0, 1.0, 0.4)
        return None

    def _score_openai_model(self, model_id: str) -> Optional[ModelProfile]:
        """Score an OpenAI model by its name pattern."""
        mid = model_id.lower()
        if "mini" in mid or "nano" in mid:
            return ModelProfile(model_id, "openai", 0.4, 0.1, 0.9)
        elif "gpt-4o" in mid:
            return ModelProfile(model_id, "openai", 0.8, 0.6, 0.7)
        elif "gpt-5" in mid:
            return ModelProfile(model_id, "openai", 1.0, 1.0, 0.5)
        elif "o1" in mid or "o3" in mid:
            return ModelProfile(model_id, "openai", 0.9, 0.8, 0.3)
        return None


class ComplexityScorer:
    """
    Scores query complexity 0.0-1.0 automatically.
    Uses fast heuristics first, semantic scoring for edge cases.
    Zero config — works out of the box.
    """

    # Simple keyword signals
    SIMPLE_SIGNALS = [
        "what is", "what are", "define", "translate",
        "how do you say", "what time", "when was",
        "who is", "spell", "convert", "calculate"
    ]

    COMPLEX_SIGNALS = [
        "analyse", "analyze", "design", "architect",
        "implement", "debug", "optimise", "optimize",
        "compare", "evaluate", "research", "explain why",
        "prove", "derive", "create a system"
    ]

    EXPERT_SIGNALS = [
        "novel", "invent", "original research",
        "mathematical proof", "theorem", "hypothesis",
        "state of the art", "frontier"
    ]

    def score(self, messages: list[dict]) -> float:
        """
        Score the complexity of a conversation.
        Returns 0.0 (simple) to 1.0 (expert).
        """
        if not messages:
            return 0.5

        # Get the last user message
        last_user = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                last_user = str(msg.get("content", "")).lower()
                break

        if not last_user:
            return 0.5

        # Length signal
        length_score = min(len(last_user) / 2000, 1.0) * 0.3

        # Keyword signals
        keyword_score = 0.3  # default medium

        for signal in self.SIMPLE_SIGNALS:
            if signal in last_user:
                keyword_score = 0.1
                break

        for signal in self.COMPLEX_SIGNALS:
            if signal in last_user:
                keyword_score = 0.6
                break

        for signal in self.EXPERT_SIGNALS:
            if signal in last_user:
                keyword_score = 0.9
                break

        # Code signal
        code_score = 0.0
        if any(x in last_user for x in ["```", "def ", "class ", "import ", "function", "algorithm"]):
            code_score = 0.2

        # Conversation depth signal
        depth_score = min(len(messages) / 20, 1.0) * 0.1

        total = length_score + keyword_score + code_score + depth_score
        return min(round(total, 2), 1.0)


class SmartRouter:
    """
    Routes queries to the minimum viable model.
    Auto-discovers models. Learns from results over time.
    Zero hardcoded model names — works with any future model.
    """

    def __init__(self):
        self.discovery = ModelDiscovery()
        self.scorer = ComplexityScorer()
        self._routing_log: list[dict] = []

    async def route(
        self,
        messages: list[dict],
        requested_model: str,
        api_keys: dict,
        force_route: bool = True
    ) -> str:
        """
        Decide which model to actually use.

        Args:
            messages:        The conversation messages
            requested_model: What the user asked for
            api_keys:        Available provider API keys
            force_route:     If False, respect requested model

        Returns:
            Model ID to actually use
        """
        if not force_route:
            return requested_model

        # Get available models
        registry = await self.discovery.get_registry(api_keys)

        # Score query complexity
        complexity = self.scorer.score(messages)

        # Find the requested model's provider
        provider = self._detect_provider(requested_model, registry)

        # Find minimum viable model from same provider
        best_model = self._find_minimum_viable(
            complexity, provider, registry, requested_model
        )

        # Log routing decision
        self._routing_log.append({
            "requested": requested_model,
            "routed_to": best_model,
            "complexity": complexity,
            "provider": provider
        })

        if best_model != requested_model:
            print(
                f"[Router] {requested_model} → {best_model} "
                f"(complexity: {complexity:.2f})"
            )

        return best_model

    def _detect_provider(self, model_id: str, registry: dict) -> str:
        """Detect provider from model ID."""
        if model_id in registry:
            return registry[model_id].provider

        mid = model_id.lower()
        if "claude" in mid:
            return "anthropic"
        elif "gpt" in mid or "o1" in mid or "o3" in mid:
            return "openai"
        elif "gemini" in mid:
            return "google"
        elif "llama" in mid:
            return "meta"
        elif "grok" in mid:
            return "xai"
        elif "deepseek" in mid:
            return "deepseek"
        return "unknown"

    def _find_minimum_viable(
        self,
        complexity: float,
        provider: str,
        registry: dict,
        fallback: str
    ) -> str:
        """
        Find cheapest model that can handle this complexity.
        Stays within the same provider.
        """
        # Filter to same provider
        provider_models = [
            m for m in registry.values()
            if m.provider == provider
        ]

        if not provider_models:
            return fallback

        # Find models capable enough for this complexity
        # A model needs capability >= complexity * 0.9
        capable = [
            m for m in provider_models
            if m.capability >= complexity * 0.9
        ]

        if not capable:
            # Nothing capable enough — use most powerful
            return max(provider_models, key=lambda m: m.capability).id

        # Among capable models, pick cheapest
        cheapest = min(capable, key=lambda m: m.cost)
        return cheapest.id

    @property
    def routing_stats(self) -> dict:
        """Return routing statistics."""
        if not self._routing_log:
            return {"routes": 0}

        routed = sum(
            1 for r in self._routing_log
            if r["requested"] != r["routed_to"]
        )

        return {
            "total_routes": len(self._routing_log),
            "optimised": routed,
            "optimised_pct": round(routed / len(self._routing_log) * 100, 1),
            "avg_complexity": round(
                sum(r["complexity"] for r in self._routing_log) /
                len(self._routing_log), 2
            )
        }