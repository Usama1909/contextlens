"""
ContextLens State Compiler
Converts conversation into typed state objects.
"""
import re, json, hashlib
from dataclasses import dataclass, field
from typing import Any

@dataclass
class StateObject:
    entities: dict = field(default_factory=dict)
    state: dict = field(default_factory=dict)
    intent: str = "unknown"
    key_facts: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    resolved: bool = False
    tokens_original: int = 0
    tokens_compiled: int = 0
    compression_ratio: float = 0.0
    content_hash: str = ""

    def to_injection_string(self) -> str:
        parts = []
        if self.entities: parts.append(f"stack:{json.dumps(self.entities, separators=(',',':'))}")
        if self.state: parts.append(f"state:{json.dumps(self.state, separators=(',',':'))}")
        if self.key_facts: parts.append(f"facts:{';'.join(self.key_facts[:3])}")
        if self.errors: parts.append(f"errors:{';'.join(self.errors[:2])}")
        return " | ".join(parts)

    def token_count(self) -> int:
        return len(self.to_injection_string().split()) * 4 // 3

class StateCompiler:
    TECH_ENTITIES = {
        "framework": ["fastapi", "flask", "django", "react", "nextjs"],
        "database": ["postgresql", "mysql", "mongodb", "redis", "sqlite"],
        "language": ["python", "javascript", "typescript", "rust", "golang"],
        "platform": ["hetzner", "railway", "aws", "docker", "kubernetes"],
        "ai": ["claude", "gpt", "anthropic", "openai", "gemini", "ollama"],
        "tool": ["nginx", "systemd", "uvicorn", "gunicorn"]
    }

    INTENT_PATTERNS = {
        "debugging": ["error", "bug", "fix", "broken", "failed", "exception"],
        "building": ["build", "create", "implement", "deploy", "setup"],
        "optimising": ["optimize", "improve", "faster", "performance"],
        "explaining": ["explain", "how", "what", "why", "understand"],
    }

    ERROR_PATTERNS = [
        r'\b([45]\d{2})\s+(?:error|Error)\b',
        r'Error:\s*([^\n]+)',
        r'Exception:\s*([^\n]+)',
    ]

    def compile(self, text: str) -> StateObject:
        s = StateObject()
        s.tokens_original = len(text.split()) * 4 // 3
        s.content_hash = hashlib.md5(text.encode()).hexdigest()[:12]
        s.entities = self._extract_entities(text)
        s.state = self._extract_state(text)
        s.intent = self._classify_intent(text)
        s.key_facts = self._extract_facts(text)
        s.errors = self._extract_errors(text)
        s.resolved = any(p in text.lower() for p in ["fixed","working","solved","resolved"])
        s.tokens_compiled = s.token_count()
        if s.tokens_original > 0:
            s.compression_ratio = round(1 - (s.tokens_compiled / s.tokens_original), 3)
        return s

    def _extract_entities(self, text: str) -> dict:
        entities = {}
        text_lower = text.lower()
        for category, terms in self.TECH_ENTITIES.items():
            found = [t for t in terms if t in text_lower]
            if found:
                entities[category] = found[0] if len(found) == 1 else found
        return entities

    def _extract_state(self, text: str) -> dict:
        state = {}
        codes = re.findall(r'\b([45]\d{2})\b', text)
        if codes: state["http_error"] = codes[0]
        ports = re.findall(r'\bport\s+(\d{4,5})\b', text.lower())
        if ports: state["port"] = ports[0]
        paths = re.findall(r'(/[a-zA-Z0-9_/.-]+\.py)', text)
        if paths: state["file"] = paths[0]
        missing = re.findall(r'missing\s+(?:field\s+)?["\']?(\w+)["\']?', text.lower())
        if missing: state["missing_field"] = missing[0]
        ips = re.findall(r'\b(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\b', text)
        if ips: state["ip"] = ips[0]
        return state

    def _classify_intent(self, text: str) -> str:
        text_lower = text.lower()
        scores = {intent: sum(1 for k in keywords if k in text_lower)
                  for intent, keywords in self.INTENT_PATTERNS.items()}
        return max(scores, key=scores.get) if any(scores.values()) else "general"

    def _extract_facts(self, text: str) -> list:
        sentences = [s.strip() for s in re.split(r'[.!?\n]+', text) if 10 < len(s.strip()) < 200]
        scored = []
        for s in sentences:
            score = sum(sum(1 for t in terms if t in s.lower()) for terms in self.TECH_ENTITIES.values())
            score += len(re.findall(r'\d+', s)) * 0.5
            scored.append((score, s))
        facts = [s for _, s in sorted(scored, reverse=True)[:3] if _ > 0]
        # Distil to atomic facts — max 8 words each
        atomic = []
        for fact in facts:
            words = fact.split()
            if len(words) > 8:
                # Keep only noun phrases and technical terms
                import re as _re
                terms = _re.findall(r'[A-Z][a-z]+|[a-z]+(?:Error|Exception|API)|\d+\.\d+|\w+(?:sql|db|py)', fact)
                atomic.append(' '.join(terms[:6]) if terms else ' '.join(words[:6]))
            else:
                atomic.append(fact)
        return atomic

    def _extract_errors(self, text: str) -> list:
        errors = []
        for pattern in self.ERROR_PATTERNS:
            errors.extend([m[:100] for m in re.findall(pattern, text) if m])
        return list(set(errors))[:3]

class ConversationStateEngine:
    def __init__(self):
        self.compiler = StateCompiler()
        self.merged_state = {}
        self.merged_entities = {}
        self.all_facts = []
        self.total_original_tokens = 0
        self.total_compiled_tokens = 0

    def process_turn(self, text: str) -> StateObject:
        s = self.compiler.compile(text)
        self.merged_state.update(s.state)
        self.merged_entities.update(s.entities)
        for fact in s.key_facts:
            if fact not in self.all_facts:
                self.all_facts.append(fact)
        self.total_original_tokens += s.tokens_original
        self.total_compiled_tokens += s.token_count()
        return s

    def get_compiled_context(self) -> str:
        parts = []
        if self.merged_entities:
            parts.append(f"Stack: {json.dumps(self.merged_entities, separators=(',',':'))}")
        if self.merged_state:
            parts.append(f"State: {json.dumps(self.merged_state, separators=(',',':'))}")
        if self.all_facts:
            parts.append(f"Facts: {' | '.join(self.all_facts[:5])}")
        return "\n".join(parts)

    @property
    def compression_ratio(self) -> float:
        if self.total_original_tokens == 0: return 0.0
        return round(1 - (self.total_compiled_tokens / self.total_original_tokens), 3)

    @property
    def tokens_saved(self) -> int:
        return max(0, self.total_original_tokens - self.total_compiled_tokens)
