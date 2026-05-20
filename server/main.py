
"""
ContextLens Proxy Server

Drop-in replacement for the Anthropic API.
Change one line in your app:

Before: ANTHROPIC_BASE_URL=https://api.anthropic.com
After:  ANTHROPIC_BASE_URL=https://your-server:8080

That's it. All API calls are now automatically compressed.
"""
from collections import defaultdict
import time

# Rate limiting — per API key
_request_counts: dict = defaultdict(list)
RATE_LIMIT = 60  # requests per minute


def check_rate_limit(api_key: str) -> bool:
    """Returns True if request is allowed, False if rate limited."""
    now = time.time()
    window = 60  # 1 minute window
    
    # Clean old requests
    _request_counts[api_key] = [
        t for t in _request_counts[api_key] 
        if now - t < window
    ]
    
    # Check limit
    if len(_request_counts[api_key]) >= RATE_LIMIT:
        return False
    
    # Record request
    _request_counts[api_key].append(now)
    return True
from fastapi import FastAPI, Request
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
from fastapi.responses import JSONResponse
import httpx
import json
import os
from pathlib import Path
from contextlens.core import ContextLens
from contextlens.router import SmartRouter

app = FastAPI(
    title="ContextLens Proxy",
    description="Drop-in Anthropic API proxy with automatic context compression",
    version="0.1.0"
)

# One engine per session (keyed by API key)

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
_engines: dict[str, ContextLens] = {}

ANTHROPIC_BASE = "https://api.anthropic.com"
OPENAI_BASE = "https://api.openai.com"

def detect_provider(model: str, headers: dict) -> str:
    """Auto-detect provider from model name or headers."""
    model = model.lower()
    if any(x in model for x in ["gpt", "o1", "o3", "o4"]):
        return "openai"
    elif any(x in model for x in ["claude"]):
        return "anthropic"
    elif any(x in model for x in ["gemini"]):
        return "google"
    elif any(x in model for x in ["llama", "deepseek", "grok"]):
        return "other"
    # Check authorization header as fallback
    auth = headers.get("authorization", "")
    if auth.startswith("Bearer sk-"):
        return "openai"
    return "anthropic"  # default
STATS_DIR = Path("stats")
STATS_DIR.mkdir(exist_ok=True)
_router = SmartRouter()

def get_engine(api_key: str) -> ContextLens:
    """Get or create a compression engine for this API key."""
    if api_key not in _engines:
        _engines[api_key] = ContextLens(
            budget="balanced",
            enable_semantic=True
        )
    return _engines[api_key]


def save_stats(api_key: str, stats: dict):
    """Persist stats to disk so they survive restarts."""
    # Use hash of key for filename — never store raw API keys
    import hashlib
    key_hash = hashlib.md5(api_key.encode()).hexdigest()[:12]
    stats_file = STATS_DIR / f"{key_hash}.json"

    existing = {}
    if stats_file.exists():
        try:
            existing = json.loads(stats_file.read_text())
        except Exception:
            existing = {}

    # Accumulate totals
    existing["calls"] = stats.get("calls", 0)
    existing["chars_saved"] = stats.get("chars_saved", 0)
    existing["tokens_saved_estimate"] = stats.get("tokens_saved_estimate", 0)
    existing["redundancy_pct"] = stats.get("redundancy_pct", 0)

    stats_file.write_text(json.dumps(existing, indent=2))


def load_stats(api_key: str) -> dict:
    """Load persisted stats for this API key."""
    import hashlib
    key_hash = hashlib.md5(api_key.encode()).hexdigest()[:12]
    stats_file = STATS_DIR / f"{key_hash}.json"

    if stats_file.exists():
        try:
            return json.loads(stats_file.read_text())
        except Exception:
            pass
    return {}


@app.get("/")
async def root():
    return {
        "service": "ContextLens Proxy",
        "version": "0.1.0",
        "status": "running",
        "docs": "/docs"
    }


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/v1/messages")
async def proxy_messages(request: Request):
    """
    Intercept Anthropic messages, compress, forward.
    """
    api_key = request.headers.get("x-api-key", "")
    if not api_key:
        raise HTTPException(status_code=401, detail="Missing x-api-key header")
	# Rate limiting
    if not check_rate_limit(api_key):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Max 60 requests per minute."
        )
    body = await request.json()
    # Smart routing — find minimum viable model
    requested_model = body.get("model", "")
    if requested_model:
        routed_model = await _router.route(
            body.get("messages", []),
            requested_model,
            {"anthropic": api_key},
            force_route=True
        )
        body["model"] = routed_model
    engine = get_engine(api_key)

    # Compress the messages
    messages = body.get("messages", [])
    if messages:
        result = engine.compress(messages)
        body["messages"] = result.compressed_messages

        if result.tokens_estimated_saved > 0:
            print(
                f"[ContextLens] "
                f"{result.original_chars} → {result.compressed_chars} chars | "
                f"~{result.tokens_estimated_saved} tokens saved | "
                f"{result.redundancy_pct}% redundancy"
            )

        # Persist stats
        save_stats(api_key, engine.session_stats)

    # Forward to Anthropic with error handling
    provider = detect_provider(body.get("model", ""), dict(request.headers))
    
    if provider == "openai":
        forward_url = f"{OPENAI_BASE}/v1/chat/completions"
        forward_headers = {
            "Authorization": request.headers.get("authorization", f"Bearer {api_key}"),
            "content-type": "application/json",
        }
    else:
        forward_url = f"{ANTHROPIC_BASE}/v1/messages"
        forward_headers = {
            "x-api-key": api_key,
            "anthropic-version": request.headers.get("anthropic-version", "2023-06-01"),
            "content-type": "application/json",
        }

    try:
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(
                forward_url,
                headers=forward_headers,
                json=body,
            )
        
        # Add savings header so user can see compression worked
        tokens_saved = result.tokens_estimated_saved if messages else 0
        redundancy = result.redundancy_pct if messages else 0
        
        response_data = response.json()
        headers_out = {
            "X-ContextLens-Tokens-Saved": str(tokens_saved),
            "X-ContextLens-Redundancy": f"{redundancy}%",
            "X-ContextLens-Version": "0.1.0",
        }
        
        return JSONResponse(
            content=response_data,
            status_code=response.status_code,
            headers=headers_out
        )
    except httpx.TimeoutException:
        raise HTTPException(
            status_code=504,
            detail="Upstream API timeout. Please try again."
        )
    except httpx.ConnectError:
        raise HTTPException(
            status_code=502,
            detail="Cannot reach upstream API. Please try again."
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Proxy error: {str(e)}"
        )

@app.get("/v1/stats")
async def stats(request: Request):
    """Return compression stats for this API key."""
    api_key = request.headers.get("x-api-key", "")
    if not api_key:
        raise HTTPException(status_code=401, detail="Missing x-api-key header")

    # Live stats from memory
    if api_key in _engines:
        live = _engines[api_key].session_stats
        save_stats(api_key, live)
        return live

    # Persisted stats from disk
    persisted = load_stats(api_key)
    if persisted:
        return persisted

    return {"message": "No stats yet — make some API calls first"}



from datetime import datetime
_captured_turns: list = []

@app.post("/ingest")
async def ingest_turn(request: Request):
    body = await request.json()
    turn = {
        "id": len(_captured_turns) + 1,
        "conv_id": body.get("convId", "unknown"),
        "platform": body.get("platform", "unknown"),
        "assistant_text": body.get("assistantText", ""),
        "user_text": body.get("userText", ""),
        "ts": body.get("ts", datetime.utcnow().isoformat()),
        "received_at": datetime.utcnow().isoformat()
    }
    _captured_turns.append(turn)
    print(f"[ContextLens] Ingested turn from {turn['platform']} | conv: {turn['conv_id'][:8]}")
    return {"status": "ok", "id": turn["id"], "turns_stored": len(_captured_turns)}

@app.get("/ingest")
async def get_turns():
    return {"turns": _captured_turns, "total": len(_captured_turns)}


import asyncpg

_db_pool = None

async def get_db_pool():
    global _db_pool
    if _db_pool is None:
        _db_pool = await asyncpg.create_pool(
            dsn="postgresql://aria:aria123@localhost:5432/aria_db",
            min_size=1,
            max_size=5
        )
    return _db_pool

@app.post("/ingest2")
@limiter.limit("60/minute")
async def ingest_turn_db(request: Request):
    body = await request.json()
    try:
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO contextlens_turns (conv_id, platform, assistant_text, user_text, ts)
                VALUES ($1, $2, $3, $4, $5)
            """,
                body.get("convId", "unknown"),
                body.get("platform", "unknown"),
                body.get("assistantText", ""),
                body.get("userText", ""),
                __import__("datetime").datetime.fromisoformat(body["ts"].replace("Z","+00:00")) if body.get("ts") else __import__("datetime").datetime.utcnow()
            )
        return {"status": "ok", "storage": "postgresql"}
    except Exception as e:
        return {"status": "error", "detail": str(e)}

@app.get("/ingest2")
async def get_turns_db(limit: int = 50):
    try:
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM contextlens_turns ORDER BY received_at DESC LIMIT $1", limit
            )
        return {"turns": [dict(r) for r in rows], "total": len(rows)}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


from sentence_transformers import SentenceTransformer
import numpy as np

_embed_model = None

def get_embed_model():
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    return _embed_model

def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

@app.post("/compress")
@limiter.limit("30/minute")
async def compress_messages(request: Request):
    body = await request.json()
    messages = body.get("messages", [])
    threshold = body.get("threshold", 0.85)

    if len(messages) < 3:
        return {"compressed": messages, "removed": 0, "redundancy_pct": 0}

    model = get_embed_model()
    texts = [m.get("content", "") for m in messages]
    embeddings = model.encode(texts)

    kept = []
    kept_embeddings = []
    removed = 0

    for i, (msg, emb) in enumerate(zip(messages, embeddings)):
        is_last = i == len(messages) - 1
        if is_last:
            kept.append(msg)
            continue

        is_duplicate = False
        for kept_emb in kept_embeddings:
            if cosine_similarity(emb, kept_emb) > threshold:
                is_duplicate = True
                removed += 1
                break

        if not is_duplicate:
            kept.append(msg)
            kept_embeddings.append(emb)

    original_chars = sum(len(m.get("content", "")) for m in messages)
    compressed_chars = sum(len(m.get("content", "")) for m in kept)
    redundancy_pct = round((original_chars - compressed_chars) / original_chars * 100) if original_chars > 0 else 0

    return {
        "compressed": kept,
        "removed": removed,
        "original_count": len(messages),
        "compressed_count": len(kept),
        "redundancy_pct": redundancy_pct,
        "tokens_saved_estimate": round((original_chars - compressed_chars) / 4)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8081)
