"""
ContextLens Proxy Server

Drop-in replacement for the Anthropic API.
Change one line in your app:

Before: ANTHROPIC_BASE_URL=https://api.anthropic.com
After:  ANTHROPIC_BASE_URL=https://your-server:8080

That's it. All API calls are now automatically compressed.
"""

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import httpx
import json
from contextlens.core import ContextLens

app = FastAPI(
    title="ContextLens Proxy",
    description="Drop-in Anthropic API proxy with automatic context compression",
    version="0.1.0"
)

# One engine per session (keyed by API key)
_engines: dict[str, ContextLens] = {}

ANTHROPIC_BASE = "https://api.anthropic.com"


def get_engine(api_key: str) -> ContextLens:
    """Get or create a compression engine for this API key."""
    if api_key not in _engines:
        _engines[api_key] = ContextLens(budget="balanced")
    return _engines[api_key]


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
    # Get the API key from headers
    api_key = request.headers.get("x-api-key", "")
    if not api_key:
        raise HTTPException(status_code=401, detail="Missing x-api-key header")

    # Read the request body
    body = await request.json()

    # Get compression engine for this user
    engine = get_engine(api_key)

    # Compress the messages
    messages = body.get("messages", [])
    if messages:
        result = engine.compress(messages)
        body["messages"] = result.compressed_messages

        # Log savings
        if result.tokens_estimated_saved > 0:
            print(
                f"[ContextLens] Compressed: "
                f"{result.original_chars} → {result.compressed_chars} chars | "
                f"~{result.tokens_estimated_saved} tokens saved | "
                f"{result.redundancy_pct}% redundancy"
            )

    # Forward to Anthropic
    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(
            f"{ANTHROPIC_BASE}/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": request.headers.get(
                    "anthropic-version", "2023-06-01"
                ),
                "content-type": "application/json",
            },
            json=body,
        )

    return JSONResponse(
        content=response.json(),
        status_code=response.status_code
    )


@app.get("/v1/stats")
async def stats(request: Request):
    """Return compression stats for this API key."""
    api_key = request.headers.get("x-api-key", "")
    if not api_key or api_key not in _engines:
        return {"message": "No stats yet — make some API calls first"}

    return _engines[api_key].session_stats


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)