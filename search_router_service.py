#!/usr/bin/env python3
"""FastAPI service exposing search_router as HTTP.

One unified endpoint every agent calls (OpenClaw, Hermes, LLM Waterfall, Codex, n8n,
Claude Code MCP, scripts). Replaces direct provider SDK calls everywhere.

Endpoints:
  GET  /healthz                       - liveness
  POST /v1/search {type, q, num, gl, hl}
  GET  /v1/search?type=&q=&num=...    - convenience GET
  GET  /v1/stats                      - provider load + metrics
  POST /v1/cache/clear                - evict one entry
  GET  /v1/types                      - list valid search types
  POST /v1/openai/chat/completions    - OpenAI-compat tool stub (function call shim)

Run locally:
    pip install fastapi uvicorn python-dotenv redis requests
    python3 scripts/search_router_service.py            # starts on :8300

Run as Docker (production):
    See Dockerfile.search-router + Coolify deploy notes in
    docs/setup/2026-05-02-search-router-service.md
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).parent))
from search_router import SearchRouter

try:
    from research.pipeline import run_research
    from research.models import ResearchRequest
    from research.persist import render_markdown
    RESEARCH_AVAILABLE = True
except Exception as _research_e:
    RESEARCH_AVAILABLE = False
    _research_import_error = str(_research_e)

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Search Router", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

router = SearchRouter()

VALID_TYPES = ["serp", "news", "ai", "deep", "scrape", "code", "academic", "social"]


class SearchRequest(BaseModel):
    type: Literal["serp", "news", "ai", "deep", "scrape", "code", "academic", "social"]
    q: str = Field(..., description="query string (or URL for scrape)")
    num: int = Field(10, ge=1, le=50)
    gl: str = "us"
    hl: str = "en"
    no_cache: bool = False


class CacheClearRequest(BaseModel):
    type: str
    q: str
    num: int = 10
    gl: str = "us"
    hl: str = "en"


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/types")
def types() -> dict[str, Any]:
    return {
        "types": VALID_TYPES,
        "descriptions": {
            "serp": "Google-style organic web results",
            "news": "News articles, fresh",
            "ai": "AI-grounded answer + citations",
            "deep": "Multi-source synthesis / research",
            "scrape": "URL → markdown / clean text",
            "code": "Code search across GitHub + Stack Overflow",
            "academic": "Scholarly papers (OpenAlex/arXiv/PubMed/Crossref)",
            "social": "Reddit + HackerNews",
        },
    }


@app.get("/v1/stats")
def stats() -> dict[str, Any]:
    return router.stats()


@app.post("/v1/search")
def search_post(req: SearchRequest) -> dict[str, Any]:
    out = router.run(req.type, req.q, num=req.num, gl=req.gl, hl=req.hl, bypass_cache=req.no_cache)
    if "error" in out:
        raise HTTPException(status_code=502, detail=out)
    return out


@app.get("/v1/search")
def search_get(
    type: str = Query(...),
    q: str = Query(...),
    num: int = Query(10, ge=1, le=50),
    gl: str = "us",
    hl: str = "en",
    no_cache: bool = False,
) -> dict[str, Any]:
    if type not in VALID_TYPES:
        raise HTTPException(status_code=400, detail=f"invalid type. Valid: {VALID_TYPES}")
    out = router.run(type, q, num=num, gl=gl, hl=hl, bypass_cache=no_cache)
    if "error" in out:
        raise HTTPException(status_code=502, detail=out)
    return out


@app.post("/v1/cache/clear")
def cache_clear(req: CacheClearRequest) -> dict[str, Any]:
    deleted = router.cache_clear(req.type, req.q, req.num, req.gl, req.hl)
    return {"deleted": deleted}


@app.post("/v1/openai/chat/completions")
def openai_compat(payload: dict[str, Any]) -> dict[str, Any]:
    """Minimal OpenAI-compat shim for tool-use calls.

    Expects payload like {messages: [...], tools: [{function: {name: 'search_web', ...}}]}.
    If the last user message asks for search-y info, returns a tool call.
    Otherwise returns a passthrough message saying it's a search-only service.
    """
    messages = payload.get("messages", [])
    last = messages[-1]["content"] if messages else ""
    return {
        "id": "search-router-1",
        "object": "chat.completion",
        "model": "search-router",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "search_web",
                                "arguments": f'{{"q": "{last[:200]}", "type": "serp", "num": 10}}',
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
    }


@app.post("/v1/research")
def research_endpoint(payload: dict[str, Any]) -> dict[str, Any]:
    """Deep research pipeline. POST {q, mode, max_sources, iterations, save}."""
    if not RESEARCH_AVAILABLE:
        raise HTTPException(status_code=503, detail=f"research module unavailable: {_research_import_error}")
    try:
        req = ResearchRequest(**payload)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"invalid request: {e}") from e
    try:
        report = run_research(req)
    except Exception as e:
        logger.exception("research pipeline failed")
        raise HTTPException(status_code=500, detail=f"research failed: {e}") from e
    return {
        "report": report.model_dump(),
        "markdown": render_markdown(report),
    }


@app.get("/v1/research/markdown")
def research_markdown(
    q: str = Query(...),
    mode: str = "general",
    max_sources: int = 30,
    iterations: int = 1,
) -> Any:
    if not RESEARCH_AVAILABLE:
        raise HTTPException(status_code=503, detail="research module unavailable")
    from fastapi.responses import PlainTextResponse
    req = ResearchRequest(q=q, mode=mode, max_sources=max_sources, iterations=iterations)
    report = run_research(req)
    return PlainTextResponse(render_markdown(report))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "8300")),
        log_level="info",
    )
