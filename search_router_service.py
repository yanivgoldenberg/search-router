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


import threading
import uuid as _uuid

_RESEARCH_JOBS: dict[str, dict[str, Any]] = {}
_RESEARCH_JOBS_LOCK = threading.Lock()


def _run_research_bg(job_id: str, req_dict: dict[str, Any]) -> None:
    from research.persist import save_job_state
    try:
        req = ResearchRequest(**req_dict)
        report = run_research(req)
        md = render_markdown(report)
        report_dict = report.model_dump()
        save_job_state(job_id, "complete", report=report_dict, markdown=md)
        with _RESEARCH_JOBS_LOCK:
            _RESEARCH_JOBS[job_id] = {"status": "complete", "report": report_dict, "markdown": md}
    except Exception as e:
        logger.exception("[research-bg] failed")
        save_job_state(job_id, "error", error=str(e))
        with _RESEARCH_JOBS_LOCK:
            _RESEARCH_JOBS[job_id] = {"status": "error", "error": str(e)}


@app.post("/v1/research")
def research_endpoint(payload: dict[str, Any]) -> dict[str, Any]:
    """Deep research pipeline. Sync mode (waits up to ~95s).
    For longer runs use POST /v1/research/start + GET /v1/research/{job_id}."""
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


@app.post("/v1/research/start")
def research_start(payload: dict[str, Any]) -> dict[str, Any]:
    """Kick off research in background. Returns job_id immediately."""
    if not RESEARCH_AVAILABLE:
        raise HTTPException(status_code=503, detail="research module unavailable")
    try:
        ResearchRequest(**payload)  # validate
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"invalid request: {e}") from e
    job_id = str(_uuid.uuid4())[:16]
    from research.persist import save_job_state
    save_job_state(job_id, "running")
    with _RESEARCH_JOBS_LOCK:
        _RESEARCH_JOBS[job_id] = {"status": "running"}
    threading.Thread(target=_run_research_bg, args=(job_id, payload), daemon=True).start()
    return {"job_id": job_id, "status": "running", "poll": f"/v1/research/{job_id}"}


@app.get("/v1/research/history")
def research_history(q: str = Query(..., description="full-text query"), limit: int = 20) -> dict[str, Any]:
    """Search past research sessions by full-text over question + report markdown."""
    if not RESEARCH_AVAILABLE:
        raise HTTPException(status_code=503, detail="research module unavailable")
    from research.persist import search_history
    return {"q": q, "results": search_history(q, limit=limit)}


@app.get("/v1/research/session/{session_id}")
def research_session(session_id: str) -> dict[str, Any]:
    """Fetch a saved research session by its UUID."""
    if not RESEARCH_AVAILABLE:
        raise HTTPException(status_code=503, detail="research module unavailable")
    from research.persist import get_session
    out = get_session(session_id)
    if out is None:
        raise HTTPException(status_code=404, detail="session not found")
    return out


@app.get("/v1/research/{job_id}")
def research_poll(job_id: str) -> dict[str, Any]:
    from research.persist import load_job_state
    with _RESEARCH_JOBS_LOCK:
        job = _RESEARCH_JOBS.get(job_id)
    if job is None:
        job = load_job_state(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return {"job_id": job_id, **job}


@app.get("/v1/research/{job_id}/markdown")
def research_poll_markdown(job_id: str) -> Any:
    from fastapi.responses import PlainTextResponse
    from research.persist import load_job_state
    with _RESEARCH_JOBS_LOCK:
        job = _RESEARCH_JOBS.get(job_id)
    if job is None:
        job = load_job_state(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    if job.get("status") == "running":
        return PlainTextResponse(f"# Job {job_id} still running...", status_code=202)
    if job.get("status") == "error":
        return PlainTextResponse(f"# Job {job_id} errored\n\n{job.get('error')}", status_code=500)
    return PlainTextResponse(job.get("markdown", ""))


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




@app.post("/v1/recall")
def recall_endpoint(payload: dict[str, Any]) -> dict[str, Any]:
    """Cross-session recall: search past research for relevant findings."""
    if not RESEARCH_AVAILABLE:
        raise HTTPException(status_code=503, detail="research module unavailable")
    q = payload.get("q") or payload.get("query") or ""
    if not q:
        raise HTTPException(status_code=400, detail="missing q")
    from research.persist import search_history
    return {"q": q, "results": search_history(q, limit=int(payload.get("limit", 20)))}


@app.post("/v1/watch")
def watch_create_endpoint(payload: dict[str, Any]) -> dict[str, Any]:
    """Create a recurring research watch. {topic, mode, schedule, alert_url}.
    schedule is a cron-style string for n8n consumption."""
    if not RESEARCH_AVAILABLE:
        raise HTTPException(status_code=503, detail="research module unavailable")
    from research.persist import watch_create
    topic = payload.get("topic", "").strip()
    if not topic:
        raise HTTPException(status_code=400, detail="missing topic")
    try:
        wid = watch_create(
            topic=topic,
            mode=payload.get("mode", "general"),
            schedule=payload.get("schedule", "0 9 * * 1"),
            alert_url=payload.get("alert_url", ""),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"watch_create raised: {e}") from e
    if wid is None:
        raise HTTPException(status_code=500, detail="watch_create returned None — check container logs for [persist] watch_create failed")
    return {"watch_id": wid, "topic": topic, "schedule": payload.get("schedule", "0 9 * * 1")}


@app.get("/v1/watch")
def watch_list_endpoint() -> dict[str, Any]:
    if not RESEARCH_AVAILABLE:
        raise HTTPException(status_code=503, detail="research module unavailable")
    from research.persist import watch_list
    return {"watches": watch_list()}


@app.delete("/v1/watch/{watch_id}")
def watch_delete_endpoint(watch_id: str) -> dict[str, Any]:
    if not RESEARCH_AVAILABLE:
        raise HTTPException(status_code=503, detail="research module unavailable")
    from research.persist import watch_delete
    deleted = watch_delete(watch_id)
    return {"deleted": deleted}




if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "8300")),
        log_level="info",
    )
