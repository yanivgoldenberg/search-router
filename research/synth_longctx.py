"""Long-context synthesis (tier='ultra').

Skips per-source extract and feeds raw source bodies directly into a single
long-context call (Llama 4 Scout 10M via OpenRouter, free tier). Lets the
model do its own grounding across all sources at once instead of working
from pre-distilled claims.

Returns the same dict shape as research.synth.synthesize so the pipeline
can swap implementations without changes downstream.
"""
from __future__ import annotations

import logging
import os
from typing import Any

from .llm import LLMError, chat_json
from .models import SourceMeta

logger = logging.getLogger(__name__)


LONGCTX_MODEL = os.environ.get("OPENROUTER_LONGCTX_MODEL", "meta-llama/llama-4-scout:free")
LONGCTX_MAX_CHARS = int(os.environ.get("LONGCTX_MAX_CHARS", "8000000"))
LONGCTX_PER_SOURCE_CHARS = int(os.environ.get("LONGCTX_PER_SOURCE_CHARS", "30000"))
LONGCTX_MAX_TOKENS = int(os.environ.get("LONGCTX_MAX_TOKENS", "8000"))


def _format_sources_block(sources: list[SourceMeta]) -> tuple[str, dict[int, str]]:
    idx_to_url: dict[int, str] = {}
    parts: list[str] = []
    total_chars = 0
    for i, s in enumerate(sources, start=1):
        body = (s.body or "")[:LONGCTX_PER_SOURCE_CHARS]
        if not body:
            continue
        idx_to_url[i] = s.url
        block = (
            f"\n===== SOURCE [{i}] =====\n"
            f"URL: {s.url}\n"
            f"TITLE: {s.title}\n"
            f"PROVIDER: {s.provider}\n"
            f"--- BODY ---\n{body}\n--- END SOURCE [{i}] ---\n"
        )
        if total_chars + len(block) > LONGCTX_MAX_CHARS:
            logger.info("[longctx] hit max char budget at source %d (%d sources packed)", i, len(parts))
            break
        parts.append(block)
        total_chars += len(block)
    return "".join(parts), idx_to_url


def synthesize_longctx(question: str, sources: list[SourceMeta]) -> dict[str, Any]:
    """Long-context synthesis. Returns same shape as synth.synthesize()."""
    bodied = [s for s in sources if s.body]
    if not bodied:
        return {
            "executive_summary": "No source bodies available for long-context synthesis.",
            "sections": [],
            "contradictions": [],
            "gaps": ["All sources fetched without retrievable bodies."],
            "what_would_change_my_mind": [],
            "counter_arguments": [],
        }

    sources_block, _idx_to_url = _format_sources_block(bodied)

    system = (
        "Output JSON only (no prose, no markdown fences before or after). You are a senior research analyst with a 10M-token context. "
        "Synthesize the user's research question using ONLY the source documents provided. Each source has a [N] citation index, use them inline.\n\n"
        "STRICT RULES:\n"
        "- Cite every factual statement using [N] indices that match the SOURCE [N] markers in the input.\n"
        "- Quote verbatim where a direct number, claim, or definition is load-bearing (use double quotes).\n"
        "- Do NOT invent facts not present in the sources.\n"
        "- If sources disagree, surface the contradiction explicitly with both [N] citations.\n"
        "- Identify gaps: important sub-questions that the sources do not answer.\n"
        "- Generate 'what would change my mind' (3-5 specific evidence types that would falsify your conclusion) "
        "and 'counter_arguments' (2-3 strongest objections to your synthesis).\n\n"
        'Output JSON shape:\n'
        '{"executive_summary":"4-6 sentences with [N] citations","sections":'
        '[{"heading":"...","body_markdown":"...with [N] citations..."}],'
        '"contradictions":["..."],"gaps":["..."],'
        '"what_would_change_my_mind":["..."],"counter_arguments":["..."]}'
    )

    user = (
        f"Research question:\n{question}\n\n"
        f"Source documents (cite using [N] markers):\n{sources_block}"
    )

    try:
        parsed: Any = chat_json(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=LONGCTX_MAX_TOKENS,
            temperature=0.2,
            backend="openrouter",
            model=LONGCTX_MODEL,
        )
    except Exception as e:
        logger.error("[longctx] OpenRouter failed: %s", e)
        return {
            "executive_summary": f"Long-context synthesis failed: {e}",
            "sections": [],
            "contradictions": [],
            "gaps": [f"longctx error: {e}"],
            "what_would_change_my_mind": [],
            "counter_arguments": [],
        }

    if not isinstance(parsed, dict):
        logger.error("[longctx] bad shape: %s", type(parsed).__name__)
        return {
            "executive_summary": "Long-context synthesis returned malformed output.",
            "sections": [],
            "contradictions": [],
            "gaps": ["longctx returned non-dict"],
            "what_would_change_my_mind": [],
            "counter_arguments": [],
        }

    return {
        "executive_summary": str(parsed.get("executive_summary", ""))[:4000],
        "sections": parsed.get("sections", []) or [],
        "contradictions": parsed.get("contradictions", []) or [],
        "gaps": parsed.get("gaps", []) or [],
        "what_would_change_my_mind": parsed.get("what_would_change_my_mind", []) or [],
        "counter_arguments": parsed.get("counter_arguments", []) or [],
    }
