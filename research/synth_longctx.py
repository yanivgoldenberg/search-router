"""Long-context synthesis (tier=ultra path).

Skips per-source extract. Feeds 10-30 raw source bodies directly into a 1M+ context model.
Free via OpenRouter's `meta-llama/llama-4-scout:free` (10M context).
"""
from __future__ import annotations

import logging
import os
from typing import Any

from .llm import LLMError, chat_json
from .models import ExtractedClaim, ResearchSection, SourceMeta

logger = logging.getLogger(__name__)


def synthesize_longctx(question: str, sources: list[SourceMeta], target_model: str = "openrouter") -> dict[str, Any]:
    """Feed raw source bodies into a long-context model in one call."""
    if not sources:
        return {"executive_summary": "No sources fetched.", "sections": [], "claims": [],
                "contradictions": [], "gaps": [], "what_would_change_my_mind": [], "counter_arguments": []}

    blocks = []
    for i, s in enumerate(sources, 1):
        body = (s.body or s.snippet or "")[:8000]
        if not body:
            continue
        blocks.append(f"[{i}] {s.title}\n  URL: {s.url}\n  ---\n{body}\n  ---")
    sources_text = "\n\n".join(blocks)

    system = (
        "Output JSON only (no prose, no markdown fences). You are a senior research analyst. "
        "Read all provided sources and synthesize an answer. Every factual claim must cite [N] indices. "
        "STRICT: do not invent facts not in the sources. Surface contradictions explicitly.\n\n"
        "Output shape:\n"
        '{"executive_summary":"3-5 sentences with [N] citations",'
        '"sections":[{"heading":"...","body_markdown":"...with [N] citations..."}],'
        '"claims":[{"text":"...","exact_quote":"<verbatim from source>","source_index":N,"confidence":0.0-1.0}],'
        '"contradictions":["..."],"gaps":["..."],'
        '"what_would_change_my_mind":["..."],"counter_arguments":["..."]}'
    )
    user = f"Question: {question}\n\nSources:\n{sources_text[:300000]}"

    try:
        out: Any = chat_json(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=8000, temperature=0.2, backend=target_model,
        )
    except Exception as e:
        logger.error("[longctx] synth failed: %s — falling back to default backend", e)
        try:
            out = chat_json(
                [{"role": "system", "content": system}, {"role": "user", "content": user}],
                max_tokens=8000, temperature=0.2,
            )
        except Exception as e2:
            return {"executive_summary": f"Long-context synth failed: {e2}",
                    "sections": [], "claims": [], "contradictions": [], "gaps": [],
                    "what_would_change_my_mind": [], "counter_arguments": []}

    if not isinstance(out, dict):
        return {"executive_summary": "Long-context synth returned invalid shape.",
                "sections": [], "claims": [], "contradictions": [], "gaps": [],
                "what_would_change_my_mind": [], "counter_arguments": []}

    # Build claim objects with source URL
    claims_raw = out.get("claims", []) or []
    claims = []
    for c in claims_raw:
        if not isinstance(c, dict):
            continue
        idx = int(c.get("source_index", 0) or 0) - 1
        url = sources[idx].url if 0 <= idx < len(sources) else ""
        claims.append(ExtractedClaim(
            text=str(c.get("text", "")),
            exact_quote=str(c.get("exact_quote", ""))[:500],
            source_url=url,
            confidence=float(c.get("confidence", 0.5) or 0.5),
        ))

    sections = []
    for s in out.get("sections", []) or []:
        if isinstance(s, dict):
            sections.append(ResearchSection(
                heading=str(s.get("heading", ""))[:200],
                body_markdown=str(s.get("body_markdown", "")),
            ))

    return {
        "executive_summary": str(out.get("executive_summary", ""))[:3000],
        "sections": sections,
        "claims": claims,
        "contradictions": [str(c) for c in (out.get("contradictions") or []) if c],
        "gaps": [str(g) for g in (out.get("gaps") or []) if g],
        "what_would_change_my_mind": [str(w) for w in (out.get("what_would_change_my_mind") or []) if w],
        "counter_arguments": [str(c) for c in (out.get("counter_arguments") or []) if c],
    }
