"""Phase 4-5: per-source LLM extract (claims + entities + numbers) in parallel."""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import os
from .llm import LLMError, chat_json
from .models import ExtractedClaim, ExtractedSourceFacts, SourceMeta

logger = logging.getLogger(__name__)


def _extract_one(question: str, src: SourceMeta) -> ExtractedSourceFacts:
    body = (src.body or "")[:10000]
    if not body:
        return ExtractedSourceFacts(url=src.url)

    system = (
        "Extract facts from the source document that are relevant to the user's research question. "
        "Output JSON only.\n\n"
        "Rules:\n"
        "- Each claim must include an EXACT verbatim quote from the source (1-2 sentences max).\n"
        "- The 'text' field is your paraphrase of the claim; 'exact_quote' is verbatim from the source.\n"
        "- Skip claims that don't have a verbatim quote in the source.\n"
        "- Confidence: 0.9 for primary/authoritative source, 0.7 for reputable, 0.5 for unclear, 0.3 for weak.\n"
        "- Skip claims unrelated to the research question.\n"
        "- entities: company / product / person / place names mentioned.\n"
        "- numbers: every numeric fact with units / context (e.g. '23% growth Q3 2025', '$4.2M Series A 2024').\n\n"
        'Output shape:\n'
        '{"summary":"2-3 sentence relevance summary","claims":[{"text":"...","exact_quote":"...","confidence":0.7}],'
        '"entities":["..."],"numbers":["..."]}'
    )
    user = (
        f"Research question: {question}\n\n"
        f"Source URL: {src.url}\n"
        f"Source title: {src.title}\n\n"
        f"--- SOURCE BODY ---\n{body}\n--- END ---"
    )
    try:
        parsed: Any = chat_json(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=2000,
            temperature=0.1,
            model=os.environ.get("GROQ_EXTRACT_MODEL", "llama-3.3-70b-versatile"),
        )
    except Exception as e:
        logger.warning("[extract] LLM failed for %s: %s", src.url[:60], e)
        return ExtractedSourceFacts(url=src.url)

    if not isinstance(parsed, dict):
        return ExtractedSourceFacts(url=src.url)

    claims_raw = parsed.get("claims", []) or []
    claims: list[ExtractedClaim] = []
    for c in claims_raw:
        if not isinstance(c, dict):
            continue
        text = (c.get("text") or "").strip()
        quote = (c.get("exact_quote") or "").strip()
        if not text or not quote:
            continue
        claims.append(ExtractedClaim(
            text=text,
            exact_quote=quote[:500],
            source_url=src.url,
            confidence=float(c.get("confidence", 0.5) or 0.5),
            sub_question=src.sub_question,
        ))

    entities = [str(e).strip() for e in (parsed.get("entities") or []) if e]
    numbers = [str(n).strip() for n in (parsed.get("numbers") or []) if n]
    return ExtractedSourceFacts(
        url=src.url,
        summary=str(parsed.get("summary", ""))[:600],
        claims=claims,
        entities=entities[:50],
        numbers=numbers[:50],
    )


def extract_all(question: str, sources: list[SourceMeta], max_workers: int = 1) -> list[ExtractedSourceFacts]:
    logger.info("[extract] LLM-extracting facts from %d sources", len(sources))
    out: list[ExtractedSourceFacts] = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(_extract_one, question, s) for s in sources]
        for fut in as_completed(futs):
            try:
                out.append(fut.result())
            except Exception as e:
                logger.warning("extract future failed: %s", e)
    total_claims = sum(len(f.claims) for f in out)
    logger.info("[extract] extracted %d claims across %d sources", total_claims, len(out))
    return out
