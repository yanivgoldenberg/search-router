"""Entity disambiguation pass — between Phase 2 (search) and Phase 4 (extract).

Groups sources by which entity they describe; rejects sources about wrong entity.
Critical for queries like 'who is yaniv goldenberg' where multiple people share the name.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from .llm import LLMError, chat_json
from .models import SourceMeta

logger = logging.getLogger(__name__)


def _classify_source(query: str, src: SourceMeta) -> dict[str, Any]:
    """Ask LLM: which entity does this source describe? Return {primary_entity, matches_query, confidence}."""
    body = (src.body or src.snippet or "")[:2000]
    if not body:
        return {"primary_entity": "unknown", "matches_query": True, "confidence": 0.3}
    system = (
        "Output JSON only. Identify the primary entity (person/company/product/concept) this source is about. "
        "Then judge: does this entity match the entity the user is asking about?\n"
        'Output: {"primary_entity":"<name>","entity_type":"person|company|product|concept|other",'
        '"matches_query":true|false,"reasoning":"<one-line>","confidence":0.0-1.0}'
    )
    user = f"User query: {query}\n\nSource title: {src.title}\nSource URL: {src.url}\n\nBody excerpt:\n{body}"
    try:
        out: Any = chat_json(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=400, temperature=0.1,
        )
        if isinstance(out, dict):
            return {
                "primary_entity": str(out.get("primary_entity", "unknown"))[:100],
                "entity_type": str(out.get("entity_type", "other")),
                "matches_query": bool(out.get("matches_query", True)),
                "reasoning": str(out.get("reasoning", ""))[:200],
                "confidence": float(out.get("confidence", 0.5) or 0.5),
            }
    except Exception as e:
        logger.debug("[disambig] failed for %s: %s", src.url[:50], e)
    return {"primary_entity": "unknown", "matches_query": True, "confidence": 0.3, "reasoning": "classify failed"}


def disambiguate_sources(query: str, sources: list[SourceMeta], max_workers: int = 4) -> tuple[list[SourceMeta], list[dict]]:
    """Returns (matching_sources, all_classifications). Rejects sources whose entity doesn't match."""
    if not sources:
        return [], []
    classifications = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_classify_source, query, s): s for s in sources}
        for fut in as_completed(futs):
            src = futs[fut]
            try:
                cls = fut.result()
                cls["url"] = src.url
                cls["title"] = src.title
                classifications.append(cls)
            except Exception as e:
                logger.warning("disambig future failed: %s", e)

    by_url = {c["url"]: c for c in classifications}
    matching: list[SourceMeta] = []
    for s in sources:
        c = by_url.get(s.url)
        if not c or c.get("matches_query", True):
            matching.append(s)
        else:
            logger.info("[disambig] reject %s — entity %s != query target (%.1f conf)",
                        s.url[:60], c.get("primary_entity",""), c.get("confidence", 0))
    logger.info("[disambig] kept %d/%d sources after entity matching", len(matching), len(sources))
    return matching, classifications
