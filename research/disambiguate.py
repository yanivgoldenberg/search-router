"""Phase 3: entity disambiguation.

Between fanout_search and per-source extract: classify each source's primary
subject entity, then drop sources that describe a different entity than the
query target. Stops 'Jacob Goldenberg PDF' from polluting a 'who is Yaniv
Goldenberg' answer.
"""
from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from .llm import LLMError, chat_json
from .models import SourceMeta

logger = logging.getLogger(__name__)


_PERSON_RE = re.compile(r"^\s*(who\s+(is|was)|tell\s+me\s+about)\s+(.+?)\s*\??\s*$", re.IGNORECASE)


def _target_entity(question: str) -> tuple[str, str] | None:
    """Return (entity_type, canonical_name) for entity-centric questions, else None.

    For now we only auto-detect 'who is X' / 'tell me about X' people queries.
    Company / product disambiguation can be added later.
    """
    m = _PERSON_RE.match(question)
    if m:
        name = m.group(3).strip().strip('"\'')
        name = re.split(r",| - ", name)[0].strip()
        if name and len(name.split()) <= 6:
            return ("person", name)
    return None


def _classify_one(target_kind: str, target_name: str, src: SourceMeta) -> dict[str, Any]:
    body = (src.body or src.snippet or "")[:1800]
    if not body:
        return {"url": src.url, "match": "unknown", "subject": "", "reason": "empty body"}

    system = (
        "Output JSON only (no prose, no markdown). You are an entity disambiguation classifier. "
        "Given a target entity (a specific person, company, or product) and a source document, decide whether the document's "
        "PRIMARY subject is the same entity, a different entity with a similar name, or unrelated.\n\n"
        "Rules:\n"
        "- 'same' = the document is about THIS exact entity (same person, company, or product).\n"
        "- 'different' = the document is about a DIFFERENT entity that happens to share a name or partial name "
        "(e.g. 'Jacob Goldenberg' is NOT 'Yaniv Goldenberg'; 'Apple Records' is NOT 'Apple Inc.').\n"
        "- 'unrelated' = the document does not substantively discuss any entity by that name.\n"
        "- 'unknown' = signals are too weak to decide.\n\n"
        "Be strict: differing first names, middle initials, employers, locations, professions, or eras = 'different'. "
        "Only mark 'same' if the document clearly identifies the SAME individual or organization.\n\n"
        'Output shape: {"match":"same|different|unrelated|unknown","subject":"actual primary subject of the doc","reason":"1 short sentence"}'
    )
    user = (
        f"Target {target_kind}: {target_name}\n\n"
        f"Source URL: {src.url}\n"
        f"Source title: {src.title}\n\n"
        f"--- SOURCE EXCERPT ---\n{body}\n--- END ---"
    )
    try:
        parsed: Any = chat_json(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=300,
            temperature=0.0,
        )
    except (LLMError, Exception) as e:
        logger.warning("[disambig] LLM failed for %s: %s", src.url[:60], e)
        return {"url": src.url, "match": "unknown", "subject": "", "reason": f"llm error: {e}"}

    if not isinstance(parsed, dict):
        return {"url": src.url, "match": "unknown", "subject": "", "reason": "bad shape"}
    match = str(parsed.get("match", "unknown")).lower().strip()
    if match not in ("same", "different", "unrelated", "unknown"):
        match = "unknown"
    return {
        "url": src.url,
        "match": match,
        "subject": str(parsed.get("subject", ""))[:200],
        "reason": str(parsed.get("reason", ""))[:300],
    }


def disambiguate_sources(
    question: str,
    sources: list[SourceMeta],
    *,
    target_override: tuple[str, str] | None = None,
    max_workers: int = 4,
) -> tuple[list[SourceMeta], list[dict[str, Any]]]:
    """Filter sources to those whose primary subject matches the target entity.

    Returns (kept_sources, decisions). 'unknown' and 'unrelated' are kept (we
    only drop confirmed 'different'-entity sources). When no target entity can
    be inferred, all sources pass through and decisions=[].
    """
    target = target_override or _target_entity(question)
    if not target:
        return sources, []

    target_kind, target_name = target
    classifiable = [s for s in sources if (s.body or s.snippet)]
    if not classifiable:
        return sources, []

    decisions: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_classify_one, target_kind, target_name, s): s for s in classifiable}
        for fut in as_completed(futures):
            try:
                decisions.append(fut.result())
            except Exception as e:
                src = futures[fut]
                decisions.append({"url": src.url, "match": "unknown", "subject": "", "reason": f"future error: {e}"})

    by_url = {d["url"]: d for d in decisions}
    kept: list[SourceMeta] = []
    dropped = 0
    for s in sources:
        d = by_url.get(s.url)
        if d and d["match"] == "different":
            dropped += 1
            logger.info("[disambig] drop %s subject=%r reason=%s", s.url[:80], d.get("subject", ""), d.get("reason", ""))
            continue
        kept.append(s)

    logger.info("[disambig] target=%s/%s kept=%d dropped=%d (of %d classified)",
                target_kind, target_name, len(kept), dropped, len(classifiable))
    return kept, decisions
