"""Phase 7-8: synthesis from verified claims. Single-model first, optional ensemble."""
from __future__ import annotations

import json
import logging
from typing import Any

from .llm import LLMError, chat, chat_json
from .models import ExtractedClaim, ResearchSection, SourceMeta

logger = logging.getLogger(__name__)


def _format_claims_block(claims: list[ExtractedClaim], sources: list[SourceMeta]) -> tuple[str, dict[int, str]]:
    url_to_idx: dict[str, int] = {}
    idx_to_url: dict[int, str] = {}
    next_idx = 1
    for s in sources:
        if s.url not in url_to_idx:
            url_to_idx[s.url] = next_idx
            idx_to_url[next_idx] = s.url
            next_idx += 1

    lines = []
    for c in claims:
        idx = url_to_idx.get(c.source_url, 0)
        if idx == 0:
            continue
        lines.append(
            f"[{idx}] (sub-Q: {c.sub_question[:60]}) (conf={c.confidence:.1f}) "
            f"CLAIM: {c.text}\n  QUOTE: \"{c.exact_quote}\""
        )
    return "\n\n".join(lines), idx_to_url


def synthesize(question: str, claims: list[ExtractedClaim], sources: list[SourceMeta]) -> dict[str, Any]:
    if not claims:
        return {
            "executive_summary": "No verified claims found across the searched sources.",
            "sections": [],
            "contradictions": [],
            "gaps": ["All sources fetched but no verbatim-quotable claims extracted."],
            "what_would_change_my_mind": [],
            "counter_arguments": [],
        }

    claims_block, idx_to_url = _format_claims_block(claims, sources)

    system = (
        "Output JSON only (no prose, no markdown fences before or after). You are a senior research analyst. Synthesize the user's research question using ONLY "
        "the verified claims provided. Each claim has a [N] citation index — use them inline.\n\n"
        "STRICT RULES:\n"
        "- Cite every factual statement using [N] indices.\n"
        "- Do NOT invent facts not in the claims.\n"
        "- If sources disagree, surface the contradiction explicitly.\n"
        "- If important sub-questions weren't answered, list them as gaps.\n"
        "- Adversarial step: also generate 'what would change my mind' (3-5 specific evidence types) "
        "  and 'counter_arguments' (2-3 strongest objections).\n\n"
        'Output JSON shape:\n'
        '{"executive_summary":"3-5 sentences with [N] citations","sections":'
        '[{"heading":"...","body_markdown":"...with [N] citations..."}],'
        '"contradictions":["..."],"gaps":["..."],'
        '"what_would_change_my_mind":["..."],"counter_arguments":["..."]}'
    )

    user = (
        f"Research question:\n{question}\n\n"
        f"Verified claims (use [N] inline):\n{claims_block[:120000]}"
    )

    try:
        parsed: Any = chat_json(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=6000,
            temperature=0.2,
        )
    except Exception as e:
        logger.error("[synth] LLM failed: %s", e)
        return {
            "executive_summary": f"Synthesis failed: {e}",
            "sections": [],
            "contradictions": [],
            "gaps": [],
            "what_would_change_my_mind": [],
            "counter_arguments": [],
        }

    sections_raw = parsed.get("sections", []) or []
    sections: list[ResearchSection] = []
    for s in sections_raw:
        if not isinstance(s, dict):
            continue
        sections.append(ResearchSection(
            heading=str(s.get("heading", ""))[:200],
            body_markdown=str(s.get("body_markdown", "")),
        ))

    return {
        "executive_summary": str(parsed.get("executive_summary", ""))[:3000],
        "sections": sections,
        "contradictions": [str(c) for c in (parsed.get("contradictions") or []) if c],
        "gaps": [str(g) for g in (parsed.get("gaps") or []) if g],
        "what_would_change_my_mind": [str(w) for w in (parsed.get("what_would_change_my_mind") or []) if w],
        "counter_arguments": [str(c) for c in (parsed.get("counter_arguments") or []) if c],
    }



def adversarial_pass(question: str, draft: dict, claims: list, sources: list) -> dict:
    """Run a skeptic-LLM pass over the draft synthesis and merge counter-arguments.

    Doubles cost but materially improves quality on contestable questions.
    """
    from .llm import LLMError, chat_json
    if not draft.get("executive_summary"):
        return draft
    summary = (draft.get("executive_summary") or "")[:2000]
    sec_text = " ".join(s.body_markdown for s in draft.get("sections", []))[:5000]
    system = (
        "You are a skeptical reviewer. Find weaknesses in the draft research below. "
        "Identify: (1) overclaims not strongly supported, (2) missing perspectives, "
        "(3) contradicting evidence the report should mention, (4) what a domain expert "
        "would push back on. Output JSON with these fields exactly:\n"
        '{"weak_claims":["..."],"missing_perspectives":["..."],"counter_arguments":["..."],'
        '"strongest_objection":"..."}'
    )
    user = f"Question: {question}\n\nDraft summary:\n{summary}\n\nDraft sections:\n{sec_text}"
    try:
        critique = chat_json(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=2500,
            temperature=0.4,
        )
    except Exception as e:
        logger.warning("[adversarial] failed: %s", e)
        return draft
    if not isinstance(critique, dict):
        return draft
    new_counter = (draft.get("counter_arguments") or []) + [str(c) for c in (critique.get("counter_arguments") or []) if c]
    if critique.get("strongest_objection"):
        new_counter.insert(0, "STRONGEST OBJECTION: " + str(critique["strongest_objection"]))
    new_gaps = (draft.get("gaps") or []) + [str(g) for g in (critique.get("missing_perspectives") or []) if g]
    weak = [str(w) for w in (critique.get("weak_claims") or []) if w]
    if weak:
        new_gaps.append("Weak claims flagged by adversarial review: " + "; ".join(weak[:3]))
    draft["counter_arguments"] = new_counter[:8]
    draft["gaps"] = new_gaps[:10]
    return draft
