"""Phase 1: decompose a research question into sub-questions + search queries."""
from __future__ import annotations

import logging
from typing import Any

from .llm import chat_json
from .models import DecomposedPlan, SubQuestion

logger = logging.getLogger(__name__)

MODE_HINTS = {
    "general": "general web research",
    "competitive": "B2B competitive intelligence — funding, headcount, products, pricing, traction",
    "academic": "scholarly literature — peer-reviewed, citations, methodology",
    "financial": "financial filings, market data, earnings, valuations, SEC EDGAR",
    "legal": "case law, statutes, court decisions, legal precedent",
    "medical": "clinical evidence, RCTs, peer-reviewed medical literature",
    "geo": "AI search visibility, generative engine optimization, citability",
    "trading": "market structure, technicals, macro, positioning, real-time",
}


def decompose(question: str, max_sub_questions: int = 8, mode: str = "general") -> DecomposedPlan:
    hint = MODE_HINTS.get(mode, MODE_HINTS["general"])
    system = (
        "You are a research planner. Given a question, break it into independent sub-questions "
        "that, when answered together, fully answer the original. For each sub-question, generate "
        "3-5 specific search queries (different phrasings, time windows, source-type hints). "
        "Each search query must be self-contained.\n\n"
        f"Research mode: {hint}\n\n"
        'Respond ONLY with this JSON shape (no prose, no markdown):\n'
        '{"sub_questions":[{"text":"...","search_queries":["...","..."],"search_type":"serp"}],'
        '"rationale":"..."}'
        '\n\nValid search_type values: serp, news, ai, deep, academic, code, social.'
    )
    user = (
        f"Original question: {question}\n\n"
        f"Generate up to {max_sub_questions} sub-questions. Quality over quantity. "
        "Avoid trivial or duplicative sub-questions."
    )

    parsed: Any = chat_json(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_tokens=2500,
        temperature=0.3,
    )

    if not isinstance(parsed, dict) or "sub_questions" not in parsed:
        raise ValueError(f"decompose returned unexpected shape: {type(parsed).__name__}")

    sub_qs = []
    for s in parsed.get("sub_questions", [])[:max_sub_questions]:
        if not isinstance(s, dict):
            continue
        text = s.get("text", "").strip()
        queries = [q for q in s.get("search_queries", []) if isinstance(q, str) and q.strip()]
        st = s.get("search_type", "serp")
        if st not in ("serp", "news", "ai", "deep", "academic", "code", "social"):
            st = "serp"
        if text and queries:
            sub_qs.append(SubQuestion(text=text, search_queries=queries[:5], search_type=st))

    if not sub_qs:
        sub_qs = [SubQuestion(text=question, search_queries=[question], search_type="serp")]

    return DecomposedPlan(
        question=question,
        sub_questions=sub_qs,
        rationale=parsed.get("rationale", "")[:500],
    )
