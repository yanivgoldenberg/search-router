"""End-to-end research pipeline orchestrator."""
from __future__ import annotations

import logging
import os
import time

from .decompose import decompose
from .extract import extract_all
from .fetch import fetch_all
from .models import (
    ExtractedClaim,
    ResearchReport,
    ResearchRequest,
    SourceMeta,
    SubQuestion,
)
from .persist import ensure_schema, save_session
from .search import fanout_search, rank_and_trim
from .synth import synthesize
from .verify import verify_claims

logger = logging.getLogger(__name__)

GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")


def _gap_subquestions(report_partial: dict, original: str) -> list[SubQuestion]:
    out = []
    for g in (report_partial.get("gaps") or [])[:5]:
        out.append(SubQuestion(text=str(g), search_queries=[str(g), f"{original} {g}"], search_type="serp"))
    return out


def run_research(req: ResearchRequest) -> ResearchReport:
    t0 = time.time()
    logger.info("[research] question=%s mode=%s max_sources=%d iterations=%d",
                req.q[:80], req.mode, req.max_sources, req.iterations)

    plan = decompose(req.q, max_sub_questions=req.max_sub_questions, mode=req.mode)
    logger.info("[research] decomposed into %d sub-questions", len(plan.sub_questions))

    all_sources: dict[str, SourceMeta] = {}
    all_extracted = []
    iterations_run = 0

    sub_qs = plan.sub_questions
    for it in range(max(1, req.iterations)):
        iterations_run = it + 1
        logger.info("[research] === iteration %d/%d ===", iterations_run, req.iterations)

        sources = fanout_search(sub_qs, num_per_query=8)
        new_sources = [s for s in sources if s.url not in all_sources]
        for s in new_sources:
            all_sources[s.url] = s

        budget = max(5, req.max_sources - sum(1 for s in all_sources.values() if s.body))
        ranked = rank_and_trim(new_sources, max_sources=budget)
        if not ranked:
            logger.info("[research] no new sources to fetch this iteration")
            break

        fetched = fetch_all(ranked)
        for s in fetched:
            all_sources[s.url] = s

        extracted = extract_all(req.q, fetched, max_workers=3)
        all_extracted.extend(extracted)

        all_claims = [c for ex in all_extracted for c in ex.claims]
        if it < req.iterations - 1:
            partial = synthesize(req.q, all_claims, list(all_sources.values()))
            sub_qs = _gap_subquestions(partial, req.q)
            if not sub_qs:
                logger.info("[research] no gaps surfaced; stopping early")
                break

    sources_list = [s for s in all_sources.values() if s.body]
    all_claims = [c for ex in all_extracted for c in ex.claims]

    verified, unverified = verify_claims(all_claims, sources_list)

    final = synthesize(req.q, verified, sources_list)

    report = ResearchReport(
        question=req.q,
        executive_summary=final.get("executive_summary", ""),
        sections=final.get("sections", []),
        claims=verified + unverified,
        sources=sources_list,
        contradictions=final.get("contradictions", []),
        gaps=final.get("gaps", []),
        what_would_change_my_mind=final.get("what_would_change_my_mind", []),
        counter_arguments=final.get("counter_arguments", []),
        elapsed_seconds=round(time.time() - t0, 2),
        sources_read=len(sources_list),
        sources_searched=len(all_sources),
        iterations=iterations_run,
        model_used=GROQ_MODEL,
    )

    if req.save:
        try:
            ensure_schema()
            sid = save_session(report, req.mode, req.tier)
            if sid:
                logger.info("[research] persisted as session_id=%s", sid)
        except Exception as e:
            logger.warning("[research] persist failed: %s", e)

    return report
