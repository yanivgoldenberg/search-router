"""End-to-end research pipeline orchestrator."""
from __future__ import annotations

import logging
import os
import time

from .decompose import decompose
from .disambiguate import disambiguate_sources
from .extract import extract_all
from .fetch import fetch_all
from .models import (
    ExtractedClaim,
    ResearchReport,
    ResearchRequest,
    SourceMeta,
    SubQuestion,
)
from .persist import ensure_schema, render_markdown, save_session
from .search import fanout_search, rank_and_trim
from .synth import adversarial_pass, self_critique_revise, synthesize
from .synth_longctx import synthesize_longctx
from .verify import verify_claims
from . import discord_notify as _discord, free_academic as _facad, siyuan_export as _siyuan


MODE_TO_PRIMARY_TYPE = {
    "general": "serp",
    "competitive": "serp",
    "academic": "academic",
    "financial": "serp",
    "legal": "serp",
    "medical": "academic",
    "geo": "serp",
    "trading": "news",
    "people": "social",
    "product": "serp",
}


def _apply_mode_search_bias(plan, mode):
    """Bias sub-question search_type toward the mode's primary source type."""
    primary = MODE_TO_PRIMARY_TYPE.get(mode, "serp")
    for sq in plan.sub_questions:
        if sq.search_type == "serp" and primary != "serp":
            # mix: half stay serp, half use primary
            sq.search_type = primary
    return plan

logger = logging.getLogger(__name__)

GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")


def _gap_subquestions(report_partial: dict, original: str) -> list[SubQuestion]:
    out = []
    for g in (report_partial.get("gaps") or [])[:5]:
        out.append(SubQuestion(text=str(g), search_queries=[str(g), f"{original} {g}"], search_type="serp"))
    return out


def run_research(req: ResearchRequest) -> ResearchReport:
    t0 = time.time()
    logger.info("[research] question=%s mode=%s tier=%s max_sources=%d iterations=%d",
                req.q[:80], req.mode, req.tier, req.max_sources, req.iterations)
    use_longctx = (req.tier == 'ultra') and bool(os.environ.get("OPENROUTER_API_KEY", "").strip())
    if req.tier == 'ultra' and not use_longctx:
        logger.info("[research] tier=ultra requested but OPENROUTER_API_KEY missing, falling back to standard extract+synth")

    # Auto-detect person-query intent: bump to people mode if query starts with "who is/was"
    detected_mode = req.mode
    import re as _re_mode
    if req.mode == "general" and _re_mode.match(r'^\s*(who\s+(is|was)|tell\s+me\s+about)\s+\w', req.q, _re_mode.IGNORECASE):
        detected_mode = "people"
        logger.info("[research] auto-detected person-query, switching to mode=people")
    plan = decompose(req.q, max_sub_questions=req.max_sub_questions, mode=detected_mode)
    plan = _apply_mode_search_bias(plan, detected_mode)
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
        import re as _re
        _terms = [t for t in _re.findall(r'\w+', req.q.lower()) if len(t) > 2 and t not in ('who','what','how','why','when','the','for','and','about','with')]
        ranked = rank_and_trim(new_sources, max_sources=budget, query_terms=_terms)
        if not ranked:
            logger.info("[research] no new sources to fetch this iteration")
            break

        fetched = fetch_all(ranked)
        for s in fetched:
            all_sources[s.url] = s

        kept, _disamb_decisions = disambiguate_sources(req.q, fetched)
        if len(kept) < len(fetched):
            logger.info("[research] disambiguation dropped %d/%d sources", len(fetched) - len(kept), len(fetched))
            dropped_urls = {s.url for s in fetched} - {s.url for s in kept}
            for url in dropped_urls:
                if url in all_sources:
                    all_sources[url].body = None

        if not use_longctx:
            extracted = extract_all(req.q, kept, max_workers=1)
            all_extracted.extend(extracted)

            all_claims = [c for ex in all_extracted for c in ex.claims]
            if it < req.iterations - 1:
                partial = synthesize(req.q, all_claims, list(all_sources.values()))
                sub_qs = _gap_subquestions(partial, req.q)
                if not sub_qs:
                    logger.info("[research] no gaps surfaced; stopping early")
                    break
        else:
            if it < req.iterations - 1:
                partial = synthesize_longctx(req.q, list(all_sources.values()))
                sub_qs = _gap_subquestions(partial, req.q)
                if not sub_qs:
                    logger.info("[research] no gaps surfaced; stopping early")
                    break

    sources_list = [s for s in all_sources.values() if s.body]
    all_claims = [c for ex in all_extracted for c in ex.claims]

    verified, unverified = verify_claims(all_claims, sources_list)

    if use_longctx:
        logger.info("[research] using long-context synthesis (OpenRouter, %d sources)", len(sources_list))
        final = synthesize_longctx(req.q, sources_list)
    else:
        final = synthesize(req.q, verified, sources_list)
    if req.tier == 'premium' or (req.tier == 'ultra' and not use_longctx):
        final = adversarial_pass(req.q, final, verified, sources_list)

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

    if req.tier == 'ultra':
        final = self_critique_revise(req.q, final, verified, sources_list)
    try:
        _discord.notify_completion(
            question=req.q,
            summary=final.get("executive_summary", "")[:1200],
            elapsed_s=round(time.time() - t0, 2),
            sources_read=len(sources_list),
            claims=len(all_claims),
            verified=sum(1 for c in verified),
        )
    except Exception as e:
        logger.debug("discord notify failed: %s", e)

    if req.save:
        try:
            ensure_schema()
            sid = save_session(report, req.mode, req.tier)
            if sid:
                logger.info("[research] persisted as session_id=%s", sid)
        except Exception as e:
            logger.warning("[research] persist failed: %s", e)
        try:
            md = render_markdown(report)
            if _siyuan.export_to_siyuan(req.q, md, req.mode):
                logger.info("[research] exported to SiYuan")
        except Exception as e:
            logger.debug("[research] siyuan export skipped: %s", e)

    return report
