"""Phase 2: fan-out search via aisearch /v1/search; dedupe + rank URLs."""
from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse

import requests

from .models import SourceMeta, SubQuestion

logger = logging.getLogger(__name__)

AISEARCH_URL = os.environ.get("AISEARCH_URL", "http://localhost:8300")
SEARCH_TIMEOUT = float(os.environ.get("AISEARCH_SEARCH_TIMEOUT", "30"))
MAX_WORKERS = int(os.environ.get("RESEARCH_SEARCH_WORKERS", "8"))


def _aisearch_one(query: str, search_type: str, num: int = 10) -> list[dict]:
    try:
        resp = requests.post(
            f"{AISEARCH_URL}/v1/search",
            json={"type": search_type, "q": query, "num": num},
            timeout=SEARCH_TIMEOUT,
        )
        if resp.status_code != 200:
            logger.warning("aisearch %s '%s' -> http %s", search_type, query[:60], resp.status_code)
            return []
        data = resp.json()
        provider = data.get("provider", "aisearch")
        results = []
        for r in data.get("results", []):
            url = r.get("url", "")
            if not url:
                continue
            results.append({
                "url": url,
                "title": r.get("title", "") or "",
                "snippet": r.get("snippet", "") or "",
                "provider": provider,
            })
        return results
    except Exception as e:
        logger.warning("aisearch '%s' failed: %s", query[:60], e)
        return []





def _wayback_search(query: str, num: int = 5) -> list[dict]:
    """Internet Archive Wayback CDX search — returns historical URLs matching the query."""
    try:
        # use IA's full-text search via webhandlers
        resp = requests.get(
            "https://archive.org/advancedsearch.php",
            params={
                "q": query,
                "fl[]": "identifier,title,description,url,date",
                "rows": num,
                "output": "json",
            },
            timeout=10,
        )
        if resp.status_code != 200:
            return []
        data = resp.json()
        results = []
        for d in data.get("response", {}).get("docs", []):
            ident = d.get("identifier", "")
            if not ident:
                continue
            url = d.get("url") or f"https://archive.org/details/{ident}"
            results.append({
                "url": url,
                "title": d.get("title", ""),
                "snippet": (d.get("description", "") or "")[:300],
                "provider": "wayback.archive",
            })
        return results
    except Exception as e:
        logger.debug("wayback %s failed: %s", query[:60], e)
        return []


def _dedupe_url(url: str) -> str:
    try:
        p = urlparse(url)
        host = p.netloc.lower().lstrip("www.")
        path = p.path.rstrip("/")
        return f"{host}{path}"
    except Exception:
        return url


def fanout_search(sub_questions: list[SubQuestion], num_per_query: int = 8) -> list[SourceMeta]:
    """Fire all search queries in parallel, dedupe by URL host+path."""
    jobs: list[tuple[SubQuestion, str, str]] = []
    for sq in sub_questions:
        for q in sq.search_queries:
            jobs.append((sq, q, sq.search_type))

    seen: dict[str, SourceMeta] = {}
    queries_total = len(jobs)
    logger.info("[search] firing %d search queries across %d sub-questions", queries_total, len(sub_questions))

    # Add wayback for "deep"/"academic" queries
    wayback_jobs = []
    facad_jobs = []
    for sq in sub_questions:
        if sq.search_type in ("deep", "academic"):
            for q in sq.search_queries[:2]:
                wayback_jobs.append((sq, q))
        if sq.search_type == "academic":
            for q in sq.search_queries[:2]:
                facad_jobs.append((sq, q, "academic"))

    from . import free_academic as _facad
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(_aisearch_one, q, st, num_per_query): (sq, q) for sq, q, st in jobs}
        for sq, q in wayback_jobs:
            futs[ex.submit(_wayback_search, q, 5)] = (sq, q)
        for sq, q, kind in facad_jobs:
            futs[ex.submit(lambda qq=q: (_facad.edgar_full_text_search(qq, 5) or {}).get("results", []))] = (sq, q)
            futs[ex.submit(lambda qq=q: (_facad.biorxiv_search(qq, 5, 'biorxiv') or {}).get("results", []))] = (sq, q)
            futs[ex.submit(lambda qq=q: (_facad.biorxiv_search(qq, 5, 'medrxiv') or {}).get("results", []))] = (sq, q)
            futs[ex.submit(lambda qq=q: (_facad.clinicaltrials_search(qq, 3) or {}).get("results", []))] = (sq, q)
        for fut in as_completed(futs):
            sq, _q = futs[fut]
            try:
                results = fut.result()
            except Exception as e:
                logger.warning("search future failed: %s", e)
                continue
            for r in results:
                k = _dedupe_url(r["url"])
                if k in seen:
                    continue
                seen[k] = SourceMeta(
                    url=r["url"],
                    title=r["title"],
                    snippet=r["snippet"],
                    provider=r["provider"],
                    sub_question=sq.text,
                )

    sources = list(seen.values())
    logger.info("[search] %d unique URLs after dedupe (from %d queries)", len(sources), queries_total)
    return sources


def rank_and_trim(sources: list[SourceMeta], max_sources: int, query_terms: list[str] | None = None) -> list[SourceMeta]:
    """Trim to top N. Boost domain-matches-query-name (huge), authoritative providers, etc."""
    qt = [t.lower() for t in (query_terms or []) if len(t) > 2]

    def score(s: SourceMeta) -> float:
        url_l = s.url.lower()
        title_l = s.title.lower() if s.title else ""
        sc = 0.0

        # === DOMAIN-MATCHES-QUERY-NAME (the unlock) ===
        # If domain contains query name tokens, MASSIVE boost — likely canonical
        if qt:
            try:
                from urllib.parse import urlparse
                host = urlparse(s.url).netloc.lower().replace("www.", "")
                domain_tokens = host.replace("-", "").replace(".", "")
                # ALL query tokens appear in domain → +5.0 (canonical match)
                if all(t.replace(" ", "") in domain_tokens for t in qt):
                    sc += 5.0
                # Most query tokens appear → +2.5
                elif sum(1 for t in qt if t.replace(" ", "") in domain_tokens) >= max(1, len(qt) - 1):
                    sc += 2.5
                # Title includes ALL query tokens → +1.5 (very relevant page)
                if title_l and all(t in title_l for t in qt):
                    sc += 1.5
            except Exception:
                pass

        # Provider trust
        if s.provider.startswith(("serper", "exa", "tavily")):
            sc += 1.0
        if s.provider.startswith(("openalex", "arxiv", "pubmed", "crossref", "sec.edgar")):
            sc += 1.5
        if "wikipedia" in url_l:
            sc += 1.5
        # Personal/social-pro signals (helpful for "who is X" queries)
        if any(d in url_l for d in ("linkedin.com/in/", "github.com/", "crunchbase.com/person/")):
            sc += 1.5
        # Snippet length
        if s.snippet:
            sc += min(len(s.snippet) / 500, 1.0)
        # PDFs - small boost (was 0.8, now 0.4 to avoid drowning canonical sites)
        if url_l.endswith(".pdf"):
            sc += 0.4
        # Penalize generic dump sites
        bad = ("pinterest", "facebook.com/", "instagram.com/", "twitter.com/intent", "tiktok.com/", "soundcloud.com/")
        if any(b in url_l for b in bad):
            sc -= 1.5
        return sc

    return sorted(sources, key=score, reverse=True)[:max_sources]
