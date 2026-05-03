"""Additional free academic + government providers (no key needed).

EDGAR full-text, USPTO patents (PatentsView free), bioRxiv/medRxiv preprints,
SEC company concept, ClinicalTrials.gov, Common Crawl historical, NIH RePORTER.
"""
from __future__ import annotations

import logging
import urllib.parse
from typing import Any

import requests

logger = logging.getLogger(__name__)

UA = "Mozilla/5.0 (compatible; YanivResearch/1.0; mailto:goldenbergyaniv@gmail.com)"
TO = 15


def _safe_get_json(url: str, params=None, headers=None) -> dict | list | None:
    try:
        h = {"User-Agent": UA, "Accept": "application/json"}
        if headers:
            h.update(headers)
        r = requests.get(url, params=params, headers=h, timeout=TO)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception as e:
        logger.debug("get %s failed: %s", url, e)
        return None


def edgar_full_text_search(query: str, num: int = 10) -> dict | None:
    """SEC EDGAR full-text search across 10-K, 10-Q, 8-K filings."""
    data = _safe_get_json(
        "https://efts.sec.gov/LATEST/search-index",
        params={"q": query, "dateRange": "custom", "forms": "10-K,10-Q,8-K", "hits": num},
    )
    if not data:
        return None
    hits = (data.get("hits") or {}).get("hits", [])
    results = []
    for h in hits[:num]:
        src = h.get("_source", {})
        adsh = src.get("adsh", "")
        accession = adsh.replace("-", "") if adsh else ""
        results.append({
            "title": f"{src.get('display_names', [''])[0]} — {src.get('form', '')} ({src.get('file_date', '')})",
            "url": f"https://www.sec.gov/Archives/edgar/data/{src.get('ciks', [''])[0]}/{accession}/{src.get('id', '')}",
            "snippet": " ".join(src.get("display_names", [])) + " — " + src.get("form", "") + " " + src.get("file_date", ""),
        })
    return {"provider": "sec.edgar.fts", "results": results, "raw": {"count": len(results)}}


def patentsview_search(query: str, num: int = 10) -> dict | None:
    """USPTO PatentsView free API search."""
    payload = {
        "q": {"_text_phrase": {"patent_title": query}},
        "f": ["patent_number", "patent_title", "patent_date", "patent_abstract"],
        "o": {"per_page": num, "page": 1, "matched_subentities_only": True},
    }
    try:
        r = requests.post(
            "https://search.patentsview.org/api/v1/patent/",
            json=payload, headers={"User-Agent": UA, "Accept": "application/json"}, timeout=TO,
        )
        if r.status_code != 200:
            return None
        data = r.json()
    except Exception as e:
        logger.debug("patentsview failed: %s", e)
        return None
    pats = data.get("patents", []) or []
    results = [
        {
            "title": f"USPTO {p.get('patent_number')}: {p.get('patent_title','')}",
            "url": f"https://patents.google.com/patent/US{p.get('patent_number','')}",
            "snippet": (p.get("patent_abstract") or "")[:500],
        }
        for p in pats[:num]
    ]
    return {"provider": "uspto.patentsview", "results": results, "raw": {"count": len(results)}}


def biorxiv_search(query: str, num: int = 10, server: str = "biorxiv") -> dict | None:
    """bioRxiv/medRxiv preprints API. server in {biorxiv, medrxiv}."""
    data = _safe_get_json(f"https://api.biorxiv.org/details/{server}/2024-01-01/2026-12-31/0", )
    if not data:
        return None
    coll = data.get("collection", []) or []
    q_low = query.lower()
    matches = [c for c in coll if q_low in (c.get("title", "") + " " + c.get("abstract", "")).lower()][:num]
    results = [
        {
            "title": c.get("title", ""),
            "url": f"https://www.{server}.org/content/{c.get('doi','')}",
            "snippet": (c.get("abstract", "") or "")[:500],
        }
        for c in matches
    ]
    return {"provider": f"{server}.preprint", "results": results, "raw": {"count": len(results)}}


def clinicaltrials_search(query: str, num: int = 10) -> dict | None:
    """ClinicalTrials.gov v2 API."""
    data = _safe_get_json(
        "https://clinicaltrials.gov/api/v2/studies",
        params={"query.term": query, "pageSize": num, "format": "json"},
    )
    if not data:
        return None
    studies = data.get("studies", []) or []
    results = []
    for s in studies[:num]:
        proto = s.get("protocolSection", {})
        ident = (proto.get("identificationModule") or {})
        desc = (proto.get("descriptionModule") or {})
        nctid = ident.get("nctId", "")
        results.append({
            "title": f"NCT{nctid}: {ident.get('briefTitle','')}",
            "url": f"https://clinicaltrials.gov/study/{nctid}",
            "snippet": (desc.get("briefSummary") or "")[:500],
        })
    return {"provider": "clinicaltrials", "results": results, "raw": {"count": len(results)}}


def openalex_citing(work_id: str, num: int = 10) -> dict | None:
    """For a given OpenAlex work, find papers that cite it (forward citation)."""
    data = _safe_get_json(
        "https://api.openalex.org/works",
        params={"filter": f"cites:{work_id}", "per-page": num, "mailto": "goldenbergyaniv@gmail.com"},
    )
    if not data:
        return None
    results = []
    for w in data.get("results", [])[:num]:
        results.append({
            "title": w.get("title", ""),
            "url": w.get("doi") or w.get("id", ""),
            "snippet": f"cited by {w.get('cited_by_count','?')} | {w.get('publication_year','')}",
        })
    return {"provider": "openalex.citing", "results": results, "raw": {"count": len(results)}}


def common_crawl_search(query: str, num: int = 5, index: str = "CC-MAIN-2025-46") -> dict | None:
    """Common Crawl CDX historical web index search by URL prefix."""
    # Best-effort: search by domain match for the query
    if " " in query:
        return None
    try:
        r = requests.get(
            f"https://index.commoncrawl.org/{index}-index",
            params={"url": query, "output": "json", "limit": num},
            headers={"User-Agent": UA},
            timeout=TO,
        )
        if r.status_code != 200:
            return None
    except Exception:
        return None
    results = []
    for line in r.text.splitlines()[:num]:
        try:
            import json as _json
            row = _json.loads(line)
            results.append({
                "title": row.get("url", ""),
                "url": row.get("url", ""),
                "snippet": f"timestamp={row.get('timestamp','')} status={row.get('status','')}",
            })
        except Exception:
            continue
    return {"provider": f"commoncrawl.{index}", "results": results, "raw": {"count": len(results)}}
