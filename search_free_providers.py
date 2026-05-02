#!/usr/bin/env python3
"""Zero-key free search providers (no quota, no rotation needed).

Each function returns the unified shape:
  {"provider": str, "results": [{"title", "url", "snippet"}], "raw": dict}
or None if the request failed.

Providers:
  ddg_html_search     - DuckDuckGo HTML SERP scrape (free, ~unlimited)
  ddg_instant_answer  - DuckDuckGo Instant Answer (zero-click, free)
  wikipedia_search    - MediaWiki opensearch + extract (free, unlimited)
  hn_algolia          - HN search via Algolia (5k/hr, free)
  hn_algolia_date     - HN search ordered by date
  reddit_search       - Reddit JSON search (free, unauthenticated, rate-limited)
  marginalia_search   - Marginalia indie web search (free, unlimited)
  searx_be_federation - searx.be public instance fallback
  github_code         - GitHub code search (no auth = 10 req/min; auth = 30/min)
  stack_exchange      - Stack Exchange (StackOverflow) (300/day no key)
  arxiv_search        - arXiv API (free, unlimited)
  openalex_search     - OpenAlex scholarly works (100k/day free)
  pubmed_search       - PubMed E-utilities (free, 3 req/sec)
  crossref_search     - Crossref DOI search (free, polite pool)
"""
from __future__ import annotations

import json
import logging
import os
import re
import urllib.parse
from typing import Any
from xml.etree import ElementTree

import requests

logger = logging.getLogger(__name__)
DEFAULT_TIMEOUT = 15
USER_AGENT = "Mozilla/5.0 (compatible; YanivSearchRouter/1.0; +https://yanivgoldenberg.com)"


def _safe_get(url: str, **kw: Any) -> requests.Response | None:
    try:
        kw.setdefault("timeout", DEFAULT_TIMEOUT)
        kw.setdefault("headers", {})
        kw["headers"].setdefault("User-Agent", USER_AGENT)
        resp = requests.get(url, **kw)
        if resp.status_code == 200:
            return resp
        logger.warning("free-provider %s -> %s", url, resp.status_code)
        return None
    except requests.RequestException as e:
        logger.warning("free-provider %s -> %s", url, e)
        return None


def ddg_html_search(query: str, num: int = 10) -> dict[str, Any] | None:
    resp = _safe_get(
        "https://html.duckduckgo.com/html/",
        params={"q": query},
        headers={"User-Agent": USER_AGENT, "Accept-Language": "en-US,en;q=0.9"},
    )
    if not resp:
        return None
    pattern = re.compile(
        r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>.*?'
        r'<a[^>]+class="result__snippet"[^>]*>(.*?)</a>',
        re.DOTALL,
    )
    cleaner = re.compile(r"<[^>]+>")
    results = []
    for m in pattern.finditer(resp.text):
        url, title, snippet = m.group(1), cleaner.sub("", m.group(2)).strip(), cleaner.sub("", m.group(3)).strip()
        if "/l/?uddg=" in url:
            try:
                qs = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
                if "uddg" in qs:
                    url = urllib.parse.unquote(qs["uddg"][0])
            except Exception:
                pass
        results.append({"title": title, "url": url, "snippet": snippet})
        if len(results) >= num:
            break
    return {"provider": "ddg.html", "results": results, "raw": {"count": len(results)}}


def ddg_instant_answer(query: str) -> dict[str, Any] | None:
    resp = _safe_get(
        "https://api.duckduckgo.com/",
        params={"q": query, "format": "json", "no_html": 1, "no_redirect": 1, "skip_disambig": 1},
    )
    if not resp:
        return None
    data = resp.json()
    answer = data.get("AbstractText") or data.get("Answer")
    results = []
    if data.get("AbstractURL"):
        results.append({
            "title": data.get("Heading", query),
            "url": data["AbstractURL"],
            "snippet": data.get("AbstractText", ""),
        })
    for topic in data.get("RelatedTopics", []):
        if "FirstURL" in topic:
            results.append({
                "title": topic.get("Text", "")[:120],
                "url": topic["FirstURL"],
                "snippet": topic.get("Text", ""),
            })
    return {"provider": "ddg.instant", "answer": answer, "results": results, "raw": data}


def wikipedia_search(query: str, num: int = 10, lang: str = "en") -> dict[str, Any] | None:
    resp = _safe_get(
        f"https://{lang}.wikipedia.org/w/api.php",
        params={
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": num,
            "format": "json",
            "utf8": 1,
        },
    )
    if not resp:
        return None
    data = resp.json()
    cleaner = re.compile(r"<[^>]+>")
    results = []
    for item in data.get("query", {}).get("search", []):
        title = item["title"]
        snippet = cleaner.sub("", item.get("snippet", ""))
        url = f"https://{lang}.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}"
        results.append({"title": title, "url": url, "snippet": snippet})
    return {"provider": "wikipedia", "results": results, "raw": data}


def hn_algolia(query: str, num: int = 20, by_date: bool = False) -> dict[str, Any] | None:
    endpoint = "search_by_date" if by_date else "search"
    resp = _safe_get(
        f"https://hn.algolia.com/api/v1/{endpoint}",
        params={"query": query, "hitsPerPage": num},
    )
    if not resp:
        return None
    data = resp.json()
    results = []
    for hit in data.get("hits", []):
        title = hit.get("title") or hit.get("story_title") or hit.get("comment_text", "")[:120]
        url = hit.get("url") or f"https://news.ycombinator.com/item?id={hit.get('objectID')}"
        snippet = hit.get("story_text") or hit.get("comment_text", "")
        results.append({"title": (title or "")[:300], "url": url, "snippet": (snippet or "")[:500]})
    return {"provider": f"hn.algolia.{endpoint}", "results": results, "raw": {"hits": len(results)}}


def reddit_search(query: str, num: int = 25, sort: str = "relevance") -> dict[str, Any] | None:
    resp = _safe_get(
        "https://www.reddit.com/search.json",
        params={"q": query, "limit": num, "sort": sort, "raw_json": 1},
        headers={"User-Agent": USER_AGENT},
    )
    if not resp:
        return None
    data = resp.json()
    results = []
    for post in data.get("data", {}).get("children", []):
        d = post.get("data", {})
        results.append({
            "title": d.get("title", ""),
            "url": "https://reddit.com" + d.get("permalink", ""),
            "snippet": (d.get("selftext", "") or d.get("link_flair_text", ""))[:500],
        })
    return {"provider": "reddit", "results": results, "raw": {"count": len(results)}}


def marginalia_search(query: str, num: int = 10) -> dict[str, Any] | None:
    resp = _safe_get(
        "https://api.marginalia.nu/public/search",
        params={"query": query, "count": num, "index": "0"},
    )
    if resp:
        try:
            data = resp.json()
            results = [
                {"title": r.get("title", ""), "url": r.get("url", ""), "snippet": r.get("description", "")}
                for r in data.get("results", [])[:num]
            ]
            return {"provider": "marginalia.api", "results": results, "raw": {"count": len(results)}}
        except ValueError:
            pass
    resp = _safe_get(
        "https://search.marginalia.nu/search",
        params={"query": query, "profile": "no-js"},
    )
    if not resp:
        return None
    pattern = re.compile(
        r'<a[^>]+class="result-title-a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>.*?'
        r'<div[^>]+class="description[^"]*"[^>]*>(.*?)</div>',
        re.DOTALL,
    )
    cleaner = re.compile(r"<[^>]+>")
    results = []
    for m in pattern.finditer(resp.text):
        results.append({
            "title": cleaner.sub("", m.group(2)).strip(),
            "url": m.group(1),
            "snippet": cleaner.sub("", m.group(3)).strip(),
        })
        if len(results) >= num:
            break
    return {"provider": "marginalia.html", "results": results, "raw": {"count": len(results)}}


def searx_be_federation(query: str, num: int = 10) -> dict[str, Any] | None:
    resp = _safe_get(
        "https://searx.be/search",
        params={"q": query, "format": "json"},
        headers={"User-Agent": USER_AGENT},
    )
    if not resp:
        return None
    try:
        data = resp.json()
    except ValueError:
        return None
    results = [
        {"title": r.get("title", ""), "url": r.get("url", ""), "snippet": r.get("content", "")}
        for r in data.get("results", [])[:num]
    ]
    return {"provider": "searx.be", "results": results, "raw": {"count": len(results)}}


def github_code(query: str, num: int = 10) -> dict[str, Any] | None:
    headers = {"User-Agent": USER_AGENT, "Accept": "application/vnd.github+json"}
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    resp = _safe_get(
        "https://api.github.com/search/code",
        params={"q": query, "per_page": num},
        headers=headers,
    )
    if not resp:
        return None
    data = resp.json()
    results = []
    for item in data.get("items", []):
        results.append({
            "title": f"{item.get('repository', {}).get('full_name', '')}/{item.get('path', '')}",
            "url": item.get("html_url", ""),
            "snippet": (item.get("text_matches", [{}])[0].get("fragment", "") if item.get("text_matches") else ""),
        })
    return {"provider": "github.code", "results": results, "raw": {"total": data.get("total_count", 0)}}


def stack_exchange(query: str, num: int = 10, site: str = "stackoverflow") -> dict[str, Any] | None:
    resp = _safe_get(
        "https://api.stackexchange.com/2.3/search/advanced",
        params={"q": query, "site": site, "pagesize": num, "order": "desc", "sort": "relevance"},
    )
    if not resp:
        return None
    data = resp.json()
    results = []
    for item in data.get("items", []):
        results.append({
            "title": item.get("title", ""),
            "url": item.get("link", ""),
            "snippet": f"score={item.get('score')} answers={item.get('answer_count')} tags={','.join(item.get('tags', []))}",
        })
    return {"provider": f"stackexchange.{site}", "results": results, "raw": {"count": len(results)}}


def arxiv_search(query: str, num: int = 10) -> dict[str, Any] | None:
    resp = _safe_get(
        "http://export.arxiv.org/api/query",
        params={"search_query": f"all:{query}", "start": 0, "max_results": num},
    )
    if not resp:
        return None
    try:
        ns = {"a": "http://www.w3.org/2005/Atom"}
        root = ElementTree.fromstring(resp.text)
    except ElementTree.ParseError:
        return None
    results = []
    for entry in root.findall("a:entry", ns):
        title_el = entry.find("a:title", ns)
        link_el = entry.find("a:id", ns)
        sum_el = entry.find("a:summary", ns)
        results.append({
            "title": (title_el.text or "").strip() if title_el is not None else "",
            "url": (link_el.text or "").strip() if link_el is not None else "",
            "snippet": (sum_el.text or "").strip()[:500] if sum_el is not None else "",
        })
    return {"provider": "arxiv", "results": results, "raw": {"count": len(results)}}


def openalex_search(query: str, num: int = 10) -> dict[str, Any] | None:
    resp = _safe_get(
        "https://api.openalex.org/works",
        params={"search": query, "per_page": num},
        headers={"User-Agent": f"{USER_AGENT} mailto:goldenbergyaniv@gmail.com"},
    )
    if not resp:
        return None
    data = resp.json()
    results = []
    for w in data.get("results", []):
        abstract = w.get("abstract_inverted_index")
        snippet = ""
        if isinstance(abstract, dict):
            words = sorted(((pos, w_) for w_, positions in abstract.items() for pos in positions), key=lambda x: x[0])
            snippet = " ".join(w_ for _, w_ in words)[:500]
        results.append({
            "title": w.get("title", ""),
            "url": w.get("doi") or w.get("id", ""),
            "snippet": snippet,
        })
    return {"provider": "openalex", "results": results, "raw": {"count": len(results)}}


def pubmed_search(query: str, num: int = 10) -> dict[str, Any] | None:
    s = _safe_get(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
        params={"db": "pubmed", "term": query, "retmax": num, "retmode": "json"},
    )
    if not s:
        return None
    ids = s.json().get("esearchresult", {}).get("idlist", [])
    if not ids:
        return {"provider": "pubmed", "results": [], "raw": {}}
    sm = _safe_get(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
        params={"db": "pubmed", "id": ",".join(ids), "retmode": "json"},
    )
    if not sm:
        return None
    data = sm.json().get("result", {})
    results = []
    for pid in ids:
        item = data.get(pid, {})
        results.append({
            "title": item.get("title", ""),
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pid}/",
            "snippet": (item.get("source", "") or "") + " " + (item.get("pubdate", "") or ""),
        })
    return {"provider": "pubmed", "results": results, "raw": {"ids": ids}}


def crossref_search(query: str, num: int = 10) -> dict[str, Any] | None:
    resp = _safe_get(
        "https://api.crossref.org/works",
        params={"query": query, "rows": num},
        headers={"User-Agent": f"{USER_AGENT} mailto:goldenbergyaniv@gmail.com"},
    )
    if not resp:
        return None
    data = resp.json().get("message", {}).get("items", [])
    results = []
    for w in data:
        title = (w.get("title") or [""])[0]
        results.append({
            "title": title,
            "url": w.get("URL", ""),
            "snippet": (w.get("abstract") or "")[:500],
        })
    return {"provider": "crossref", "results": results, "raw": {"count": len(results)}}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Free zero-key search providers")
    parser.add_argument("provider", choices=[
        "ddg-html", "ddg-instant", "wikipedia", "hn", "hn-date", "reddit",
        "marginalia", "searxbe", "github", "stackoverflow", "arxiv",
        "openalex", "pubmed", "crossref",
    ])
    parser.add_argument("query")
    parser.add_argument("--num", type=int, default=10)
    args = parser.parse_args()
    fn = {
        "ddg-html": ddg_html_search,
        "ddg-instant": lambda q, num: ddg_instant_answer(q),
        "wikipedia": wikipedia_search,
        "hn": hn_algolia,
        "hn-date": lambda q, num: hn_algolia(q, num, by_date=True),
        "reddit": reddit_search,
        "marginalia": marginalia_search,
        "searxbe": searx_be_federation,
        "github": github_code,
        "stackoverflow": stack_exchange,
        "arxiv": arxiv_search,
        "openalex": openalex_search,
        "pubmed": pubmed_search,
        "crossref": crossref_search,
    }[args.provider]
    out = fn(args.query, args.num) if args.provider != "ddg-instant" else fn(args.query, 0)
    print(json.dumps(out, indent=2, ensure_ascii=False))
