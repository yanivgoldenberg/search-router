#!/usr/bin/env python3
"""Unified search router with Redis cache + multi-provider cascade.

Routes a query to the cheapest viable engine for its type, with key rotation
inside each provider. Caches results in Redis for 7 days.

Supported types:
  serp      - Google-style organic results
  news      - News articles
  ai        - AI-grounded answer (Tavily/Exa/You.com smart)
  scrape    - URL-to-markdown
  deep      - Multi-source synthesis

Cascade order (per type):
  serp:   cache -> Serper -> SearXNG -> Brave* -> Exa(keyword)
  news:   cache -> Serper(news) -> You.com(news) -> SearXNG(news)
  ai:     cache -> Tavily -> Exa(deep-lite) -> You.com(smart)
  scrape: cache -> Jina(r.jina.ai) -> Firecrawl -> Browserless*
  deep:   cache -> Exa(deep) -> Tavily(advanced) -> You.com(research)

(*) Optional providers; skipped silently if not wired.

Env vars: SERPER_API_KEYS, EXA_API_KEYS, YOUCOM_API_KEYS, TAVILY_API_KEYS,
          FIRECRAWL_API_KEY, JINA_API_KEY (optional), REDIS_HOST, REDIS_PORT,
          REDIS_PASSWORD, SEARXNG_URL.

Usage:
    python3 scripts/search_router.py serp "best ai writing tools 2026" --num 10
    python3 scripts/search_router.py ai "what is RAG" --json
    python3 scripts/search_router.py scrape https://example.com/article
    python3 scripts/search_router.py deep "transformer scaling laws" --json
    python3 scripts/search_router.py news "openai" --num 5
    python3 scripts/search_router.py stats
    python3 scripts/search_router.py cache-clear "best ai writing tools 2026"

Module API:
    from scripts.search_router import SearchRouter
    r = SearchRouter()
    out = r.run("serp", "best ai writing tools", num=10)
    # out = {"provider": "serper", "cached": False, "results": [...], "raw": {...}}
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

import requests
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

ENV_PATH = Path("/config/workspace/.env")
if ENV_PATH.exists():
    load_dotenv(ENV_PATH)

sys.path.insert(0, str(Path(__file__).parent))

try:
    import redis
except Exception as _redis_e:
    logger.warning("redis module unavailable (%s) - cache disabled", _redis_e)
    redis = None

try:
    from serper_search import SerperClient
except Exception:
    SerperClient = None
try:
    from exa_search import ExaClient
except Exception:
    ExaClient = None
try:
    from youcom_search import YouComClient
except Exception:
    YouComClient = None
try:
    from scrapingbee_search import ScrapingBeeClient
except Exception:
    ScrapingBeeClient = None
try:
    from serpapi_search import SerpApiClient
except Exception:
    SerpApiClient = None
try:
    import search_free_providers as freep
except Exception:
    freep = None
try:
    from brave_search import BraveClient
except Exception:
    BraveClient = None
try:
    from mojeek_search import MojeekClient
except Exception:
    MojeekClient = None
try:
    import reddit_oauth_search as reddit_oauth
except Exception:
    reddit_oauth = None

CACHE_TTL_SECONDS = 7 * 24 * 3600
CACHE_PREFIX = "search:"


def _cache_key(provider_hint: str, query: str, params: dict[str, Any]) -> str:
    blob = json.dumps({"p": provider_hint, "q": query, "k": params}, sort_keys=True)
    return CACHE_PREFIX + hashlib.sha256(blob.encode()).hexdigest()[:32]


class SearchRouter:
    def __init__(self) -> None:
        self.redis = self._connect_redis()
        self.serper = self._safe_init(SerperClient)
        self.exa = self._safe_init(ExaClient)
        self.youcom = self._safe_init(YouComClient)
        self.scrapingbee = self._safe_init(ScrapingBeeClient)
        self.serpapi = self._safe_init(SerpApiClient)
        self.brave = self._safe_init(BraveClient)
        self.mojeek = self._safe_init(MojeekClient)
        self.reddit_oauth_ok = bool(os.environ.get("REDDIT_CLIENT_ID") and os.environ.get("REDDIT_CLIENT_SECRET"))
        self.tavily_keys = [k.strip() for k in os.environ.get("TAVILY_API_KEYS", "").split(",") if k.strip()] or (
            [os.environ["TAVILY_API_KEY"]] if os.environ.get("TAVILY_API_KEY") else []
        )
        self.firecrawl_key = os.environ.get("FIRECRAWL_API_KEY", "").strip()
        self.firecrawl_url = os.environ.get("FIRECRAWL_API_URL", "https://api.firecrawl.dev").strip()
        self.searxng_url = os.environ.get("SEARXNG_URL", "http://searxng:8080").strip()
        self.jina_key = os.environ.get("JINA_API_KEY", "").strip()
        self._tavily_idx = 0
        self.metrics: dict[str, int] = {}

    @staticmethod
    def _safe_init(cls: Any) -> Any:
        if cls is None:
            return None
        try:
            return cls()
        except Exception as e:
            logger.warning("Provider init failed for %s: %s", getattr(cls, "__name__", cls), e)
            return None

    def _connect_redis(self) -> Any:
        if redis is None:
            return None
        host = os.environ.get("REDIS_HOST", "redis-cache")
        port = int(os.environ.get("REDIS_PORT", "6379"))
        password = os.environ.get("REDIS_PASSWORD") or None
        try:
            r = redis.Redis(host=host, port=port, password=password, socket_timeout=2, decode_responses=True)
            r.ping()
            return r
        except Exception as e:
            logger.warning("Redis cache unavailable (%s:%s): %s", host, port, e)
            return None

    def _cache_get(self, key: str) -> dict[str, Any] | None:
        if not self.redis:
            return None
        try:
            blob = self.redis.get(key)
            if blob:
                return json.loads(blob)
        except Exception as e:
            logger.warning("Redis get failed: %s", e)
        return None

    def _cache_set(self, key: str, value: dict[str, Any]) -> None:
        if not self.redis:
            return
        try:
            self.redis.setex(key, CACHE_TTL_SECONDS, json.dumps(value))
        except Exception as e:
            logger.warning("Redis set failed: %s", e)

    def _bump(self, name: str) -> None:
        self.metrics[name] = self.metrics.get(name, 0) + 1

    def _serper(self, query: str, num: int, gl: str, hl: str, endpoint: str = "search") -> dict[str, Any] | None:
        if not self.serper:
            return None
        out = self.serper.call(endpoint, query, num=num, gl=gl, hl=hl)
        if "error" in out:
            return None
        organic = out.get("organic", []) or out.get("news", [])
        return {
            "provider": f"serper.{endpoint}",
            "results": [
                {"title": r.get("title", ""), "url": r.get("link", ""), "snippet": r.get("snippet", "")}
                for r in organic[:num]
            ],
            "raw": out,
        }

    def _searxng(self, query: str, num: int, category: str = "general") -> dict[str, Any] | None:
        try:
            resp = requests.get(
                f"{self.searxng_url}/search",
                params={"q": query, "format": "json", "categories": category},
                timeout=15,
            )
            if resp.status_code != 200:
                return None
            data = resp.json()
            results = data.get("results", [])[:num]
            return {
                "provider": "searxng",
                "results": [
                    {"title": r.get("title", ""), "url": r.get("url", ""), "snippet": r.get("content", "")}
                    for r in results
                ],
                "raw": data,
            }
        except Exception as e:
            logger.warning("SearXNG request failed: %s", e)
            return None

    def _exa_search(self, query: str, num: int, type: str = "auto") -> dict[str, Any] | None:
        if not self.exa:
            return None
        out = self.exa.search(query, type=type, num_results=num, highlights=True)
        if "error" in out:
            return None
        return {
            "provider": f"exa.{type}",
            "results": [
                {
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "snippet": " ".join(r.get("highlights", []))[:500],
                }
                for r in out.get("results", [])[:num]
            ],
            "raw": out,
        }

    def _tavily(self, query: str, num: int, search_depth: str = "basic") -> dict[str, Any] | None:
        if not self.tavily_keys:
            return None
        for _ in range(len(self.tavily_keys)):
            key = self.tavily_keys[self._tavily_idx % len(self.tavily_keys)]
            self._tavily_idx += 1
            try:
                resp = requests.post(
                    "https://api.tavily.com/search",
                    json={
                        "api_key": key,
                        "query": query,
                        "max_results": num,
                        "search_depth": search_depth,
                        "include_answer": True,
                    },
                    timeout=30,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    return {
                        "provider": f"tavily.{search_depth}",
                        "answer": data.get("answer"),
                        "results": [
                            {"title": r.get("title", ""), "url": r.get("url", ""), "snippet": r.get("content", "")}
                            for r in data.get("results", [])[:num]
                        ],
                        "raw": data,
                    }
                if resp.status_code in (401, 429):
                    continue
            except Exception as e:
                logger.warning("Tavily request failed: %s", e)
                continue
        return None

    def _youcom_search(self, query: str, num: int, news: bool = False) -> dict[str, Any] | None:
        if not self.youcom:
            return None
        out = self.youcom.news(query, num_results=num) if news else self.youcom.search(query, num_results=num)
        if "error" in out:
            return None
        web = ((out.get("results") or {}).get("web")) if isinstance(out.get("results"), dict) else None
        hits = web or out.get("hits") or (out.get("news") or {}).get("results") or []
        if not isinstance(hits, list):
            return None
        return {
            "provider": f"youcom.{'news' if news else 'search'}",
            "results": [
                {
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "snippet": (r.get("snippets", [None])[0] if r.get("snippets") else r.get("description", "")) or "",
                }
                for r in hits[:num]
            ],
            "raw": {"count": len(hits)},
        }

    def _youcom_smart(self, query: str) -> dict[str, Any] | None:
        if not self.youcom:
            return None
        out = self.youcom.smart(query)
        if "error" in out:
            return None
        return {"provider": "youcom.smart", "answer": out.get("answer") or out.get("response"), "results": [], "raw": out}

    def _brave(self, query: str, num: int, country: str, news: bool = False) -> dict[str, Any] | None:
        if not self.brave:
            return None
        out = self.brave.news(query, num=num, country=country.upper()) if news else self.brave.search(query, num=num, country=country.upper())
        if "error" in out:
            return None
        items = (out.get("web") or {}).get("results", []) or out.get("results", [])
        return {
            "provider": f"brave.{'news' if news else 'web'}",
            "results": [
                {"title": r.get("title", ""), "url": r.get("url", ""), "snippet": r.get("description", "")}
                for r in items[:num]
            ],
            "raw": {"count": len(items)},
        }

    def _mojeek(self, query: str, num: int) -> dict[str, Any] | None:
        if not self.mojeek:
            return None
        out = self.mojeek.search(query, num=num)
        if "error" in out:
            return None
        items = out.get("response", {}).get("results", [])
        return {
            "provider": "mojeek",
            "results": [
                {"title": r.get("title", ""), "url": r.get("url", ""), "snippet": r.get("desc", "")}
                for r in items[:num]
            ],
            "raw": {"count": len(items)},
        }

    def _reddit_oauth(self, query: str, num: int) -> dict[str, Any] | None:
        if not (self.reddit_oauth_ok and reddit_oauth):
            return None
        out = reddit_oauth.search(query, num=num)
        if "error" in out:
            return None
        children = (out.get("data") or {}).get("children", [])
        return {
            "provider": "reddit.oauth",
            "results": [
                {
                    "title": c.get("data", {}).get("title", ""),
                    "url": "https://reddit.com" + c.get("data", {}).get("permalink", ""),
                    "snippet": (c.get("data", {}).get("selftext", "") or "")[:500],
                }
                for c in children[:num]
            ],
            "raw": {"count": len(children)},
        }

    def _serpapi(self, query: str, num: int, gl: str, hl: str, engine: str = "google") -> dict[str, Any] | None:
        if not self.serpapi:
            return None
        out = self.serpapi.call(engine, query, num=num, gl=gl, hl=hl)
        if "error" in out:
            return None
        organic = out.get("organic_results") or out.get("news_results") or []
        return {
            "provider": f"serpapi.{engine}",
            "results": [
                {"title": r.get("title", ""), "url": r.get("link", ""), "snippet": r.get("snippet", "")}
                for r in organic[:num]
            ],
            "raw": out,
        }

    def _scrapingbee_search(self, query: str, num: int, country: str) -> dict[str, Any] | None:
        if not self.scrapingbee:
            return None
        out = self.scrapingbee.search(query, num_results=num, country_code=country)
        if "error" in out:
            return None
        return {
            "provider": "scrapingbee.google",
            "results": [
                {"title": r.get("title", ""), "url": r.get("url", ""), "snippet": r.get("description", "")}
                for r in out.get("organic_results", [])[:num]
            ],
            "raw": out,
        }

    def _scrapingbee_scrape(self, url: str) -> dict[str, Any] | None:
        if not self.scrapingbee:
            return None
        out = self.scrapingbee.scrape(url, render_js=True)
        if "error" in out:
            return None
        return {
            "provider": "scrapingbee.scrape",
            "results": [{"title": "", "url": url, "snippet": (out.get("content") or "")[:5000]}],
            "raw": {"length": len(out.get("content", ""))},
        }

    def _jina_read(self, url: str) -> dict[str, Any] | None:
        try:
            headers = {"Accept": "application/json", "X-Retain-Images": "none"}
            if self.jina_key:
                headers["Authorization"] = f"Bearer {self.jina_key}"
            resp = requests.get(f"https://r.jina.ai/{url}", headers=headers, timeout=30)
            if resp.status_code != 200:
                return None
            try:
                data = resp.json()
            except ValueError:
                data = {"content": resp.text}
            return {
                "provider": "jina.reader",
                "results": [
                    {
                        "title": data.get("data", {}).get("title", "") if isinstance(data.get("data"), dict) else "",
                        "url": url,
                        "snippet": (data.get("data", {}).get("content", "") if isinstance(data.get("data"), dict) else data.get("content", ""))[:5000],
                    }
                ],
                "raw": data,
            }
        except Exception as e:
            logger.warning("Jina reader failed: %s", e)
            return None

    def _firecrawl_scrape(self, url: str) -> dict[str, Any] | None:
        if not self.firecrawl_key:
            return None
        try:
            resp = requests.post(
                f"{self.firecrawl_url}/v1/scrape",
                headers={"Authorization": f"Bearer {self.firecrawl_key}", "Content-Type": "application/json"},
                json={"url": url, "formats": ["markdown"]},
                timeout=60,
            )
            if resp.status_code != 200:
                return None
            data = resp.json()
            md = data.get("data", {}).get("markdown", "")
            return {
                "provider": "firecrawl",
                "results": [{"title": data.get("data", {}).get("title", ""), "url": url, "snippet": md[:5000]}],
                "raw": data,
            }
        except Exception as e:
            logger.warning("Firecrawl failed: %s", e)
            return None

    def run(
        self,
        type: str,
        query: str,
        num: int = 10,
        gl: str = "us",
        hl: str = "en",
        bypass_cache: bool = False,
    ) -> dict[str, Any]:
        params = {"num": num, "gl": gl, "hl": hl}
        ck = _cache_key(type, query, params)

        if not bypass_cache:
            cached = self._cache_get(ck)
            if cached:
                self._bump("cache_hit")
                cached["cached"] = True
                return cached
        self._bump("cache_miss")

        cascade: list[Callable[[], dict[str, Any] | None]] = []

        if type == "serp":
            cascade = [
                lambda: self._serper(query, num, gl, hl, "search"),
                lambda: self._brave(query, num, gl),
                lambda: self._searxng(query, num, "general"),
                lambda: self._mojeek(query, num),
                lambda: self._exa_search(query, num, "keyword"),
                lambda: self._scrapingbee_search(query, num, gl),
                lambda: self._serpapi(query, num, gl, hl, "google"),
                lambda: (freep.ddg_html_search(query, num) if freep else None),
                lambda: (freep.marginalia_search(query, num) if freep else None),
            ]
        elif type == "news":
            cascade = [
                lambda: self._serper(query, num, gl, hl, "news"),
                lambda: self._brave(query, num, gl, news=True),
                lambda: self._youcom_search(query, num, news=True),
                lambda: self._searxng(query, num, "news"),
                lambda: self._serpapi(query, num, gl, hl, "google_news"),
                lambda: (freep.hn_algolia(query, num, by_date=True) if freep else None),
            ]
        elif type == "ai":
            cascade = [
                lambda: self._tavily(query, num, "basic"),
                lambda: self._exa_search(query, num, "deep-lite"),
                lambda: self._youcom_smart(query),
                lambda: (freep.ddg_instant_answer(query) if freep else None),
                lambda: (freep.wikipedia_search(query, num) if freep else None),
            ]
        elif type == "deep":
            cascade = [
                lambda: self._exa_search(query, num, "deep"),
                lambda: self._tavily(query, num, "advanced"),
                lambda: self._youcom_smart(query),
                lambda: (freep.openalex_search(query, num) if freep else None),
                lambda: (freep.arxiv_search(query, num) if freep else None),
            ]
        elif type == "scrape":
            cascade = [
                lambda: self._jina_read(query),
                lambda: self._firecrawl_scrape(query),
                lambda: self._scrapingbee_scrape(query),
            ]
        elif type == "code":
            cascade = [
                lambda: (freep.github_code(query, num) if freep else None),
                lambda: (freep.stack_exchange(query, num, "stackoverflow") if freep else None),
                lambda: self._serper(query, num, gl, hl, "search"),
            ]
        elif type == "academic":
            cascade = [
                lambda: (freep.openalex_search(query, num) if freep else None),
                lambda: (freep.arxiv_search(query, num) if freep else None),
                lambda: (freep.pubmed_search(query, num) if freep else None),
                lambda: (freep.crossref_search(query, num) if freep else None),
                lambda: self._serpapi(query, num, gl, hl, "google_scholar"),
            ]
        elif type == "social":
            cascade = [
                lambda: self._reddit_oauth(query, num),
                lambda: (freep.hn_algolia(query, num) if freep else None),
                lambda: self._serper(f"{query} site:reddit.com", num, gl, hl, "search"),
            ]
        else:
            return {"error": f"unknown type: {type}", "valid_types": ["serp", "news", "ai", "deep", "scrape", "code", "academic", "social"]}

        last_err: list[str] = []
        for attempt in cascade:
            t0 = time.time()
            try:
                out = attempt()
            except Exception as e:
                last_err.append(str(e))
                continue
            if out:
                out["cached"] = False
                out["latency_ms"] = int((time.time() - t0) * 1000)
                out["query"] = query
                out["type"] = type
                self._bump(f"hit.{out.get('provider', 'unknown')}")
                self._cache_set(ck, out)
                return out
            last_err.append(f"empty from {attempt}")

        self._bump("all_failed")
        return {"error": "all providers failed", "type": type, "query": query, "attempts": last_err}

    def stats(self) -> dict[str, Any]:
        return {
            "providers_loaded": {
                "serper": bool(self.serper),
                "exa": bool(self.exa),
                "youcom": bool(self.youcom),
                "scrapingbee": bool(self.scrapingbee),
                "serpapi": bool(self.serpapi),
                "brave": bool(self.brave),
                "mojeek": bool(self.mojeek),
                "reddit_oauth": self.reddit_oauth_ok,
                "free_providers": bool(freep),
                "tavily_keys": len(self.tavily_keys),
                "firecrawl": bool(self.firecrawl_key),
                "jina_key_set": bool(self.jina_key),
                "searxng_url": self.searxng_url,
                "redis": bool(self.redis),
            },
            "metrics": self.metrics,
        }

    def cache_clear(self, type: str, query: str, num: int = 10, gl: str = "us", hl: str = "en") -> bool:
        if not self.redis:
            return False
        ck = _cache_key(type, query, {"num": num, "gl": gl, "hl": hl})
        return bool(self.redis.delete(ck))


def main() -> int:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    parser = argparse.ArgumentParser(description="Unified search router (cache + cascade)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    for t in ("serp", "news", "ai", "deep", "scrape", "code", "academic", "social"):
        p = sub.add_parser(t, help=f"Run {t} cascade")
        p.add_argument("query", help="query (or URL for scrape)")
        p.add_argument("--num", type=int, default=10)
        p.add_argument("--gl", default="us")
        p.add_argument("--hl", default="en")
        p.add_argument("--no-cache", action="store_true")
        p.add_argument("--json", action="store_true", dest="as_json")

    sub.add_parser("stats", help="show provider load + metrics")

    cc = sub.add_parser("cache-clear", help="evict cached entry")
    cc.add_argument("type")
    cc.add_argument("query")
    cc.add_argument("--num", type=int, default=10)
    cc.add_argument("--gl", default="us")
    cc.add_argument("--hl", default="en")

    args = parser.parse_args()
    router = SearchRouter()

    if args.cmd == "stats":
        print(json.dumps(router.stats(), indent=2))
        return 0

    if args.cmd == "cache-clear":
        ok = router.cache_clear(args.type, args.query, args.num, args.gl, args.hl)
        print(json.dumps({"deleted": ok}))
        return 0

    out = router.run(args.cmd, args.query, num=args.num, gl=args.gl, hl=args.hl, bypass_cache=args.no_cache)

    if args.as_json:
        print(json.dumps(out, indent=2, ensure_ascii=False))
        return 0 if "error" not in out else 1

    if "error" in out:
        print(f"ERROR: {out['error']}", file=sys.stderr)
        return 1
    print(f"# provider={out.get('provider')}  cached={out.get('cached')}  latency={out.get('latency_ms')}ms")
    if out.get("answer"):
        print(f"\nANSWER: {out['answer']}\n")
    for i, r in enumerate(out.get("results", [])[: args.num], 1):
        print(f"{i}. {r.get('title', '')}\n   {r.get('url', '')}\n   {(r.get('snippet') or '')[:300]}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
