#!/usr/bin/env python3
"""You.com Search API wrapper with key rotation.

Endpoints:
  /search   - Web search (snippets + URLs + AI snippets)
  /news     - News search

Smart/Research endpoints (chat-style, separate base URL):
  POST https://chat-api.you.com/smart    - LLM-grounded answer
  POST https://chat-api.you.com/research - Deep research (slow, expensive)

Free tier: 1,000 calls/key/mo (web + news). Smart/Research counted separately.

Env vars (from /config/workspace/.env):
  YOUCOM_API_KEYS  - comma-separated pool
  YOUCOM_API_KEY   - single key fallback

Usage:
    python3 scripts/youcom_search.py search "ai safety research" --num 10
    python3 scripts/youcom_search.py news "openai" --num 5
    python3 scripts/youcom_search.py smart "what is RAG" --json
    python3 scripts/youcom_search.py keys-status
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

ENV_PATH = Path("/config/workspace/.env")
if ENV_PATH.exists():
    load_dotenv(ENV_PATH)

YOU_SEARCH_BASE = "https://api.you.com/v1"
YOU_CHAT_BASE = "https://chat-api.you.com"
DEFAULT_TIMEOUT = 60


def _load_keys() -> list[str]:
    pool = os.environ.get("YOUCOM_API_KEYS", "").strip()
    if pool:
        keys = [k.strip() for k in pool.split(",") if k.strip()]
        if keys:
            return keys
    single = os.environ.get("YOUCOM_API_KEY", "").strip()
    return [single] if single else []


class YouComClient:
    def __init__(self, keys: list[str] | None = None) -> None:
        self.keys = keys if keys is not None else _load_keys()
        if not self.keys:
            raise RuntimeError(
                "No You.com keys found. Set YOUCOM_API_KEYS or YOUCOM_API_KEY in /config/workspace/.env"
            )
        self._cycle = itertools.cycle(self.keys)
        self._exhausted: set[str] = set()

    def _next_key(self) -> str:
        for _ in range(len(self.keys)):
            k = next(self._cycle)
            if k not in self._exhausted:
                return k
        raise RuntimeError("All You.com keys exhausted")

    def _request(self, method: str, url: str, **kw: Any) -> dict[str, Any]:
        last_err: str | None = None
        for _ in range(len(self.keys) + 1):
            try:
                key = self._next_key()
            except RuntimeError as e:
                last_err = str(e)
                break
            headers = kw.pop("headers", {}) | {"X-API-Key": key}
            try:
                resp = requests.request(method, url, headers=headers, timeout=DEFAULT_TIMEOUT, **kw)
            except requests.RequestException as e:
                last_err = f"network: {e}"
                logger.warning("You.com request failed (key=%s): %s", key[:8], e)
                continue
            if resp.status_code == 200:
                try:
                    return resp.json()
                except ValueError:
                    return {"raw": resp.text}
            if resp.status_code in (401, 403):
                logger.warning("You.com key invalid (key=%s): %s", key[:8], resp.text[:200])
                self._exhausted.add(key)
                last_err = f"http {resp.status_code}: {resp.text[:200]}"
                continue
            if resp.status_code == 429:
                logger.warning("You.com key rate-limited (key=%s)", key[:8])
                self._exhausted.add(key)
                last_err = "http 429"
                continue
            last_err = f"http {resp.status_code}: {resp.text[:200]}"
            logger.error("You.com unexpected status %s: %s", resp.status_code, resp.text[:200])
            time.sleep(0.5)
        return {"error": last_err or "all keys failed", "url": url}

    def search(self, query: str, num_results: int = 10, country: str | None = None) -> dict[str, Any]:
        params: dict[str, Any] = {"query": query, "num_web_results": num_results}
        if country:
            params["country"] = country
        return self._request("GET", f"{YOU_SEARCH_BASE}/search", params=params)

    def news(self, query: str, num_results: int = 10, country: str | None = None) -> dict[str, Any]:
        params: dict[str, Any] = {"query": query, "count": num_results}
        if country:
            params["country"] = country
        return self._request("GET", f"{YOU_SEARCH_BASE}/news", params=params)

    def smart(self, query: str) -> dict[str, Any]:
        return self._request(
            "POST",
            f"{YOU_CHAT_BASE}/smart",
            json={"query": query},
            headers={"Content-Type": "application/json"},
        )

    def research(self, query: str) -> dict[str, Any]:
        return self._request(
            "POST",
            f"{YOU_CHAT_BASE}/research",
            json={"query": query},
            headers={"Content-Type": "application/json"},
        )

    def status(self) -> dict[str, Any]:
        return {
            "total_keys": len(self.keys),
            "exhausted": len(self._exhausted),
            "live": len(self.keys) - len(self._exhausted),
            "keys_preview": [f"{k[:12]}..." for k in self.keys],
        }


def _print_results(result: dict[str, Any], num: int) -> None:
    if "error" in result:
        print(f"ERROR: {result['error']}", file=sys.stderr)
        return
    web = ((result.get("results") or {}).get("web")) if isinstance(result.get("results"), dict) else None
    hits = web or result.get("hits") or (result.get("news") or {}).get("results") or result.get("results") or []
    if not isinstance(hits, list):
        hits = []
    for i, item in enumerate(hits[:num], 1):
        title = item.get("title", "")
        url = item.get("url", "")
        snippets = item.get("snippets", []) or [item.get("description", "")]
        snippet = snippets[0] if snippets else ""
        print(f"{i}. {title}\n   {url}\n   {snippet[:300]}\n")


def main() -> int:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    parser = argparse.ArgumentParser(description="You.com Search API client with key rotation")
    sub = parser.add_subparsers(dest="cmd", required=True)

    for ep in ("search", "news"):
        p = sub.add_parser(ep, help=f"You.com /{ep}")
        p.add_argument("query")
        p.add_argument("--num", type=int, default=10)
        p.add_argument("--country", default=None)
        p.add_argument("--json", action="store_true", dest="as_json")

    for ep in ("smart", "research"):
        p = sub.add_parser(ep, help=f"You.com /{ep} (chat-style)")
        p.add_argument("query")
        p.add_argument("--json", action="store_true", dest="as_json")

    sub.add_parser("keys-status", help="show key pool status")

    args = parser.parse_args()
    client = YouComClient()

    if args.cmd == "keys-status":
        print(json.dumps(client.status(), indent=2))
        return 0

    if args.cmd in ("search", "news"):
        fn = client.search if args.cmd == "search" else client.news
        result = fn(args.query, num_results=args.num, country=args.country)
    else:
        fn = client.smart if args.cmd == "smart" else client.research
        result = fn(args.query)

    if args.as_json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        if args.cmd in ("search", "news"):
            _print_results(result, args.num)
        else:
            print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if "error" not in result else 1


if __name__ == "__main__":
    sys.exit(main())
