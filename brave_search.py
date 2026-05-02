#!/usr/bin/env python3
"""Brave Search API wrapper with key rotation.

Sign up: https://api.search.brave.com  (free tier: 2,000/key/mo, 1 req/sec)

Endpoints:
  /res/v1/web/search   - Web SERP
  /res/v1/news/search  - News
  /res/v1/images/search - Images
  /res/v1/videos/search - Videos
  /res/v1/summarizer/search - AI summary (paid plans only)

Env vars:
  BRAVE_API_KEYS  - comma-separated pool
  BRAVE_API_KEY   - single key fallback

Usage:
    python3 scripts/brave_search.py search "best ai tools" --num 10
    python3 scripts/brave_search.py news "openai" --num 5
    python3 scripts/brave_search.py keys-status
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

BRAVE_BASE = "https://api.search.brave.com/res/v1"
DEFAULT_TIMEOUT = 15


def _load_keys() -> list[str]:
    pool = os.environ.get("BRAVE_API_KEYS", "").strip()
    if pool:
        keys = [k.strip() for k in pool.split(",") if k.strip()]
        if keys:
            return keys
    single = os.environ.get("BRAVE_API_KEY", "").strip()
    return [single] if single else []


class BraveClient:
    def __init__(self, keys: list[str] | None = None) -> None:
        self.keys = keys if keys is not None else _load_keys()
        if not self.keys:
            raise RuntimeError("No Brave keys. Set BRAVE_API_KEYS or BRAVE_API_KEY in .env")
        self._cycle = itertools.cycle(self.keys)
        self._exhausted: set[str] = set()

    def _next_key(self) -> str:
        for _ in range(len(self.keys)):
            k = next(self._cycle)
            if k not in self._exhausted:
                return k
        raise RuntimeError("All Brave keys exhausted")

    def _get(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        last_err = None
        for _ in range(len(self.keys) + 1):
            try:
                key = self._next_key()
            except RuntimeError as e:
                last_err = str(e); break
            headers = {"X-Subscription-Token": key, "Accept": "application/json"}
            try:
                resp = requests.get(f"{BRAVE_BASE}{path}", params=params, headers=headers, timeout=DEFAULT_TIMEOUT)
            except requests.RequestException as e:
                last_err = f"network: {e}"; continue
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code in (401, 403):
                self._exhausted.add(key); last_err = f"http {resp.status_code}"; continue
            if resp.status_code == 429:
                self._exhausted.add(key); last_err = "http 429"; continue
            last_err = f"http {resp.status_code}: {resp.text[:200]}"
            time.sleep(0.5)
        return {"error": last_err or "all keys failed"}

    def search(self, query: str, num: int = 10, country: str = "US") -> dict[str, Any]:
        return self._get("/web/search", {"q": query, "count": min(num, 20), "country": country})

    def news(self, query: str, num: int = 10, country: str = "US") -> dict[str, Any]:
        return self._get("/news/search", {"q": query, "count": min(num, 20), "country": country})

    def images(self, query: str, num: int = 10) -> dict[str, Any]:
        return self._get("/images/search", {"q": query, "count": min(num, 100)})

    def status(self) -> dict[str, Any]:
        return {"total_keys": len(self.keys), "exhausted": len(self._exhausted),
                "live": len(self.keys) - len(self._exhausted),
                "keys_preview": [f"{k[:10]}..." for k in self.keys]}


def main() -> int:
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    for ep in ("search", "news", "images"):
        sp = sub.add_parser(ep); sp.add_argument("query")
        sp.add_argument("--num", type=int, default=10)
        sp.add_argument("--country", default="US")
        sp.add_argument("--json", action="store_true", dest="as_json")
    sub.add_parser("keys-status")

    args = p.parse_args()
    c = BraveClient()
    if args.cmd == "keys-status":
        print(json.dumps(c.status(), indent=2)); return 0
    fn = {"search": c.search, "news": c.news, "images": c.images}[args.cmd]
    out = fn(args.query, num=args.num, country=getattr(args, "country", "US")) if args.cmd != "images" else fn(args.query, num=args.num)
    if args.as_json:
        print(json.dumps(out, indent=2, ensure_ascii=False)); return 0 if "error" not in out else 1
    if "error" in out:
        print(f"ERROR: {out['error']}", file=sys.stderr); return 1
    items = (out.get("web") or {}).get("results", []) or out.get("results", [])
    for i, r in enumerate(items[: args.num], 1):
        print(f"{i}. {r.get('title','')}\n   {r.get('url','')}\n   {(r.get('description') or '')[:300]}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
