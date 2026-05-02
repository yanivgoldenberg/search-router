#!/usr/bin/env python3
"""SerpAPI (serpapi.com) wrapper with key rotation.

One endpoint, many engines via `engine=` param:
  google, google_news, google_images, google_scholar, google_maps, google_jobs,
  bing, duckduckgo, yahoo, yandex, baidu, youtube, ebay, walmart, amazon,
  apple_app_store, google_play, etc.

Free tier: 100 searches/key/mo.

Env vars (from /config/workspace/.env):
  SERPAPI_API_KEYS  - comma-separated pool
  SERPAPI_API_KEY   - single key fallback

Usage:
    python3 scripts/serpapi_search.py search "best ai writing tools" --num 10
    python3 scripts/serpapi_search.py search "ai" --engine bing --num 10
    python3 scripts/serpapi_search.py news "openai" --num 5
    python3 scripts/serpapi_search.py scholar "transformers" --num 5
    python3 scripts/serpapi_search.py images "logo" --num 8
    python3 scripts/serpapi_search.py keys-status
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

SERPAPI_BASE = "https://serpapi.com/search.json"
DEFAULT_TIMEOUT = 30

CMD_TO_ENGINE = {
    "search": "google",
    "news": "google_news",
    "images": "google_images",
    "scholar": "google_scholar",
    "maps": "google_maps",
    "jobs": "google_jobs",
}


def _load_keys() -> list[str]:
    pool = os.environ.get("SERPAPI_API_KEYS", "").strip()
    if pool:
        keys = [k.strip() for k in pool.split(",") if k.strip()]
        if keys:
            return keys
    single = os.environ.get("SERPAPI_API_KEY", "").strip()
    return [single] if single else []


class SerpApiClient:
    def __init__(self, keys: list[str] | None = None) -> None:
        self.keys = keys if keys is not None else _load_keys()
        if not self.keys:
            raise RuntimeError(
                "No SerpAPI keys found. Set SERPAPI_API_KEYS or SERPAPI_API_KEY in /config/workspace/.env"
            )
        self._cycle = itertools.cycle(self.keys)
        self._exhausted: set[str] = set()

    def _next_key(self) -> str:
        for _ in range(len(self.keys)):
            k = next(self._cycle)
            if k not in self._exhausted:
                return k
        raise RuntimeError("All SerpAPI keys exhausted")

    def call(
        self,
        engine: str,
        query: str,
        num: int = 10,
        gl: str = "us",
        hl: str = "en",
        location: str | None = None,
        engine_override: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        eng = engine_override or engine
        last_err: str | None = None
        for _ in range(len(self.keys) + 1):
            try:
                key = self._next_key()
            except RuntimeError as e:
                last_err = str(e)
                break
            params: dict[str, Any] = {
                "engine": eng,
                "q": query,
                "num": num,
                "gl": gl,
                "hl": hl,
                "api_key": key,
            }
            if location:
                params["location"] = location
            if extra:
                params.update(extra)
            try:
                resp = requests.get(SERPAPI_BASE, params=params, timeout=DEFAULT_TIMEOUT)
            except requests.RequestException as e:
                last_err = f"network: {e}"
                logger.warning("SerpAPI request failed (key=%s): %s", key[:8], e)
                continue
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code in (401, 403):
                self._exhausted.add(key)
                last_err = f"http {resp.status_code}: {resp.text[:200]}"
                continue
            if resp.status_code == 429:
                self._exhausted.add(key)
                last_err = "http 429"
                continue
            last_err = f"http {resp.status_code}: {resp.text[:200]}"
            logger.error("SerpAPI unexpected status %s: %s", resp.status_code, resp.text[:200])
            time.sleep(0.5)
        return {"error": last_err or "all keys failed", "engine": eng, "query": query}

    def status(self) -> dict[str, Any]:
        return {
            "total_keys": len(self.keys),
            "exhausted": len(self._exhausted),
            "live": len(self.keys) - len(self._exhausted),
            "keys_preview": [f"{k[:12]}..." for k in self.keys],
        }


def main() -> int:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    parser = argparse.ArgumentParser(description="SerpAPI client with key rotation")
    sub = parser.add_subparsers(dest="cmd", required=True)

    for cmd in CMD_TO_ENGINE:
        p = sub.add_parser(cmd, help=f"SerpAPI {cmd}")
        p.add_argument("query")
        p.add_argument("--num", type=int, default=10)
        p.add_argument("--gl", default="us")
        p.add_argument("--hl", default="en")
        p.add_argument("--location", default=None)
        p.add_argument("--engine", default=None, help="override engine, e.g. bing, duckduckgo")
        p.add_argument("--json", action="store_true", dest="as_json")

    sub.add_parser("keys-status", help="show key pool status")

    args = parser.parse_args()
    client = SerpApiClient()

    if args.cmd == "keys-status":
        print(json.dumps(client.status(), indent=2))
        return 0

    engine = CMD_TO_ENGINE.get(args.cmd, "google")
    out = client.call(
        engine, args.query, num=args.num, gl=args.gl, hl=args.hl,
        location=args.location, engine_override=args.engine,
    )
    if args.as_json:
        print(json.dumps(out, indent=2, ensure_ascii=False)[:30000])
        return 0 if "error" not in out else 1
    if "error" in out:
        print(f"ERROR: {out['error']}", file=sys.stderr)
        return 1
    organic = out.get("organic_results") or out.get("news_results") or out.get("images_results") or []
    for i, r in enumerate(organic[: args.num], 1):
        title = r.get("title", "")
        link = r.get("link", r.get("source", ""))
        snippet = r.get("snippet", r.get("description", ""))
        print(f"{i}. {title}\n   {link}\n   {snippet[:300]}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
