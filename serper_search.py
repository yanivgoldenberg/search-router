#!/usr/bin/env python3
"""Serper.dev SERP API wrapper with key rotation.

Free tier: 2,500 queries/key/month. Pool 3 keys for ~7,500/mo.

Endpoints:
  /search   - Google web SERP (organic + answerBox + knowledgeGraph + people-also-ask)
  /images   - Google Images
  /news     - Google News
  /places   - Google Maps / Places
  /scholar  - Google Scholar
  /shopping - Google Shopping
  /videos   - Google Videos
  /webpage  - Render + return HTML/markdown of a URL (paid)
  /reviews  - Place reviews
  /autocomplete - Google autocomplete

Env vars (from /config/workspace/.env):
  SERPER_API_KEYS  - comma-separated pool (preferred, enables rotation)
  SERPER_API_KEY   - single key (fallback)

Usage:
    python3 scripts/serper_search.py search "best ai writing tools" --num 20
    python3 scripts/serper_search.py search "ai" --gl il --hl he --num 10
    python3 scripts/serper_search.py news "openai" --num 10 --json
    python3 scripts/serper_search.py images "logo design" --num 8
    python3 scripts/serper_search.py places "coffee" --location "Tel Aviv,Israel"
    python3 scripts/serper_search.py scholar "transformer architecture"
    python3 scripts/serper_search.py autocomplete "best ai"
    python3 scripts/serper_search.py keys-status
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

SERPER_BASE = "https://google.serper.dev"
DEFAULT_TIMEOUT = 20

ENDPOINTS = {
    "search": "/search",
    "news": "/news",
    "images": "/images",
    "places": "/places",
    "scholar": "/scholar",
    "shopping": "/shopping",
    "videos": "/videos",
    "webpage": "/webpage",
    "reviews": "/reviews",
    "autocomplete": "/autocomplete",
}


def _load_keys() -> list[str]:
    pool = os.environ.get("SERPER_API_KEYS", "").strip()
    if pool:
        keys = [k.strip() for k in pool.split(",") if k.strip()]
        if keys:
            return keys
    single = os.environ.get("SERPER_API_KEY", "").strip()
    return [single] if single else []


class SerperClient:
    def __init__(self, keys: list[str] | None = None) -> None:
        self.keys = keys if keys is not None else _load_keys()
        if not self.keys:
            raise RuntimeError(
                "No Serper keys found. Set SERPER_API_KEYS or SERPER_API_KEY in /config/workspace/.env"
            )
        self._cycle = itertools.cycle(self.keys)
        self._exhausted: set[str] = set()

    def _next_key(self) -> str:
        for _ in range(len(self.keys)):
            k = next(self._cycle)
            if k not in self._exhausted:
                return k
        raise RuntimeError("All Serper keys exhausted (HTTP 429)")

    def call(
        self,
        endpoint: str,
        query: str,
        num: int = 10,
        gl: str = "us",
        hl: str = "en",
        location: str | None = None,
        page: int = 1,
        autocorrect: bool = True,
        tbs: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        path = ENDPOINTS.get(endpoint)
        if not path:
            raise ValueError(f"Unknown endpoint: {endpoint}. Choices: {list(ENDPOINTS)}")

        payload: dict[str, Any] = {
            "q": query,
            "num": num,
            "gl": gl,
            "hl": hl,
            "page": page,
            "autocorrect": autocorrect,
        }
        if location:
            payload["location"] = location
        if tbs:
            payload["tbs"] = tbs
        if extra:
            payload.update(extra)

        last_err: str | None = None
        for attempt in range(len(self.keys) + 1):
            try:
                key = self._next_key()
            except RuntimeError as e:
                last_err = str(e)
                break

            headers = {"X-API-KEY": key, "Content-Type": "application/json"}
            try:
                resp = requests.post(
                    f"{SERPER_BASE}{path}",
                    headers=headers,
                    json=payload,
                    timeout=DEFAULT_TIMEOUT,
                )
            except requests.RequestException as e:
                last_err = f"network: {e}"
                logger.warning("Serper request failed (key=%s): %s", key[:8], e)
                continue

            if resp.status_code == 200:
                return resp.json()
            if resp.status_code in (401, 403):
                logger.warning("Serper key invalid/forbidden (key=%s): %s", key[:8], resp.text[:200])
                self._exhausted.add(key)
                last_err = f"http {resp.status_code}: {resp.text[:200]}"
                continue
            if resp.status_code == 429:
                logger.warning("Serper key rate-limited (key=%s)", key[:8])
                self._exhausted.add(key)
                last_err = "http 429: quota or rate limit"
                continue
            last_err = f"http {resp.status_code}: {resp.text[:200]}"
            logger.error("Serper unexpected status %s: %s", resp.status_code, resp.text[:200])
            time.sleep(0.5)

        return {"error": last_err or "all keys failed", "endpoint": endpoint, "query": query}

    def status(self) -> dict[str, Any]:
        return {
            "total_keys": len(self.keys),
            "exhausted": len(self._exhausted),
            "live": len(self.keys) - len(self._exhausted),
            "keys_preview": [f"{k[:8]}..." for k in self.keys],
        }


def main() -> int:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    parser = argparse.ArgumentParser(description="Serper.dev SERP client with key rotation")
    sub = parser.add_subparsers(dest="cmd", required=True)

    for ep in ENDPOINTS:
        if ep == "autocomplete":
            p = sub.add_parser(ep, help=f"Serper {ep}")
            p.add_argument("query", help="query string")
            p.add_argument("--gl", default="us")
            p.add_argument("--hl", default="en")
            p.add_argument("--json", action="store_true", dest="as_json")
            continue
        p = sub.add_parser(ep, help=f"Serper {ep}")
        p.add_argument("query", help="query string")
        p.add_argument("--num", type=int, default=10)
        p.add_argument("--gl", default="us", help="country code (e.g. il, us, gb)")
        p.add_argument("--hl", default="en", help="language code (e.g. he, en)")
        p.add_argument("--location", default=None, help="full location string")
        p.add_argument("--page", type=int, default=1)
        p.add_argument("--tbs", default=None, help="time/range filter, e.g. qdr:d (day), qdr:w (week)")
        p.add_argument("--json", action="store_true", dest="as_json")

    sub.add_parser("keys-status", help="show key pool status")

    args = parser.parse_args()
    client = SerperClient()

    if args.cmd == "keys-status":
        print(json.dumps(client.status(), indent=2))
        return 0

    kwargs: dict[str, Any] = {"query": args.query, "gl": args.gl, "hl": args.hl}
    if args.cmd != "autocomplete":
        kwargs["num"] = args.num
        kwargs["page"] = args.page
        if args.location:
            kwargs["location"] = args.location
        if args.tbs:
            kwargs["tbs"] = args.tbs

    result = client.call(args.cmd, **kwargs)

    if args.as_json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        if "error" in result:
            print(f"ERROR: {result['error']}", file=sys.stderr)
            return 1
        organic = result.get("organic", []) or result.get("news", []) or result.get("images", []) or result.get("places", [])
        for i, item in enumerate(organic[: kwargs.get("num", 10)], 1):
            title = item.get("title", "")
            link = item.get("link", item.get("imageUrl", ""))
            snippet = item.get("snippet", item.get("description", ""))
            print(f"{i}. {title}\n   {link}\n   {snippet}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
