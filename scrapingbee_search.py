#!/usr/bin/env python3
"""ScrapingBee API wrapper with key rotation.

ScrapingBee renders pages with rotating proxies + headless browser, and exposes
a Google SERP endpoint built on top of that.

Endpoints:
  GET https://app.scrapingbee.com/api/v1/             - render any URL
  GET https://app.scrapingbee.com/api/v1/store/google - Google SERP

Free tier: 1,000 API credits/key/mo. SERP queries cost ~25 credits each;
plain HTML scrapes cost 1; JS-rendered scrapes cost 5; premium proxy +20.

Env vars (from /config/workspace/.env):
  SCRAPINGBEE_API_KEYS  - comma-separated pool
  SCRAPINGBEE_API_KEY   - single key fallback

Usage:
    python3 scripts/scrapingbee_search.py search "best ai writing tools" --num 10
    python3 scripts/scrapingbee_search.py scrape https://example.com
    python3 scripts/scrapingbee_search.py scrape https://x.com --render-js --premium
    python3 scripts/scrapingbee_search.py keys-status
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

SB_RENDER = "https://app.scrapingbee.com/api/v1/"
SB_GOOGLE = "https://app.scrapingbee.com/api/v1/store/google"
DEFAULT_TIMEOUT = 90


def _load_keys() -> list[str]:
    pool = os.environ.get("SCRAPINGBEE_API_KEYS", "").strip()
    if pool:
        keys = [k.strip() for k in pool.split(",") if k.strip()]
        if keys:
            return keys
    single = os.environ.get("SCRAPINGBEE_API_KEY", "").strip()
    return [single] if single else []


class ScrapingBeeClient:
    def __init__(self, keys: list[str] | None = None) -> None:
        self.keys = keys if keys is not None else _load_keys()
        if not self.keys:
            raise RuntimeError(
                "No ScrapingBee keys found. Set SCRAPINGBEE_API_KEYS or SCRAPINGBEE_API_KEY in /config/workspace/.env"
            )
        self._cycle = itertools.cycle(self.keys)
        self._exhausted: set[str] = set()

    def _next_key(self) -> str:
        for _ in range(len(self.keys)):
            k = next(self._cycle)
            if k not in self._exhausted:
                return k
        raise RuntimeError("All ScrapingBee keys exhausted")

    def _get(self, base: str, params: dict[str, Any]) -> dict[str, Any]:
        last_err: str | None = None
        for _ in range(len(self.keys) + 1):
            try:
                key = self._next_key()
            except RuntimeError as e:
                last_err = str(e)
                break
            params = dict(params, api_key=key)
            try:
                resp = requests.get(base, params=params, timeout=DEFAULT_TIMEOUT)
            except requests.RequestException as e:
                last_err = f"network: {e}"
                logger.warning("ScrapingBee request failed (key=%s): %s", key[:8], e)
                continue
            if resp.status_code == 200:
                ctype = resp.headers.get("content-type", "")
                if "application/json" in ctype:
                    return resp.json()
                return {"content": resp.text, "url": params.get("url"), "status": 200}
            if resp.status_code in (401, 403):
                self._exhausted.add(key)
                last_err = f"http {resp.status_code}: {resp.text[:200]}"
                logger.warning("ScrapingBee key invalid (key=%s)", key[:8])
                continue
            if resp.status_code == 429:
                self._exhausted.add(key)
                last_err = "http 429"
                continue
            last_err = f"http {resp.status_code}: {resp.text[:200]}"
            logger.error("ScrapingBee unexpected status %s: %s", resp.status_code, resp.text[:200])
            time.sleep(0.5)
        return {"error": last_err or "all keys failed"}

    def search(self, query: str, num_results: int = 10, country_code: str = "us") -> dict[str, Any]:
        params: dict[str, Any] = {
            "search": query,
            "country_code": country_code,
            "nb_results": num_results,
        }
        return self._get(SB_GOOGLE, params)

    def scrape(
        self,
        url: str,
        render_js: bool = False,
        premium_proxy: bool = False,
        country_code: str | None = None,
        wait: int | None = None,
        return_json: bool = False,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"url": url}
        params["render_js"] = "true" if render_js else "false"
        if premium_proxy:
            params["premium_proxy"] = "true"
        if country_code:
            params["country_code"] = country_code
        if wait is not None:
            params["wait"] = wait
        if return_json:
            params["json_response"] = "true"
        return self._get(SB_RENDER, params)

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
    parser = argparse.ArgumentParser(description="ScrapingBee client with key rotation")
    sub = parser.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("search", help="Google SERP via ScrapingBee")
    s.add_argument("query")
    s.add_argument("--num", type=int, default=10)
    s.add_argument("--country", default="us")
    s.add_argument("--json", action="store_true", dest="as_json")

    sc = sub.add_parser("scrape", help="Render and return any URL")
    sc.add_argument("url")
    sc.add_argument("--render-js", action="store_true")
    sc.add_argument("--premium", action="store_true")
    sc.add_argument("--country", default=None)
    sc.add_argument("--wait", type=int, default=None)
    sc.add_argument("--return-json", action="store_true")
    sc.add_argument("--json", action="store_true", dest="as_json")

    sub.add_parser("keys-status", help="show key pool status")

    args = parser.parse_args()
    client = ScrapingBeeClient()

    if args.cmd == "keys-status":
        print(json.dumps(client.status(), indent=2))
        return 0

    if args.cmd == "search":
        out = client.search(args.query, num_results=args.num, country_code=args.country)
        if args.as_json:
            print(json.dumps(out, indent=2, ensure_ascii=False))
        else:
            if "error" in out:
                print(f"ERROR: {out['error']}", file=sys.stderr)
                return 1
            for i, r in enumerate(out.get("organic_results", [])[: args.num], 1):
                print(f"{i}. {r.get('title', '')}\n   {r.get('url', '')}\n   {r.get('description', '')[:300]}\n")
        return 0

    if args.cmd == "scrape":
        out = client.scrape(
            args.url,
            render_js=args.render_js,
            premium_proxy=args.premium,
            country_code=args.country,
            wait=args.wait,
            return_json=args.return_json,
        )
        if args.as_json:
            print(json.dumps(out, indent=2, ensure_ascii=False)[:20000])
        else:
            if "error" in out:
                print(f"ERROR: {out['error']}", file=sys.stderr)
                return 1
            print(out.get("content", "")[:5000])
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
