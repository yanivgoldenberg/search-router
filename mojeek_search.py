#!/usr/bin/env python3
"""Mojeek Search API wrapper with key rotation.

Sign up: https://www.mojeek.com/services/search/web-search-api/
Independent crawler/index, free tier on request.

Env vars:
  MOJEEK_API_KEYS  - comma-separated pool
  MOJEEK_API_KEY   - single key fallback
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

MOJEEK_BASE = "https://api.mojeek.com/search"
DEFAULT_TIMEOUT = 15


def _load_keys() -> list[str]:
    pool = os.environ.get("MOJEEK_API_KEYS", "").strip()
    if pool:
        keys = [k.strip() for k in pool.split(",") if k.strip()]
        if keys:
            return keys
    single = os.environ.get("MOJEEK_API_KEY", "").strip()
    return [single] if single else []


class MojeekClient:
    def __init__(self, keys: list[str] | None = None) -> None:
        self.keys = keys if keys is not None else _load_keys()
        if not self.keys:
            raise RuntimeError("No Mojeek keys. Set MOJEEK_API_KEYS or MOJEEK_API_KEY in .env")
        self._cycle = itertools.cycle(self.keys)
        self._exhausted: set[str] = set()

    def _next_key(self) -> str:
        for _ in range(len(self.keys)):
            k = next(self._cycle)
            if k not in self._exhausted:
                return k
        raise RuntimeError("All Mojeek keys exhausted")

    def search(self, query: str, num: int = 10, region: str | None = None) -> dict[str, Any]:
        last_err = None
        for _ in range(len(self.keys) + 1):
            try:
                key = self._next_key()
            except RuntimeError as e:
                last_err = str(e)
                break
            params: dict[str, Any] = {"q": query, "api_key": key, "fmt": "json", "t": min(num, 50)}
            if region:
                params["reg"] = region
            try:
                resp = requests.get(MOJEEK_BASE, params=params, timeout=DEFAULT_TIMEOUT)
            except requests.RequestException as e:
                last_err = f"network: {e}"
                continue
            if resp.status_code == 200:
                try:
                    return resp.json()
                except ValueError:
                    return {"raw": resp.text}
            if resp.status_code in (401, 403):
                self._exhausted.add(key)
                last_err = f"http {resp.status_code}"
                continue
            if resp.status_code == 429:
                self._exhausted.add(key)
                last_err = "http 429"
                continue
            last_err = f"http {resp.status_code}: {resp.text[:200]}"
            time.sleep(0.5)
        return {"error": last_err or "all keys failed"}

    def status(self) -> dict[str, Any]:
        return {
            "total_keys": len(self.keys),
            "exhausted": len(self._exhausted),
            "live": len(self.keys) - len(self._exhausted),
            "keys_preview": [f"{k[:10]}..." for k in self.keys],
        }


def main() -> int:
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("search")
    s.add_argument("query")
    s.add_argument("--num", type=int, default=10)
    s.add_argument("--region", default=None)
    s.add_argument("--json", action="store_true", dest="as_json")
    sub.add_parser("keys-status")

    args = p.parse_args()
    c = MojeekClient()
    if args.cmd == "keys-status":
        print(json.dumps(c.status(), indent=2))
        return 0
    out = c.search(args.query, num=args.num, region=args.region)
    if args.as_json:
        print(json.dumps(out, indent=2, ensure_ascii=False))
        return 0 if "error" not in out else 1
    if "error" in out:
        print(f"ERROR: {out['error']}", file=sys.stderr)
        return 1
    for i, r in enumerate(out.get("response", {}).get("results", [])[: args.num], 1):
        print(f"{i}. {r.get('title', '')}\n   {r.get('url', '')}\n   {(r.get('desc') or '')[:300]}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
