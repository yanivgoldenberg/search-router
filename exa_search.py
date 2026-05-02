#!/usr/bin/env python3
"""Exa.ai search + contents wrapper with key rotation.

Canonical reference: https://docs.exa.ai/reference/search-api-guide-for-coding-agents

Endpoints:
  /search    - Find URLs (and optionally fetch contents) via neural / keyword / auto search
  /contents  - Extract clean content for URLs you already have

Search types (default: auto):
  auto             ~1s   smart (default)
  fast             ~450ms basic
  instant          ~250ms chat/voice
  deep-lite        ~4s   cheap synthesis
  deep             4-15s research
  deep-reasoning   12-40s hardest

Env vars (from /config/workspace/.env):
  EXA_API_KEYS  - comma-separated pool (preferred for rotation)
  EXA_API_KEY   - single key fallback

Usage:
    python3 scripts/exa_search.py search "ai safety research" --num 10
    python3 scripts/exa_search.py search "rust vs go" --type deep --highlights
    python3 scripts/exa_search.py search "openai" --include arxiv.org,github.com
    python3 scripts/exa_search.py search "best laptops" --max-age-hours 24 --json
    python3 scripts/exa_search.py contents https://example.com/a https://example.com/b
    python3 scripts/exa_search.py contents https://x.com/y --max-chars 5000
    python3 scripts/exa_search.py keys-status
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

EXA_BASE = "https://api.exa.ai"
DEFAULT_TIMEOUT = 60

VALID_TYPES = {"auto", "fast", "instant", "deep-lite", "deep", "deep-reasoning", "neural", "keyword"}


def _load_keys() -> list[str]:
    pool = os.environ.get("EXA_API_KEYS", "").strip()
    if pool:
        keys = [k.strip() for k in pool.split(",") if k.strip()]
        if keys:
            return keys
    single = os.environ.get("EXA_API_KEY", "").strip()
    return [single] if single else []


class ExaClient:
    def __init__(self, keys: list[str] | None = None) -> None:
        self.keys = keys if keys is not None else _load_keys()
        if not self.keys:
            raise RuntimeError(
                "No Exa keys found. Set EXA_API_KEYS or EXA_API_KEY in /config/workspace/.env"
            )
        self._cycle = itertools.cycle(self.keys)
        self._exhausted: set[str] = set()

    def _next_key(self) -> str:
        for _ in range(len(self.keys)):
            k = next(self._cycle)
            if k not in self._exhausted:
                return k
        raise RuntimeError("All Exa keys exhausted")

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        last_err: str | None = None
        for _ in range(len(self.keys) + 1):
            try:
                key = self._next_key()
            except RuntimeError as e:
                last_err = str(e)
                break

            headers = {"x-api-key": key, "Content-Type": "application/json"}
            try:
                resp = requests.post(
                    f"{EXA_BASE}{path}", headers=headers, json=payload, timeout=DEFAULT_TIMEOUT
                )
            except requests.RequestException as e:
                last_err = f"network: {e}"
                logger.warning("Exa request failed (key=%s): %s", key[:8], e)
                continue

            if resp.status_code == 200:
                return resp.json()
            if resp.status_code in (401, 403):
                logger.warning("Exa key invalid (key=%s): %s", key[:8], resp.text[:200])
                self._exhausted.add(key)
                last_err = f"http {resp.status_code}: {resp.text[:200]}"
                continue
            if resp.status_code == 429:
                logger.warning("Exa key rate-limited (key=%s)", key[:8])
                self._exhausted.add(key)
                last_err = "http 429"
                continue
            last_err = f"http {resp.status_code}: {resp.text[:200]}"
            logger.error("Exa unexpected status %s: %s", resp.status_code, resp.text[:200])
            time.sleep(0.5)

        return {"error": last_err or "all keys failed", "path": path}

    def search(
        self,
        query: str,
        *,
        type: str = "auto",
        num_results: int = 10,
        include_domains: list[str] | None = None,
        exclude_domains: list[str] | None = None,
        category: str | None = None,
        max_age_hours: int | None = None,
        highlights: bool = True,
        text: bool = False,
        text_max_chars: int | None = None,
        text_verbosity: str | None = None,
        summary: bool | str | None = None,
        output_schema: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if type not in VALID_TYPES:
            raise ValueError(f"Invalid type {type}. Choices: {sorted(VALID_TYPES)}")

        contents: dict[str, Any] = {}
        if highlights:
            contents["highlights"] = True
        if text:
            text_block: dict[str, Any] = {}
            if text_max_chars:
                text_block["maxCharacters"] = text_max_chars
            if text_verbosity:
                text_block["verbosity"] = text_verbosity
            contents["text"] = text_block or True
        if summary is not None:
            if isinstance(summary, str):
                contents["summary"] = {"query": summary}
            else:
                contents["summary"] = bool(summary)
        if max_age_hours is not None:
            contents["maxAgeHours"] = max_age_hours

        payload: dict[str, Any] = {
            "query": query,
            "type": type,
            "numResults": num_results,
        }
        if contents:
            payload["contents"] = contents
        if include_domains:
            payload["includeDomains"] = include_domains
        if exclude_domains:
            payload["excludeDomains"] = exclude_domains
        if category:
            payload["category"] = category
        if output_schema:
            payload["outputSchema"] = output_schema
        if extra:
            payload.update(extra)

        return self._post("/search", payload)

    def contents(
        self,
        urls: list[str],
        *,
        highlights: bool = True,
        text: bool = False,
        text_max_chars: int | None = None,
        max_age_hours: int | None = None,
        summary: bool | str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"urls": urls}
        if highlights:
            payload["highlights"] = True
        if text:
            text_block: dict[str, Any] = {}
            if text_max_chars:
                text_block["maxCharacters"] = text_max_chars
            payload["text"] = text_block or True
        if summary is not None:
            if isinstance(summary, str):
                payload["summary"] = {"query": summary}
            else:
                payload["summary"] = bool(summary)
        if max_age_hours is not None:
            payload["maxAgeHours"] = max_age_hours

        return self._post("/contents", payload)

    def status(self) -> dict[str, Any]:
        return {
            "total_keys": len(self.keys),
            "exhausted": len(self._exhausted),
            "live": len(self.keys) - len(self._exhausted),
            "keys_preview": [f"{k[:8]}..." for k in self.keys],
        }


def _print_search_results(result: dict[str, Any], num: int) -> None:
    if "error" in result:
        print(f"ERROR: {result['error']}", file=sys.stderr)
        return
    if "output" in result:
        print(json.dumps(result["output"], indent=2, ensure_ascii=False))
        return
    for i, item in enumerate(result.get("results", [])[:num], 1):
        title = item.get("title", "")
        url = item.get("url", "")
        score = item.get("score")
        score_str = f" (score={score:.3f})" if isinstance(score, (int, float)) else ""
        print(f"{i}. {title}{score_str}\n   {url}")
        for h in item.get("highlights", [])[:2]:
            print(f"     • {h.strip()[:200]}")
        if item.get("text"):
            print(f"     {item['text'][:300]}...")
        print()


def main() -> int:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    parser = argparse.ArgumentParser(description="Exa.ai search client with key rotation")
    sub = parser.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("search", help="Exa /search")
    s.add_argument("query")
    s.add_argument("--type", default="auto", choices=sorted(VALID_TYPES))
    s.add_argument("--num", type=int, default=10, dest="num_results")
    s.add_argument("--include", default=None, help="comma-separated includeDomains")
    s.add_argument("--exclude", default=None, help="comma-separated excludeDomains")
    s.add_argument("--category", default=None)
    s.add_argument("--max-age-hours", type=int, default=None, dest="max_age_hours")
    s.add_argument("--highlights", action="store_true", default=True)
    s.add_argument("--no-highlights", dest="highlights", action="store_false")
    s.add_argument("--text", action="store_true", default=False)
    s.add_argument("--max-chars", type=int, default=None, dest="text_max_chars")
    s.add_argument("--text-verbosity", choices=["compact", "full"], default=None)
    s.add_argument("--summary", default=None, help="true | <query bias string>")
    s.add_argument("--schema", default=None, help="path to JSON outputSchema file")
    s.add_argument("--json", action="store_true", dest="as_json")

    c = sub.add_parser("contents", help="Exa /contents (URLs you already have)")
    c.add_argument("urls", nargs="+")
    c.add_argument("--highlights", action="store_true", default=True)
    c.add_argument("--no-highlights", dest="highlights", action="store_false")
    c.add_argument("--text", action="store_true", default=False)
    c.add_argument("--max-chars", type=int, default=None, dest="text_max_chars")
    c.add_argument("--max-age-hours", type=int, default=None, dest="max_age_hours")
    c.add_argument("--summary", default=None)
    c.add_argument("--json", action="store_true", dest="as_json")

    sub.add_parser("keys-status", help="show key pool status")

    args = parser.parse_args()
    client = ExaClient()

    if args.cmd == "keys-status":
        print(json.dumps(client.status(), indent=2))
        return 0

    if args.cmd == "search":
        include = [d.strip() for d in args.include.split(",")] if args.include else None
        exclude = [d.strip() for d in args.exclude.split(",")] if args.exclude else None
        summary: bool | str | None
        if args.summary is None:
            summary = None
        elif args.summary.lower() == "true":
            summary = True
        else:
            summary = args.summary
        schema = json.loads(Path(args.schema).read_text()) if args.schema else None

        result = client.search(
            args.query,
            type=args.type,
            num_results=args.num_results,
            include_domains=include,
            exclude_domains=exclude,
            category=args.category,
            max_age_hours=args.max_age_hours,
            highlights=args.highlights,
            text=args.text,
            text_max_chars=args.text_max_chars,
            text_verbosity=args.text_verbosity,
            summary=summary,
            output_schema=schema,
        )
        if args.as_json:
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            _print_search_results(result, args.num_results)
        return 0 if "error" not in result else 1

    if args.cmd == "contents":
        summary: bool | str | None
        if args.summary is None:
            summary = None
        elif args.summary.lower() == "true":
            summary = True
        else:
            summary = args.summary
        result = client.contents(
            args.urls,
            highlights=args.highlights,
            text=args.text,
            text_max_chars=args.text_max_chars,
            max_age_hours=args.max_age_hours,
            summary=summary,
        )
        if args.as_json or not result.get("results"):
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            for i, item in enumerate(result.get("results", []), 1):
                print(f"{i}. {item.get('title', '')}\n   {item.get('url', '')}")
                for h in item.get("highlights", [])[:3]:
                    print(f"     • {h.strip()[:200]}")
                if item.get("text"):
                    print(f"     {item['text'][:500]}...")
                print()
        return 0 if "error" not in result else 1

    return 1


if __name__ == "__main__":
    sys.exit(main())
