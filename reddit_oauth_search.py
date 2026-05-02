#!/usr/bin/env python3
"""Reddit Search via OAuth (the only path that works in 2026).

Setup (one-time):
  1. Go to https://www.reddit.com/prefs/apps
  2. Click "create another app" -> select "script" type
  3. name=anything, redirect uri=http://localhost (unused for script type)
  4. Note the client_id (under app name) and client_secret
  5. Add to /config/workspace/.env:
       REDDIT_CLIENT_ID=...
       REDDIT_CLIENT_SECRET=...
       REDDIT_USER_AGENT="web:yaniv-search:v1.0 (by /u/yourusername)"

OAuth flow uses the "client credentials" grant (read-only, no user login needed).

Usage:
    python3 scripts/reddit_oauth_search.py search "openai" --num 10
    python3 scripts/reddit_oauth_search.py subreddit programming --num 25
    python3 scripts/reddit_oauth_search.py status
"""
from __future__ import annotations

import argparse
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

OAUTH_URL = "https://www.reddit.com/api/v1/access_token"
API_BASE = "https://oauth.reddit.com"
DEFAULT_TIMEOUT = 15

_token_cache: dict[str, Any] = {"token": None, "expires_at": 0}


def _client_creds() -> tuple[str, str, str]:
    cid = os.environ.get("REDDIT_CLIENT_ID", "").strip()
    cs = os.environ.get("REDDIT_CLIENT_SECRET", "").strip()
    ua = os.environ.get("REDDIT_USER_AGENT", "").strip() or "web:yaniv-search:v1.0"
    if not cid or not cs:
        raise RuntimeError(
            "Reddit OAuth not configured. Set REDDIT_CLIENT_ID + REDDIT_CLIENT_SECRET in .env. "
            "Register a script-type app at https://www.reddit.com/prefs/apps"
        )
    return cid, cs, ua


def _get_token() -> str:
    now = time.time()
    if _token_cache["token"] and _token_cache["expires_at"] > now + 30:
        return _token_cache["token"]
    cid, cs, ua = _client_creds()
    resp = requests.post(
        OAUTH_URL,
        auth=(cid, cs),
        data={"grant_type": "client_credentials"},
        headers={"User-Agent": ua},
        timeout=DEFAULT_TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()
    _token_cache["token"] = data["access_token"]
    _token_cache["expires_at"] = now + data.get("expires_in", 3600)
    return _token_cache["token"]


def _api(path: str, params: dict[str, Any]) -> dict[str, Any]:
    try:
        token = _get_token()
        _, _, ua = _client_creds()
    except Exception as e:
        return {"error": str(e)}
    try:
        resp = requests.get(
            f"{API_BASE}{path}",
            params=params,
            headers={"Authorization": f"Bearer {token}", "User-Agent": ua},
            timeout=DEFAULT_TIMEOUT,
        )
    except requests.RequestException as e:
        return {"error": f"network: {e}"}
    if resp.status_code != 200:
        return {"error": f"http {resp.status_code}: {resp.text[:200]}"}
    return resp.json()


def search(query: str, num: int = 25, sort: str = "relevance", time_filter: str = "all") -> dict[str, Any]:
    return _api("/search", {"q": query, "limit": num, "sort": sort, "t": time_filter, "raw_json": 1, "type": "link"})


def subreddit(name: str, num: int = 25, sort: str = "hot") -> dict[str, Any]:
    return _api(f"/r/{name}/{sort}", {"limit": num, "raw_json": 1})


def search_subreddits(query: str, num: int = 10) -> dict[str, Any]:
    return _api("/subreddits/search", {"q": query, "limit": num, "raw_json": 1})


def status() -> dict[str, Any]:
    try:
        cid, _, ua = _client_creds()
        token = _get_token()
        return {"configured": True, "client_id_preview": cid[:6] + "...", "user_agent": ua, "token_acquired": bool(token)}
    except Exception as e:
        return {"configured": False, "error": str(e)}


def main() -> int:
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("search")
    s.add_argument("query")
    s.add_argument("--num", type=int, default=25)
    s.add_argument("--sort", default="relevance", choices=["relevance", "hot", "top", "new", "comments"])
    s.add_argument("--time", default="all", choices=["hour", "day", "week", "month", "year", "all"], dest="time_filter")
    s.add_argument("--json", action="store_true", dest="as_json")

    sr = sub.add_parser("subreddit")
    sr.add_argument("name")
    sr.add_argument("--num", type=int, default=25)
    sr.add_argument("--sort", default="hot", choices=["hot", "new", "top", "rising"])
    sr.add_argument("--json", action="store_true", dest="as_json")

    ss = sub.add_parser("subreddits")
    ss.add_argument("query")
    ss.add_argument("--num", type=int, default=10)
    ss.add_argument("--json", action="store_true", dest="as_json")

    sub.add_parser("status")

    args = p.parse_args()

    if args.cmd == "status":
        print(json.dumps(status(), indent=2))
        return 0

    if args.cmd == "search":
        out = search(args.query, num=args.num, sort=args.sort, time_filter=args.time_filter)
    elif args.cmd == "subreddit":
        out = subreddit(args.name, num=args.num, sort=args.sort)
    else:
        out = search_subreddits(args.query, num=args.num)

    if getattr(args, "as_json", False):
        print(json.dumps(out, indent=2, ensure_ascii=False))
        return 0 if "error" not in out else 1
    if "error" in out:
        print(f"ERROR: {out['error']}", file=sys.stderr)
        return 1
    for i, post in enumerate((out.get("data") or {}).get("children", [])[: args.num], 1):
        d = post.get("data", {})
        print(f"{i}. {d.get('title') or d.get('display_name', '')}")
        print(f"   r/{d.get('subreddit') or d.get('display_name', '')}  score={d.get('score', 0)}  comments={d.get('num_comments', 0)}")
        print(f"   https://reddit.com{d.get('permalink', '')}")
        if d.get("selftext"):
            print(f"   {d['selftext'][:300]}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
