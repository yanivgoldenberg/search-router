"""Post research completion to Discord webhook."""
from __future__ import annotations

import logging
import os

import requests

logger = logging.getLogger(__name__)


def notify_completion(question: str, summary: str, job_id: str = "", elapsed_s: float = 0.0,
                      sources_read: int = 0, claims: int = 0, verified: int = 0) -> bool:
    webhook = os.environ.get("DISCORD_WEBHOOK_URL", "").strip()
    if not webhook:
        return False
    base = os.environ.get("AISEARCH_PUBLIC_URL", "https://aisearch.yanivgoldenberg.com")
    poll_url = f"{base}/v1/research/{job_id}/markdown" if job_id else ""
    msg = (
        f"**Research complete** ({elapsed_s:.0f}s, {sources_read} sources, {verified}/{claims} verified)\n"
        f"Q: {question[:200]}\n\n"
        f"{summary[:1200]}\n\n"
        + (f"<{poll_url}>" if poll_url else "")
    )
    try:
        r = requests.post(webhook, json={"content": msg[:1900]}, timeout=8)
        return r.status_code in (200, 204)
    except Exception as e:
        logger.debug("[discord] notify failed: %s", e)
        return False
