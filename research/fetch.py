"""Phase 3: parallel scrape via Jina Reader (free, unlimited). Falls back to plain HTTP."""
from __future__ import annotations

import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

from .models import SourceMeta

logger = logging.getLogger(__name__)

JINA_KEY = os.environ.get("JINA_API_KEY", "").strip()
JINA_URL = "https://r.jina.ai/"
FETCH_TIMEOUT = float(os.environ.get("RESEARCH_FETCH_TIMEOUT", "30"))
MAX_BODY_CHARS = int(os.environ.get("RESEARCH_MAX_BODY_CHARS", "60000"))
MAX_WORKERS = int(os.environ.get("RESEARCH_FETCH_WORKERS", "10"))


def _jina_fetch(url: str) -> str | None:
    headers = {"Accept": "text/markdown", "X-Retain-Images": "none", "X-No-Cache": "false"}
    if JINA_KEY:
        headers["Authorization"] = f"Bearer {JINA_KEY}"
    try:
        resp = requests.get(f"{JINA_URL}{url}", headers=headers, timeout=FETCH_TIMEOUT)
        if resp.status_code == 200 and resp.text and len(resp.text) > 200:
            return resp.text[:MAX_BODY_CHARS]
    except requests.RequestException as e:
        logger.debug("jina %s -> %s", url[:60], e)
    return None


def _http_fetch(url: str) -> str | None:
    try:
        resp = requests.get(
            url,
            timeout=FETCH_TIMEOUT,
            headers={"User-Agent": "Mozilla/5.0 (compatible; YanivResearch/1.0)"},
        )
        if resp.status_code == 200 and resp.text:
            text = re.sub(r"<script.*?</script>", " ", resp.text, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r"<style.*?</style>", " ", text, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r"<[^>]+>", " ", text)
            text = re.sub(r"\s+", " ", text)
            return text[:MAX_BODY_CHARS]
    except requests.RequestException as e:
        logger.debug("http %s -> %s", url[:60], e)
    return None


def _fetch_one(src: SourceMeta) -> SourceMeta:
    body = _jina_fetch(src.url)
    if not body:
        body = _http_fetch(src.url)
    if body:
        src.body = body
        src.word_count = len(body.split())
    return src


def fetch_all(sources: list[SourceMeta]) -> list[SourceMeta]:
    """Fetch full bodies in parallel. Drops sources that fail to fetch."""
    logger.info("[fetch] fetching %d sources", len(sources))
    out: list[SourceMeta] = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(_fetch_one, s): s for s in sources}
        for fut in as_completed(futs):
            try:
                src = fut.result()
                if src.body:
                    out.append(src)
            except Exception as e:
                logger.warning("fetch future failed: %s", e)
    logger.info("[fetch] %d/%d sources successfully fetched", len(out), len(sources))
    return out
