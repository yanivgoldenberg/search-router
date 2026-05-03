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
MAX_BODY_CHARS = int(os.environ.get("RESEARCH_MAX_BODY_CHARS", "12000"))
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




try:
    import fitz  # PyMuPDF
    HAVE_PDF = True
except ImportError:
    HAVE_PDF = False


def _pdf_fetch(url: str) -> str | None:
    """Download a PDF and extract full text via PyMuPDF."""
    if not HAVE_PDF:
        return None
    try:
        resp = requests.get(url, timeout=FETCH_TIMEOUT, headers={"User-Agent": "Mozilla/5.0 (compatible; YanivResearch/1.0)"}, stream=True)
        if resp.status_code != 200:
            return None
        ctype = resp.headers.get("content-type", "").lower()
        is_pdf = "pdf" in ctype or url.lower().endswith(".pdf")
        if not is_pdf:
            head = resp.raw.read(5)
            is_pdf = head[:4] == b"%PDF"
            if is_pdf:
                pdf_bytes = head + resp.raw.read()
            else:
                return None
        else:
            pdf_bytes = resp.content
        if len(pdf_bytes) > 50 * 1024 * 1024:  # 50MB cap
            return None
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        chunks = []
        for page in doc:
            chunks.append(page.get_text("text"))
            if len(" ".join(chunks)) > MAX_BODY_CHARS * 3:
                break
        doc.close()
        text = " ".join(chunks)
        text = " ".join(text.split())  # collapse whitespace
        return text[:MAX_BODY_CHARS]
    except Exception as e:
        logger.debug("pdf %s -> %s", url[:60], e)
        return None


def _fetch_one(src: SourceMeta) -> SourceMeta:
    if src.url.lower().endswith(".pdf"):
        body = _pdf_fetch(src.url) or _jina_fetch(src.url)
    else:
        body = _jina_fetch(src.url)
    if not body:
        body = _http_fetch(src.url)
    if not body and src.url.lower().endswith(".pdf"):
        body = _pdf_fetch(src.url)
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
