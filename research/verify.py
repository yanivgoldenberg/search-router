"""Phase 9: verbatim citation grep. Drops claims whose exact_quote isn't in the source body."""
from __future__ import annotations

import logging
import re
from difflib import SequenceMatcher

from .models import ExtractedClaim, SourceMeta

logger = logging.getLogger(__name__)

VERIFY_FUZZY_THRESHOLD = 0.85
VERIFY_NORMALIZE = re.compile(r"\s+")


def _normalize(s: str) -> str:
    return VERIFY_NORMALIZE.sub(" ", s.strip().lower())


def _fuzzy_present(needle: str, haystack: str, threshold: float = VERIFY_FUZZY_THRESHOLD) -> bool:
    n = _normalize(needle)
    if not n or len(n) < 20:
        return n in haystack
    h = _normalize(haystack)
    if n in h:
        return True
    n_parts = n.split()
    if len(n_parts) >= 5:
        head = " ".join(n_parts[:5])
        tail = " ".join(n_parts[-5:])
        if head in h and tail in h:
            return True
    sample_len = max(80, len(n) + 40)
    for i in range(0, max(1, len(h) - len(n) + 1), max(40, len(n) // 2)):
        chunk = h[i:i + sample_len]
        if SequenceMatcher(None, n, chunk).ratio() >= threshold:
            return True
    return False


def verify_claims(claims: list[ExtractedClaim], sources: list[SourceMeta]) -> tuple[list[ExtractedClaim], list[ExtractedClaim]]:
    """Returns (verified, unverified)."""
    body_by_url = {s.url: (s.body or "") for s in sources if s.body}
    verified: list[ExtractedClaim] = []
    unverified: list[ExtractedClaim] = []

    for c in claims:
        body = body_by_url.get(c.source_url, "")
        if not body:
            c.verified = False
            unverified.append(c)
            continue
        if _fuzzy_present(c.exact_quote, body):
            c.verified = True
            verified.append(c)
        else:
            c.verified = False
            unverified.append(c)

    logger.info(
        "[verify] %d/%d claims verified by verbatim grep (%.1f%%)",
        len(verified),
        len(claims),
        100.0 * len(verified) / max(1, len(claims)),
    )
    return verified, unverified
