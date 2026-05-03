"""Auto-export each completed research to SiYuan notebook 'Research'."""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

import requests

logger = logging.getLogger(__name__)


def export_to_siyuan(question: str, markdown: str, mode: str = "general") -> bool:
    base = os.environ.get("SIYUAN_API_URL", "").strip() or "https://notes.yanivgoldenberg.com"
    token = os.environ.get("SIYUAN_API_TOKEN", "").strip()
    if not token:
        return False
    notebook = os.environ.get("SIYUAN_RESEARCH_NOTEBOOK", "20240101000000-research")
    title = (question[:80] or "research").strip().replace("/", "-")
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d-%H%M%S")
    path = f"/Research/{ts}-{mode}-{title}"
    try:
        # Try canonical SiYuan endpoint
        r = requests.post(
            f"{base}/api/filetree/createDocWithMd",
            headers={"Authorization": f"Token {token}", "Content-Type": "application/json"},
            json={"notebook": notebook, "path": path, "markdown": markdown[:50000]},
            timeout=10,
        )
        if r.status_code == 200 and r.json().get("code") == 0:
            return True
        logger.debug("[siyuan] %s -> %s %s", path, r.status_code, r.text[:200])
    except Exception as e:
        logger.debug("[siyuan] export failed: %s", e)
    return False
