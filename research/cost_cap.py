"""Per-provider monthly cost cap. Free providers tracked at $0; paid logged for awareness."""
from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

try:
    import redis as _redis
except Exception:
    _redis = None

_RC = None


def _redis_client():
    global _RC
    if _RC is not None:
        return _RC
    if _redis is None:
        return None
    try:
        host = os.environ.get("REDIS_HOST", "redis-cache")
        port = int(os.environ.get("REDIS_PORT", "6379"))
        password = os.environ.get("REDIS_PASSWORD") or None
        c = _redis.Redis(host=host, port=port, password=password, socket_timeout=2, decode_responses=True)
        c.ping()
        _RC = c
        return c
    except Exception as e:
        logger.warning("[cap] redis unavailable: %s", e)
        return None


def _key(provider: str, kind: str = "calls") -> str:
    yyyymm = datetime.now(timezone.utc).strftime("%Y%m")
    return f"cap:{provider}:{yyyymm}:{kind}"


def record_call(provider: str, tokens_in: int = 0, tokens_out: int = 0, est_cost_cents: float = 0.0) -> None:
    c = _redis_client()
    if not c:
        return
    try:
        p = c.pipeline()
        p.incr(_key(provider, "calls"))
        p.incrby(_key(provider, "tokens_in"), tokens_in)
        p.incrby(_key(provider, "tokens_out"), tokens_out)
        p.incrbyfloat(_key(provider, "cost_cents"), est_cost_cents)
        p.expire(_key(provider, "calls"), 60 * 60 * 24 * 40)  # ~40 days
        p.execute()
    except Exception as e:
        logger.debug("[cap] record_call failed: %s", e)


def get_usage(provider: str | None = None) -> dict:
    c = _redis_client()
    if not c:
        return {}
    yyyymm = datetime.now(timezone.utc).strftime("%Y%m")
    pat = f"cap:{provider or '*'}:{yyyymm}:*"
    out: dict = {}
    try:
        for key in c.scan_iter(match=pat, count=100):
            parts = key.split(":")
            if len(parts) != 4:
                continue
            _, prov, _ym, kind = parts
            out.setdefault(prov, {})[kind] = c.get(key)
        for prov, vals in out.items():
            if "calls" in vals: vals["calls"] = int(vals["calls"])
            if "tokens_in" in vals: vals["tokens_in"] = int(vals["tokens_in"])
            if "tokens_out" in vals: vals["tokens_out"] = int(vals["tokens_out"])
            if "cost_cents" in vals: vals["cost_cents"] = float(vals["cost_cents"])
        return out
    except Exception as e:
        logger.debug("[cap] get_usage failed: %s", e)
        return {}


def cap_exceeded(provider: str, monthly_cap_cents: float) -> bool:
    if monthly_cap_cents <= 0:
        return False
    c = _redis_client()
    if not c:
        return False
    try:
        cents = c.get(_key(provider, "cost_cents"))
        return float(cents or 0) >= monthly_cap_cents
    except Exception:
        return False
