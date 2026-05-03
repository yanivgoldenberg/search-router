"""LLM client.

Multi-provider round-robin with adaptive throttling.
Default cascade: Groq -> Cerebras -> SambaNova -> Together -> OpenRouter -> HuggingFace.

Each provider has its own rate-limit bucket, so round-robin multiplies effective
tokens-per-minute. Adaptive throttling reads rate-limit headers and waits just
long enough to avoid 429s.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any

import requests

logger = logging.getLogger(__name__)

WATERFALL_URL = os.environ.get("LLM_WATERFALL_URL", "").rstrip("/")
WATERFALL_MODEL = os.environ.get("LLM_WATERFALL_MODEL", "waterfall")
WATERFALL_KEY = os.environ.get("LLM_WATERFALL_KEY", "").strip()
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
CEREBRAS_URL = "https://api.cerebras.ai/v1/chat/completions"
SAMBANOVA_URL = "https://api.sambanova.ai/v1/chat/completions"
TOGETHER_URL = "https://api.together.xyz/v1/chat/completions"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

GROQ_DEFAULT_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
DEFAULT_TIMEOUT = 90

# Per-provider rate limit state (token bucket tracking from response headers)
_BUCKET_LOCK = threading.Lock()
_BUCKETS: dict[str, dict[str, float]] = {}


class LLMError(Exception):
    pass


def _bucket_check(provider: str, est_tokens: int) -> float:
    """Return wait_seconds before this call is safe. 0 if no wait needed."""
    with _BUCKET_LOCK:
        b = _BUCKETS.get(provider)
        if not b:
            return 0.0
        remaining = b.get("remaining_tokens", 100000.0)
        reset_at = b.get("reset_at", 0.0)
        now = time.time()
        if remaining < est_tokens and reset_at > now:
            wait = reset_at - now
            return min(wait, 60.0)
    return 0.0


def _bucket_update(provider: str, headers: dict[str, str]) -> None:
    """Update token-bucket state from response headers."""
    try:
        remaining = headers.get("x-ratelimit-remaining-tokens")
        reset_str = headers.get("x-ratelimit-reset-tokens", "")
        if remaining is None:
            return
        reset_seconds = 60.0
        if reset_str.endswith("s"):
            try:
                reset_seconds = float(reset_str.rstrip("ms").rstrip("s"))
            except ValueError:
                pass
        with _BUCKET_LOCK:
            _BUCKETS[provider] = {
                "remaining_tokens": float(remaining),
                "reset_at": time.time() + reset_seconds,
            }
    except Exception as e:
        logger.debug("bucket_update %s failed: %s", provider, e)


def _have_key(env_var: str) -> bool:
    return bool(os.environ.get(env_var, "").strip())


def chat(
    messages: list[dict[str, str]],
    *,
    model: str | None = None,
    max_tokens: int = 4000,
    temperature: float = 0.2,
    json_mode: bool = False,
    backend: str = "auto",
) -> str:
    """Send chat messages, return assistant text. Round-robin across providers."""
    last_err: str | None = None
    est_tokens = sum(len(m.get("content", "")) for m in messages) // 4 + max_tokens

    candidates: list[tuple[str, callable]] = []
    if backend in ("auto", "waterfall") and WATERFALL_URL:
        candidates.append(("waterfall", lambda: _waterfall(messages, model or WATERFALL_MODEL, max_tokens, temperature, json_mode)))
        if backend == "auto" and os.environ.get("LLM_WATERFALL_EXCLUSIVE", "1") == "1":
            return _run_candidates(candidates, est_tokens)
    if backend in ("auto", "groq") and _have_key("GROQ_API_KEY"):
        candidates.append(("groq", lambda: _groq(messages, model or GROQ_DEFAULT_MODEL, max_tokens, temperature, json_mode)))
    if backend in ("auto", "cerebras") and _have_key("CEREBRAS_API_KEY"):
        candidates.append(("cerebras", lambda: _cerebras(messages, max_tokens, temperature, json_mode)))
    if backend in ("auto", "sambanova") and _have_key("SAMBANOVA_API_KEY"):
        candidates.append(("sambanova", lambda: _sambanova(messages, max_tokens, temperature, json_mode)))
    if backend in ("auto", "together") and _have_key("TOGETHER_API_KEY"):
        candidates.append(("together", lambda: _together(messages, max_tokens, temperature, json_mode)))
    if backend in ("auto", "openrouter") and _have_key("OPENROUTER_API_KEY"):
        candidates.append(("openrouter", lambda: _openrouter(messages, max_tokens, temperature, json_mode, model_override=model)))
    if backend in ("auto", "anthropic") and _have_key("ANTHROPIC_API_KEY"):
        candidates.append(("anthropic", lambda: _anthropic(messages, ANTHROPIC_MODEL, max_tokens, temperature, json_mode)))

    return _run_candidates(candidates, est_tokens)


def _run_candidates(candidates, est_tokens: int) -> str:
    if not candidates:
        raise LLMError("no LLM provider configured")
    last_err: str | None = None

    def _load(name_fn):
        name = name_fn[0]
        with _BUCKET_LOCK:
            b = _BUCKETS.get(name, {})
            return -float(b.get("remaining_tokens", 1e9))

    sorted_candidates = sorted(candidates, key=_load)

    for name, fn in sorted_candidates:
        wait = _bucket_check(name, est_tokens)
        if wait > 30:
            logger.info("provider %s would wait %.1fs (est %d tokens), trying next", name, wait, est_tokens)
            continue
        if wait > 0:
            logger.info("provider %s pacing %.1fs (est %d tokens)", name, wait, est_tokens)
            time.sleep(wait)
        try:
            return fn()
        except LLMError as e:
            last_err = f"{name}: {e}"
            logger.warning("provider %s failed (%s), trying next", name, e)
            continue

    raise LLMError(last_err or "all providers failed")


def chat_json(
    messages: list[dict[str, str]],
    *,
    model: str | None = None,
    max_tokens: int = 4000,
    temperature: float = 0.1,
    backend: str = "auto",
) -> dict[str, Any] | list[Any]:
    raw = chat(messages, model=model, max_tokens=max_tokens, temperature=temperature, json_mode=True, backend=backend)
    return _parse_json(raw)


def _parse_json(raw: str) -> Any:
    """Robust JSON parse — handles fences, prose, common Llama quirks (key=val instead of key:val)."""
    import re as _re
    candidates = []
    candidates.append(raw)
    s = raw.strip()
    if s.startswith("```"):
        s = s.split("```", 2)[1]
        if s.startswith("json"):
            s = s[4:]
        s = s.split("```", 1)[0]
    candidates.append(s.strip())
    # Extract widest balanced JSON block
    for c in ("{", "["):
        i = raw.find(c)
        if i >= 0:
            end = raw.rfind("]" if c == "[" else "}")
            if end > i:
                candidates.append(raw[i:end + 1])
    # Try Llama's `"key=[` typo fix: replace `"keyName=[` with `"keyName":[`
    for cand in list(candidates):
        fixed = _re.sub(r'"(\w+)\s*=\s*\[', r'"\1":[', cand)
        fixed = _re.sub(r'"(\w+)\s*=\s*"', r'"\1":"', fixed)
        if fixed != cand:
            candidates.append(fixed)
    for cand in candidates:
        try:
            return json.loads(cand)
        except (json.JSONDecodeError, ValueError):
            continue
    raise LLMError(f"could not parse JSON: {raw[:300]}")


def _openai_compat_call(provider: str, url: str, key: str, model: str, messages, max_tokens, temperature, json_mode) -> str:
    payload: dict[str, Any] = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature}
    # NOTE: we deliberately do NOT use response_format=json_object — Groq's strict
    # validator rejects valid-but-not-bare JSON. Our _parse_json strips ``` fences.
    last_err = ""
    for attempt in range(3):
        try:
            resp = requests.post(url, headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"}, json=payload, timeout=DEFAULT_TIMEOUT)
        except requests.RequestException as e:
            last_err = f"network: {e}"
            time.sleep(2 + attempt * 2)
            continue
        _bucket_update(provider, dict(resp.headers))
        if resp.status_code == 200:
            data = resp.json()
            usage = data.get("usage", {}) or {}
            try:
                from . import cost_cap as _cc
                _cc.record_call(provider, int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0)), 0.0)
            except Exception:
                pass
            return data["choices"][0]["message"]["content"]
        if resp.status_code == 429:
            retry_after = resp.headers.get("retry-after", "")
            try:
                wait = float(retry_after) if retry_after else 5 + attempt * 5
            except ValueError:
                wait = 5 + attempt * 5
            wait = min(wait, 30)
            logger.info("%s 429, sleeping %.1fs (attempt %d)", provider, wait, attempt + 1)
            time.sleep(wait)
            last_err = "rate limited"
            continue
        if resp.status_code in (500, 502, 503, 504):
            time.sleep(2 + attempt * 2)
            last_err = f"http {resp.status_code}"
            continue
        raise LLMError(f"{provider} http {resp.status_code}: {resp.text[:300]}")
    raise LLMError(f"{provider} exhausted: {last_err}")


def _groq(messages, model, max_tokens, temperature, json_mode) -> str:
    return _openai_compat_call("groq", GROQ_URL, os.environ["GROQ_API_KEY"], model, messages, max_tokens, temperature, json_mode)


def _waterfall(messages, model, max_tokens, temperature, json_mode) -> str:
    url = f"{WATERFALL_URL}/v1/chat/completions"
    return _openai_compat_call("waterfall", url, WATERFALL_KEY or "none", model, messages, max_tokens, temperature, json_mode)


def _cerebras(messages, max_tokens, temperature, json_mode) -> str:
    model = os.environ.get("CEREBRAS_MODEL", "llama-3.3-70b")
    return _openai_compat_call("cerebras", CEREBRAS_URL, os.environ["CEREBRAS_API_KEY"], model, messages, max_tokens, temperature, json_mode)


def _sambanova(messages, max_tokens, temperature, json_mode) -> str:
    model = os.environ.get("SAMBANOVA_MODEL", "Meta-Llama-3.3-70B-Instruct")
    return _openai_compat_call("sambanova", SAMBANOVA_URL, os.environ["SAMBANOVA_API_KEY"], model, messages, max_tokens, temperature, json_mode)


def _together(messages, max_tokens, temperature, json_mode) -> str:
    model = os.environ.get("TOGETHER_MODEL", "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free")
    return _openai_compat_call("together", TOGETHER_URL, os.environ["TOGETHER_API_KEY"], model, messages, max_tokens, temperature, json_mode)


def _openrouter(messages, max_tokens, temperature, json_mode, model_override: str | None = None) -> str:
    model = model_override or os.environ.get("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct:free")
    headers_extra = {"HTTP-Referer": "https://aisearch.yanivgoldenberg.com", "X-Title": "aisearch-research"}
    payload: dict[str, Any] = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature}
    # NOTE: we deliberately do NOT use response_format=json_object — Groq's strict
    # validator rejects valid-but-not-bare JSON. Our _parse_json strips ``` fences.
    key = os.environ["OPENROUTER_API_KEY"]
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json", **headers_extra}
    last_err = ""
    for attempt in range(3):
        try:
            resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=DEFAULT_TIMEOUT)
        except requests.RequestException as e:
            last_err = f"network: {e}"
            time.sleep(2)
            continue
        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"]
        if resp.status_code == 429:
            time.sleep(5 + attempt * 5)
            last_err = "rate limited"
            continue
        raise LLMError(f"openrouter http {resp.status_code}: {resp.text[:300]}")
    raise LLMError(f"openrouter exhausted: {last_err}")


def _anthropic(messages, model, max_tokens, temperature, json_mode) -> str:
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not key:
        raise LLMError("ANTHROPIC_API_KEY not set")
    system = ""
    msgs = []
    for m in messages:
        if m["role"] == "system":
            system = (system + "\n" + m["content"]).strip() if system else m["content"]
        else:
            msgs.append(m)
    if json_mode:
        system = (system + "\n\nRespond ONLY with valid JSON, no prose.").strip()
    try:
        resp = requests.post(
            ANTHROPIC_URL,
            headers={"x-api-key": key, "anthropic-version": "2023-06-01", "Content-Type": "application/json"},
            json={"model": model, "max_tokens": max_tokens, "temperature": temperature, "system": system, "messages": msgs},
            timeout=DEFAULT_TIMEOUT,
        )
    except requests.RequestException as e:
        raise LLMError(f"anthropic network: {e}") from e
    if resp.status_code != 200:
        raise LLMError(f"anthropic http {resp.status_code}: {resp.text[:300]}")
    return resp.json()["content"][0]["text"]
