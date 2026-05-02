"""LLM client. Default Groq (free, fast, Llama 3.3 70B). Anthropic fallback.

Used by every research phase: decompose, extract, synth, verify-claims, judge.
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

import requests

logger = logging.getLogger(__name__)

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
GROQ_DEFAULT_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_LARGE_CTX_MODEL = os.environ.get("GROQ_LARGE_MODEL", "llama-3.3-70b-versatile")
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
DEFAULT_TIMEOUT = 90


class LLMError(Exception):
    pass


def chat(
    messages: list[dict[str, str]],
    *,
    model: str | None = None,
    max_tokens: int = 4000,
    temperature: float = 0.2,
    json_mode: bool = False,
    backend: str = "auto",
) -> str:
    """Send chat messages, return assistant text. Raises LLMError on hard failure."""
    last_err: str | None = None
    if backend in ("auto", "groq"):
        try:
            return _groq(messages, model or GROQ_DEFAULT_MODEL, max_tokens, temperature, json_mode)
        except LLMError as e:
            last_err = f"groq: {e}"
            logger.warning("Groq failed (%s), falling back to Anthropic", e)
    if backend in ("auto", "anthropic"):
        try:
            return _anthropic(messages, ANTHROPIC_MODEL, max_tokens, temperature, json_mode)
        except LLMError as e:
            last_err = f"anthropic: {e}"
    raise LLMError(last_err or "no backend available")


def chat_json(
    messages: list[dict[str, str]],
    *,
    model: str | None = None,
    max_tokens: int = 4000,
    temperature: float = 0.1,
    backend: str = "auto",
) -> dict[str, Any] | list[Any]:
    """Chat with JSON output enforcement. Returns parsed dict/list."""
    raw = chat(messages, model=model, max_tokens=max_tokens, temperature=temperature, json_mode=True, backend=backend)
    return _parse_json(raw)


def _parse_json(raw: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    s = raw.strip()
    if s.startswith("```"):
        s = s.split("```", 2)[1]
        if s.startswith("json"):
            s = s[4:]
        s = s.split("```", 1)[0]
    try:
        return json.loads(s.strip())
    except json.JSONDecodeError:
        for c in ("[", "{"):
            i = raw.find(c)
            if i >= 0:
                end = raw.rfind("]" if c == "[" else "}")
                if end > i:
                    try:
                        return json.loads(raw[i:end + 1])
                    except json.JSONDecodeError:
                        continue
    raise LLMError(f"could not parse JSON: {raw[:300]}")


def _groq(messages: list[dict[str, str]], model: str, max_tokens: int, temperature: float, json_mode: bool) -> str:
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        raise LLMError("GROQ_API_KEY not set")
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    last_err: str = ""
    for attempt in range(4):
        try:
            resp = requests.post(
                GROQ_URL,
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json=payload,
                timeout=DEFAULT_TIMEOUT,
            )
        except requests.RequestException as e:
            last_err = f"network: {e}"
            time.sleep(2 + attempt * 2)
            continue
        if resp.status_code == 200:
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        if resp.status_code == 429:
            retry_after = resp.headers.get("retry-after", "")
            try:
                wait = float(retry_after) if retry_after else 6 + attempt * 4
            except ValueError:
                wait = 6 + attempt * 4
            wait = min(wait, 60)
            logger.info("groq 429, sleeping %.1fs (attempt %d)", wait, attempt + 1)
            time.sleep(wait)
            last_err = "rate limited"
            continue
        if resp.status_code in (500, 502, 503, 504):
            time.sleep(2 + attempt * 2)
            last_err = f"http {resp.status_code}"
            continue
        raise LLMError(f"groq http {resp.status_code}: {resp.text[:300]}")
    raise LLMError(f"groq exhausted retries: {last_err}")


def _anthropic(messages: list[dict[str, str]], model: str, max_tokens: int, temperature: float, json_mode: bool) -> str:
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
    data = resp.json()
    return data["content"][0]["text"]
