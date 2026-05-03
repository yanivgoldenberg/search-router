"""Multilingual forced-search: translate query into N languages, fan out, dedupe.
Then translate non-English bodies back to English before extract.

Adds ~30% relevant sources for non-English-dominant topics (Israeli market, EU policy, etc).
"""
from __future__ import annotations

import logging
from typing import Any

from .llm import LLMError, chat_json

logger = logging.getLogger(__name__)


DEFAULT_LANGS = [
    {"name": "English", "code": "en", "country": "us"},
    {"name": "Hebrew", "code": "he", "country": "il"},
    {"name": "Mandarin Chinese", "code": "zh", "country": "cn"},
    {"name": "German", "code": "de", "country": "de"},
    {"name": "Spanish", "code": "es", "country": "es"},
]


def translate_query(query: str, languages: list[dict] | None = None) -> dict[str, str]:
    """Translate query into target languages. Returns {lang_code: translated_query}."""
    langs = languages or DEFAULT_LANGS
    target_names = [l["name"] for l in langs if l["code"] != "en"]
    out = {"en": query}
    if not target_names:
        return out
    system = (
        "Output JSON only (no prose). Translate the user's query into each target language. "
        'Return shape: {"translations":{"<lang_name>":"<translated_query>"}}'
    )
    user = f"Query: {query}\n\nTarget languages: {', '.join(target_names)}"
    try:
        parsed: Any = chat_json(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=600, temperature=0.1,
        )
        translations = parsed.get("translations", {}) if isinstance(parsed, dict) else {}
        for l in langs:
            if l["code"] == "en":
                continue
            tx = translations.get(l["name"]) or translations.get(l["name"].lower())
            if tx:
                out[l["code"]] = str(tx).strip()
    except Exception as e:
        logger.warning("[multilingual] translate failed: %s", e)
    return out


def back_translate_to_english(text: str, source_lang: str = "auto") -> str:
    """Translate a non-English body back to English. Cap at 8000 chars."""
    text = text[:8000]
    if not text:
        return text
    system = "Output English only (no prose, no commentary). Translate the input to English, preserving structure."
    user = f"Source language: {source_lang}\n\n{text}"
    try:
        from .llm import chat
        return chat(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=4000, temperature=0.1,
        )
    except Exception as e:
        logger.debug("[multilingual] back-translate failed: %s", e)
        return text


def expand_subquestion_for_languages(sub_question_text: str, base_queries: list[str], lang_translations: dict[str, str]) -> dict[str, list[str]]:
    """Map {lang_code: [translated query variants]}. Caller fans out to aisearch with gl/hl per lang."""
    out = {}
    for code, _qtext in lang_translations.items():
        # If we have a translated query for this lang, use it; else fall back to base
        if code == "en":
            out[code] = base_queries
        else:
            # Best-effort: use the translated full sub-question text as a single query for that lang
            out[code] = [lang_translations[code]]
    return out
