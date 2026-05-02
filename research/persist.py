"""Phase 10: persist research session to Postgres + render markdown report."""
from __future__ import annotations

import json
import logging
import os
import uuid
from typing import Any

from .models import ResearchReport

logger = logging.getLogger(__name__)

try:
    import psycopg2
    import psycopg2.extras
except ImportError:
    psycopg2 = None


PG_HOST = os.environ.get("POSTGRES_HOST", "main-db")
PG_PORT = int(os.environ.get("POSTGRES_PORT", "5432"))
PG_USER = os.environ.get("POSTGRES_USER", "postgres")
PG_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "")
PG_DB = os.environ.get("RESEARCH_DB", os.environ.get("POSTGRES_DB", "projects"))


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS research_sessions (
    id           UUID PRIMARY KEY,
    question     TEXT NOT NULL,
    mode         TEXT NOT NULL DEFAULT 'general',
    tier         TEXT NOT NULL DEFAULT 'free',
    started_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    elapsed_s    REAL,
    sources_searched INTEGER,
    sources_read  INTEGER,
    iterations    INTEGER,
    model_used    TEXT,
    report_md     TEXT,
    report_json   JSONB
);
CREATE INDEX IF NOT EXISTS research_sessions_question_idx ON research_sessions USING gin (to_tsvector('english', question));
CREATE INDEX IF NOT EXISTS research_sessions_started_idx ON research_sessions (started_at DESC);

CREATE TABLE IF NOT EXISTS research_claims (
    id           BIGSERIAL PRIMARY KEY,
    session_id   UUID NOT NULL REFERENCES research_sessions(id) ON DELETE CASCADE,
    claim_text   TEXT NOT NULL,
    exact_quote  TEXT,
    source_url   TEXT,
    sub_question TEXT,
    confidence   REAL,
    verified     BOOLEAN
);
CREATE INDEX IF NOT EXISTS research_claims_session_idx ON research_claims (session_id);
CREATE INDEX IF NOT EXISTS research_claims_url_idx ON research_claims (source_url);
"""


def _connect() -> Any:
    if psycopg2 is None:
        raise RuntimeError("psycopg2 not installed")
    return psycopg2.connect(host=PG_HOST, port=PG_PORT, user=PG_USER, password=PG_PASSWORD, dbname=PG_DB)


def ensure_schema() -> bool:
    if psycopg2 is None:
        logger.warning("[persist] psycopg2 unavailable, skipping schema setup")
        return False
    try:
        with _connect() as conn, conn.cursor() as cur:
            cur.execute(SCHEMA_SQL)
            conn.commit()
        return True
    except Exception as e:
        logger.warning("[persist] schema setup failed: %s", e)
        return False


def save_session(report: ResearchReport, mode: str, tier: str) -> str | None:
    if psycopg2 is None:
        return None
    session_id = str(uuid.uuid4())
    try:
        with _connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO research_sessions
                  (id, question, mode, tier, completed_at, elapsed_s, sources_searched,
                   sources_read, iterations, model_used, report_md, report_json)
                VALUES (%s, %s, %s, %s, NOW(), %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    session_id,
                    report.question,
                    mode,
                    tier,
                    report.elapsed_seconds,
                    report.sources_searched,
                    report.sources_read,
                    report.iterations,
                    report.model_used,
                    render_markdown(report),
                    json.dumps(report.model_dump()),
                ),
            )
            for c in report.claims:
                cur.execute(
                    """
                    INSERT INTO research_claims
                      (session_id, claim_text, exact_quote, source_url, sub_question, confidence, verified)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """,
                    (session_id, c.text, c.exact_quote, c.source_url, c.sub_question, c.confidence, c.verified),
                )
            conn.commit()
        return session_id
    except Exception as e:
        logger.warning("[persist] save failed: %s", e)
        return None


def render_markdown(report: ResearchReport) -> str:
    lines: list[str] = []
    lines.append(f"# {report.question}\n")
    lines.append(
        f"_Researched in {report.elapsed_seconds:.1f}s · {report.sources_read}/{report.sources_searched} "
        f"sources · {len(report.claims)} verified claims · {report.iterations} pass(es) · model: {report.model_used}_\n"
    )
    lines.append("## Executive summary\n")
    lines.append(report.executive_summary + "\n")

    for sec in report.sections:
        lines.append(f"\n## {sec.heading}\n")
        lines.append(sec.body_markdown + "\n")

    if report.contradictions:
        lines.append("\n## Contradictions surfaced\n")
        for c in report.contradictions:
            lines.append(f"- {c}")
        lines.append("")

    if report.counter_arguments:
        lines.append("\n## Counter-arguments / skeptical view\n")
        for c in report.counter_arguments:
            lines.append(f"- {c}")
        lines.append("")

    if report.what_would_change_my_mind:
        lines.append("\n## What would change my mind\n")
        for w in report.what_would_change_my_mind:
            lines.append(f"- {w}")
        lines.append("")

    if report.gaps:
        lines.append("\n## Gaps (not enough evidence)\n")
        for g in report.gaps:
            lines.append(f"- {g}")
        lines.append("")

    lines.append("\n## Sources\n")
    url_to_idx: dict[str, int] = {}
    next_idx = 1
    for s in report.sources:
        if s.url not in url_to_idx:
            url_to_idx[s.url] = next_idx
            next_idx += 1
    for s in report.sources:
        idx = url_to_idx.get(s.url, 0)
        lines.append(f"[{idx}] {s.title or '(untitled)'} — {s.url}")

    unverified = [c for c in report.claims if c.verified is False]
    if unverified:
        lines.append("\n## Unverified claims (dropped from main report)\n")
        for c in unverified[:30]:
            lines.append(f"- {c.text} _(quote not found in {c.source_url})_")

    return "\n".join(lines)
