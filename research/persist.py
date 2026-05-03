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


def save_job_state(job_id: str, status: str, report=None, markdown="", error="") -> bool:
    if psycopg2 is None:
        return False
    try:
        with _connect() as conn, conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS research_jobs (
                    id          TEXT PRIMARY KEY,
                    status      TEXT NOT NULL,
                    started_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    report      JSONB,
                    markdown    TEXT,
                    error       TEXT
                )
            """)
            import json as _json
            cur.execute("""
                INSERT INTO research_jobs (id, status, report, markdown, error)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                  status = EXCLUDED.status,
                  report = EXCLUDED.report,
                  markdown = EXCLUDED.markdown,
                  error = EXCLUDED.error,
                  updated_at = NOW()
            """, (job_id, status, _json.dumps(report) if report else None, markdown, error))
            conn.commit()
        return True
    except Exception as e:
        logger.warning("[persist] save_job_state failed: %s", e)
        return False


def load_job_state(job_id: str) -> dict | None:
    if psycopg2 is None:
        return None
    try:
        with _connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT status, report, markdown, error FROM research_jobs WHERE id = %s", (job_id,))
            row = cur.fetchone()
            if not row:
                return None
            status, report, markdown, error = row
            out = {"status": status}
            if report:
                out["report"] = report
            if markdown:
                out["markdown"] = markdown
            if error:
                out["error"] = error
            return out
    except Exception as e:
        logger.warning("[persist] load_job_state failed: %s", e)
        return None


WATCHER_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS research_watches (
    id          UUID PRIMARY KEY,
    topic       TEXT NOT NULL,
    mode        TEXT NOT NULL DEFAULT 'general',
    schedule    TEXT NOT NULL,
    last_run_at TIMESTAMPTZ,
    last_session_id UUID,
    alert_url   TEXT,
    enabled     BOOLEAN NOT NULL DEFAULT true,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""


def ensure_watcher_schema() -> bool:
    if psycopg2 is None:
        return False
    try:
        with _connect() as conn, conn.cursor() as cur:
            cur.execute(WATCHER_SCHEMA_SQL)
            conn.commit()
        return True
    except Exception as e:
        logger.warning("[persist] watcher schema setup failed: %s", e)
        return False


def search_history(query: str, limit: int = 20) -> list[dict]:
    if psycopg2 is None:
        return []
    try:
        with _connect() as conn, conn.cursor() as cur:
            cur.execute("""
                SELECT id, question, mode, started_at, sources_read, model_used,
                       LEFT(report_md, 400) as preview
                FROM research_sessions
                WHERE to_tsvector('english', question || ' ' || COALESCE(report_md, ''))
                      @@ plainto_tsquery('english', %s)
                ORDER BY started_at DESC
                LIMIT %s
            """, (query, limit))
            return [
                {
                    "session_id": str(row[0]),
                    "question": row[1],
                    "mode": row[2],
                    "started_at": row[3].isoformat() if row[3] else None,
                    "sources_read": row[4],
                    "model_used": row[5],
                    "preview": row[6],
                }
                for row in cur.fetchall()
            ]
    except Exception as e:
        logger.warning("[persist] search_history failed: %s", e)
        return []


def get_session(session_id: str) -> dict | None:
    if psycopg2 is None:
        return None
    try:
        with _connect() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT id, question, mode, started_at, completed_at, elapsed_s, sources_read, model_used, report_md, report_json FROM research_sessions WHERE id = %s",
                (session_id,),
            )
            row = cur.fetchone()
            if not row:
                return None
            return {
                "session_id": str(row[0]),
                "question": row[1],
                "mode": row[2],
                "started_at": row[3].isoformat() if row[3] else None,
                "completed_at": row[4].isoformat() if row[4] else None,
                "elapsed_s": row[5],
                "sources_read": row[6],
                "model_used": row[7],
                "markdown": row[8],
                "report": row[9],
            }
    except Exception as e:
        logger.warning("[persist] get_session failed: %s", e)
        return None


def watch_create(topic: str, mode: str, schedule: str, alert_url: str = "") -> str | None:
    if psycopg2 is None:
        return None
    import uuid as _uuid
    wid = str(_uuid.uuid4())
    try:
        ensure_watcher_schema()
        with _connect() as conn, conn.cursor() as cur:
            cur.execute(
                "INSERT INTO research_watches (id, topic, mode, schedule, alert_url) VALUES (%s, %s, %s, %s, %s)",
                (wid, topic, mode, schedule, alert_url),
            )
            conn.commit()
        return wid
    except Exception as e:
        logger.exception("[persist] watch_create failed: %s", e)
        return None


def watch_list() -> list[dict]:
    if psycopg2 is None:
        return []
    try:
        ensure_watcher_schema()
        with _connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT id, topic, mode, schedule, last_run_at, alert_url, enabled FROM research_watches ORDER BY created_at DESC")
            return [
                {
                    "id": str(r[0]),
                    "topic": r[1],
                    "mode": r[2],
                    "schedule": r[3],
                    "last_run_at": r[4].isoformat() if r[4] else None,
                    "alert_url": r[5],
                    "enabled": r[6],
                }
                for r in cur.fetchall()
            ]
    except Exception as e:
        logger.warning("[persist] watch_list failed: %s", e)
        return []


def watch_delete(watch_id: str) -> bool:
    if psycopg2 is None:
        return False
    try:
        with _connect() as conn, conn.cursor() as cur:
            cur.execute("DELETE FROM research_watches WHERE id = %s", (watch_id,))
            conn.commit()
        return cur.rowcount > 0
    except Exception as e:
        logger.warning("[persist] watch_delete failed: %s", e)
        return False
