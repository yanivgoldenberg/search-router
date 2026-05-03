"""Pydantic data models for the research pipeline."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, Optional

from pydantic import BaseModel, Field


class SubQuestion(BaseModel):
    text: str
    search_queries: list[str] = Field(default_factory=list)
    search_type: Literal["serp", "news", "ai", "deep", "academic", "code", "social"] = "serp"


class DecomposedPlan(BaseModel):
    question: str
    sub_questions: list[SubQuestion]
    rationale: str = ""


class SourceMeta(BaseModel):
    url: str
    title: str = ""
    snippet: str = ""
    body: Optional[str] = None
    fetched_at: Optional[str] = None
    provider: str = ""
    sub_question: str = ""
    word_count: int = 0
    credibility: Optional[str] = None


class ExtractedClaim(BaseModel):
    text: str
    exact_quote: str = ""
    source_url: str = ""
    confidence: float = 0.5
    sub_question: str = ""
    verified: Optional[bool] = None


class ExtractedSourceFacts(BaseModel):
    url: str
    summary: str = ""
    claims: list[ExtractedClaim] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    numbers: list[str] = Field(default_factory=list)


class ResearchSection(BaseModel):
    heading: str
    body_markdown: str
    citation_indices: list[int] = Field(default_factory=list)


class ResearchReport(BaseModel):
    question: str
    executive_summary: str
    sections: list[ResearchSection]
    claims: list[ExtractedClaim]
    sources: list[SourceMeta]
    contradictions: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    what_would_change_my_mind: list[str] = Field(default_factory=list)
    counter_arguments: list[str] = Field(default_factory=list)
    elapsed_seconds: float = 0.0
    sources_read: int = 0
    sources_searched: int = 0
    iterations: int = 1
    model_used: str = ""
    started_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ResearchRequest(BaseModel):
    q: str
    max_sources: int = 30
    max_sub_questions: int = 12
    iterations: int = 3
    mode: Literal["general", "competitive", "academic", "financial", "legal", "medical", "geo", "trading", "people", "product"] = "general"
    tier: Literal["free", "premium", "ultra"] = "free"
    session_id: Optional[str] = None
    save: bool = True
    include_full_bodies: bool = False
