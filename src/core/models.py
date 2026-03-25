from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


# ── ArXiv Paper Model ──

class Author(BaseModel):
    name: str
    is_first_author: bool = False
    h_index: int | None = None
    citation_count: int | None = None


class CitationStats(BaseModel):
    total_citations: int = 0
    avg_citation_age_years: float | None = None
    median_h_index_citing_authors: float | None = None
    top_citing_categories: list[str] = Field(default_factory=list)


class ReferencesStats(BaseModel):
    total_references: int = 0
    avg_reference_age_years: float | None = None
    top_referenced_categories: list[str] = Field(default_factory=list)


class Paper(BaseModel):
    arxiv_id: str
    title: str
    abstract: str
    authors: list[Author] = Field(default_factory=list)
    categories: list[str] = Field(default_factory=list)
    primary_category: str | None = None
    submitted_date: datetime | None = None
    updated_date: datetime | None = None
    published_date: datetime | None = None
    doi: str | None = None
    journal_ref: str | None = None
    comments: str | None = None
    page_count: int | None = None
    has_github: bool = False
    github_urls: list[str] = Field(default_factory=list)
    pdf_url: str | None = None
    abstract_url: str | None = None

    # Embeddings (stored but not returned by default)
    title_embedding: list[float] | None = None
    abstract_embedding: list[float] | None = None
    paragraph_embeddings: list[list[float]] | None = None

    # Citation data
    citation_stats: CitationStats = Field(default_factory=CitationStats)
    references_stats: ReferencesStats = Field(default_factory=ReferencesStats)

    # Computed
    first_author: str | None = None
    first_author_h_index: int | None = None


# ── Search Models ──

class SimilarityLevel(str, Enum):
    TITLE = "title"
    ABSTRACT = "abstract"
    PARAGRAPH = "paragraph"


class SortField(str, Enum):
    RELEVANCE = "relevance"
    DATE = "date"
    CITATIONS = "citations"
    H_INDEX = "h_index"
    PAGE_COUNT = "page_count"
    UPDATED = "updated"


class SortOrder(str, Enum):
    ASC = "asc"
    DESC = "desc"


class DateRange(BaseModel):
    gte: datetime | None = None
    lte: datetime | None = None


class SemanticQuery(BaseModel):
    text: str = Field(..., max_length=2000)
    level: SimilarityLevel = SimilarityLevel.ABSTRACT
    weight: float = Field(default=1.0, ge=0.0, le=10.0)


class SearchRequest(BaseModel):
    # Text search
    query: str | None = Field(default=None, max_length=2000)
    title_query: str | None = Field(default=None, max_length=500)
    abstract_query: str | None = Field(default=None, max_length=2000)

    # Semantic similarity
    semantic: SemanticQuery | None = None

    # Fuzzy matching
    fuzzy: str | None = Field(default=None, max_length=500)
    fuzzy_fuzziness: int = Field(default=2, ge=0, le=3)

    # Regex search
    title_regex: str | None = Field(default=None, max_length=200)
    abstract_regex: str | None = Field(default=None, max_length=200)
    author_regex: str | None = Field(default=None, max_length=200)

    # Author filters
    author: str | None = None
    first_author: str | None = None
    min_h_index: int | None = Field(default=None, ge=0)
    max_h_index: int | None = Field(default=None, ge=0)
    min_first_author_h_index: int | None = Field(default=None, ge=0)
    min_median_h_index_citing: float | None = Field(default=None, ge=0)

    # Citation filters
    min_citations: int | None = Field(default=None, ge=0)
    max_citations: int | None = Field(default=None, ge=0)
    min_references: int | None = Field(default=None, ge=0)

    # Category filters
    categories: list[str] | None = None
    primary_category: str | None = None
    exclude_categories: list[str] | None = None

    # Date filters
    submitted_date: DateRange | None = None
    updated_date: DateRange | None = None

    # Paper metadata filters
    has_github: bool | None = None
    min_page_count: int | None = Field(default=None, ge=0)
    max_page_count: int | None = Field(default=None, ge=0)
    has_doi: bool | None = None
    has_journal_ref: bool | None = None

    # Boolean / match control
    minimum_should_match: str | None = Field(
        default=None,
        description="ES minimum_should_match spec, e.g. '2', '75%'"
    )
    operator: str = Field(default="or", pattern="^(and|or)$")

    # Sorting
    sort_by: SortField = SortField.RELEVANCE
    sort_order: SortOrder = SortOrder.DESC

    # Pagination
    offset: int = Field(default=0, ge=0, le=50000)
    limit: int = Field(default=20, ge=1, le=200)

    # Response control
    include_embeddings: bool = False
    highlight: bool = True

    @field_validator("title_regex", "abstract_regex", "author_regex")
    @classmethod
    def validate_regex_safety(cls, v: str | None) -> str | None:
        if v is None:
            return v
        import re
        # Block dangerous patterns that could cause ReDoS
        dangerous = [
            r"(.+)+",
            r"(.*)*",
            r"([a-zA-Z]+)*",
            r"(a|a)+",
        ]
        for pat in dangerous:
            if pat in v:
                raise ValueError(f"Potentially dangerous regex pattern blocked: {pat}")
        # Validate syntax
        try:
            re.compile(v)
        except re.error as exc:
            raise ValueError(f"Invalid regex: {exc}") from exc
        return v


class SearchHit(BaseModel):
    arxiv_id: str
    title: str
    abstract: str
    authors: list[Author] = Field(default_factory=list)
    categories: list[str] = Field(default_factory=list)
    primary_category: str | None = None
    submitted_date: datetime | None = None
    updated_date: datetime | None = None
    published_date: datetime | None = None
    doi: str | None = None
    journal_ref: str | None = None
    page_count: int | None = None
    has_github: bool = False
    github_urls: list[str] = Field(default_factory=list)
    first_author: str | None = None
    first_author_h_index: int | None = None
    citation_stats: CitationStats = Field(default_factory=CitationStats)
    references_stats: ReferencesStats = Field(default_factory=ReferencesStats)
    score: float | None = None
    highlights: dict[str, list[str]] | None = None


class SearchResponse(BaseModel):
    total: int
    hits: list[SearchHit]
    took_ms: int
    offset: int
    limit: int


class StatsResponse(BaseModel):
    total_papers: int
    categories: dict[str, int]
    date_range: dict[str, str | None]
    papers_with_github: int
    avg_page_count: float | None
    avg_citations: float | None


class HealthResponse(BaseModel):
    status: str
    elasticsearch: str
    redis: str
    total_papers: int
