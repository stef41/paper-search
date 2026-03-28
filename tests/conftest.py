"""Shared test fixtures and helpers."""
from __future__ import annotations

import asyncio
import os
import time
from datetime import datetime, timezone
from typing import AsyncGenerator, Generator

import pytest
import httpx
from elasticsearch import AsyncElasticsearch
from fastapi.testclient import TestClient

# Set test env before importing app
os.environ.setdefault("ES_HOST", "elasticsearch")
os.environ.setdefault("ES_PORT", "9200")
os.environ.setdefault("ES_INDEX", "arxiv_papers_test")
os.environ.setdefault("REDIS_HOST", "redis")
os.environ.setdefault("REDIS_PORT", "6379")
os.environ.setdefault("API_KEYS", "test-key-1,test-key-2")
os.environ.setdefault("RATE_LIMIT_PER_MINUTE", "1000")
os.environ.setdefault("RATE_LIMIT_BURST", "100")
os.environ.setdefault("EMBEDDING_DIM", "384")
os.environ.setdefault("MAX_QUERY_LENGTH", "2000")
os.environ.setdefault("MAX_REGEX_LENGTH", "200")
os.environ.setdefault("REGEX_TIMEOUT_MS", "1000")
os.environ.setdefault("CORS_ORIGINS", "*")


from src.core.config import get_settings, reset_settings
from src.core.elasticsearch import get_es_client, close_es_client, ensure_index
from src.core.redis import get_redis_client, close_redis_client
from src.api.main import create_app


VALID_API_KEY = "test-key-1"
VALID_API_KEY_2 = "test-key-2"
INVALID_API_KEY = "invalid-key-xxx"

TEST_INDEX = "arxiv_papers_test"


# ── Sample Data ──

def make_paper(
    arxiv_id: str = "2401.00001",
    title: str = "Neural Networks for Natural Language Processing",
    abstract: str = "We present a novel approach to natural language processing using deep neural networks. Our method achieves state-of-the-art results on multiple benchmarks.",
    authors: list | None = None,
    categories: list | None = None,
    primary_category: str = "cs.CL",
    submitted_date: str = "2024-01-15T00:00:00+00:00",
    has_github: bool = False,
    github_urls: list | None = None,
    page_count: int | None = 12,
    doi: str | None = None,
    journal_ref: str | None = None,
    first_author: str | None = None,
    first_author_h_index: int | None = None,
    citation_stats: dict | None = None,
    references_stats: dict | None = None,
    title_embedding: list | None = None,
    abstract_embedding: list | None = None,
) -> dict:
    if authors is None:
        authors = [
            {"name": "John Smith", "is_first_author": True, "h_index": 45, "citation_count": 12000},
            {"name": "Jane Doe", "is_first_author": False, "h_index": 32, "citation_count": 8000},
        ]
    if categories is None:
        categories = ["cs.CL", "cs.AI"]
    if github_urls is None and has_github:
        github_urls = ["https://github.com/example/repo"]
    elif github_urls is None:
        github_urls = []
    if citation_stats is None:
        citation_stats = {
            "total_citations": 150,
            "avg_citation_age_years": 2.5,
            "median_h_index_citing_authors": 25.0,
            "top_citing_categories": ["cs.CL", "cs.AI"],
        }
    if references_stats is None:
        references_stats = {
            "total_references": 42,
            "avg_reference_age_years": 5.0,
            "top_referenced_categories": ["cs.CL", "cs.LG"],
        }

    dim = int(os.environ.get("EMBEDDING_DIM", "384"))
    if title_embedding is None:
        title_embedding = [0.01] * dim
    if abstract_embedding is None:
        abstract_embedding = [0.02] * dim

    return {
        "arxiv_id": arxiv_id,
        "title": title,
        "abstract": abstract,
        "authors": authors,
        "categories": categories,
        "primary_category": primary_category,
        "submitted_date": submitted_date,
        "updated_date": submitted_date,
        "published_date": None,
        "doi": doi,
        "journal_ref": journal_ref,
        "comments": f"{page_count} pages" if page_count else None,
        "page_count": page_count,
        "has_github": has_github,
        "github_urls": github_urls,
        "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}",
        "abstract_url": f"https://arxiv.org/abs/{arxiv_id}",
        "first_author": first_author or (authors[0]["name"] if authors else None),
        "first_author_h_index": first_author_h_index or (authors[0].get("h_index") if authors else None),
        "title_embedding": title_embedding,
        "abstract_embedding": abstract_embedding,
        "citation_stats": citation_stats,
        "references_stats": references_stats,
        "domains": list(set(c.split(".")[0] for c in categories)),
    }


SAMPLE_PAPERS = [
    make_paper(
        arxiv_id="2401.00001",
        title="Neural Networks for Natural Language Processing",
        abstract="We present a novel approach to natural language processing using deep neural networks. Our method achieves state-of-the-art results on multiple benchmarks including GLUE and SuperGLUE.",
        authors=[
            {"name": "John Smith", "is_first_author": True, "h_index": 45, "citation_count": 12000},
            {"name": "Jane Doe", "is_first_author": False, "h_index": 32, "citation_count": 8000},
        ],
        categories=["cs.CL", "cs.AI"],
        primary_category="cs.CL",
        submitted_date="2024-01-15T00:00:00+00:00",
        has_github=True,
        github_urls=["https://github.com/jsmith/nlp-nn"],
        page_count=12,
        doi="10.1234/example.2024.001",
        citation_stats={
            "total_citations": 150,
            "avg_citation_age_years": 2.5,
            "median_h_index_citing_authors": 25.0,
            "top_citing_categories": ["cs.CL", "cs.AI"],
        },
        references_stats={
            "total_references": 42,
            "avg_reference_age_years": 5.0,
            "top_referenced_categories": ["cs.CL", "cs.LG"],
        },
    ),
    make_paper(
        arxiv_id="2401.00002",
        title="Reinforcement Learning with Human Feedback",
        abstract="This paper explores reinforcement learning from human feedback (RLHF) for training large language models. We propose a new reward model architecture.",
        authors=[
            {"name": "Alice Johnson", "is_first_author": True, "h_index": 60, "citation_count": 25000},
            {"name": "Bob Williams", "is_first_author": False, "h_index": 55, "citation_count": 20000},
        ],
        categories=["cs.LG", "cs.AI"],
        primary_category="cs.LG",
        submitted_date="2024-02-20T00:00:00+00:00",
        has_github=True,
        github_urls=["https://github.com/ajohnson/rlhf"],
        page_count=20,
        journal_ref="NeurIPS 2024",
        citation_stats={
            "total_citations": 500,
            "avg_citation_age_years": 1.5,
            "median_h_index_citing_authors": 35.0,
            "top_citing_categories": ["cs.LG", "cs.CL"],
        },
        references_stats={
            "total_references": 65,
            "avg_reference_age_years": 3.0,
            "top_referenced_categories": ["cs.LG", "cs.AI"],
        },
    ),
    make_paper(
        arxiv_id="2401.00003",
        title="Quantum Computing Algorithms for Optimization",
        abstract="We survey quantum computing algorithms for combinatorial optimization problems. Our analysis covers QAOA, VQE, and quantum annealing approaches.",
        authors=[
            {"name": "Carol Davis", "is_first_author": True, "h_index": 30, "citation_count": 5000},
        ],
        categories=["quant-ph", "cs.DS"],
        primary_category="quant-ph",
        submitted_date="2024-03-10T00:00:00+00:00",
        has_github=False,
        page_count=35,
        citation_stats={
            "total_citations": 20,
            "avg_citation_age_years": 0.5,
            "median_h_index_citing_authors": 15.0,
            "top_citing_categories": ["quant-ph"],
        },
        references_stats={
            "total_references": 80,
            "avg_reference_age_years": 8.0,
            "top_referenced_categories": ["quant-ph", "cs.DS"],
        },
    ),
    make_paper(
        arxiv_id="2401.00004",
        title="Generative Adversarial Networks for Image Synthesis",
        abstract="A comprehensive study of generative adversarial networks applied to high-resolution image synthesis. We introduce StyleGAN-X.",
        authors=[
            {"name": "David Lee", "is_first_author": True, "h_index": 15, "citation_count": 2000},
            {"name": "Eva Martinez", "is_first_author": False, "h_index": 10, "citation_count": 1000},
            {"name": "Frank Brown", "is_first_author": False, "h_index": 8, "citation_count": 500},
        ],
        categories=["cs.CV", "cs.AI"],
        primary_category="cs.CV",
        submitted_date="2023-06-15T00:00:00+00:00",
        has_github=True,
        github_urls=["https://github.com/dlee/stylegan-x"],
        page_count=8,
        citation_stats={
            "total_citations": 5,
            "avg_citation_age_years": 0.2,
            "median_h_index_citing_authors": 10.0,
            "top_citing_categories": ["cs.CV"],
        },
        references_stats={
            "total_references": 30,
            "avg_reference_age_years": 4.0,
            "top_referenced_categories": ["cs.CV", "cs.LG"],
        },
    ),
    make_paper(
        arxiv_id="2401.00005",
        title="Transformer Architecture Improvements for Long Sequences",
        abstract="We propose modifications to the standard transformer architecture that enable efficient processing of sequences up to 1 million tokens long.",
        authors=[
            {"name": "Grace Kim", "is_first_author": True, "h_index": 72, "citation_count": 50000},
            {"name": "Henry Zhang", "is_first_author": False, "h_index": 65, "citation_count": 40000},
        ],
        categories=["cs.LG", "cs.CL", "cs.AI"],
        primary_category="cs.LG",
        submitted_date="2024-04-01T00:00:00+00:00",
        has_github=True,
        github_urls=["https://github.com/gkim/long-transformer"],
        page_count=25,
        doi="10.1234/example.2024.005",
        journal_ref="ICML 2024",
        citation_stats={
            "total_citations": 1200,
            "avg_citation_age_years": 0.8,
            "median_h_index_citing_authors": 40.0,
            "top_citing_categories": ["cs.LG", "cs.CL"],
        },
        references_stats={
            "total_references": 55,
            "avg_reference_age_years": 3.5,
            "top_referenced_categories": ["cs.LG", "cs.CL"],
        },
    ),
    make_paper(
        arxiv_id="2301.99999",
        title="A Survey of Computer Vision Techniques",
        abstract="This survey covers recent advances in computer vision including object detection, segmentation, and pose estimation methods.",
        authors=[
            {"name": "Irene Park", "is_first_author": True, "h_index": 5, "citation_count": 200},
        ],
        categories=["cs.CV"],
        primary_category="cs.CV",
        submitted_date="2023-01-01T00:00:00+00:00",
        has_github=False,
        page_count=50,
        citation_stats={
            "total_citations": 0,
            "avg_citation_age_years": None,
            "median_h_index_citing_authors": None,
            "top_citing_categories": [],
        },
        references_stats={
            "total_references": 120,
            "avg_reference_age_years": 6.0,
            "top_referenced_categories": ["cs.CV", "cs.LG"],
        },
    ),
]


# ── Fixtures ──

@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
async def es_client() -> AsyncGenerator[AsyncElasticsearch, None]:
    settings = get_settings()
    client = AsyncElasticsearch(
        hosts=[settings.es_url],
        request_timeout=30,
    )

    # Wait for ES to be ready
    for _ in range(60):
        try:
            health = await client.cluster.health(wait_for_status="yellow", timeout="5s")
            if health["status"] in ("green", "yellow"):
                break
        except Exception:
            await asyncio.sleep(1)

    yield client
    await client.close()


@pytest.fixture(scope="session")
async def indexed_papers(es_client: AsyncElasticsearch) -> list[dict]:
    settings = get_settings()
    index = settings.es_index

    # Delete if exists
    if await es_client.indices.exists(index=index):
        await es_client.indices.delete(index=index)

    # Create index
    await ensure_index(es_client, index, settings.embedding_dim)

    # Index all sample papers
    for paper in SAMPLE_PAPERS:
        await es_client.index(index=index, id=paper["arxiv_id"], document=paper)

    # Force refresh for test visibility
    await es_client.indices.refresh(index=index)

    # Wait to ensure all docs are searchable
    for _ in range(30):
        count = await es_client.count(index=index)
        if count["count"] >= len(SAMPLE_PAPERS):
            break
        await asyncio.sleep(0.5)

    return SAMPLE_PAPERS


@pytest.fixture(scope="session")
def app(indexed_papers):
    reset_settings()
    application = create_app()
    return application


@pytest.fixture(scope="session")
def client(app) -> Generator:
    with TestClient(app) as c:
        yield c


def auth_headers(api_key: str = VALID_API_KEY) -> dict:
    return {"X-API-Key": api_key}
