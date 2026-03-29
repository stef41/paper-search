"""
ArXiv API seeder — fast initial data loading via the ArXiv search API.

Usage:
    python -m src.ingestion.seed                       # seed all major CS categories
    python -m src.ingestion.seed --categories cs.AI cs.LG cs.CL
    python -m src.ingestion.seed --max-papers 10000
    python -m src.ingestion.seed --skip-embeddings     # much faster, add embeddings later
"""
from __future__ import annotations

import argparse
import asyncio
import re
import time
from datetime import datetime, timezone
from typing import Any
from lxml.etree import fromstring, XMLParser

import httpx
import structlog
from elasticsearch import AsyncElasticsearch
from elasticsearch.helpers import async_bulk

from src.core.config import get_settings
from src.core.elasticsearch import get_es_client, ensure_index, close_es_client
from src.core.embeddings import encode_texts
from src.ingestion.state import save_state, get_state

logger = structlog.get_logger()

ARXIV_API_URL = "https://export.arxiv.org/api/query"

ATOM_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
    "opensearch": "http://a9.com/-/spec/opensearch/1.1/",
}

# Major CS categories for default seeding
DEFAULT_CATEGORIES = [
    "cs.AI", "cs.CL", "cs.CV", "cs.LG", "cs.NE",
    "cs.IR", "cs.RO", "cs.SE", "cs.DS", "cs.CR",
    "cs.DC", "cs.HC", "cs.MA", "cs.PL", "cs.SY",
    "stat.ML",
    "math.OC",
    "eess.SP", "eess.IV",
    "quant-ph",
]

GITHUB_RE = re.compile(r"https?://github\.com/[\w\-\.]+/[\w\-\.]+")


def parse_api_entry(entry) -> dict | None:
    """Parse a single Atom entry from the ArXiv API response."""
    arxiv_id_raw = entry.findtext("atom:id", "", ATOM_NS)
    if not arxiv_id_raw:
        return None

    # Extract arxiv_id: "http://arxiv.org/abs/2401.00001v1" -> "2401.00001"
    if "/abs/" not in arxiv_id_raw:
        return None  # skip error/non-paper entries
    arxiv_id = arxiv_id_raw.split("/abs/")[-1]
    # Remove version
    arxiv_id = re.sub(r"v\d+$", "", arxiv_id)

    title = entry.findtext("atom:title", "", ATOM_NS).strip()
    title = re.sub(r"\s+", " ", title)  # normalize whitespace

    abstract = entry.findtext("atom:summary", "", ATOM_NS).strip()
    abstract = re.sub(r"\s+", " ", abstract)

    if not title or not abstract:
        return None

    # Authors
    authors = []
    for i, author_el in enumerate(entry.findall("atom:author", ATOM_NS)):
        name = author_el.findtext("atom:name", "", ATOM_NS).strip()
        if name:
            authors.append({
                "name": name,
                "is_first_author": i == 0,
                "h_index": None,
                "citation_count": None,
            })

    # Categories
    categories = []
    primary_category = None
    prim_el = entry.find("arxiv:primary_category", ATOM_NS)
    if prim_el is not None:
        primary_category = prim_el.get("term", "")
        categories.append(primary_category)

    for cat_el in entry.findall("atom:category", ATOM_NS):
        term = cat_el.get("term", "")
        if term and term not in categories:
            categories.append(term)

    # Dates
    published = entry.findtext("atom:published", "", ATOM_NS)
    updated = entry.findtext("atom:updated", "", ATOM_NS)

    submitted_date = None
    if published:
        try:
            submitted_date = datetime.fromisoformat(published.replace("Z", "+00:00")).isoformat()
        except ValueError:
            pass

    updated_date = None
    if updated:
        try:
            updated_date = datetime.fromisoformat(updated.replace("Z", "+00:00")).isoformat()
        except ValueError:
            pass

    # DOI and journal ref
    doi = entry.findtext("arxiv:doi", None, ATOM_NS)
    journal_ref = entry.findtext("arxiv:journal_ref", None, ATOM_NS)
    comments = entry.findtext("arxiv:comment", None, ATOM_NS)

    # Page count from comments
    page_count = None
    if comments:
        m = re.search(r"(\d+)\s*pages?", comments, re.IGNORECASE)
        if m:
            page_count = int(m.group(1))

    # GitHub URLs
    github_urls = list(set(GITHUB_RE.findall(abstract or "") + GITHUB_RE.findall(comments or "")))

    # PDF link
    pdf_url = None
    for link in entry.findall("atom:link", ATOM_NS):
        if link.get("title") == "pdf":
            pdf_url = link.get("href")
            break

    first_author = authors[0]["name"] if authors else None

    return {
        "arxiv_id": arxiv_id,
        "title": title,
        "abstract": abstract,
        "authors": authors,
        "categories": categories,
        "domains": list(set(c.split(".")[0] for c in categories)),
        "primary_category": primary_category,
        "submitted_date": submitted_date,
        "updated_date": updated_date,
        "published_date": submitted_date,
        "doi": doi,
        "journal_ref": journal_ref,
        "comments": comments,
        "page_count": page_count,
        "has_github": len(github_urls) > 0,
        "github_urls": github_urls,
        "pdf_url": pdf_url,
        "abstract_url": f"https://arxiv.org/abs/{arxiv_id}",
        "first_author": first_author,
        "first_author_h_index": None,
        "citation_stats": {
            "total_citations": 0,
            "avg_citation_age_years": None,
            "median_h_index_citing_authors": None,
            "top_citing_categories": [],
        },
        "references_stats": {
            "total_references": 0,
            "avg_reference_age_years": None,
            "top_referenced_categories": [],
        },
    }


async def fetch_api_page(
    http: httpx.AsyncClient,
    search_query: str,
    start: int,
    max_results: int = 200,
) -> tuple[list[dict], int]:
    """Fetch one page from ArXiv API. Returns (papers, total_results)."""
    params = {
        "search_query": search_query,
        "start": start,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }

    for attempt in range(5):
        try:
            resp = await http.get(ARXIV_API_URL, params=params, timeout=60)
            if resp.status_code == 503:
                wait = int(resp.headers.get("Retry-After", 30))
                logger.warning("arxiv_503_retry", wait=wait, attempt=attempt)
                await asyncio.sleep(wait)
                continue
            resp.raise_for_status()
            break
        except (httpx.HTTPError, httpx.TimeoutException) as e:
            logger.warning("arxiv_api_error", error=str(e), attempt=attempt)
            await asyncio.sleep(5 * (attempt + 1))
    else:
        return [], 0

    parser = XMLParser(resolve_entities=False, no_network=True)
    root = fromstring(resp.content, parser=parser)

    total_str = root.findtext("opensearch:totalResults", "0", ATOM_NS)
    total = int(total_str)

    papers = []
    for entry in root.findall("atom:entry", ATOM_NS):
        paper = parse_api_entry(entry)
        if paper:
            papers.append(paper)

    return papers, total


def add_embeddings_batch(papers: list[dict]) -> list[dict]:
    """Add title and abstract embeddings to a batch of papers."""
    if not papers:
        return papers

    titles = [p["title"] for p in papers]
    abstracts = [p["abstract"] for p in papers]

    title_embs = encode_texts(titles, batch_size=64)
    abstract_embs = encode_texts(abstracts, batch_size=64)

    for i, paper in enumerate(papers):
        if i < len(title_embs):
            paper["title_embedding"] = title_embs[i]
        if i < len(abstract_embs):
            paper["abstract_embedding"] = abstract_embs[i]

    return papers


async def index_batch(
    client: AsyncElasticsearch, index: str, papers: list[dict]
) -> int:
    """Bulk index papers into Elasticsearch."""
    if not papers:
        return 0

    actions = [
        {"_index": index, "_id": p["arxiv_id"], "_source": p}
        for p in papers
    ]
    success, errors = await async_bulk(client, actions, raise_on_error=False)
    if errors:
        logger.warning("bulk_index_errors", count=len(errors))
    return success


async def seed_category(
    http: httpx.AsyncClient,
    es: AsyncElasticsearch,
    index: str,
    category: str,
    max_papers: int,
    skip_embeddings: bool,
    batch_size: int = 200,
) -> int:
    """Seed papers from a single ArXiv category."""
    search_query = f"cat:{category}"
    total_indexed = 0
    start = 0

    logger.info("seeding_category", category=category, max_papers=max_papers)

    while start < max_papers:
        fetch_size = min(batch_size, max_papers - start)
        papers, total_available = await fetch_api_page(http, search_query, start, fetch_size)

        if not papers:
            logger.info("no_more_papers", category=category, start=start)
            break

        if not skip_embeddings:
            papers = add_embeddings_batch(papers)

        count = await index_batch(es, index, papers)
        total_indexed += count
        start += fetch_size

        logger.info(
            "seed_progress",
            category=category,
            indexed=total_indexed,
            available=total_available,
            batch=count,
        )

        # ArXiv API requires 3-second delay between requests
        await asyncio.sleep(3)

        if start >= total_available:
            break

    return total_indexed


async def run_seed(
    categories: list[str] | None = None,
    max_papers_per_category: int = 2000,
    skip_embeddings: bool = False,
) -> int:
    """Main seeding entry point."""
    settings = get_settings()
    es = await get_es_client()
    await ensure_index(es, settings.es_index, settings.embedding_dim)

    cats = categories or DEFAULT_CATEGORIES
    total = 0

    logger.info(
        "seed_start",
        categories=cats,
        max_per_cat=max_papers_per_category,
        skip_embeddings=skip_embeddings,
    )

    async with httpx.AsyncClient() as http:
        for cat in cats:
            state = await get_state(es, f"seed:{cat}")
            already_done = state and state.get("status") == "completed"

            if already_done:
                prev = state.get("total_harvested", 0)
                logger.info("category_already_seeded", category=cat, count=prev)
                total += prev
                continue

            await save_state(es, f"seed:{cat}", status="running")

            count = await seed_category(
                http, es, settings.es_index, cat,
                max_papers_per_category, skip_embeddings,
            )
            total += count

            await save_state(
                es, f"seed:{cat}",
                total_harvested=count,
                status="completed",
            )

            logger.info("category_seeded", category=cat, count=count, total=total)

    # Refresh index to make all docs searchable
    await es.indices.refresh(index=settings.es_index)

    logger.info("seed_complete", total=total)
    return total


def main():
    parser = argparse.ArgumentParser(description="Seed ArXiv papers from the ArXiv API")
    parser.add_argument(
        "--categories", nargs="+", default=None,
        help="ArXiv categories to seed (default: major CS categories)",
    )
    parser.add_argument(
        "--max-papers", type=int, default=2000,
        help="Max papers per category (default: 2000)",
    )
    parser.add_argument(
        "--skip-embeddings", action="store_true",
        help="Skip embedding generation for faster seeding",
    )
    args = parser.parse_args()

    async def _run():
        try:
            await run_seed(
                categories=args.categories,
                max_papers_per_category=args.max_papers,
                skip_embeddings=args.skip_embeddings,
            )
        finally:
            await close_es_client()

    asyncio.run(_run())


if __name__ == "__main__":
    main()
