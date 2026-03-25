"""
Semantic Scholar enrichment — adds citation counts, h-index, and reference data.

Uses the free Semantic Scholar API (no key needed, 100 req/5min).
Run this after initial seeding to enrich papers with real citation data.

Usage:
    python -m src.ingestion.enrich
    python -m src.ingestion.enrich --batch-size 50 --max-papers 10000
"""
from __future__ import annotations

import argparse
import asyncio
import statistics
import time
from typing import Any

import httpx
import structlog
from elasticsearch import AsyncElasticsearch

from src.core.config import get_settings
from src.core.elasticsearch import get_es_client, ensure_index

logger = structlog.get_logger()

S2_API = "https://api.semanticscholar.org/graph/v1"
S2_FIELDS = "citationCount,influentialCitationCount,references,citations.citationCount,citations.fieldsOfStudy,authors.hIndex,authors.citationCount,externalIds"

# Semantic Scholar rate limit: 100 requests per 5 minutes (free tier)
S2_DELAY = 3.1  # seconds between requests


async def fetch_s2_paper(
    http: httpx.AsyncClient,
    arxiv_id: str,
) -> dict | None:
    """Fetch paper data from Semantic Scholar by ArXiv ID."""
    url = f"{S2_API}/paper/ARXIV:{arxiv_id}"
    params = {"fields": S2_FIELDS}

    for attempt in range(3):
        try:
            resp = await http.get(url, params=params, timeout=30)

            if resp.status_code == 404:
                return None
            if resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", 60))
                logger.warning("s2_rate_limited", wait=wait)
                await asyncio.sleep(wait)
                continue
            if resp.status_code == 200:
                return resp.json()

            logger.warning("s2_error", status=resp.status_code, arxiv_id=arxiv_id)
            await asyncio.sleep(5)

        except (httpx.HTTPError, httpx.TimeoutException) as e:
            logger.warning("s2_request_error", error=str(e), attempt=attempt)
            await asyncio.sleep(5)

    return None


def compute_enrichment(s2_data: dict) -> dict:
    """Compute enrichment fields from Semantic Scholar data."""
    result: dict[str, Any] = {}

    # Author h-index data
    authors = s2_data.get("authors", [])
    if authors:
        h_indices = [a.get("hIndex") for a in authors if a.get("hIndex") is not None]
        if h_indices:
            result["first_author_h_index"] = h_indices[0] if h_indices else None

    # Citation stats
    citation_count = s2_data.get("citationCount", 0) or 0
    citations = s2_data.get("citations", []) or []

    citing_h_indices = []
    citing_categories = []
    for cit in citations:
        if isinstance(cit, dict):
            cc = cit.get("citationCount")
            if cc is not None:
                citing_h_indices.append(cc)
            fos = cit.get("fieldsOfStudy") or []
            citing_categories.extend(fos)

    median_h = None
    if citing_h_indices:
        median_h = statistics.median(citing_h_indices)

    # Top citing categories
    cat_counts: dict[str, int] = {}
    for c in citing_categories:
        cat_counts[c] = cat_counts.get(c, 0) + 1
    top_cats = sorted(cat_counts.keys(), key=lambda k: cat_counts[k], reverse=True)[:5]

    result["citation_stats"] = {
        "total_citations": citation_count,
        "avg_citation_age_years": None,  # Would need publication dates of citing papers
        "median_h_index_citing_authors": median_h,
        "top_citing_categories": top_cats,
    }

    # Reference stats
    references = s2_data.get("references", []) or []
    ref_categories = []
    for ref in references:
        if isinstance(ref, dict):
            fos = ref.get("fieldsOfStudy") or []
            ref_categories.extend(fos)

    ref_cat_counts: dict[str, int] = {}
    for c in ref_categories:
        ref_cat_counts[c] = ref_cat_counts.get(c, 0) + 1
    top_ref_cats = sorted(ref_cat_counts.keys(), key=lambda k: ref_cat_counts[k], reverse=True)[:5]

    result["references_stats"] = {
        "total_references": len(references),
        "avg_reference_age_years": None,
        "top_referenced_categories": top_ref_cats,
    }

    # Update author-level data
    if authors:
        enriched_authors = []
        for i, a in enumerate(authors):
            enriched_authors.append({
                "h_index": a.get("hIndex"),
                "citation_count": a.get("citationCount"),
            })
        result["_author_enrichment"] = enriched_authors

    return result


async def enrich_papers(
    max_papers: int | None = None,
    batch_size: int = 50,
) -> int:
    """Enrich indexed papers with Semantic Scholar data."""
    settings = get_settings()
    es = await get_es_client()

    logger.info("enrichment_start", max_papers=max_papers)

    # Scroll through papers that haven't been enriched yet
    # (citation_stats.total_citations == 0 as a proxy)
    query = {
        "query": {
            "bool": {
                "must": [
                    {"term": {"citation_stats.total_citations": 0}},
                ],
            }
        },
        "size": batch_size,
        "sort": [{"submitted_date": {"order": "desc"}}],
        "_source": ["arxiv_id", "authors"],
    }

    total_enriched = 0
    start_time = time.monotonic()

    async with httpx.AsyncClient() as http:
        # Use search_after for efficient pagination
        search_after = None

        while True:
            if max_papers and total_enriched >= max_papers:
                break

            body = dict(query)
            if search_after:
                body["search_after"] = search_after

            resp = await es.search(index=settings.es_index, body=body)
            hits = resp["hits"]["hits"]

            if not hits:
                break

            for hit in hits:
                if max_papers and total_enriched >= max_papers:
                    break

                arxiv_id = hit["_source"]["arxiv_id"]
                existing_authors = hit["_source"].get("authors", [])

                s2_data = await fetch_s2_paper(http, arxiv_id)

                if s2_data:
                    enrichment = compute_enrichment(s2_data)

                    # Merge author enrichment
                    author_enrich = enrichment.pop("_author_enrichment", [])
                    update_doc = enrichment

                    if author_enrich and existing_authors:
                        for i, ea in enumerate(existing_authors):
                            if i < len(author_enrich):
                                ea["h_index"] = author_enrich[i].get("h_index")
                                ea["citation_count"] = author_enrich[i].get("citation_count")
                        update_doc["authors"] = existing_authors

                    await es.update(
                        index=settings.es_index,
                        id=arxiv_id,
                        body={"doc": update_doc},
                    )
                    total_enriched += 1

                    if total_enriched % 10 == 0:
                        elapsed = time.monotonic() - start_time
                        logger.info(
                            "enrichment_progress",
                            enriched=total_enriched,
                            elapsed=f"{elapsed:.0f}s",
                        )

                # Rate limit
                await asyncio.sleep(S2_DELAY)

            search_after = hits[-1]["sort"]

    elapsed = time.monotonic() - start_time
    logger.info(
        "enrichment_complete",
        total=total_enriched,
        elapsed=f"{elapsed:.0f}s",
    )
    return total_enriched


def main():
    parser = argparse.ArgumentParser(description="Enrich papers with Semantic Scholar data")
    parser.add_argument("--max-papers", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=50)
    args = parser.parse_args()

    asyncio.run(enrich_papers(
        max_papers=args.max_papers,
        batch_size=args.batch_size,
    ))


if __name__ == "__main__":
    main()
