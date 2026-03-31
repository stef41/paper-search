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
import json
import statistics
import time
from datetime import datetime, timezone
from typing import Any

import httpx
import structlog
from elasticsearch import AsyncElasticsearch

from src.core.config import get_settings
from src.core.elasticsearch import get_es_client, ensure_index, close_es_client

logger = structlog.get_logger()

S2_API = "https://api.semanticscholar.org/graph/v1"
S2_FIELDS = "citationCount,influentialCitationCount,references.externalIds,references.fieldsOfStudy,citations.externalIds,citations.citationCount,citations.fieldsOfStudy,citations.authors.hIndex,authors.hIndex,authors.citationCount,externalIds"

# Semantic Scholar rate limit: 100 requests per 5 minutes (free tier)
S2_DELAY = 3.1  # seconds between requests


async def fetch_s2_paper(
    http: httpx.AsyncClient,
    arxiv_id: str,
) -> dict | None | bool:
    """Fetch paper data from Semantic Scholar by ArXiv ID.

    Returns:
        dict: paper data on success
        False: definitive 404 (paper not in S2)
        None: transient failure (should retry later)
    """
    url = f"{S2_API}/paper/ARXIV:{arxiv_id}"
    params = {"fields": S2_FIELDS}

    for attempt in range(3):
        try:
            resp = await http.get(url, params=params, timeout=30)

            if resp.status_code == 404:
                return False
            if resp.status_code == 429:
                try:
                    wait = min(int(resp.headers.get("Retry-After", 60)), 120)
                except (ValueError, TypeError):
                    wait = 60
                logger.warning("s2_rate_limited", wait=wait)
                await asyncio.sleep(wait)
                continue
            if resp.status_code == 200:
                try:
                    return resp.json()
                except (ValueError, json.JSONDecodeError):
                    logger.warning("s2_json_error", arxiv_id=arxiv_id)
                    return None

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
    if authors and isinstance(authors[0], dict):
        result["first_author_h_index"] = authors[0].get("hIndex")

    # Citation stats
    citation_count = s2_data.get("citationCount", 0) or 0
    citations = s2_data.get("citations", []) or []

    citing_h_indices = []
    citing_categories = []
    for cit in citations:
        if isinstance(cit, dict):
            # Collect h-indices of citing papers' authors
            for a in (cit.get("authors") or []):
                if isinstance(a, dict):
                    h = a.get("hIndex")
                    if h is not None:
                        citing_h_indices.append(h)
            fos = cit.get("fieldsOfStudy") or []
            citing_categories.extend(fos)

    median_h_index_citing = None
    if citing_h_indices:
        median_h_index_citing = statistics.median(citing_h_indices)

    # Top citing categories
    cat_counts: dict[str, int] = {}
    for c in citing_categories:
        cat_counts[c] = cat_counts.get(c, 0) + 1
    top_cats = sorted(cat_counts.keys(), key=lambda k: cat_counts[k], reverse=True)[:10]

    result["citation_stats"] = {
        "total_citations": citation_count,
        "avg_citation_age_years": None,  # Would need publication dates of citing papers
        "median_h_index_citing_authors": median_h_index_citing,
        "top_citing_categories": top_cats,
    }

    # Reference stats
    references = s2_data.get("references", []) or []
    ref_categories = []
    ref_arxiv_ids: list[str] = []
    for ref in references:
        if isinstance(ref, dict):
            fos = ref.get("fieldsOfStudy") or []
            ref_categories.extend(fos)
            ext = ref.get("externalIds") or {}
            if isinstance(ext, dict) and ext.get("ArXiv"):
                ref_arxiv_ids.append(ext["ArXiv"])

    ref_cat_counts: dict[str, int] = {}
    for c in ref_categories:
        ref_cat_counts[c] = ref_cat_counts.get(c, 0) + 1
    top_ref_cats = sorted(ref_cat_counts.keys(), key=lambda k: ref_cat_counts[k], reverse=True)[:5]

    result["references_stats"] = {
        "total_references": len(references),
        "avg_reference_age_years": None,
        "top_referenced_categories": top_ref_cats,
    }

    # Extract citation and reference arxiv IDs for graph building
    if ref_arxiv_ids:
        result["reference_ids"] = list(dict.fromkeys(ref_arxiv_ids))

    cited_by_arxiv_ids: list[str] = []
    for cit in citations:
        if isinstance(cit, dict):
            ext = cit.get("externalIds") or {}
            if isinstance(ext, dict) and ext.get("ArXiv"):
                cited_by_arxiv_ids.append(ext["ArXiv"])
    if cited_by_arxiv_ids:
        result["cited_by_ids"] = list(dict.fromkeys(cited_by_arxiv_ids))

    # Update author-level data
    if authors:
        enriched_authors = []
        for i, a in enumerate(authors):
            if isinstance(a, dict):
                enriched_authors.append({
                    "h_index": a.get("hIndex"),
                    "citation_count": a.get("citationCount"),
                })
            else:
                enriched_authors.append({"h_index": None, "citation_count": None})
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
    query = {
        "query": {
            "bool": {
                "must_not": [
                    {"exists": {"field": "enriched_at"}},
                ],
            }
        },
        "size": batch_size,
        "sort": [{"submitted_date": {"order": "desc"}}, {"arxiv_id": {"order": "asc"}}],
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

                if s2_data and s2_data is not False:
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

                    try:
                        await es.update(
                            index=settings.es_index,
                            id=arxiv_id,
                            body={"doc": {
                                **update_doc,
                                "enrichment_source": "semantic_scholar",
                                "enriched_at": datetime.now(timezone.utc).isoformat(),
                            }},
                        )
                    except Exception as e:
                        logger.warning("paper_update_failed", arxiv_id=arxiv_id, error=str(e))
                        await asyncio.sleep(S2_DELAY)
                        continue
                    total_enriched += 1
                elif s2_data is False:
                    # Definitive 404 — paper not in Semantic Scholar
                    try:
                        await es.update(
                            index=settings.es_index,
                            id=arxiv_id,
                            body={"doc": {
                                "enrichment_source": "s2_not_found",
                                "enriched_at": datetime.now(timezone.utc).isoformat(),
                            }},
                        )
                    except Exception as e:
                        logger.warning("paper_update_failed", arxiv_id=arxiv_id, error=str(e))
                        await asyncio.sleep(S2_DELAY)
                        continue

                if total_enriched % 10 == 0 and total_enriched > 0:
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

    async def _run():
        try:
            await enrich_papers(
                max_papers=args.max_papers,
                batch_size=args.batch_size,
            )
        finally:
            await close_es_client()

    asyncio.run(_run())


if __name__ == "__main__":
    main()
