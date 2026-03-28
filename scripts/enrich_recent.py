#!/usr/bin/env python3
"""Fast targeted enrichment — enrich recent papers with Semantic Scholar data.

Focuses on papers from the last 2 years that are most likely to have citation
data, enriching enough to demo rising_interdisciplinary queries.
"""
import asyncio
import sys
import time
import httpx
import statistics

S2_API = "https://api.semanticscholar.org/graph/v1"
S2_FIELDS = "citationCount,influentialCitationCount,citations.fieldsOfStudy,authors.hIndex,authors.citationCount,references"

ES_URL = "http://localhost:9200"
INDEX = "arxiv_papers"


async def fetch_s2(http: httpx.AsyncClient, arxiv_id: str) -> dict | None:
    url = f"{S2_API}/paper/ARXIV:{arxiv_id}"
    for attempt in range(3):
        try:
            r = await http.get(url, params={"fields": S2_FIELDS}, timeout=30)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 404:
                return None
            if r.status_code == 429:
                wait = int(r.headers.get("Retry-After", 60))
                print(f"    rate-limited, waiting {wait}s")
                await asyncio.sleep(wait)
                continue
            await asyncio.sleep(3)
        except Exception as e:
            print(f"    error: {e}")
            await asyncio.sleep(5)
    return None


def compute_enrichment(s2: dict) -> dict:
    cites = s2.get("citationCount", 0) or 0

    # Citing categories
    citing_cats: dict[str, int] = {}
    for c in (s2.get("citations") or []):
        if isinstance(c, dict):
            for fos in (c.get("fieldsOfStudy") or []):
                citing_cats[fos] = citing_cats.get(fos, 0) + 1
    top_citing = sorted(citing_cats.keys(), key=lambda k: -citing_cats[k])[:10]

    # Author h-indices
    authors = s2.get("authors") or []
    h_indices = [a.get("hIndex") for a in authors if a.get("hIndex") is not None]

    refs = s2.get("references") or []

    return {
        "citation_stats.total_citations": cites,
        "citation_stats.top_citing_categories": top_citing,
        "citation_stats.median_h_index_citing_authors": statistics.median(h_indices) if h_indices else None,
        "references_stats.total_references": len(refs),
        "first_author_h_index": authors[0].get("hIndex") if authors else None,
    }


async def main():
    batch_size = int(sys.argv[1]) if len(sys.argv) > 1 else 200

    async with httpx.AsyncClient() as http:
        # Find recent papers (last 2 years) that haven't been enriched
        query = {
            "query": {
                "bool": {
                    "filter": [
                        {"range": {"submitted_date": {"gte": "2024-01-01"}}},
                        {"term": {"citation_stats.total_citations": 0}},
                    ]
                }
            },
            "size": batch_size,
            "sort": [{"submitted_date": {"order": "desc"}}],
            "_source": ["arxiv_id"],
        }

        r = await http.post(f"{ES_URL}/{INDEX}/_search", json=query, timeout=30)
        hits = r.json()["hits"]["hits"]
        print(f"Found {len(hits)} recent papers to enrich")

        enriched = 0
        skipped = 0

        for i, hit in enumerate(hits):
            arxiv_id = hit["_source"]["arxiv_id"]
            doc_id = hit["_id"]

            s2 = await fetch_s2(http, arxiv_id)
            if not s2:
                skipped += 1
                print(f"  [{i+1}/{len(hits)}] {arxiv_id} — not found on S2")
                await asyncio.sleep(3.1)
                continue

            fields = compute_enrichment(s2)
            cites = fields["citation_stats.total_citations"]
            cats = fields["citation_stats.top_citing_categories"]

            # Update ES
            update_body = {"doc": {
                "citation_stats": {
                    "total_citations": cites,
                    "top_citing_categories": cats,
                    "median_h_index_citing_authors": fields["citation_stats.median_h_index_citing_authors"],
                },
                "references_stats": {
                    "total_references": fields["references_stats.total_references"],
                },
                "first_author_h_index": fields["first_author_h_index"],
            }}
            await http.post(f"{ES_URL}/{INDEX}/_update/{doc_id}", json=update_body, timeout=10)
            enriched += 1

            status = f"cites={cites}"
            if cats:
                status += f" cats={cats[:3]}"
            print(f"  [{i+1}/{len(hits)}] {arxiv_id} — {status}")

            await asyncio.sleep(3.1)  # Semantic Scholar rate limit

        print(f"\nDone: {enriched} enriched, {skipped} skipped")


if __name__ == "__main__":
    asyncio.run(main())
