#!/usr/bin/env python3
"""Bulk enrichment via Semantic Scholar batch API.

Uses the POST /paper/batch endpoint to fetch up to 500 papers per request,
bypassing the per-paper rate limit issue.
"""
import asyncio
import sys
import statistics
import httpx

S2_API = "https://api.semanticscholar.org/graph/v1"
S2_FIELDS = "citationCount,citations.externalIds,citations.fieldsOfStudy,authors.hIndex,references.externalIds,externalIds"
ES_URL = "http://localhost:9200"
INDEX = "arxiv_papers"
BATCH_SIZE = 100  # S2 allows up to 500 per batch request


async def main():
    target = int(sys.argv[1]) if len(sys.argv) > 1 else 500

    async with httpx.AsyncClient(timeout=60) as http:
        # Fetch recent un-enriched papers from ES
        query = {
            "query": {
                "bool": {
                    "filter": [
                        {"range": {"submitted_date": {"gte": "2024-01-01"}}},
                        {"term": {"citation_stats.total_citations": 0}},
                    ]
                }
            },
            "size": target,
            "sort": [{"submitted_date": {"order": "desc"}}],
            "_source": ["arxiv_id"],
        }
        r = await http.post(f"{ES_URL}/{INDEX}/_search", json=query)
        hits = r.json()["hits"]["hits"]
        print(f"Found {len(hits)} papers to enrich")

        total_enriched = 0
        total_skipped = 0

        # Process in batches
        for batch_start in range(0, len(hits), BATCH_SIZE):
            batch = hits[batch_start:batch_start + BATCH_SIZE]
            arxiv_ids = [h["_source"]["arxiv_id"] for h in batch]
            doc_ids = {h["_source"]["arxiv_id"]: h["_id"] for h in batch}

            # S2 batch lookup
            s2_ids = [f"ARXIV:{aid}" for aid in arxiv_ids]
            body = {"ids": s2_ids}
            params = {"fields": S2_FIELDS}

            print(f"\nBatch {batch_start//BATCH_SIZE + 1}: fetching {len(s2_ids)} papers from S2...")

            for attempt in range(5):
                try:
                    resp = await http.post(
                        f"{S2_API}/paper/batch",
                        json=body,
                        params=params,
                    )
                    if resp.status_code == 200:
                        break
                    if resp.status_code == 429:
                        wait = int(resp.headers.get("Retry-After", 30))
                        print(f"  Rate limited, waiting {wait}s...")
                        await asyncio.sleep(wait)
                        continue
                    print(f"  S2 returned {resp.status_code}: {resp.text[:200]}")
                    await asyncio.sleep(10)
                except Exception as e:
                    print(f"  Error: {e}")
                    await asyncio.sleep(10)
            else:
                print("  Skipping batch after 5 retries")
                continue

            results = resp.json()
            # results is a list, same order as input (null entries for not found)

            bulk_body = []
            batch_enriched = 0

            for arxiv_id, s2_data in zip(arxiv_ids, results):
                if s2_data is None:
                    total_skipped += 1
                    continue

                cites = s2_data.get("citationCount", 0) or 0

                # Extract citing categories
                citing_cats: dict[str, int] = {}
                for c in (s2_data.get("citations") or []):
                    if isinstance(c, dict):
                        for fos in (c.get("fieldsOfStudy") or []):
                            citing_cats[fos] = citing_cats.get(fos, 0) + 1
                top_citing = sorted(citing_cats.keys(), key=lambda k: -citing_cats[k])[:10]

                # Author h-indices
                authors = s2_data.get("authors") or []
                h_indices = [a.get("hIndex") for a in authors if a.get("hIndex")]
                median_h = statistics.median(h_indices) if h_indices else None

                refs = s2_data.get("references") or []
                citations_list = s2_data.get("citations") or []

                # Extract arxiv IDs from references
                ref_arxiv_ids = []
                for ref in refs:
                    if isinstance(ref, dict):
                        ext = ref.get("externalIds") or {}
                        aid = ext.get("ArXiv")
                        if aid:
                            ref_arxiv_ids.append(aid)

                # Extract arxiv IDs from citations (papers that cite this one)
                cited_by_arxiv_ids = []
                for cit in citations_list:
                    if isinstance(cit, dict):
                        ext = cit.get("externalIds") or {}
                        aid = ext.get("ArXiv")
                        if aid:
                            cited_by_arxiv_ids.append(aid)

                # Build bulk update
                doc_id = doc_ids[arxiv_id]
                bulk_body.append(f'{{"update":{{"_id":"{doc_id}"}}}}')
                import json
                update = {
                    "doc": {
                        "citation_stats": {
                            "total_citations": cites,
                            "top_citing_categories": top_citing,
                            "median_h_index_citing_authors": median_h,
                        },
                        "references_stats": {
                            "total_references": len(refs),
                        },
                        "first_author_h_index": h_indices[0] if h_indices else None,
                        "reference_ids": ref_arxiv_ids,
                        "cited_by_ids": cited_by_arxiv_ids,
                    }
                }
                bulk_body.append(json.dumps(update))
                batch_enriched += 1

                if cites > 0:
                    cat_str = ",".join(top_citing[:3]) if top_citing else "none"
                    print(f"  {arxiv_id}: {cites} cites, refs={len(ref_arxiv_ids)}, cited_by={len(cited_by_arxiv_ids)}, cats=[{cat_str}]")

            # Bulk update ES
            if bulk_body:
                bulk_data = "\n".join(bulk_body) + "\n"
                br = await http.post(
                    f"{ES_URL}/{INDEX}/_bulk",
                    content=bulk_data.encode(),
                    headers={"Content-Type": "application/x-ndjson"},
                )
                if br.status_code != 200:
                    print(f"  ES bulk error: {br.status_code}")
                else:
                    errors = br.json().get("errors", False)
                    if errors:
                        print(f"  ES bulk had errors")

            total_enriched += batch_enriched
            print(f"  Batch done: {batch_enriched} enriched, {len(batch) - batch_enriched} skipped")

            # Small delay between batches
            await asyncio.sleep(2)

        # Refresh index
        await http.post(f"{ES_URL}/{INDEX}/_refresh")
        print(f"\nTotal: {total_enriched} enriched, {total_skipped} not on S2")


if __name__ == "__main__":
    asyncio.run(main())
