#!/usr/bin/env python3
"""Targeted enrichment — enrich papers matching a specific ES query.

Usage:
    python scripts/enrich_targeted.py '{"query":"transformer","categories":["cs.LG"]}' 200
"""
import asyncio
import json
import sys
import statistics
from datetime import datetime, timezone
import httpx

S2_API = "https://api.semanticscholar.org/graph/v1"
S2_FIELDS = "citationCount,citations.externalIds,citations.fieldsOfStudy,citations.citationCount,authors.hIndex,references.externalIds,externalIds"
ES_URL = "http://localhost:9200"
INDEX = "arxiv_papers"
BATCH_SIZE = 100


async def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/enrich_targeted.py '<search_json>' [max_papers]")
        sys.exit(1)

    search_params = json.loads(sys.argv[1])
    target = int(sys.argv[2]) if len(sys.argv) > 2 else 200

    async with httpx.AsyncClient(timeout=60) as http:
        # Build ES query from the search params
        must = []
        filters = []

        if search_params.get("query"):
            must.append({"multi_match": {"query": search_params["query"], "fields": ["title^3", "abstract^2"]}})
        if search_params.get("categories"):
            filters.append({"terms": {"categories": search_params["categories"]}})
        if search_params.get("submitted_date"):
            date_range = {}
            if search_params["submitted_date"].get("gte"):
                date_range["gte"] = search_params["submitted_date"]["gte"]
            if date_range:
                filters.append({"range": {"submitted_date": date_range}})

        # Only papers without reference_ids
        filters.append({
            "bool": {
                "should": [
                    {"bool": {"must_not": [{"exists": {"field": "reference_ids"}}]}},
                    {"script": {"script": "doc.containsKey('reference_ids') && doc['reference_ids'].size() == 0"}},
                ],
                "minimum_should_match": 1,
            }
        })

        query = {
            "query": {"bool": {"must": must or [{"match_all": {}}], "filter": filters}},
            "size": target,
            "sort": [{"submitted_date": {"order": "desc"}}],
            "_source": ["arxiv_id"],
        }
        r = await http.post(f"{ES_URL}/{INDEX}/_search", json=query)
        hits = r.json()["hits"]["hits"]
        print(f"Found {len(hits)} papers to enrich")

        total_enriched = 0
        total_skipped = 0

        for batch_start in range(0, len(hits), BATCH_SIZE):
            batch = hits[batch_start:batch_start + BATCH_SIZE]
            arxiv_ids = [h["_source"]["arxiv_id"] for h in batch]
            doc_ids = {h["_source"]["arxiv_id"]: h["_id"] for h in batch}

            s2_ids = [f"ARXIV:{aid}" for aid in arxiv_ids]
            print(f"\nBatch {batch_start//BATCH_SIZE + 1}: fetching {len(s2_ids)} papers...")

            for attempt in range(5):
                try:
                    resp = await http.post(f"{S2_API}/paper/batch", json={"ids": s2_ids}, params={"fields": S2_FIELDS})
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
                print("  Skipping batch")
                continue

            results = resp.json()
            bulk_body = []
            batch_enriched = 0

            for arxiv_id, s2_data in zip(arxiv_ids, results):
                if s2_data is None:
                    total_skipped += 1
                    continue

                cites = s2_data.get("citationCount", 0) or 0
                citations_list = s2_data.get("citations") or []
                refs = s2_data.get("references") or []
                authors = s2_data.get("authors") or []

                # Citing categories
                citing_cats = {}
                for c in citations_list:
                    if isinstance(c, dict):
                        for fos in (c.get("fieldsOfStudy") or []):
                            citing_cats[fos] = citing_cats.get(fos, 0) + 1
                top_citing = sorted(citing_cats.keys(), key=lambda k: -citing_cats[k])[:10]

                # Author h-indices
                h_indices = [a.get("hIndex") for a in authors if a.get("hIndex") is not None]

                # Median citation count of citing papers
                citing_citation_counts = []
                for cit in citations_list:
                    if isinstance(cit, dict):
                        cc = cit.get("citationCount")
                        if cc is not None:
                            citing_citation_counts.append(cc)
                median_h = statistics.median(citing_citation_counts) if citing_citation_counts else None

                # Reference arxiv IDs
                ref_ids = []
                for ref in refs:
                    if isinstance(ref, dict):
                        ext = ref.get("externalIds") or {}
                        aid = ext.get("ArXiv")
                        if aid:
                            ref_ids.append(aid)

                # Cited-by arxiv IDs
                cited_by_ids = []
                for cit in citations_list:
                    if isinstance(cit, dict):
                        ext = cit.get("externalIds") or {}
                        aid = ext.get("ArXiv")
                        if aid:
                            cited_by_ids.append(aid)

                doc_id = doc_ids[arxiv_id]
                bulk_body.append(json.dumps({"update": {"_id": doc_id}}))
                update = {"doc": {
                    "citation_stats": {"total_citations": cites, "top_citing_categories": top_citing, "median_h_index_citing_authors": median_h},
                    "references_stats": {"total_references": len(refs)},
                    "first_author_h_index": authors[0].get("hIndex") if authors else None,
                    "reference_ids": ref_ids,
                    "cited_by_ids": cited_by_ids,
                    "enrichment_source": "semantic_scholar",
                    "enriched_at": datetime.now(timezone.utc).isoformat(),
                }}
                bulk_body.append(json.dumps(update))
                batch_enriched += 1
                if ref_ids:
                    print(f"  {arxiv_id}: {cites} cites, {len(ref_ids)} refs, {len(cited_by_ids)} cited_by")

            if bulk_body:
                bulk_data = "\n".join(bulk_body) + "\n"
                br = await http.post(f"{ES_URL}/{INDEX}/_bulk", content=bulk_data.encode(), headers={"Content-Type": "application/x-ndjson"})
                if br.status_code != 200:
                    print(f"  WARNING: bulk update returned HTTP {br.status_code}")
                elif br.json().get("errors"):
                    errs = sum(1 for it in br.json().get("items", []) if "error" in it.get("update", {}))
                    print(f"  WARNING: {errs} bulk update errors")

            total_enriched += batch_enriched
            print(f"  Batch done: {batch_enriched} enriched")
            await asyncio.sleep(2)

        await http.post(f"{ES_URL}/{INDEX}/_refresh")
        print(f"\nTotal: {total_enriched} enriched, {total_skipped} not on S2")

if __name__ == "__main__":
    asyncio.run(main())
