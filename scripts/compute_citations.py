#!/usr/bin/env python3
"""Self-compute citation counts and cited_by links from internal reference data.

This script requires NO external API calls. It works by inverting the reference
graph already stored in Elasticsearch:
  - Paper A has reference_ids: [B, C, D]
  - Therefore: B is cited by A, C is cited by A, D is cited by A

The script:
  1. Scans all papers that have reference_ids (populated by OpenAlex or S2)
  2. Builds an inverted index: for each referenced paper, collect all citers
  3. Bulk-updates cited_by_ids and citation_stats.total_citations in ES

This is O(n) over enriched papers and requires zero API calls.

Usage:
  python3 scripts/compute_citations.py [--dry-run] [--batch-size 5000]
"""
import argparse
import asyncio
import json
import time
from collections import defaultdict

import httpx

ES_URL = "http://localhost:9200"
INDEX = "arxiv_papers"
SCROLL_SIZE = 2000  # papers per scroll page
BULK_SIZE = 5000  # updates per bulk request


async def scan_references(http: httpx.AsyncClient) -> dict[str, list[str]]:
    """Scan all papers with reference_ids and build the inverted citation map.

    Returns: {cited_paper_arxiv_id: [citing_paper_arxiv_id, ...]}
    """
    cited_by: dict[str, list[str]] = defaultdict(list)
    total_refs = 0
    total_papers = 0

    # Use scroll API to iterate over all papers with reference_ids
    query = {
        "query": {"exists": {"field": "reference_ids"}},
        "size": SCROLL_SIZE,
        "_source": ["arxiv_id", "reference_ids"],
    }

    r = await http.post(
        f"{ES_URL}/{INDEX}/_search?scroll=5m",
        json=query,
    )
    data = r.json()
    if r.status_code != 200 or "hits" not in data:
        print(f"ES search failed (status {r.status_code}): {str(data)[:300]}")
        return dict(cited_by)
    scroll_id = data.get("_scroll_id")
    hits = data["hits"]["hits"]

    try:
        while hits:
            for h in hits:
                src = h["_source"]
                citing_id = src["arxiv_id"]
                refs = src.get("reference_ids") or []
                total_papers += 1
                total_refs += len(refs)

                for ref_id in set(refs):
                    cited_by[ref_id].append(citing_id)

            # Get next scroll page
            r = await http.post(
                f"{ES_URL}/_search/scroll",
                json={"scroll": "5m", "scroll_id": scroll_id},
            )
            data = r.json()
            if r.status_code != 200 or "hits" not in data:
                print(f"ES scroll failed (status {r.status_code}): {str(data)[:300]}")
                break
            scroll_id = data.get("_scroll_id")
            hits = data["hits"]["hits"]
    finally:
        # Clear scroll
        if scroll_id:
            await http.request(
                "DELETE",
                f"{ES_URL}/_search/scroll",
                json={"scroll_id": scroll_id},
            )

    print(f"Scanned {total_papers} papers with {total_refs} total references")
    print(f"Built citation map for {len(cited_by)} unique cited papers")

    return dict(cited_by)


async def resolve_arxiv_ids_to_doc_ids(
    http: httpx.AsyncClient, arxiv_ids: list[str]
) -> dict[str, str]:
    """Batch-resolve arxiv_ids to ES document _ids using mget."""
    mapping: dict[str, str] = {}

    for i in range(0, len(arxiv_ids), 1000):
        batch = arxiv_ids[i:i + 1000]

        # Use _search with terms filter
        query = {
            "query": {"terms": {"arxiv_id": batch}},
            "size": len(batch),
            "_source": ["arxiv_id"],
        }
        r = await http.post(f"{ES_URL}/{INDEX}/_search", json=query)
        resp_data = r.json()
        if r.status_code != 200 or "hits" not in resp_data:
            print(f"  ES resolve failed (status {r.status_code}): {str(resp_data)[:200]}")
            continue
        for h in resp_data["hits"]["hits"]:
            mapping[h["_source"]["arxiv_id"]] = h["_id"]

    return mapping


async def bulk_update_citations(
    http: httpx.AsyncClient,
    cited_by: dict[str, list[str]],
    doc_id_map: dict[str, str],
    batch_size: int,
    dry_run: bool,
) -> tuple[int, int]:
    """Bulk-update cited_by_ids and citation counts in ES.

    Returns (updated, skipped).
    """
    updated = 0
    skipped = 0
    bulk_body: list[str] = []

    for arxiv_id, citers in cited_by.items():
        doc_id = doc_id_map.get(arxiv_id)
        if not doc_id:
            # Paper is referenced but not in our index
            skipped += 1
            continue

        doc = {
            "cited_by_ids": citers,
            "citation_stats": {
                "total_citations": len(citers),
            },
        }

        bulk_body.append(json.dumps({"update": {"_id": doc_id}}))
        bulk_body.append(json.dumps({"doc": doc}))
        updated += 1

        # Flush bulk buffer
        if len(bulk_body) >= batch_size * 2:
            if not dry_run:
                bulk_data = "\n".join(bulk_body) + "\n"
                r = await http.post(
                    f"{ES_URL}/{INDEX}/_bulk",
                    content=bulk_data.encode(),
                    headers={"Content-Type": "application/x-ndjson"},
                    timeout=120,
                )
                if r.status_code != 200:
                    print(f"  ES bulk error: {r.status_code}")
                elif r.json().get("errors"):
                    errs = [
                        item["update"]["error"]["reason"]
                        for item in r.json()["items"]
                        if "error" in item.get("update", {})
                    ]
                    print(f"  ES bulk had {len(errs)} errors: {errs[:3]}")
            print(f"  Flushed {len(bulk_body) // 2} updates (total: {updated})")
            bulk_body = []

    # Flush remaining
    if bulk_body and not dry_run:
        bulk_data = "\n".join(bulk_body) + "\n"
        r = await http.post(
            f"{ES_URL}/{INDEX}/_bulk",
            content=bulk_data.encode(),
            headers={"Content-Type": "application/x-ndjson"},
            timeout=120,
        )
        if r.status_code != 200:
            print(f"  ES bulk error: {r.status_code}")
    if bulk_body:
        print(f"  Flushed final {len(bulk_body) // 2} updates (total: {updated})")

    return updated, skipped


async def main():
    parser = argparse.ArgumentParser(
        description="Self-compute citation counts from internal reference data"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Just scan and report, don't update ES")
    parser.add_argument("--batch-size", type=int, default=BULK_SIZE,
                        help=f"Bulk update batch size (default: {BULK_SIZE})")
    args = parser.parse_args()

    print("=" * 60)
    print("Self-Compute Citation Counts")
    print(f"  Dry run: {args.dry_run}")
    print(f"  Batch size: {args.batch_size}")
    print("=" * 60)

    t0 = time.time()

    async with httpx.AsyncClient(timeout=60) as http:
        # Step 1: Scan reference graph
        print("\n[1/3] Scanning reference graph...")
        cited_by = await scan_references(http)

        if not cited_by:
            print("No reference data found. Run enrich_openalex.py first.")
            return

        # Stats
        citation_counts = [len(v) for v in cited_by.values()]
        max_cited = max(citation_counts)
        avg_cited = sum(citation_counts) / len(citation_counts)
        papers_with_1plus = sum(1 for c in citation_counts if c >= 1)

        print(f"\nCitation distribution:")
        print(f"  Papers with ≥1 internal citation: {papers_with_1plus}")
        print(f"  Max citations: {max_cited}")
        print(f"  Avg citations: {avg_cited:.1f}")

        if args.dry_run:
            # Show top cited papers
            top = sorted(cited_by.items(), key=lambda x: len(x[1]), reverse=True)[:10]
            print(f"\nTop 10 most-cited papers (internal):")
            for aid, citers in top:
                print(f"  {aid}: {len(citers)} citations")
            print(f"\nDone (dry run). Elapsed: {time.time() - t0:.1f}s")
            return

        # Step 2: Resolve arxiv IDs to ES doc IDs
        print("\n[2/3] Resolving document IDs...")
        all_cited_ids = list(cited_by.keys())
        doc_id_map = await resolve_arxiv_ids_to_doc_ids(http, all_cited_ids)
        print(f"  Resolved {len(doc_id_map)}/{len(all_cited_ids)} papers in our index")

        # Step 3: Bulk update
        print("\n[3/3] Updating citations in ES...")
        updated, skipped = await bulk_update_citations(
            http, cited_by, doc_id_map, args.batch_size, args.dry_run
        )

        # Refresh index
        await http.post(f"{ES_URL}/{INDEX}/_refresh")

        elapsed = time.time() - t0
        print(f"\n{'=' * 60}")
        print(f"DONE in {elapsed:.1f}s")
        print(f"  Papers updated with cited_by_ids: {updated}")
        print(f"  Referenced papers not in our index: {skipped}")
        print(f"{'=' * 60}")


if __name__ == "__main__":
    asyncio.run(main())
