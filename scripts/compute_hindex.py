#!/usr/bin/env python3
"""Self-compute author h-index from internal citation data.

This script requires NO external API calls. It works by:
  1. Scanning all papers with citation_stats.total_citations > 0
  2. Grouping papers by author
  3. Computing h-index for each author: h = max h such that h papers have ≥h citations
  4. Updating the h_index in the nested authors array and first_author_h_index

Requires compute_citations.py to have been run first (or OpenAlex enrichment).

Usage:
  python3 scripts/compute_hindex.py [--dry-run] [--min-papers 2]
"""
import argparse
import asyncio
import json
import time
from collections import defaultdict

import httpx

ES_URL = "http://localhost:9200"
INDEX = "arxiv_papers"
SCROLL_SIZE = 5000
BULK_SIZE = 5000


def compute_h_index(citation_counts: list[int]) -> int:
    """Compute h-index: max h such that h papers have ≥ h citations."""
    sorted_counts = sorted(citation_counts, reverse=True)
    h = 0
    for i, c in enumerate(sorted_counts):
        if c >= i + 1:
            h = i + 1
        else:
            break
    return h


async def scan_papers_with_citations(http: httpx.AsyncClient) -> dict[str, list[tuple[str, int]]]:
    """Scan all papers with citation counts > 0.

    Returns: {author_name: [(arxiv_id, citation_count), ...]}
    """
    author_papers: dict[str, list[tuple[str, int]]] = defaultdict(list)
    total = 0

    query = {
        "query": {
            "range": {"citation_stats.total_citations": {"gt": 0}}
        },
        "size": SCROLL_SIZE,
        "_source": ["arxiv_id", "authors", "first_author", "citation_stats.total_citations"],
    }

    r = await http.post(f"{ES_URL}/{INDEX}/_search?scroll=5m", json=query)
    data = r.json()
    if r.status_code != 200 or "hits" not in data:
        print(f"ES search failed (status {r.status_code}): {str(data)[:300]}")
        return dict(author_papers)
    scroll_id = data.get("_scroll_id")
    hits = data["hits"]["hits"]

    while hits:
        for h in hits:
            src = h["_source"]
            citations = (src.get("citation_stats") or {}).get("total_citations", 0) or 0
            authors = src.get("authors") or []
            arxiv_id = src["arxiv_id"]
            total += 1

            for a in authors:
                if isinstance(a, dict):
                    name = a.get("name", "").strip()
                else:
                    name = str(a).strip()
                if name:
                    author_papers[name].append((arxiv_id, citations))

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

    if scroll_id:
        await http.request("DELETE", f"{ES_URL}/_search/scroll", json={"scroll_id": scroll_id})

    print(f"Scanned {total} papers with citations > 0")
    print(f"Found {len(author_papers)} unique authors")

    return dict(author_papers)


async def update_h_indices(
    http: httpx.AsyncClient,
    h_indices: dict[str, int],
    batch_size: int,
    dry_run: bool,
) -> int:
    """Update h_index in ES for all papers by each author.

    Scans all papers and updates the nested authors array + first_author_h_index.
    """
    updated = 0
    bulk_body: list[str] = []

    # Scan ALL papers (we need to update the authors array)
    query = {
        "query": {"match_all": {}},
        "size": SCROLL_SIZE,
        "_source": ["arxiv_id", "authors", "first_author"],
    }

    r = await http.post(f"{ES_URL}/{INDEX}/_search?scroll=5m", json=query)
    data = r.json()
    if r.status_code != 200 or "hits" not in data:
        print(f"ES search failed (status {r.status_code}): {str(data)[:300]}")
        return 0
    scroll_id = data.get("_scroll_id")
    hits = data["hits"]["hits"]

    while hits:
        for h in hits:
            src = h["_source"]
            doc_id = h["_id"]
            authors = src.get("authors") or []
            first_author = src.get("first_author", "")

            has_update = False
            updated_authors = []

            for a in authors:
                if isinstance(a, dict):
                    ac = dict(a)
                    name = ac.get("name", "").strip()
                    h_val = h_indices.get(name)
                    if h_val is not None and ac.get("h_index") != h_val:
                        ac["h_index"] = h_val
                        has_update = True
                    updated_authors.append(ac)
                else:
                    updated_authors.append(a)

            if not has_update:
                continue

            doc: dict = {"authors": updated_authors}

            # Update first_author_h_index
            if first_author:
                h_val = h_indices.get(first_author.strip())
                if h_val is not None:
                    doc["first_author_h_index"] = h_val

            bulk_body.append(json.dumps({"update": {"_id": doc_id}}))
            bulk_body.append(json.dumps({"doc": doc}))
            updated += 1

            if len(bulk_body) >= batch_size * 2:
                if not dry_run:
                    bulk_data = "\n".join(bulk_body) + "\n"
                    r2 = await http.post(
                        f"{ES_URL}/{INDEX}/_bulk",
                        content=bulk_data.encode(),
                        headers={"Content-Type": "application/x-ndjson"},
                        timeout=120,
                    )
                    if r2.status_code != 200:
                        print(f"  ES bulk error: {r2.status_code}")
                print(f"  Flushed {len(bulk_body) // 2} updates (total: {updated})")
                bulk_body = []

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

    if scroll_id:
        await http.request("DELETE", f"{ES_URL}/_search/scroll", json={"scroll_id": scroll_id})

    # Flush remaining
    if bulk_body and not dry_run:
        bulk_data = "\n".join(bulk_body) + "\n"
        r2 = await http.post(
            f"{ES_URL}/{INDEX}/_bulk",
            content=bulk_data.encode(),
            headers={"Content-Type": "application/x-ndjson"},
            timeout=120,
        )
        if r2.status_code != 200:
            print(f"  ES bulk error: {r2.status_code}")
    if bulk_body:
        print(f"  Flushed final {len(bulk_body) // 2} updates (total: {updated})")

    return updated


async def main():
    parser = argparse.ArgumentParser(
        description="Self-compute author h-index from internal citation data"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Just scan and report, don't update ES")
    parser.add_argument("--min-papers", type=int, default=1,
                        help="Minimum papers by an author to compute h-index (default: 1)")
    parser.add_argument("--batch-size", type=int, default=BULK_SIZE,
                        help=f"Bulk update batch size (default: {BULK_SIZE})")
    args = parser.parse_args()

    print("=" * 60)
    print("Self-Compute Author H-Index")
    print(f"  Dry run: {args.dry_run}")
    print(f"  Min papers per author: {args.min_papers}")
    print("=" * 60)

    t0 = time.time()

    async with httpx.AsyncClient(timeout=60) as http:
        # Step 1: Scan papers with citations
        print("\n[1/3] Scanning papers with citations...")
        author_papers = await scan_papers_with_citations(http)

        if not author_papers:
            print("No citation data found. Run compute_citations.py first.")
            return

        # Step 2: Compute h-index for each author
        print("\n[2/3] Computing h-indices...")
        h_indices: dict[str, int] = {}
        for name, papers in author_papers.items():
            if len(papers) < args.min_papers:
                continue
            citations = [c for _, c in papers]
            h = compute_h_index(citations)
            if h > 0:
                h_indices[name] = h

        # Stats
        h_values = list(h_indices.values())
        if h_values:
            max_h = max(h_values)
            avg_h = sum(h_values) / len(h_values)
            authors_h1plus = sum(1 for h in h_values if h >= 1)

            print(f"  Authors with h > 0: {authors_h1plus}")
            print(f"  Max h-index: {max_h}")
            print(f"  Avg h-index: {avg_h:.2f}")

            top = sorted(h_indices.items(), key=lambda x: x[1], reverse=True)[:15]
            print(f"\n  Top 15 authors by h-index (internal):")
            for name, h in top:
                n_papers = len(author_papers[name])
                print(f"    {name}: h={h} ({n_papers} papers in index)")

        if args.dry_run:
            print(f"\nDone (dry run). Elapsed: {time.time() - t0:.1f}s")
            return

        # Step 3: Update ES
        print("\n[3/3] Updating author h-indices in ES...")
        updated = await update_h_indices(http, h_indices, args.batch_size, args.dry_run)

        # Refresh
        await http.post(f"{ES_URL}/{INDEX}/_refresh")

        elapsed = time.time() - t0
        print(f"\n{'=' * 60}")
        print(f"DONE in {elapsed:.1f}s")
        print(f"  Authors with computed h-index: {len(h_indices)}")
        print(f"  Papers updated: {updated}")
        print(f"{'=' * 60}")


if __name__ == "__main__":
    asyncio.run(main())
