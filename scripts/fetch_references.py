#!/usr/bin/env python3
"""Fetch reference lists from OpenAlex. ALL citation counting is done locally.

OpenAlex is used ONLY to answer: "which papers does paper X reference?"
The answer is stored as reference_ids (arxiv IDs of referenced papers).

ALL citation metrics are computed locally by compute_citations.py and
compute_hindex.py — never imported from external sources.

Flow:
  1. This script: OpenAlex → reference_ids (which papers does X cite?)
  2. compute_citations.py: invert graph → cited_by_ids + citation counts
  3. compute_hindex.py: group by author → h-index

Usage:
  python3 scripts/fetch_references.py [max_papers] [--email you@example.com]
  python3 scripts/fetch_references.py 10000 --mode unenriched
  python3 scripts/fetch_references.py 5000 --mode stale --stale-days 30
"""
import argparse
import asyncio
import json
import os
import time
from datetime import datetime, timedelta, timezone

import httpx

ES_URL = "http://localhost:9200"
INDEX = "arxiv_papers"
OPENALEX_API = "https://api.openalex.org"
BATCH_SIZE = 50  # papers per OpenAlex DOI query
REQ_DELAY = 0.105  # ~9.5 req/s
CACHE_FILE = "data/openalex_id_cache.json"  # local cache: OpenAlex ID → arxiv ID


async def _checked_bulk(http: httpx.AsyncClient, bulk_body: list[str]) -> None:
    """Post a bulk request and warn on per-item errors."""
    resp = await http.post(
        f"{ES_URL}/{INDEX}/_bulk",
        content=("\n".join(bulk_body) + "\n").encode(),
        headers={"Content-Type": "application/x-ndjson"},
    )
    result = resp.json()
    if result.get("errors"):
        failed = sum(1 for item in result.get("items", [])
                     if "error" in (item.get("update") or item.get("index") or {}))
        print(f"    WARNING: {failed} bulk operations failed")


def arxiv_to_doi(arxiv_id: str) -> str:
    return f"10.48550/arxiv.{arxiv_id}"


def doi_to_arxiv(doi: str) -> str | None:
    if "10.48550/arxiv." in doi:
        return doi.split("10.48550/arxiv.")[1]
    return None


# ─── Local OpenAlex ID → arxiv ID cache ───

_oa_cache: dict[str, str] = {}


def load_cache() -> None:
    global _oa_cache
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE) as f:
            _oa_cache = json.load(f)
        print(f"  Loaded {len(_oa_cache):,} cached OpenAlex→arxiv mappings")


def save_cache() -> None:
    os.makedirs(os.path.dirname(CACHE_FILE) or ".", exist_ok=True)
    with open(CACHE_FILE, "w") as f:
        json.dump(_oa_cache, f)
    print(f"  Saved {len(_oa_cache):,} cached OpenAlex→arxiv mappings")


async def resolve_openalex_ids(
    http: httpx.AsyncClient, oa_ids: list[str], email: str
) -> dict[str, str]:
    """Resolve OpenAlex work IDs to arxiv IDs. Uses local cache first."""
    result: dict[str, str] = {}
    to_resolve: list[str] = []

    for oid in oa_ids:
        cached = _oa_cache.get(oid)
        if cached is not None:
            if cached:  # non-empty = has arxiv ID
                result[oid] = cached
        else:
            to_resolve.append(oid)

    if not to_resolve:
        return result

    # Batch-query OpenAlex for uncached IDs
    for i in range(0, len(to_resolve), BATCH_SIZE):
        batch = to_resolve[i:i + BATCH_SIZE]
        short_ids = [oid.replace("https://openalex.org/", "") for oid in batch]
        filter_val = "|".join(short_ids)

        params = {
            "filter": f"openalex:{filter_val}",
            "select": "id,ids",
            "per_page": "200",
        }
        if email:
            params["mailto"] = email

        try:
            r = await http.get(f"{OPENALEX_API}/works", params=params)
            if r.status_code == 429:
                wait = min(int(r.headers.get("Retry-After", 60)), 120)
                print(f"    Rate limited, waiting {wait}s...")
                await asyncio.sleep(wait)
                r = await http.get(f"{OPENALEX_API}/works", params=params)

            if r.status_code == 200:
                found_ids = set()
                for w in r.json().get("results", []):
                    oa_id = w["id"]
                    doi = w.get("ids", {}).get("doi", "")
                    arxiv_id = doi_to_arxiv(doi.replace("https://doi.org/", ""))
                    found_ids.add(oa_id)
                    if arxiv_id:
                        result[oa_id] = arxiv_id
                        _oa_cache[oa_id] = arxiv_id
                    else:
                        _oa_cache[oa_id] = ""  # cache negative: no arxiv DOI

                # Mark IDs not returned by OpenAlex as empty (not found)
                for oid in batch:
                    if oid not in found_ids and oid not in _oa_cache:
                        _oa_cache[oid] = ""

        except Exception as e:
            print(f"    Ref resolution error: {e}")

        await asyncio.sleep(REQ_DELAY)

    return result


async def fetch_papers(http: httpx.AsyncClient, mode: str, limit: int, stale_days: int) -> list[dict]:
    """Fetch papers needing reference enrichment."""
    if mode == "unenriched":
        # Match papers with no enrichment_source, OR papers enriched by
        # Semantic Scholar (enrich.py) that still lack reference_ids —
        # S2 enrichment sets citation stats but not reference links.
        query = {
            "query": {"bool": {"should": [
                {"bool": {"must_not": [{"exists": {"field": "enrichment_source"}}]}},
                {"bool": {
                    "must": [{"term": {"enrichment_source": "semantic_scholar"}}],
                    "must_not": [{"exists": {"field": "reference_ids"}}],
                }},
            ], "minimum_should_match": 1}},
            "size": min(limit, 10000),
            "sort": [{"submitted_date": {"order": "desc"}}],
            "_source": ["arxiv_id"],
        }
    elif mode == "stale":
        cutoff = (datetime.now(timezone.utc) - timedelta(days=stale_days)).strftime("%Y-%m-%dT%H:%M:%SZ")
        query = {
            "query": {
                "bool": {
                    "must": [{"term": {"enrichment_source": "openalex"}}],
                    "should": [
                        {"range": {"enriched_at": {"lt": cutoff}}},
                        {"bool": {"must_not": [{"exists": {"field": "enriched_at"}}]}},
                    ],
                    "minimum_should_match": 1,
                }
            },
            "size": min(limit, 10000),
            "sort": [{"enriched_at": {"order": "asc", "missing": "_first"}}],
            "_source": ["arxiv_id"],
        }
    elif mode == "retry":
        cutoff = (datetime.now(timezone.utc) - timedelta(days=stale_days)).strftime("%Y-%m-%dT%H:%M:%SZ")
        query = {
            "query": {
                "bool": {
                    "must": [{"term": {"enrichment_source": "openalex_not_found"}}],
                    "should": [
                        {"range": {"enriched_at": {"lt": cutoff}}},
                        {"bool": {"must_not": [{"exists": {"field": "enriched_at"}}]}},
                    ],
                    "minimum_should_match": 1,
                }
            },
            "size": min(limit, 10000),
            "sort": [{"enriched_at": {"order": "asc", "missing": "_first"}}],
            "_source": ["arxiv_id"],
        }
    else:
        return []

    r = await http.post(f"{ES_URL}/{INDEX}/_search", json=query)
    if r.status_code != 200:
        print(f"ES search failed (status {r.status_code}): {str(r.text)[:300]}")
        return []
    data = r.json()
    hits = data.get("hits", {}).get("hits", [])
    return [{"arxiv_id": h["_source"]["arxiv_id"], "_id": h["_id"]} for h in hits]


async def process_batch(
    http: httpx.AsyncClient, papers: list[dict], email: str
) -> tuple[int, int, int]:
    """Process one batch: get references from OpenAlex, resolve to arxiv IDs.

    Returns (found_on_oa, not_found_on_oa, refs_resolved).
    """
    doi_filter = "|".join(arxiv_to_doi(p["arxiv_id"]) for p in papers)
    params = {
        "filter": f"doi:{doi_filter}",
        "select": "id,referenced_works,referenced_works_count,ids",
        "per_page": "200",
    }
    if email:
        params["mailto"] = email

    for attempt in range(3):
        try:
            r = await http.get(f"{OPENALEX_API}/works", params=params)
            if r.status_code == 200:
                break
            if r.status_code == 429:
                wait = min(int(r.headers.get("Retry-After", 60)), 120)
                print(f"    Rate limited, waiting {wait}s...")
                await asyncio.sleep(wait)
                continue
            await asyncio.sleep(5)
        except Exception as e:
            print(f"    Error: {e}")
            await asyncio.sleep(5)
    else:
        return 0, len(papers), 0

    results = r.json().get("results", [])
    oa_by_doi: dict[str, dict] = {}
    for w in results:
        doi = w.get("ids", {}).get("doi", "").replace("https://doi.org/", "")
        oa_by_doi[doi] = w
        # Cache this paper's own OpenAlex ID → arxiv ID mapping
        oa_id = w.get("id", "")
        arxiv_id = doi_to_arxiv(doi)
        if oa_id and arxiv_id:
            _oa_cache[oa_id] = arxiv_id

    # Collect ALL unique referenced OpenAlex IDs across this batch
    all_ref_ids: set[str] = set()
    for w in results:
        all_ref_ids.update(w.get("referenced_works", []))

    # Resolve them to arxiv IDs (using cache + API)
    uncached = [rid for rid in all_ref_ids if rid not in _oa_cache]
    if uncached:
        await resolve_openalex_ids(http, uncached, email)

    # Build ES bulk update
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    bulk_body: list[str] = []
    found = 0
    total_refs = 0

    for p in papers:
        doi = arxiv_to_doi(p["arxiv_id"])
        oa = oa_by_doi.get(doi)

        if oa is None:
            bulk_body.append(json.dumps({"update": {"_id": p["_id"]}}))
            bulk_body.append(json.dumps({"doc": {
                "enrichment_source": "openalex_not_found",
                "enriched_at": now_str,
            }}))
            continue

        # Resolve references to arxiv IDs
        ref_arxiv_ids: list[str] = []
        for ref_oa_id in oa.get("referenced_works", []):
            cached = _oa_cache.get(ref_oa_id, "")
            if cached:
                ref_arxiv_ids.append(cached)

        ref_count = oa.get("referenced_works_count", 0) or 0

        doc: dict = {
            "enrichment_source": "openalex",
            "enriched_at": now_str,
            "references_stats": {"total_references": ref_count},
        }
        if ref_arxiv_ids:
            doc["reference_ids"] = ref_arxiv_ids

        bulk_body.append(json.dumps({"update": {"_id": p["_id"]}}))
        bulk_body.append(json.dumps({"doc": doc}))
        found += 1
        total_refs += len(ref_arxiv_ids)

    if bulk_body:
        await _checked_bulk(http, bulk_body)

    return found, len(papers) - found, total_refs


async def main():
    parser = argparse.ArgumentParser(description="Fetch reference lists from OpenAlex (NO citation counts)")
    parser.add_argument("max_papers", type=int, nargs="?", default=10000,
                        help="Max papers to process (default: 10000)")
    parser.add_argument("--email", type=str, default="",
                        help="Email for OpenAlex polite pool")
    parser.add_argument("--mode", choices=["unenriched", "stale", "retry"],
                        default="unenriched",
                        help="Which papers to process")
    parser.add_argument("--stale-days", type=int, default=30,
                        help="For --mode stale: refresh references older than N days")
    args = parser.parse_args()

    t0 = time.time()
    print("=" * 60)
    print("Fetch References from OpenAlex")
    print("  (citation counts are computed locally, NOT from OpenAlex)")
    print(f"  Target: {args.max_papers:,} papers, mode={args.mode}")
    if args.mode == "stale":
        print(f"  Stale threshold: {args.stale_days} days")
    print("=" * 60)

    load_cache()

    total_found = 0
    total_not_found = 0
    total_refs = 0
    processed = 0

    async with httpx.AsyncClient(timeout=60) as http:
        while processed < args.max_papers:
            papers = await fetch_papers(http, args.mode, args.max_papers - processed, args.stale_days)
            if not papers:
                break

            print(f"\nFetched {len(papers)} papers to process")

            for i in range(0, len(papers), BATCH_SIZE):
                batch = papers[i:i + BATCH_SIZE]
                batch_num = (processed + i) // BATCH_SIZE + 1

                found, not_found, refs = await process_batch(http, batch, args.email)
                total_found += found
                total_not_found += not_found
                total_refs += refs
                processed += len(batch)

                if batch_num % 20 == 0:
                    elapsed = time.time() - t0
                    print(f"  [{processed:,} processed] found={total_found:,} not_found={total_not_found:,} refs={total_refs:,} cache={len(_oa_cache):,} ({elapsed:.0f}s)")

                await asyncio.sleep(REQ_DELAY)

            await http.post(f"{ES_URL}/{INDEX}/_refresh")

    save_cache()

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"DONE in {elapsed:.0f}s")
    print(f"  Processed:      {processed:,}")
    print(f"  Found on OA:    {total_found:,}")
    print(f"  Not found:      {total_not_found:,}")
    print(f"  Refs resolved:  {total_refs:,}")
    print(f"  Cache size:     {len(_oa_cache):,}")
    print(f"{'='*60}")
    print()
    print("Next steps:")
    print("  python3 scripts/compute_citations.py   # invert graph → citation counts")
    print("  python3 scripts/compute_hindex.py       # compute author h-indices")


if __name__ == "__main__":
    asyncio.run(main())
