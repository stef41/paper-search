#!/usr/bin/env python3
"""Continuous enrichment pipeline — run daily via cron or systemd timer.

ALL citation counting is done locally. OpenAlex is used ONLY to discover
which papers a given paper references (the reference list).

Pipeline steps:
  1. Fetch reference lists from OpenAlex (resolve to arxiv IDs)
  2. Self-compute cited_by_ids by inverting the reference graph (zero API calls)
  3. Self-compute author h-indices from internal citation data (zero API calls)
  4. Compute title & abstract embeddings for unembedded papers (zero API calls)

Usage:
  # Full daily run
  python3 scripts/enrich_pipeline.py

  # Only recompute self-derived data (no API calls)
  python3 scripts/enrich_pipeline.py --skip-openalex

  # Custom limits
  python3 scripts/enrich_pipeline.py --new-limit 50000 --email you@example.com

Cron example (daily at 2am):
  0 2 * * * cd /data/users/zacharie/arxiv-agent && python3 scripts/enrich_pipeline.py --email you@example.com >> logs/enrich.log 2>&1
"""
import argparse
import asyncio
import json
import os
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone

import httpx

ES_URL = "http://localhost:9200"
INDEX = "arxiv_papers"
OPENALEX_API = "https://api.openalex.org"
BATCH_SIZE = 50  # papers per OpenAlex request
REQ_DELAY = 0.105  # ~9.5 req/s to stay under 10 req/s

_bulk_errors_total = 0


async def _checked_bulk(http: httpx.AsyncClient, bulk_body: list[str], **kwargs) -> None:
    """Post a bulk request and warn on per-item errors."""
    global _bulk_errors_total
    resp = await http.post(
        f"{ES_URL}/{INDEX}/_bulk",
        content=("\n".join(bulk_body) + "\n").encode(),
        headers={"Content-Type": "application/x-ndjson"},
        **kwargs,
    )
    result = resp.json()
    if result.get("errors"):
        failed = sum(1 for item in result.get("items", [])
                     if "error" in (item.get("update") or item.get("index") or {}))
        _bulk_errors_total += failed
        print(f"    WARNING: {failed} bulk operations failed")
SCROLL_SIZE = 5000
BULK_SIZE = 5000
CACHE_FILE = "data/openalex_id_cache.json"


# ─── Helpers ───

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
) -> None:
    """Resolve uncached OpenAlex work IDs to arxiv IDs, populating _oa_cache."""
    to_resolve = [oid for oid in oa_ids if oid not in _oa_cache]
    if not to_resolve:
        return

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
                    _oa_cache[oa_id] = arxiv_id or ""

                for oid in batch:
                    if oid not in found_ids and oid not in _oa_cache:
                        _oa_cache[oid] = ""
        except Exception as e:
            print(f"    Ref resolution error: {e}")

        await asyncio.sleep(REQ_DELAY)


# ─── Step 1: OpenAlex reference fetching (NO citation counts) ───

async def openalex_fetch_refs_batch(
    http: httpx.AsyncClient, papers: list[dict], email: str
) -> tuple[int, int, int]:
    """Fetch references for a batch. Returns (found, not_found, refs_resolved)."""
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
            print(f"    Request error: {e}")
            await asyncio.sleep(5)
    else:
        return 0, len(papers), 0

    results = r.json().get("results", [])
    oa_by_doi: dict[str, dict] = {}
    for w in results:
        doi = w.get("ids", {}).get("doi", "").replace("https://doi.org/", "")
        oa_by_doi[doi] = w
        # Cache this paper's own mapping
        oa_id = w.get("id", "")
        arxiv_id = doi_to_arxiv(doi)
        if oa_id and arxiv_id:
            _oa_cache[oa_id] = arxiv_id

    # Collect all unique referenced works and resolve uncached ones
    all_ref_ids: set[str] = set()
    for w in results:
        all_ref_ids.update(w.get("referenced_works", []))
    uncached = [rid for rid in all_ref_ids if rid not in _oa_cache]
    if uncached:
        await resolve_openalex_ids(http, uncached, email)

    # Build ES bulk update — ONLY reference_ids, NO citation counts
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

        # Only store reference data — citation counts come from compute_citations.py
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


async def run_openalex(
    http: httpx.AsyncClient, mode: str, limit: int, email: str, stale_days: int
) -> tuple[int, int, int]:
    """Run OpenAlex reference fetching for a given mode.
    Returns (found, not_found, refs_resolved)."""
    if mode == "new":
        query = {
            "query": {"bool": {"must_not": [{"exists": {"field": "enrichment_source"}}]}},
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
    elif mode == "retry_not_found":
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
        return 0, 0, 0

    total_found = 0
    total_not_found = 0
    total_refs = 0
    processed = 0

    while processed < limit:
        query["size"] = min(limit - processed, 10000)
        r = await http.post(f"{ES_URL}/{INDEX}/_search", json=query)
        if r.status_code != 200:
            print(f"    ES search failed (status {r.status_code}): {str(r.text)[:300]}")
            break
        hits = r.json().get("hits", {}).get("hits", [])
        if not hits:
            break

        papers = [{"arxiv_id": h["_source"]["arxiv_id"], "_id": h["_id"]} for h in hits]

        for i in range(0, len(papers), BATCH_SIZE):
            if processed >= limit:
                break
            batch = papers[i:i + BATCH_SIZE]
            found, not_found, refs = await openalex_fetch_refs_batch(http, batch, email)
            total_found += found
            total_not_found += not_found
            total_refs += refs
            processed += len(batch)

            if processed % 2000 == 0:
                print(f"    [{mode}] {processed:,} done: {total_found:,} found, {total_refs:,} refs, cache={len(_oa_cache):,}")

            await asyncio.sleep(REQ_DELAY)

        await http.post(f"{ES_URL}/{INDEX}/_refresh")

        if len(hits) < query["size"]:
            break

    return total_found, total_not_found, total_refs


# ─── Step 2: Self-compute citations (zero API calls) ───

async def compute_citations(http: httpx.AsyncClient) -> tuple[int, int]:
    """Invert reference graph to build cited_by_ids + citation counts.
    Returns (updated, external)."""
    cited_by: dict[str, list[str]] = defaultdict(list)
    total_papers = 0

    query = {
        "query": {"exists": {"field": "reference_ids"}},
        "size": SCROLL_SIZE,
        "_source": ["arxiv_id", "reference_ids"],
    }
    r = await http.post(f"{ES_URL}/{INDEX}/_search?scroll=5m", json=query)
    data = r.json()
    if r.status_code != 200 or "hits" not in data:
        print(f"    ES search failed (status {r.status_code}): {str(data)[:300]}")
        return 0, 0
    scroll_id = data.get("_scroll_id")
    hits = data["hits"]["hits"]

    try:
        while hits:
            for h in hits:
                src = h["_source"]
                total_papers += 1
                for ref_id in (src.get("reference_ids") or []):
                    cited_by[ref_id].append(src["arxiv_id"])

            r = await http.post(f"{ES_URL}/_search/scroll", json={"scroll": "5m", "scroll_id": scroll_id})
            data = r.json()
            if r.status_code != 200 or "hits" not in data:
                print(f"    ES scroll failed (status {r.status_code}): {str(data)[:300]}")
                break
            scroll_id = data.get("_scroll_id")
            hits = data["hits"]["hits"]
    finally:
        if scroll_id:
            await http.request("DELETE", f"{ES_URL}/_search/scroll", json={"scroll_id": scroll_id})

    print(f"    Built citation graph: {total_papers:,} papers with refs → {len(cited_by):,} cited papers")

    if not cited_by:
        return 0, 0

    # Resolve arxiv_ids to doc _ids
    all_ids = list(cited_by.keys())
    doc_map: dict[str, str] = {}
    for i in range(0, len(all_ids), 1000):
        batch = all_ids[i:i + 1000]
        r = await http.post(f"{ES_URL}/{INDEX}/_search", json={
            "query": {"terms": {"arxiv_id": batch}}, "size": len(batch), "_source": ["arxiv_id"]
        })
        if r.status_code != 200:
            print(f"    ES resolve failed (status {r.status_code}): {str(r.text)[:200]}")
            continue
        for h in r.json()["hits"]["hits"]:
            doc_map[h["_source"]["arxiv_id"]] = h["_id"]

    # Bulk update
    updated = 0
    external = 0
    bulk_body: list[str] = []

    for arxiv_id, citers in cited_by.items():
        doc_id = doc_map.get(arxiv_id)
        if not doc_id:
            external += 1
            continue
        bulk_body.append(json.dumps({"update": {"_id": doc_id}}))
        bulk_body.append(json.dumps({"doc": {
            "cited_by_ids": citers,
            "citation_stats": {"total_citations": len(citers)},
        }}))
        updated += 1

        if len(bulk_body) >= BULK_SIZE * 2:
            await _checked_bulk(http, bulk_body, timeout=120)
            bulk_body = []

    if bulk_body:
        await _checked_bulk(http, bulk_body, timeout=120)

    return updated, external


# ─── Step 3: Self-compute h-index (zero API calls) ───

def _h_index(citations: list[int]) -> int:
    sorted_c = sorted(citations, reverse=True)
    h = 0
    for i, c in enumerate(sorted_c):
        if c >= i + 1:
            h = i + 1
        else:
            break
    return h


async def compute_hindex(http: httpx.AsyncClient) -> tuple[int, int]:
    """Compute h-index for all authors from internal citations.
    Returns (authors, papers_updated)."""
    author_citations: dict[str, list[int]] = defaultdict(list)

    query = {
        "query": {"range": {"citation_stats.total_citations": {"gt": 0}}},
        "size": SCROLL_SIZE,
        "_source": ["arxiv_id", "authors", "citation_stats.total_citations"],
    }
    r = await http.post(f"{ES_URL}/{INDEX}/_search?scroll=5m", json=query)
    data = r.json()
    if r.status_code != 200 or "hits" not in data:
        print(f"    ES search failed (status {r.status_code}): {str(data)[:300]}")
        return 0, 0
    scroll_id = data.get("_scroll_id")
    hits = data["hits"]["hits"]

    try:
        while hits:
            for h in hits:
                src = h["_source"]
                cites = (src.get("citation_stats") or {}).get("total_citations", 0) or 0
                for a in (src.get("authors") or []):
                    name = a.get("name", "").strip() if isinstance(a, dict) else str(a).strip()
                    if name:
                        author_citations[name].append(cites)

            r = await http.post(f"{ES_URL}/_search/scroll", json={"scroll": "5m", "scroll_id": scroll_id})
            data = r.json()
            if r.status_code != 200 or "hits" not in data:
                print(f"    ES scroll failed (status {r.status_code}): {str(data)[:300]}")
                break
            scroll_id = data.get("_scroll_id")
            hits = data["hits"]["hits"]
    finally:
        if scroll_id:
            await http.request("DELETE", f"{ES_URL}/_search/scroll", json={"scroll_id": scroll_id})
        await http.request("DELETE", f"{ES_URL}/_search/scroll", json={"scroll_id": scroll_id})

    # Compute h-index per author
    h_indices: dict[str, int] = {}
    for name, cites in author_citations.items():
        h = _h_index(cites)
        if h > 0:
            h_indices[name] = h

    if not h_indices:
        return 0, 0

    # Scan ALL papers and update authors with h_index
    updated = 0
    query = {"query": {"match_all": {}}, "size": SCROLL_SIZE, "_source": ["authors", "first_author"]}
    r = await http.post(f"{ES_URL}/{INDEX}/_search?scroll=5m", json=query)
    data = r.json()
    if r.status_code != 200 or "hits" not in data:
        print(f"    ES search failed (status {r.status_code}): {str(data)[:300]}")
        return len(h_indices), 0
    scroll_id = data.get("_scroll_id")
    hits = data["hits"]["hits"]
    bulk_body: list[str] = []

    try:
        while hits:
            for h in hits:
                src = h["_source"]
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
                if first_author:
                    h_val = h_indices.get(first_author.strip())
                    if h_val is not None:
                        doc["first_author_h_index"] = h_val

                bulk_body.append(json.dumps({"update": {"_id": h["_id"]}}))
                bulk_body.append(json.dumps({"doc": doc}))
                updated += 1

                if len(bulk_body) >= BULK_SIZE * 2:
                    await _checked_bulk(http, bulk_body, timeout=120)
                    bulk_body = []

            r = await http.post(f"{ES_URL}/_search/scroll", json={"scroll": "5m", "scroll_id": scroll_id})
            data = r.json()
            if r.status_code != 200 or "hits" not in data:
                print(f"    ES scroll failed (status {r.status_code}): {str(data)[:300]}")
                break
            scroll_id = data.get("_scroll_id")
            hits = data["hits"]["hits"]
    finally:
        if scroll_id:
            await http.request("DELETE", f"{ES_URL}/_search/scroll", json={"scroll_id": scroll_id})

    if bulk_body:
        await _checked_bulk(http, bulk_body, timeout=120)

    return len(h_indices), updated


# ─── Main orchestrator ───

async def main():
    parser = argparse.ArgumentParser(
        description="Enrichment pipeline — citation counts are 100%% local, OpenAlex used ONLY for reference lists")
    parser.add_argument("--new-limit", type=int, default=10000,
                        help="Max new papers to fetch references for (default: 10000)")
    parser.add_argument("--stale-days", type=int, default=30,
                        help="Re-fetch references older than N days (default: 30)")
    parser.add_argument("--stale-limit", type=int, default=5000,
                        help="Max stale papers to refresh per run (default: 5000)")
    parser.add_argument("--retry-limit", type=int, default=2000,
                        help="Max previously-not-found papers to retry (default: 2000)")
    parser.add_argument("--email", type=str, default="",
                        help="Email for OpenAlex polite pool")
    parser.add_argument("--skip-openalex", action="store_true",
                        help="Skip OpenAlex API calls, only run local computation")
    parser.add_argument("--skip-refresh", action="store_true",
                        help="Skip stale refresh, only fetch for new papers")
    parser.add_argument("--skip-recompute", action="store_true",
                        help="Skip local computation (citations + h-index)")
    parser.add_argument("--skip-embeddings", action="store_true",
                        help="Skip embedding computation")
    parser.add_argument("--embed-batch-size", type=int, default=256,
                        help="Embedding encode batch size (default: 256)")
    args = parser.parse_args()

    t0 = time.time()
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    print("=" * 70)
    print(f"ENRICHMENT PIPELINE — {now}")
    print("  Citation counts: 100% local (inverted reference graph)")
    print("  OpenAlex: reference lists only (no external counts)")
    print("=" * 70)

    load_cache()

    async with httpx.AsyncClient(timeout=60) as http:
        # ── Current state ──
        r = await http.post(f"{ES_URL}/{INDEX}/_search", json={
            "size": 0,
            "aggs": {
                "unenriched": {"filter": {"bool": {"must_not": [{"exists": {"field": "enrichment_source"}}]}}},
                "enriched": {"filter": {"term": {"enrichment_source": "openalex"}}},
                "not_found": {"filter": {"term": {"enrichment_source": "openalex_not_found"}}},
                "has_refs": {"filter": {"exists": {"field": "reference_ids"}}},
                "has_cited": {"filter": {"exists": {"field": "cited_by_ids"}}},
                "has_hindex": {"filter": {"exists": {"field": "first_author_h_index"}}},
                "cites_gt0": {"filter": {"range": {"citation_stats.total_citations": {"gt": 0}}}},
                "has_embeddings": {"filter": {"exists": {"field": "title_embedding"}}},
            }
        })
        a = r.json()["aggregations"]
        print(f"\nCurrent state:")
        print(f"  Unenriched:     {a['unenriched']['doc_count']:>10,}")
        print(f"  Enriched (OA):  {a['enriched']['doc_count']:>10,}")
        print(f"  Not found (OA): {a['not_found']['doc_count']:>10,}")
        print(f"  Has refs:       {a['has_refs']['doc_count']:>10,}")
        print(f"  Has cited_by:   {a['has_cited']['doc_count']:>10,}")
        print(f"  Has h-index:    {a['has_hindex']['doc_count']:>10,}")
        print(f"  Citations > 0:  {a['cites_gt0']['doc_count']:>10,}")
        print(f"  Has embeddings: {a['has_embeddings']['doc_count']:>10,}")

        # ── Step 1: Fetch reference lists from OpenAlex ──
        if not args.skip_openalex:
            print(f"\n{'─'*70}")
            print(f"STEP 1: Fetch references for new papers (limit: {args.new_limit:,})")
            print(f"  (OpenAlex provides reference lists ONLY — no citation counts)")
            print(f"{'─'*70}")
            new_found, new_nf, new_refs = await run_openalex(
                http, "new", args.new_limit, args.email, args.stale_days
            )
            print(f"  → {new_found:,} found, {new_nf:,} not on OA, {new_refs:,} refs resolved")

            if not args.skip_refresh:
                print(f"\n{'─'*70}")
                print(f"STEP 1b: Refresh stale references (>{args.stale_days} days, limit: {args.stale_limit:,})")
                print(f"{'─'*70}")
                stale_found, stale_nf, stale_refs = await run_openalex(
                    http, "stale", args.stale_limit, args.email, args.stale_days
                )
                print(f"  → {stale_found:,} refreshed, {stale_refs:,} refs resolved")

                print(f"\n{'─'*70}")
                print(f"STEP 1c: Retry not-found papers (>{args.stale_days} days, limit: {args.retry_limit:,})")
                print(f"{'─'*70}")
                retry_found, retry_nf, retry_refs = await run_openalex(
                    http, "retry_not_found", args.retry_limit, args.email, args.stale_days
                )
                print(f"  → {retry_found:,} newly found, {retry_refs:,} refs resolved")

        save_cache()

        # ── Step 2: Self-compute citation counts (zero API calls) ──
        if not args.skip_recompute:
            print(f"\n{'─'*70}")
            print(f"STEP 2: Self-compute cited_by_ids from reference graph (local only)")
            print(f"{'─'*70}")
            cite_updated, cite_external = await compute_citations(http)
            print(f"  → {cite_updated:,} papers with citation counts, {cite_external:,} external refs")

            await http.post(f"{ES_URL}/{INDEX}/_refresh")

            # ── Step 3: Self-compute h-index (zero API calls) ──
            print(f"\n{'─'*70}")
            print(f"STEP 3: Self-compute author h-indices (local only)")
            print(f"{'─'*70}")
            h_authors, h_papers = await compute_hindex(http)
            print(f"  → {h_authors:,} authors, {h_papers:,} papers updated")

        # ── Step 4: Compute embeddings for unembedded papers ──
        if not args.skip_embeddings:
            print(f"\n{'─'*70}")
            print(f"STEP 4: Compute title + abstract embeddings (local, GPU)")
            print(f"{'─'*70}")
            import subprocess
            embed_cmd = [
                "python3", "-u", "scripts/compute_embeddings.py",
                "--batch-size", str(args.embed_batch_size),
            ]
            result = subprocess.run(embed_cmd, capture_output=True, text=True)
            print(result.stdout)
            if result.returncode != 0:
                print(f"  Embedding error: {result.stderr[-500:] if result.stderr else 'unknown'}")

        # ── Final report ──
        await http.post(f"{ES_URL}/{INDEX}/_refresh")

        r = await http.post(f"{ES_URL}/{INDEX}/_search", json={
            "size": 0,
            "aggs": {
                "unenriched": {"filter": {"bool": {"must_not": [{"exists": {"field": "enrichment_source"}}]}}},
                "enriched": {"filter": {"term": {"enrichment_source": "openalex"}}},
                "not_found": {"filter": {"term": {"enrichment_source": "openalex_not_found"}}},
                "has_refs": {"filter": {"exists": {"field": "reference_ids"}}},
                "has_cited": {"filter": {"exists": {"field": "cited_by_ids"}}},
                "has_hindex": {"filter": {"exists": {"field": "first_author_h_index"}}},
                "cites_gt0": {"filter": {"range": {"citation_stats.total_citations": {"gt": 0}}}},
                "has_embeddings": {"filter": {"exists": {"field": "title_embedding"}}},
            }
        })
        a = r.json()["aggregations"]

        elapsed = time.time() - t0
        print(f"\n{'='*70}")
        print(f"PIPELINE COMPLETE — {elapsed:.0f}s")
        print(f"  All citation counts are 100% self-computed from internal data")
        print(f"{'='*70}")
        print(f"  Unenriched:     {a['unenriched']['doc_count']:>10,}")
        print(f"  Enriched (OA):  {a['enriched']['doc_count']:>10,}")
        print(f"  Not found (OA): {a['not_found']['doc_count']:>10,}")
        print(f"  Has refs:       {a['has_refs']['doc_count']:>10,}")
        print(f"  Has cited_by:   {a['has_cited']['doc_count']:>10,}")
        print(f"  Has h-index:    {a['has_hindex']['doc_count']:>10,}")
        print(f"  Citations > 0:  {a['cites_gt0']['doc_count']:>10,}")
        print(f"  Has embeddings: {a['has_embeddings']['doc_count']:>10,}")
        print(f"  OA→arxiv cache: {len(_oa_cache):>10,}")
        print(f"{'='*70}")


if __name__ == "__main__":
    asyncio.run(main())
