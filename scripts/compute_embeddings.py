#!/usr/bin/env python3
"""Compute title + abstract embeddings for all papers.

Uses sentence-transformers/all-MiniLM-L6-v2 (384-dim, cosine).
Designed to run incrementally: only embeds papers that don't have
title_embedding yet. Safe to re-run — existing embeddings are skipped.

Usage:
  python3 scripts/compute_embeddings.py                    # all unembedded
  python3 scripts/compute_embeddings.py --max-papers 100000
  python3 scripts/compute_embeddings.py --batch-size 512   # GPU-tuned
"""
import argparse
import json
import time

import httpx
import torch
from sentence_transformers import SentenceTransformer

ES_URL = "http://localhost:9200"
INDEX = "arxiv_papers"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SCROLL_SIZE = 500
BULK_SIZE = 500


def main():
    parser = argparse.ArgumentParser(description="Compute embeddings for papers")
    parser.add_argument("--max-papers", type=int, default=0,
                        help="Max papers to embed (0 = all)")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Encoding batch size (default: 256, GPU can handle 512+)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model {MODEL_NAME} on {device}...")
    model = SentenceTransformer(MODEL_NAME, device=device)

    t0 = time.time()

    # Count unembedded papers
    with httpx.Client(timeout=60) as http:
        r = http.post(f"{ES_URL}/{INDEX}/_count", json={
            "query": {"bool": {"must_not": [{"exists": {"field": "title_embedding"}}]}}
        })
        count_data = r.json()
        if r.status_code != 200 or "count" not in count_data:
            print(f"ES count failed (status {r.status_code}): {str(count_data)[:300]}")
            return
        unembedded = count_data["count"]

    target = unembedded if args.max_papers == 0 else min(args.max_papers, unembedded)
    print(f"Papers without embeddings: {unembedded:,}")
    print(f"Target: {target:,}")
    print(f"Batch size: {args.batch_size}")
    print("=" * 60)

    total = 0
    scroll_id = None

    with httpx.Client(timeout=120) as http:
        # Initial search
        r = http.post(f"{ES_URL}/{INDEX}/_search?scroll=10m", json={
            "query": {"bool": {"must_not": [{"exists": {"field": "title_embedding"}}]}},
            "size": SCROLL_SIZE,
            "_source": ["arxiv_id", "title", "abstract"],
            "sort": [{"_doc": "asc"}],
        })
        data = r.json()
        if r.status_code != 200 or "hits" not in data:
            print(f"ES search failed (status {r.status_code}): {str(data)[:300]}")
            return
        scroll_id = data.get("_scroll_id")
        hits = data["hits"]["hits"]

        try:
            while hits:
                if args.max_papers > 0 and total >= args.max_papers:
                    break

                # Extract texts
                doc_ids = []
                titles = []
                abstracts = []
                for h in hits:
                    src = h["_source"]
                    doc_ids.append(h["_id"])
                    titles.append(src.get("title", "") or "")
                    abstracts.append(src.get("abstract", "") or "")

                # Encode in batches
                title_embs = model.encode(
                    titles, batch_size=args.batch_size,
                    normalize_embeddings=True, show_progress_bar=False,
                )
                abstract_embs = model.encode(
                    abstracts, batch_size=args.batch_size,
                    normalize_embeddings=True, show_progress_bar=False,
                )

                # Bulk update to ES
                bulk_lines: list[str] = []
                for i, doc_id in enumerate(doc_ids):
                    bulk_lines.append(json.dumps({"update": {"_id": doc_id}}))
                    doc = {
                        "title_embedding": title_embs[i].tolist(),
                        "abstract_embedding": abstract_embs[i].tolist(),
                    }
                    bulk_lines.append(json.dumps({"doc": doc}))

                    if len(bulk_lines) >= BULK_SIZE * 2:
                        br = http.post(
                            f"{ES_URL}/{INDEX}/_bulk",
                            content=("\n".join(bulk_lines) + "\n").encode(),
                            headers={"Content-Type": "application/x-ndjson"},
                        )
                        if br.status_code != 200:
                            print(f"  WARNING: ES bulk returned {br.status_code}")
                        else:
                            br_json = br.json()
                            if br_json.get("errors"):
                                errs = sum(1 for it in br_json.get("items", []) if "error" in it.get("update", {}))
                                print(f"  WARNING: {errs} bulk update errors")
                        bulk_lines = []

                if bulk_lines:
                    br = http.post(
                        f"{ES_URL}/{INDEX}/_bulk",
                        content=("\n".join(bulk_lines) + "\n").encode(),
                        headers={"Content-Type": "application/x-ndjson"},
                    )
                    if br.status_code != 200:
                        print(f"  WARNING: ES bulk returned {br.status_code}")
                    else:
                        br_json = br.json()
                        if br_json.get("errors"):
                            errs = sum(1 for it in br_json.get("items", []) if "error" in it.get("update", {}))
                            print(f"  WARNING: {errs} bulk update errors")

                total += len(hits)
                elapsed = time.time() - t0
                rate = total / elapsed if elapsed > 0 else 0
                eta = (target - total) / rate if rate > 0 else 0

                if total % 5000 < SCROLL_SIZE:
                    print(f"  [{total:>10,}/{target:,}] {rate:.0f} papers/s, "
                          f"elapsed {elapsed:.0f}s, ETA {eta:.0f}s")

                # Next scroll page
                r = http.post(f"{ES_URL}/_search/scroll", json={
                    "scroll": "10m", "scroll_id": scroll_id,
                })
                data = r.json()
                if r.status_code != 200 or "hits" not in data:
                    print(f"ES scroll failed (status {r.status_code}): {str(data)[:300]}")
                    break
                scroll_id = data.get("_scroll_id")
                hits = data.get("hits", {}).get("hits", [])

        finally:
            # Cleanup
            if scroll_id:
                http.request("DELETE", f"{ES_URL}/_search/scroll",
                             json={"scroll_id": scroll_id})

        http.post(f"{ES_URL}/{INDEX}/_refresh")

    elapsed = time.time() - t0
    rate = total / elapsed if elapsed > 0 else 0
    print(f"\n{'='*60}")
    print(f"DONE: {total:,} papers embedded in {elapsed:.0f}s ({rate:.0f}/s)")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Device: {device}")
    print(f"  Fields: title_embedding (384d), abstract_embedding (384d)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
