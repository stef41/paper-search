"""
Backfill embeddings for papers that were imported without them.

Usage:
    python -m src.ingestion.embed_backfill
    python -m src.ingestion.embed_backfill --batch-size 100 --max-papers 50000
"""
from __future__ import annotations

import argparse
import asyncio
import time

import structlog

from src.core.config import get_settings
from src.core.elasticsearch import get_es_client, ensure_index, close_es_client
from src.core.embeddings import encode_texts

logger = structlog.get_logger()


async def backfill_embeddings(
    batch_size: int = 100,
    max_papers: int | None = None,
) -> int:
    """Generate embeddings for papers that don't have them."""
    settings = get_settings()
    es = await get_es_client()

    query = {
        "query": {
            "bool": {
                "must_not": [
                    {"exists": {"field": "title_embedding"}},
                ],
            }
        },
        "size": batch_size,
        "_source": ["arxiv_id", "title", "abstract"],
        "sort": [{"_doc": "asc"}],
    }

    total = 0
    start_time = time.monotonic()
    search_after = None

    while True:
        if max_papers and total >= max_papers:
            break

        body = dict(query)
        if search_after:
            body["search_after"] = search_after

        resp = await es.search(index=settings.es_index, body=body)
        hits = resp["hits"]["hits"]

        if not hits:
            break

        titles = [h["_source"]["title"] for h in hits]
        abstracts = [h["_source"]["abstract"] for h in hits]
        ids = [h["_source"]["arxiv_id"] for h in hits]

        title_embs = encode_texts(titles, batch_size=64)
        abstract_embs = encode_texts(abstracts, batch_size=64)

        # Bulk update
        ops = []
        for i, aid in enumerate(ids):
            ops.append({"update": {"_index": settings.es_index, "_id": aid}})
            doc = {}
            if i < len(title_embs):
                doc["title_embedding"] = title_embs[i]
            if i < len(abstract_embs):
                doc["abstract_embedding"] = abstract_embs[i]
            ops.append({"doc": doc})

        if ops:
            bulk_resp = await es.bulk(body=ops)
            if bulk_resp.get("errors"):
                failed = sum(1 for item in bulk_resp.get("items", []) if "error" in item.get("update", {}))
                logger.warning("bulk_embed_errors", failed=failed, batch=len(ids))

        total += len(hits)
        elapsed = time.monotonic() - start_time
        rate = total / elapsed if elapsed > 0 else 0

        logger.info(
            "embed_progress",
            total=total,
            rate=f"{rate:.0f}/s",
            elapsed=f"{elapsed:.0f}s",
        )

        search_after = hits[-1]["sort"]

    elapsed = time.monotonic() - start_time
    logger.info("embed_backfill_complete", total=total, elapsed=f"{elapsed:.0f}s")
    return total


async def _main(batch_size: int, max_papers: int | None) -> None:
    try:
        await backfill_embeddings(batch_size=batch_size, max_papers=max_papers)
    finally:
        await close_es_client()


def main():
    parser = argparse.ArgumentParser(description="Backfill embeddings for papers")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--max-papers", type=int, default=None)
    args = parser.parse_args()

    asyncio.run(_main(
        batch_size=args.batch_size,
        max_papers=args.max_papers,
    ))


if __name__ == "__main__":
    main()
