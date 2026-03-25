#!/usr/bin/env python3
"""
Full ArXiv OAI-PMH harvest — downloads ALL ~2.5M papers into Elasticsearch.

Strategy:
  - Harvest year-by-year (1986–2026) for granular progress tracking
  - Skip embeddings (use embed_backfill later)
  - Optimize ES settings for bulk throughput
  - Save progress per-year so we can resume on crash
  - 3-second delay between requests (ArXiv rate limit)

Usage:
  python3 scripts/harvest_full.py                # Harvest everything
  python3 scripts/harvest_full.py --start-year 2020  # Start from 2020
  python3 scripts/harvest_full.py --resume       # Resume interrupted harvest
"""
import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone

os.environ.setdefault("ES_HOST", "localhost")
os.environ.setdefault("REDIS_HOST", "localhost")
os.environ.setdefault("API_KEYS", "harvest-key")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.config import reset_settings
reset_settings()

import structlog
from elasticsearch import AsyncElasticsearch

from src.core.config import get_settings
from src.core.elasticsearch import get_es_client, ensure_index, close_es_client
from src.ingestion.worker import run_ingestion_cycle

logger = structlog.get_logger()

PROGRESS_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), ".harvest_progress.json"
)


def load_progress() -> dict:
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {}


def save_progress(progress: dict):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=2)


async def optimize_es_for_bulk():
    """Temporarily tune ES settings for maximum bulk indexing throughput."""
    settings = get_settings()
    es = await get_es_client()
    await ensure_index(es, settings.es_index, settings.embedding_dim)

    try:
        await es.indices.put_settings(
            index=settings.es_index,
            body={
                "index": {
                    "refresh_interval": "60s",
                    "number_of_replicas": 0,
                    "translog.durability": "async",
                    "translog.flush_threshold_size": "1gb",
                }
            }
        )
        logger.info("es_optimized_for_bulk")
    except Exception as e:
        logger.warning("es_optimization_partial", error=str(e))


async def restore_es_settings():
    """Restore ES settings after bulk import."""
    settings = get_settings()
    es = await get_es_client()

    try:
        await es.indices.put_settings(
            index=settings.es_index,
            body={
                "index": {
                    "refresh_interval": "1s",
                    "number_of_replicas": 0,
                    "translog.durability": "request",
                }
            }
        )
        await es.indices.refresh(index=settings.es_index)
        logger.info("es_settings_restored")
    except Exception as e:
        logger.warning("es_restore_error", error=str(e))


async def get_current_count() -> int:
    settings = get_settings()
    es = await get_es_client()
    try:
        resp = await es.count(index=settings.es_index)
        return resp["count"]
    except Exception:
        return 0


async def harvest_full(start_year: int = 1986, resume: bool = False):
    """Harvest all ArXiv papers year by year."""
    progress = load_progress() if resume else {}
    current_year = datetime.now(timezone.utc).year

    # Optimize ES for bulk loading
    await optimize_es_for_bulk()

    initial_count = await get_current_count()
    overall_start = time.monotonic()

    print(f"\n{'='*60}")
    print(f"  FULL ARXIV HARVEST")
    print(f"  Years: {start_year} → {current_year}")
    print(f"  Current papers in DB: {initial_count:,}")
    print(f"  Skip embeddings: YES (run embed_backfill later)")
    print(f"  Progress file: {PROGRESS_FILE}")
    print(f"{'='*60}\n")

    total_harvested = 0

    for year in range(start_year, current_year + 1):
        year_key = str(year)

        # Skip completed years
        if year_key in progress and progress[year_key].get("status") == "completed":
            count = progress[year_key].get("count", 0)
            total_harvested += count
            print(f"  ⏭  {year}: already completed ({count:,} papers)")
            continue

        from_date = f"{year}-01-01"
        until_date = f"{year}-12-31" if year < current_year else datetime.now(timezone.utc).strftime("%Y-%m-%d")

        print(f"\n  📥 {year}: harvesting {from_date} → {until_date} ...")
        year_start = time.monotonic()

        try:
            count = await run_ingestion_cycle(
                from_date=from_date,
                until_date=until_date,
                skip_embeddings=True,
            )
            elapsed = time.monotonic() - year_start

            total_harvested += count
            current_total = await get_current_count()

            progress[year_key] = {
                "status": "completed",
                "count": count,
                "elapsed_s": int(elapsed),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            save_progress(progress)

            print(f"  ✓  {year}: {count:,} papers in {elapsed:.0f}s | DB total: {current_total:,}")

        except Exception as e:
            elapsed = time.monotonic() - year_start
            progress[year_key] = {
                "status": "failed",
                "error": str(e),
                "elapsed_s": int(elapsed),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            save_progress(progress)
            print(f"  ✗  {year}: FAILED after {elapsed:.0f}s — {e}")
            print(f"     Run with --resume to retry")

    # Restore ES settings
    await restore_es_settings()

    # Final stats
    final_count = await get_current_count()
    total_elapsed = time.monotonic() - overall_start

    print(f"\n{'='*60}")
    print(f"  HARVEST COMPLETE")
    print(f"  Papers harvested this run: {total_harvested:,}")
    print(f"  Total papers in database: {final_count:,}")
    print(f"  Total time: {total_elapsed/3600:.1f} hours")
    print(f"  Target: >2,000,000 — {'✓ MET' if final_count >= 2_000_000 else '✗ NOT MET'}")
    print(f"{'='*60}\n")

    await close_es_client()


def main():
    parser = argparse.ArgumentParser(description="Full ArXiv OAI-PMH harvest")
    parser.add_argument("--start-year", type=int, default=1986,
                        help="First year to harvest (default: 1986)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last progress checkpoint")
    args = parser.parse_args()

    asyncio.run(harvest_full(
        start_year=args.start_year,
        resume=args.resume,
    ))


if __name__ == "__main__":
    main()
