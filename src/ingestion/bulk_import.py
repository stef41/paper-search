"""
Kaggle ArXiv bulk importer — import from the Cornell ArXiv JSON snapshot.

The dataset is at: https://www.kaggle.com/datasets/Cornell-University/arxiv
Download the arxiv-metadata-oai-snapshot.json file (~4GB) and mount it.

Usage:
    python -m src.ingestion.bulk_import --file /data/arxiv-metadata-oai-snapshot.json
    python -m src.ingestion.bulk_import --file /data/arxiv-metadata-oai-snapshot.json --skip-embeddings
    python -m src.ingestion.bulk_import --file /data/arxiv-metadata-oai-snapshot.json --max-papers 100000
    python -m src.ingestion.bulk_import --file /data/arxiv-metadata-oai-snapshot.json --categories cs.AI cs.LG
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog
from elasticsearch import AsyncElasticsearch
from elasticsearch.helpers import async_bulk

from src.core.config import get_settings
from src.core.elasticsearch import get_es_client, ensure_index, close_es_client
from src.core.embeddings import encode_texts
from src.ingestion.state import save_state, get_state

logger = structlog.get_logger()

GITHUB_RE = re.compile(r"https?://github\.com/[\w\-\.]+/[\w\-\.]+")


def parse_kaggle_record(record: dict) -> dict | None:
    """Parse a single JSON record from the Kaggle ArXiv snapshot."""
    arxiv_id = record.get("id", "")
    if not arxiv_id:
        return None

    title = record.get("title", "").strip()
    title = re.sub(r"\s+", " ", title)

    abstract = record.get("abstract", "").strip()
    abstract = re.sub(r"\s+", " ", abstract)

    if not title or not abstract:
        return None

    # Categories
    categories_str = record.get("categories", "")
    categories = categories_str.split() if categories_str else []
    primary_category = categories[0] if categories else None

    # Authors — Kaggle format: "LastName, FirstName and LastName, FirstName"
    # or parsed authors list
    authors = []
    authors_parsed = record.get("authors_parsed", [])
    if authors_parsed:
        for i, parts in enumerate(authors_parsed):
            if isinstance(parts, list) and len(parts) >= 2:
                name = f"{parts[1].strip()} {parts[0].strip()}".strip()
            elif isinstance(parts, list) and len(parts) == 1:
                name = parts[0].strip()
            else:
                continue
            if name:
                authors.append({
                    "name": name,
                    "is_first_author": i == 0,
                    "h_index": None,
                    "citation_count": None,
                })
    else:
        # Fallback: parse from authors string
        # Split on ' and ' only — the ',\s*(?=[A-Z])' split incorrectly
        # fragments "Last, First" name pairs into separate entries.
        authors_str = record.get("authors", "")
        if authors_str:
            for i, name in enumerate(re.split(r"\s+and\s+", authors_str)):
                name = name.strip().rstrip(",")
                if name:
                    authors.append({
                        "name": name,
                        "is_first_author": i == 0,
                        "h_index": None,
                        "citation_count": None,
                    })

    # Dates
    versions = record.get("versions", [])
    submitted_date = None
    updated_date = None

    if versions:
        # First version = submitted, last version = updated
        first_v = versions[0].get("created", "")
        last_v = versions[-1].get("created", "")
        if first_v:
            try:
                submitted_date = _parse_date(first_v)
            except ValueError:
                pass
        if last_v:
            try:
                updated_date = _parse_date(last_v)
            except ValueError:
                pass

    if not submitted_date:
        update_date_str = record.get("update_date", "")
        if update_date_str:
            try:
                submitted_date = datetime.fromisoformat(update_date_str).replace(
                    tzinfo=timezone.utc
                ).isoformat()
            except ValueError:
                pass

    # DOI and journal ref
    doi = record.get("doi", None) or None
    journal_ref = record.get("journal-ref", None) or None
    comments = record.get("comments", None) or None

    # Page count
    page_count = None
    if comments:
        m = re.search(r"(\d+)\s*pages?", comments, re.IGNORECASE)
        if m:
            page_count = int(m.group(1))

    # GitHub URLs
    github_urls = list(set(
        GITHUB_RE.findall(abstract or "") +
        GITHUB_RE.findall(comments or "")
    ))

    first_author = authors[0]["name"] if authors else None

    return {
        "arxiv_id": arxiv_id,
        "title": title,
        "abstract": abstract,
        "authors": authors,
        "categories": categories,
        "domains": list(set(c.split(".")[0] for c in categories)),
        "primary_category": primary_category,
        "submitted_date": submitted_date,
        "updated_date": updated_date or submitted_date,
        "published_date": submitted_date,
        "doi": doi,
        "journal_ref": journal_ref,
        "comments": comments,
        "page_count": page_count,
        "has_github": len(github_urls) > 0,
        "github_urls": github_urls,
        "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}",
        "abstract_url": f"https://arxiv.org/abs/{arxiv_id}",
        "first_author": first_author,
        "first_author_h_index": None,
        "citation_stats": {
            "total_citations": 0,
            "avg_citation_age_years": None,
            "median_h_index_citing_authors": None,
            "top_citing_categories": [],
        },
        "references_stats": {
            "total_references": 0,
            "avg_reference_age_years": None,
            "top_referenced_categories": [],
        },
    }


def _parse_date(date_str: str) -> str:
    """Parse various date formats from ArXiv data."""
    # Try multiple formats
    for fmt in [
        "%a, %d %b %Y %H:%M:%S %Z",
        "%Y-%m-%d",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%SZ",
    ]:
        try:
            dt = datetime.strptime(date_str.strip(), fmt)
            return dt.replace(tzinfo=timezone.utc).isoformat()
        except ValueError:
            continue

    # Fallback: try isoformat
    dt = datetime.fromisoformat(date_str.strip())
    if dt.tzinfo is not None:
        return dt.astimezone(timezone.utc).isoformat()
    return dt.replace(tzinfo=timezone.utc).isoformat()


def stream_kaggle_file(
    filepath: str,
    filter_categories: list[str] | None = None,
    max_papers: int | None = None,
    skip_ids: set | None = None,
) -> Any:
    """Stream and parse the Kaggle JSON file line by line."""
    count = 0
    skipped = 0

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if max_papers and count >= max_papers:
                break

            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue

            # Category filter
            if filter_categories:
                cats = record.get("categories", "").split()
                if not any(c in filter_categories for c in cats):
                    continue

            # Skip already-indexed
            if skip_ids and record.get("id") in skip_ids:
                continue

            try:
                parsed = parse_kaggle_record(record)
            except Exception:
                skipped += 1
                continue
            if parsed:
                count += 1
                yield parsed

    logger.info("stream_complete", total=count, skipped=skipped)


async def run_bulk_import(
    filepath: str,
    filter_categories: list[str] | None = None,
    max_papers: int | None = None,
    skip_embeddings: bool = False,
    batch_size: int = 500,
) -> int:
    """Import papers from Kaggle JSON snapshot."""
    settings = get_settings()
    es = await get_es_client()
    await ensure_index(es, settings.es_index, settings.embedding_dim)

    logger.info(
        "bulk_import_start",
        file=filepath,
        categories=filter_categories,
        max_papers=max_papers,
        skip_embeddings=skip_embeddings,
    )

    total_indexed = 0
    batch: list[dict] = []
    start_time = time.monotonic()

    for paper in stream_kaggle_file(filepath, filter_categories, max_papers):
        batch.append(paper)

        if len(batch) >= batch_size:
            if not skip_embeddings:
                _add_embeddings(batch)

            count = await _index_batch(es, settings.es_index, batch)
            total_indexed += count

            elapsed = time.monotonic() - start_time
            rate = total_indexed / elapsed if elapsed > 0 else 0
            logger.info(
                "bulk_progress",
                indexed=total_indexed,
                rate=f"{rate:.0f}/s",
                elapsed=f"{elapsed:.0f}s",
            )

            batch = []

    # Index remaining
    if batch:
        if not skip_embeddings:
            _add_embeddings(batch)
        count = await _index_batch(es, settings.es_index, batch)
        total_indexed += count

    await es.indices.refresh(index=settings.es_index)

    elapsed = time.monotonic() - start_time
    rate = total_indexed / elapsed if elapsed > 0 else 0

    await save_state(
        es, "bulk_import",
        total_harvested=total_indexed,
        status="completed",
        metadata={"file": filepath, "rate": f"{rate:.0f}/s", "elapsed": f"{elapsed:.0f}s"},
    )

    logger.info(
        "bulk_import_complete",
        total=total_indexed,
        elapsed=f"{elapsed:.0f}s",
        rate=f"{rate:.0f}/s",
    )
    return total_indexed


def _add_embeddings(papers: list[dict]) -> None:
    """Add embeddings to a batch of papers in-place."""
    titles = [p["title"] for p in papers]
    abstracts = [p["abstract"] for p in papers]

    title_embs = encode_texts(titles, batch_size=64)
    abstract_embs = encode_texts(abstracts, batch_size=64)

    for i, paper in enumerate(papers):
        if i < len(title_embs):
            paper["title_embedding"] = title_embs[i]
        if i < len(abstract_embs):
            paper["abstract_embedding"] = abstract_embs[i]


async def _index_batch(client: AsyncElasticsearch, index: str, papers: list[dict]) -> int:
    """Bulk index papers."""
    actions = [
        {"_index": index, "_id": p["arxiv_id"], "_source": p}
        for p in papers
    ]
    success, errors = await async_bulk(client, actions, raise_on_error=False)
    if errors:
        logger.warning("bulk_errors", count=len(errors))
    return success


def main():
    parser = argparse.ArgumentParser(
        description="Bulk import ArXiv papers from Kaggle JSON snapshot"
    )
    parser.add_argument(
        "--file", required=True,
        help="Path to arxiv-metadata-oai-snapshot.json",
    )
    parser.add_argument(
        "--categories", nargs="+", default=None,
        help="Filter to these categories only",
    )
    parser.add_argument(
        "--max-papers", type=int, default=None,
        help="Maximum papers to import",
    )
    parser.add_argument(
        "--skip-embeddings", action="store_true",
        help="Skip embedding generation",
    )
    parser.add_argument(
        "--batch-size", type=int, default=500,
        help="Batch size for indexing (default: 500)",
    )
    args = parser.parse_args()

    if not Path(args.file).exists():
        logger.error("file_not_found", file=args.file)
        print(f"Error: File not found: {args.file}")
        print("Download from: https://www.kaggle.com/datasets/Cornell-University/arxiv")
        return

    async def _run():
        try:
            await run_bulk_import(
                filepath=args.file,
                filter_categories=args.categories,
                max_papers=args.max_papers,
                skip_embeddings=args.skip_embeddings,
                batch_size=args.batch_size,
            )
        finally:
            await close_es_client()

    asyncio.run(_run())


if __name__ == "__main__":
    main()
