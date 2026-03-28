from __future__ import annotations

import argparse
import asyncio
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Any
from xml.etree.ElementTree import fromstring

import httpx
import structlog
from elasticsearch import AsyncElasticsearch
from elasticsearch.helpers import async_bulk
from tenacity import retry, stop_after_attempt, wait_exponential

from src.core.config import get_settings
from src.core.elasticsearch import get_es_client, ensure_index, close_es_client
from src.core.embeddings import encode_texts
from src.ingestion.state import save_state, get_state

logger = structlog.get_logger()

OAI_NAMESPACES = {
    "oai": "http://www.openarchives.org/OAI/2.0/",
    "dc": "http://purl.org/dc/elements/1.1/",
    "arx": "http://arxiv.org/OAI/arXivRaw/",
}

GITHUB_RE = re.compile(
    r"https?://github\.com/[\w\-\.]+/[\w\-\.]+"
)


def extract_github_urls(text: str) -> list[str]:
    if not text:
        return []
    return list(set(GITHUB_RE.findall(text)))


def estimate_page_count(comments: str | None) -> int | None:
    if not comments:
        return None
    match = re.search(r"(\d+)\s*pages?", comments, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def parse_oai_record(record_xml: Any) -> dict | None:
    header = record_xml.find("oai:header", OAI_NAMESPACES)
    if header is None:
        return None

    status = header.get("status", "")
    if status == "deleted":
        return None

    identifier = header.findtext("oai:identifier", "", OAI_NAMESPACES)
    arxiv_id = identifier.replace("oai:arXiv.org:", "")

    metadata = record_xml.find("oai:metadata", OAI_NAMESPACES)
    if metadata is None:
        return None

    arx = metadata.find("arx:arXivRaw", OAI_NAMESPACES)
    if arx is None:
        return None

    title = arx.findtext("arx:title", "", OAI_NAMESPACES).strip()
    abstract = arx.findtext("arx:abstract", "", OAI_NAMESPACES).strip()
    categories_str = arx.findtext("arx:categories", "", OAI_NAMESPACES).strip()
    categories = categories_str.split() if categories_str else []
    comments = arx.findtext("arx:comments", None, OAI_NAMESPACES)
    doi = arx.findtext("arx:doi", None, OAI_NAMESPACES)
    journal_ref = arx.findtext("arx:journal-ref", None, OAI_NAMESPACES)

    # Parse dates — prefer version dates from arXivRaw metadata over OAI datestamp
    submitted_date = None
    updated_date = None

    versions = arx.findall("arx:version", OAI_NAMESPACES)
    if versions:
        # v1 date = original submission date
        v1_date_str = versions[0].findtext("arx:date", "", OAI_NAMESPACES)
        if v1_date_str:
            try:
                submitted_date = datetime.strptime(
                    v1_date_str.strip(), "%a, %d %b %Y %H:%M:%S %Z"
                ).replace(tzinfo=timezone.utc)
            except ValueError:
                pass
        # Latest version date = updated date
        if len(versions) > 1:
            last_date_str = versions[-1].findtext("arx:date", "", OAI_NAMESPACES)
            if last_date_str:
                try:
                    updated_date = datetime.strptime(
                        last_date_str.strip(), "%a, %d %b %Y %H:%M:%S %Z"
                    ).replace(tzinfo=timezone.utc)
                except ValueError:
                    pass

    # Fallback to OAI datestamp if version dates unavailable
    if submitted_date is None:
        datestamp = header.findtext("oai:datestamp", "", OAI_NAMESPACES)
        if datestamp:
            try:
                submitted_date = datetime.fromisoformat(datestamp).replace(tzinfo=timezone.utc)
            except ValueError:
                pass
    if updated_date is None:
        updated_date = submitted_date

    # Parse authors
    authors_str = arx.findtext("arx:authors", "", OAI_NAMESPACES)
    author_list = []
    if authors_str:
        for i, name in enumerate(authors_str.split(",")):
            name = name.strip()
            if name:
                author_list.append({
                    "name": name,
                    "is_first_author": i == 0,
                    "h_index": None,
                    "citation_count": None,
                })

    # GitHub detection
    github_urls = extract_github_urls(abstract or "")
    if comments:
        github_urls.extend(extract_github_urls(comments))
    github_urls = list(set(github_urls))

    page_count = estimate_page_count(comments)

    first_author = author_list[0]["name"] if author_list else None

    return {
        "arxiv_id": arxiv_id,
        "title": title,
        "abstract": abstract,
        "authors": author_list,
        "categories": categories,
        "primary_category": categories[0] if categories else None,
        "submitted_date": submitted_date.isoformat() if submitted_date else None,
        "updated_date": updated_date.isoformat() if updated_date else None,
        "published_date": None,
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


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2, min=4, max=60))
async def fetch_oai_page(
    client: httpx.AsyncClient, base_url: str, params: dict
) -> tuple[list[dict], str | None]:
    resp = await client.get(base_url, params=params, timeout=120)
    resp.raise_for_status()

    root = fromstring(resp.content)
    records_el = root.findall(".//oai:record", OAI_NAMESPACES)

    papers = []
    for rec in records_el:
        parsed = parse_oai_record(rec)
        if parsed:
            papers.append(parsed)

    # Check for resumption token
    token_el = root.find(".//oai:resumptionToken", OAI_NAMESPACES)
    token = None
    if token_el is not None and token_el.text:
        token = token_el.text.strip()

    return papers, token


def add_embeddings(papers: list[dict]) -> list[dict]:
    if not papers:
        return papers

    titles = [p["title"] for p in papers]
    abstracts = [p["abstract"] for p in papers]

    title_embs = encode_texts(titles)
    abstract_embs = encode_texts(abstracts)

    for i, paper in enumerate(papers):
        paper["title_embedding"] = title_embs[i]
        paper["abstract_embedding"] = abstract_embs[i]

    return papers


async def index_batch(
    client: AsyncElasticsearch, index: str, papers: list[dict]
) -> int:
    actions = []
    for paper in papers:
        actions.append({
            "_index": index,
            "_id": paper["arxiv_id"],
            "_source": paper,
        })

    success, errors = await async_bulk(client, actions, raise_on_error=False)
    if errors:
        logger.warning("bulk_index_errors", count=len(errors))
    return success


async def run_ingestion_cycle(
    from_date: str | None = None,
    until_date: str | None = None,
    skip_embeddings: bool = False,
    set_spec: str | None = None,
):
    """Run one OAI-PMH harvest cycle with state tracking.

    Args:
        from_date: Harvest records from this date (YYYY-MM-DD). Auto-detected from state if None.
        until_date: Harvest records until this date (YYYY-MM-DD). Defaults to today.
        skip_embeddings: Skip embedding generation for faster indexing.
        set_spec: OAI set to harvest (e.g. "cs" for all CS). None = all.
    """
    settings = get_settings()
    es = await get_es_client()
    await ensure_index(es, settings.es_index, settings.embedding_dim)

    # Determine date range from state
    source_key = f"oai:{set_spec or 'all'}"
    state = await get_state(es, source_key)

    if from_date is None and state and state.get("last_harvested_date"):
        from_date = state["last_harvested_date"]
        logger.info("resuming_from_state", from_date=from_date)

    # Check for resumption token in state
    resumption_token = None
    if state and state.get("resumption_token") and state.get("status") == "interrupted":
        resumption_token = state["resumption_token"]
        logger.info("resuming_with_token", token=resumption_token[:50])

    if until_date is None:
        until_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    logger.info(
        "ingestion_cycle_start",
        from_date=from_date,
        until_date=until_date,
        set_spec=set_spec,
        has_resume_token=resumption_token is not None,
    )

    total_indexed = 0
    start_time = time.monotonic()
    current_date = from_date  # Track latest seen date

    async with httpx.AsyncClient(follow_redirects=True) as http:
        # Build initial params
        if resumption_token:
            params = {
                "verb": "ListRecords",
                "resumptionToken": resumption_token,
            }
        else:
            params: dict[str, str] = {
                "verb": "ListRecords",
                "metadataPrefix": "arXivRaw",
            }
            if from_date:
                params["from"] = from_date
            if until_date:
                params["until"] = until_date
            if set_spec:
                params["set"] = set_spec

        page_count = 0

        while True:
            try:
                papers, token = await fetch_oai_page(
                    http, settings.arxiv_oai_base_url, params
                )
            except Exception as e:
                logger.error("oai_fetch_error", error=str(e), page=page_count)
                # Save interrupted state so we can resume
                await save_state(
                    es, source_key,
                    last_harvested_date=current_date,
                    resumption_token=params.get("resumptionToken"),
                    total_harvested=total_indexed,
                    status="interrupted",
                )
                raise

            page_count += 1

            if papers:
                # Track latest date seen
                for p in papers:
                    if p.get("submitted_date") and (not current_date or p["submitted_date"] > current_date):
                        current_date = p["submitted_date"][:10]

                if not skip_embeddings:
                    papers = add_embeddings(papers)

                count = await index_batch(es, settings.es_index, papers)
                total_indexed += count

                elapsed = time.monotonic() - start_time
                rate = total_indexed / elapsed if elapsed > 0 else 0

                logger.info(
                    "indexed_batch",
                    count=count,
                    total=total_indexed,
                    page=page_count,
                    rate=f"{rate:.0f}/s",
                    has_more=token is not None,
                )

            # Save progress periodically (every 10 pages)
            if page_count % 10 == 0:
                await save_state(
                    es, source_key,
                    last_harvested_date=current_date,
                    resumption_token=token,
                    total_harvested=total_indexed,
                    status="running",
                )

            if not token:
                break

            params = {
                "verb": "ListRecords",
                "resumptionToken": token,
            }

            # Respect ArXiv rate limits (3 seconds between requests)
            await asyncio.sleep(3)

    # Refresh to make docs searchable
    await es.indices.refresh(index=settings.es_index)

    elapsed = time.monotonic() - start_time
    rate = total_indexed / elapsed if elapsed > 0 else 0

    await save_state(
        es, source_key,
        last_harvested_date=until_date,
        resumption_token=None,
        total_harvested=total_indexed,
        status="completed",
    )

    logger.info(
        "ingestion_cycle_complete",
        total=total_indexed,
        pages=page_count,
        elapsed=f"{elapsed:.0f}s",
        rate=f"{rate:.0f}/s",
    )
    return total_indexed


async def main():
    """Continuous harvesting loop with automatic incremental updates."""
    settings = get_settings()
    logger.info("ingestion_worker_start", interval_hours=settings.ingestion_interval_hours)

    while True:
        try:
            await run_ingestion_cycle()
        except Exception:
            logger.exception("ingestion_cycle_error")

        logger.info("sleeping", hours=settings.ingestion_interval_hours)
        await asyncio.sleep(settings.ingestion_interval_hours * 3600)


def cli_main():
    """CLI entry point for manual harvesting."""
    parser = argparse.ArgumentParser(description="ArXiv OAI-PMH harvester")
    parser.add_argument("--from-date", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--until-date", help="End date (YYYY-MM-DD)")
    parser.add_argument("--set", dest="set_spec", help="OAI set (e.g. 'cs')")
    parser.add_argument("--skip-embeddings", action="store_true")
    parser.add_argument("--once", action="store_true", help="Run once, don't loop")
    args = parser.parse_args()

    if args.once:
        asyncio.run(run_ingestion_cycle(
            from_date=args.from_date,
            until_date=args.until_date,
            skip_embeddings=args.skip_embeddings,
            set_spec=args.set_spec,
        ))
    else:
        asyncio.run(main())


if __name__ == "__main__":
    cli_main()
