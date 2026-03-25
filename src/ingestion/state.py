"""Harvest state persistence — tracks ingestion progress in Elasticsearch."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import structlog
from elasticsearch import AsyncElasticsearch

logger = structlog.get_logger()

STATE_INDEX = ".arxiv_harvest_state"


async def ensure_state_index(client: AsyncElasticsearch) -> None:
    if not await client.indices.exists(index=STATE_INDEX):
        await client.indices.create(
            index=STATE_INDEX,
            body={
                "settings": {"number_of_shards": 1, "number_of_replicas": 0},
                "mappings": {
                    "properties": {
                        "source": {"type": "keyword"},
                        "last_harvested_date": {"type": "date"},
                        "resumption_token": {"type": "keyword"},
                        "total_harvested": {"type": "long"},
                        "last_run": {"type": "date"},
                        "status": {"type": "keyword"},
                        "metadata": {"type": "object", "enabled": False},
                    }
                },
            },
        )


async def get_state(client: AsyncElasticsearch, source: str) -> dict | None:
    await ensure_state_index(client)
    try:
        resp = await client.get(index=STATE_INDEX, id=source)
        return resp["_source"]
    except Exception:
        return None


async def save_state(
    client: AsyncElasticsearch,
    source: str,
    *,
    last_harvested_date: str | None = None,
    resumption_token: str | None = None,
    total_harvested: int = 0,
    status: str = "running",
    metadata: dict | None = None,
) -> None:
    await ensure_state_index(client)
    doc: dict[str, Any] = {
        "source": source,
        "last_run": datetime.now(timezone.utc).isoformat(),
        "total_harvested": total_harvested,
        "status": status,
    }
    if last_harvested_date:
        doc["last_harvested_date"] = last_harvested_date
    if resumption_token is not None:
        doc["resumption_token"] = resumption_token
    if metadata:
        doc["metadata"] = metadata

    await client.index(index=STATE_INDEX, id=source, document=doc)
