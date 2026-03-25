from __future__ import annotations

import redis.asyncio as redis
import structlog

from src.core.config import get_settings

logger = structlog.get_logger()

_pool: redis.Redis | None = None


async def get_redis_client() -> redis.Redis:
    global _pool
    if _pool is None:
        settings = get_settings()
        _pool = redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5,
        )
    return _pool


async def close_redis_client() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None
