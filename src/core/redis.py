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
        kwargs: dict = dict(
            host=settings.redis_host,
            port=settings.redis_port,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5,
        )
        if settings.redis_password:
            kwargs["password"] = settings.redis_password
        _pool = redis.Redis(**kwargs)
    return _pool


async def close_redis_client() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None
