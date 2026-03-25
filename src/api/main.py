from __future__ import annotations

import structlog
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi.responses import JSONResponse

from src.core.config import get_settings
from src.core.elasticsearch import get_es_client, close_es_client, ensure_index
from src.core.redis import get_redis_client, close_redis_client
from src.core.models import (
    HealthResponse,
    SearchRequest,
    SearchResponse,
    StatsResponse,
)
from src.core.search import SearchEngine
from src.core.security import verify_api_key, validate_search_request
from src.core.embeddings import encode_text, get_cached_embedding, cache_embedding

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    es = await get_es_client()
    await ensure_index(es, settings.es_index, settings.embedding_dim)
    await get_redis_client()
    logger.info("app_started")
    yield
    await close_es_client()
    await close_redis_client()
    logger.info("app_stopped")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="ArXiv Search Engine for Agents",
        description="High-performance ArXiv search with semantic similarity, citation analytics, and advanced filters",
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Rate limiting
    limiter = Limiter(key_func=get_remote_address)
    app.state.limiter = limiter

    @app.exception_handler(RateLimitExceeded)
    async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
        return JSONResponse(
            status_code=429,
            content={"detail": "Rate limit exceeded. Please slow down."},
        )

    # CORS
    origins = [o.strip() for o in settings.cors_origins.split(",")]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["X-API-Key", "Content-Type"],
        max_age=3600,
    )

    # Security headers middleware
    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Cache-Control"] = "no-store"
        response.headers["Content-Security-Policy"] = "default-src 'none'"
        return response

    # ── Routes ──

    @app.get("/health", response_model=HealthResponse)
    async def health():
        es_status = "unknown"
        redis_status = "unknown"
        total = 0
        try:
            es = await get_es_client()
            info = await es.cluster.health()
            es_status = info["status"]
            settings = get_settings()
            count = await es.count(index=settings.es_index)
            total = count["count"]
        except Exception:
            es_status = "error"

        try:
            r = await get_redis_client()
            await r.ping()
            redis_status = "ok"
        except Exception:
            redis_status = "error"

        status = "healthy" if es_status in ("green", "yellow") and redis_status == "ok" else "degraded"
        return HealthResponse(
            status=status,
            elasticsearch=es_status,
            redis=redis_status,
            total_papers=total,
        )

    @app.post(
        "/search",
        response_model=SearchResponse,
        dependencies=[Depends(verify_api_key)],
    )
    @limiter.limit(f"{settings.rate_limit_per_minute}/minute")
    async def search(request: Request, body: SearchRequest):
        validate_search_request(body)

        # Handle semantic embedding
        embedding = None
        if body.semantic:
            redis_client = await get_redis_client()
            embedding = await get_cached_embedding(
                redis_client, body.semantic.text, body.semantic.level.value
            )
            if embedding is None:
                embedding = encode_text(body.semantic.text)
                await cache_embedding(
                    redis_client, body.semantic.text, body.semantic.level.value, embedding
                )

        es = await get_es_client()
        settings = get_settings()
        engine = SearchEngine(es, settings.es_index)
        return await engine.search(body, embedding)

    @app.get(
        "/stats",
        response_model=StatsResponse,
        dependencies=[Depends(verify_api_key)],
    )
    @limiter.limit(f"{settings.rate_limit_per_minute}/minute")
    async def stats(request: Request):
        es = await get_es_client()
        settings_obj = get_settings()
        engine = SearchEngine(es, settings_obj.es_index)
        return await engine.get_stats()

    @app.get(
        "/paper/{arxiv_id:path}",
        dependencies=[Depends(verify_api_key)],
    )
    @limiter.limit(f"{settings.rate_limit_per_minute}/minute")
    async def get_paper(request: Request, arxiv_id: str):
        es = await get_es_client()
        settings_obj = get_settings()
        engine = SearchEngine(es, settings_obj.es_index)
        paper = await engine.get_paper(arxiv_id)
        if paper is None:
            raise HTTPException(status_code=404, detail="Paper not found")
        return paper

    return app


app = create_app()
