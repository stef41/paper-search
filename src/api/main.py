from __future__ import annotations

import asyncio
import re
import time
import structlog
from collections import defaultdict
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi.responses import JSONResponse
from starlette.responses import Response


def _get_client_ip(request: Request) -> str:
    """Get the real client IP.

    When behind a trusted reverse proxy (nginx, Cloudflare, ALB, etc.),
    reads X-Forwarded-For. Otherwise uses the raw socket IP.
    Set TRUSTED_PROXIES env var to a comma-separated list of proxy IPs
    (e.g. "127.0.0.1,10.0.0.0/8") to enable proxy header trust.
    Without TRUSTED_PROXIES, X-Forwarded-For is ignored (safe default).
    """
    import ipaddress

    raw_ip = request.client.host if request.client else "unknown"

    trusted = get_settings().trusted_proxies
    if not trusted:
        return raw_ip

    # Check if the direct connection is from a trusted proxy
    trusted_nets = []
    for cidr in trusted.split(","):
        cidr = cidr.strip()
        if not cidr:
            continue
        try:
            trusted_nets.append(ipaddress.ip_network(cidr, strict=False))
        except ValueError:
            continue

    try:
        client_addr = ipaddress.ip_address(raw_ip)
    except ValueError:
        return raw_ip

    # Handle IPv4-mapped IPv6 (e.g. ::ffff:127.0.0.1 from dual-stack)
    if isinstance(client_addr, ipaddress.IPv6Address) and client_addr.ipv4_mapped:
        client_addr = client_addr.ipv4_mapped

    is_trusted = any(client_addr in net for net in trusted_nets)
    if not is_trusted:
        return raw_ip

    # Connection is from a trusted proxy — read the forwarded header.
    # Walk right-to-left, skipping trusted proxies, to find the real client IP.
    forwarded = request.headers.get("X-Forwarded-For", "")
    if forwarded:
        ips = [ip.strip() for ip in forwarded.split(",")]
        for ip in reversed(ips):
            try:
                addr = ipaddress.ip_address(ip)
                if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped:
                    addr = addr.ipv4_mapped
            except ValueError:
                continue
            if not any(addr in net for net in trusted_nets):
                return ip
        # All entries are trusted proxies or invalid — use raw socket IP
        return raw_ip

    return raw_ip

from src.core.config import get_settings
from src.core.elasticsearch import get_es_client, close_es_client, ensure_index
from src.core.redis import get_redis_client, close_redis_client
from src.core.models import (
    HealthResponse,
    SearchRequest,
    SearchResponse,
    StatsResponse,
    GraphSearchRequest,
    GraphResponse,
    SemanticQuery,
)
from src.core.search import SearchEngine
from src.core.graph import GraphEngine
from src.core.security import verify_api_key, validate_search_request
from src.core.embeddings import encode_text, get_cached_embedding, cache_embedding

logger = structlog.get_logger()


async def _resolve_embeddings(
    sem_list: list[SemanticQuery] | None,
) -> list[tuple[SemanticQuery, list[float]]]:
    """Compute (or cache-lookup) embeddings for each SemanticQuery."""
    if not sem_list:
        return []
    from src.core.redis import get_redis_client
    redis_client = await get_redis_client()
    result: list[tuple[SemanticQuery, list[float]]] = []
    for sq in sem_list:
        if not sq.text or not sq.text.strip():
            continue
        emb = None
        try:
            emb = await get_cached_embedding(redis_client, sq.text, sq.level.value)
        except Exception:
            pass  # Redis down — fall through to compute
        if emb is None:
            emb = await asyncio.get_running_loop().run_in_executor(None, encode_text, sq.text)
            try:
                await cache_embedding(redis_client, sq.text, sq.level.value, emb)
            except Exception:
                pass  # cache write failure is non-fatal
        result.append((sq, emb))
    return result


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    if not settings.api_key_list or any(k == "changeme-key-1" for k in settings.api_key_list):
        logger.warning("SECURITY: using default or empty API key — set API_KEYS env var before deploying")
    es = await get_es_client()
    try:
        await ensure_index(es, settings.es_index, settings.embedding_dim)
        await get_redis_client()
        logger.info("app_started")
        yield
    finally:
        try:
            await close_es_client()
        except Exception:
            logger.error("failed_to_close_es")
        try:
            await close_redis_client()
        except Exception:
            logger.error("failed_to_close_redis")
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

    # Rate limiting (uses raw socket IP, not X-Forwarded-For)
    limiter = Limiter(key_func=_get_client_ip)
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

    # Security headers middleware (registered before ddos_guard so it wraps it —
    # Starlette reverses registration order, making earlier = outer)
    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Cache-Control"] = "no-store"
        response.headers["Content-Security-Policy"] = (
            "default-src 'none'" if request.url.path not in ("/docs", "/redoc", "/openapi.json")
            else "default-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; img-src 'self' https://fastapi.tiangolo.com data:; font-src 'self' data:"
        )
        return response

    # ── IP-level DDoS rate limiter (registered after security_headers so it's inner) ──
    _ddos_limit, _ddos_window_str = settings.ddos_rate_limit.split("/")
    _ddos_max = int(_ddos_limit)
    _window_map = {"second": 1, "minute": 60, "hour": 3600}
    if _ddos_window_str not in _window_map:
        raise ValueError(f"Invalid ddos_rate_limit window '{_ddos_window_str}'. Must be one of: {list(_window_map)}")
    _ddos_window = _window_map[_ddos_window_str]
    _ip_hits: dict[str, list[float]] = defaultdict(list)
    _ip_last_sweep: float = 0.0

    @app.middleware("http")
    async def ddos_guard(request: Request, call_next):
        nonlocal _ip_last_sweep
        client_ip = _get_client_ip(request)
        now = time.monotonic()
        cutoff = now - _ddos_window

        # Periodic sweep: evict stale IPs every 60 seconds
        if now - _ip_last_sweep > 60:
            stale = [ip for ip, h in _ip_hits.items() if not h or h[-1] < cutoff]
            for ip in stale:
                del _ip_hits[ip]
            # Emergency cap: prevent unbounded growth under distributed attacks
            if len(_ip_hits) > 100_000:
                sorted_ips = sorted(_ip_hits, key=lambda ip: _ip_hits[ip][-1] if _ip_hits[ip] else 0)
                for ip in sorted_ips[:len(sorted_ips) // 2]:
                    del _ip_hits[ip]
            _ip_last_sweep = now

        # Prune old entries and append current
        hits = _ip_hits[client_ip]
        # Fast prune: find first entry within window
        lo = 0
        while lo < len(hits) and hits[lo] < cutoff:
            lo += 1
        if lo:
            del hits[:lo]

        if len(hits) >= _ddos_max:
            logger.warning("ddos_blocked", ip=client_ip, hits=len(hits), window=_ddos_window)
            resp = Response(
                content='{"detail":"Too many requests from this IP. Slow down."}',
                status_code=429,
                media_type="application/json",
            )
            resp.headers["X-Content-Type-Options"] = "nosniff"
            resp.headers["X-Frame-Options"] = "DENY"
            resp.headers["Cache-Control"] = "no-store"
            return resp

        hits.append(now)
        response = await call_next(request)
        return response

    # ── Generic JSON error handler ──

    @app.exception_handler(Exception)
    async def _unhandled_exception_handler(request: Request, exc: Exception):
        logger.error("unhandled_exception", path=request.url.path, error=str(exc))
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"},
        )

    # ── Routes ──

    @app.get("/health")
    async def health():
        es_status = "unknown"
        redis_status = "unknown"
        try:
            es = await get_es_client()
            info = await es.cluster.health()
            es_status = info["status"]
        except Exception:
            es_status = "error"

        try:
            r = await get_redis_client()
            await r.ping()
            redis_status = "ok"
        except Exception:
            redis_status = "error"

        status = "healthy" if es_status in ("green", "yellow") and redis_status == "ok" else "degraded"
        code = 200 if status == "healthy" else 503
        return JSONResponse(content={"status": status}, status_code=code)

    @app.post(
        "/search",
        response_model=SearchResponse,
        dependencies=[Depends(verify_api_key)],
    )
    @limiter.limit(f"{settings.rate_limit_per_minute}/minute")
    async def search(request: Request, body: SearchRequest):
        validate_search_request(body)

        # Handle semantic embeddings (single or list)
        sem_list = body.semantic if isinstance(body.semantic, list) else ([body.semantic] if body.semantic else None)
        try:
            embeddings = await _resolve_embeddings(sem_list)
        except Exception as e:
            logger.error("embedding_generation_failed", error=str(e))
            raise HTTPException(status_code=503, detail="Semantic search temporarily unavailable")

        es = await get_es_client()
        settings = get_settings()
        engine = SearchEngine(es, settings.es_index)
        return await engine.search(body, embeddings=embeddings)

    @app.post(
        "/graph",
        response_model=GraphResponse,
        dependencies=[Depends(verify_api_key)],
    )
    @limiter.limit(f"{settings.rate_limit_per_minute}/minute")
    async def graph(request: Request, body: GraphSearchRequest):
        # Validate any search filters present
        sr = body.to_search_request()
        validate_search_request(sr)

        # Handle semantic embeddings (single or list)
        sem_list = body.semantic if isinstance(body.semantic, list) else ([body.semantic] if body.semantic else None)
        try:
            embeddings = await _resolve_embeddings(sem_list)
        except Exception as e:
            logger.error("embedding_generation_failed", error=str(e))
            raise HTTPException(status_code=503, detail="Semantic search temporarily unavailable")

        es = await get_es_client()
        settings_obj = get_settings()
        engine = GraphEngine(es, settings_obj.es_index)
        try:
            result = await asyncio.wait_for(
                engine.execute(body.graph, sr, embeddings=embeddings),
                timeout=120,
            )
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="Graph query timed out")

        # Surface clear input-validation / not-found errors as proper HTTP codes.
        # "no papers found", "need at least N papers", etc. are data-insufficiency
        # responses, not client errors — leave those as 200 with metadata.
        err = result.metadata.get("error", "")
        if "required" in err or "same paper" in err or "max " in err:
            raise HTTPException(status_code=400, detail=err)
        if "not found" in err or "not in subgraph" in err:
            raise HTTPException(status_code=404, detail=err)

        return result

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
        # Validate arxiv_id format to prevent path traversal / abuse
        if not re.match(r'^(\d{4}\.\d{4,5}(v\d+)?|[a-zA-Z][a-zA-Z0-9._-]*/\d{7}(v\d+)?)$', arxiv_id):
            raise HTTPException(status_code=400, detail="Invalid ArXiv ID format")
        es = await get_es_client()
        settings_obj = get_settings()
        engine = SearchEngine(es, settings_obj.es_index)
        paper = await engine.get_paper(arxiv_id)
        if paper is None:
            # Retry without version suffix (2301.12345v2 → 2301.12345)
            stripped = re.sub(r'v\d+$', '', arxiv_id)
            if stripped != arxiv_id:
                paper = await engine.get_paper(stripped)
        if paper is None:
            raise HTTPException(status_code=404, detail="Paper not found")
        return paper

    return app


app = create_app()
