from __future__ import annotations

import hashlib
import hmac
import re
import time
from typing import Any

import structlog
from fastapi import HTTPException, Request, Security
from fastapi.security import APIKeyHeader

from src.core.config import get_settings

logger = structlog.get_logger()

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str | None = Security(api_key_header)) -> str:
    if api_key is None:
        raise HTTPException(status_code=401, detail="Missing API key")

    settings = get_settings()
    valid_keys = settings.api_key_list

    # Constant-time comparison to prevent timing attacks
    for valid_key in valid_keys:
        if hmac.compare_digest(api_key.encode(), valid_key.encode()):
            return api_key

    raise HTTPException(status_code=403, detail="Invalid API key")


def sanitize_query(text: str, max_length: int = 2000) -> str:
    if not text:
        return text
    # Truncate
    text = text[:max_length]
    # Remove null bytes
    text = text.replace("\x00", "")
    return text


def validate_regex_pattern(pattern: str, max_length: int = 200) -> str:
    if len(pattern) > max_length:
        raise HTTPException(
            status_code=400,
            detail=f"Regex pattern too long (max {max_length} chars)",
        )

    # Block known ReDoS patterns
    redos_patterns = [
        r"\([^)]*[+*][^)]*\)[+*]",
        r"\([^)]*\|[^)]*\)[+*]",
    ]
    for rdos in redos_patterns:
        if re.search(rdos, pattern):
            raise HTTPException(
                status_code=400,
                detail="Potentially dangerous regex pattern (ReDoS risk)",
            )

    try:
        re.compile(pattern)
    except re.error as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid regex pattern: {exc}",
        )

    return pattern


def validate_search_request(request: Any) -> None:
    settings = get_settings()

    if request.query and len(request.query) > settings.max_query_length:
        raise HTTPException(
            status_code=400,
            detail=f"Query too long (max {settings.max_query_length} chars)",
        )

    for field in ("title_regex", "abstract_regex", "author_regex"):
        val = getattr(request, field, None)
        if val:
            validate_regex_pattern(val, settings.max_regex_length)

    if request.offset + request.limit > 50200:
        raise HTTPException(
            status_code=400,
            detail="Cannot paginate beyond 50200 results. Use narrower filters.",
        )
