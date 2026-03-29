from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
import structlog

from src.core.config import get_settings

logger = structlog.get_logger()

_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        settings = get_settings()
        _model = SentenceTransformer(settings.semantic_model)
        logger.info("loaded_embedding_model", model=settings.semantic_model)
    return _model


def encode_text(text: str) -> list[float]:
    model = _get_model()
    embedding = model.encode(text, normalize_embeddings=True)
    return embedding.tolist()


def encode_texts(texts: list[str], batch_size: int = 64) -> list[list[float]]:
    if not texts:
        return []
    model = _get_model()
    embeddings = model.encode(texts, batch_size=batch_size, normalize_embeddings=True)
    return embeddings.tolist()


def encode_paragraphs(abstract: str) -> list[list[float]]:
    paragraphs = [p.strip() for p in abstract.split("\n") if p.strip()]
    if not paragraphs:
        return []
    return encode_texts(paragraphs)


def cache_key_for_text(text: str, level: str) -> str:
    settings = get_settings()
    h = hashlib.sha256(f"{settings.semantic_model}:{level}:{text}".encode()).hexdigest()[:32]
    return f"emb:{h}"


async def get_cached_embedding(
    redis_client: Any, text: str, level: str
) -> list[float] | None:
    key = cache_key_for_text(text, level)
    cached = await redis_client.get(key)
    if cached:
        try:
            return json.loads(cached)
        except (json.JSONDecodeError, ValueError):
            return None
    return None


async def cache_embedding(
    redis_client: Any, text: str, level: str, embedding: list[float], ttl: int = 3600
) -> None:
    key = cache_key_for_text(text, level)
    await redis_client.setex(key, ttl, json.dumps(embedding))
