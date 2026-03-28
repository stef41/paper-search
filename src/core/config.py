from __future__ import annotations

import os
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore",
    )

    # Elasticsearch
    es_host: str = "elasticsearch"
    es_port: int = 9200
    es_index: str = "arxiv_papers"

    # Redis
    redis_host: str = "redis"
    redis_port: int = 6379
    redis_password: str = ""

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_keys: str = "changeme-key-1"
    rate_limit_per_minute: int = 120
    rate_limit_burst: int = 20
    ddos_rate_limit: str = "600/minute"

    # Ingestion
    arxiv_oai_base_url: str = "https://oaipmh.arxiv.org/oai"
    ingestion_batch_size: int = 500
    ingestion_interval_hours: int = 6
    semantic_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Embeddings
    embedding_dim: int = 384

    # Security
    cors_origins: str = "*"
    max_query_length: int = 2000
    max_regex_length: int = 200
    regex_timeout_ms: int = 1000

    @property
    def es_url(self) -> str:
        return f"http://{self.es_host}:{self.es_port}"

    @property
    def api_key_list(self) -> list[str]:
        return [k.strip() for k in self.api_keys.split(",") if k.strip()]


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings() -> None:
    global _settings
    _settings = None
