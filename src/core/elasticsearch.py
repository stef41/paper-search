from __future__ import annotations

import structlog
from elasticsearch import AsyncElasticsearch

from src.core.config import get_settings

logger = structlog.get_logger()

_client: AsyncElasticsearch | None = None


async def get_es_client() -> AsyncElasticsearch:
    global _client
    if _client is None:
        settings = get_settings()
        _client = AsyncElasticsearch(
            hosts=[settings.es_url],
            request_timeout=30,
            max_retries=3,
            retry_on_timeout=True,
        )
    return _client


async def close_es_client() -> None:
    global _client
    if _client is not None:
        await _client.close()
        _client = None


def get_index_mapping(embedding_dim: int) -> dict:
    return {
        "settings": {
            "number_of_shards": 2,
            "number_of_replicas": 0,
            "refresh_interval": "5s",
            "analysis": {
                "analyzer": {
                    "arxiv_analyzer": {
                        "type": "custom",
                        "tokenizer": "standard",
                        "filter": [
                            "lowercase",
                            "stop",
                            "snowball",
                            "arxiv_synonym",
                        ],
                    },
                    "keyword_lowercase": {
                        "type": "custom",
                        "tokenizer": "keyword",
                        "filter": ["lowercase"],
                    },
                },
                "filter": {
                    "arxiv_synonym": {
                        "type": "synonym",
                        "synonyms": [
                            "nn, neural network",
                            "cnn, convolutional neural network",
                            "rnn, recurrent neural network",
                            "llm, large language model",
                            "nlp, natural language processing",
                            "cv, computer vision",
                            "rl, reinforcement learning",
                            "gan, generative adversarial network",
                            "vae, variational autoencoder",
                            "bert, bidirectional encoder representations from transformers",
                            "gpt, generative pretrained transformer",
                        ],
                    }
                },
            },
            "index": {
                "max_regex_length": 1000,
            },
        },
        "mappings": {
            "properties": {
                "arxiv_id": {"type": "keyword"},
                "title": {
                    "type": "text",
                    "analyzer": "arxiv_analyzer",
                    "fields": {
                        "raw": {"type": "keyword"},
                        "suggest": {
                            "type": "text",
                            "analyzer": "simple",
                        },
                    },
                },
                "abstract": {
                    "type": "text",
                    "analyzer": "arxiv_analyzer",
                    "fields": {
                        "raw": {"type": "keyword", "ignore_above": 32766},
                    },
                },
                "authors": {
                    "type": "nested",
                    "include_in_root": True,
                    "properties": {
                        "name": {
                            "type": "text",
                            "fields": {
                                "raw": {"type": "keyword"},
                                "lowercase": {
                                    "type": "text",
                                    "analyzer": "keyword_lowercase",
                                },
                            },
                        },
                        "is_first_author": {"type": "boolean"},
                        "h_index": {"type": "integer"},
                        "citation_count": {"type": "integer"},
                    },
                },
                "categories": {"type": "keyword"},
                "primary_category": {"type": "keyword"},
                "submitted_date": {"type": "date"},
                "updated_date": {"type": "date"},
                "published_date": {"type": "date"},
                "doi": {"type": "keyword"},
                "journal_ref": {
                    "type": "text",
                    "fields": {"raw": {"type": "keyword"}},
                },
                "comments": {"type": "text"},
                "page_count": {"type": "integer"},
                "has_github": {"type": "boolean"},
                "github_urls": {"type": "keyword"},
                "pdf_url": {"type": "keyword"},
                "abstract_url": {"type": "keyword"},
                "first_author": {
                    "type": "text",
                    "fields": {"raw": {"type": "keyword"}},
                },
                "first_author_h_index": {"type": "integer"},
                # Embeddings
                "title_embedding": {
                    "type": "dense_vector",
                    "dims": embedding_dim,
                    "index": True,
                    "similarity": "cosine",
                },
                "abstract_embedding": {
                    "type": "dense_vector",
                    "dims": embedding_dim,
                    "index": True,
                    "similarity": "cosine",
                },
                # Citation stats
                "citation_stats": {
                    "type": "object",
                    "properties": {
                        "total_citations": {"type": "integer"},
                        "avg_citation_age_years": {"type": "float"},
                        "median_h_index_citing_authors": {"type": "float"},
                        "top_citing_categories": {"type": "keyword"},
                    },
                },
                "references_stats": {
                    "type": "object",
                    "properties": {
                        "total_references": {"type": "integer"},
                        "avg_reference_age_years": {"type": "float"},
                        "top_referenced_categories": {"type": "keyword"},
                    },
                },
                # Citation links (arxiv IDs of papers)
                "reference_ids": {"type": "keyword"},
                "cited_by_ids": {"type": "keyword"},
                "enrichment_source": {"type": "keyword"},
                "enriched_at": {"type": "date"},
            }
        },
    }


async def ensure_index(client: AsyncElasticsearch, index: str, embedding_dim: int) -> None:
    try:
        exists = await client.indices.exists(index=index)
    except Exception:
        exists = False
    if not exists:
        mapping = get_index_mapping(embedding_dim)
        await client.indices.create(index=index, body=mapping)
        logger.info("created_index", index=index)
    else:
        logger.info("index_exists", index=index)
