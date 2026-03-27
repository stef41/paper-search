from __future__ import annotations

import time
from typing import Any

import structlog
from elasticsearch import AsyncElasticsearch

from src.core.config import get_settings
from src.core.models import (
    SearchHit,
    SearchRequest,
    SearchResponse,
    SemanticMode,
    SemanticQuery,
    SimilarityLevel,
    SortField,
    StatsResponse,
)

logger = structlog.get_logger()


class QueryBuilder:
    """Builds Elasticsearch queries from SearchRequest."""

    def __init__(
        self,
        request: SearchRequest,
        embedding: list[float] | None = None,
        embeddings: list[tuple[SemanticQuery, list[float]]] | None = None,
    ):
        self.req = request
        # Support both old single-embedding API and new multi-embedding API
        if embeddings is not None:
            self._embeddings = embeddings
        elif embedding is not None and request.semantic:
            sem_list = request.semantic if isinstance(request.semantic, list) else [request.semantic]
            self._embeddings = [(sem_list[0], embedding)]
        else:
            self._embeddings = []

    def build(self) -> dict[str, Any]:
        body: dict[str, Any] = {}
        query = self._build_query()

        # Wrap query in function_score if there are exclude semantics
        # (applies continuous cosine-similarity penalty in scoring phase)
        exclude_functions = self._build_exclude_functions()
        if exclude_functions and query:
            query = {
                "function_score": {
                    "query": query,
                    "functions": exclude_functions,
                    "score_mode": "multiply",
                    "boost_mode": "multiply",
                }
            }

        if query:
            body["query"] = query

        # KNN for semantic search (boost mode)
        knn = self._build_knn()
        if knn:
            body["knn"] = knn

        body["sort"] = self._build_sort()
        body["from"] = self.req.offset
        body["size"] = self.req.limit

        if self.req.highlight:
            body["highlight"] = {
                "fields": {
                    "title": {"number_of_fragments": 0},
                    "abstract": {"fragment_size": 200, "number_of_fragments": 3},
                },
                "pre_tags": ["<em>"],
                "post_tags": ["</em>"],
            }

        return body

    def _build_query(self) -> dict[str, Any] | None:
        must: list[dict] = []
        should: list[dict] = []
        filter_clauses: list[dict] = []
        must_not: list[dict] = []
        # Store filter state for KNN reuse
        self._filter_clauses = filter_clauses
        self._must_not = must_not

        # Full-text queries
        if self.req.query:
            should.append({
                "multi_match": {
                    "query": self.req.query,
                    "fields": ["title^3", "abstract^2", "authors.name"],
                    "type": "best_fields",
                    "operator": self.req.operator,
                    **({"minimum_should_match": self.req.minimum_should_match}
                       if self.req.minimum_should_match else {}),
                }
            })

        if self.req.title_query:
            should.append({
                "match": {
                    "title": {
                        "query": self.req.title_query,
                        "boost": 3,
                        "operator": self.req.operator,
                    }
                }
            })

        if self.req.abstract_query:
            should.append({
                "match": {
                    "abstract": {
                        "query": self.req.abstract_query,
                        "boost": 2,
                        "operator": self.req.operator,
                    }
                }
            })

        # Fuzzy matching
        if self.req.fuzzy:
            should.append({
                "multi_match": {
                    "query": self.req.fuzzy,
                    "fields": ["title^3", "abstract^2", "authors.name"],
                    "fuzziness": self.req.fuzzy_fuzziness,
                    "prefix_length": 2,
                }
            })

        # Regex searches
        if self.req.title_regex:
            must.append({"regexp": {"title.raw": {"value": self.req.title_regex, "flags": "ALL"}}})

        if self.req.abstract_regex:
            must.append({"regexp": {"abstract.raw": {"value": self.req.abstract_regex, "flags": "ALL"}}})

        if self.req.author_regex:
            must.append({
                "nested": {
                    "path": "authors",
                    "query": {
                        "regexp": {"authors.name.raw": {"value": self.req.author_regex, "flags": "ALL"}}
                    },
                }
            })

        # Author filters
        if self.req.author:
            must.append({
                "nested": {
                    "path": "authors",
                    "query": {
                        "match": {"authors.name": self.req.author}
                    },
                }
            })

        if self.req.first_author:
            must.append({
                "nested": {
                    "path": "authors",
                    "query": {
                        "bool": {
                            "must": [
                                {"match": {"authors.name": self.req.first_author}},
                                {"term": {"authors.is_first_author": True}},
                            ]
                        }
                    },
                }
            })

        # H-index filters
        if self.req.min_h_index is not None:
            must.append({
                "nested": {
                    "path": "authors",
                    "query": {
                        "range": {"authors.h_index": {"gte": self.req.min_h_index}}
                    },
                }
            })

        if self.req.max_h_index is not None:
            must.append({
                "nested": {
                    "path": "authors",
                    "query": {
                        "range": {"authors.h_index": {"lte": self.req.max_h_index}}
                    },
                }
            })

        if self.req.min_first_author_h_index is not None:
            filter_clauses.append({
                "range": {"first_author_h_index": {"gte": self.req.min_first_author_h_index}}
            })

        if self.req.min_median_h_index_citing is not None:
            filter_clauses.append({
                "range": {
                    "citation_stats.median_h_index_citing_authors": {
                        "gte": self.req.min_median_h_index_citing
                    }
                }
            })

        # Citation filters
        if self.req.min_citations is not None:
            filter_clauses.append({
                "range": {"citation_stats.total_citations": {"gte": self.req.min_citations}}
            })

        if self.req.max_citations is not None:
            filter_clauses.append({
                "range": {"citation_stats.total_citations": {"lte": self.req.max_citations}}
            })

        if self.req.min_references is not None:
            filter_clauses.append({
                "range": {"references_stats.total_references": {"gte": self.req.min_references}}
            })

        # Category filters
        if self.req.categories and len(self.req.categories) > 0:
            filter_clauses.append({"terms": {"categories": self.req.categories}})

        if self.req.primary_category:
            filter_clauses.append({"term": {"primary_category": self.req.primary_category}})

        if self.req.exclude_categories:
            must_not.append({"terms": {"categories": self.req.exclude_categories}})

        # Date filters
        if self.req.submitted_date:
            date_range: dict[str, Any] = {}
            if self.req.submitted_date.gte:
                date_range["gte"] = self.req.submitted_date.gte.isoformat()
            if self.req.submitted_date.lte:
                date_range["lte"] = self.req.submitted_date.lte.isoformat()
            if date_range:
                filter_clauses.append({"range": {"submitted_date": date_range}})

        if self.req.updated_date:
            date_range_u: dict[str, Any] = {}
            if self.req.updated_date.gte:
                date_range_u["gte"] = self.req.updated_date.gte.isoformat()
            if self.req.updated_date.lte:
                date_range_u["lte"] = self.req.updated_date.lte.isoformat()
            if date_range_u:
                filter_clauses.append({"range": {"updated_date": date_range_u}})

        # Metadata filters
        if self.req.has_github is not None:
            filter_clauses.append({"term": {"has_github": self.req.has_github}})

        if self.req.min_page_count is not None:
            filter_clauses.append({"range": {"page_count": {"gte": self.req.min_page_count}}})

        if self.req.max_page_count is not None:
            filter_clauses.append({"range": {"page_count": {"lte": self.req.max_page_count}}})

        if self.req.has_doi is not None:
            if self.req.has_doi:
                filter_clauses.append({"exists": {"field": "doi"}})
            else:
                must_not.append({"exists": {"field": "doi"}})

        if self.req.has_journal_ref is not None:
            if self.req.has_journal_ref:
                filter_clauses.append({"exists": {"field": "journal_ref"}})
            else:
                must_not.append({"exists": {"field": "journal_ref"}})

        # Build final bool query
        if not any([must, should, filter_clauses, must_not]):
            return {"match_all": {}}

        bool_query: dict[str, Any] = {}
        if must:
            bool_query["must"] = must
        if should:
            bool_query["should"] = should
            if not must and not filter_clauses:
                bool_query["minimum_should_match"] = 1
        if filter_clauses:
            bool_query["filter"] = filter_clauses
        if must_not:
            bool_query["must_not"] = must_not

        return {"bool": bool_query}

    def _build_knn(self) -> dict[str, Any] | list[dict[str, Any]] | None:
        boost_entries = [
            (sq, emb) for sq, emb in self._embeddings
            if sq.mode == SemanticMode.BOOST
        ]
        if not boost_entries:
            return None

        field_map = {
            SimilarityLevel.TITLE: "title_embedding",
            SimilarityLevel.ABSTRACT: "abstract_embedding",
        }

        # Build filter clause shared across all KNN entries
        knn_filter_parts: list[dict] = []
        if hasattr(self, "_filter_clauses") and self._filter_clauses:
            knn_filter_parts.extend(self._filter_clauses)
        if hasattr(self, "_must_not") and self._must_not:
            knn_filter_parts.append({"bool": {"must_not": self._must_not}})
        knn_filter = (
            {"bool": {"filter": knn_filter_parts}} if len(knn_filter_parts) > 1
            else knn_filter_parts[0] if knn_filter_parts else None
        )

        knns: list[dict[str, Any]] = []
        for sq, emb in boost_entries:
            field = field_map.get(sq.level, "abstract_embedding")
            knn: dict[str, Any] = {
                "field": field,
                "query_vector": emb,
                "k": min(self.req.limit, 100),
                "num_candidates": min(self.req.limit * 10, 1000),
                "boost": sq.weight,
            }
            if knn_filter:
                knn["filter"] = knn_filter
            knns.append(knn)

        return knns[0] if len(knns) == 1 else knns

    def _build_exclude_functions(self) -> list[dict[str, Any]] | None:
        """Build function_score functions for exclude-mode semantics.

        Uses script_score with negative cosineSimilarity so that papers
        similar to the excluded text get a continuous penalty proportional
        to their similarity — no hard cutoff.
        """
        exclude_entries = [
            (sq, emb) for sq, emb in self._embeddings
            if sq.mode == SemanticMode.EXCLUDE
        ]
        if not exclude_entries:
            return None

        field_map = {
            SimilarityLevel.TITLE: "title_embedding",
            SimilarityLevel.ABSTRACT: "abstract_embedding",
        }

        functions: list[dict[str, Any]] = []
        for sq, emb in exclude_entries:
            field = field_map.get(sq.level, "abstract_embedding")
            # cosineSimilarity returns [-1, 1], so (1 + cos) is [0, 2].
            # High similarity → large denominator → score near 0 (penalised).
            # Low similarity  → denominator ≈ 1 → score near 1 (untouched).
            functions.append({
                "script_score": {
                    "script": {
                        "source": f"1.0 / (1.0 + params.w * (1.0 + cosineSimilarity(params.v, '{field}')))",
                        "params": {
                            "v": emb,
                            "w": sq.weight,
                        },
                    },
                },
                "filter": {"exists": {"field": field}},
            })

        return functions

    def _build_sort(self) -> list[dict[str, Any] | str]:
        sort_map = {
            SortField.DATE: "submitted_date",
            SortField.CITATIONS: "citation_stats.total_citations",
            SortField.H_INDEX: "first_author_h_index",
            SortField.PAGE_COUNT: "page_count",
            SortField.UPDATED: "updated_date",
        }

        if self.req.sort_by == SortField.RELEVANCE:
            return ["_score", {"submitted_date": {"order": "desc", "missing": "_last"}}]

        field = sort_map[self.req.sort_by]
        return [
            {field: {"order": self.req.sort_order.value, "missing": "_last"}},
            "_score",
        ]


class SearchEngine:
    """Executes search queries against Elasticsearch."""

    def __init__(self, client: AsyncElasticsearch, index: str):
        self.client = client
        self.index = index

    async def search(
        self,
        request: SearchRequest,
        embedding: list[float] | None = None,
        embeddings: list[tuple[SemanticQuery, list[float]]] | None = None,
    ) -> SearchResponse:
        builder = QueryBuilder(request, embedding=embedding, embeddings=embeddings)
        body = builder.build()

        logger.debug("search_query", body=body)

        start = time.monotonic()
        result = await self.client.search(
            index=self.index,
            body=body,
            request_timeout=10,
            track_total_hits=True,
            timeout="5s",
        )
        took_ms = int((time.monotonic() - start) * 1000)

        total = result["hits"]["total"]["value"]
        hits = []
        for hit in result["hits"]["hits"]:
            source = hit["_source"]
            highlights = {}
            if "highlight" in hit:
                highlights = hit["highlight"]

            # Always strip embeddings from responses (prevent model extraction)
            source.pop("title_embedding", None)
            source.pop("abstract_embedding", None)
            source.pop("paragraph_embeddings", None)

            hits.append(SearchHit(
                score=hit.get("_score"),
                highlights=highlights if highlights else None,
                **{k: v for k, v in source.items()
                   if k not in ("title_embedding", "abstract_embedding", "paragraph_embeddings")},
            ))

        return SearchResponse(
            total=total,
            hits=hits,
            took_ms=took_ms,
            offset=request.offset,
            limit=request.limit,
        )

    async def get_stats(self) -> StatsResponse:
        count_resp = await self.client.count(index=self.index)
        total = count_resp["count"]

        aggs_body = {
            "size": 0,
            "aggs": {
                "categories": {"terms": {"field": "categories", "size": 100}},
                "min_date": {"min": {"field": "submitted_date"}},
                "max_date": {"max": {"field": "submitted_date"}},
                "github_count": {"filter": {"term": {"has_github": True}}},
                "avg_pages": {"avg": {"field": "page_count"}},
                "avg_citations": {"avg": {"field": "citation_stats.total_citations"}},
            },
        }
        aggs_resp = await self.client.search(index=self.index, body=aggs_body)
        aggs = aggs_resp["aggregations"]

        cats = {b["key"]: b["doc_count"] for b in aggs["categories"]["buckets"]}

        return StatsResponse(
            total_papers=total,
            categories=cats,
            date_range={
                "min": aggs["min_date"].get("value_as_string"),
                "max": aggs["max_date"].get("value_as_string"),
            },
            papers_with_github=aggs["github_count"]["doc_count"],
            avg_page_count=aggs["avg_pages"]["value"],
            avg_citations=aggs["avg_citations"]["value"],
        )

    async def get_paper(self, arxiv_id: str) -> dict | None:
        try:
            resp = await self.client.search(
                index=self.index,
                body={"query": {"term": {"arxiv_id": arxiv_id}}, "size": 1},
            )
            hits = resp["hits"]["hits"]
            if hits:
                return hits[0]["_source"]
            return None
        except Exception:
            logger.exception("get_paper_error", arxiv_id=arxiv_id)
            return None
