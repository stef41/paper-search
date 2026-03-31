"""Graph query engine — expressive graph-style queries using Elasticsearch aggregations.

Supports 16 graph query types:
  1.  Category diversity:          papers spanning many subcategories
  2.  Co-authorship ego network:   collaboration graph around an author
  3.  Author bridge detection:     authors publishing across disjoint fields
  4.  Cross-category flow:         how ideas flow between category pairs
  5.  Interdisciplinary papers:    papers whose categories have low co-occurrence
  6.  Rising interdisciplinary:    recent breakout papers cited across fields
  7.  Citation traversal:          follow citation links, aggregate by category/author/year
  8.  Paper citation network:      direct paper→paper citation graph
  9.  Author influence:            author→author influence via citation paths
  10. Temporal evolution:           category publication volume over time
  11. Paper similarity:             semantic similarity network using embeddings
  12. Domain collaboration:         domain-level co-occurrence graph
  13. Author topic evolution:       how an author's topics shift over time
  14. GitHub landscape:             code-availability patterns by category/time
  15. Bibliographic coupling:       papers sharing many references
  16. Co-citation:                  papers frequently cited together

All queries compose with the existing SearchRequest filters — date ranges,
citation thresholds, h-index, GitHub, regex, etc. are all respected.
"""
from __future__ import annotations

import asyncio
import contextvars
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field as dc_field
from itertools import combinations, islice
from typing import Any

import structlog
from elasticsearch import AsyncElasticsearch

from src.core.models import (
    Aggregation,
    GraphQuery,
    GraphQueryType,
    GraphResponse,
    GraphNode,
    GraphEdge,
    PathFilter,
    PatternNode,
    PatternEdge,
    PipelineStep,
    SubgraphFilter,
    WhereCondition,
    SearchRequest,
    SemanticMode,
    SemanticQuery,
    SimilarityLevel,
)
from src.core.search import QueryBuilder

logger = structlog.get_logger()


def _safe_int(val: Any, default: int = 0) -> int:
    """Convert a value to int, returning *default* on failure."""
    try:
        return int(val)
    except (TypeError, ValueError, OverflowError):
        return default


# ═══════════════════════════════════════════════════════════════════════════════
# Schema abstraction: decouple graph algorithms from ArXiv-specific field names
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class FieldMapping:
    """Maps generic graph concepts to domain-specific ES field names.

    Change ONLY this mapping to adapt the engine to a different data source
    (e.g., Scopus, OpenAlex, DBLP, or any citation network).
    """
    node_id: str = "arxiv_id"
    node_label: str = "title"
    node_categories: str = "categories"
    node_primary_category: str = "primary_category"
    node_authors: str = "authors"
    node_author_name: str = "name"     # nested field inside authors
    node_timestamp: str = "submitted_date"
    node_metrics: str = "citation_stats"
    node_metrics_total: str = "total_citations"
    outgoing_edges: str = "reference_ids"
    incoming_edges: str = "cited_by_ids"
    embedding_vector: str = "abstract_embedding"
    has_code: str = "has_github"

    @property
    def subgraph_fields(self) -> list[str]:
        """Fields needed for citation subgraph construction."""
        return [self.node_id, self.node_label, self.node_categories,
                self.node_primary_category, self.node_authors, self.node_timestamp,
                self.node_metrics, self.outgoing_edges, self.incoming_edges,
                self.has_code]

    def extract_id(self, src: dict) -> str:
        return src.get(self.node_id) or ""

    def extract_label(self, src: dict) -> str:
        return src.get(self.node_label, src.get(self.node_id, ""))

    def extract_authors(self, src: dict, max_n: int = 50) -> list[str]:
        authors = src.get(self.node_authors) or []
        return [a.get(self.node_author_name, "") if isinstance(a, dict) else (str(a) if a is not None else "")
                for a in authors[:max_n]]

    def extract_citations(self, src: dict) -> int:
        cs = src.get(self.node_metrics) or {}
        return (cs.get(self.node_metrics_total) or 0) if isinstance(cs, dict) else 0

    def extract_outgoing(self, src: dict) -> list[str]:
        return src.get(self.outgoing_edges, []) or []

    def extract_incoming(self, src: dict) -> list[str]:
        return src.get(self.incoming_edges, []) or []


@dataclass
class EdgeType:
    """Definition of a graph edge type."""
    name: str
    description: str
    directed: bool = True
    source_type: str = "paper"
    target_type: str = "paper"


# ── Edge Type Registry ──
# All recognized edge types. Handlers and pattern matching validate against this.
EDGE_REGISTRY: dict[str, EdgeType] = {
    "cites": EdgeType("cites", "Paper A references paper B", directed=True),
    "cited_by": EdgeType("cited_by", "Paper A is cited by paper B", directed=True),
    "co_authored": EdgeType("co_authored", "Papers share at least one author", directed=False),
    "same_category": EdgeType("same_category", "Papers share at least one category", directed=False),
    "similar_to": EdgeType("similar_to", "Papers are semantically similar (embedding cosine)", directed=False),
    "contains": EdgeType("contains", "Community/component contains a paper", directed=True, source_type="community", target_type="paper"),
    "inter_community": EdgeType("inter_community", "Edge between communities", directed=False, source_type="community", target_type="community"),
    "mutual_citation": EdgeType("mutual_citation", "A cites B and B cites A", directed=False),
    "co_occurs": EdgeType("co_occurs", "Categories co-occur on papers", directed=False, source_type="category", target_type="category"),
    "published_in": EdgeType("published_in", "Author published in category", directed=True, source_type="author", target_type="category"),
    "influences": EdgeType("influences", "Author influences another via citations", directed=True, source_type="author", target_type="author"),
    "bibliographic_coupling": EdgeType("bibliographic_coupling", "Papers share references", directed=False),
    "cocitation": EdgeType("cocitation", "Papers are co-cited together", directed=False),
    "path_link": EdgeType("path_link", "Sequential link in a path", directed=True),
    "walk_transition": EdgeType("walk_transition", "Random walk transition", directed=True),
    "bridges": EdgeType("bridges", "Bridge edge whose removal disconnects graph", directed=False),
    "triangle_edge": EdgeType("triangle_edge", "Part of a citation triangle", directed=True),
    "chain_link": EdgeType("chain_link", "Part of a citation chain", directed=True),
    "star_spoke": EdgeType("star_spoke", "Spoke in a star pattern", directed=True),
    "predicted": EdgeType("predicted", "Predicted future link", directed=False),
    "mst_edge": EdgeType("mst_edge", "Edge in minimum spanning tree", directed=False),
}


def get_edge_type(name: str) -> EdgeType:
    """Get an edge type by name; returns a default if not in registry."""
    return EDGE_REGISTRY.get(name, EdgeType(name, f"Custom edge type: {name}"))


# Default mapping for ArXiv
ARXIV_FIELDS = FieldMapping()

# Per-coroutine request state (safe under concurrent asyncio requests)
_ctx_active_id_filter: contextvars.ContextVar[list[str] | None] = contextvars.ContextVar(
    "_ctx_active_id_filter", default=None
)
_ctx_embeddings: contextvars.ContextVar[list[tuple[SemanticQuery, list[float]]] | None] = contextvars.ContextVar(
    "_ctx_embeddings", default=None
)
_ctx_first_boost_emb: contextvars.ContextVar[list[float] | None] = contextvars.ContextVar(
    "_ctx_first_boost_emb", default=None
)
_ctx_projection_direction: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "_ctx_projection_direction", default=None
)


class GraphEngine:
    """Executes graph-style queries against the ES index."""

    MAX_AGG_BUCKETS = 5000
    MAX_RESULTS = 10000

    def __init__(self, client: AsyncElasticsearch, index: str,
                 fields: FieldMapping | None = None):
        self.client = client
        self.index = index
        self.F = fields or ARXIV_FIELDS

    async def execute(
        self,
        graph_query: GraphQuery,
        search_request: SearchRequest | None = None,
        embedding: list[float] | None = None,
        embeddings: list[tuple[SemanticQuery, list[float]]] | None = None,
    ) -> GraphResponse:
        start = time.monotonic()

        # Normalize: prefer new multi-embedding, fallback to single
        if embeddings is not None:
            _ctx_embeddings.set(embeddings)
        elif embedding is not None and search_request and search_request.semantic:
            sem_list = search_request.semantic if isinstance(search_request.semantic, list) else [search_request.semantic]
            _ctx_embeddings.set([(sem_list[0], embedding)])
        else:
            _ctx_embeddings.set([])

        # Extract first boost embedding for backward-compat helpers
        boost_embs = [(sq, emb) for sq, emb in _ctx_embeddings.get() if sq.mode == SemanticMode.BOOST]
        _ctx_first_boost_emb.set(boost_embs[0][1] if boost_embs else None)

        handler = {
            GraphQueryType.CATEGORY_DIVERSITY: self._category_diversity,
            GraphQueryType.COAUTHOR_NETWORK: self._coauthor_network,
            GraphQueryType.AUTHOR_BRIDGE: self._author_bridge,
            GraphQueryType.CROSS_CATEGORY_FLOW: self._cross_category_flow,
            GraphQueryType.INTERDISCIPLINARY: self._interdisciplinary,
            GraphQueryType.RISING_INTERDISCIPLINARY: self._rising_interdisciplinary,
            GraphQueryType.CITATION_TRAVERSAL: self._citation_traversal,
            GraphQueryType.PAPER_CITATION_NETWORK: self._paper_citation_network,
            GraphQueryType.AUTHOR_INFLUENCE: self._author_influence,
            GraphQueryType.TEMPORAL_EVOLUTION: self._temporal_evolution,
            GraphQueryType.PAPER_SIMILARITY: self._paper_similarity,
            GraphQueryType.DOMAIN_COLLABORATION: self._domain_collaboration,
            GraphQueryType.AUTHOR_TOPIC_EVOLUTION: self._author_topic_evolution,
            GraphQueryType.GITHUB_LANDSCAPE: self._github_landscape,
            GraphQueryType.BIBLIOGRAPHIC_COUPLING: self._bibliographic_coupling,
            GraphQueryType.COCITATION: self._cocitation,
            GraphQueryType.MULTIHOP_CITATION: self._multihop_citation,
            GraphQueryType.SHORTEST_CITATION_PATH: self._shortest_citation_path,
            GraphQueryType.PAGERANK: self._pagerank,
            GraphQueryType.COMMUNITY_DETECTION: self._community_detection,
            GraphQueryType.CITATION_PATTERNS: self._citation_patterns,
            GraphQueryType.CONNECTED_COMPONENTS: self._connected_components,
            GraphQueryType.WEIGHTED_SHORTEST_PATH: self._weighted_shortest_path,
            GraphQueryType.BETWEENNESS_CENTRALITY: self._betweenness_centrality,
            GraphQueryType.CLOSENESS_CENTRALITY: self._closeness_centrality,
            GraphQueryType.STRONGLY_CONNECTED_COMPONENTS: self._strongly_connected_components,
            GraphQueryType.TOPOLOGICAL_SORT: self._topological_sort,
            GraphQueryType.LINK_PREDICTION: self._link_prediction,
            GraphQueryType.LOUVAIN_COMMUNITY: self._louvain_community,
            GraphQueryType.DEGREE_CENTRALITY: self._degree_centrality,
            GraphQueryType.EIGENVECTOR_CENTRALITY: self._eigenvector_centrality,
            GraphQueryType.KCORE_DECOMPOSITION: self._kcore_decomposition,
            GraphQueryType.ARTICULATION_POINTS: self._articulation_points,
            GraphQueryType.INFLUENCE_MAXIMIZATION: self._influence_maximization,
            GraphQueryType.HITS: self._hits,
            GraphQueryType.HARMONIC_CENTRALITY: self._harmonic_centrality,
            GraphQueryType.KATZ_CENTRALITY: self._katz_centrality,
            GraphQueryType.ALL_SHORTEST_PATHS: self._all_shortest_paths,
            GraphQueryType.K_SHORTEST_PATHS: self._k_shortest_paths,
            GraphQueryType.RANDOM_WALK: self._random_walk,
            GraphQueryType.TRIANGLE_COUNT: self._triangle_count,
            GraphQueryType.GRAPH_DIAMETER: self._graph_diameter,
            GraphQueryType.LEIDEN_COMMUNITY: self._leiden_community,
            GraphQueryType.BRIDGE_EDGES: self._bridge_edges,
            GraphQueryType.MIN_CUT: self._min_cut,
            GraphQueryType.MINIMUM_SPANNING_TREE: self._minimum_spanning_tree,
            GraphQueryType.NODE_SIMILARITY: self._node_similarity,
            GraphQueryType.BIPARTITE_PROJECTION: self._bipartite_projection,
            GraphQueryType.ADAMIC_ADAR_INDEX: self._adamic_adar_index,
            GraphQueryType.PATTERN_MATCH: self._pattern_match,
            GraphQueryType.PIPELINE: self._pipeline,
            GraphQueryType.SUBGRAPH_PROJECTION: self._subgraph_projection,
            GraphQueryType.TRAVERSE: self._traverse,
            GraphQueryType.GRAPH_UNION: self._graph_union,
            GraphQueryType.GRAPH_INTERSECTION: self._graph_intersection,
        }[graph_query.type]

        try:
            result = await handler(graph_query, search_request, _ctx_first_boost_emb.get())
            result.took_ms = int((time.monotonic() - start) * 1000)

            # ── Post-processing: aggregations (applies to any graph type) ──
            if graph_query.aggregations:
                result = self._apply_aggregations(result, graph_query.aggregations)

            return result
        finally:
            _ctx_embeddings.set([])
            _ctx_first_boost_emb.set(None)
            _ctx_active_id_filter.set(None)
            _ctx_projection_direction.set(None)

    def _apply_aggregations(
        self,
        result: GraphResponse,
        aggregations: list[Aggregation],
    ) -> GraphResponse:
        """Compute aggregation functions over result nodes (like Cypher RETURN)."""
        agg_results: dict[str, Any] = {}
        paper_nodes = [n for n in result.nodes if n.type == "paper"]

        for agg in aggregations:
            # Skip duplicate aliases — keep the first occurrence
            if agg.alias in agg_results:
                continue
            fn = agg.function
            field = agg.field

            if fn == "count":
                agg_results[agg.alias] = len(paper_nodes) if field is None else sum(
                    1 for n in paper_nodes if n.properties.get(field) is not None)

            elif fn in ("sum", "avg", "min", "max"):
                vals: list[float] = []
                for n in paper_nodes:
                    v = n.properties.get(field)
                    if v is None and field == "citations":
                        v = n.properties.get("total_citations")
                    if v is None and field == "total_citations":
                        v = n.properties.get("citations")
                    if v is not None:
                        try:
                            vals.append(float(v))
                        except (TypeError, ValueError):
                            pass
                if fn == "sum":
                    agg_results[agg.alias] = sum(vals) if vals else 0
                elif fn == "avg":
                    agg_results[agg.alias] = round(sum(vals) / len(vals), 4) if vals else None
                elif fn == "min":
                    agg_results[agg.alias] = min(vals) if vals else None
                elif fn == "max":
                    agg_results[agg.alias] = max(vals) if vals else None

            elif fn == "collect":
                collected: list[Any] = []
                for n in paper_nodes:
                    v = n.properties.get(field)
                    if v is not None:
                        if isinstance(v, list):
                            collected.extend(v)
                        else:
                            collected.append(v)
                agg_results[agg.alias] = collected[:50000]

            elif fn == "group_count":
                groups: Counter[str] = Counter()
                for n in paper_nodes:
                    v = n.properties.get(field)
                    if v is not None:
                        if isinstance(v, list):
                            for item in v:
                                groups[str(item)] += 1
                        else:
                            groups[str(v)] += 1
                agg_results[agg.alias] = dict(groups.most_common(5000))

        result.metadata["aggregations"] = agg_results
        return result

    @staticmethod
    def _node_matches_path_filter(src: dict, filters: dict[str, Any]) -> bool:
        """Check if a paper matches path filter criteria."""
        if "categories" in filters:
            if not set(filters["categories"]) & set(src.get("categories") or []):
                return False
        if "primary_category" in filters:
            if src.get("primary_category") != filters["primary_category"]:
                return False
        if "has_github" in filters:
            if src.get("has_github") != filters["has_github"]:
                return False
        if "min_citations" in filters:
            cs = src.get("citation_stats") or {}
            tc = (cs.get("total_citations") or 0) if isinstance(cs, dict) else 0
            if tc < _safe_int(filters["min_citations"]):
                return False
        if "max_citations" in filters:
            cs = src.get("citation_stats") or {}
            tc = (cs.get("total_citations") or 0) if isinstance(cs, dict) else 0
            if tc > _safe_int(filters["max_citations"]):
                return False
        if "date_from" in filters:
            sd = src.get("submitted_date") or ""
            if not sd:
                return False
            if sd[:10] < filters["date_from"][:10]:
                return False
        if "date_to" in filters:
            sd = src.get("submitted_date") or ""
            if not sd:
                return False
            if sd[:10] > filters["date_to"][:10]:
                return False
        return True

    def _filter_paths(
        self,
        paths: list[list[str]],
        paper_cache: dict[str, dict],
        pf: PathFilter,
    ) -> list[list[str]]:
        """Apply PathFilter to a list of paths, returning those that pass."""
        result: list[list[str]] = []
        for path in paths:
            hops = len(path) - 1
            if pf.min_path_length is not None and hops < pf.min_path_length:
                continue
            if pf.max_path_length is not None and hops > pf.max_path_length:
                continue
            if pf.all_nodes_match:
                if not all(self._node_matches_path_filter(
                    paper_cache.get(pid, {}), pf.all_nodes_match) for pid in path):
                    continue
            if pf.any_node_matches:
                # Check intermediate nodes (exclude source and target)
                intermediates = path[1:-1] if len(path) > 2 else []
                if intermediates and not any(self._node_matches_path_filter(
                    paper_cache.get(pid, {}), pf.any_node_matches) for pid in intermediates):
                    continue
            result.append(path)
        return result

    # ── helpers ──

    def _base_query(
        self,
        search_request: SearchRequest | None,
        embedding: list[float] | None = None,
    ) -> dict[str, Any]:
        """Build an ES query dict incorporating all SearchRequest filters."""
        if search_request is None:
            q: dict[str, Any] = {"match_all": {}}
        else:
            # Pass embeddings so exclude-mode semantic queries are available
            exclude_embs = [(sq, emb) for sq, emb in _ctx_embeddings.get() if sq.mode == SemanticMode.EXCLUDE]
            builder = QueryBuilder(search_request, embedding, embeddings=exclude_embs if exclude_embs else None)
            q = builder._build_query() or {"match_all": {}}
            # Wrap in function_score if exclude semantics are present
            exclude_functions = builder._build_exclude_functions()
            if exclude_functions and q:
                q = {
                    "function_score": {
                        "query": q,
                        "functions": exclude_functions,
                        "score_mode": "multiply",
                        "boost_mode": "multiply",
                    }
                }
        # If a pipeline/projection set an active ID filter, apply it
        _id_filter = _ctx_active_id_filter.get()
        if _id_filter is not None:
            if not _id_filter:
                # Empty filter means previous step produced no papers;
                # return a query that matches nothing.
                return {"bool": {"must_not": [{"match_all": {}}]}}
            q = {"bool": {"must": [q], "filter": [{"terms": {"arxiv_id": _id_filter}}]}}
        return q

    def _build_knn(
        self,
        search_request: SearchRequest | None,
        embedding: list[float] | None = None,
    ) -> dict[str, Any] | list[dict[str, Any]] | None:
        """Build KNN clause(s) for boost-mode semantic similarity, or None."""
        if search_request is None or embedding is None:
            return None
        # Build using multi-embedding if available, else single
        boost_embs = [(sq, emb) for sq, emb in _ctx_embeddings.get() if sq.mode == SemanticMode.BOOST]
        if boost_embs:
            builder = QueryBuilder(search_request, embeddings=boost_embs)
        else:
            builder = QueryBuilder(search_request, embedding=embedding)
        builder._build_query()  # populate filter state for KNN
        return builder._build_knn()

    async def _do_search(
        self,
        body: dict[str, Any],
        sr: SearchRequest | None = None,
        emb: list[float] | None = None,
    ) -> dict:
        """Execute an ES search, injecting KNN if semantic similarity is active."""
        body = {**body}  # avoid mutating caller's dict
        knn = self._build_knn(sr, emb)
        if knn:
            # Apply the body's query as KNN filter so KNN results respect
            # any handler-specific constraints added beyond _base_query.
            q = body.get("query")
            requested_size = body.get("size", 50)
            entries = knn if isinstance(knn, list) else [knn]
            for entry in entries:
                if q:
                    entry["filter"] = q
                # Override k to match the graph handler's actual result size
                # (the default k is derived from SearchRequest pagination
                # which is unrelated to the graph query's size needs).
                entry["k"] = min(max(requested_size, entry.get("k", 20)), 500)
                entry["num_candidates"] = min(entry["k"] * 10, 5000)
            body["knn"] = knn
        body["timeout"] = "10s"
        return await self.client.options(request_timeout=15).search(
            index=self.index, body=body,
        )

    async def _semantic_prefilter(
        self,
        sr: SearchRequest | None,
        emb: list[float] | None,
        max_candidates: int = 500,
    ) -> list[str] | None:
        """Run a KNN-only search to get semantically similar paper IDs.

        Used as a pre-filter for queries where function_score with
        boost_mode='replace' would override KNN scoring.
        Returns None if semantic is not active or if no embeddings exist.
        """
        knn = self._build_knn(sr, emb)
        if not knn:
            return None

        # Normalize to single KNN entry (use first for prefilter)
        if isinstance(knn, list):
            knn = knn[0]
        knn["k"] = min(max_candidates, 5000)
        knn["num_candidates"] = min(max_candidates * 5, 10000)
        body: dict[str, Any] = {"knn": knn, "size": min(max_candidates, 10000), "_source": ["arxiv_id"]}
        # Also apply text/filter constraints if any
        base_q = self._base_query(sr, emb)
        if base_q != {"match_all": {}}:
            knn["filter"] = base_q

        try:
            body["timeout"] = "10s"
            resp = await self.client.options(request_timeout=15).search(
                index=self.index, body=body,
            )
            ids = [h["_source"]["arxiv_id"] for h in resp["hits"]["hits"]]
            if not ids:
                # No embeddings stored yet — skip pre-filter gracefully
                return None
            return ids[:10000]
        except Exception:
            # KNN may fail if no documents have the embedding field
            logger.warning("semantic_prefilter_failed", exc_info=True)
            return None

    async def _agg_search(
        self,
        query: dict[str, Any],
        aggs: dict[str, Any],
        size: int = 0,
        sr: SearchRequest | None = None,
        emb: list[float] | None = None,
    ) -> dict:
        body: dict[str, Any] = {"query": query, "aggs": aggs, "size": size}
        knn = self._build_knn(sr, emb)
        if knn:
            # Apply the full query as KNN filter so KNN results are
            # constrained by the handler's extra filters (e.g. author clause),
            # not just the base SearchRequest filters.
            entries = knn if isinstance(knn, list) else [knn]
            for entry in entries:
                entry["filter"] = query
                # Override k to provide enough KNN candidates for aggregations
                # (default k from SearchRequest pagination is too low).
                entry["k"] = min(max(200, entry.get("k", 20)), 500)
                entry["num_candidates"] = min(entry["k"] * 10, 5000)
            body["knn"] = knn
        body["timeout"] = "10s"
        return await self.client.options(request_timeout=15).search(
            index=self.index,
            body=body,
        )

    # ── 1. Category diversity ──

    async def _category_diversity(
        self,
        gq: GraphQuery,
        sr: SearchRequest | None,
        emb: list[float] | None,
    ) -> GraphResponse:
        """Find papers tagged with many distinct (sub-)categories.

        Returns papers sorted by number of unique categories, showing which
        categories each paper spans. Great for finding interdisciplinary work.
        """
        min_cats = gq.min_categories or 3
        limit = min(gq.limit or 50, self.MAX_RESULTS)
        query = self._base_query(sr, emb)

        # If semantic similarity is active, pre-filter to semantically
        # similar papers (function_score with boost_mode=replace would
        # otherwise override KNN scoring)
        sem_ids = await self._semantic_prefilter(sr, emb, max_candidates=max(limit * 20, 500))
        if sem_ids is not None:
            query = {"bool": {"must": [query], "filter": [{"terms": {"arxiv_id": sem_ids}}]}}

        # Use a script_score query to score by number of categories
        body = {
            "query": {
                "function_score": {
                    "query": query,
                    "script_score": {
                        "script": {
                            "source": "doc['categories'].length"
                        }
                    },
                    "boost_mode": "replace",
                }
            },
            "size": limit,
            "min_score": min_cats,
            "_source": ["arxiv_id", "title", "categories", "primary_category",
                         "authors", "submitted_date", "citation_stats", "has_github"],
        }
        resp = await self._do_search(body) if sem_ids is not None else await self._do_search(body, sr, emb)

        nodes: list[GraphNode] = []
        edges: list[GraphEdge] = []
        cat_set: set[str] = set()

        for hit in resp["hits"]["hits"]:
            src = hit["_source"]
            cats = src.get("categories") or []
            if len(cats) < min_cats:
                continue
            cat_set.update(cats)
            nodes.append(self._make_paper_node(src, {"category_count": len(cats)}))
            # Add edges from paper → each category
            for cat in cats:
                edges.append(GraphEdge(source=src.get("arxiv_id", ""), target=cat, relation="in_category"))

        # Add category nodes
        for cat in cat_set:
            nodes.append(GraphNode(id=cat, label=cat, type="category"))

        return GraphResponse(
            nodes=nodes,
            edges=edges,
            total=resp["hits"]["total"]["value"],
            took_ms=0,
            metadata={"min_categories": min_cats, "papers_returned": len(resp["hits"]["hits"])},
        )

    # ── 2. Co-authorship ego network ──

    async def _coauthor_network(
        self,
        gq: GraphQuery,
        sr: SearchRequest | None,
        emb: list[float] | None,
    ) -> GraphResponse:
        """Build the co-authorship graph around a given author.

        Finds all papers by the seed author (with filters applied), extracts
        co-authors, and optionally goes one hop deeper to find 2nd-degree
        collaborators.
        """
        seed = gq.seed_author
        if not seed:
            return GraphResponse(nodes=[], edges=[], total=0, took_ms=0,
                                 metadata={"error": "seed_author required"})

        depth = min(gq.depth or 1, 2)
        limit = min(gq.limit or 100, self.MAX_RESULTS)

        # Build the combined query: user filters + must match seed author
        base_q = self._base_query(sr, emb)
        author_clause = {
            "nested": {
                "path": "authors",
                "query": {"match_phrase": {"authors.name": seed}},
            }
        }
        combined = {"bool": {"must": [base_q, author_clause]}}

        # Aggregate co-authors from the seed author's papers
        aggs = {
            "coauthors": {
                "nested": {"path": "authors"},
                "aggs": {
                    "names": {
                        "terms": {"field": "authors.name.raw", "size": self.MAX_AGG_BUCKETS}
                    }
                },
            }
        }
        resp = await self._agg_search(combined, aggs, size=0, sr=sr, emb=emb)
        total_papers = resp["hits"]["total"]["value"]

        author_papers: dict[str, int] = {}
        for bucket in resp["aggregations"]["coauthors"]["names"]["buckets"]:
            author_papers[bucket["key"]] = bucket["doc_count"]

        # Build nodes and edges
        nodes: list[GraphNode] = []
        edges: list[GraphEdge] = []
        seen_authors: set[str] = set()

        # Seed author node — resolve canonical name from aggregation (case-insensitive)
        seed_lower = seed.lower()
        canonical_seed = seed
        seed_count = 0
        for k in list(author_papers.keys()):
            if k.lower() == seed_lower:
                if not seed_count:  # use first match as canonical name
                    canonical_seed = k
                seed_count += author_papers.pop(k)
        if not seed_count:
            seed_count = total_papers
        seed = canonical_seed
        nodes.append(GraphNode(
            id=seed, label=seed, type="author",
            properties={"paper_count": seed_count, "depth": 0},
        ))
        seen_authors.add(seed)

        # 1st-degree co-authors — reserve budget for 2nd-degree if depth>=2
        first_limit = max(1, limit // 2) if depth >= 2 else limit
        first_degree = sorted(author_papers.items(), key=lambda x: -x[1])[:first_limit]
        for name, count in first_degree:
            if name not in seen_authors and name.lower() != seed_lower:
                nodes.append(GraphNode(
                    id=name, label=name, type="author",
                    properties={"paper_count": count, "depth": 1},
                ))
                seen_authors.add(name)
            edges.append(GraphEdge(
                source=seed, target=name, relation="co_authored",
                weight=count,
            ))

        # Optional 2nd-degree expansion
        if depth >= 2 and first_degree:
            # Pick top co-authors to expand (limit to avoid explosion)
            to_expand = [name for name, _ in first_degree[:20]]
            for coauthor in to_expand:
                sub_clause = {
                    "nested": {
                        "path": "authors",
                        "query": {"match_phrase": {"authors.name": coauthor}},
                    }
                }
                sub_combined = {"bool": {"must": [base_q, sub_clause]}}
                sub_resp = await self._agg_search(sub_combined, aggs, size=0, sr=sr, emb=emb)
                for bucket in sub_resp["aggregations"]["coauthors"]["names"]["buckets"]:
                    name2 = bucket["key"]
                    if name2.lower() == coauthor.lower() or name2.lower() == seed_lower:
                        continue
                    if name2 not in seen_authors:
                        nodes.append(GraphNode(
                            id=name2, label=name2, type="author",
                            properties={"paper_count": bucket["doc_count"], "depth": 2},
                        ))
                        seen_authors.add(name2)
                    edges.append(GraphEdge(
                        source=coauthor, target=name2, relation="co_authored",
                        weight=bucket["doc_count"],
                    ))

        return GraphResponse(
            nodes=nodes[:limit], edges=[e for e in edges if e.source in {n.id for n in nodes[:limit]} and e.target in {n.id for n in nodes[:limit]}],
            total=total_papers, took_ms=0,
            metadata={"seed_author": seed, "depth": depth,
                       "unique_coauthors": len(seen_authors) - 1},
        )

    # ── 3. Author bridge detection ──

    async def _author_bridge(
        self,
        gq: GraphQuery,
        sr: SearchRequest | None,
        emb: list[float] | None,
    ) -> GraphResponse:
        """Find authors who publish across multiple distinct category groups.

        An author is a 'bridge' if they publish in categories A and B that
        rarely co-occur. This uses a two-step aggregation: get per-author
        category spread, then score by diversity.
        """
        min_cats = gq.min_categories or 3
        limit = min(gq.limit or 50, self.MAX_RESULTS)
        source_categories = gq.source_categories or []
        target_categories = gq.target_categories or []
        query = self._base_query(sr, emb)

        # If source/target categories are given, restrict to papers that
        # have at least one from either set
        if source_categories or target_categories:
            all_cats = list(set(source_categories + target_categories))
            query = {"bool": {"must": [query], "filter": [{"terms": {"categories": all_cats}}]}}

        aggs = {
            "authors_nested": {
                "nested": {"path": "authors"},
                "aggs": {
                    "by_author": {
                        "terms": {"field": "authors.name.raw", "size": self.MAX_AGG_BUCKETS},
                        "aggs": {
                            "back_to_root": {
                                "reverse_nested": {},
                                "aggs": {
                                    "cats": {
                                        "terms": {"field": "categories", "size": 50}
                                    }
                                },
                            }
                        },
                    }
                },
            }
        }
        resp = await self._agg_search(query, aggs, sr=sr, emb=emb)

        # Score each author by category diversity
        scored: list[tuple[str, int, list[str], int]] = []
        for bucket in resp["aggregations"]["authors_nested"]["by_author"]["buckets"]:
            name = bucket["key"]
            paper_count = bucket["doc_count"]
            cats = [c["key"] for c in bucket["back_to_root"]["cats"]["buckets"]]
            if len(cats) < min_cats:
                continue

            # If source/target specified, require presence in both
            if source_categories and target_categories:
                has_source = any(c in source_categories for c in cats)
                has_target = any(c in target_categories for c in cats)
                if not (has_source and has_target):
                    continue

            scored.append((name, len(cats), cats, paper_count))

        scored.sort(key=lambda x: (-x[1], -x[3]))
        scored = scored[:limit]

        nodes: list[GraphNode] = []
        edges: list[GraphEdge] = []
        cat_set: set[str] = set()

        for name, cat_count, cats, paper_count in scored:
            nodes.append(GraphNode(
                id=name, label=name, type="author",
                properties={
                    "category_count": cat_count,
                    "categories": cats,
                    "paper_count": paper_count,
                },
            ))
            for cat in cats:
                cat_set.add(cat)
                edges.append(GraphEdge(source=name, target=cat, relation="published_in"))

        for cat in cat_set:
            nodes.append(GraphNode(id=cat, label=cat, type="category"))

        return GraphResponse(
            nodes=nodes, edges=edges,
            total=len(scored), took_ms=0,
            metadata={"min_categories": min_cats},
        )

    # ── 4. Cross-category flow ──

    async def _cross_category_flow(
        self,
        gq: GraphQuery,
        sr: SearchRequest | None,
        emb: list[float] | None,
    ) -> GraphResponse:
        """Show how often category pairs co-occur on the same paper.

        Returns a weighted bipartite graph of categories. High-weight edges
        mean strong cross-pollination. Filter by source_categories /
        target_categories to focus on specific flows.
        """
        limit = min(gq.limit or 100, self.MAX_RESULTS)
        query = self._base_query(sr, emb)

        aggs = {
            "cats": {
                "terms": {"field": "categories", "size": self.MAX_AGG_BUCKETS},
                "aggs": {
                    "co_cats": {
                        "terms": {"field": "categories", "size": 50}
                    }
                },
            }
        }
        resp = await self._agg_search(query, aggs, sr=sr, emb=emb)

        # Build pair counts
        pair_counts: Counter[tuple[str, str]] = Counter()
        cat_counts: dict[str, int] = {}
        for bucket in resp["aggregations"]["cats"]["buckets"]:
            cat_a = bucket["key"]
            cat_counts[cat_a] = bucket["doc_count"]
            for co_bucket in bucket["co_cats"]["buckets"]:
                cat_b = co_bucket["key"]
                if cat_a == cat_b:
                    continue  # skip self-pair
                a, b = (cat_a, cat_b) if cat_a < cat_b else (cat_b, cat_a)
                pair_counts[(a, b)] = max(pair_counts.get((a, b), 0), co_bucket["doc_count"])

        # Filter by source/target if specified
        src_cats = set(gq.source_categories) if gq.source_categories else None
        tgt_cats = set(gq.target_categories) if gq.target_categories else None

        filtered_pairs = []
        for (a, b), count in pair_counts.most_common():
            if src_cats and tgt_cats:
                if not ((a in src_cats and b in tgt_cats) or (b in src_cats and a in tgt_cats)):
                    continue
            elif src_cats:
                if a not in src_cats and b not in src_cats:
                    continue
            elif tgt_cats:
                if a not in tgt_cats and b not in tgt_cats:
                    continue
            filtered_pairs.append((a, b, count))

        filtered_pairs = filtered_pairs[:limit]

        nodes: list[GraphNode] = []
        edges: list[GraphEdge] = []
        seen_cats: set[str] = set()

        for a, b, count in filtered_pairs:
            for cat in (a, b):
                if cat not in seen_cats:
                    nodes.append(GraphNode(
                        id=cat, label=cat, type="category",
                        properties={"paper_count": cat_counts.get(cat, 0)},
                    ))
                    seen_cats.add(cat)
            edges.append(GraphEdge(source=a, target=b, relation="co_occurs", weight=count))

        return GraphResponse(
            nodes=nodes, edges=edges,
            total=len(filtered_pairs), took_ms=0,
            metadata={"unique_categories": len(seen_cats)},
        )

    # ── 5. Interdisciplinary papers ──

    async def _interdisciplinary(
        self,
        gq: GraphQuery,
        sr: SearchRequest | None,
        emb: list[float] | None,
    ) -> GraphResponse:
        """Find interdisciplinary papers: those whose categories are rarely
        combined together.

        Two-step process:
        1. Compute global category co-occurrence frequencies
        2. Score each paper by the rarity of its category combinations
           (lower co-occurrence = more interdisciplinary)

        This is exactly the "papers cited across diverse subcategories" idea —
        papers sitting at unusual intersections.
        """
        limit = min(gq.limit or 50, self.MAX_RESULTS)
        min_cats = gq.min_categories or 2
        query = self._base_query(sr, emb)

        # Step 1: global category co-occurrence
        cooc_aggs = {
            "cats": {
                "terms": {"field": "categories", "size": self.MAX_AGG_BUCKETS},
                "aggs": {
                    "co_cats": {
                        "terms": {"field": "categories", "size": 100}
                    }
                },
            },
            "total": {"value_count": {"field": "arxiv_id"}},
        }
        cooc_resp = await self._agg_search({"match_all": {}}, cooc_aggs)

        total_papers = cooc_resp["aggregations"]["total"]["value"]
        cat_doc_count: dict[str, int] = {}
        pair_freq: dict[tuple[str, str], float] = {}

        # First pass: collect all category doc counts
        for bucket in cooc_resp["aggregations"]["cats"]["buckets"]:
            cat_doc_count[bucket["key"]] = bucket["doc_count"]

        # Second pass: compute pair co-occurrence frequencies
        for bucket in cooc_resp["aggregations"]["cats"]["buckets"]:
            cat_a = bucket["key"]
            for co_bucket in bucket["co_cats"]["buckets"]:
                cat_b = co_bucket["key"]
                if cat_a >= cat_b:
                    continue
                # Jaccard-like co-occurrence frequency
                pair = (cat_a, cat_b)
                co_count = co_bucket["doc_count"]
                union = cat_doc_count.get(cat_a, 1) + cat_doc_count.get(cat_b, 1) - co_count
                pair_freq[pair] = co_count / max(union, 1)

        # Step 2: find papers with multi-category tags, score by rarity
        # If semantic is active, pre-filter to semantically similar papers
        sem_ids = await self._semantic_prefilter(sr, emb, max_candidates=max(limit * 20, 500))
        inner_q = query
        if sem_ids is not None:
            inner_q = {"bool": {"must": [query], "filter": [{"terms": {"arxiv_id": sem_ids}}]}}

        body = {
            "query": {
                "function_score": {
                    "query": inner_q,
                    "script_score": {
                        "script": {"source": "doc['categories'].length"}
                    },
                    "boost_mode": "replace",
                }
            },
            "size": min(limit * 4, 800),  # oversample to filter
            "min_score": min_cats,
            "_source": ["arxiv_id", "title", "categories", "primary_category",
                         "authors", "submitted_date", "citation_stats", "has_github"],
        }
        papers_resp = await self._do_search(body) if sem_ids is not None else await self._do_search(body, sr, emb)

        # Score each paper by median rarity of its category pairs
        scored_papers = []
        for hit in papers_resp["hits"]["hits"]:
            src = hit["_source"]
            cats = src.get("categories") or []
            if len(cats) < min_cats:
                continue

            # Compute interdisciplinary score:
            # average (1 - co-occurrence) across all category pairs
            pairs = list(combinations(sorted(cats), 2))
            if not pairs:
                continue
            rarity_scores = []
            for a, b in pairs:
                freq = pair_freq.get((a, b), 0.0)
                rarity_scores.append(1.0 - freq)

            score = sum(rarity_scores) / len(rarity_scores)
            scored_papers.append((src, score, len(cats)))

        # Sort by interdisciplinary score descending
        scored_papers.sort(key=lambda x: -x[1])
        scored_papers = scored_papers[:limit]

        nodes: list[GraphNode] = []
        edges: list[GraphEdge] = []
        cat_set: set[str] = set()

        for src, score, cat_count in scored_papers:
            cats = src.get("categories") or []
            cat_set.update(cats)
            nodes.append(self._make_paper_node(src, {"category_count": cat_count, "interdisciplinary_score": round(score, 4)}))
            for cat in cats:
                edges.append(GraphEdge(
                    source=src.get("arxiv_id", ""), target=cat, relation="in_category",
                ))

        for cat in cat_set:
            nodes.append(GraphNode(
                id=cat, label=cat, type="category",
                properties={"paper_count": cat_doc_count.get(cat, 0)},
            ))

        return GraphResponse(
            nodes=nodes, edges=edges,
            total=len(scored_papers), took_ms=0,
            metadata={
                "min_categories": min_cats,
                "papers_scored": len(scored_papers),
            },
        )

    # ── 6. Rising interdisciplinary ──

    async def _rising_interdisciplinary(
        self,
        gq: GraphQuery,
        sr: SearchRequest | None,
        emb: list[float] | None,
    ) -> GraphResponse:
        """Find recent papers that are in the top N% most-cited within a
        citation window AND whose citers span many distinct categories.

        This is the "breakout interdisciplinary paper" detector:
        - Paper must be younger than `recency_months` (default 6)
        - Its citation count must be >= the `citation_percentile` threshold
          computed over all papers from the last `citation_window_years`
        - It must be cited by papers from >= `min_citing_categories` distinct
          categories (as recorded in `citation_stats.top_citing_categories`)
        - All standard SearchRequest filters apply on top

        Steps:
          1. Compute percentile threshold for citation count over the window
          2. Query for recent papers exceeding that threshold
          3. Filter & score by citing-category diversity
        """
        from datetime import datetime, timedelta, timezone

        limit = min(gq.limit or 50, self.MAX_RESULTS)
        percentile = gq.citation_percentile  # e.g. 90 → top 10%
        recency_months = gq.recency_months
        window_years = gq.citation_window_years
        min_citing_cats = gq.min_citing_categories

        now = datetime.now(timezone.utc)
        recency_cutoff = (now - timedelta(days=recency_months * 30)).isoformat()
        window_cutoff = (now - timedelta(days=window_years * 365)).isoformat()

        base_q = self._base_query(sr, emb)

        # Step 1: compute the citation percentile threshold
        # over papers from the last `window_years`
        percentile_agg = {
            "query": {
                "bool": {
                    "must": [{"match_all": {}}],
                    "filter": [
                        {"range": {"submitted_date": {"gte": window_cutoff}}},
                        {"range": {"citation_stats.total_citations": {"gt": 0}}},
                    ],
                }
            },
            "size": 0,
            "aggs": {
                "citation_pct": {
                    "percentiles": {
                        "field": "citation_stats.total_citations",
                        "percents": [percentile],
                    }
                },
                "total_with_citations": {
                    "value_count": {"field": "citation_stats.total_citations"}
                },
            },
        }
        pct_resp = await self._do_search(percentile_agg)
        pct_values = pct_resp["aggregations"]["citation_pct"]["values"]
        # ES returns keys like "90.0"
        threshold = pct_values.get(str(float(percentile)), 0)
        total_with_cites = pct_resp["aggregations"]["total_with_citations"]["value"]

        if threshold < 1:
            # Not enough enriched data to compute a meaningful percentile
            return GraphResponse(
                nodes=[], edges=[], total=0, took_ms=0,
                metadata={
                    "error": "insufficient_citation_data",
                    "detail": f"Only {int(total_with_cites)} papers have citation data in the {window_years}-year window. "
                              f"Run `python -m src.ingestion.enrich` to populate citation counts from Semantic Scholar.",
                    "citation_percentile": percentile,
                    "threshold": threshold,
                    "papers_with_citations": int(total_with_cites),
                },
            )

        # Step 2: find recent papers exceeding the threshold
        # Combine: base_q (user filters) + recency + citation threshold
        import math
        combined_filter = [
            {"range": {"submitted_date": {"gte": recency_cutoff}}},
            {"range": {"citation_stats.total_citations": {"gte": math.ceil(threshold)}}},
        ]
        combined_q = {
            "bool": {
                "must": [base_q],
                "filter": combined_filter,
            }
        }

        body = {
            "query": combined_q,
            "size": min(limit * 4, 800),  # oversample to filter by citing diversity
            "sort": [{"citation_stats.total_citations": {"order": "desc"}}],
            "_source": [
                "arxiv_id", "title", "categories", "primary_category",
                "authors", "submitted_date", "citation_stats", "has_github",
            ],
        }
        papers_resp = await self._do_search(body, sr, emb)

        # Step 3: filter by citing-category diversity & score
        scored: list[tuple[dict, int, list[str]]] = []
        for hit in papers_resp["hits"]["hits"]:
            src = hit["_source"]
            citing_cats = (src.get("citation_stats") or {}).get("top_citing_categories") or []
            if len(citing_cats) < min_citing_cats:
                continue
            scored.append((src, len(citing_cats), citing_cats))

        # Sort by number of distinct citing categories descending
        scored.sort(key=lambda x: (-x[1], -((x[0].get("citation_stats") or {}).get("total_citations") or 0)))
        scored = scored[:limit]

        nodes: list[GraphNode] = []
        edges: list[GraphEdge] = []
        cat_set: set[str] = set()

        for src, citing_cat_count, citing_cats in scored:
            cites = (src.get("citation_stats") or {}).get("total_citations") or 0
            own_cats = src.get("categories") or []
            cat_set.update(own_cats)
            cat_set.update(citing_cats)

            nodes.append(self._make_paper_node(src, {"total_citations": cites, "citing_category_count": citing_cat_count, "top_citing_categories": citing_cats}))
            # Edges: paper → its own categories
            for cat in own_cats:
                edges.append(GraphEdge(
                    source=src.get("arxiv_id", ""), target=cat, relation="in_category",
                ))
            # Edges: paper ← citing categories (shows where citations come from)
            for cat in citing_cats:
                edges.append(GraphEdge(
                    source=cat, target=src.get("arxiv_id", ""), relation="cited_from_category",
                ))

        # Add category nodes
        for cat in cat_set:
            nodes.append(GraphNode(id=cat, label=cat, type="category"))

        return GraphResponse(
            nodes=nodes, edges=edges,
            total=papers_resp["hits"]["total"]["value"], took_ms=0,
            metadata={
                "citation_threshold": int(threshold),
                "citation_percentile": percentile,
                "recency_months": recency_months,
                "citation_window_years": window_years,
                "min_citing_categories": min_citing_cats,
                "papers_with_citations_in_window": int(total_with_cites),
                "papers_matching": len(scored),
            },
        )

    # ── 7. Citation traversal ──

    async def _citation_traversal(
        self,
        gq: GraphQuery,
        sr: SearchRequest | None,
        emb: list[float] | None,
    ) -> GraphResponse:
        """Traverse citation links: given a set of seed papers (defined by
        search filters or a specific paper), follow their references or
        citers and analyze the resulting set.

        direction='references'  → "what do these papers cite?"
        direction='cited_by'    → "what papers cite these?"

        The traversed set is then aggregated by category/author/year.
        This enables queries like:
          - "among all papers cited by transformer papers from 2024, which
            categories appear most?"
          - "who cites adversarial robustness papers? show me by field"
        """
        limit = min(gq.limit or 50, self.MAX_RESULTS)
        direction = gq.direction  # "references" or "cited_by"
        field = "reference_ids" if direction == "references" else "cited_by_ids"
        aggregate_by = gq.aggregate_by  # "category", "author", "year"

        # Step 1: find seed papers
        if gq.seed_arxiv_id:
            # Single paper seed
            seed_query = {"term": {"arxiv_id": gq.seed_arxiv_id}}
        elif gq.seed_arxiv_ids:
            seed_query = {"terms": {"arxiv_id": gq.seed_arxiv_ids[:10000]}}
        else:
            base = self._base_query(sr, emb)
            # Require seeds to actually have citation link data
            seed_query = {
                "bool": {
                    "must": [base],
                    "filter": [{"exists": {"field": field}}],
                }
            }

        # Fetch seed papers and their citation links
        seed_body = {
            "query": seed_query,
            "size": min(limit * 2, 10000),
            "_source": ["arxiv_id", "title", "categories", "primary_category",
                         "authors", "submitted_date", "citation_stats", "has_github", field],
        }
        seed_resp = await self._do_search(seed_body, sr, emb)

        seed_hits = seed_resp["hits"]["hits"]
        if not seed_hits:
            return GraphResponse(
                nodes=[], edges=[], total=0, took_ms=0,
                metadata={"error": "no_seed_papers", "direction": direction},
            )

        # Collect all linked arxiv IDs
        linked_ids: set[str] = set()
        seed_papers: list[dict] = []
        for hit in seed_hits:
            src = hit["_source"]
            ids = src.get(field, []) or []
            linked_ids.update(ids)
            seed_papers.append(src)

        if not linked_ids:
            return GraphResponse(
                nodes=[], edges=[], total=0, took_ms=0,
                metadata={
                    "error": "no_citation_links",
                    "detail": f"Seed papers have no {field}. Run enrichment: python scripts/enrich_bulk.py",
                    "seed_papers": len(seed_papers),
                    "direction": direction,
                },
            )

        # Step 2: fetch the linked papers from our index
        linked_query = {
            "query": {"terms": {"arxiv_id": list(linked_ids)[:10000]}},
            "size": min(len(linked_ids), 10000),
            "_source": ["arxiv_id", "title", "categories", "primary_category",
                         "authors", "submitted_date", "citation_stats"],
        }
        linked_resp = await self._do_search(linked_query)

        traversed = []
        for hit in linked_resp["hits"]["hits"]:
            traversed.append(hit["_source"])

        # Step 3: aggregate
        nodes: list[GraphNode] = []
        edges: list[GraphEdge] = []

        # Add seed papers as nodes
        for sp in seed_papers[:500]:
            nodes.append(self._make_paper_node(sp, {
                "role": "seed",
                f"{field}_count": len(sp.get(field) or []),
            }))

        relation = "cites" if direction == "references" else "cited_by"

        if aggregate_by in ("author", "year"):
            nodes.append(GraphNode(
                id="seed_set", label="Seed Papers", type="group",
                properties={"paper_count": len(seed_papers)},
            ))

        if aggregate_by == "category":
            # Count by category across all traversed papers
            cat_counts: dict[str, int] = {}
            cat_papers: dict[str, list[str]] = defaultdict(list)
            for tp in traversed:
                for cat in tp.get("categories") or []:
                    cat_counts[cat] = cat_counts.get(cat, 0) + 1
                    cat_papers[cat].append(tp.get("arxiv_id", ""))

            sorted_cats = sorted(cat_counts.items(), key=lambda x: -x[1])[:limit]
            for cat, count in sorted_cats:
                nodes.append(GraphNode(
                    id=cat, label=cat, type="category",
                    properties={"paper_count": count},
                ))
            # Edges from seed → category (aggregated)
            for sp in seed_papers[:500]:
                ref_set = set(sp.get(field) or [])
                for cat, _ in sorted_cats:
                    overlap = len(ref_set & set(cat_papers[cat]))
                    if overlap > 0:
                        edges.append(GraphEdge(
                            source=sp.get("arxiv_id", ""), target=cat,
                            relation=relation, weight=overlap,
                        ))

        elif aggregate_by == "author":
            author_counts: dict[str, int] = {}
            for tp in traversed:
                for a in tp.get("authors") or []:
                    name = a.get("name", "") if isinstance(a, dict) else (str(a) if a is not None else "")
                    if name:
                        author_counts[name] = author_counts.get(name, 0) + 1

            sorted_authors = sorted(author_counts.items(), key=lambda x: -x[1])[:limit]
            for name, count in sorted_authors:
                nodes.append(GraphNode(
                    id=name, label=name, type="author",
                    properties={"paper_count": count},
                ))
                edges.append(GraphEdge(
                    source="seed_set", target=name,
                    relation=f"{relation}_author", weight=count,
                ))

        elif aggregate_by == "year":
            year_counts: dict[str, int] = {}
            for tp in traversed:
                sd = tp.get("submitted_date", "")
                if sd:
                    yr = str(sd)[:4]
                    year_counts[yr] = year_counts.get(yr, 0) + 1

            sorted_years = sorted(year_counts.items())
            for yr, count in sorted_years:
                nodes.append(GraphNode(
                    id=yr, label=yr, type="year",
                    properties={"paper_count": count},
                ))
                edges.append(GraphEdge(
                    source="seed_set", target=yr,
                    relation=f"{relation}_year", weight=count,
                ))

        return GraphResponse(
            nodes=nodes, edges=edges,
            total=len(traversed), took_ms=0,
            metadata={
                "direction": direction,
                "aggregate_by": aggregate_by,
                "seed_papers": len(seed_papers),
                "linked_ids_found": len(linked_ids),
                "traversed_in_index": len(traversed),
                "coverage": round(len(traversed) / max(len(linked_ids), 1), 3),
            },
        )

    # ── 8. Paper citation network ──

    async def _paper_citation_network(
        self,
        gq: GraphQuery,
        sr: SearchRequest | None,
        emb: list[float] | None,
    ) -> GraphResponse:
        """Direct paper-to-paper citation graph.  Unlike citation_traversal
        (which aggregates), this returns actual paper nodes connected by
        citation edges — ideal for network visualization."""
        limit = min(gq.limit or 50, self.MAX_RESULTS)
        direction = gq.direction
        field = "reference_ids" if direction == "references" else "cited_by_ids"

        # Step 1: find seed papers
        if gq.seed_arxiv_id:
            seed_query = {"term": {"arxiv_id": gq.seed_arxiv_id}}
        elif gq.seed_arxiv_ids:
            seed_query = {"terms": {"arxiv_id": gq.seed_arxiv_ids[:10000]}}
        else:
            base = self._base_query(sr, emb)
            seed_query = {
                "bool": {"must": [base], "filter": [{"exists": {"field": field}}]}
            }

        seed_body: dict[str, Any] = {
            "query": seed_query,
            "size": min(limit * 10, 10000),
            "_source": [
                "arxiv_id", "title", "categories", "primary_category",
                "authors", "submitted_date", "citation_stats", "has_github", field,
            ],
        }
        seed_resp = await self._do_search(seed_body, sr, emb)
        seed_hits = seed_resp["hits"]["hits"]
        if not seed_hits:
            return GraphResponse(
                nodes=[], edges=[], total=0, took_ms=0,
                metadata={"error": "no_seed_papers"},
            )

        linked_ids: set[str] = set()
        seed_papers: list[dict] = []
        for hit in seed_hits:
            src = hit["_source"]
            ids = src.get(field, []) or []
            linked_ids.update(ids)
            seed_papers.append(src)

        if not linked_ids:
            return GraphResponse(
                nodes=[], edges=[], total=0, took_ms=0,
                metadata={"error": "no_citation_links", "seed_papers": len(seed_papers)},
            )

        # Step 2: fetch linked papers
        linked_resp = await self._do_search({
            "query": {"terms": {"arxiv_id": list(linked_ids)[:10000]}},
            "size": min(len(linked_ids), limit * 5, 500),
            "_source": [
                "arxiv_id", "title", "categories", "primary_category",
                "authors", "submitted_date", "citation_stats", "has_github",
            ],
        })
        linked_map: dict[str, dict] = {}
        for hit in linked_resp["hits"]["hits"]:
            s = hit["_source"]
            linked_map[s.get("arxiv_id", "")] = s

        nodes: list[GraphNode] = []
        edges: list[GraphEdge] = []
        seen: set[str] = set()

        for sp in seed_papers:
            aid = sp.get("arxiv_id", "")
            if aid not in seen:
                seen.add(aid)
                nodes.append(self._make_paper_node(sp, {"role": "seed"}))

        edge_count = 0
        for sp in seed_papers:
            sp_id = sp.get("arxiv_id", "")
            for ref_id in set(sp.get(field, []) or []):
                if ref_id in linked_map and ref_id != sp_id and edge_count < limit * 10:
                    lp = linked_map[ref_id]
                    if ref_id not in seen:
                        seen.add(ref_id)
                        nodes.append(self._make_paper_node(lp, {"role": "linked"}))
                    src_id = sp.get("arxiv_id", "") if direction == "references" else ref_id
                    tgt_id = ref_id if direction == "references" else sp.get("arxiv_id", "")
                    edges.append(GraphEdge(source=src_id, target=tgt_id, relation="cites"))
                    edge_count += 1

        trimmed = nodes[:limit * 3]
        trimmed_ids = {n.id for n in trimmed}
        return GraphResponse(
            nodes=trimmed, edges=[e for e in edges if e.source in trimmed_ids and e.target in trimmed_ids],
            total=len(linked_ids), took_ms=0,
            metadata={
                "direction": direction,
                "seed_papers": len(seed_papers),
                "linked_found_in_index": len(linked_map),
                "total_linked_ids": len(linked_ids),
            },
        )

    # ── 9. Author influence ──

    async def _author_influence(
        self,
        gq: GraphQuery,
        sr: SearchRequest | None,
        emb: list[float] | None,
    ) -> GraphResponse:
        """Author-to-author influence graph.  For a seed author, follow
        citation links (cited_by or references) from their papers and
        aggregate by author — producing a weighted author→author network."""
        seed = gq.seed_author
        if not seed:
            return GraphResponse(
                nodes=[], edges=[], total=0, took_ms=0,
                metadata={"error": "seed_author required"},
            )

        limit = min(gq.limit or 50, self.MAX_RESULTS)
        direction = gq.direction
        field = "reference_ids" if direction == "references" else "cited_by_ids"

        base_q = self._base_query(sr, emb)
        author_clause = {
            "nested": {
                "path": "authors",
                "query": {"match_phrase": {"authors.name": seed}},
            }
        }
        combined = {
            "bool": {
                "must": [base_q, author_clause],
                "filter": [{"exists": {"field": field}}],
            }
        }

        seed_body: dict[str, Any] = {
            "query": combined,
            "size": 200,
            "_source": ["arxiv_id", field],
        }
        seed_resp = await self._do_search(seed_body, sr, emb)

        linked_ids: set[str] = set()
        for hit in seed_resp["hits"]["hits"]:
            ids = hit["_source"].get(field, []) or []
            linked_ids.update(ids)

        if not linked_ids:
            return GraphResponse(
                nodes=[], edges=[], total=0, took_ms=0,
                metadata={
                    "error": "no_citation_links",
                    "seed_papers": len(seed_resp["hits"]["hits"]),
                },
            )

        aggs = {
            "authors_nested": {
                "nested": {"path": "authors"},
                "aggs": {
                    "by_author": {
                        "terms": {"field": "authors.name.raw", "size": self.MAX_AGG_BUCKETS}
                    }
                },
            }
        }
        linked_resp = await self._agg_search(
            {"terms": {"arxiv_id": list(linked_ids)[:10000]}}, aggs,
        )

        nodes: list[GraphNode] = [
            GraphNode(
                id=seed, label=seed, type="author",
                properties={"role": "seed", "paper_count": len(seed_resp["hits"]["hits"])},
            )
        ]
        edges: list[GraphEdge] = []

        for bucket in linked_resp["aggregations"]["authors_nested"]["by_author"]["buckets"][:limit]:
            name = bucket["key"]
            if name.lower() == seed.lower():
                continue
            nodes.append(GraphNode(
                id=name, label=name, type="author",
                properties={"paper_count": bucket["doc_count"]},
            ))
            if direction == "references":
                # Seed references their work → they influenced seed
                edges.append(GraphEdge(source=name, target=seed, relation="influences", weight=bucket["doc_count"]))
            else:
                # Their work cites seed → seed influenced them
                edges.append(GraphEdge(source=seed, target=name, relation="influences", weight=bucket["doc_count"]))

        return GraphResponse(
            nodes=nodes, edges=edges,
            total=len(linked_ids), took_ms=0,
            metadata={
                "seed_author": seed,
                "direction": direction,
                "seed_papers": len(seed_resp["hits"]["hits"]),
                "total_linked_ids": len(linked_ids),
                "unique_influencing_authors": len(nodes) - 1,
            },
        )

    # ── 10. Temporal evolution ──

    async def _temporal_evolution(
        self,
        gq: GraphQuery,
        sr: SearchRequest | None,
        emb: list[float] | None,
    ) -> GraphResponse:
        """Category publication volume over time.  Returns a bipartite graph
        of category nodes and time-period nodes, with edges weighted by
        paper count — ideal for visualizing field growth trends."""
        limit = min(gq.limit or 50, self.MAX_RESULTS)
        interval = gq.time_interval
        es_interval = {"month": "month", "quarter": "quarter", "year": "year"}[interval]
        query = self._base_query(sr, emb)

        aggs = {
            "over_time": {
                "date_histogram": {
                    "field": "submitted_date",
                    "calendar_interval": es_interval,
                },
                "aggs": {
                    "by_category": {
                        "terms": {"field": "categories", "size": 30}
                    }
                },
            }
        }
        resp = await self._agg_search(query, aggs, sr=sr, emb=emb)

        nodes: list[GraphNode] = []
        edges: list[GraphEdge] = []
        seen_cats: set[str] = set()
        seen_times: set[str] = set()

        for tb in resp["aggregations"]["over_time"]["buckets"]:
            time_key = tb["key_as_string"]
            if time_key not in seen_times:
                seen_times.add(time_key)
                nodes.append(GraphNode(
                    id=time_key, label=time_key, type="time",
                    properties={"total_papers": tb["doc_count"]},
                ))
            for cb in tb["by_category"]["buckets"][:limit]:
                cat = cb["key"]
                if cat not in seen_cats:
                    seen_cats.add(cat)
                    nodes.append(GraphNode(id=cat, label=cat, type="category"))
                edges.append(GraphEdge(
                    source=cat, target=time_key,
                    relation="published_in", weight=cb["doc_count"],
                ))

        return GraphResponse(
            nodes=nodes, edges=edges,
            total=len(seen_times), took_ms=0,
            metadata={
                "interval": interval,
                "time_periods": len(seen_times),
                "unique_categories": len(seen_cats),
            },
        )

    # ── 11. Paper similarity ──

    async def _paper_similarity(
        self,
        gq: GraphQuery,
        sr: SearchRequest | None,
        emb: list[float] | None,
    ) -> GraphResponse:
        """Semantic similarity graph.  Uses abstract embeddings to connect
        papers above a cosine-similarity threshold — great for discovering
        hidden connections between papers that share semantic content."""
        import numpy as np

        limit = min(gq.limit or 50, self.MAX_RESULTS)
        threshold = gq.similarity_threshold

        if not _ctx_first_boost_emb.get():
            return GraphResponse(
                nodes=[], edges=[], total=0, took_ms=0,
                metadata={"error": "semantic boost query required for paper_similarity"},
            )

        query = self._base_query(sr, emb)
        body: dict[str, Any] = {
            "query": query,
            "size": limit,
            "_source": [
                "arxiv_id", "title", "categories", "primary_category",
                "authors", "submitted_date", "citation_stats", "has_github",
                "abstract_embedding",
            ],
        }
        resp = await self._do_search(body, sr, emb)

        papers: list[tuple[dict, list[float]]] = []
        for hit in resp["hits"]["hits"]:
            src = hit["_source"]
            vec = src.get("abstract_embedding")
            if vec:
                papers.append((src, vec))

        if len(papers) < 2:
            return GraphResponse(
                nodes=[], edges=[], total=0, took_ms=0,
                metadata={"error": "not enough papers with embeddings"},
            )

        nodes: list[GraphNode] = []
        for src, _ in papers:
            nodes.append(self._make_paper_node(src))

        # Compute pairwise cosine similarity
        vecs = np.array([v for _, v in papers])
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        normed = vecs / norms
        sim_matrix = normed @ normed.T

        edges: list[GraphEdge] = []
        for i in range(len(papers)):
            for j in range(i + 1, len(papers)):
                sim = float(sim_matrix[i][j])
                if sim >= threshold:
                    edges.append(GraphEdge(
                        source=papers[i][0].get("arxiv_id", ""),
                        target=papers[j][0].get("arxiv_id", ""),
                        relation="similar_to",
                        weight=round(sim, 4),
                    ))

        return GraphResponse(
            nodes=nodes, edges=edges,
            total=len(papers), took_ms=0,
            metadata={
                "similarity_threshold": threshold,
                "papers_with_embeddings": len(papers),
                "edges_above_threshold": len(edges),
            },
        )

    # ── 12. Domain collaboration ──

    async def _domain_collaboration(
        self,
        gq: GraphQuery,
        sr: SearchRequest | None,
        emb: list[float] | None,
    ) -> GraphResponse:
        """Domain-level collaboration network.  Shows how top-level ArXiv
        domains (cs, physics, math, q-bio …) co-occur on the same papers —
        a higher-level view than cross_category_flow."""
        limit = min(gq.limit or 50, self.MAX_RESULTS)
        query = self._base_query(sr, emb)

        aggs = {
            "domains": {
                "terms": {"field": "domains", "size": 50},
                "aggs": {
                    "co_domains": {
                        "terms": {"field": "domains", "size": 50}
                    }
                },
            }
        }
        resp = await self._agg_search(query, aggs, sr=sr, emb=emb)

        domain_counts: dict[str, int] = {}
        pair_counts: Counter[tuple[str, str]] = Counter()

        for bucket in resp["aggregations"]["domains"]["buckets"]:
            dom_a = bucket["key"]
            domain_counts[dom_a] = bucket["doc_count"]
            for co_bucket in bucket["co_domains"]["buckets"]:
                dom_b = co_bucket["key"]
                if dom_a == dom_b:
                    continue
                a, b = (dom_a, dom_b) if dom_a < dom_b else (dom_b, dom_a)
                pair_counts[(a, b)] = max(pair_counts.get((a, b), 0), co_bucket["doc_count"])

        # Optional source/target filter (reuse category params for domains)
        src_doms = set(gq.source_categories) if gq.source_categories else None
        tgt_doms = set(gq.target_categories) if gq.target_categories else None

        filtered: list[tuple[str, str, int]] = []
        for (a, b), count in pair_counts.most_common():
            if src_doms and tgt_doms:
                if not ((a in src_doms and b in tgt_doms) or (b in src_doms and a in tgt_doms)):
                    continue
            elif src_doms:
                if a not in src_doms and b not in src_doms:
                    continue
            elif tgt_doms:
                if a not in tgt_doms and b not in tgt_doms:
                    continue
            filtered.append((a, b, count))
        filtered = filtered[:limit]

        nodes: list[GraphNode] = []
        edges: list[GraphEdge] = []
        seen: set[str] = set()

        for a, b, count in filtered:
            for dom in (a, b):
                if dom not in seen:
                    seen.add(dom)
                    nodes.append(GraphNode(
                        id=dom, label=dom, type="domain",
                        properties={"paper_count": domain_counts.get(dom, 0)},
                    ))
            edges.append(GraphEdge(source=a, target=b, relation="co_occurs", weight=count))

        return GraphResponse(
            nodes=nodes, edges=edges,
            total=len(filtered), took_ms=0,
            metadata={"unique_domains": len(seen)},
        )

    # ── 13. Author topic evolution ──

    async def _author_topic_evolution(
        self,
        gq: GraphQuery,
        sr: SearchRequest | None,
        emb: list[float] | None,
    ) -> GraphResponse:
        """How an author's research topics evolve over time.  Returns a
        tripartite graph: author → time-period → category, showing the
        category distribution per time bucket."""
        limit = min(gq.limit or 50, self.MAX_RESULTS)
        seed = gq.seed_author
        if not seed:
            return GraphResponse(
                nodes=[], edges=[], total=0, took_ms=0,
                metadata={"error": "seed_author required"},
            )

        interval = gq.time_interval
        es_interval = {"month": "month", "quarter": "quarter", "year": "year"}[interval]

        base_q = self._base_query(sr, emb)
        author_clause = {
            "nested": {
                "path": "authors",
                "query": {"match_phrase": {"authors.name": seed}},
            }
        }
        combined = {"bool": {"must": [base_q, author_clause]}}

        aggs = {
            "over_time": {
                "date_histogram": {
                    "field": "submitted_date",
                    "calendar_interval": es_interval,
                },
                "aggs": {
                    "by_category": {
                        "terms": {"field": "categories", "size": 30}
                    }
                },
            },
            "total_papers": {"value_count": {"field": "arxiv_id"}},
        }
        resp = await self._agg_search(combined, aggs, sr=sr, emb=emb)

        total = int(resp["aggregations"]["total_papers"]["value"])
        nodes: list[GraphNode] = [
            GraphNode(
                id=seed, label=seed, type="author",
                properties={"total_papers": total},
            )
        ]
        edges: list[GraphEdge] = []
        seen_cats: set[str] = set()
        seen_times: set[str] = set()

        for tb in resp["aggregations"]["over_time"]["buckets"]:
            time_key = tb["key_as_string"]
            if time_key not in seen_times:
                seen_times.add(time_key)
                nodes.append(GraphNode(
                    id=time_key, label=time_key, type="time",
                    properties={"paper_count": tb["doc_count"]},
                ))
                edges.append(GraphEdge(
                    source=seed, target=time_key,
                    relation="published_in", weight=tb["doc_count"],
                ))
            for cb in tb["by_category"]["buckets"]:
                cat = cb["key"]
                if cat not in seen_cats:
                    seen_cats.add(cat)
                    nodes.append(GraphNode(id=cat, label=cat, type="category"))
                edges.append(GraphEdge(
                    source=time_key, target=cat,
                    relation="researched", weight=cb["doc_count"],
                ))

        return GraphResponse(
            nodes=nodes, edges=edges,
            total=total, took_ms=0,
            metadata={
                "seed_author": seed,
                "interval": interval,
                "time_periods": len(seen_times),
                "unique_categories": len(seen_cats),
            },
        )

    # ── 14. GitHub landscape ──

    async def _github_landscape(
        self,
        gq: GraphQuery,
        sr: SearchRequest | None,
        emb: list[float] | None,
    ) -> GraphResponse:
        """GitHub code-availability landscape.  Shows which categories and
        domains have the most papers with GitHub links, plus adoption rates
        and trends over time."""
        limit = min(gq.limit or 50, self.MAX_RESULTS)
        query = self._base_query(sr, emb)
        interval = gq.time_interval
        es_interval = {"month": "month", "quarter": "quarter", "year": "year"}[interval]

        github_query = {
            "bool": {"must": [query], "filter": [{"term": {"has_github": True}}]}
        }

        aggs = {
            "by_category": {
                "terms": {"field": "categories", "size": self.MAX_AGG_BUCKETS},
            },
            "by_domain": {
                "terms": {"field": "domains", "size": 50},
            },
            "over_time": {
                "date_histogram": {
                    "field": "submitted_date",
                    "calendar_interval": es_interval,
                },
            },
            "total_github": {"value_count": {"field": "arxiv_id"}},
        }
        resp = await self._agg_search(github_query, aggs, sr=sr, emb=emb)

        # Get total papers per category for adoption-rate computation
        total_aggs = {
            "by_category": {
                "terms": {"field": "categories", "size": self.MAX_AGG_BUCKETS},
            },
        }
        total_resp = await self._agg_search(query, total_aggs, sr=sr, emb=emb)

        total_by_cat: dict[str, int] = {}
        for b in total_resp["aggregations"]["by_category"]["buckets"]:
            total_by_cat[b["key"]] = b["doc_count"]

        nodes: list[GraphNode] = []
        edges: list[GraphEdge] = []

        cat_buckets = resp["aggregations"]["by_category"]["buckets"][:limit]
        for b in cat_buckets:
            cat = b["key"]
            github_count = b["doc_count"]
            total = total_by_cat.get(cat, github_count)
            rate = round(github_count / max(total, 1), 4)
            nodes.append(GraphNode(
                id=cat, label=cat, type="category",
                properties={
                    "github_papers": github_count,
                    "total_papers": total,
                    "github_rate": rate,
                },
            ))

        cat_ids = {n.id for n in nodes}
        for b in resp["aggregations"]["by_domain"]["buckets"]:
            if b["key"] not in cat_ids:
                nodes.append(GraphNode(
                    id=b["key"], label=b["key"], type="domain",
                    properties={"github_papers": b["doc_count"]},
                ))

        for tb in resp["aggregations"]["over_time"]["buckets"]:
            time_key = tb["key_as_string"]
            nodes.append(GraphNode(
                id=f"time_{time_key}", label=time_key, type="time",
                properties={"github_papers": tb["doc_count"]},
            ))

        # Edges: category → its domain
        domain_ids = {n.id for n in nodes if n.type == "domain"}
        for b in cat_buckets:
            cat = b["key"]
            domain = cat.split(".")[0] if "." in cat else cat
            if domain in domain_ids and domain != cat:
                edges.append(GraphEdge(
                    source=cat, target=domain,
                    relation="belongs_to", weight=b["doc_count"],
                ))

        return GraphResponse(
            nodes=nodes, edges=edges,
            total=int(resp["aggregations"]["total_github"]["value"]), took_ms=0,
            metadata={
                "total_github_papers": int(resp["aggregations"]["total_github"]["value"]),
                "categories_with_github": len(cat_buckets),
                "interval": interval,
            },
        )

    # ── 15. Bibliographic coupling ──

    async def _bibliographic_coupling(
        self,
        gq: GraphQuery,
        sr: SearchRequest | None,
        emb: list[float] | None,
    ) -> GraphResponse:
        """Bibliographic coupling: papers sharing references.  Two papers
        with many shared references are likely studying the same problem.
        Edge weight = number of shared reference IDs."""
        limit = min(gq.limit or 50, self.MAX_RESULTS)

        if gq.seed_arxiv_ids:
            seed_query = {"terms": {"arxiv_id": gq.seed_arxiv_ids[:10000]}}
        elif gq.seed_arxiv_id:
            # Fetch the seed paper first, then find papers sharing its references
            seed_body: dict[str, Any] = {
                "query": {"term": {"arxiv_id": gq.seed_arxiv_id}},
                "size": 1,
                "_source": ["arxiv_id", "reference_ids"],
            }
            seed_resp = await self._do_search(seed_body, sr, emb)
            seed_refs: list[str] = []
            for hit in seed_resp["hits"]["hits"]:
                seed_refs = hit["_source"].get("reference_ids", []) or []
            if not seed_refs:
                return GraphResponse(
                    nodes=[], edges=[], total=0, took_ms=0,
                    metadata={"error": "seed paper has no references for coupling"},
                )
            # Find papers that reference at least one of the same references
            seed_query = {
                "bool": {
                    "filter": [{"terms": {"reference_ids": seed_refs}}],
                    "must": [{"exists": {"field": "reference_ids"}}],
                }
            }
        else:
            base = self._base_query(sr, emb)
            seed_query = {
                "bool": {"must": [base], "filter": [{"exists": {"field": "reference_ids"}}]}
            }

        body: dict[str, Any] = {
            "query": seed_query,
            "size": limit,
            "_source": [
                "arxiv_id", "title", "categories", "primary_category",
                "authors", "submitted_date", "citation_stats", "has_github", "reference_ids",
            ],
        }
        resp = await self._do_search(body, sr, emb)

        papers: list[tuple[dict, set[str]]] = []
        for hit in resp["hits"]["hits"]:
            src = hit["_source"]
            refs = src.get("reference_ids", []) or []
            if refs:
                papers.append((src, set(refs)))

        if len(papers) < 2:
            return GraphResponse(
                nodes=[], edges=[], total=0, took_ms=0,
                metadata={"error": "need at least 2 papers with references"},
            )

        nodes: list[GraphNode] = []
        for src, refs in papers:
            nodes.append(self._make_paper_node(src, {"reference_count": len(refs)}))

        edges: list[GraphEdge] = []
        for i in range(len(papers)):
            for j in range(i + 1, len(papers)):
                shared = len(papers[i][1] & papers[j][1])
                if shared > 0:
                    edges.append(GraphEdge(
                        source=papers[i][0].get("arxiv_id", ""),
                        target=papers[j][0].get("arxiv_id", ""),
                        relation="bibliographic_coupling",
                        weight=shared,
                    ))

        return GraphResponse(
            nodes=nodes, edges=edges,
            total=len(papers), took_ms=0,
            metadata={
                "papers_analyzed": len(papers),
                "coupling_edges": len(edges),
            },
        )

    # ── 16. Co-citation ──

    async def _cocitation(
        self,
        gq: GraphQuery,
        sr: SearchRequest | None,
        emb: list[float] | None,
    ) -> GraphResponse:
        """Co-citation analysis: papers frequently cited together.  Two
        papers have a co-citation edge when many other papers cite both.
        Edge weight = number of co-citing papers."""
        limit = min(gq.limit or 50, self.MAX_RESULTS)

        if gq.seed_arxiv_ids:
            seed_query = {"terms": {"arxiv_id": gq.seed_arxiv_ids[:10000]}}
        elif gq.seed_arxiv_id:
            # Fetch the seed paper first, then find papers cited alongside it
            seed_body: dict[str, Any] = {
                "query": {"term": {"arxiv_id": gq.seed_arxiv_id}},
                "size": 1,
                "_source": ["arxiv_id", "cited_by_ids"],
            }
            seed_resp = await self._do_search(seed_body, sr, emb)
            seed_citers: list[str] = []
            for hit in seed_resp["hits"]["hits"]:
                seed_citers = hit["_source"].get("cited_by_ids", []) or []
            seed_citers = seed_citers[:10000]
            if not seed_citers:
                return GraphResponse(
                    nodes=[], edges=[], total=0, took_ms=0,
                    metadata={"error": "seed paper has no citations for co-citation"},
                )
            # Find papers also cited by the same papers
            seed_query = {
                "bool": {
                    "filter": [{"terms": {"cited_by_ids": seed_citers}}],
                    "must": [{"exists": {"field": "cited_by_ids"}}],
                }
            }
        else:
            base = self._base_query(sr, emb)
            seed_query = {
                "bool": {"must": [base], "filter": [{"exists": {"field": "cited_by_ids"}}]}
            }

        body: dict[str, Any] = {
            "query": seed_query,
            "size": limit,
            "_source": [
                "arxiv_id", "title", "categories", "primary_category",
                "authors", "submitted_date", "citation_stats", "has_github", "cited_by_ids",
            ],
        }
        resp = await self._do_search(body, sr, emb)

        papers: list[tuple[dict, set[str]]] = []
        for hit in resp["hits"]["hits"]:
            src = hit["_source"]
            citers = src.get("cited_by_ids", []) or []
            if citers:
                papers.append((src, set(citers)))

        if len(papers) < 2:
            return GraphResponse(
                nodes=[], edges=[], total=0, took_ms=0,
                metadata={"error": "need at least 2 papers with citation data"},
            )

        nodes: list[GraphNode] = []
        for src, citers in papers:
            nodes.append(self._make_paper_node(src))

        edges: list[GraphEdge] = []
        for i in range(len(papers)):
            for j in range(i + 1, len(papers)):
                shared = len(papers[i][1] & papers[j][1])
                if shared > 0:
                    edges.append(GraphEdge(
                        source=papers[i][0].get("arxiv_id", ""),
                        target=papers[j][0].get("arxiv_id", ""),
                        relation="cocitation",
                        weight=shared,
                    ))

        return GraphResponse(
            nodes=nodes, edges=edges,
            total=len(papers), took_ms=0,
            metadata={
                "papers_analyzed": len(papers),
                "cocitation_edges": len(edges),
            },
        )

    # ── 17. Multi-hop citation traversal ──

    async def _multihop_citation(
        self,
        gq: GraphQuery,
        sr: SearchRequest | None,
        emb: list[float] | None,
    ) -> GraphResponse:
        """Walk N hops along citation links (up to 5).  Unlike citation_traversal
        which aggregates, this returns the actual paper graph at each hop depth.
        Papers are tagged with their hop distance from the seed set."""
        max_hops = min(gq.max_hops, 50)
        limit = min(gq.limit or 50, self.MAX_RESULTS)
        direction = gq.direction
        field = "reference_ids" if direction == "references" else "cited_by_ids"

        # Step 1: find seed papers
        if gq.seed_arxiv_id:
            seed_query: dict[str, Any] = {"term": {"arxiv_id": gq.seed_arxiv_id}}
        elif gq.seed_arxiv_ids:
            seed_query = {"terms": {"arxiv_id": gq.seed_arxiv_ids[:10000]}}
        else:
            base = self._base_query(sr, emb)
            seed_query = {
                "bool": {"must": [base], "filter": [{"exists": {"field": field}}]}
            }

        seed_body: dict[str, Any] = {
            "query": seed_query,
            "size": min(limit * 10, 10000),
            "_source": ["arxiv_id", "title", "categories", "primary_category",
                         "authors", "submitted_date", "citation_stats", "has_github", field],
        }
        seed_resp = await self._do_search(seed_body, sr, emb)
        if not seed_resp["hits"]["hits"]:
            return GraphResponse(nodes=[], edges=[], total=0, took_ms=0,
                                 metadata={"error": "no_seed_papers"})

        _FIELDS = ["arxiv_id", "title", "categories", "primary_category",
                    "authors", "submitted_date", "citation_stats", "has_github"]

        nodes: list[GraphNode] = []
        edges: list[GraphEdge] = []
        seen: dict[str, int] = {}  # arxiv_id → hop depth

        def _add_paper(src: dict, hop: int) -> None:
            aid = src.get("arxiv_id", "")
            if not aid or aid in seen:
                return
            seen[aid] = hop
            nodes.append(self._make_paper_node(src, {"hop": hop}))

        # Seed papers = hop 0
        current_frontier: dict[str, dict] = {}
        for hit in seed_resp["hits"]["hits"]:
            src = hit["_source"]
            aid = src.get("arxiv_id", "")
            if not aid:
                continue
            _add_paper(src, 0)
            current_frontier[aid] = src

        # Walk hops
        for hop in range(1, max_hops + 1):
            next_ids: set[str] = set()
            for aid, src in current_frontier.items():
                for linked_id in set(src.get(field, []) or []):
                    if linked_id == aid:
                        continue
                    # Add edge for all links (including back-edges to already-seen nodes)
                    if direction == "references":
                        edges.append(GraphEdge(source=aid, target=linked_id, relation="cites"))
                    else:
                        edges.append(GraphEdge(source=linked_id, target=aid, relation="cites"))
                    # Only traverse forward to unseen nodes
                    if linked_id not in seen:
                        next_ids.add(linked_id)

            if not next_ids:
                break

            # Fetch next hop papers (capped)
            fetch_ids = list(next_ids)[:min(limit * 10, 10000)]
            hop_resp = await self._do_search({
                "query": {"terms": {"arxiv_id": fetch_ids}},
                "size": len(fetch_ids),
                "_source": _FIELDS + [field],
            })

            next_frontier: dict[str, dict] = {}
            for hit in hop_resp["hits"]["hits"]:
                src = hit["_source"]
                aid = src.get("arxiv_id", "")
                if not aid:
                    continue
                _add_paper(src, hop)
                next_frontier[aid] = src

            current_frontier = next_frontier
            if len(nodes) >= limit * 10:
                break

        # Filter out dangling edges (nodes not found in ES or beyond fetch cap)
        trimmed = nodes[:limit * 3]
        trimmed_ids = {n.id for n in trimmed}
        edges = [e for e in edges if e.source in trimmed_ids and e.target in trimmed_ids]

        return GraphResponse(
            nodes=trimmed, edges=edges,
            total=len(seen), took_ms=0,
            metadata={
                "direction": direction,
                "max_hops": max_hops,
                "hops_reached": max((seen[a] for a in seen), default=0),
                "papers_per_hop": dict(Counter(seen.values())),
            },
        )

    # ── 18. Shortest citation path ──

    async def _shortest_citation_path(
        self,
        gq: GraphQuery,
        sr: SearchRequest | None,
        emb: list[float] | None,
    ) -> GraphResponse:
        """BFS shortest path between two papers along citation links.
        Uses bidirectional BFS for efficiency — expands from both ends
        and meets in the middle."""
        source_id = gq.seed_arxiv_id
        target_id = gq.target_arxiv_id
        if not source_id or not target_id:
            return GraphResponse(nodes=[], edges=[], total=0, took_ms=0,
                                 metadata={"error": "both seed_arxiv_id and target_arxiv_id required"})
        if source_id == target_id:
            return GraphResponse(nodes=[], edges=[], total=0, took_ms=0,
                                 metadata={"error": "source and target are the same paper"})

        max_depth = min(gq.max_hops, 50)
        _FIELDS = ["arxiv_id", "title", "categories", "primary_category",
                    "authors", "submitted_date", "citation_stats",
                    "reference_ids", "cited_by_ids", "has_github"]

        # BFS state: id → (parent_id, direction_used)
        forward_parents: dict[str, str | None] = {source_id: None}
        backward_parents: dict[str, str | None] = {target_id: None}
        forward_frontier: set[str] = {source_id}
        backward_frontier: set[str] = {target_id}
        paper_cache: dict[str, dict] = {}
        meeting_point: str | None = None

        async def _fetch_papers(ids: list[str]) -> dict[str, dict]:
            if not ids:
                return {}
            resp = await self._do_search({
                "query": {"terms": {"arxiv_id": ids[:10000]}},
                "size": min(len(ids), 10000),
                "_source": _FIELDS,
            })
            result: dict[str, dict] = {}
            for hit in resp["hits"]["hits"]:
                s = hit["_source"]
                aid = s.get("arxiv_id", "")
                result[aid] = s
                paper_cache[aid] = s
            return result

        # Pre-fetch source and target
        initial = await _fetch_papers([source_id, target_id])
        if source_id not in initial or target_id not in initial:
            return GraphResponse(nodes=[], edges=[], total=0, took_ms=0,
                                 metadata={"error": "source or target paper not found in index"})

        for depth in range(max_depth):
            # Expand the smaller frontier (skip empty ones)
            if forward_frontier and (not backward_frontier or len(forward_frontier) <= len(backward_frontier)):
                papers = await _fetch_papers(list(forward_frontier))
                next_forward: set[str] = set()
                candidates: list[str] = []
                for fid in forward_frontier:
                    src = papers.get(fid, paper_cache.get(fid, {}))
                    # Follow both directions for connectivity
                    neighbors = set(src.get("reference_ids", []) or []) | set(src.get("cited_by_ids", []) or [])
                    for nid in neighbors:
                        if nid in backward_parents:
                            candidates.append(nid)
                            if nid not in forward_parents:
                                forward_parents[nid] = fid
                        elif nid not in forward_parents:
                            forward_parents[nid] = fid
                            next_forward.add(nid)
                if candidates:
                    # Pick meeting point with shortest backward chain
                    def _bwd_depth(mp: str) -> int:
                        d = 0
                        c: str | None = backward_parents.get(mp)
                        while c is not None:
                            d += 1
                            c = backward_parents.get(c)
                        return d
                    meeting_point = min(candidates, key=_bwd_depth)
                    break
                forward_frontier = next_forward
            else:
                papers = await _fetch_papers(list(backward_frontier))
                next_backward: set[str] = set()
                candidates_b: list[str] = []
                for bid in backward_frontier:
                    src = papers.get(bid, paper_cache.get(bid, {}))
                    neighbors = set(src.get("reference_ids", []) or []) | set(src.get("cited_by_ids", []) or [])
                    for nid in neighbors:
                        if nid in forward_parents:
                            candidates_b.append(nid)
                            if nid not in backward_parents:
                                backward_parents[nid] = bid
                        elif nid not in backward_parents:
                            backward_parents[nid] = bid
                            next_backward.add(nid)
                if candidates_b:
                    def _fwd_depth(mp: str) -> int:
                        d = 0
                        c: str | None = forward_parents.get(mp)
                        while c is not None:
                            d += 1
                            c = forward_parents.get(c)
                        return d
                    meeting_point = min(candidates_b, key=_fwd_depth)
                    break
                backward_frontier = next_backward

            if not forward_frontier and not backward_frontier:
                break

        if not meeting_point:
            return GraphResponse(nodes=[], edges=[], total=0, took_ms=0,
                                 metadata={
                                     "error": "no_path_found",
                                     "max_depth_searched": max_depth,
                                     "forward_explored": len(forward_parents),
                                     "backward_explored": len(backward_parents),
                                 })

        # Reconstruct path
        path: list[str] = []
        # Forward: source → meeting
        cur: str | None = meeting_point
        forward_segment: list[str] = []
        while cur is not None:
            forward_segment.append(cur)
            cur = forward_parents.get(cur)
        forward_segment.reverse()
        # Backward: meeting → target
        cur = backward_parents.get(meeting_point)
        backward_segment: list[str] = []
        while cur is not None:
            backward_segment.append(cur)
            cur = backward_parents.get(cur)
        path = forward_segment + backward_segment

        # Apply path filter if specified
        if gq.path_filter and path:
            filtered = self._filter_paths([path], paper_cache, gq.path_filter)
            if not filtered:
                return GraphResponse(nodes=[], edges=[], total=0, took_ms=0,
                                     metadata={
                                         "error": "path found but rejected by path_filter",
                                         "path_length": len(path) - 1,
                                         "path_filter_applied": True,
                                     })

        # Fetch all path papers
        missing = [pid for pid in path if pid not in paper_cache]
        if missing:
            await _fetch_papers(missing)

        nodes: list[GraphNode] = []
        edges: list[GraphEdge] = []
        for i, pid in enumerate(path):
            src = paper_cache.get(pid, {"arxiv_id": pid})
            nodes.append(self._make_paper_node(src, {"path_position": i}))
            if i > 0:
                edges.append(GraphEdge(
                    source=path[i - 1], target=pid, relation="path_link",
                ))

        return GraphResponse(
            nodes=nodes, edges=edges,
            total=len(path), took_ms=0,
            metadata={
                "source": source_id,
                "target": target_id,
                "path_length": len(path) - 1,
                "path_ids": path,
                "forward_explored": len(forward_parents),
                "backward_explored": len(backward_parents),
            },
        )

    # ── 19. PageRank approximation ──

    async def _pagerank(
        self,
        gq: GraphQuery,
        sr: SearchRequest | None,
        emb: list[float] | None,
    ) -> GraphResponse:
        """Iterative PageRank on the citation graph.  Seeds from search
        filters or paper IDs, fetches their citation neighborhood, and
        runs PageRank iterations to find the most influential papers."""
        limit = min(gq.limit or 50, self.MAX_RESULTS)
        damping = gq.damping_factor
        max_iter = gq.iterations

        paper_data, out_edges, in_edges, undirected = await self._build_citation_subgraph(
            gq, sr, emb, size_multiplier=4)

        if not out_edges:
            return GraphResponse(nodes=[], edges=[], total=0, took_ms=0,
                                 metadata={"error": "no_citation_links_in_subgraph",
                                            "papers_in_subgraph": len(paper_data)})

        # Run PageRank
        all_nodes = set(paper_data.keys())
        N = len(all_nodes)
        ranks: dict[str, float] = {a: 1.0 / N for a in all_nodes}

        converged_at = max_iter
        did_converge = False
        for it in range(max_iter):
            # Redistribute rank from dangling nodes (no outgoing edges)
            dangling_sum = sum(ranks[n] for n in all_nodes if not out_edges.get(n))
            new_ranks: dict[str, float] = {}
            for node in all_nodes:
                rank_sum = 0.0
                for in_node in in_edges.get(node, set()):
                    out_degree = len(out_edges.get(in_node, set()))
                    if out_degree > 0:
                        rank_sum += ranks[in_node] / out_degree
                new_ranks[node] = (1.0 - damping) / N + damping * (rank_sum + dangling_sum / N)
            max_delta = max(abs(new_ranks[n] - ranks[n]) for n in all_nodes)
            ranks = new_ranks
            if max_delta < 1e-6:
                converged_at = it + 1
                did_converge = True
                break

        # Return top-ranked papers
        sorted_ranks = sorted(ranks.items(), key=lambda x: -x[1])[:limit]

        nodes: list[GraphNode] = []
        edges_out: list[GraphEdge] = []

        for aid, rank in sorted_ranks:
            src = paper_data.get(aid, {"arxiv_id": aid})
            nodes.append(self._make_paper_node(src, {
                "pagerank": round(rank, 8),
                "in_degree": len(in_edges.get(aid, set())),
                "out_degree": len(out_edges.get(aid, set())),
            }))

        ranked_set = {aid for aid, _ in sorted_ranks}
        for aid, _ in sorted_ranks:
            for target in out_edges.get(aid, set()):
                if target in ranked_set:
                    edges_out.append(GraphEdge(
                        source=aid, target=target, relation="cites",
                        weight=ranks.get(target, 0),
                    ))

        return GraphResponse(
            nodes=nodes, edges=edges_out,
            total=N, took_ms=0,
            metadata={
                "damping_factor": damping,
                "iterations": converged_at,
                "converged": did_converge,
                "papers_in_subgraph": N,
                "edges_in_subgraph": sum(len(v) for v in out_edges.values()),
                "max_pagerank": round(sorted_ranks[0][1], 8) if sorted_ranks else 0,
            },
        )

    # ── 20. Community detection (Label Propagation) ──

    async def _community_detection(
        self,
        gq: GraphQuery,
        sr: SearchRequest | None,
        emb: list[float] | None,
    ) -> GraphResponse:
        """Label Propagation community detection on the citation/co-authorship
        graph.  Each paper starts with its own label; iteratively adopts the
        most frequent label among its neighbors.  Returns communities as
        cluster nodes connected to their member papers."""
        import random

        limit = min(gq.limit or 50, self.MAX_RESULTS)
        max_iter = min(gq.iterations, 500)

        paper_data, out_edges, in_edges, undirected = await self._build_citation_subgraph(
            gq, sr, emb, size_multiplier=4)

        if len(paper_data) < 2:
            return GraphResponse(nodes=[], edges=[], total=0, took_ms=0,
                                 metadata={"error": "need at least 2 papers with citation links"})

        # Merge undirected citation edges with co-authorship edges
        adj: dict[str, set[str]] = defaultdict(set)
        for aid, nbrs in undirected.items():
            adj[aid] |= nbrs

        # Also co-authorship edges (papers sharing authors)
        node_ids = set(paper_data.keys())
        author_papers: dict[str, list[str]] = defaultdict(list)
        for aid, src in paper_data.items():
            for a in (src.get("authors") or [])[:50]:
                name = a.get("name", "") if isinstance(a, dict) else (str(a) if a is not None else "")
                if name and aid not in author_papers[name]:
                    author_papers[name].append(aid)
        for name, papers in author_papers.items():
            if len(papers) > 1:
                for i in range(len(papers)):
                    for j in range(i + 1, min(len(papers), i + 10)):
                        adj[papers[i]].add(papers[j])
                        adj[papers[j]].add(papers[i])

        # Label Propagation
        labels: dict[str, str] = {aid: aid for aid in node_ids}
        node_list = list(node_ids)

        converged_at = max_iter
        for _iter in range(max_iter):
            random.shuffle(node_list)
            changed = False
            for nid in node_list:
                neighbors = adj.get(nid, set())
                if not neighbors:
                    continue
                label_counts: Counter[str] = Counter()
                for nbr in neighbors:
                    label_counts[labels[nbr]] += 1
                most_common = label_counts.most_common(1)[0][0]
                if labels[nid] != most_common:
                    labels[nid] = most_common
                    changed = True
            if not changed:
                converged_at = _iter + 1
                break

        # Step 3: Build community clusters
        communities: dict[str, list[str]] = defaultdict(list)
        for aid, label in labels.items():
            communities[label].append(aid)

        # Sort communities by size
        sorted_comms = sorted(communities.items(), key=lambda x: -len(x[1]))

        nodes: list[GraphNode] = []
        edges_out: list[GraphEdge] = []

        for comm_idx, (label, members) in enumerate(sorted_comms[:limit]):
            # Community node
            comm_id = f"community_{comm_idx}"
            # Dominant category in this community
            cat_counts: Counter[str] = Counter()
            for mid in members:
                for cat in paper_data.get(mid, {}).get("categories") or []:
                    cat_counts[cat] += 1
            top_cats = [c for c, _ in cat_counts.most_common(3)]

            nodes.append(GraphNode(
                id=comm_id,
                label=f"Community {comm_idx} ({len(members)} papers)",
                type="community",
                properties={
                    "size": len(members),
                    "top_categories": top_cats,
                    "member_ids": members[:200],
                },
            ))

            # Add member papers (capped per community)
            for mid in members[:100]:
                src = paper_data.get(mid, {"arxiv_id": mid})
                nodes.append(self._make_paper_node(src, {"community": comm_idx}))
                edges_out.append(GraphEdge(
                    source=comm_id, target=mid, relation="contains",
                ))

        # Add inter-community edges
        comm_of: dict[str, int] = {}
        for comm_idx, (_, members) in enumerate(sorted_comms[:limit]):
            for mid in members:
                comm_of[mid] = comm_idx
        inter_comm: Counter[tuple[int, int]] = Counter()
        for aid in node_ids:
            ca = comm_of.get(aid)
            if ca is None:
                continue
            for nbr in adj.get(aid, set()):
                cb = comm_of.get(nbr)
                if cb is not None and ca < cb:
                    inter_comm[(ca, cb)] += 1
        for (ca, cb), weight in inter_comm.most_common(limit):
            edges_out.append(GraphEdge(
                source=f"community_{ca}", target=f"community_{cb}",
                relation="inter_community", weight=weight,
            ))

        return GraphResponse(
            nodes=nodes, edges=edges_out,
            total=len(paper_data), took_ms=0,
            metadata={
                "papers_in_subgraph": len(paper_data),
                "edges_in_subgraph": sum(len(v) for v in adj.values()) // 2,
                "communities_found": len(sorted_comms),
                "largest_community": len(sorted_comms[0][1]) if sorted_comms else 0,
                "iterations": converged_at,
            },
        )

    # ── 21. Citation patterns ──

    async def _citation_patterns(
        self,
        gq: GraphQuery,
        sr: SearchRequest | None,
        emb: list[float] | None,
    ) -> GraphResponse:
        """Subgraph pattern matching on the citation graph.
        Finds structural patterns:
          - mutual:   A cites B AND B cites A
          - triangle: A→B→C→A (citation triangle)
          - star:     one paper cited by many in the result set (hub)
          - chain:    A→B→C→D (longest citation chain)
        """
        limit = min(gq.limit or 50, self.MAX_RESULTS)
        pattern = gq.pattern

        paper_data, out_edges, in_edges, undirected = await self._build_citation_subgraph(
            gq, sr, emb, size_multiplier=10)

        node_ids = set(paper_data.keys())
        # Use directed edges for pattern matching
        cites = out_edges

        nodes: list[GraphNode] = []
        edges_out: list[GraphEdge] = []
        seen_papers: set[str] = set()

        def _ensure_node(aid: str) -> None:
            if aid in seen_papers:
                return
            seen_papers.add(aid)
            src = paper_data.get(aid, {"arxiv_id": aid})
            nodes.append(self._make_paper_node(src))

        found_patterns = 0

        if pattern == "mutual":
            # Find A↔B: A cites B AND B cites A
            for a in node_ids:
                for b in cites.get(a, set()):
                    if a in cites.get(b, set()) and a < b:
                        _ensure_node(a)
                        _ensure_node(b)
                        edges_out.append(GraphEdge(source=a, target=b, relation="mutual_citation"))
                        edges_out.append(GraphEdge(source=b, target=a, relation="mutual_citation"))
                        found_patterns += 1
                        if found_patterns >= limit:
                            break
                if found_patterns >= limit:
                    break

        elif pattern == "triangle":
            # Find A→B→C→A
            seen_tris: set[str] = set()
            for a in node_ids:
                for b in cites.get(a, set()):
                    for c in cites.get(b, set()):
                        if a in cites.get(c, set()) and len({a, b, c}) == 3:
                            tri = tuple(sorted([a, b, c]))
                            tri_key = "|".join(tri)
                            if tri_key in seen_tris:
                                continue
                            seen_tris.add(tri_key)
                            _ensure_node(a)
                            _ensure_node(b)
                            _ensure_node(c)
                            edges_out.append(GraphEdge(source=a, target=b, relation="triangle_edge"))
                            edges_out.append(GraphEdge(source=b, target=c, relation="triangle_edge"))
                            edges_out.append(GraphEdge(source=c, target=a, relation="triangle_edge"))
                            found_patterns += 1
                            if found_patterns >= limit:
                                break
                    if found_patterns >= limit:
                        break
                if found_patterns >= limit:
                    break

        elif pattern == "star":
            # Find papers with highest in-degree within the subgraph (hubs)
            in_degree: Counter[str] = Counter()
            for a in node_ids:
                for b in cites.get(a, set()):
                    in_degree[b] += 1
            for hub_id, degree in in_degree.most_common(limit):
                if degree < 2:
                    break
                if hub_id not in seen_papers:
                    _ensure_node(hub_id)
                    nodes[-1].properties["pattern_role"] = "hub"
                    nodes[-1].properties["in_degree"] = degree
                else:
                    # Already added — find and update in-place
                    for _n in nodes:
                        if _n.id == hub_id:
                            _n.properties["pattern_role"] = "hub"
                            _n.properties["in_degree"] = degree
                            break
                for a in node_ids:
                    if hub_id in cites.get(a, set()):
                        _ensure_node(a)
                        edges_out.append(GraphEdge(source=a, target=hub_id,
                                                   relation="star_spoke"))
                found_patterns += 1

        elif pattern == "chain":
            # Find longest citation chains via iterative DFS (no memo —
            # result depends on visited set, so caching is unsound).
            _MAX_CHAIN_DEPTH = 200  # cap to avoid runaway exploration

            def _chain_len(node: str, visited: set[str]) -> int:
                # Iterative DFS with explicit stack
                # Stack entries: (node, neighbor_iterator, current_best)
                stack: list[tuple[str, list[str], int]] = []
                nbrs = [n for n in cites.get(node, set()) if n not in visited]
                stack.append((node, nbrs, 0))
                result = 0
                while stack:
                    if len(stack) > _MAX_CHAIN_DEPTH:
                        # Too deep — return what we have
                        result = max(result, len(stack) - 1)
                        popped_node, _, popped_best = stack.pop()
                        visited.discard(popped_node)
                        if stack:
                            parent = stack[-1]
                            stack[-1] = (parent[0], parent[1], max(parent[2], 1 + popped_best))
                        continue
                    cur, cur_nbrs, best = stack[-1]
                    if cur_nbrs:
                        nxt = cur_nbrs.pop()
                        visited.add(nxt)
                        nxt_nbrs = [n for n in cites.get(nxt, set()) if n not in visited]
                        stack.append((nxt, nxt_nbrs, 0))
                    else:
                        stack.pop()
                        visited.discard(cur)
                        if stack:
                            parent = stack[-1]
                            # Update parent's best with 1 + child's best
                            stack[-1] = (parent[0], parent[1], max(parent[2], 1 + best))
                        else:
                            result = max(result, best)
                return result

            chains: list[tuple[str, int]] = []
            for a in node_ids:
                cl = _chain_len(a, {a})
                if cl >= 2:
                    chains.append((a, cl))
            chains.sort(key=lambda x: -x[1])

            # Reconstruct best chains
            seen_edges: set[tuple[str, str]] = set()
            for start_id, chain_length in chains[:limit]:
                _ensure_node(start_id)
                cur = start_id
                visited_chain: set[str] = {cur}
                for _ in range(chain_length):
                    best_next = None
                    best_len = -1
                    for nxt in cites.get(cur, set()):
                        if nxt not in visited_chain:
                            cl = _chain_len(nxt, visited_chain | {nxt})
                            if cl > best_len:
                                best_len = cl
                                best_next = nxt
                    if best_next is None:
                        break
                    _ensure_node(best_next)
                    ek = (cur, best_next)
                    if ek not in seen_edges:
                        seen_edges.add(ek)
                        edges_out.append(GraphEdge(source=cur, target=best_next,
                                                   relation="chain_link"))
                    visited_chain.add(best_next)
                    cur = best_next
                found_patterns += 1

        return GraphResponse(
            nodes=nodes, edges=edges_out,
            total=found_patterns, took_ms=0,
            metadata={
                "pattern": pattern,
                "patterns_found": found_patterns,
                "papers_in_subgraph": len(paper_data),
                "edges_in_subgraph": sum(len(v) for v in cites.values()),
            },
        )

    # ── 22. Connected components ──

    async def _connected_components(
        self,
        gq: GraphQuery,
        sr: SearchRequest | None,
        emb: list[float] | None,
    ) -> GraphResponse:
        """Find connected components (paper clusters) in the citation graph.
        Uses iterative BFS.  Returns component summary nodes connected to
        their member papers."""
        limit = min(gq.limit or 50, self.MAX_RESULTS)

        base = self._base_query(sr, emb)
        combined = {
            "bool": {
                "must": [base],
                "filter": [
                    {"bool": {"should": [
                        {"exists": {"field": "reference_ids"}},
                        {"exists": {"field": "cited_by_ids"}},
                    ]}}
                ],
            }
        }

        resp = await self._do_search({
            "query": combined,
            "size": min(limit * 10, 10000),
            "_source": ["arxiv_id", "title", "categories", "primary_category",
                         "authors", "submitted_date", "citation_stats", "has_github",
                         "reference_ids", "cited_by_ids"],
        }, sr, emb)

        paper_data: dict[str, dict] = {}
        for hit in resp["hits"]["hits"]:
            s = hit["_source"]
            paper_data[s.get("arxiv_id", "")] = s

        node_ids = set(paper_data.keys())
        # Build undirected adjacency
        adj: dict[str, set[str]] = defaultdict(set)
        for aid, src in paper_data.items():
            for rid in (src.get("reference_ids", []) or []):
                if rid in node_ids and rid != aid:
                    adj[aid].add(rid)
                    adj[rid].add(aid)
            for cid in (src.get("cited_by_ids", []) or []):
                if cid in node_ids and cid != aid:
                    adj[aid].add(cid)
                    adj[cid].add(aid)

        # BFS to find connected components
        visited: set[str] = set()
        components: list[list[str]] = []

        for start in node_ids:
            if start in visited:
                continue
            component: list[str] = []
            queue = [start]
            visited.add(start)
            while queue:
                cur = queue.pop(0)
                component.append(cur)
                for nbr in adj.get(cur, set()):
                    if nbr not in visited:
                        visited.add(nbr)
                        queue.append(nbr)
            components.append(component)

        components.sort(key=lambda x: -len(x))

        nodes: list[GraphNode] = []
        edges_out: list[GraphEdge] = []

        for comp_idx, members in enumerate(components[:limit]):
            comp_id = f"component_{comp_idx}"
            cat_counts: Counter[str] = Counter()
            for mid in members:
                for cat in paper_data.get(mid, {}).get("categories") or []:
                    cat_counts[cat] += 1
            top_cats = [c for c, _ in cat_counts.most_common(3)]

            nodes.append(GraphNode(
                id=comp_id,
                label=f"Component {comp_idx} ({len(members)} papers)",
                type="component",
                properties={
                    "size": len(members),
                    "top_categories": top_cats,
                    "member_ids": members[:30],
                },
            ))

            for mid in members[:100]:
                src = paper_data.get(mid, {"arxiv_id": mid})
                nodes.append(self._make_paper_node(src, {"component": comp_idx}))
                edges_out.append(GraphEdge(
                    source=comp_id, target=mid, relation="contains",
                ))

        return GraphResponse(
            nodes=nodes, edges=edges_out,
            total=len(paper_data), took_ms=0,
            metadata={
                "papers_in_subgraph": len(paper_data),
                "edges_in_subgraph": sum(len(v) for v in adj.values()) // 2,
                "components_found": len(components),
                "largest_component": len(components[0]) if components else 0,
                "isolated_papers": sum(1 for c in components if len(c) == 1),
            },
        )

    # ── Helper: build a citation subgraph from search results ──

    async def _build_citation_subgraph(
        self,
        gq: GraphQuery,
        sr: SearchRequest | None,
        emb: list[float] | None,
        size_multiplier: int = 10,
    ) -> tuple[dict[str, dict], dict[str, set[str]], dict[str, set[str]], dict[str, set[str]]]:
        """Build a directed citation subgraph with neighbor expansion.

        After fetching the initial result set, collects all referenced/citing
        paper IDs and fetches those too, so there are actual edges between
        papers even when the initial set is sparsely connected.

        Returns:
            paper_data: {arxiv_id: source_dict}
            out_edges:  {arxiv_id: set of papers this paper cites}
            in_edges:   {arxiv_id: set of papers that cite this paper}
            undirected:  {arxiv_id: set of all neighbors (both directions)}
        """
        limit = min(gq.limit or 50, self.MAX_RESULTS)
        F = self.F

        if gq.seed_arxiv_id:
            seed_query: dict[str, Any] = {"term": {F.node_id: gq.seed_arxiv_id}}
        elif gq.seed_arxiv_ids:
            seed_query = {"terms": {F.node_id: gq.seed_arxiv_ids[:10000]}}
        else:
            base = self._base_query(sr, emb)
            seed_query = {
                "bool": {
                    "must": [base],
                    "filter": [
                        {"bool": {"should": [
                            {"exists": {"field": F.outgoing_edges}},
                            {"exists": {"field": F.incoming_edges}},
                        ]}}
                    ],
                }
            }

        _FIELDS = F.subgraph_fields

        fetch_size = min(limit * size_multiplier, 10000)
        if gq.seed_arxiv_ids:
            fetch_size = min(max(fetch_size, len(gq.seed_arxiv_ids)), 10000)

        resp = await self._do_search({
            "query": seed_query,
            "size": fetch_size,
            "_source": _FIELDS,
        }, sr, emb)

        paper_data: dict[str, dict] = {}
        for hit in resp["hits"]["hits"]:
            s = hit["_source"]
            paper_data[F.extract_id(s)] = s

        # ── Expansion pass: fetch neighbor papers ──
        # Respect projection direction when set by _subgraph_projection
        proj_dir = _ctx_projection_direction.get()
        expand_outgoing = proj_dir in (None, "references", "both")
        expand_incoming = proj_dir in (None, "cited_by", "both")

        neighbor_ids: set[str] = set()
        for src in paper_data.values():
            if expand_outgoing:
                for rid in F.extract_outgoing(src):
                    if rid not in paper_data:
                        neighbor_ids.add(rid)
            if expand_incoming:
                for cid in F.extract_incoming(src):
                    if cid not in paper_data:
                        neighbor_ids.add(cid)

        # Fetch neighbors in batches (parallel)
        neighbor_ids_list = list(neighbor_ids)[:10000]
        if neighbor_ids_list:
            batches = [neighbor_ids_list[i:i + 200] for i in range(0, len(neighbor_ids_list), 200)]
            results = await asyncio.gather(*(
                self.client.options(request_timeout=15).search(
                    index=self.index,
                    body={
                        "query": {"terms": {F.node_id: batch}},
                        "size": len(batch),
                        "_source": _FIELDS,
                        "timeout": "10s",
                    },
                ) for batch in batches
            ), return_exceptions=True)
            for nbr_resp in results:
                if isinstance(nbr_resp, BaseException):
                    continue
                for hit in nbr_resp.get("hits", {}).get("hits", []):
                    s = hit["_source"]
                    nid = F.extract_id(s)
                    if nid not in paper_data:
                        paper_data[nid] = s

        node_ids = set(paper_data.keys())
        out_edges: dict[str, set[str]] = defaultdict(set)
        in_edges: dict[str, set[str]] = defaultdict(set)
        undirected: dict[str, set[str]] = defaultdict(set)

        # Respect projection direction when set by _subgraph_projection
        proj_dir = _ctx_projection_direction.get()
        follow_outgoing = proj_dir in (None, "references", "both")
        follow_incoming = proj_dir in (None, "cited_by", "both")

        for aid, src in paper_data.items():
            if follow_outgoing:
                for rid in F.extract_outgoing(src):
                    if rid in node_ids and rid != aid:
                        out_edges[aid].add(rid)
                        in_edges[rid].add(aid)
                        undirected[aid].add(rid)
                        undirected[rid].add(aid)
            if follow_incoming:
                for cid in F.extract_incoming(src):
                    if cid in node_ids and cid != aid:
                        out_edges[cid].add(aid)
                        in_edges[aid].add(cid)
                        undirected[aid].add(cid)
                        undirected[cid].add(aid)

        return paper_data, out_edges, in_edges, undirected

    def _make_paper_node(self, src: dict, extra_props: dict | None = None) -> GraphNode:
        """Create a standard paper GraphNode from an ES source dict.

        Uses self.F (FieldMapping) so changing the mapping automatically
        adapts all node construction across all 52 handlers."""
        F = self.F
        props: dict[str, Any] = {
            "categories": src.get(F.node_categories) or [],
            "primary_category": src.get(F.node_primary_category),
            "citations": F.extract_citations(src),
            "submitted_date": src.get(F.node_timestamp),
            "authors": F.extract_authors(src),
            "has_github": src.get(F.has_code),
        }
        if extra_props:
            props.update(extra_props)
        return GraphNode(
            id=F.extract_id(src),
            label=F.extract_label(src),
            type="paper",
            properties=props,
        )

    # ── 23. Weighted shortest path (Dijkstra) ──

    async def _weighted_shortest_path(
        self,
        gq: GraphQuery,
        sr: SearchRequest | None,
        emb: list[float] | None,
    ) -> GraphResponse:
        """Dijkstra shortest path with configurable edge weights.

        weight_field='citations': cost = 1 / (1 + total_citations) — prefer paths
        through highly-cited papers (lower cost).
        weight_field='uniform': all edges cost 1 (equivalent to BFS).
        """
        import heapq

        source_id = gq.seed_arxiv_id
        target_id = gq.target_arxiv_id
        if not source_id or not target_id:
            return GraphResponse(nodes=[], edges=[], total=0, took_ms=0,
                                 metadata={"error": "both seed_arxiv_id and target_arxiv_id required"})
        if source_id == target_id:
            return GraphResponse(nodes=[], edges=[], total=0, took_ms=0,
                                 metadata={"error": "source and target are the same paper"})

        weight_field = gq.weight_field
        max_depth = min(gq.max_hops, 50)

        _FIELDS = ["arxiv_id", "title", "categories", "primary_category",
                    "authors", "submitted_date", "citation_stats",
                    "reference_ids", "cited_by_ids", "has_github"]

        paper_cache: dict[str, dict] = {}

        async def _fetch(ids: list[str]) -> dict[str, dict]:
            if not ids:
                return {}
            # Filter out already cached
            to_fetch = [i for i in ids if i not in paper_cache]
            if to_fetch:
                resp = await self._do_search({
                    "query": {"terms": {"arxiv_id": to_fetch[:10000]}},
                    "size": min(len(to_fetch), 10000),
                    "_source": _FIELDS,
                })
                for hit in resp["hits"]["hits"]:
                    s = hit["_source"]
                    paper_cache[s.get("arxiv_id", "")] = s
            return {i: paper_cache[i] for i in ids if i in paper_cache}

        # Fetch source and target
        initial = await _fetch([source_id, target_id])
        if source_id not in initial or target_id not in initial:
            return GraphResponse(nodes=[], edges=[], total=0, took_ms=0,
                                 metadata={"error": "source or target paper not found"})

        def _edge_cost(paper: dict) -> float:
            if weight_field == "uniform":
                return 1.0
            cit = (paper.get("citation_stats") or {}).get("total_citations") or 0
            return 1.0 / (1.0 + cit)

        # Dijkstra
        # dist[node] = (cost, parent)
        dist: dict[str, float] = {source_id: 0.0}
        parent: dict[str, str | None] = {source_id: None}
        visited: set[str] = set()
        heap: list[tuple[float, str]] = [(0.0, source_id)]
        hops_used: dict[str, int] = {source_id: 0}

        found = False
        while heap:
            cost, current = heapq.heappop(heap)
            if current in visited:
                continue
            visited.add(current)

            if current == target_id:
                found = True
                break

            cur_hops = hops_used.get(current, 0)
            if cur_hops >= max_depth:
                continue

            # Expand neighbors
            cur_paper = paper_cache.get(current, {})
            neighbors: set[str] = set()
            for rid in (cur_paper.get("reference_ids", []) or []):
                neighbors.add(rid)
            for cid in (cur_paper.get("cited_by_ids", []) or []):
                neighbors.add(cid)

            if neighbors:
                to_fetch = list(neighbors)[:10000]
                await _fetch(to_fetch)
                neighbors = set(to_fetch)  # only explore fetched neighbors

            for nbr in neighbors:
                if nbr in visited:
                    continue
                nbr_paper = paper_cache.get(nbr, {})
                if not nbr_paper:
                    continue
                new_cost = cost + _edge_cost(nbr_paper)
                if nbr not in dist or new_cost < dist[nbr]:
                    dist[nbr] = new_cost
                    parent[nbr] = current
                    hops_used[nbr] = cur_hops + 1
                    heapq.heappush(heap, (new_cost, nbr))

        if not found:
            return GraphResponse(nodes=[], edges=[], total=0, took_ms=0,
                                 metadata={
                                     "error": "no_path_found",
                                     "weight_field": weight_field,
                                     "explored": len(visited),
                                 })

        # Reconstruct path
        path: list[str] = []
        cur: str | None = target_id
        while cur is not None:
            path.append(cur)
            cur = parent.get(cur)
        path.reverse()

        # Fetch all path papers
        await _fetch(path)

        nodes: list[GraphNode] = []
        edges_out: list[GraphEdge] = []
        total_cost = 0.0

        for i, pid in enumerate(path):
            src = paper_cache.get(pid, {"arxiv_id": pid})
            node_cost = _edge_cost(src) if i > 0 else 0.0
            total_cost += node_cost
            nodes.append(self._make_paper_node(src, {
                "path_position": i,
                "cumulative_cost": round(total_cost, 6),
                "edge_cost": round(node_cost, 6),
            }))
            if i > 0:
                edges_out.append(GraphEdge(
                    source=path[i - 1], target=pid,
                    relation="weighted_path", weight=node_cost,
                ))

        return GraphResponse(
            nodes=nodes, edges=edges_out,
            total=len(path), took_ms=0,
            metadata={
                "source": source_id,
                "target": target_id,
                "path_length": len(path) - 1,
                "total_cost": round(total_cost, 6),
                "weight_field": weight_field,
                "explored": len(visited),
                "path_ids": path,
            },
        )

    # ── 24. Betweenness centrality ──

    async def _betweenness_centrality(
        self,
        gq: GraphQuery,
        sr: SearchRequest | None,
        emb: list[float] | None,
    ) -> GraphResponse:
        """Approximate betweenness centrality via Brandes' algorithm on
        the citation subgraph. Identifies papers that act as critical
        bridges — removing them would disconnect many shortest paths."""
        limit = min(gq.limit or 50, self.MAX_RESULTS)

        paper_data, out_edges, in_edges, undirected = await self._build_citation_subgraph(
            gq, sr, emb, size_multiplier=8)

        if len(paper_data) < 3:
            return GraphResponse(nodes=[], edges=[], total=0, took_ms=0,
                                 metadata={"error": "need at least 3 connected papers"})

        node_list = list(paper_data.keys())
        N = len(node_list)
        betweenness: dict[str, float] = {n: 0.0 for n in node_list}

        # Brandes' algorithm (undirected variant for efficiency)
        # Sample at most 100 source nodes for approximation on large graphs
        import random
        sample = node_list if N <= 500 else random.sample(node_list, min(N, 500))

        for s in sample:
            # BFS
            S: list[str] = []
            P: dict[str, list[str]] = {n: [] for n in node_list}
            sigma: dict[str, int] = {n: 0 for n in node_list}
            sigma[s] = 1
            d: dict[str, int] = {n: -1 for n in node_list}
            d[s] = 0
            Q: list[str] = [s]
            qi = 0
            while qi < len(Q):
                v = Q[qi]
                qi += 1
                S.append(v)
                for w in undirected.get(v, set()):
                    if d[w] < 0:
                        Q.append(w)
                        d[w] = d[v] + 1
                    if d[w] == d[v] + 1:
                        sigma[w] += sigma[v]
                        P[w].append(v)

            delta: dict[str, float] = {n: 0.0 for n in node_list}
            while S:
                w = S.pop()
                for v in P[w]:
                    if sigma[w] > 0:
                        delta[v] += (sigma[v] / sigma[w]) * (1 + delta[w])
                if w != s:
                    betweenness[w] += delta[w]

        # Normalize (undirected: divide by 2 to correct double-counting)
        if N > 2:
            scale = 1.0 / ((N - 1) * (N - 2))
            if len(sample) < N:
                scale *= N / len(sample)  # Approximate scaling
            for n in node_list:
                betweenness[n] = betweenness[n] * scale / 2.0

        # Sort and return top nodes
        sorted_bc = sorted(betweenness.items(), key=lambda x: -x[1])[:limit]

        nodes: list[GraphNode] = []
        edges_out: list[GraphEdge] = []
        ranked_set = {aid for aid, _ in sorted_bc}

        for rank, (aid, bc) in enumerate(sorted_bc):
            src = paper_data.get(aid, {"arxiv_id": aid})
            nodes.append(self._make_paper_node(src, {
                "betweenness": round(bc, 8),
                "rank": rank,
                "degree": len(undirected.get(aid, set())),
            }))

        for aid, _ in sorted_bc:
            for target in out_edges.get(aid, set()):
                if target in ranked_set:
                    edges_out.append(GraphEdge(
                        source=aid, target=target, relation="cites",
                        weight=betweenness.get(target, 0),
                    ))

        return GraphResponse(
            nodes=nodes, edges=edges_out,
            total=N, took_ms=0,
            metadata={
                "papers_in_subgraph": N,
                "edges_in_subgraph": sum(len(v) for v in out_edges.values()),
                "max_betweenness": round(sorted_bc[0][1], 8) if sorted_bc else 0,
                "sampled_sources": len(sample),
            },
        )

    # ── 25. Closeness centrality ──

    async def _closeness_centrality(
        self,
        gq: GraphQuery,
        sr: SearchRequest | None,
        emb: list[float] | None,
    ) -> GraphResponse:
        """Closeness centrality — how close a paper is to all others.

        closeness(v) = (n-1) / sum_of_shortest_path_distances_from_v

        Papers with high closeness can quickly reach or be reached by most
        other papers in the citation network."""
        limit = min(gq.limit or 50, self.MAX_RESULTS)

        paper_data, out_edges, in_edges, undirected = await self._build_citation_subgraph(
            gq, sr, emb, size_multiplier=8)

        if len(paper_data) < 3:
            return GraphResponse(nodes=[], edges=[], total=0, took_ms=0,
                                 metadata={"error": "need at least 3 connected papers"})

        node_list = list(paper_data.keys())
        N = len(node_list)
        closeness: dict[str, float] = {}

        for s in node_list:
            # BFS from s
            dist: dict[str, int] = {s: 0}
            queue = [s]
            qi = 0
            while qi < len(queue):
                v = queue[qi]
                qi += 1
                for w in undirected.get(v, set()):
                    if w not in dist:
                        dist[w] = dist[v] + 1
                        queue.append(w)

            reachable = len(dist) - 1  # exclude self
            if reachable > 0:
                total_dist = sum(dist.values())
                # Wasserman-Faust normalization for disconnected graphs
                closeness[s] = (reachable / (N - 1)) * (reachable / total_dist) if total_dist > 0 else 0.0
            else:
                closeness[s] = 0.0

        sorted_cc = sorted(closeness.items(), key=lambda x: -x[1])[:limit]
        ranked_set = {aid for aid, _ in sorted_cc}

        nodes: list[GraphNode] = []
        edges_out: list[GraphEdge] = []

        for rank, (aid, cc) in enumerate(sorted_cc):
            src = paper_data.get(aid, {"arxiv_id": aid})
            nodes.append(self._make_paper_node(src, {
                "closeness": round(cc, 8),
                "rank": rank,
                "degree": len(undirected.get(aid, set())),
            }))

        for aid, _ in sorted_cc:
            for target in out_edges.get(aid, set()):
                if target in ranked_set:
                    edges_out.append(GraphEdge(
                        source=aid, target=target, relation="cites",
                        weight=closeness.get(target, 0),
                    ))

        return GraphResponse(
            nodes=nodes, edges=edges_out,
            total=N, took_ms=0,
            metadata={
                "papers_in_subgraph": N,
                "edges_in_subgraph": sum(len(v) for v in out_edges.values()),
                "max_closeness": round(sorted_cc[0][1], 8) if sorted_cc else 0,
                "min_closeness": round(sorted_cc[-1][1], 8) if sorted_cc else 0,
            },
        )

    # ── 26. Strongly connected components (Tarjan) ──

    async def _strongly_connected_components(
        self,
        gq: GraphQuery,
        sr: SearchRequest | None,
        emb: list[float] | None,
    ) -> GraphResponse:
        """Tarjan's algorithm for strongly connected components on the
        DIRECTED citation graph. An SCC is a maximal set of papers where
        every paper can reach every other through citation links.

        Reveals citation clusters with mutual influence."""
        limit = min(gq.limit or 50, self.MAX_RESULTS)

        paper_data, out_edges, in_edges, _ = await self._build_citation_subgraph(
            gq, sr, emb, size_multiplier=8)

        if len(paper_data) < 2:
            return GraphResponse(nodes=[], edges=[], total=0, took_ms=0,
                                 metadata={"error": "need at least 2 papers with citation links"})

        node_list = list(paper_data.keys())

        # Iterative Tarjan (avoids stack overflow on large graphs)
        index_counter = [0]
        indices: dict[str, int] = {}
        lowlinks: dict[str, int] = {}
        on_stack: dict[str, bool] = {}
        stack: list[str] = []
        sccs: list[list[str]] = []

        # Iterative version using explicit call stack
        for start in node_list:
            if start in indices:
                continue
            work_stack: list[tuple[str, int]] = [(start, 0)]
            # (node, iterator_position over out_edges)
            out_lists: dict[str, list[str]] = {}

            while work_stack:
                v, pos = work_stack[-1]

                if pos == 0:
                    # First visit
                    indices[v] = lowlinks[v] = index_counter[0]
                    index_counter[0] += 1
                    stack.append(v)
                    on_stack[v] = True
                    out_lists[v] = list(out_edges.get(v, set()))

                if pos < len(out_lists.get(v, [])):
                    w = out_lists[v][pos]
                    work_stack[-1] = (v, pos + 1)
                    if w not in indices:
                        work_stack.append((w, 0))
                    elif on_stack.get(w, False):
                        lowlinks[v] = min(lowlinks[v], indices[w])
                else:
                    # Done with all successors
                    if lowlinks[v] == indices[v]:
                        scc: list[str] = []
                        while True:
                            w = stack.pop()
                            on_stack[w] = False
                            scc.append(w)
                            if w == v:
                                break
                        sccs.append(scc)

                    work_stack.pop()
                    if work_stack:
                        parent = work_stack[-1][0]
                        lowlinks[parent] = min(lowlinks[parent], lowlinks[v])

        # Sort SCCs by size (non-trivial first)
        sccs.sort(key=lambda x: -len(x))

        nodes: list[GraphNode] = []
        edges_out: list[GraphEdge] = []

        scc_of: dict[str, int] = {}
        for idx, scc in enumerate(sccs[:limit]):
            for mid in scc:
                scc_of[mid] = idx

            scc_id = f"scc_{idx}"
            cat_counts: Counter[str] = Counter()
            for mid in scc:
                for cat in paper_data.get(mid, {}).get("categories") or []:
                    cat_counts[cat] += 1

            nodes.append(GraphNode(
                id=scc_id,
                label=f"SCC {idx} ({len(scc)} papers)",
                type="scc",
                properties={
                    "size": len(scc),
                    "is_trivial": len(scc) == 1,
                    "top_categories": [c for c, _ in cat_counts.most_common(3)],
                    "member_ids": scc[:200],
                },
            ))

            for mid in scc[:100]:
                src = paper_data.get(mid, {"arxiv_id": mid})
                nodes.append(self._make_paper_node(src, {"scc": idx}))
                edges_out.append(GraphEdge(
                    source=scc_id, target=mid, relation="contains",
                ))

        # Inter-SCC edges
        inter: Counter[tuple[int, int]] = Counter()
        for aid in node_list:
            sa = scc_of.get(aid)
            if sa is None:
                continue
            for nbr in out_edges.get(aid, set()):
                sb = scc_of.get(nbr)
                if sb is not None and sa != sb:
                    inter[(sa, sb)] += 1

        for (sa, sb), w in inter.most_common(limit):
            edges_out.append(GraphEdge(
                source=f"scc_{sa}", target=f"scc_{sb}",
                relation="scc_edge", weight=w,
            ))

        nontrivial = sum(1 for scc in sccs if len(scc) > 1)
        return GraphResponse(
            nodes=nodes, edges=edges_out,
            total=len(paper_data), took_ms=0,
            metadata={
                "papers_in_subgraph": len(paper_data),
                "directed_edges": sum(len(v) for v in out_edges.values()),
                "sccs_found": len(sccs),
                "nontrivial_sccs": nontrivial,
                "largest_scc": len(sccs[0]) if sccs else 0,
            },
        )

    # ── 27. Topological sort ──

    async def _topological_sort(
        self,
        gq: GraphQuery,
        sr: SearchRequest | None,
        emb: list[float] | None,
    ) -> GraphResponse:
        """Topological ordering of the citation DAG.

        Orders papers so that if paper A cites paper B, B appears before A.
        This represents the intellectual lineage: foundational papers first,
        derivative works later.

        Cycles (mutual citations) are broken by submitted_date tiebreak.
        Returns papers with their topological depth."""
        limit = min(gq.limit or 50, self.MAX_RESULTS)

        paper_data, out_edges, in_edges, _ = await self._build_citation_subgraph(
            gq, sr, emb, size_multiplier=8)

        if not paper_data:
            return GraphResponse(nodes=[], edges=[], total=0, took_ms=0,
                                 metadata={"error": "no papers found"})

        node_list = list(paper_data.keys())
        N = len(node_list)

        # Kahn's algorithm — iterative topological sort
        # in_degree = number of papers that cite this paper AND are in the subgraph
        # (i.e., how many arrows point TO this paper from within)
        # direction: A cites B means A → B in out_edges.
        # For topological sort, B (cited paper) should come first.
        # So we reverse: in_degree[v] = number of papers v cites that are in our set.
        in_deg: dict[str, int] = {n: 0 for n in node_list}
        for n in node_list:
            in_deg[n] = len(out_edges.get(n, set()))

        queue: list[str] = [n for n in node_list if in_deg[n] == 0]
        # Sort initial queue by date (oldest first) for deterministic ordering
        queue.sort(key=lambda x: paper_data.get(x, {}).get("submitted_date", "") or "")

        topo_order: list[str] = []
        depth: dict[str, int] = {}

        while queue:
            v = queue.pop(0)
            topo_order.append(v)
            if v not in depth:
                depth[v] = 0
            # For each paper that cites v (v is in their references)
            for u in in_edges.get(v, set()):
                in_deg[u] -= 1
                if in_deg[u] == 0:
                    queue.append(u)
                depth_candidate = depth[v] + 1
                depth[u] = max(depth.get(u, 0), depth_candidate)
            queue.sort(key=lambda x: paper_data.get(x, {}).get("submitted_date", "") or "")

        # Handle cycles — add remaining nodes sorted by date
        topo_set = set(topo_order)
        remaining = [n for n in node_list if n not in topo_set]
        remaining.sort(key=lambda x: paper_data.get(x, {}).get("submitted_date", "") or "")
        base_depth = (max(depth.values()) + 1) if depth else 0
        for n in remaining:
            topo_order.append(n)
            depth[n] = base_depth

        nodes: list[GraphNode] = []
        edges_out: list[GraphEdge] = []
        result_ids = set(topo_order[:limit])

        for position, aid in enumerate(topo_order[:limit]):
            src = paper_data.get(aid, {"arxiv_id": aid})
            nodes.append(self._make_paper_node(src, {
                "topo_position": position,
                "topo_depth": depth.get(aid, 0),
            }))

        # Add citation edges between returned nodes
        for aid in topo_order[:limit]:
            for target in out_edges.get(aid, set()):
                if target in result_ids:
                    edges_out.append(GraphEdge(
                        source=aid, target=target, relation="cites",
                    ))

        max_depth = max(depth.values()) if depth else 0
        return GraphResponse(
            nodes=nodes, edges=edges_out,
            total=N, took_ms=0,
            metadata={
                "papers_in_subgraph": N,
                "papers_in_order": len(topo_order),
                "cycles_broken": len(remaining),
                "max_depth": max_depth,
                "depth_distribution": dict(Counter(depth[n] for n in topo_order[:limit])),
            },
        )

    # ── 28. Link prediction ──

    async def _link_prediction(
        self,
        gq: GraphQuery,
        sr: SearchRequest | None,
        emb: list[float] | None,
    ) -> GraphResponse:
        """Predict likely future citations using graph-based heuristics.

        Methods:
        - common_neighbors:          |N(u) ∩ N(v)|
        - jaccard:                   |N(u) ∩ N(v)| / |N(u) ∪ N(v)|
        - adamic_adar:               Σ_{w ∈ N(u)∩N(v)} 1/log(|N(w)|)
        - preferential_attachment:   |N(u)| × |N(v)|

        Returns the top predicted links as edges ranked by score."""
        import math

        limit = min(gq.limit or 50, self.MAX_RESULTS)
        method = gq.prediction_method

        paper_data, out_edges, in_edges, undirected = await self._build_citation_subgraph(
            gq, sr, emb, size_multiplier=6)

        if len(paper_data) < 3:
            return GraphResponse(nodes=[], edges=[], total=0, took_ms=0,
                                 metadata={"error": "need at least 3 papers"})

        node_list = list(paper_data.keys())
        # Only predict links between pairs that are NOT already connected
        existing: set[tuple[str, str]] = set()
        for a in node_list:
            for b in undirected.get(a, set()):
                existing.add((a, b))

        predictions: list[tuple[str, str, float]] = []

        # Cap active nodes for O(n²) methods to prevent DoS
        active_nodes = sorted(node_list, key=lambda x: -len(undirected.get(x, set())))
        active_nodes = active_nodes[:min(len(active_nodes), 2000)]

        if method == "common_neighbors":
            for i, a in enumerate(active_nodes):
                na = undirected.get(a, set())
                if not na:
                    continue
                for b in active_nodes[i + 1:]:
                    if (a, b) in existing:
                        continue
                    nb = undirected.get(b, set())
                    score = len(na & nb)
                    if score > 0:
                        predictions.append((a, b, float(score)))

        elif method == "jaccard":
            for i, a in enumerate(active_nodes):
                na = undirected.get(a, set())
                if not na:
                    continue
                for b in active_nodes[i + 1:]:
                    if (a, b) in existing:
                        continue
                    nb = undirected.get(b, set())
                    union = na | nb
                    if union:
                        score = len(na & nb) / len(union)
                        if score > 0:
                            predictions.append((a, b, score))

        elif method == "adamic_adar":
            for i, a in enumerate(active_nodes):
                na = undirected.get(a, set())
                if not na:
                    continue
                for b in active_nodes[i + 1:]:
                    if (a, b) in existing:
                        continue
                    nb = undirected.get(b, set())
                    common = na & nb
                    if common:
                        score = sum(
                            1.0 / math.log(len(undirected.get(w, set())))
                            for w in common
                            if len(undirected.get(w, set())) > 1
                        )
                        if score > 0:
                            predictions.append((a, b, score))

        elif method == "preferential_attachment":
            # Only compute for nodes that have some connections
            connected = [(n, len(undirected.get(n, set()))) for n in node_list
                         if undirected.get(n)]
            connected.sort(key=lambda x: -x[1])
            # Limit computation — take top connected nodes
            top_nodes = connected[:min(200, len(connected))]
            for i, (a, da) in enumerate(top_nodes):
                for b, db in top_nodes[i + 1:]:
                    if (a, b) not in existing:
                        predictions.append((a, b, float(da * db)))

        predictions.sort(key=lambda x: -x[2])
        top_predictions = predictions[:limit]

        nodes: list[GraphNode] = []
        edges_out: list[GraphEdge] = []
        seen_nodes: set[str] = set()

        for a, b, score in top_predictions:
            for pid in (a, b):
                if pid not in seen_nodes:
                    seen_nodes.add(pid)
                    src = paper_data.get(pid, {"arxiv_id": pid})
                    nodes.append(self._make_paper_node(src, {
                        "degree": len(undirected.get(pid, set())),
                    }))
            edges_out.append(GraphEdge(
                source=a, target=b,
                relation="predicted",
                weight=round(score, 6),
            ))

        return GraphResponse(
            nodes=nodes, edges=edges_out,
            total=len(predictions), took_ms=0,
            metadata={
                "method": method,
                "papers_in_subgraph": len(paper_data),
                "existing_edges": len(existing) // 2,
                "predictions_total": len(predictions),
                "predictions_returned": len(top_predictions),
                "max_score": round(top_predictions[0][2], 6) if top_predictions else 0,
            },
        )

    # ── 29. Louvain community detection ──

    async def _louvain_community(
        self,
        gq: GraphQuery,
        sr: SearchRequest | None,
        emb: list[float] | None,
    ) -> GraphResponse:
        """Louvain modularity-optimization community detection.

        Phase 1: Each node starts as its own community. Greedily move nodes
                 to the neighboring community that maximizes modularity gain.
        Phase 2: Aggregate communities into super-nodes and repeat.
        Returns communities with their member papers and inter-community edges.
        """
        import random

        limit = min(gq.limit or 50, self.MAX_RESULTS)
        max_iter = min(gq.iterations, 500)

        paper_data, out_edges, in_edges, undirected = await self._build_citation_subgraph(
            gq, sr, emb, size_multiplier=8)

        if len(paper_data) < 2:
            return GraphResponse(nodes=[], edges=[], total=0, took_ms=0,
                                 metadata={"error": "need at least 2 papers"})

        # Build weighted undirected adjacency (also co-authorship)
        # undirected already has both directions, so adj[aid][nbr] is sufficient
        adj: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        for aid in paper_data:
            for nbr in undirected.get(aid, set()):
                adj[aid][nbr] += 1.0

        # Co-authorship edges
        author_papers: dict[str, list[str]] = defaultdict(list)
        for aid, src in paper_data.items():
            for a in (src.get("authors") or [])[:50]:
                name = a.get("name", "") if isinstance(a, dict) else (str(a) if a is not None else "")
                if name and aid not in author_papers[name]:
                    author_papers[name].append(aid)
        for name, papers in author_papers.items():
            if len(papers) > 1:
                for i in range(len(papers)):
                    for j in range(i + 1, min(len(papers), i + 10)):
                        adj[papers[i]][papers[j]] += 0.5
                        adj[papers[j]][papers[i]] += 0.5

        # Total edge weight (m)
        m = sum(sum(nbrs.values()) for nbrs in adj.values()) / 2.0
        if m == 0:
            return GraphResponse(nodes=[], edges=[], total=0, took_ms=0,
                                 metadata={"error": "no edges in subgraph"})

        node_list = list(paper_data.keys())
        # Node strength (sum of edge weights)
        strength: dict[str, float] = {n: sum(adj[n].values()) for n in node_list}
        # Community assignment
        comm: dict[str, int] = {n: i for i, n in enumerate(node_list)}
        # Community total strength
        comm_strength: dict[int, float] = {i: strength[n] for i, n in enumerate(node_list)}
        # Community internal weight — initialize properly
        comm_internal: dict[int, float] = defaultdict(float)
        for n in node_list:
            for nbr, w in adj[n].items():
                if comm[n] == comm[nbr]:
                    comm_internal[comm[n]] += w / 2.0  # each edge counted from both sides

        for _ in range(max_iter):
            moved = False
            random.shuffle(node_list)
            for node in node_list:
                old_comm = comm[node]
                ki = strength[node]

                # Compute weights from node to each neighboring community
                nbr_comms: dict[int, float] = defaultdict(float)
                for nbr, w in adj[node].items():
                    nbr_comms[comm[nbr]] += w

                ki_old = nbr_comms.get(old_comm, 0.0)

                # Remove node from its community for gain calculation
                sigma_old = comm_strength[old_comm] - ki

                best_comm = old_comm
                best_gain = 0.0
                for c, ki_in in nbr_comms.items():
                    if c == old_comm:
                        continue
                    sigma_tot = comm_strength.get(c, 0.0)
                    # Standard Louvain gain: gain_into_c - loss_from_old
                    gain = (ki_in - ki_old) / m - ki * (sigma_tot - sigma_old) / (2.0 * m * m)
                    if gain > best_gain:
                        best_gain = gain
                        best_comm = c

                if best_comm != old_comm:
                    # Update community bookkeeping
                    comm_strength[old_comm] -= ki
                    comm_internal[old_comm] -= ki_old
                    comm[node] = best_comm
                    comm_strength[best_comm] = comm_strength.get(best_comm, 0.0) + ki
                    comm_internal[best_comm] += nbr_comms.get(best_comm, 0.0)
                    moved = True

            if not moved:
                break

        # Compute modularity
        modularity = 0.0
        for c in set(comm.values()):
            lc = comm_internal.get(c, 0.0)
            sc = comm_strength.get(c, 0.0)
            modularity += lc / m - (sc / (2.0 * m)) ** 2

        # Build communities
        communities: dict[int, list[str]] = defaultdict(list)
        for n, c in comm.items():
            communities[c].append(n)
        sorted_comms = sorted(communities.items(), key=lambda x: -len(x[1]))

        nodes: list[GraphNode] = []
        edges_out: list[GraphEdge] = []

        for idx, (cid, members) in enumerate(sorted_comms[:limit]):
            cat_counts: Counter[str] = Counter()
            for mid in members:
                for cat in paper_data.get(mid, {}).get("categories") or []:
                    cat_counts[cat] += 1
            top_cats = [c for c, _ in cat_counts.most_common(3)]

            comm_node_id = f"louvain_{idx}"
            nodes.append(GraphNode(
                id=comm_node_id,
                label=f"Community {idx} ({len(members)} papers)",
                type="community",
                properties={
                    "size": len(members),
                    "top_categories": top_cats,
                    "member_ids": members[:200],
                },
            ))

            for mid in members[:100]:
                src = paper_data.get(mid, {"arxiv_id": mid})
                nodes.append(self._make_paper_node(src, {"community": idx}))
                edges_out.append(GraphEdge(
                    source=comm_node_id, target=mid, relation="contains",
                ))

        # Inter-community edges
        comm_of = {n: comm[n] for n in node_list}
        comm_idx_map = {cid: idx for idx, (cid, _) in enumerate(sorted_comms[:limit])}
        inter_comm: Counter[tuple[int, int]] = Counter()
        for aid in node_list:
            ca = comm_idx_map.get(comm_of.get(aid, -1))
            if ca is None:
                continue
            for nbr in undirected.get(aid, set()):
                cb = comm_idx_map.get(comm_of.get(nbr, -1))
                if cb is not None and ca < cb:
                    inter_comm[(ca, cb)] += 1
        for (ca, cb), w in inter_comm.most_common(limit):
            edges_out.append(GraphEdge(
                source=f"louvain_{ca}", target=f"louvain_{cb}",
                relation="inter_community", weight=w,
            ))

        return GraphResponse(
            nodes=nodes, edges=edges_out,
            total=len(paper_data), took_ms=0,
            metadata={
                "papers_in_subgraph": len(paper_data),
                "communities_found": len(communities),
                "modularity": round(modularity, 6),
                "largest_community": len(sorted_comms[0][1]) if sorted_comms else 0,
                "algorithm": "louvain",
            },
        )

    # ── 30. Degree centrality ──

    async def _degree_centrality(
        self,
        gq: GraphQuery,
        sr: SearchRequest | None,
        emb: list[float] | None,
    ) -> GraphResponse:
        """Degree centrality: rank papers by number of connections.

        Modes:
        - in:    number of papers that cite this paper (in-degree)
        - out:   number of papers this paper cites (out-degree)
        - total: sum of in + out (default)

        Returns top-N papers by degree, with citation edges between them."""
        limit = min(gq.limit or 50, self.MAX_RESULTS)
        mode = gq.degree_mode  # "in", "out", "total"

        paper_data, out_edges, in_edges, undirected = await self._build_citation_subgraph(
            gq, sr, emb, size_multiplier=8)

        if not paper_data:
            return GraphResponse(nodes=[], edges=[], total=0, took_ms=0,
                                 metadata={"error": "no papers found"})

        # Compute degree for each paper
        degree: dict[str, int] = {}
        for aid in paper_data:
            if mode == "in":
                degree[aid] = len(in_edges.get(aid, set()))
            elif mode == "out":
                degree[aid] = len(out_edges.get(aid, set()))
            else:  # total
                degree[aid] = len(in_edges.get(aid, set())) + len(out_edges.get(aid, set()))

        N = len(paper_data)
        # Normalized centrality — max total degree in directed graph is 2*(N-1)
        norm = max(2 * (N - 1), 1) if mode == "total" else max(N - 1, 1)

        # Sort by degree descending
        ranked = sorted(degree.items(), key=lambda x: -x[1])[:limit]

        nodes: list[GraphNode] = []
        edges_out: list[GraphEdge] = []
        result_ids = {aid for aid, _ in ranked}

        for rank, (aid, deg) in enumerate(ranked):
            src = paper_data.get(aid, {"arxiv_id": aid})
            nodes.append(self._make_paper_node(src, {
                "degree": deg,
                "degree_centrality": round(deg / norm, 6),
                "degree_mode": mode,
                "rank": rank,
            }))

        # Add edges between result papers
        for aid in result_ids:
            for target in out_edges.get(aid, set()):
                if target in result_ids:
                    edges_out.append(GraphEdge(
                        source=aid, target=target, relation="cites",
                    ))

        max_deg = ranked[0][1] if ranked else 0
        avg_deg = sum(d for _, d in degree.items()) / N if N > 0 else 0
        return GraphResponse(
            nodes=nodes, edges=edges_out,
            total=N, took_ms=0,
            metadata={
                "papers_in_subgraph": N,
                "mode": mode,
                "max_degree": max_deg,
                "avg_degree": round(avg_deg, 2),
                "edges_in_subgraph": sum(len(s) for s in out_edges.values()),
            },
        )

    # ── 31. Eigenvector centrality ──

    async def _eigenvector_centrality(
        self,
        gq: GraphQuery,
        sr: SearchRequest | None,
        emb: list[float] | None,
    ) -> GraphResponse:
        """Eigenvector centrality via power iteration on the directed
        citation graph.  A node is important if it is cited by other
        important nodes (recursive prestige)."""
        import math

        limit = min(gq.limit or 50, self.MAX_RESULTS)
        max_iter = min(gq.iterations, 500)

        paper_data, out_edges, in_edges, undirected = await self._build_citation_subgraph(
            gq, sr, emb, size_multiplier=8)

        if len(paper_data) < 2:
            return GraphResponse(nodes=[], edges=[], total=0, took_ms=0,
                                 metadata={"error": "need at least 2 papers"})

        node_list = list(paper_data.keys())
        N = len(node_list)
        idx_map = {aid: i for i, aid in enumerate(node_list)}

        # Initialize scores uniformly
        scores = [1.0 / N] * N

        iteration = 0
        diff = 0.0
        for iteration in range(max_iter):
            new_scores = [0.0] * N
            for i, aid in enumerate(node_list):
                for nbr in in_edges.get(aid, set()):
                    j = idx_map.get(nbr)
                    if j is not None:
                        new_scores[i] += scores[j]
            # Normalize (L2 norm)
            norm = math.sqrt(sum(s * s for s in new_scores)) or 1.0
            new_scores = [s / norm for s in new_scores]
            # Check convergence
            diff = sum(abs(new_scores[i] - scores[i]) for i in range(N))
            scores = new_scores
            if diff < 1e-6:
                break

        # Rank by eigenvector centrality
        scored = [(node_list[i], scores[i]) for i in range(N)]
        scored.sort(key=lambda x: -x[1])
        top = scored[:limit]

        nodes: list[GraphNode] = []
        edges_out: list[GraphEdge] = []
        result_ids = {aid for aid, _ in top}

        for rank, (aid, sc) in enumerate(top):
            src = paper_data.get(aid, {"arxiv_id": aid})
            nodes.append(self._make_paper_node(src, {
                "eigenvector_centrality": round(sc, 8),
                "rank": rank,
            }))

        for aid in result_ids:
            for target in out_edges.get(aid, set()):
                if target in result_ids:
                    edges_out.append(GraphEdge(
                        source=aid, target=target, relation="cites",
                    ))

        return GraphResponse(
            nodes=nodes, edges=edges_out,
            total=N, took_ms=0,
            metadata={
                "papers_in_subgraph": N,
                "edges_in_subgraph": sum(len(s) for s in out_edges.values()),
                "max_eigenvector": round(top[0][1], 8) if top else 0,
                "iterations_run": iteration + 1 if paper_data else 0,
                "converged": diff < 1e-6 if len(paper_data) >= 2 else True,
            },
        )

    # ── 32. K-core decomposition ──

    async def _kcore_decomposition(
        self,
        gq: GraphQuery,
        sr: SearchRequest | None,
        emb: list[float] | None,
    ) -> GraphResponse:
        """K-core decomposition: iteratively peels low-degree nodes to find
        the densest subgraph.

        Each node gets a coreness value = the highest k such that the node
        belongs to the k-core (a subgraph where every node has degree >= k).

        Returns papers labeled with their coreness, grouped by core level."""
        limit = min(gq.limit or 50, self.MAX_RESULTS)

        paper_data, out_edges, in_edges, undirected = await self._build_citation_subgraph(
            gq, sr, emb, size_multiplier=8)

        if not paper_data:
            return GraphResponse(nodes=[], edges=[], total=0, took_ms=0,
                                 metadata={"error": "no papers found"})

        # Build adjacency (mutable)
        adj: dict[str, set[str]] = {aid: set(undirected.get(aid, set())) for aid in paper_data}
        degree: dict[str, int] = {aid: len(adj[aid]) for aid in paper_data}
        coreness: dict[str, int] = {}
        remaining = set(paper_data.keys())

        k = 0
        while remaining:
            # Find all nodes with degree <= k
            to_remove: list[str] = []
            changed = True
            while changed:
                changed = False
                for n in list(remaining):
                    if degree.get(n, 0) <= k:
                        to_remove.append(n)
                        remaining.discard(n)
                        changed = True
                        # Update neighbors' degrees
                        for nbr in adj.get(n, set()):
                            if nbr in remaining:
                                degree[nbr] -= 1

            for n in to_remove:
                coreness[n] = k
            k += 1

        max_core = max(coreness.values()) if coreness else 0

        # Core distribution
        core_dist: Counter[int] = Counter(coreness.values())

        # Return highest-coreness papers
        ranked = sorted(coreness.items(), key=lambda x: (-x[1], x[0]))[:limit]

        nodes: list[GraphNode] = []
        edges_out: list[GraphEdge] = []
        result_ids = {aid for aid, _ in ranked}

        for aid, core_val in ranked:
            src = paper_data.get(aid, {"arxiv_id": aid})
            nodes.append(self._make_paper_node(src, {
                "coreness": core_val,
                "max_coreness": max_core,
            }))

        for aid in result_ids:
            for target in out_edges.get(aid, set()):
                if target in result_ids:
                    edges_out.append(GraphEdge(
                        source=aid, target=target, relation="cites",
                    ))

        return GraphResponse(
            nodes=nodes, edges=edges_out,
            total=len(paper_data), took_ms=0,
            metadata={
                "papers_in_subgraph": len(paper_data),
                "max_coreness": max_core,
                "core_distribution": dict(sorted(core_dist.items())),
                "edges_in_subgraph": sum(len(s) for s in out_edges.values()),
            },
        )

    # ── 33. Articulation points ──

    async def _articulation_points(
        self,
        gq: GraphQuery,
        sr: SearchRequest | None,
        emb: list[float] | None,
    ) -> GraphResponse:
        """Find articulation points (cut vertices) in the citation graph.

        An articulation point is a node whose removal disconnects the graph.
        These are critical bridge papers between research communities.

        Uses iterative Tarjan's algorithm to avoid stack overflow."""
        limit = min(gq.limit or 50, self.MAX_RESULTS)

        paper_data, out_edges, in_edges, undirected = await self._build_citation_subgraph(
            gq, sr, emb, size_multiplier=8)

        if len(paper_data) < 3:
            return GraphResponse(nodes=[], edges=[], total=0, took_ms=0,
                                 metadata={"error": "need at least 3 papers"})

        node_list = list(paper_data.keys())

        # Iterative Tarjan's for articulation points
        disc: dict[str, int] = {}
        low: dict[str, int] = {}
        parent: dict[str, str | None] = {}
        ap_set: set[str] = set()
        timer = 0

        for start in node_list:
            if start in disc:
                continue
            # Iterative DFS
            stack: list[tuple[str, str | None, int]] = [(start, None, 0)]
            parent[start] = None

            while stack:
                u, par, nbr_idx = stack.pop()

                if nbr_idx == 0:
                    disc[u] = low[u] = timer
                    timer += 1

                neighbors = sorted(undirected.get(u, set()))
                found_next = False
                for i in range(nbr_idx, len(neighbors)):
                    v = neighbors[i]
                    if v not in disc:
                        parent[v] = u
                        stack.append((u, par, i + 1))  # resume u later
                        stack.append((v, u, 0))         # explore v
                        found_next = True
                        break
                    elif v != par:
                        low[u] = min(low[u], disc[v])

                if not found_next and par is not None:
                    # Backtrack: update parent's low value
                    low[par] = min(low[par], low[u])
                    # Check if par is an articulation point
                    if parent[par] is not None and low[u] >= disc[par]:
                        ap_set.add(par)
                    # Root check: if par is root and has 2+ children in DFS tree
                    if parent[par] is None:
                        children = sum(1 for n in node_list if parent.get(n) == par)
                        if children > 1:
                            ap_set.add(par)

        # Sort articulation points by degree (most connected first)
        ap_ranked = sorted(ap_set, key=lambda x: -len(undirected.get(x, set())))[:limit]

        nodes: list[GraphNode] = []
        edges_out: list[GraphEdge] = []
        result_ids = set(ap_ranked)

        for rank, aid in enumerate(ap_ranked):
            src = paper_data.get(aid, {"arxiv_id": aid})
            nbrs = undirected.get(aid, set())
            nodes.append(self._make_paper_node(src, {
                "is_articulation_point": True,
                "degree": len(nbrs),
                "rank": rank,
            }))

        # Add edges between articulation points
        for aid in result_ids:
            for target in out_edges.get(aid, set()):
                if target in result_ids:
                    edges_out.append(GraphEdge(
                        source=aid, target=target, relation="cites",
                    ))

        # Also add some neighbor papers for each AP to show what they bridge
        neighbor_limit = max(1, (limit * 2) // max(len(ap_ranked), 1))
        for aid in ap_ranked[:10]:
            nbrs = sorted(undirected.get(aid, set()) - result_ids)
            for nbr_id in nbrs[:neighbor_limit]:
                if nbr_id in paper_data and nbr_id not in result_ids:
                    result_ids.add(nbr_id)
                    src = paper_data.get(nbr_id, {"arxiv_id": nbr_id})
                    nodes.append(self._make_paper_node(src, {
                        "is_articulation_point": False,
                        "bridged_by": aid,
                    }))
                    edges_out.append(GraphEdge(
                        source=aid, target=nbr_id, relation="bridges",
                    ))

        return GraphResponse(
            nodes=nodes, edges=edges_out,
            total=len(paper_data), took_ms=0,
            metadata={
                "papers_in_subgraph": len(paper_data),
                "edges_in_subgraph": sum(len(s) for s in out_edges.values()),
                "articulation_points_found": len(ap_set),
                "articulation_points_returned": len(ap_ranked),
            },
        )

    # ── 34. Influence maximization ──

    async def _influence_maximization(
        self,
        gq: GraphQuery,
        sr: SearchRequest | None,
        emb: list[float] | None,
    ) -> GraphResponse:
        """Greedy influence maximization using the Independent Cascade model.

        Finds the k papers (seeds) that, if they started a cascade, would
        reach the maximum number of other papers in the citation graph.

        Uses Monte Carlo simulation (100 runs per candidate) for estimated
        influence spread."""
        import random

        limit = min(gq.limit or 50, self.MAX_RESULTS)
        k = min(gq.influence_seeds, 20)  # cap seeds to prevent combinatorial explosion
        num_simulations = 20

        paper_data, out_edges, in_edges, undirected = await self._build_citation_subgraph(
            gq, sr, emb, size_multiplier=8)

        if len(paper_data) < 3:
            return GraphResponse(nodes=[], edges=[], total=0, took_ms=0,
                                 metadata={"error": "need at least 3 papers"})

        node_list = list(paper_data.keys())
        N = len(node_list)

        # Propagation probability per edge (based on influence in-degree).
        # Influence flows cited→citing, so in-degree in the influence graph
        # = number of papers v cites = len(out_edges[v]).
        influence_in_deg: dict[str, int] = {n: max(len(out_edges.get(n, set())), 1) for n in node_list}

        def simulate_spread(seeds: set[str]) -> int:
            """Run one IC simulation from seeds, return total activated."""
            active = set(seeds)
            frontier = list(seeds)
            while frontier:
                next_frontier: list[str] = []
                for u in frontier:
                    # u tries to activate its cited_by neighbors (u influences those who cite it)
                    for v in in_edges.get(u, set()):
                        if v not in active:
                            p = 1.0 / influence_in_deg[v]
                            if random.random() < p:
                                active.add(v)
                                next_frontier.append(v)
                    # Also: u's references (u spreading backwards)
                    for v in out_edges.get(u, set()):
                        if v not in active:
                            p = 0.5 / max(len(out_edges.get(u, set())), 1)
                            if random.random() < p:
                                active.add(v)
                                next_frontier.append(v)
                frontier = next_frontier
            return len(active)

        # Greedy selection of seeds
        selected_seeds: list[str] = []
        seed_set: set[str] = set()
        marginal_gains: dict[str, float] = {}
        prev_spread = 0.0

        # Pre-filter candidates (top by degree for efficiency)
        candidates = sorted(node_list, key=lambda x: -len(undirected.get(x, set())))
        candidates = candidates[:min(len(candidates), 500)]

        for _ in range(k):
            best_node = ""
            best_spread = 0.0

            for cand in candidates:
                if cand in seed_set:
                    continue
                test_seeds = seed_set | {cand}
                total = sum(simulate_spread(test_seeds) for _ in range(num_simulations))
                avg_spread = total / num_simulations
                if avg_spread > best_spread:
                    best_spread = avg_spread
                    best_node = cand

            if best_node:
                selected_seeds.append(best_node)
                seed_set.add(best_node)
                marginal_gains[best_node] = best_spread - prev_spread
                prev_spread = best_spread

        # Build response
        nodes: list[GraphNode] = []
        edges_out: list[GraphEdge] = []
        result_ids = set(selected_seeds)

        for rank, aid in enumerate(selected_seeds):
            src = paper_data.get(aid, {"arxiv_id": aid})
            nodes.append(self._make_paper_node(src, {
                "seed_rank": rank,
                "estimated_spread": round(marginal_gains.get(aid, 0), 2),
                "degree": len(undirected.get(aid, set())),
                "is_influence_seed": True,
            }))

        # Add some influenced papers
        influenced_set: set[str] = set()
        for _ in range(10):
            active = set(seed_set)
            frontier = list(seed_set)
            while frontier:
                next_frontier: list[str] = []
                for u in frontier:
                    for v in in_edges.get(u, set()):
                        if v not in active:
                            p = 1.0 / influence_in_deg[v]
                            if random.random() < p:
                                active.add(v)
                                next_frontier.append(v)
                    for v in out_edges.get(u, set()):
                        if v not in active:
                            p = 0.5 / max(len(out_edges.get(u, set())), 1)
                            if random.random() < p:
                                active.add(v)
                                next_frontier.append(v)
                frontier = next_frontier
            influenced_set |= (active - seed_set)

        # Add top influenced papers
        influenced_ranked = sorted(influenced_set,
                                    key=lambda x: -len(undirected.get(x, set())))
        for aid in influenced_ranked[:max(limit - len(selected_seeds), 0)]:
            if aid not in result_ids and aid in paper_data:
                result_ids.add(aid)
                src = paper_data.get(aid, {"arxiv_id": aid})
                nodes.append(self._make_paper_node(src, {
                    "is_influence_seed": False,
                    "degree": len(undirected.get(aid, set())),
                }))

        # Add edges
        for aid in result_ids:
            for target in out_edges.get(aid, set()):
                if target in result_ids:
                    edges_out.append(GraphEdge(
                        source=aid, target=target, relation="cites",
                    ))
            for target in in_edges.get(aid, set()):
                if target in result_ids and target != aid:
                    edges_out.append(GraphEdge(
                        source=aid, target=target, relation="influences",
                    ))

        total_spread = sum(marginal_gains.values()) / k if k > 0 else 0
        return GraphResponse(
            nodes=nodes, edges=edges_out,
            total=N, took_ms=0,
            metadata={
                "papers_in_subgraph": N,
                "edges_in_subgraph": sum(len(s) for s in out_edges.values()),
                "seeds_selected": len(selected_seeds),
                "seed_ids": selected_seeds,
                "estimated_total_spread": round(prev_spread if selected_seeds else 0, 2),
                "influenced_papers_sampled": len(influenced_set),
                "simulations_per_candidate": num_simulations,
            },
        )

    # ── 35. HITS (Hubs & Authorities) ──

    async def _hits(
        self,
        gq: GraphQuery,
        sr: SearchRequest | None,
        emb: list[float] | None,
    ) -> GraphResponse:
        """HITS algorithm (Hyperlink-Induced Topic Search).

        Computes hub and authority scores for each paper:
        - Authority: paper cited by many good hubs (important papers)
        - Hub: paper that cites many good authorities (survey papers)
        """
        import math

        limit = min(gq.limit or 50, self.MAX_RESULTS)
        max_iter = min(gq.iterations, 500)

        paper_data, out_edges, in_edges, undirected = await self._build_citation_subgraph(
            gq, sr, emb, size_multiplier=8)

        if len(paper_data) < 2:
            return GraphResponse(nodes=[], edges=[], total=0, took_ms=0,
                                 metadata={"error": "need at least 2 papers"})

        node_list = list(paper_data.keys())
        N = len(node_list)
        idx = {aid: i for i, aid in enumerate(node_list)}

        auth = [1.0 / N] * N
        hub = [1.0 / N] * N

        converged_at = max_iter
        for it in range(max_iter):
            # Authority update: auth(v) = sum of hub(u) for all u→v
            new_auth = [0.0] * N
            for i, aid in enumerate(node_list):
                for citer in in_edges.get(aid, set()):
                    j = idx.get(citer)
                    if j is not None:
                        new_auth[i] += hub[j]
            # Hub update: hub(u) = sum of auth(v) for all u→v (use NEW auth)
            new_hub = [0.0] * N
            for i, aid in enumerate(node_list):
                for cited in out_edges.get(aid, set()):
                    j = idx.get(cited)
                    if j is not None:
                        new_hub[i] += new_auth[j]
            # Normalize
            norm_a = math.sqrt(sum(a * a for a in new_auth)) or 1.0
            norm_h = math.sqrt(sum(h * h for h in new_hub)) or 1.0
            new_auth = [a / norm_a for a in new_auth]
            new_hub = [h / norm_h for h in new_hub]
            # Convergence check
            max_delta = max(max(abs(new_auth[i] - auth[i]) for i in range(N)),
                           max(abs(new_hub[i] - hub[i]) for i in range(N)))
            auth = new_auth
            hub = new_hub
            if max_delta < 1e-6:
                converged_at = it + 1
                break

        scored = [(node_list[i], auth[i], hub[i]) for i in range(N)]
        # Sort by authority score
        by_auth = sorted(scored, key=lambda x: -x[1])[:limit]
        by_hub = sorted(scored, key=lambda x: -x[2])[:limit]

        nodes: list[GraphNode] = []
        edges_out: list[GraphEdge] = []
        seen = set()

        for rank, (aid, a, h) in enumerate(by_auth):
            src = paper_data.get(aid, {"arxiv_id": aid})
            nodes.append(self._make_paper_node(src, {
                "authority": round(a, 8),
                "hub": round(h, 8),
                "auth_rank": rank,
            }))
            seen.add(aid)

        # Add top hubs not already in results
        for rank, (aid, a, h) in enumerate(by_hub):
            if aid not in seen and len(nodes) < limit * 2:
                src = paper_data.get(aid, {"arxiv_id": aid})
                nodes.append(self._make_paper_node(src, {
                    "authority": round(a, 8),
                    "hub": round(h, 8),
                    "hub_rank": rank,
                }))
                seen.add(aid)

        for aid in seen:
            for target in out_edges.get(aid, set()):
                if target in seen:
                    edges_out.append(GraphEdge(source=aid, target=target, relation="cites"))

        return GraphResponse(
            nodes=nodes, edges=edges_out,
            total=N, took_ms=0,
            metadata={
                "papers_in_subgraph": N,
                "edges_in_subgraph": sum(len(s) for s in out_edges.values()),
                "max_authority": round(by_auth[0][1], 8) if by_auth else 0,
                "max_hub": round(by_hub[0][2], 8) if by_hub else 0,
                "iterations": converged_at,
            },
        )

    # ── 36. Harmonic centrality ──

    async def _harmonic_centrality(
        self,
        gq: GraphQuery,
        sr: SearchRequest | None,
        emb: list[float] | None,
    ) -> GraphResponse:
        """Harmonic centrality: sum of inverse distances to all other nodes.

        H(v) = Σ_{u≠v} 1/d(v,u)
        Unlike closeness, naturally handles disconnected graphs (unreachable
        nodes contribute 0 instead of making the whole thing undefined)."""
        from collections import deque

        limit = min(gq.limit or 50, self.MAX_RESULTS)

        paper_data, out_edges, in_edges, undirected = await self._build_citation_subgraph(
            gq, sr, emb, size_multiplier=8)

        if len(paper_data) < 2:
            return GraphResponse(nodes=[], edges=[], total=0, took_ms=0,
                                 metadata={"error": "need at least 2 papers"})

        node_list = list(paper_data.keys())
        N = len(node_list)

        harmonic: dict[str, float] = {}
        for v in node_list:
            # BFS from v
            dist: dict[str, int] = {v: 0}
            queue = deque([v])
            total = 0.0
            while queue:
                u = queue.popleft()
                for nbr in undirected.get(u, set()):
                    if nbr not in dist:
                        dist[nbr] = dist[u] + 1
                        total += 1.0 / dist[nbr]
                        queue.append(nbr)
            harmonic[v] = total

        # Normalize: divide by N-1
        norm = max(N - 1, 1)
        for v in node_list:
            harmonic[v] /= norm

        ranked = sorted(harmonic.items(), key=lambda x: -x[1])[:limit]

        nodes: list[GraphNode] = []
        edges_out: list[GraphEdge] = []
        result_ids = {aid for aid, _ in ranked}

        for rank, (aid, hc) in enumerate(ranked):
            src = paper_data.get(aid, {"arxiv_id": aid})
            nodes.append(self._make_paper_node(src, {
                "harmonic_centrality": round(hc, 8),
                "rank": rank,
            }))

        for aid in result_ids:
            for target in out_edges.get(aid, set()):
                if target in result_ids:
                    edges_out.append(GraphEdge(source=aid, target=target, relation="cites"))

        return GraphResponse(
            nodes=nodes, edges=edges_out,
            total=N, took_ms=0,
            metadata={
                "papers_in_subgraph": N,
                "edges_in_subgraph": sum(len(s) for s in out_edges.values()),
                "max_harmonic": round(ranked[0][1], 8) if ranked else 0,
                "min_harmonic": round(ranked[-1][1], 8) if ranked else 0,
            },
        )

    # ── 37. Katz centrality ──

    async def _katz_centrality(
        self,
        gq: GraphQuery,
        sr: SearchRequest | None,
        emb: list[float] | None,
    ) -> GraphResponse:
        """Katz centrality: counts all paths from a node, weighted by length.

        K(v) = α·A·K + 1  (iterative power method)
        α (attenuation) = damping_factor * (1/λ_max) to ensure convergence."""
        import math

        limit = min(gq.limit or 50, self.MAX_RESULTS)
        max_iter = min(gq.iterations, 500)
        alpha = gq.damping_factor * 0.5  # conservative α

        paper_data, out_edges, in_edges, undirected = await self._build_citation_subgraph(
            gq, sr, emb, size_multiplier=8)

        if len(paper_data) < 2:
            return GraphResponse(nodes=[], edges=[], total=0, took_ms=0,
                                 metadata={"error": "need at least 2 papers"})

        node_list = list(paper_data.keys())
        N = len(node_list)
        idx = {aid: i for i, aid in enumerate(node_list)}

        # Estimate max in-degree as proxy for spectral radius; cap alpha for convergence
        max_in_degree = max((len(in_edges.get(n, set())) for n in node_list), default=1) or 1
        alpha = min(alpha, 0.9 / max_in_degree)

        katz = [1.0] * N

        converged_at = max_iter
        for it in range(max_iter):
            new_katz = [1.0] * N  # β = 1
            for i, aid in enumerate(node_list):
                for citer in in_edges.get(aid, set()):
                    j = idx.get(citer)
                    if j is not None:
                        new_katz[i] += alpha * katz[j]
            max_delta = max(abs(new_katz[i] - katz[i]) for i in range(N))
            katz = new_katz
            if max_delta < 1e-6:
                converged_at = it + 1
                break

        scored = sorted(
            [(node_list[i], katz[i]) for i in range(N)],
            key=lambda x: -x[1],
        )[:limit]

        nodes: list[GraphNode] = []
        edges_out: list[GraphEdge] = []
        result_ids = {aid for aid, _ in scored}

        for rank, (aid, kz) in enumerate(scored):
            src = paper_data.get(aid, {"arxiv_id": aid})
            nodes.append(self._make_paper_node(src, {
                "katz_centrality": round(kz, 8),
                "rank": rank,
            }))

        for aid in result_ids:
            for target in out_edges.get(aid, set()):
                if target in result_ids:
                    edges_out.append(GraphEdge(source=aid, target=target, relation="cites"))

        return GraphResponse(
            nodes=nodes, edges=edges_out,
            total=N, took_ms=0,
            metadata={
                "papers_in_subgraph": N,
                "edges_in_subgraph": sum(len(s) for s in out_edges.values()),
                "max_katz": round(scored[0][1], 8) if scored else 0,
                "alpha": round(alpha, 4),
                "iterations": converged_at,
            },
        )

    # ── 38. All shortest paths ──

    async def _all_shortest_paths(
        self,
        gq: GraphQuery,
        sr: SearchRequest | None,
        emb: list[float] | None,
    ) -> GraphResponse:
        """Find ALL shortest paths between source and target (not just one).

        Uses BFS to find shortest distance, then backtracking DFS to
        enumerate all paths of that length."""
        from collections import deque

        if not gq.seed_arxiv_id or not gq.target_arxiv_id:
            return GraphResponse(nodes=[], edges=[], total=0, took_ms=0,
                                 metadata={"error": "both seed_arxiv_id and target_arxiv_id required"})
        if gq.seed_arxiv_id == gq.target_arxiv_id:
            return GraphResponse(nodes=[], edges=[], total=0, took_ms=0,
                                 metadata={"error": "source and target are the same paper"})

        source = gq.seed_arxiv_id
        target = gq.target_arxiv_id
        max_hops = gq.max_hops

        # Hop-by-hop BFS like weighted_shortest_path
        _FIELDS = ["arxiv_id", "title", "categories", "primary_category",
                    "authors", "submitted_date", "citation_stats",
                    "reference_ids", "cited_by_ids", "has_github"]

        paper_cache: dict[str, dict] = {}
        adj: dict[str, set[str]] = defaultdict(set)

        # BFS to find shortest distance and build graph
        dist: dict[str, int] = {source: 0}
        queue = deque([source])

        while queue:
            u = queue.popleft()
            if dist[u] >= max_hops:
                continue
            if len(dist) > 50000:
                break
            if u not in paper_cache:
                resp = await self.client.options(request_timeout=15).search(
                    index=self.index,
                    body={"query": {"term": {"arxiv_id": u}}, "size": 1, "_source": _FIELDS, "timeout": "10s"},
                )
                hits = resp["hits"]["hits"]
                if hits:
                    paper_cache[u] = hits[0]["_source"]
            src_data = paper_cache.get(u)
            if not src_data:
                continue
            neighbors = set()
            for rid in (src_data.get("reference_ids", []) or []):
                neighbors.add(rid)
            for cid in (src_data.get("cited_by_ids", []) or []):
                neighbors.add(cid)

            for nbr in neighbors:
                adj[u].add(nbr)
                adj[nbr].add(u)
                if nbr not in dist:
                    dist[nbr] = dist[u] + 1
                    queue.append(nbr)

        if target not in dist:
            return GraphResponse(nodes=[], edges=[], total=0, took_ms=0,
                                 metadata={"path_found": False, "source": source, "target": target})

        shortest_dist = dist[target]

        # Enumerate all shortest paths via iterative backtracking
        all_paths: list[list[str]] = []
        MAX_PATHS = 500

        # Iterative DFS to avoid RecursionError on deep/wide graphs
        stack: list[tuple[str, list[str]]] = [(source, [source])]
        while stack and len(all_paths) < MAX_PATHS:
            current, path = stack.pop()
            if current == target:
                all_paths.append(path)
                continue
            if len(path) > shortest_dist + 1:
                continue
            current_dist = dist.get(current, float('inf'))
            for nbr in adj.get(current, set()):
                if dist.get(nbr, float('inf')) == current_dist + 1:
                    stack.append((nbr, path + [nbr]))

        # Apply path filter if specified
        if gq.path_filter and all_paths:
            # Need paper data to filter, so fetch first
            tmp_ids: set[str] = set()
            for p in all_paths:
                tmp_ids.update(p)
            to_fetch_filter = [pid for pid in tmp_ids if pid not in paper_cache]
            if to_fetch_filter:
                for batch_start in range(0, len(to_fetch_filter), 200):
                    batch = to_fetch_filter[batch_start:batch_start + 200]
                    batch_resp = await self.client.options(request_timeout=15).search(
                        index=self.index,
                        body={"query": {"terms": {"arxiv_id": batch}}, "size": len(batch), "_source": _FIELDS, "timeout": "10s"},
                    )
                    for hit in batch_resp["hits"]["hits"]:
                        s = hit["_source"]
                        paper_cache[s.get("arxiv_id", "")] = s
            all_paths = self._filter_paths(all_paths, paper_cache, gq.path_filter)

        # Fetch paper data for all nodes in paths
        all_ids = set()
        for p in all_paths:
            all_ids.update(p)

        to_fetch = [pid for pid in all_ids if pid not in paper_cache]
        if to_fetch:
            for batch_start in range(0, len(to_fetch), 200):
                batch = to_fetch[batch_start:batch_start + 200]
                batch_resp = await self.client.options(request_timeout=15).search(
                    index=self.index,
                    body={"query": {"terms": {"arxiv_id": batch}}, "size": len(batch), "_source": _FIELDS, "timeout": "10s"},
                )
                for hit in batch_resp["hits"]["hits"]:
                    s = hit["_source"]
                    paper_cache[s.get("arxiv_id", "")] = s

        nodes: list[GraphNode] = []
        edges_out: list[GraphEdge] = []
        seen_nodes: set[str] = set()
        seen_edges: set[tuple[str, str]] = set()

        for path_idx, path in enumerate(all_paths):
            for i, pid in enumerate(path):
                if pid not in seen_nodes:
                    seen_nodes.add(pid)
                    src = paper_cache.get(pid, {"arxiv_id": pid})
                    nodes.append(self._make_paper_node(src, {
                        "on_path": True,
                        "path_position": i,
                    }))
                if i < len(path) - 1:
                    edge_key = (path[i], path[i + 1])
                    if edge_key not in seen_edges:
                        seen_edges.add(edge_key)
                        edges_out.append(GraphEdge(
                            source=path[i], target=path[i + 1],
                            relation="shortest_path",
                            weight=path_idx + 1,
                        ))

        return GraphResponse(
            nodes=nodes, edges=edges_out,
            total=len(all_paths), took_ms=0,
            metadata={
                "source": source,
                "target": target,
                "path_found": True,
                "shortest_distance": shortest_dist,
                "paths_found": len(all_paths),
                "paths_capped": len(all_paths) >= MAX_PATHS,
                "unique_nodes": len(seen_nodes),
            },
        )

    # ── 39. K-shortest paths (Yen's) ──

    async def _k_shortest_paths(
        self,
        gq: GraphQuery,
        sr: SearchRequest | None,
        emb: list[float] | None,
    ) -> GraphResponse:
        """Yen's K-shortest paths algorithm.

        Finds the K shortest simple paths between source and target,
        where paths may have different lengths (not just shortest)."""
        from collections import deque
        import heapq

        if not gq.seed_arxiv_id or not gq.target_arxiv_id:
            return GraphResponse(nodes=[], edges=[], total=0, took_ms=0,
                                 metadata={"error": "both seed_arxiv_id and target_arxiv_id required"})
        if gq.seed_arxiv_id == gq.target_arxiv_id:
            return GraphResponse(nodes=[], edges=[], total=0, took_ms=0,
                                 metadata={"error": "source and target are the same paper"})

        source = gq.seed_arxiv_id
        target = gq.target_arxiv_id
        K = gq.k_paths
        max_hops = gq.max_hops

        _FIELDS = ["arxiv_id", "title", "categories", "primary_category",
                    "authors", "submitted_date", "citation_stats",
                    "reference_ids", "cited_by_ids", "has_github"]

        paper_cache: dict[str, dict] = {}

        async def fetch(pid: str) -> dict | None:
            if pid in paper_cache:
                return paper_cache[pid]
            resp = await self.client.options(request_timeout=15).search(
                index=self.index,
                body={"query": {"term": {"arxiv_id": pid}}, "size": 1, "_source": _FIELDS, "timeout": "10s"},
            )
            if resp["hits"]["hits"]:
                paper_cache[pid] = resp["hits"]["hits"][0]["_source"]
                return paper_cache[pid]
            return None

        def get_neighbors(pid: str) -> set[str]:
            src = paper_cache.get(pid)
            if not src:
                return set()
            nbrs = set()
            for rid in (src.get("reference_ids", []) or []):
                nbrs.add(rid)
            for cid in (src.get("cited_by_ids", []) or []):
                nbrs.add(cid)
            return nbrs

        # BFS shortest path
        async def bfs_path(src_id: str, tgt_id: str, excluded_nodes: set[str], excluded_edges: set[tuple[str, str]]) -> list[str] | None:
            if src_id in excluded_nodes or tgt_id in excluded_nodes:
                return None
            prev: dict[str, str | None] = {src_id: None}
            depth: dict[str, int] = {src_id: 0}
            queue = deque([src_id])
            while queue:
                u = queue.popleft()
                if u == tgt_id:
                    path = []
                    cur: str | None = tgt_id
                    while cur is not None:
                        path.append(cur)
                        cur = prev[cur]
                    return path[::-1]
                if len(prev) > 50000:
                    return None
                if depth[u] >= max_hops:
                    continue
                await fetch(u)
                for nbr in get_neighbors(u):
                    if nbr not in prev and nbr not in excluded_nodes and (u, nbr) not in excluded_edges:
                        prev[nbr] = u
                        depth[nbr] = depth[u] + 1
                        queue.append(nbr)
            return None

        # Find first shortest path
        await fetch(source)
        first_path = await bfs_path(source, target, set(), set())
        if not first_path:
            return GraphResponse(nodes=[], edges=[], total=0, took_ms=0,
                                 metadata={"path_found": False, "source": source, "target": target})

        A: list[list[str]] = [first_path]
        B: list[tuple[int, list[str]]] = []  # (cost, path)

        for k in range(1, K):
            prev_path = A[k - 1]
            for i in range(len(prev_path) - 1):
                spur_node = prev_path[i]
                root_path = prev_path[:i + 1]

                excluded_edges: set[tuple[str, str]] = set()
                for p in A:
                    if p[:i + 1] == root_path and i + 1 < len(p):
                        excluded_edges.add((p[i], p[i + 1]))

                excluded_nodes = set(root_path[:-1])

                await fetch(spur_node)
                spur_path = await bfs_path(spur_node, target, excluded_nodes, excluded_edges)
                if spur_path:
                    total_path = root_path[:-1] + spur_path
                    if total_path not in A:
                        heapq.heappush(B, (len(total_path), total_path))

            # Pop from B, skipping duplicates
            while B:
                _, next_path = heapq.heappop(B)
                if next_path not in A:
                    A.append(next_path)
                    break
            else:
                break

        # Apply path filter if specified
        if gq.path_filter and A:
            # Fetch paper data for filtering
            filter_ids: set[str] = set()
            for pa in A:
                filter_ids.update(pa)
            for pid in filter_ids:
                await fetch(pid)
            A = self._filter_paths(A, paper_cache, gq.path_filter)

        # Build response
        all_ids: set[str] = set()
        for path in A:
            all_ids.update(path)

        for pid in all_ids:
            await fetch(pid)

        nodes: list[GraphNode] = []
        edges_out: list[GraphEdge] = []
        seen_edges: set[tuple[str, str]] = set()

        for pid in all_ids:
            src = paper_cache.get(pid, {"arxiv_id": pid})
            path_indices = [idx for idx, p in enumerate(A) if pid in p]
            nodes.append(self._make_paper_node(src, {"on_paths": path_indices}))

        for path_idx, path in enumerate(A):
            for i in range(len(path) - 1):
                ek = (path[i], path[i + 1])
                if ek not in seen_edges:
                    seen_edges.add(ek)
                    edges_out.append(GraphEdge(
                        source=path[i], target=path[i + 1],
                        relation="path_step", weight=path_idx + 1,
                    ))

        return GraphResponse(
            nodes=nodes, edges=edges_out,
            total=len(A), took_ms=0,
            metadata={
                "source": source,
                "target": target,
                "k_requested": K,
                "paths_found": len(A),
                "path_lengths": [len(p) - 1 for p in A],
                "unique_nodes": len(all_ids),
            },
        )

    # ── 40. Random walk / Personalized PageRank ──

    async def _random_walk(
        self,
        gq: GraphQuery,
        sr: SearchRequest | None,
        emb: list[float] | None,
    ) -> GraphResponse:
        """Random walk with restart (Personalized PageRank).

        Simulates many random walks from a seed paper, with probability
        `teleport_prob` of jumping back to the seed at each step.
        Returns papers ranked by visit frequency."""
        import random

        limit = min(gq.limit or 50, self.MAX_RESULTS)

        if not gq.seed_arxiv_id:
            return GraphResponse(nodes=[], edges=[], total=0, took_ms=0,
                                 metadata={"error": "seed_arxiv_id required for random walk"})

        paper_data, out_edges, in_edges, undirected = await self._build_citation_subgraph(
            gq, sr, emb, size_multiplier=8)

        seed = gq.seed_arxiv_id
        if seed not in paper_data:
            return GraphResponse(nodes=[], edges=[], total=0, took_ms=0,
                                 metadata={"error": f"seed paper {seed} not found"})

        walk_len = gq.walk_length
        n_walks = gq.num_walks
        tp = gq.teleport_prob

        visit_count: Counter[str] = Counter()
        transition_count: Counter[tuple[str, str]] = Counter()

        for _ in range(n_walks):
            current = seed
            visit_count[current] += 1
            for step in range(walk_len):
                if random.random() < tp:
                    current = seed
                    visit_count[current] += 1
                    continue
                nbrs = list(undirected.get(current, set()))
                if not nbrs:
                    current = seed
                    visit_count[current] += 1
                    continue
                next_node = random.choice(nbrs)
                transition_count[(current, next_node)] += 1
                current = next_node
                visit_count[current] += 1

        # Normalize visit counts
        total_visits = sum(visit_count.values())
        ranked = visit_count.most_common(limit)

        nodes: list[GraphNode] = []
        edges_out: list[GraphEdge] = []
        result_ids = {aid for aid, _ in ranked}

        for rank, (aid, count) in enumerate(ranked):
            src = paper_data.get(aid, {"arxiv_id": aid})
            nodes.append(self._make_paper_node(src, {
                "visit_count": count,
                "visit_probability": round(count / total_visits, 6),
                "is_seed": aid == seed,
                "rank": rank,
            }))

        # Top transitions
        top_trans = transition_count.most_common(limit * 2)
        for (a, b), w in top_trans:
            if a in result_ids and b in result_ids:
                edges_out.append(GraphEdge(
                    source=a, target=b, relation="walk_transition", weight=w,
                ))

        return GraphResponse(
            nodes=nodes, edges=edges_out,
            total=len(visit_count), took_ms=0,
            metadata={
                "seed": seed,
                "papers_in_subgraph": len(paper_data),
                "walk_length": walk_len,
                "num_walks": n_walks,
                "teleport_prob": tp,
                "unique_visited": len(visit_count),
                "total_visits": total_visits,
            },
        )

    # ── 41. Triangle count / Clustering coefficient ──

    async def _triangle_count(
        self,
        gq: GraphQuery,
        sr: SearchRequest | None,
        emb: list[float] | None,
    ) -> GraphResponse:
        """Triangle counting and local/global clustering coefficient.

        Local CC(v) = 2T(v) / (deg(v)*(deg(v)-1))  where T(v) = triangles at v.
        Global CC = 3 * total_triangles / total_triplets."""
        limit = min(gq.limit or 50, self.MAX_RESULTS)

        paper_data, out_edges, in_edges, undirected = await self._build_citation_subgraph(
            gq, sr, emb, size_multiplier=8)

        if len(paper_data) < 3:
            return GraphResponse(nodes=[], edges=[], total=0, took_ms=0,
                                 metadata={"error": "need at least 3 papers"})

        node_list = list(paper_data.keys())

        # Count triangles for each node
        tri_count: dict[str, int] = {n: 0 for n in node_list}
        total_triangles = 0

        for v in node_list:
            nbrs_v = undirected.get(v, set())
            nbrs_list = sorted(nbrs_v)
            for i in range(len(nbrs_list)):
                for j in range(i + 1, len(nbrs_list)):
                    u, w = nbrs_list[i], nbrs_list[j]
                    if w in undirected.get(u, set()):
                        tri_count[v] += 1

        # Each triangle counted 3 times (once per vertex)
        total_triangles = sum(tri_count.values()) // 3

        # Local clustering coefficient
        local_cc: dict[str, float] = {}
        for v in node_list:
            deg = len(undirected.get(v, set()))
            if deg < 2:
                local_cc[v] = 0.0
            else:
                local_cc[v] = (2.0 * tri_count[v]) / (deg * (deg - 1))

        # Global clustering coefficient
        total_triplets = sum(
            len(undirected.get(v, set())) * (len(undirected.get(v, set())) - 1) // 2
            for v in node_list
        )
        global_cc = (3.0 * total_triangles / total_triplets) if total_triplets > 0 else 0.0

        # Sort by triangle count descended
        ranked = sorted(node_list, key=lambda v: (-tri_count[v], -local_cc[v]))[:limit]

        nodes: list[GraphNode] = []
        edges_out: list[GraphEdge] = []
        result_ids = set(ranked)

        for rank, aid in enumerate(ranked):
            src = paper_data.get(aid, {"arxiv_id": aid})
            nodes.append(self._make_paper_node(src, {
                "triangles": tri_count[aid],
                "clustering_coefficient": round(local_cc[aid], 6),
                "degree": len(undirected.get(aid, set())),
                "rank": rank,
            }))

        for aid in result_ids:
            for target in out_edges.get(aid, set()):
                if target in result_ids:
                    edges_out.append(GraphEdge(source=aid, target=target, relation="cites"))

        avg_cc = sum(local_cc.values()) / len(node_list) if node_list else 0

        return GraphResponse(
            nodes=nodes, edges=edges_out,
            total=len(paper_data), took_ms=0,
            metadata={
                "papers_in_subgraph": len(paper_data),
                "edges_in_subgraph": sum(len(s) for s in out_edges.values()),
                "total_triangles": total_triangles,
                "global_clustering_coefficient": round(global_cc, 6),
                "average_clustering_coefficient": round(avg_cc, 6),
                "max_triangles": max(tri_count.values()) if tri_count else 0,
            },
        )

    # ── 42. Graph diameter / Eccentricity ──

    async def _graph_diameter(
        self,
        gq: GraphQuery,
        sr: SearchRequest | None,
        emb: list[float] | None,
    ) -> GraphResponse:
        """Compute graph diameter, radius, and eccentricity of each node.

        Eccentricity(v) = max distance from v to any reachable node.
        Diameter = max eccentricity (longest shortest path).
        Radius = min eccentricity.
        Center = nodes with eccentricity == radius."""
        from collections import deque

        limit = min(gq.limit or 50, self.MAX_RESULTS)

        paper_data, out_edges, in_edges, undirected = await self._build_citation_subgraph(
            gq, sr, emb, size_multiplier=8)

        if len(paper_data) < 2:
            return GraphResponse(nodes=[], edges=[], total=0, took_ms=0,
                                 metadata={"error": "need at least 2 papers"})

        node_list = list(paper_data.keys())
        N = len(node_list)

        # BFS from each node to compute eccentricity
        eccentricity: dict[str, int] = {}
        reachable_count: dict[str, int] = {}

        for v in node_list:
            dist: dict[str, int] = {v: 0}
            queue = deque([v])
            max_dist = 0
            while queue:
                u = queue.popleft()
                for nbr in undirected.get(u, set()):
                    if nbr not in dist:
                        dist[nbr] = dist[u] + 1
                        max_dist = max(max_dist, dist[nbr])
                        queue.append(nbr)
            eccentricity[v] = max_dist
            reachable_count[v] = len(dist) - 1

        # Find largest connected component for diameter/radius/center
        component_of: dict[str, int] = {}
        comp_id = 0
        comp_sizes: dict[int, int] = {}
        for start in node_list:
            if start in component_of:
                continue
            q2 = deque([start])
            component_of[start] = comp_id
            size = 0
            while q2:
                u = q2.popleft()
                size += 1
                for nbr in undirected.get(u, set()):
                    if nbr not in component_of:
                        component_of[nbr] = comp_id
                        q2.append(nbr)
            comp_sizes[comp_id] = size
            comp_id += 1
        largest_comp = max(comp_sizes, key=comp_sizes.get) if comp_sizes else 0
        lc_nodes = {v for v in node_list if component_of.get(v) == largest_comp}

        diameter = max((eccentricity[v] for v in lc_nodes), default=0)
        radius = min((eccentricity[v] for v in lc_nodes), default=0)
        center = [v for v in lc_nodes if eccentricity[v] == radius]

        # Sort by eccentricity (center nodes first, largest component prioritized)
        ranked = sorted(node_list, key=lambda v: (
            0 if v in lc_nodes else 1,
            eccentricity[v],
            -reachable_count[v],
        ))[:limit]

        nodes: list[GraphNode] = []
        edges_out: list[GraphEdge] = []
        result_ids = set(ranked)

        for rank, aid in enumerate(ranked):
            src = paper_data.get(aid, {"arxiv_id": aid})
            nodes.append(self._make_paper_node(src, {
                "eccentricity": eccentricity[aid],
                "reachable": reachable_count[aid],
                "is_center": aid in center,
                "rank": rank,
            }))

        for aid in result_ids:
            for target in out_edges.get(aid, set()):
                if target in result_ids:
                    edges_out.append(GraphEdge(source=aid, target=target, relation="cites"))

        return GraphResponse(
            nodes=nodes, edges=edges_out,
            total=N, took_ms=0,
            metadata={
                "papers_in_subgraph": N,
                "edges_in_subgraph": sum(len(s) for s in out_edges.values()),
                "diameter": diameter,
                "radius": radius,
                "center_size": len(center),
                "center_ids": center[:100],
            },
        )

    # ── 43. Leiden community detection ──

    async def _leiden_community(
        self,
        gq: GraphQuery,
        sr: SearchRequest | None,
        emb: list[float] | None,
    ) -> GraphResponse:
        """Leiden algorithm — improved Louvain with guaranteed well-connected
        communities.

        Adds a refinement step: after moving nodes (like Louvain), each
        community is refined by checking if all nodes are still well-connected
        within their community."""
        import random

        limit = min(gq.limit or 50, self.MAX_RESULTS)
        max_iter = min(gq.iterations, 500)

        paper_data, out_edges, in_edges, undirected = await self._build_citation_subgraph(
            gq, sr, emb, size_multiplier=8)

        if len(paper_data) < 2:
            return GraphResponse(nodes=[], edges=[], total=0, took_ms=0,
                                 metadata={"error": "need at least 2 papers"})

        adj: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        for aid in paper_data:
            for nbr in undirected.get(aid, set()):
                adj[aid][nbr] += 1.0
        author_papers: dict[str, list[str]] = defaultdict(list)
        for aid, src in paper_data.items():
            for a in (src.get("authors") or [])[:50]:
                name = a.get("name", "") if isinstance(a, dict) else (str(a) if a is not None else "")
                if name and aid not in author_papers[name]:
                    author_papers[name].append(aid)
        for name, papers in author_papers.items():
            if len(papers) > 1:
                for i in range(len(papers)):
                    for j in range(i + 1, min(len(papers), i + 10)):
                        adj[papers[i]][papers[j]] += 0.5
                        adj[papers[j]][papers[i]] += 0.5

        m = sum(sum(nbrs.values()) for nbrs in adj.values()) / 2.0
        if m == 0:
            return GraphResponse(nodes=[], edges=[], total=0, took_ms=0,
                                 metadata={"error": "no edges in subgraph"})

        node_list = list(paper_data.keys())
        strength: dict[str, float] = {n: sum(adj[n].values()) for n in node_list}
        comm: dict[str, int] = {n: i for i, n in enumerate(node_list)}

        iteration = 0
        for iteration in range(max_iter):
            # Phase 1: Local moving (same as Louvain)
            moved = False
            random.shuffle(node_list)
            comm_strength: dict[int, float] = defaultdict(float)
            for n in node_list:
                comm_strength[comm[n]] += strength[n]

            for node in node_list:
                old_comm = comm[node]
                ki = strength[node]

                nbr_comms: dict[int, float] = defaultdict(float)
                for nbr, w in adj[node].items():
                    nbr_comms[comm[nbr]] += w

                ki_old = nbr_comms.get(old_comm, 0.0)

                # Remove node from its community for gain calculation
                sigma_old = comm_strength[old_comm] - ki

                best_comm = old_comm
                best_gain = 0.0
                for c, ki_in in nbr_comms.items():
                    if c == old_comm:
                        continue
                    sigma_tot = comm_strength.get(c, 0.0)
                    # Standard modularity gain: gain_into_c - loss_from_old
                    gain = (ki_in - ki_old) / m - ki * (sigma_tot - sigma_old) / (2.0 * m * m)
                    if gain > best_gain:
                        best_gain = gain
                        best_comm = c

                if best_comm != old_comm:
                    comm_strength[old_comm] -= ki
                    comm[node] = best_comm
                    comm_strength[best_comm] += ki
                    moved = True

            # Phase 2: Refinement — ensure each community is well-connected
            # For each community, check connectivity via BFS
            communities: dict[int, list[str]] = defaultdict(list)
            for n, c in comm.items():
                communities[c].append(n)

            new_comm_id = max(comm.values()) + 1
            for cid, members in list(communities.items()):
                if len(members) <= 1:
                    continue
                # BFS to find connected components within community
                comm_adj: dict[str, set[str]] = defaultdict(set)
                member_set = set(members)
                for n in members:
                    for nbr in adj[n]:
                        if nbr in member_set:
                            comm_adj[n].add(nbr)
                            comm_adj[nbr].add(n)

                visited: set[str] = set()
                components: list[list[str]] = []
                for n in members:
                    if n in visited:
                        continue
                    comp: list[str] = []
                    stack = [n]
                    while stack:
                        u = stack.pop()
                        if u in visited:
                            continue
                        visited.add(u)
                        comp.append(u)
                        for nbr in comm_adj[u]:
                            if nbr not in visited:
                                stack.append(nbr)
                    components.append(comp)

                # If community has >1 component, split
                if len(components) > 1:
                    moved = True  # splits may enable new Phase 1 moves
                    for comp in components[1:]:
                        for n in comp:
                            comm[n] = new_comm_id
                        new_comm_id += 1

            if not moved:
                break

        # Compute modularity
        comm_strength_final: dict[int, float] = defaultdict(float)
        comm_internal: dict[int, float] = defaultdict(float)
        for n in node_list:
            c = comm[n]
            comm_strength_final[c] += strength[n]
            for nbr, w in adj[n].items():
                if comm[nbr] == c:
                    comm_internal[c] += w

        modularity = 0.0
        for c in set(comm.values()):
            lc = comm_internal.get(c, 0.0) / 2.0
            sc = comm_strength_final.get(c, 0.0)
            modularity += lc / m - (sc / (2.0 * m)) ** 2

        # Build response
        communities_final: dict[int, list[str]] = defaultdict(list)
        for n, c in comm.items():
            communities_final[c].append(n)
        sorted_comms = sorted(communities_final.items(), key=lambda x: -len(x[1]))

        nodes: list[GraphNode] = []
        edges_out: list[GraphEdge] = []

        for idx, (cid, members) in enumerate(sorted_comms[:limit]):
            cat_counts: Counter[str] = Counter()
            for mid in members:
                for cat in paper_data.get(mid, {}).get("categories") or []:
                    cat_counts[cat] += 1
            top_cats = [c for c, _ in cat_counts.most_common(3)]
            comm_node_id = f"leiden_{idx}"
            nodes.append(GraphNode(
                id=comm_node_id,
                label=f"Community {idx} ({len(members)} papers)",
                type="community",
                properties={"size": len(members), "top_categories": top_cats, "member_ids": members[:200]},
            ))
            for mid in members[:100]:
                src = paper_data.get(mid, {"arxiv_id": mid})
                nodes.append(self._make_paper_node(src, {"community": idx}))
                edges_out.append(GraphEdge(source=comm_node_id, target=mid, relation="contains"))

        # Inter-community edges
        comm_idx_map = {cid: idx for idx, (cid, _) in enumerate(sorted_comms[:limit])}
        inter: Counter[tuple[int, int]] = Counter()
        for aid in node_list:
            ca = comm_idx_map.get(comm.get(aid, -1))
            if ca is None:
                continue
            for nbr in undirected.get(aid, set()):
                cb = comm_idx_map.get(comm.get(nbr, -1))
                if cb is not None and ca < cb:
                    inter[(ca, cb)] += 1
        for (ca, cb), w in inter.most_common(limit):
            edges_out.append(GraphEdge(
                source=f"leiden_{ca}", target=f"leiden_{cb}",
                relation="inter_community", weight=w,
            ))

        return GraphResponse(
            nodes=nodes, edges=edges_out,
            total=len(paper_data), took_ms=0,
            metadata={
                "papers_in_subgraph": len(paper_data),
                "communities_found": len(communities_final),
                "modularity": round(modularity, 6),
                "largest_community": len(sorted_comms[0][1]) if sorted_comms else 0,
                "algorithm": "leiden",
                "iterations": iteration + 1 if paper_data else 0,
            },
        )

    # ── 44. Bridge edges ──

    async def _bridge_edges(
        self,
        gq: GraphQuery,
        sr: SearchRequest | None,
        emb: list[float] | None,
    ) -> GraphResponse:
        """Find bridge edges — edges whose removal increases connected components.

        A bridge edge connects two otherwise-disconnected parts of the graph.
        Uses iterative Tarjan's bridge-finding algorithm."""
        limit = min(gq.limit or 50, self.MAX_RESULTS)

        paper_data, out_edges, in_edges, undirected = await self._build_citation_subgraph(
            gq, sr, emb, size_multiplier=8)

        if len(paper_data) < 3:
            return GraphResponse(nodes=[], edges=[], total=0, took_ms=0,
                                 metadata={"error": "need at least 3 papers"})

        node_list = list(paper_data.keys())
        disc: dict[str, int] = {}
        low: dict[str, int] = {}
        parent: dict[str, str | None] = {}
        bridges: list[tuple[str, str]] = []
        timer = 0

        for start in node_list:
            if start in disc:
                continue
            stack: list[tuple[str, str | None, int]] = [(start, None, 0)]
            parent[start] = None

            while stack:
                u, par, nbr_idx = stack.pop()
                if nbr_idx == 0:
                    disc[u] = low[u] = timer
                    timer += 1

                neighbors = sorted(undirected.get(u, set()))
                found_next = False
                for i in range(nbr_idx, len(neighbors)):
                    v = neighbors[i]
                    if v not in disc:
                        parent[v] = u
                        stack.append((u, par, i + 1))
                        stack.append((v, u, 0))
                        found_next = True
                        break
                    elif v != par:
                        low[u] = min(low[u], disc[v])

                if not found_next and par is not None:
                    low[par] = min(low[par], low[u])
                    if low[u] > disc[par]:
                        bridges.append((par, u))

        # Sort bridges by combined degree (most impactful first)
        bridges.sort(key=lambda b: -(len(undirected.get(b[0], set())) + len(undirected.get(b[1], set()))))
        top_bridges = bridges[:limit]

        nodes: list[GraphNode] = []
        edges_out: list[GraphEdge] = []
        seen: set[str] = set()

        for a, b in top_bridges:
            for pid in (a, b):
                if pid not in seen:
                    seen.add(pid)
                    src = paper_data.get(pid, {"arxiv_id": pid})
                    nodes.append(self._make_paper_node(src, {
                        "is_bridge_endpoint": True,
                        "degree": len(undirected.get(pid, set())),
                    }))
            edges_out.append(GraphEdge(
                source=a, target=b, relation="bridges",
                weight=len(undirected.get(a, set())) + len(undirected.get(b, set())),
            ))

        return GraphResponse(
            nodes=nodes, edges=edges_out,
            total=len(paper_data), took_ms=0,
            metadata={
                "papers_in_subgraph": len(paper_data),
                "edges_in_subgraph": sum(len(s) for s in out_edges.values()),
                "bridges_found": len(bridges),
                "bridges_returned": len(top_bridges),
            },
        )

    # ── 45. Min-cut ──

    async def _min_cut(
        self,
        gq: GraphQuery,
        sr: SearchRequest | None,
        emb: list[float] | None,
    ) -> GraphResponse:
        """Minimum cut between two papers using BFS-based max-flow
        (Edmonds-Karp / Ford-Fulkerson with BFS).

        The min-cut is the minimum number of edges that must be removed
        to disconnect source from target."""
        from collections import deque

        if not gq.seed_arxiv_id or not gq.target_arxiv_id:
            return GraphResponse(nodes=[], edges=[], total=0, took_ms=0,
                                 metadata={"error": "both seed_arxiv_id and target_arxiv_id required"})
        if gq.seed_arxiv_id == gq.target_arxiv_id:
            return GraphResponse(nodes=[], edges=[], total=0, took_ms=0,
                                 metadata={"error": "source and target are the same paper"})

        paper_data, out_edges, in_edges, undirected = await self._build_citation_subgraph(
            gq, sr, emb, size_multiplier=8)

        source = gq.seed_arxiv_id
        target = gq.target_arxiv_id

        # Ensure both endpoints are in the subgraph
        _FIELDS = ["arxiv_id", "title", "categories", "primary_category",
                    "authors", "submitted_date", "citation_stats",
                    "reference_ids", "cited_by_ids", "has_github"]
        for pid in (source, target):
            if pid not in paper_data:
                resp = await self.client.options(request_timeout=15).search(
                    index=self.index,
                    body={"query": {"term": {"arxiv_id": pid}}, "size": 1, "_source": _FIELDS, "timeout": "10s"},
                )
                if resp["hits"]["hits"]:
                    s = resp["hits"]["hits"][0]["_source"]
                    paper_data[pid] = s
                    for rid in (s.get("reference_ids", []) or []):
                        if rid in paper_data and rid != pid:
                            out_edges.setdefault(pid, set()).add(rid)
                            in_edges.setdefault(rid, set()).add(pid)
                            undirected.setdefault(pid, set()).add(rid)
                            undirected.setdefault(rid, set()).add(pid)
                    for cid in (s.get("cited_by_ids", []) or []):
                        if cid in paper_data and cid != pid:
                            in_edges.setdefault(pid, set()).add(cid)
                            out_edges.setdefault(cid, set()).add(pid)
                            undirected.setdefault(pid, set()).add(cid)
                            undirected.setdefault(cid, set()).add(pid)

        if source not in paper_data or target not in paper_data:
            return GraphResponse(nodes=[], edges=[], total=0, took_ms=0,
                                 metadata={"error": "source or target not in subgraph"})

        node_list = list(paper_data.keys())

        # Build capacity graph (undirected, capacity 1 per edge)
        cap: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for u in node_list:
            for v in undirected.get(u, set()):
                cap[u][v] = 1
                cap[v][u] = 1

        # Edmonds-Karp (BFS augmenting paths)
        flow: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        max_flow = 0

        while True:
            # BFS to find augmenting path
            parent_map: dict[str, str | None] = {source: None}
            queue = deque([source])
            found = False
            while queue and not found:
                u = queue.popleft()
                for v in cap[u]:
                    if v not in parent_map and cap[u][v] - flow[u][v] > 0:
                        parent_map[v] = u
                        if v == target:
                            found = True
                            break
                        queue.append(v)

            if not found:
                break

            # Find bottleneck
            path_flow = float('inf')
            v = target
            while v != source:
                u = parent_map[v]
                path_flow = min(path_flow, cap[u][v] - flow[u][v])
                v = u

            # Update flow
            v = target
            while v != source:
                u = parent_map[v]
                flow[u][v] += path_flow
                flow[v][u] -= path_flow
                v = u
            max_flow += path_flow

        # Find min-cut edges: BFS from source in residual graph
        reachable: set[str] = set()
        queue = deque([source])
        reachable.add(source)
        while queue:
            u = queue.popleft()
            for v in cap[u]:
                if v not in reachable and cap[u][v] - flow[u][v] > 0:
                    reachable.add(v)
                    queue.append(v)

        cut_edges: list[tuple[str, str]] = []
        for u in reachable:
            for v in cap[u]:
                if v not in reachable and cap[u][v] > 0:
                    cut_edges.append((u, v))

        nodes: list[GraphNode] = []
        edges_out: list[GraphEdge] = []
        seen: set[str] = set()

        # Add source and target
        for pid in [source, target]:
            seen.add(pid)
            src = paper_data.get(pid, {"arxiv_id": pid})
            nodes.append(self._make_paper_node(src, {
                "side": "source" if pid in reachable else "target",
                "is_endpoint": True,
            }))

        # Add cut edge endpoints
        for u, v in cut_edges:
            for pid in (u, v):
                if pid not in seen:
                    seen.add(pid)
                    src = paper_data.get(pid, {"arxiv_id": pid})
                    nodes.append(self._make_paper_node(src, {
                        "side": "source" if pid in reachable else "target",
                        "is_cut_endpoint": True,
                    }))
            edges_out.append(GraphEdge(source=u, target=v, relation="min_cut_edge"))

        return GraphResponse(
            nodes=nodes, edges=edges_out,
            total=len(paper_data), took_ms=0,
            metadata={
                "source": source,
                "target": target,
                "min_cut_value": max_flow,
                "cut_edges": len(cut_edges),
                "source_side_size": len(reachable),
                "target_side_size": len(paper_data) - len(reachable),
                "papers_in_subgraph": len(paper_data),
            },
        )

    # ── 46. Minimum spanning tree ──

    async def _minimum_spanning_tree(
        self,
        gq: GraphQuery,
        sr: SearchRequest | None,
        emb: list[float] | None,
    ) -> GraphResponse:
        """Kruskal's minimum spanning tree on the citation graph.

        Edge weight = 1 / (1 + shared_connections) so strongly connected
        papers have lower cost. Returns the MST edges and their endpoints."""
        limit = min(gq.limit or 50, self.MAX_RESULTS)

        paper_data, out_edges, in_edges, undirected = await self._build_citation_subgraph(
            gq, sr, emb, size_multiplier=8)

        if len(paper_data) < 2:
            return GraphResponse(nodes=[], edges=[], total=0, took_ms=0,
                                 metadata={"error": "need at least 2 papers"})

        node_list = list(paper_data.keys())

        # Union-Find
        parent_uf: dict[str, str] = {n: n for n in node_list}
        rank_uf: dict[str, int] = {n: 0 for n in node_list}

        def find(x: str) -> str:
            while parent_uf[x] != x:
                parent_uf[x] = parent_uf[parent_uf[x]]
                x = parent_uf[x]
            return x

        def union(x: str, y: str) -> bool:
            rx, ry = find(x), find(y)
            if rx == ry:
                return False
            if rank_uf[rx] < rank_uf[ry]:
                rx, ry = ry, rx
            parent_uf[ry] = rx
            if rank_uf[rx] == rank_uf[ry]:
                rank_uf[rx] += 1
            return True

        # Build edges with weight = 1/(1 + shared neighbors)
        edges_list: list[tuple[float, str, str]] = []
        seen_pairs: set[tuple[str, str]] = set()
        for u in node_list:
            for v in undirected.get(u, set()):
                pair = (min(u, v), max(u, v))
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    shared = len(undirected.get(u, set()) & undirected.get(v, set()))
                    weight = 1.0 / (1.0 + shared)
                    edges_list.append((weight, u, v))

        edges_list.sort()

        mst_edges: list[tuple[str, str, float]] = []
        for w, u, v in edges_list:
            if union(u, v):
                mst_edges.append((u, v, w))
                if len(mst_edges) == len(node_list) - 1:
                    break

        nodes: list[GraphNode] = []
        edges_out: list[GraphEdge] = []
        seen_nodes: set[str] = set()

        # Limit to top edges
        for u, v, w in mst_edges[:limit]:
            for pid in (u, v):
                if pid not in seen_nodes:
                    seen_nodes.add(pid)
                    src = paper_data.get(pid, {"arxiv_id": pid})
                    nodes.append(self._make_paper_node(src, {
                        "in_mst": True,
                        "degree": len(undirected.get(pid, set())),
                    }))
            edges_out.append(GraphEdge(
                source=u, target=v, relation="mst_edge",
                weight=round(w, 6),
            ))

        total_weight = sum(w for _, _, w in mst_edges)
        components = len(set(find(n) for n in node_list))

        return GraphResponse(
            nodes=nodes, edges=edges_out,
            total=len(paper_data), took_ms=0,
            metadata={
                "papers_in_subgraph": len(paper_data),
                "mst_edges": len(mst_edges),
                "total_weight": round(total_weight, 4),
                "components": components,
                "is_connected": components == 1,
            },
        )

    # ── 47. Node similarity ──

    async def _node_similarity(
        self,
        gq: GraphQuery,
        sr: SearchRequest | None,
        emb: list[float] | None,
    ) -> GraphResponse:
        """Structural node similarity based on neighborhood overlap.

        Methods:
        - jaccard:  |N(u) ∩ N(v)| / |N(u) ∪ N(v)|
        - overlap:  |N(u) ∩ N(v)| / min(|N(u)|, |N(v)|)
        - cosine:   |N(u) ∩ N(v)| / sqrt(|N(u)| * |N(v)|)

        Returns top paper pairs by similarity score."""
        import math

        limit = min(gq.limit or 50, self.MAX_RESULTS)
        method = gq.similarity_method

        paper_data, out_edges, in_edges, undirected = await self._build_citation_subgraph(
            gq, sr, emb, size_multiplier=6)

        if len(paper_data) < 3:
            return GraphResponse(nodes=[], edges=[], total=0, took_ms=0,
                                 metadata={"error": "need at least 3 papers"})

        node_list = list(paper_data.keys())

        # Only consider nodes with at least 1 neighbor
        active = [n for n in node_list if undirected.get(n)]

        # Compute similarities for top-degree nodes (cap for performance)
        active.sort(key=lambda x: -len(undirected.get(x, set())))
        active = active[:min(len(active), 2000)]

        pairs: list[tuple[str, str, float]] = []
        for i in range(len(active)):
            u = active[i]
            nu = undirected.get(u, set())
            for j in range(i + 1, len(active)):
                v = active[j]
                nv = undirected.get(v, set())
                inter = len(nu & nv)
                if inter == 0:
                    continue
                if method == "jaccard":
                    score = inter / len(nu | nv)
                elif method == "overlap":
                    score = inter / min(len(nu), len(nv))
                else:  # cosine
                    score = inter / math.sqrt(len(nu) * len(nv))
                pairs.append((u, v, score))

        pairs.sort(key=lambda x: -x[2])
        top_pairs = pairs[:limit]

        nodes: list[GraphNode] = []
        edges_out: list[GraphEdge] = []
        seen: set[str] = set()

        for u, v, score in top_pairs:
            for pid in (u, v):
                if pid not in seen:
                    seen.add(pid)
                    src = paper_data.get(pid, {"arxiv_id": pid})
                    nodes.append(self._make_paper_node(src, {
                        "degree": len(undirected.get(pid, set())),
                    }))
            edges_out.append(GraphEdge(
                source=u, target=v,
                relation="similar_to",
                weight=round(score, 6),
            ))

        return GraphResponse(
            nodes=nodes, edges=edges_out,
            total=len(paper_data), took_ms=0,
            metadata={
                "method": method,
                "papers_in_subgraph": len(paper_data),
                "pairs_computed": len(pairs),
                "pairs_returned": len(top_pairs),
                "max_similarity": round(top_pairs[0][2], 6) if top_pairs else 0,
            },
        )

    # ── 48. Bipartite projection ──

    async def _bipartite_projection(
        self,
        gq: GraphQuery,
        sr: SearchRequest | None,
        emb: list[float] | None,
    ) -> GraphResponse:
        """Bipartite graph projection.

        The paper-category (or paper-author) bipartite graph is projected
        onto one side:
        - papers:     two papers are connected if they share a category/author
        - categories: two categories are connected if they appear on the same paper
        - authors:    two authors are connected if they co-authored a paper

        Edge weight = number of shared items from the other side."""
        limit = min(gq.limit or 50, self.MAX_RESULTS)
        side = gq.projection_side

        paper_data, out_edges, in_edges, undirected = await self._build_citation_subgraph(
            gq, sr, emb, size_multiplier=6)

        if not paper_data:
            return GraphResponse(nodes=[], edges=[], total=0, took_ms=0,
                                 metadata={"error": "no papers found"})

        if side == "categories":
            # Project onto categories
            cat_papers: dict[str, set[str]] = defaultdict(set)
            for aid, src in paper_data.items():
                for cat in src.get("categories") or []:
                    cat_papers[cat].add(aid)

            cat_list = sorted(cat_papers.keys(), key=lambda c: -len(cat_papers[c]))
            pairs: list[tuple[str, str, int]] = []
            for i in range(len(cat_list)):
                for j in range(i + 1, len(cat_list)):
                    shared = len(cat_papers[cat_list[i]] & cat_papers[cat_list[j]])
                    if shared > 0:
                        pairs.append((cat_list[i], cat_list[j], shared))
            pairs.sort(key=lambda x: -x[2])
            top = pairs[:limit]

            nodes = []
            edges_out = []
            seen: set[str] = set()
            for a, b, w in top:
                for cat in (a, b):
                    if cat not in seen:
                        seen.add(cat)
                        nodes.append(GraphNode(
                            id=cat, label=cat, type="category",
                            properties={"paper_count": len(cat_papers[cat])},
                        ))
                edges_out.append(GraphEdge(source=a, target=b, relation="co_occurs", weight=w))

            return GraphResponse(
                nodes=nodes, edges=edges_out,
                total=len(cat_papers), took_ms=0,
                metadata={"projection_side": "categories", "categories_found": len(cat_papers),
                           "pairs_returned": len(top), "papers_in_subgraph": len(paper_data)},
            )

        elif side == "authors":
            # Co-authorship projection
            paper_authors: dict[str, list[str]] = {}
            for aid, src in paper_data.items():
                paper_authors[aid] = [name for a in (src.get("authors") or [])[:15]
                                       if a
                                       for name in [a.get("name", "") if isinstance(a, dict) else (str(a) if a is not None else "")]
                                       if name]

            author_coauth: Counter[tuple[str, str]] = Counter()
            for aid, authors in paper_authors.items():
                for i in range(len(authors)):
                    for j in range(i + 1, len(authors)):
                        pair = tuple(sorted([authors[i], authors[j]]))
                        author_coauth[pair] += 1

            top = author_coauth.most_common(limit)
            nodes = []
            edges_out = []
            seen_auth: set[str] = set()
            for (a, b), w in top:
                for auth in (a, b):
                    if auth not in seen_auth:
                        seen_auth.add(auth)
                        nodes.append(GraphNode(id=auth, label=auth, type="author", properties={}))
                edges_out.append(GraphEdge(source=a, target=b, relation="co_authored", weight=w))

            return GraphResponse(
                nodes=nodes, edges=edges_out,
                total=len(seen_auth), took_ms=0,
                metadata={"projection_side": "authors", "authors_found": len(seen_auth),
                           "pairs_returned": len(top), "papers_in_subgraph": len(paper_data)},
            )

        else:  # papers — connected by shared categories
            paper_cats: dict[str, set[str]] = {}
            for aid, src in paper_data.items():
                paper_cats[aid] = set(src.get("categories") or [])

            paper_list = sorted(paper_data.keys(), key=lambda x: -len(paper_cats.get(x, set())))
            paper_list = paper_list[:min(len(paper_list), 2000)]

            pairs_p: list[tuple[str, str, int]] = []
            for i in range(len(paper_list)):
                for j in range(i + 1, len(paper_list)):
                    shared = len(paper_cats.get(paper_list[i], set()) & paper_cats.get(paper_list[j], set()))
                    if shared > 0:
                        pairs_p.append((paper_list[i], paper_list[j], shared))
            pairs_p.sort(key=lambda x: -x[2])
            top_p = pairs_p[:limit]

            nodes = []
            edges_out = []
            seen_p: set[str] = set()
            for a, b, w in top_p:
                for pid in (a, b):
                    if pid not in seen_p:
                        seen_p.add(pid)
                        src = paper_data.get(pid, {"arxiv_id": pid})
                        nodes.append(self._make_paper_node(src, {
                            "category_count": len(paper_cats.get(pid, set())),
                        }))
                edges_out.append(GraphEdge(source=a, target=b, relation="shared_category", weight=w))

            return GraphResponse(
                nodes=nodes, edges=edges_out,
                total=len(paper_data), took_ms=0,
                metadata={"projection_side": "papers", "pairs_returned": len(top_p),
                           "papers_in_subgraph": len(paper_data)},
            )

    # ── 49. Adamic-Adar index (full pairwise) ──

    async def _adamic_adar_index(
        self,
        gq: GraphQuery,
        sr: SearchRequest | None,
        emb: list[float] | None,
    ) -> GraphResponse:
        """Adamic-Adar similarity index for all paper pairs.

        AA(u,v) = Σ_{w ∈ N(u)∩N(v)} 1/log(|N(w)|)

        Higher scores indicate papers that share many low-degree common
        neighbors (specialist connections)."""
        import math

        limit = min(gq.limit or 50, self.MAX_RESULTS)

        paper_data, out_edges, in_edges, undirected = await self._build_citation_subgraph(
            gq, sr, emb, size_multiplier=6)

        if len(paper_data) < 3:
            return GraphResponse(nodes=[], edges=[], total=0, took_ms=0,
                                 metadata={"error": "need at least 3 papers"})

        node_list = list(paper_data.keys())
        # Pre-compute 1/log(deg) for each node
        inv_log_deg: dict[str, float] = {}
        for n in node_list:
            deg = len(undirected.get(n, set()))
            if deg > 1:
                inv_log_deg[n] = 1.0 / math.log(deg)
            else:
                inv_log_deg[n] = 0.0

        # Top-degree nodes for performance
        active = sorted(node_list, key=lambda x: -len(undirected.get(x, set())))
        active = active[:min(len(active), 2000)]

        pairs: list[tuple[str, str, float]] = []
        for i in range(len(active)):
            u = active[i]
            nu = undirected.get(u, set())
            for j in range(i + 1, len(active)):
                v = active[j]
                nv = undirected.get(v, set())
                common = nu & nv
                if not common:
                    continue
                score = sum(inv_log_deg.get(w, 0.0) for w in common)
                if score > 0:
                    pairs.append((u, v, score))

        pairs.sort(key=lambda x: -x[2])
        top = pairs[:limit]

        nodes: list[GraphNode] = []
        edges_out: list[GraphEdge] = []
        seen: set[str] = set()

        for u, v, score in top:
            for pid in (u, v):
                if pid not in seen:
                    seen.add(pid)
                    src = paper_data.get(pid, {"arxiv_id": pid})
                    nodes.append(self._make_paper_node(src, {
                        "degree": len(undirected.get(pid, set())),
                    }))
            edges_out.append(GraphEdge(
                source=u, target=v,
                relation="adamic_adar",
                weight=round(score, 6),
            ))

        return GraphResponse(
            nodes=nodes, edges=edges_out,
            total=len(paper_data), took_ms=0,
            metadata={
                "papers_in_subgraph": len(paper_data),
                "pairs_computed": len(pairs),
                "pairs_returned": len(top),
                "max_score": round(top[0][2], 6) if top else 0,
            },
        )

    # ══════════════════════════════════════════════════════════════════
    # 50. PATTERN MATCH — Declarative structural pattern matching
    # ══════════════════════════════════════════════════════════════════

    async def _pattern_match(
        self,
        gq: GraphQuery,
        sr: SearchRequest | None,
        emb: list[float] | None,
    ) -> GraphResponse:
        """Match arbitrary structural patterns in the citation graph.

        Like Cypher's MATCH clause: define node aliases with property filters
        and edges with relation types, and the engine finds all matching
        subgraphs.

        Pattern nodes can filter on: categories, primary_category, min_citations,
        max_citations, has_github, date_from, date_to.

        Pattern edges support: cites, cited_by, co_authored, same_category.
        Variable-length edges via min_hops/max_hops.
        """
        from collections import deque

        if not gq.pattern_nodes or not gq.pattern_edges:
            return GraphResponse(nodes=[], edges=[], total=0, took_ms=0,
                                 metadata={"error": "pattern_nodes and pattern_edges required"})

        limit = min(gq.limit or 20, self.MAX_RESULTS)
        p_nodes = gq.pattern_nodes
        p_edges = gq.pattern_edges

        # Validate aliases
        aliases = {n.alias for n in p_nodes}
        for e in p_edges:
            if e.source not in aliases or e.target not in aliases:
                return GraphResponse(nodes=[], edges=[], total=0, took_ms=0,
                                     metadata={"error": f"Edge refers to unknown alias: {e.source} or {e.target}"})

        if len(p_nodes) > 5:
            return GraphResponse(nodes=[], edges=[], total=0, took_ms=0,
                                 metadata={"error": "max 5 pattern nodes allowed"})

        _FIELDS = ["arxiv_id", "title", "abstract", "categories", "primary_category",
                    "authors", "submitted_date", "updated_date", "citation_stats",
                    "references_stats", "reference_ids", "cited_by_ids",
                    "has_github", "github_urls", "page_count",
                    "doi", "journal_ref", "comments", "domains",
                    "first_author", "first_author_h_index"]

        # ── Step 1: Find candidates for the anchor node (first pattern node) ──
        anchor = p_nodes[0]
        anchor_query = self._pattern_node_es_query(anchor, sr, emb)

        resp = await self._do_search({
            "query": anchor_query,
            "size": min(limit * 25, 10000),
            "_source": _FIELDS,
        }, sr, emb)

        paper_cache: dict[str, dict] = {}
        for hit in resp["hits"]["hits"]:
            s = hit["_source"]
            paper_cache[s.get("arxiv_id", "")] = s

        if not paper_cache:
            return GraphResponse(nodes=[], edges=[], total=0, took_ms=0,
                                 metadata={"error": "no papers match anchor node filters"})

        # ── Step 2: BFS-expand to cover edges, fetching neighbors ──
        max_expansion = gq.max_expansion
        already_attempted: set[str] = set()  # Track IDs we already tried to fetch
        for _ in range(max_expansion):
            if len(paper_cache) >= 50000:
                break
            neighbor_ids: set[str] = set()
            for src in paper_cache.values():
                for rid in (src.get("reference_ids", []) or []):
                    if rid not in paper_cache and rid not in already_attempted:
                        neighbor_ids.add(rid)
                for cid in (src.get("cited_by_ids", []) or []):
                    if cid not in paper_cache and cid not in already_attempted:
                        neighbor_ids.add(cid)

            to_fetch = list(neighbor_ids)[:10000]
            if not to_fetch:
                break

            # Parallel ES batch fetches
            batches = [to_fetch[i:i + 200] for i in range(0, len(to_fetch), 200)]
            results = await asyncio.gather(*(
                self.client.options(request_timeout=15).search(
                    index=self.index,
                    body={
                        "query": {"terms": {"arxiv_id": batch}},
                        "size": len(batch),
                        "_source": _FIELDS,
                        "timeout": "10s",
                    },
                ) for batch in batches
            ), return_exceptions=True)
            for nbr_resp in results:
                if isinstance(nbr_resp, BaseException):
                    continue
                for hit in nbr_resp.get("hits", {}).get("hits", []):
                    s = hit["_source"]
                    paper_cache[s.get("arxiv_id", "")] = s
            already_attempted.update(to_fetch)

        # ── Step 3: Build adjacency structures ──
        out_edges: dict[str, set[str]] = defaultdict(set)
        in_edges: dict[str, set[str]] = defaultdict(set)

        used_relations = {e.relation for e in p_edges}

        for aid, src in paper_cache.items():
            for rid in (src.get("reference_ids", []) or []):
                if rid in paper_cache and rid != aid:
                    out_edges[aid].add(rid)
                    in_edges[rid].add(aid)

        # Lazy co-author lookup: build inverted index (author→papers, paper→authors)
        # but DON'T precompute all pairs — look up on demand in O(k) per call
        _paper_authors: dict[str, list[str]] = {}  # paper_id → list of author names
        _author_papers: dict[str, list[str]] = defaultdict(list)  # author_name → list of paper_ids
        if "co_authored" in used_relations:
            for aid, src in paper_cache.items():
                names = []
                for a in (src.get("authors") or [])[:50]:
                    name = a.get("name", "") if isinstance(a, dict) else (str(a) if a is not None else "")
                    if name:
                        names.append(name)
                        _author_papers[name].append(aid)
                _paper_authors[aid] = names

        # Lazy category lookup: build inverted index (cat→papers, paper→cats)
        _paper_cats: dict[str, list[str]] = {}  # paper_id → list of categories
        _cat_papers: dict[str, list[str]] = defaultdict(list)  # category → list of paper_ids
        if "same_category" in used_relations:
            for aid, src in paper_cache.items():
                cats = src.get("categories") or []
                _paper_cats[aid] = cats
                for cat in cats:
                    _cat_papers[cat].append(aid)

        def _get_coauthor_neighbors(paper_id: str) -> set[str]:
            """O(authors * avg_papers_per_author) instead of O(n²) precomputation."""
            result: set[str] = set()
            for author in _paper_authors.get(paper_id, []):
                for pid in _author_papers[author]:
                    if pid != paper_id:
                        result.add(pid)
            return result

        def _get_category_neighbors(paper_id: str) -> set[str]:
            """O(categories * avg_papers_per_cat) instead of O(n²) precomputation."""
            result: set[str] = set()
            for cat in _paper_cats.get(paper_id, []):
                for pid in _cat_papers[cat]:
                    if pid != paper_id:
                        result.add(pid)
            return result

        def get_neighbors(src_id: str, relation: str) -> set[str]:
            if relation == "cites":
                return out_edges.get(src_id, set())
            elif relation == "cited_by":
                return in_edges.get(src_id, set())
            elif relation == "co_authored":
                return _get_coauthor_neighbors(src_id)
            elif relation == "same_category":
                return _get_category_neighbors(src_id)
            return set()

        _reachable_cache: dict[tuple[str, str, int, int], set[str]] = {}

        def reachable(src_id: str, relation: str, min_h: int, max_h: int) -> set[str]:
            """Variable-length traversal from src_id, with memoization."""
            key = (src_id, relation, min_h, max_h)
            if key in _reachable_cache:
                return _reachable_cache[key]
            if min_h == 1 and max_h == 1:
                result = get_neighbors(src_id, relation)
                _reachable_cache[key] = result
                return result
            result: set[str] = set()
            current_frontier = {src_id}
            visited = {src_id}
            early_visited: set[str] = set()  # nodes found at hop < min_h
            for hop in range(1, max_h + 1):
                next_frontier: set[str] = set()
                for n in current_frontier:
                    for nbr in get_neighbors(n, relation):
                        if nbr not in visited:
                            next_frontier.add(nbr)
                        elif hop >= min_h and nbr in early_visited:
                            result.add(nbr)  # also reachable via longer path
                next_frontier -= visited
                visited |= next_frontier
                if hop < min_h:
                    early_visited |= next_frontier
                if hop >= min_h:
                    result |= next_frontier
                current_frontier = next_frontier
                if not current_frontier:
                    break
            _reachable_cache[key] = result
            return result

        def matches_filter(aid: str, pnode: PatternNode) -> bool:
            src = paper_cache.get(aid)
            if not src:
                return False
            f = pnode.filters
            if "categories" in f and isinstance(f["categories"], list):
                if not set(f["categories"]) & set(src.get("categories") or []):
                    return False
            if "primary_category" in f:
                if src.get("primary_category") != f["primary_category"]:
                    return False
            if "min_citations" in f:
                cs = src.get("citation_stats") or {}
                tc = (cs.get("total_citations") or 0) if isinstance(cs, dict) else 0
                if tc < _safe_int(f["min_citations"]):
                    return False
            if "max_citations" in f:
                cs = src.get("citation_stats") or {}
                tc = (cs.get("total_citations") or 0) if isinstance(cs, dict) else 0
                if tc > _safe_int(f["max_citations"]):
                    return False
            if "has_github" in f and src.get("has_github") != f["has_github"]:
                return False
            if "date_from" in f:
                sd = src.get("submitted_date", "")
                if sd and sd[:10] < f["date_from"][:10]:
                    return False
            if "date_to" in f:
                sd = src.get("submitted_date", "")
                if sd and sd[:10] > f["date_to"][:10]:
                    return False
            return True

        # ── Step 4: Find matching subgraphs via backtracking search ──
        alias_order = [n.alias for n in p_nodes]
        alias_to_pnode = {n.alias: n for n in p_nodes}
        # Build adjacency requirements per alias
        edges_from: dict[str, list[PatternEdge]] = defaultdict(list)
        edges_to: dict[str, list[PatternEdge]] = defaultdict(list)
        for e in p_edges:
            edges_from[e.source].append(e)
            edges_to[e.target].append(e)

        # Identify aliases only reachable via optional edges
        alias_has_required_edge: dict[str, bool] = {a: False for a in alias_order}
        for e in p_edges:
            if not e.optional:
                alias_has_required_edge[e.source] = True
                alias_has_required_edge[e.target] = True

        # WHERE condition helpers
        where_conditions = gq.where or []
        _MISSING = object()  # Sentinel: property missing on a real paper

        def _resolve_property(alias: str, prop: str, assignment: dict[str, str | None]) -> Any:
            """Resolve alias.property to a value from paper_cache."""
            aid = assignment.get(alias)
            if aid is None:
                return None
            src = paper_cache.get(aid)
            if not src:
                return _MISSING
            if prop == "citations":
                cs = src.get("citation_stats") or {}
                return (cs.get("total_citations") or 0) if isinstance(cs, dict) else 0
            elif prop in ("date", "submitted_date"):
                return src.get("submitted_date", "")
            elif prop == "primary_category":
                return src.get("primary_category", "")
            elif prop == "categories":
                return src.get("categories") or []
            elif prop == "has_github":
                return src.get("has_github", False)
            elif prop == "page_count":
                return src.get("page_count", 0) or 0
            elif prop == "title":
                return src.get("title", "")
            elif prop == "authors":
                return [a.get("name", "") if isinstance(a, dict) else (str(a) if a is not None else "")
                        for a in src.get("authors") or []]
            # Handle dotted nested properties like "citation_stats.total_citations"
            if "." in prop:
                obj = src
                for part in prop.split("."):
                    if isinstance(obj, dict):
                        obj = obj.get(part)
                    else:
                        return _MISSING
                return obj if obj is not None else _MISSING
            val = src.get(prop)
            return val if val is not None else _MISSING

        _KNOWN_PROPERTIES = {
            "arxiv_id", "title", "abstract", "authors",
            "categories", "domains", "primary_category",
            "submitted_date", "updated_date", "published_date", "date",
            "doi", "journal_ref", "comments", "page_count",
            "has_github", "github_urls", "pdf_url", "abstract_url",
            "first_author", "first_author_h_index",
            "citation_stats", "references_stats",
            "reference_ids", "cited_by_ids",
            "enrichment_source", "enriched_at",
            "citations",
        }
        _KNOWN_NESTED_PREFIXES = {"citation_stats", "references_stats"}

        def _resolve_value(ref: str, assignment: dict[str, str | None]) -> Any:
            """Resolve either alias.property or literal value."""
            if "." in ref:
                parts = ref.split(".", 1)
                # Only interpret as alias.property when the alias is a known
                # pattern node AND the property is recognized.  This avoids
                # treating dotted literals like "cs.AI" as alias references.
                if parts[0] in alias_order and (
                    parts[1] in _KNOWN_PROPERTIES
                    or parts[1].split(".")[0] in _KNOWN_NESTED_PREFIXES
                ):
                    return _resolve_property(parts[0], parts[1], assignment)
            # Try as literal
            if ref.lower() == "true":
                return True
            if ref.lower() == "false":
                return False
            try:
                return int(ref)
            except ValueError:
                pass
            try:
                return float(ref)
            except ValueError:
                pass
            return ref

        def _check_where(assignment: dict[str, str | None]) -> bool:
            """Check all WHERE conditions against current assignment."""
            for wc in where_conditions:
                # Only check conditions where all referenced aliases are assigned
                skip = False
                for ref in (wc.left, wc.right):
                    if "." in ref:
                        alias_part, prop_part = ref.split(".", 1)
                        if (alias_part in alias_order
                                and (prop_part in _KNOWN_PROPERTIES
                                     or prop_part.split(".")[0] in _KNOWN_NESTED_PREFIXES)
                                and alias_part not in assignment):
                            skip = True
                            break
                if skip:
                    continue

                left_val = _resolve_value(wc.left, assignment)
                right_val = _resolve_value(wc.right, assignment)
                if left_val is None or right_val is None:
                    continue  # Optional alias unbound — skip condition
                if left_val is _MISSING or right_val is _MISSING:
                    # Property missing on a real paper
                    if wc.op in ("!=", "not_in"):
                        continue  # null != anything → condition passes
                    return False  # All other comparisons fail

                try:
                    def _compare(a: Any, b: Any) -> tuple[Any, Any]:
                        """Try numeric, fall back to string for ordered comparison."""
                        try:
                            return float(a), float(b)
                        except (TypeError, ValueError):
                            sa, sb = str(a), str(b)
                            # Normalise dates to YYYY-MM-DD for fair comparison
                            if len(sa) >= 10 and sa[4:5] == "-" and sa[7:8] == "-":
                                sa = sa[:10]
                            if len(sb) >= 10 and sb[4:5] == "-" and sb[7:8] == "-":
                                sb = sb[:10]
                            return sa, sb

                    if wc.op in (">", "<", ">=", "<=", "==", "!="):
                        cmp_l, cmp_r = _compare(left_val, right_val)
                        if wc.op == ">" and not (cmp_l > cmp_r):
                            return False
                        elif wc.op == "<" and not (cmp_l < cmp_r):
                            return False
                        elif wc.op == ">=" and not (cmp_l >= cmp_r):
                            return False
                        elif wc.op == "<=" and not (cmp_l <= cmp_r):
                            return False
                        elif wc.op == "==" and cmp_l != cmp_r:
                            return False
                        elif wc.op == "!=" and cmp_l == cmp_r:
                            return False
                    elif wc.op == "in":
                        if isinstance(right_val, list):
                            if str(left_val) not in [str(x) for x in right_val]:
                                return False
                        else:
                            cmp_l, cmp_r = _compare(left_val, right_val)
                            if cmp_l != cmp_r:
                                return False
                    elif wc.op == "not_in":
                        if isinstance(right_val, list):
                            if str(left_val) in [str(x) for x in right_val]:
                                return False
                        else:
                            cmp_l, cmp_r = _compare(left_val, right_val)
                            if cmp_l == cmp_r:
                                return False
                    elif wc.op == "contains":
                        if isinstance(left_val, list):
                            if str(right_val) not in [str(x) for x in left_val]:
                                return False
                        elif str(right_val) not in str(left_val):
                            return False
                except (TypeError, ValueError):
                    pass
            return True

        matches: list[dict[str, str | None]] = []
        MAX_MATCHES = limit * 5

        def backtrack(idx: int, assignment: dict[str, str | None]):
            if len(matches) >= MAX_MATCHES:
                return
            if idx == len(alias_order):
                # Full assignment — check WHERE conditions
                if _check_where(assignment):
                    matches.append(dict(assignment))
                return

            alias = alias_order[idx]
            pnode = alias_to_pnode[alias]

            # Determine candidates from edge constraints to already-assigned nodes
            candidates: set[str] | None = None
            all_edges_optional = True

            optional_candidates: set[str] | None = None

            for pe in edges_from.get(alias, []):
                if pe.target in assignment and assignment[pe.target] is not None:
                    rev_rel = "cited_by" if pe.relation == "cites" else (
                        "cites" if pe.relation == "cited_by" else pe.relation)
                    nbrs = reachable(assignment[pe.target], rev_rel, pe.min_hops, pe.max_hops)
                    if not pe.optional:
                        all_edges_optional = False
                        candidates = nbrs if candidates is None else (candidates & nbrs)
                    else:
                        optional_candidates = nbrs if optional_candidates is None else (optional_candidates | nbrs)

            for pe in edges_to.get(alias, []):
                if pe.source in assignment and assignment[pe.source] is not None:
                    nbrs = reachable(assignment[pe.source], pe.relation, pe.min_hops, pe.max_hops)
                    if not pe.optional:
                        all_edges_optional = False
                        candidates = nbrs if candidates is None else (candidates & nbrs)
                    else:
                        optional_candidates = nbrs if optional_candidates is None else (optional_candidates | nbrs)

            # Use optional candidates only when no required edge constrained us
            if candidates is None and optional_candidates is not None:
                candidates = optional_candidates

            if candidates is None:
                # No edge constraint: use all anchor candidates
                candidates = set(paper_cache.keys())
                all_edges_optional = True

            # Apply node filters
            used = {v for v in assignment.values() if v is not None}
            found_any = False
            cand_iter = islice(candidates, 5000)  # O(k) vs O(n log n) for sorted
            for cand in cand_iter:
                if cand in used:
                    continue
                if not matches_filter(cand, pnode):
                    continue
                found_any = True
                assignment[alias] = cand
                backtrack(idx + 1, assignment)
                del assignment[alias]
                if len(matches) >= MAX_MATCHES:
                    return

            # OPTIONAL MATCH: if no candidate found and all edges are optional, assign None
            if not found_any and all_edges_optional and not alias_has_required_edge.get(alias, True):
                assignment[alias] = None
                backtrack(idx + 1, assignment)
                del assignment[alias]

        backtrack(0, {})

        # ── Step 5: Build response from matches ──
        nodes_out: list[GraphNode] = []
        edges_out: list[GraphEdge] = []
        seen_nodes: set[str] = set()
        seen_edges: set[tuple[str, str, str]] = set()

        for match_idx, match in enumerate(matches[:limit]):
            for alias, aid in match.items():
                if aid is not None and aid not in seen_nodes:
                    seen_nodes.add(aid)
                    src = paper_cache.get(aid, {"arxiv_id": aid})
                    nodes_out.append(self._make_paper_node(src, {
                        "pattern_alias": alias,
                        "match_index": match_idx,
                    }))

            for pe in p_edges:
                src_id = match.get(pe.source)
                tgt_id = match.get(pe.target)
                if src_id and tgt_id:
                    # For optional edges, verify the relationship actually exists
                    if pe.optional:
                        if tgt_id not in reachable(src_id, pe.relation, pe.min_hops, pe.max_hops):
                            continue
                    ek = (src_id, tgt_id, pe.relation)
                    if ek not in seen_edges:
                        seen_edges.add(ek)
                        edges_out.append(GraphEdge(
                            source=src_id, target=tgt_id,
                            relation=pe.relation, weight=match_idx + 1,
                        ))

        has_optional = any(e.optional for e in p_edges)
        has_where = len(where_conditions) > 0

        return GraphResponse(
            nodes=nodes_out, edges=edges_out,
            total=len(matches), took_ms=0,
            metadata={
                "matches_found": len(matches),
                "matches_returned": min(len(matches), limit),
                "matches_capped": len(matches) >= MAX_MATCHES,
                "pattern_nodes": len(p_nodes),
                "pattern_edges": len(p_edges),
                "papers_in_subgraph": len(paper_cache),
                "unique_result_nodes": len(seen_nodes),
                "where_conditions": len(where_conditions) if has_where else 0,
                "optional_edges": sum(1 for e in p_edges if e.optional) if has_optional else 0,
            },
        )

    def _pattern_node_es_query(
        self,
        pnode: PatternNode,
        sr: SearchRequest | None,
        emb: list[float] | None,
    ) -> dict[str, Any]:
        """Build an ES query for a single pattern node."""
        musts: list[dict] = []
        filters: list[dict] = []

        base = self._base_query(sr, emb)
        musts.append(base)

        f = pnode.filters
        if "categories" in f and isinstance(f["categories"], list) and f["categories"]:
            filters.append({"terms": {"categories": [str(c) for c in f["categories"]]}})
        if "primary_category" in f:
            filters.append({"term": {"primary_category": str(f["primary_category"])}})
        if "min_citations" in f:
            filters.append({"range": {"citation_stats.total_citations": {"gte": _safe_int(f["min_citations"])}}})
        if "max_citations" in f:
            filters.append({"range": {"citation_stats.total_citations": {"lte": _safe_int(f["max_citations"])}}})
        if "has_github" in f:
            filters.append({"term": {"has_github": bool(f["has_github"])}})
        if "date_from" in f:
            filters.append({"range": {"submitted_date": {"gte": str(f["date_from"])}}})
        if "date_to" in f:
            filters.append({"range": {"submitted_date": {"lte": str(f["date_to"])}}})


        if filters:
            return {"bool": {"must": musts, "filter": filters}}
        if len(musts) == 1:
            return musts[0]
        return {"bool": {"must": musts}}

    # ══════════════════════════════════════════════════════════════════
    # 51. PIPELINE — Chain multiple graph algorithms
    # ══════════════════════════════════════════════════════════════════

    async def _pipeline(
        self,
        gq: GraphQuery,
        sr: SearchRequest | None,
        emb: list[float] | None,
    ) -> GraphResponse:
        """Execute a pipeline of graph algorithms.

        Each step runs an algorithm, and the output nodes become the input
        seed paper IDs for the next step. Optionally filter between steps
        by property value.

        Example pipeline:
        1. pagerank(limit=50) → get top 50 by PageRank
        2. community_detection(limit=20) → detect communities in those 50
        3. triangle_count(limit=10) → count triangles in those communities
        """
        if not gq.pipeline_steps or len(gq.pipeline_steps) < 2:
            return GraphResponse(nodes=[], edges=[], total=0, took_ms=0,
                                 metadata={"error": "pipeline_steps requires at least 2 steps"})

        if len(gq.pipeline_steps) > 5:
            return GraphResponse(nodes=[], edges=[], total=0, took_ms=0,
                                 metadata={"error": "max 5 pipeline steps allowed"})

        final_limit = min(gq.limit or 50, self.MAX_RESULTS)
        step_results: list[dict[str, Any]] = []
        current_paper_ids: list[str] | None = None

        for step_idx, step in enumerate(gq.pipeline_steps):
            # Validate step type
            try:
                step_type = GraphQueryType(step.type)
            except ValueError:
                return GraphResponse(nodes=[], edges=[], total=0, took_ms=0,
                                     metadata={"error": f"Unknown type in step {step_idx}: {step.type}"})

            # Don't allow nested pipelines (recursive)
            if step_type == GraphQueryType.PIPELINE:
                return GraphResponse(nodes=[], edges=[], total=0, took_ms=0,
                                     metadata={"error": f"Step {step_idx}: cannot nest pipeline inside pipeline"})

            # Build GraphQuery for this step
            step_params = {
                "type": step.type,
                "limit": step.limit,
            }
            # Merge step.params but don't allow overriding type/limit
            step_params.update({k: v for k, v in step.params.items()
                                if k not in ("type", "limit")})

            # Feed previous step's node IDs as seeds
            if current_paper_ids is not None:
                if len(current_paper_ids) == 0:
                    return GraphResponse(nodes=[], edges=[], total=0, took_ms=0,
                                         metadata={
                                             "error": f"Pipeline stopped at step {step_idx}: 0 papers from previous step",
                                             "completed_steps": step_idx,
                                             "step_results": step_results,
                                         })
                step_params["seed_arxiv_ids"] = current_paper_ids[:2000]
                # Also set singular seed for handlers that require it
                # (traverse, shortest_path, random_walk, etc.)
                if "seed_arxiv_id" not in step_params:
                    step_params["seed_arxiv_id"] = current_paper_ids[0]

            try:
                step_gq = GraphQuery(**step_params)
            except Exception as e:
                return GraphResponse(nodes=[], edges=[], total=0, took_ms=0,
                                     metadata={"error": f"Invalid params in step {step_idx}: {e}"})

            # Use the same handler dispatch
            handler = {
                GraphQueryType.CATEGORY_DIVERSITY: self._category_diversity,
                GraphQueryType.COAUTHOR_NETWORK: self._coauthor_network,
                GraphQueryType.AUTHOR_BRIDGE: self._author_bridge,
                GraphQueryType.CROSS_CATEGORY_FLOW: self._cross_category_flow,
                GraphQueryType.INTERDISCIPLINARY: self._interdisciplinary,
                GraphQueryType.RISING_INTERDISCIPLINARY: self._rising_interdisciplinary,
                GraphQueryType.CITATION_TRAVERSAL: self._citation_traversal,
                GraphQueryType.PAPER_CITATION_NETWORK: self._paper_citation_network,
                GraphQueryType.AUTHOR_INFLUENCE: self._author_influence,
                GraphQueryType.TEMPORAL_EVOLUTION: self._temporal_evolution,
                GraphQueryType.PAPER_SIMILARITY: self._paper_similarity,
                GraphQueryType.DOMAIN_COLLABORATION: self._domain_collaboration,
                GraphQueryType.AUTHOR_TOPIC_EVOLUTION: self._author_topic_evolution,
                GraphQueryType.GITHUB_LANDSCAPE: self._github_landscape,
                GraphQueryType.BIBLIOGRAPHIC_COUPLING: self._bibliographic_coupling,
                GraphQueryType.COCITATION: self._cocitation,
                GraphQueryType.MULTIHOP_CITATION: self._multihop_citation,
                GraphQueryType.SHORTEST_CITATION_PATH: self._shortest_citation_path,
                GraphQueryType.PAGERANK: self._pagerank,
                GraphQueryType.COMMUNITY_DETECTION: self._community_detection,
                GraphQueryType.CITATION_PATTERNS: self._citation_patterns,
                GraphQueryType.CONNECTED_COMPONENTS: self._connected_components,
                GraphQueryType.WEIGHTED_SHORTEST_PATH: self._weighted_shortest_path,
                GraphQueryType.BETWEENNESS_CENTRALITY: self._betweenness_centrality,
                GraphQueryType.CLOSENESS_CENTRALITY: self._closeness_centrality,
                GraphQueryType.STRONGLY_CONNECTED_COMPONENTS: self._strongly_connected_components,
                GraphQueryType.TOPOLOGICAL_SORT: self._topological_sort,
                GraphQueryType.LINK_PREDICTION: self._link_prediction,
                GraphQueryType.LOUVAIN_COMMUNITY: self._louvain_community,
                GraphQueryType.DEGREE_CENTRALITY: self._degree_centrality,
                GraphQueryType.EIGENVECTOR_CENTRALITY: self._eigenvector_centrality,
                GraphQueryType.KCORE_DECOMPOSITION: self._kcore_decomposition,
                GraphQueryType.ARTICULATION_POINTS: self._articulation_points,
                GraphQueryType.INFLUENCE_MAXIMIZATION: self._influence_maximization,
                GraphQueryType.HITS: self._hits,
                GraphQueryType.HARMONIC_CENTRALITY: self._harmonic_centrality,
                GraphQueryType.KATZ_CENTRALITY: self._katz_centrality,
                GraphQueryType.ALL_SHORTEST_PATHS: self._all_shortest_paths,
                GraphQueryType.K_SHORTEST_PATHS: self._k_shortest_paths,
                GraphQueryType.RANDOM_WALK: self._random_walk,
                GraphQueryType.TRIANGLE_COUNT: self._triangle_count,
                GraphQueryType.GRAPH_DIAMETER: self._graph_diameter,
                GraphQueryType.LEIDEN_COMMUNITY: self._leiden_community,
                GraphQueryType.BRIDGE_EDGES: self._bridge_edges,
                GraphQueryType.MIN_CUT: self._min_cut,
                GraphQueryType.MINIMUM_SPANNING_TREE: self._minimum_spanning_tree,
                GraphQueryType.NODE_SIMILARITY: self._node_similarity,
                GraphQueryType.BIPARTITE_PROJECTION: self._bipartite_projection,
                GraphQueryType.ADAMIC_ADAR_INDEX: self._adamic_adar_index,
                GraphQueryType.PATTERN_MATCH: self._pattern_match,
                GraphQueryType.SUBGRAPH_PROJECTION: self._subgraph_projection,
                GraphQueryType.TRAVERSE: self._traverse,
                GraphQueryType.GRAPH_UNION: self._graph_union,
                GraphQueryType.GRAPH_INTERSECTION: self._graph_intersection,
            }.get(step_type)

            if handler is None:
                return GraphResponse(nodes=[], edges=[], total=0, took_ms=0,
                                     metadata={"error": f"No handler for step type: {step.type}"})

            # First step uses the original search request, subsequent steps don't
            step_sr = sr if step_idx == 0 and current_paper_ids is None else None
            step_emb = emb if step_idx == 0 and current_paper_ids is None else None

            # Restrict handlers that use _base_query to previous step's IDs
            prev_filter = _ctx_active_id_filter.get()
            if current_paper_ids is not None:
                _ctx_active_id_filter.set(current_paper_ids[:2000])
            try:
                result = await handler(step_gq, step_sr, step_emb)
            finally:
                _ctx_active_id_filter.set(prev_filter)

            # Extract paper IDs from result nodes
            paper_ids = [n.id for n in result.nodes if n.type == "paper"]

            # Apply inter-step filter if specified
            if step.filter_property and paper_ids:
                filtered_ids = []
                for n in result.nodes:
                    if n.type != "paper":
                        continue
                    val = n.properties.get(step.filter_property)
                    if val is None:
                        continue
                    try:
                        val = float(val)
                    except (TypeError, ValueError):
                        continue
                    if step.filter_min is not None and val < step.filter_min:
                        continue
                    if step.filter_max is not None and val > step.filter_max:
                        continue
                    filtered_ids.append(n.id)
                paper_ids = filtered_ids

            step_results.append({
                "step": step_idx,
                "type": step.type,
                "nodes_produced": len(result.nodes),
                "paper_ids_out": len(paper_ids),
                "filtered": step.filter_property is not None,
            })

            current_paper_ids = paper_ids

        # The final step's result IS the pipeline output (already executed in the loop)
        final_result = result

        # If the last step had a filter, trim the result to the filtered IDs
        if gq.pipeline_steps and gq.pipeline_steps[-1].filter_property and current_paper_ids is not None:
            retained = set(current_paper_ids)
            final_result.nodes = [n for n in final_result.nodes if n.type != "paper" or n.id in retained]
            retained_ids = {n.id for n in final_result.nodes}
            final_result.edges = [e for e in final_result.edges
                                  if e.source in retained_ids and e.target in retained_ids]
            # Remove non-paper nodes that lost all their edges (orphans)
            connected = set()
            for e in final_result.edges:
                connected.add(e.source)
                connected.add(e.target)
            final_result.nodes = [n for n in final_result.nodes if n.type == "paper" or n.id in connected]
            final_result.total = len(final_result.nodes)

        # Annotate with pipeline metadata
        final_result.metadata["pipeline_steps"] = step_results
        final_result.metadata["total_steps"] = len(gq.pipeline_steps)
        final_result.metadata["pipeline_output_type"] = gq.pipeline_steps[-1].type

        # Enforce the pipeline-level limit
        if len(final_result.nodes) > final_limit:
            final_result.nodes = final_result.nodes[:final_limit]
            retained_ids = {n.id for n in final_result.nodes}
            final_result.edges = [e for e in final_result.edges
                                  if e.source in retained_ids and e.target in retained_ids]
            final_result.total = len(final_result.nodes)

        return final_result

    # ══════════════════════════════════════════════════════════════════
    # 52. SUBGRAPH PROJECTION — Define precise subgraph, then run algo
    # ══════════════════════════════════════════════════════════════════

    async def _subgraph_projection(
        self,
        gq: GraphQuery,
        sr: SearchRequest | None,
        emb: list[float] | None,
    ) -> GraphResponse:
        """Project a precisely-defined subgraph, then run any algorithm on it.

        Unlike regular graph queries where the search context loosely defines
        the working set, this lets you explicitly control:
        - Which categories to include/exclude
        - Date ranges
        - Citation thresholds
        - Direction of edges (references only, cited_by only, or both)
        - Max nodes

        Then runs any existing algorithm on that exact subgraph.

        Example: "Run PageRank on only cs.AI papers from 2024 with 10+ citations"
        """
        if not gq.subgraph_filter or not gq.subgraph_algorithm:
            return GraphResponse(nodes=[], edges=[], total=0, took_ms=0,
                                 metadata={"error": "subgraph_filter and subgraph_algorithm required"})

        # Validate the target algorithm
        try:
            algo_type = GraphQueryType(gq.subgraph_algorithm)
        except ValueError:
            return GraphResponse(nodes=[], edges=[], total=0, took_ms=0,
                                 metadata={"error": f"Unknown algorithm: {gq.subgraph_algorithm}"})

        if algo_type in (GraphQueryType.PIPELINE, GraphQueryType.PATTERN_MATCH, GraphQueryType.SUBGRAPH_PROJECTION):
            return GraphResponse(nodes=[], edges=[], total=0, took_ms=0,
                                 metadata={"error": f"Cannot nest {gq.subgraph_algorithm} inside subgraph_projection"})

        sf = gq.subgraph_filter
        limit = min(gq.limit or 50, self.MAX_RESULTS)

        _FIELDS = ["arxiv_id", "title", "categories", "primary_category",
                    "authors", "submitted_date", "citation_stats",
                    "reference_ids", "cited_by_ids", "has_github"]

        # ── Step 1: Build ES query from subgraph filter ──
        filters: list[dict] = []
        musts: list[dict] = []

        # Start with the regular search context if provided
        if sr is not None:
            base = self._base_query(sr, emb)
            musts.append(base)

        if sf.categories:
            filters.append({"terms": {"categories": sf.categories}})
        if sf.exclude_categories:
            filters.append({"bool": {"must_not": [{"terms": {"categories": sf.exclude_categories}}]}})
        if sf.primary_category:
            filters.append({"term": {"primary_category": sf.primary_category}})
        if sf.date_from:
            filters.append({"range": {"submitted_date": {"gte": sf.date_from}}})
        if sf.date_to:
            filters.append({"range": {"submitted_date": {"lte": sf.date_to}}})
        if sf.min_citations is not None:
            filters.append({"range": {"citation_stats.total_citations": {"gte": sf.min_citations}}})
        if sf.max_citations is not None:
            filters.append({"range": {"citation_stats.total_citations": {"lte": sf.max_citations}}})
        if sf.has_github is not None:
            filters.append({"term": {"has_github": sf.has_github}})
        if sf.authors:
            filters.append({"nested": {"path": "authors", "query": {"bool": {"should": [{"match": {"authors.name": a}} for a in sf.authors]}}}})

        # Require citation data
        filters.append({"bool": {"should": [
            {"exists": {"field": "reference_ids"}},
            {"exists": {"field": "cited_by_ids"}},
        ]}})

        if sf.seed_arxiv_ids:
            seed_clause: dict = {"terms": {"arxiv_id": sf.seed_arxiv_ids[:10000]}}
            query = {"bool": {"must": [seed_clause] + musts, "filter": filters}} if (musts or filters) else seed_clause
        elif musts or filters:
            query = {"bool": {}}
            if musts:
                query["bool"]["must"] = musts
            if filters:
                query["bool"]["filter"] = filters
        else:
            query = {"bool": {"filter": filters}} if filters else {"match_all": {}}

        # ── Step 2: Fetch the projected subgraph ──
        resp = await self._do_search({
            "query": query,
            "size": min(sf.max_nodes, 10000),
            "_source": _FIELDS,
        }, sr, emb)

        paper_data: dict[str, dict] = {}
        for hit in resp["hits"]["hits"]:
            s = hit["_source"]
            paper_data[s.get("arxiv_id", "")] = s

        if not paper_data:
            return GraphResponse(nodes=[], edges=[], total=0, took_ms=0,
                                 metadata={"error": "no papers match subgraph filter"})

        # If seed IDs provided, expand neighborhood
        if sf.seed_arxiv_ids:
            neighbor_ids: set[str] = set()
            for src in paper_data.values():
                if sf.direction in ("references", "both"):
                    for rid in (src.get("reference_ids", []) or []):
                        neighbor_ids.add(rid)
                if sf.direction in ("cited_by", "both"):
                    for cid in (src.get("cited_by_ids", []) or []):
                        neighbor_ids.add(cid)

            to_fetch = [nid for nid in neighbor_ids if nid not in paper_data][:10000]
            fetch_batches = [to_fetch[i:i + 100] for i in range(0, len(to_fetch), 100)]
            async def _fetch_subgraph_batch(batch: list[str]) -> dict:
                nbr_query: dict[str, Any] = {"terms": {"arxiv_id": batch}}
                nbr_filters: list[dict[str, Any]] = [nbr_query]
                if sf.categories:
                    nbr_filters.append({"terms": {"categories": sf.categories}})
                if sf.exclude_categories:
                    nbr_filters.append({"bool": {"must_not": [{"terms": {"categories": sf.exclude_categories}}]}})
                if sf.primary_category:
                    nbr_filters.append({"term": {"primary_category": sf.primary_category}})
                if sf.date_from:
                    nbr_filters.append({"range": {"submitted_date": {"gte": sf.date_from}}})
                if sf.date_to:
                    nbr_filters.append({"range": {"submitted_date": {"lte": sf.date_to}}})
                if sf.min_citations is not None:
                    nbr_filters.append({"range": {"citation_stats.total_citations": {"gte": sf.min_citations}}})
                if sf.max_citations is not None:
                    nbr_filters.append({"range": {"citation_stats.total_citations": {"lte": sf.max_citations}}})
                if sf.has_github is not None:
                    nbr_filters.append({"term": {"has_github": sf.has_github}})
                if sf.authors:
                    nbr_filters.append({"nested": {"path": "authors", "query": {"bool": {"should": [{"match": {"authors.name": a}} for a in sf.authors]}}}})
                if len(nbr_filters) > 1:
                    nbr_query = {"bool": {"filter": nbr_filters}}
                return await self.client.options(request_timeout=15).search(
                    index=self.index,
                    body={"query": nbr_query, "size": len(batch), "_source": _FIELDS, "timeout": "10s"},
                )
            if fetch_batches:
                batch_results = await asyncio.gather(*(_fetch_subgraph_batch(b) for b in fetch_batches), return_exceptions=True)
                for nbr_resp in batch_results:
                    if isinstance(nbr_resp, BaseException):
                        continue
                    for hit in nbr_resp.get("hits", {}).get("hits", []):
                        s = hit["_source"]
                        paper_data[s.get("arxiv_id", "")] = s

        # ── Step 3: Filter edges by direction ──
        # Build edge structures respecting the direction filter
        out_edges: dict[str, set[str]] = defaultdict(set)
        in_edges: dict[str, set[str]] = defaultdict(set)
        undirected: dict[str, set[str]] = defaultdict(set)

        for aid, src in paper_data.items():
            if sf.direction in ("references", "both"):
                for rid in (src.get("reference_ids", []) or []):
                    if rid in paper_data and rid != aid:
                        out_edges[aid].add(rid)
                        in_edges[rid].add(aid)
                        undirected[aid].add(rid)
                        undirected[rid].add(aid)
            if sf.direction in ("cited_by", "both"):
                for cid in (src.get("cited_by_ids", []) or []):
                    if cid in paper_data and cid != aid:
                        in_edges[aid].add(cid)
                        out_edges[cid].add(aid)
                        undirected[aid].add(cid)
                        undirected[cid].add(aid)

        # ── Step 4: Run the algorithm on the projected subgraph ──
        algo_params = {
            "type": gq.subgraph_algorithm,
            "limit": limit,
        }
        algo_params.update({k: v for k, v in gq.subgraph_params.items()
                            if k not in ("type", "limit")})

        # For algorithms needing seed_arxiv_ids, pass paper IDs from the subgraph
        if sf.seed_arxiv_ids:
            algo_params["seed_arxiv_ids"] = sf.seed_arxiv_ids[:10000]

        try:
            algo_gq = GraphQuery(**algo_params)
        except Exception as e:
            return GraphResponse(nodes=[], edges=[], total=0, took_ms=0,
                                 metadata={"error": f"Invalid subgraph_params: {e}"})

        # Constrain the algorithm to our exact projected paper IDs.
        # This seed list is honoured by _build_citation_subgraph; for handlers
        # that use _base_query instead, we propagate the subgraph filter
        # constraints through a synthetic SearchRequest.
        projected_ids = list(paper_data.keys())[:10000]
        algo_gq.seed_arxiv_ids = projected_ids

        # Build synthetic SearchRequest carrying the subgraph filter constraints
        # so that handlers using _base_query / _agg_search also stay within scope.
        projected_sr_kwargs: dict[str, Any] = {}
        if sf.categories:
            projected_sr_kwargs["categories"] = sf.categories
        if sf.primary_category:
            projected_sr_kwargs["primary_category"] = sf.primary_category
        if sf.min_citations is not None:
            projected_sr_kwargs["min_citations"] = sf.min_citations
        if sf.max_citations is not None:
            projected_sr_kwargs["max_citations"] = sf.max_citations
        if sf.has_github is not None:
            projected_sr_kwargs["has_github"] = sf.has_github
        if sf.exclude_categories:
            projected_sr_kwargs["exclude_categories"] = sf.exclude_categories
        if sf.date_from or sf.date_to:
            from src.core.models import DateRange
            dt_kwargs: dict[str, Any] = {}
            if sf.date_from:
                dt_kwargs["gte"] = sf.date_from
            if sf.date_to:
                dt_kwargs["lte"] = sf.date_to
            projected_sr_kwargs["submitted_date"] = DateRange(**dt_kwargs)
        projected_sr = SearchRequest(**projected_sr_kwargs)

        handler = {
            GraphQueryType.CATEGORY_DIVERSITY: self._category_diversity,
            GraphQueryType.COAUTHOR_NETWORK: self._coauthor_network,
            GraphQueryType.AUTHOR_BRIDGE: self._author_bridge,
            GraphQueryType.CROSS_CATEGORY_FLOW: self._cross_category_flow,
            GraphQueryType.INTERDISCIPLINARY: self._interdisciplinary,
            GraphQueryType.RISING_INTERDISCIPLINARY: self._rising_interdisciplinary,
            GraphQueryType.CITATION_TRAVERSAL: self._citation_traversal,
            GraphQueryType.PAPER_CITATION_NETWORK: self._paper_citation_network,
            GraphQueryType.AUTHOR_INFLUENCE: self._author_influence,
            GraphQueryType.TEMPORAL_EVOLUTION: self._temporal_evolution,
            GraphQueryType.PAPER_SIMILARITY: self._paper_similarity,
            GraphQueryType.DOMAIN_COLLABORATION: self._domain_collaboration,
            GraphQueryType.AUTHOR_TOPIC_EVOLUTION: self._author_topic_evolution,
            GraphQueryType.GITHUB_LANDSCAPE: self._github_landscape,
            GraphQueryType.BIBLIOGRAPHIC_COUPLING: self._bibliographic_coupling,
            GraphQueryType.COCITATION: self._cocitation,
            GraphQueryType.MULTIHOP_CITATION: self._multihop_citation,
            GraphQueryType.SHORTEST_CITATION_PATH: self._shortest_citation_path,
            GraphQueryType.PAGERANK: self._pagerank,
            GraphQueryType.COMMUNITY_DETECTION: self._community_detection,
            GraphQueryType.CITATION_PATTERNS: self._citation_patterns,
            GraphQueryType.CONNECTED_COMPONENTS: self._connected_components,
            GraphQueryType.WEIGHTED_SHORTEST_PATH: self._weighted_shortest_path,
            GraphQueryType.BETWEENNESS_CENTRALITY: self._betweenness_centrality,
            GraphQueryType.CLOSENESS_CENTRALITY: self._closeness_centrality,
            GraphQueryType.STRONGLY_CONNECTED_COMPONENTS: self._strongly_connected_components,
            GraphQueryType.TOPOLOGICAL_SORT: self._topological_sort,
            GraphQueryType.LINK_PREDICTION: self._link_prediction,
            GraphQueryType.LOUVAIN_COMMUNITY: self._louvain_community,
            GraphQueryType.DEGREE_CENTRALITY: self._degree_centrality,
            GraphQueryType.EIGENVECTOR_CENTRALITY: self._eigenvector_centrality,
            GraphQueryType.KCORE_DECOMPOSITION: self._kcore_decomposition,
            GraphQueryType.ARTICULATION_POINTS: self._articulation_points,
            GraphQueryType.INFLUENCE_MAXIMIZATION: self._influence_maximization,
            GraphQueryType.HITS: self._hits,
            GraphQueryType.HARMONIC_CENTRALITY: self._harmonic_centrality,
            GraphQueryType.KATZ_CENTRALITY: self._katz_centrality,
            GraphQueryType.ALL_SHORTEST_PATHS: self._all_shortest_paths,
            GraphQueryType.K_SHORTEST_PATHS: self._k_shortest_paths,
            GraphQueryType.RANDOM_WALK: self._random_walk,
            GraphQueryType.TRIANGLE_COUNT: self._triangle_count,
            GraphQueryType.GRAPH_DIAMETER: self._graph_diameter,
            GraphQueryType.LEIDEN_COMMUNITY: self._leiden_community,
            GraphQueryType.BRIDGE_EDGES: self._bridge_edges,
            GraphQueryType.MIN_CUT: self._min_cut,
            GraphQueryType.MINIMUM_SPANNING_TREE: self._minimum_spanning_tree,
            GraphQueryType.NODE_SIMILARITY: self._node_similarity,
            GraphQueryType.BIPARTITE_PROJECTION: self._bipartite_projection,
            GraphQueryType.ADAMIC_ADAR_INDEX: self._adamic_adar_index,
            GraphQueryType.TRAVERSE: self._traverse,
            GraphQueryType.GRAPH_UNION: self._graph_union,
            GraphQueryType.GRAPH_INTERSECTION: self._graph_intersection,
        }.get(algo_type)

        if handler is None:
            return GraphResponse(nodes=[], edges=[], total=0, took_ms=0,
                                 metadata={"error": f"No handler for algorithm: {gq.subgraph_algorithm}"})

        # Pass projected_sr so handlers using _base_query are limited to projected IDs.
        # seed_arxiv_ids on algo_gq constrains _build_citation_subgraph users.
        prev_filter = _ctx_active_id_filter.get()
        prev_direction = _ctx_projection_direction.get()
        _ctx_active_id_filter.set(projected_ids)
        _ctx_projection_direction.set(sf.direction)
        try:
            result = await handler(algo_gq, projected_sr, None)
        finally:
            _ctx_active_id_filter.set(prev_filter)
            _ctx_projection_direction.set(prev_direction)

        # Annotate with subgraph metadata
        result.metadata["subgraph_projection"] = {
            "papers_in_projection": len(paper_data),
            "edges_in_projection": sum(len(s) for s in out_edges.values()),
            "direction": sf.direction,
            "algorithm": gq.subgraph_algorithm,
            "filters_applied": {
                k: v for k, v in sf.model_dump(exclude_none=True).items()
                if k not in ("max_nodes", "direction")
            },
        }

        return result

    # ── 53. General traversal (BFS/DFS with user predicates) ──────────────

    async def _traverse(
        self,
        gq: GraphQuery,
        sr: SearchRequest | None,
        emb: list[float] | None,
    ) -> GraphResponse:
        """General-purpose BFS traversal with user-defined predicates.

        traverse_direction: outgoing | incoming | both
        traverse_predicate: filter nodes during expansion
        traverse_until: stop conditions (max_nodes, max_depth, category, min_citations)
        """
        from collections import deque

        F = self.F
        source_id = gq.seed_arxiv_id
        if not source_id:
            return GraphResponse(
                nodes=[], edges=[], total=0, took_ms=0,
                metadata={"error": "seed_arxiv_id required for traverse"},
            )

        limit = min(gq.limit or 50, self.MAX_RESULTS)
        max_depth = min(gq.max_hops, 50)
        direction = gq.traverse_direction
        predicate = gq.traverse_predicate
        until = gq.traverse_until
        until_max_nodes = min(until.get("max_nodes", limit * 3), self.MAX_RESULTS)
        until_max_depth = min(until.get("max_depth", max_depth), 50)
        until_category = until.get("category")
        until_min_cit = _safe_int(until["min_citations"]) if until.get("min_citations") is not None else None
        collect_edges = gq.collect_edges

        _FIELDS = F.subgraph_fields
        paper_cache: dict[str, dict] = {}
        visited: set[str] = set()
        nodes_out: list[GraphNode] = []
        edges_out: list[GraphEdge] = []
        seen_edges: set[tuple[str, str]] = set()

        queue: deque[tuple[str, int]] = deque()
        queue.append((source_id, 0))
        visited.add(source_id)
        found_target = False

        while queue and len(nodes_out) < until_max_nodes:
            current_id, depth = queue.popleft()

            # Fetch paper if not cached
            if current_id not in paper_cache:
                resp = await self.client.options(request_timeout=15).search(
                    index=self.index,
                    body={
                        "query": {"term": {F.node_id: current_id}},
                        "size": 1,
                        "_source": _FIELDS,
                        "timeout": "10s",
                    },
                )
                if resp["hits"]["hits"]:
                    paper_cache[current_id] = resp["hits"]["hits"][0]["_source"]

            src = paper_cache.get(current_id)
            if not src:
                continue

            # Apply traverse_predicate filter (skip seed at depth 0)
            if predicate and depth > 0:
                if "min_citations" in predicate and F.extract_citations(src) < _safe_int(predicate["min_citations"]):
                    continue
                if "max_citations" in predicate and F.extract_citations(src) > _safe_int(predicate["max_citations"]):
                    continue
                if "categories" in predicate:
                    if not set(predicate["categories"]) & set(src.get(F.node_categories) or []):
                        continue
                if "primary_category" in predicate:
                    if src.get(F.node_primary_category) != predicate["primary_category"]:
                        continue
                if "has_github" in predicate:
                    if src.get(F.has_code) != predicate["has_github"]:
                        continue
                if "date_from" in predicate:
                    ts = src.get(F.node_timestamp, "")
                    if ts and ts[:10] < predicate["date_from"][:10]:
                        continue
                if "date_to" in predicate:
                    ts = src.get(F.node_timestamp, "")
                    if ts and ts[:10] > predicate["date_to"][:10]:
                        continue

            nodes_out.append(self._make_paper_node(src, {"depth": depth}))

            # Check until (stop) conditions
            if until_category and until_category in (src.get(F.node_categories) or []):
                found_target = True
                break
            if until_min_cit is not None and F.extract_citations(src) >= until_min_cit:
                found_target = True
                break

            if depth >= until_max_depth:
                continue

            # Collect neighbor IDs based on direction
            outgoing_ids: list[str] = []
            incoming_ids: list[str] = []
            if direction in ("outgoing", "both"):
                outgoing_ids = F.extract_outgoing(src)
            if direction in ("incoming", "both"):
                incoming_ids = F.extract_incoming(src)
            neighbor_ids: list[str] = outgoing_ids + incoming_ids

            # Batch-fetch neighbors not yet cached
            to_fetch = [nid for nid in neighbor_ids if nid not in paper_cache and nid not in visited][:2000]
            if to_fetch:
                batches = [to_fetch[i:i + 200] for i in range(0, len(to_fetch), 200)]
                batch_results = await asyncio.gather(*(
                    self.client.options(request_timeout=15).search(
                        index=self.index,
                        body={
                            "query": {"terms": {F.node_id: batch}},
                            "size": len(batch),
                            "_source": _FIELDS,
                            "timeout": "10s",
                        },
                    ) for batch in batches
                ), return_exceptions=True)
                for br in batch_results:
                    if isinstance(br, BaseException):
                        continue
                    for hit in br.get("hits", {}).get("hits", []):
                        s = hit["_source"]
                        paper_cache[F.extract_id(s)] = s

            for nid in neighbor_ids:
                if nid not in visited and len(nodes_out) < until_max_nodes:
                    visited.add(nid)
                    queue.append((nid, depth + 1))

            # Record edges separately — a node can appear in both outgoing and incoming
            if collect_edges:
                for nid in outgoing_ids:
                    if nid == current_id:
                        continue
                    ek = (current_id, nid)
                    if nid in visited and ek not in seen_edges:
                        seen_edges.add(ek)
                        edges_out.append(GraphEdge(source=current_id, target=nid, relation="cites"))
                for nid in incoming_ids:
                    if nid == current_id:
                        continue
                    ek = (nid, current_id)
                    if nid in visited and ek not in seen_edges:
                        seen_edges.add(ek)
                        edges_out.append(GraphEdge(source=nid, target=current_id, relation="cites"))

        # Filter out dangling edges (nodes that failed predicate or weren't fetched)
        if collect_edges:
            node_ids = {n.id for n in nodes_out}
            edges_out = [e for e in edges_out if e.source in node_ids and e.target in node_ids]

        # Enforce gq.limit on output (until_max_nodes may be larger)
        nodes_out = nodes_out[:limit]
        if collect_edges:
            node_ids = {n.id for n in nodes_out}
            edges_out = [e for e in edges_out if e.source in node_ids and e.target in node_ids]

        return GraphResponse(
            nodes=nodes_out,
            edges=edges_out if collect_edges else [],
            total=len(nodes_out),
            took_ms=0,
            metadata={
                "direction": direction,
                "max_depth": until_max_depth,
                "papers_in_subgraph": len(nodes_out),
                "predicate_filters": list(predicate.keys()) if predicate else [],
                "until_conditions": list(until.keys()) if until else [],
                "target_found": found_target,
            },
        )

    # ── 54. Graph union ──────────────────────────────────────────────────

    async def _graph_union(
        self,
        gq: GraphQuery,
        sr: SearchRequest | None,
        emb: list[float] | None,
    ) -> GraphResponse:
        """Union of two sub-query results: all nodes and edges from both."""
        return await self._graph_set_op(gq, sr, emb, mode="union")

    # ── 55. Graph intersection ───────────────────────────────────────────

    async def _graph_intersection(
        self,
        gq: GraphQuery,
        sr: SearchRequest | None,
        emb: list[float] | None,
    ) -> GraphResponse:
        """Intersection of two sub-query results: only nodes in both."""
        return await self._graph_set_op(gq, sr, emb, mode="intersection")

    async def _graph_set_op(
        self,
        gq: GraphQuery,
        sr: SearchRequest | None,
        emb: list[float] | None,
        mode: str,
        _depth: int = 0,
    ) -> GraphResponse:
        """Execute two sub-queries and combine their graph results."""
        _MAX_SET_DEPTH = 3
        if _depth >= _MAX_SET_DEPTH:
            return GraphResponse(
                nodes=[], edges=[], total=0, took_ms=0,
                metadata={"error": f"set operation nesting too deep (max {_MAX_SET_DEPTH})"},
            )

        if not gq.set_queries or len(gq.set_queries) < 2:
            return GraphResponse(
                nodes=[], edges=[], total=0, took_ms=0,
                metadata={"error": "set_queries requires at least 2 sub-queries"},
            )

        # Block recursive nesting that could cause exponential blowup
        _BLOCKED_NESTED = {
            GraphQueryType.GRAPH_UNION, GraphQueryType.GRAPH_INTERSECTION,
            GraphQueryType.PIPELINE, GraphQueryType.SUBGRAPH_PROJECTION,
        }

        # Save parent embeddings and ID filter before sub-execute() calls,
        # because execute()'s finally block clears the context vars.
        saved_embeddings = _ctx_embeddings.get()
        saved_id_filter = _ctx_active_id_filter.get()

        sub_results: list[GraphResponse] = []
        for i, sq_dict in enumerate(gq.set_queries[:2]):
            try:
                sub_gq = GraphQuery(**sq_dict)
            except Exception as e:
                return GraphResponse(
                    nodes=[], edges=[], total=0, took_ms=0,
                    metadata={"error": f"Invalid sub-query {i + 1}: {e}"},
                )
            if sub_gq.type in _BLOCKED_NESTED:
                return GraphResponse(
                    nodes=[], edges=[], total=0, took_ms=0,
                    metadata={"error": f"Sub-query {i + 1}: {sub_gq.type.value} cannot be nested inside {mode}"},
                )
            _ctx_active_id_filter.set(saved_id_filter)
            try:
                sub_result = await self.execute(sub_gq, sr, embeddings=saved_embeddings)
            except Exception as e:
                return GraphResponse(
                    nodes=[], edges=[], total=0, took_ms=0,
                    metadata={"error": f"Sub-query {i + 1} failed: {e}"},
                )
            sub_results.append(sub_result)

        r1, r2 = sub_results

        # Disambiguate synthetic IDs (community_0, component_0, …) that
        # collide between sub-queries.  Paper/author/category IDs are
        # globally meaningful so same-ID = same-entity; synthetic group
        # IDs are positional and must be renamed to avoid false matches.
        _SYNTHETIC_TYPES = {"community", "component", "scc"}
        _id_set_1 = {n.id for n in r1.nodes}
        _renames: dict[str, str] = {}
        for n in r2.nodes:
            if n.id in _id_set_1 and n.type in _SYNTHETIC_TYPES:
                new_id = f"sq2_{n.id}"
                _renames[n.id] = new_id
                n.id = new_id
        if _renames:
            for e in r2.edges:
                e.source = _renames.get(e.source, e.source)
                e.target = _renames.get(e.target, e.target)

        ids1 = {n.id for n in r1.nodes}
        ids2 = {n.id for n in r2.nodes}
        result_limit = min(gq.limit or self.MAX_RESULTS, self.MAX_RESULTS)

        if mode == "union":
            node_map: dict[str, GraphNode] = {}
            for n in r1.nodes + r2.nodes:
                if n.id not in node_map:
                    node_map[n.id] = n
            nodes = list(node_map.values())[:result_limit]
            retained_ids = {n.id for n in nodes}
            edge_set: set[tuple[str, str, str]] = set()
            edges: list[GraphEdge] = []
            for e in r1.edges + r2.edges:
                ek = (e.source, e.target, e.relation)
                if ek not in edge_set and e.source in retained_ids and e.target in retained_ids:
                    edge_set.add(ek)
                    edges.append(e)
        else:
            shared = ids1 & ids2
            nodes = [n for n in r1.nodes if n.id in shared][:result_limit]
            retained_ids = {n.id for n in nodes}
            # True edge intersection: only edges present in BOTH results
            edge_keys_1 = {(e.source, e.target, e.relation) for e in r1.edges}
            edge_keys_2 = {(e.source, e.target, e.relation) for e in r2.edges}
            shared_edge_keys = edge_keys_1 & edge_keys_2
            edge_map = {(e.source, e.target, e.relation): e for e in r1.edges}
            edges = [
                edge_map[ek] for ek in shared_edge_keys
                if ek[0] in retained_ids and ek[1] in retained_ids
            ]

        return GraphResponse(
            nodes=nodes,
            edges=edges,
            total=len(nodes),
            took_ms=0,
            metadata={
                "operation": mode,
                "query_1_nodes": len(ids1),
                "query_2_nodes": len(ids2),
                "result_nodes": len(nodes),
                "result_edges": len(edges),
            },
        )
