from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


# ── ArXiv Paper Model ──

class Author(BaseModel):
    name: str
    is_first_author: bool = False
    h_index: int | None = None
    citation_count: int | None = None


class CitationStats(BaseModel):
    total_citations: int = 0
    avg_citation_age_years: float | None = None
    median_h_index_citing_authors: float | None = None
    top_citing_categories: list[str] = Field(default_factory=list)


class ReferencesStats(BaseModel):
    total_references: int = 0
    avg_reference_age_years: float | None = None
    top_referenced_categories: list[str] = Field(default_factory=list)


class Paper(BaseModel):
    arxiv_id: str
    title: str
    abstract: str
    authors: list[Author] = Field(default_factory=list)
    categories: list[str] = Field(default_factory=list)
    primary_category: str | None = None
    submitted_date: datetime | None = None
    updated_date: datetime | None = None
    published_date: datetime | None = None
    doi: str | None = None
    journal_ref: str | None = None
    comments: str | None = None
    page_count: int | None = None
    has_github: bool = False
    github_urls: list[str] = Field(default_factory=list)
    pdf_url: str | None = None
    abstract_url: str | None = None
    domains: list[str] = Field(default_factory=list)

    # Embeddings (stored but not returned by default)
    title_embedding: list[float] | None = None
    abstract_embedding: list[float] | None = None
    paragraph_embeddings: list[list[float]] | None = None

    # Citation data
    citation_stats: CitationStats = Field(default_factory=CitationStats)
    references_stats: ReferencesStats = Field(default_factory=ReferencesStats)

    # Citation links (arxiv IDs)
    reference_ids: list[str] = Field(default_factory=list)  # papers this paper cites
    cited_by_ids: list[str] = Field(default_factory=list)   # papers that cite this paper

    # Computed
    first_author: str | None = None
    first_author_h_index: int | None = None


# ── Search Models ──

class SimilarityLevel(str, Enum):
    TITLE = "title"
    ABSTRACT = "abstract"
    PARAGRAPH = "paragraph"


class SortField(str, Enum):
    RELEVANCE = "relevance"
    DATE = "date"
    CITATIONS = "citations"
    H_INDEX = "h_index"
    PAGE_COUNT = "page_count"
    UPDATED = "updated"


class SortOrder(str, Enum):
    ASC = "asc"
    DESC = "desc"


class DateRange(BaseModel):
    gte: datetime | None = None
    lte: datetime | None = None

    @model_validator(mode="after")
    def _check_range(self) -> "DateRange":
        if self.gte is not None and self.lte is not None and self.gte > self.lte:
            raise ValueError(
                f"DateRange gte ({self.gte.date()}) must not exceed lte ({self.lte.date()})"
            )
        return self


class SemanticMode(str, Enum):
    BOOST = "boost"
    EXCLUDE = "exclude"


class SemanticQuery(BaseModel):
    text: str = Field(..., max_length=2000)
    level: SimilarityLevel = SimilarityLevel.ABSTRACT
    weight: float = Field(default=1.0, gt=0.0, le=10.0)
    mode: SemanticMode = SemanticMode.BOOST


class SearchRequest(BaseModel):
    # Text search
    query: str | None = Field(default=None, max_length=2000)
    title_query: str | None = Field(default=None, max_length=500)
    abstract_query: str | None = Field(default=None, max_length=2000)

    # Semantic similarity (single SemanticQuery or list)
    semantic: SemanticQuery | list[SemanticQuery] | None = None

    @field_validator("semantic", mode="before")
    @classmethod
    def _normalize_semantic(cls, v: Any) -> list[SemanticQuery] | None:
        if v is None:
            return None
        if isinstance(v, dict):
            return [v]
        if isinstance(v, SemanticQuery):
            return [v]
        if isinstance(v, list):
            return v
        return v

    # Fuzzy matching
    fuzzy: str | None = Field(default=None, max_length=500)
    fuzzy_fuzziness: int = Field(default=2, ge=0, le=3)

    # Regex search
    title_regex: str | None = Field(default=None, max_length=200)
    abstract_regex: str | None = Field(default=None, max_length=200)
    author_regex: str | None = Field(default=None, max_length=200)

    # Author filters
    author: str | None = None
    first_author: str | None = None
    min_h_index: int | None = Field(default=None, ge=0)
    max_h_index: int | None = Field(default=None, ge=0)
    min_first_author_h_index: int | None = Field(default=None, ge=0)
    min_median_h_index_citing: float | None = Field(default=None, ge=0)

    # Citation filters
    min_citations: int | None = Field(default=None, ge=0)
    max_citations: int | None = Field(default=None, ge=0)
    min_references: int | None = Field(default=None, ge=0)

    # Category filters
    categories: list[str] | None = Field(default=None, max_length=50)
    primary_category: str | None = None
    exclude_categories: list[str] | None = Field(default=None, max_length=50)

    # Date filters
    submitted_date: DateRange | None = None
    updated_date: DateRange | None = None

    # Paper metadata filters
    has_github: bool | None = None
    min_page_count: int | None = Field(default=None, ge=0)
    max_page_count: int | None = Field(default=None, ge=0)
    has_doi: bool | None = None
    has_journal_ref: bool | None = None

    # Boolean / match control
    minimum_should_match: str | None = Field(
        default=None,
        pattern=r"^\d+%?$",
        description="ES minimum_should_match spec, e.g. '2', '75%'"
    )
    operator: str = Field(default="or", pattern="^(and|or)$")

    # Sorting
    sort_by: SortField = SortField.RELEVANCE
    sort_order: SortOrder = SortOrder.DESC

    # Pagination
    offset: int = Field(default=0, ge=0, le=50000)
    limit: int = Field(default=20, ge=1, le=200)

    @field_validator("limit")
    @classmethod
    def validate_result_window(cls, v: int, info: Any) -> int:
        offset = info.data.get("offset", 0)
        if offset + v > 50200:
            raise ValueError(
                f"offset ({offset}) + limit ({v}) = {offset + v} exceeds max result window (50200)"
            )
        return v

    # Response control
    include_embeddings: bool = False  # Admin-only; ignored for non-admin requests
    highlight: bool = True

    @field_validator("title_regex", "abstract_regex", "author_regex")
    @classmethod
    def validate_regex_safety(cls, v: str | None) -> str | None:
        if v is None:
            return v
        import re
        # Block dangerous patterns that could cause ReDoS
        dangerous = [
            r"(.+)+",
            r"(.*)*",
            r"([a-zA-Z]+)*",
            r"(a|a)+",
        ]
        for pat in dangerous:
            if pat in v:
                raise ValueError(f"Potentially dangerous regex pattern blocked: {pat}")
        # Catch quantified groups containing quantifiers: (x+)+, (x*)+, etc.
        import re as _re
        if _re.search(r'\([^)]*[+*][^)]*\)[+*]', v):
            raise ValueError("Potentially dangerous regex pattern (nested quantifiers)")
        # Validate syntax
        try:
            re.compile(v)
        except re.error as exc:
            raise ValueError(f"Invalid regex: {exc}") from exc
        return v


class SearchHit(BaseModel):
    arxiv_id: str
    title: str
    abstract: str
    authors: list[Author] = Field(default_factory=list)
    categories: list[str] = Field(default_factory=list)
    primary_category: str | None = None
    submitted_date: datetime | None = None
    updated_date: datetime | None = None
    published_date: datetime | None = None
    doi: str | None = None
    journal_ref: str | None = None
    page_count: int | None = None
    has_github: bool = False
    github_urls: list[str] = Field(default_factory=list)
    first_author: str | None = None
    first_author_h_index: int | None = None
    comments: str | None = None
    pdf_url: str | None = None
    abstract_url: str | None = None
    domains: list[str] = Field(default_factory=list)
    reference_ids: list[str] = Field(default_factory=list)
    cited_by_ids: list[str] = Field(default_factory=list)
    citation_stats: CitationStats = Field(default_factory=CitationStats)
    references_stats: ReferencesStats = Field(default_factory=ReferencesStats)
    enrichment_source: str | None = None
    enriched_at: datetime | None = None
    score: float | None = None
    highlights: dict[str, list[str]] | None = None


class SearchResponse(BaseModel):
    total: int
    hits: list[SearchHit]
    took_ms: int
    offset: int
    limit: int


class StatsResponse(BaseModel):
    total_papers: int
    categories: dict[str, int]
    date_range: dict[str, str | None]
    papers_with_github: int
    avg_page_count: float | None
    avg_citations: float | None


class HealthResponse(BaseModel):
    status: str


# ── Graph Query Models ──

class GraphQueryType(str, Enum):
    CATEGORY_DIVERSITY = "category_diversity"
    COAUTHOR_NETWORK = "coauthor_network"
    AUTHOR_BRIDGE = "author_bridge"
    CROSS_CATEGORY_FLOW = "cross_category_flow"
    INTERDISCIPLINARY = "interdisciplinary"
    RISING_INTERDISCIPLINARY = "rising_interdisciplinary"
    CITATION_TRAVERSAL = "citation_traversal"
    PAPER_CITATION_NETWORK = "paper_citation_network"
    AUTHOR_INFLUENCE = "author_influence"
    TEMPORAL_EVOLUTION = "temporal_evolution"
    PAPER_SIMILARITY = "paper_similarity"
    DOMAIN_COLLABORATION = "domain_collaboration"
    AUTHOR_TOPIC_EVOLUTION = "author_topic_evolution"
    GITHUB_LANDSCAPE = "github_landscape"
    BIBLIOGRAPHIC_COUPLING = "bibliographic_coupling"
    COCITATION = "cocitation"
    MULTIHOP_CITATION = "multihop_citation"
    SHORTEST_CITATION_PATH = "shortest_citation_path"
    PAGERANK = "pagerank"
    COMMUNITY_DETECTION = "community_detection"
    CITATION_PATTERNS = "citation_patterns"
    CONNECTED_COMPONENTS = "connected_components"
    WEIGHTED_SHORTEST_PATH = "weighted_shortest_path"
    BETWEENNESS_CENTRALITY = "betweenness_centrality"
    CLOSENESS_CENTRALITY = "closeness_centrality"
    STRONGLY_CONNECTED_COMPONENTS = "strongly_connected_components"
    TOPOLOGICAL_SORT = "topological_sort"
    LINK_PREDICTION = "link_prediction"
    LOUVAIN_COMMUNITY = "louvain_community"
    DEGREE_CENTRALITY = "degree_centrality"
    EIGENVECTOR_CENTRALITY = "eigenvector_centrality"
    KCORE_DECOMPOSITION = "kcore_decomposition"
    ARTICULATION_POINTS = "articulation_points"
    INFLUENCE_MAXIMIZATION = "influence_maximization"
    HITS = "hits"
    HARMONIC_CENTRALITY = "harmonic_centrality"
    KATZ_CENTRALITY = "katz_centrality"
    ALL_SHORTEST_PATHS = "all_shortest_paths"
    K_SHORTEST_PATHS = "k_shortest_paths"
    RANDOM_WALK = "random_walk"
    TRIANGLE_COUNT = "triangle_count"
    GRAPH_DIAMETER = "graph_diameter"
    LEIDEN_COMMUNITY = "leiden_community"
    BRIDGE_EDGES = "bridge_edges"
    MIN_CUT = "min_cut"
    MINIMUM_SPANNING_TREE = "minimum_spanning_tree"
    NODE_SIMILARITY = "node_similarity"
    BIPARTITE_PROJECTION = "bipartite_projection"
    ADAMIC_ADAR_INDEX = "adamic_adar_index"
    PATTERN_MATCH = "pattern_match"
    PIPELINE = "pipeline"
    SUBGRAPH_PROJECTION = "subgraph_projection"
    TRAVERSE = "traverse"
    GRAPH_UNION = "graph_union"
    GRAPH_INTERSECTION = "graph_intersection"


class PatternNode(BaseModel):
    """A node in a structural pattern."""
    alias: str = Field(max_length=20, description="Node alias (e.g. 'a', 'b', 'c')")
    type: str = Field(default="paper", pattern="^(paper|author|category)$")
    filters: dict[str, Any] = Field(default_factory=dict,
        description="Property filters: categories, primary_category, min_citations, max_citations, has_github, date_from, date_to")


class PatternEdge(BaseModel):
    """An edge in a structural pattern."""
    source: str = Field(max_length=20, description="Source node alias")
    target: str = Field(max_length=20, description="Target node alias")
    relation: str = Field(default="cites", pattern="^(cites|cited_by|co_authored|same_category)$")
    min_hops: int = Field(default=1, ge=1, le=5)
    max_hops: int = Field(default=1, ge=1, le=10)
    optional: bool = Field(default=False, description="If true, the match succeeds even when this edge has no target (OPTIONAL MATCH)")

    @model_validator(mode="after")
    def _check_hops(self) -> "PatternEdge":
        if self.min_hops > self.max_hops:
            raise ValueError(
                f"min_hops ({self.min_hops}) cannot exceed max_hops ({self.max_hops})"
            )
        return self


class WhereCondition(BaseModel):
    """Cross-node predicate for pattern matching (like Cypher WHERE)."""
    left: str = Field(max_length=100, description="Left operand: alias.property (e.g. 'a.citations')")
    op: str = Field(pattern="^(>|<|>=|<=|==|!=|in|not_in|contains)$",
        description="Comparison operator")
    right: str = Field(max_length=200, description="Right operand: alias.property or literal value")


class Aggregation(BaseModel):
    """Aggregation function applied to graph query results (like Cypher RETURN count/avg/sum)."""
    function: str = Field(pattern="^(count|sum|avg|min|max|collect|group_count)$",
        description="Aggregation function")
    field: str | None = Field(default=None, max_length=100,
        description="Node property to aggregate (e.g. 'citations', 'primary_category'). None for count.")
    alias: str = Field(max_length=50, description="Key name in aggregations response")


class PathFilter(BaseModel):
    """Filter paths by length or node properties (path variable binding)."""
    min_path_length: int | None = Field(default=None, ge=0, description="Minimum path hops")
    max_path_length: int | None = Field(default=None, ge=0, description="Maximum path hops")
    all_nodes_match: dict[str, Any] | None = Field(default=None,
        description="Every node on path must match these filters (categories, has_github, etc.)")
    any_node_matches: dict[str, Any] | None = Field(default=None,
        description="At least one intermediate node must match these filters")

    @model_validator(mode="after")
    def _check_path_length(self) -> "PathFilter":
        if (self.min_path_length is not None and self.max_path_length is not None
                and self.min_path_length > self.max_path_length):
            raise ValueError(
                f"min_path_length ({self.min_path_length}) cannot exceed max_path_length ({self.max_path_length})"
            )
        return self


class PipelineStep(BaseModel):
    """A single step in a graph algorithm pipeline."""
    type: str = Field(description="Graph query type to run at this step")
    limit: int = Field(default=50, ge=1, le=10000)
    # Optional overrides for algorithm params
    params: dict[str, Any] = Field(default_factory=dict,
        description="Algorithm-specific params (damping_factor, iterations, etc.)")
    # Filter on previous step output
    filter_property: str | None = Field(default=None, max_length=100,
        description="Property from previous step to filter on")
    filter_min: float | None = Field(default=None,
        description="Min value for filter_property")
    filter_max: float | None = Field(default=None,
        description="Max value for filter_property")


class SubgraphFilter(BaseModel):
    """Defines which nodes/edges to include in a projected subgraph."""
    categories: list[str] | None = Field(default=None, max_length=50, description="Only papers in these categories")
    exclude_categories: list[str] | None = Field(default=None, max_length=50, description="Exclude papers in these categories")
    primary_category: str | None = Field(default=None, description="Only papers with this primary category")
    date_from: str | None = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}", description="Papers submitted after this date (YYYY-MM-DD)")
    date_to: str | None = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}", description="Papers submitted before this date (YYYY-MM-DD)")
    min_citations: int | None = Field(default=None, ge=0)
    max_citations: int | None = Field(default=None, ge=0)
    has_github: bool | None = None
    authors: list[str] | None = Field(default=None, max_length=100, description="Only papers by these authors")
    seed_arxiv_ids: list[str] | None = Field(default=None, max_length=10000, description="Start from these specific papers")
    direction: str = Field(default="both", pattern="^(references|cited_by|both)$",
        description="Which citation edges to include")
    max_nodes: int = Field(default=500, ge=10, le=10000, description="Max nodes in the projected subgraph")


class GraphQuery(BaseModel):
    """Defines a graph-style query that runs on top of the paper index."""

    type: GraphQueryType

    # Co-author network
    seed_author: str | None = Field(default=None, max_length=200,
        description="Seed author for coauthor_network queries")
    depth: int = Field(default=1, ge=1, le=5,
        description="Hop depth for network expansion")

    # Category diversity / interdisciplinary / author bridge
    min_categories: int | None = Field(default=None, ge=2, le=50,
        description="Minimum number of distinct categories")

    # Cross-category flow / author bridge directional filters
    source_categories: list[str] | None = Field(default=None, max_length=50,
        description="Source categories for flow / bridge queries")
    target_categories: list[str] | None = Field(default=None, max_length=50,
        description="Target categories for flow / bridge queries")

    # Rising interdisciplinary
    citation_percentile: float = Field(default=90.0, ge=50.0, le=99.9,
        description="Minimum citation percentile (e.g. 90 = top 10%%)")
    recency_months: int = Field(default=6, ge=1, le=36,
        description="Paper must be younger than this many months")
    citation_window_years: int = Field(default=2, ge=1, le=10,
        description="Window for computing citation percentile")
    min_citing_categories: int = Field(default=3, ge=2, le=10,
        description="Minimum distinct categories among citing papers")

    # Citation traversal
    direction: str = Field(default="references", pattern="^(references|cited_by)$",
        description="Traversal direction: 'references' (what do seed papers cite?) or 'cited_by' (what cites them?)")
    seed_arxiv_id: str | None = Field(default=None, max_length=50,
        description="Specific paper to start traversal from (alternative to using search filters as seed)")
    aggregate_by: str = Field(default="category", pattern="^(category|author|year)$",
        description="How to aggregate the traversed papers")

    # Multi-paper seed (for bibliographic_coupling, cocitation, paper_citation_network)
    seed_arxiv_ids: list[str] | None = Field(default=None, max_length=10000,
        description="Multiple paper IDs as seeds (alternative to search filters)")

    # Temporal queries
    time_interval: str = Field(default="year", pattern="^(month|quarter|year)$",
        description="Time bucket size for temporal queries")

    # Paper similarity
    similarity_threshold: float = Field(default=0.5, ge=0.0, le=1.0,
        description="Minimum cosine similarity for paper_similarity edges")

    # Multi-hop traversal
    max_hops: int = Field(default=2, ge=1, le=50,
        description="Maximum number of hops for multihop_citation traversal")

    # Shortest path
    target_arxiv_id: str | None = Field(default=None, max_length=50,
        description="Destination paper for shortest_citation_path")

    # PageRank
    damping_factor: float = Field(default=0.85, ge=0.1, le=0.99,
        description="PageRank damping factor (default 0.85)")
    iterations: int = Field(default=20, ge=1, le=500,
        description="PageRank iterations")

    # Citation patterns
    pattern: str = Field(default="mutual", pattern="^(mutual|triangle|star|chain)$",
        description="Citation pattern to find: mutual (A↔B), triangle (A→B→C→A), star (hub citing many), chain (A→B→C→D)")

    # Weighted shortest path
    weight_field: str = Field(default="citations", pattern="^(citations|uniform)$",
        description="Edge weight for weighted shortest path: 'citations' (lower cost for highly-cited) or 'uniform' (all edges weight 1)")

    # Link prediction
    prediction_method: str = Field(default="common_neighbors", pattern="^(common_neighbors|jaccard|adamic_adar|preferential_attachment)$",
        description="Link prediction method: common_neighbors, jaccard, adamic_adar, preferential_attachment")

    # Influence maximization
    influence_seeds: int = Field(default=5, ge=1, le=200,
        description="Number of seed papers to select for influence maximization")

    # Degree centrality
    degree_mode: str = Field(default="total", pattern="^(in|out|total)$",
        description="Degree mode: in (cited_by), out (references), total (both)")

    # K-shortest paths
    k_paths: int = Field(default=3, ge=1, le=20,
        description="Number of shortest paths to find (Yen's algorithm)")

    # Random walk
    walk_length: int = Field(default=10, ge=1, le=100,
        description="Number of steps in random walk")
    num_walks: int = Field(default=100, ge=1, le=1000,
        description="Number of random walks to simulate")
    teleport_prob: float = Field(default=0.15, ge=0.0, le=1.0,
        description="Probability of teleporting back to start (personalized PageRank)")

    # Node similarity
    similarity_method: str = Field(default="jaccard", pattern="^(jaccard|overlap|cosine)$",
        description="Node similarity method: jaccard (|A∩B|/|A∪B|), overlap (|A∩B|/min(|A|,|B|)), cosine (|A∩B|/√(|A|·|B|))")

    # Bipartite projection
    projection_side: str = Field(default="papers", pattern="^(papers|categories|authors)$",
        description="Which side to project onto: papers, categories, or authors")

    # ── Pattern matching ──
    pattern_nodes: list[PatternNode] | None = Field(default=None,
        description="Nodes in the structural pattern to match")
    pattern_edges: list[PatternEdge] | None = Field(default=None,
        description="Edges in the structural pattern to match")
    where: list[WhereCondition] | None = Field(default=None,
        description="Cross-node predicates for pattern matching (like Cypher WHERE)")
    max_expansion: int = Field(default=5, ge=1, le=50,
        description="Max BFS expansion rounds for pattern_match neighbor discovery")

    # ── Pipeline composition ──
    pipeline_steps: list[PipelineStep] | None = Field(default=None,
        description="Ordered list of graph algorithm steps to chain")

    # ── Subgraph projection ──
    subgraph_filter: SubgraphFilter | None = Field(default=None,
        description="Define the subgraph to project before running the algorithm")
    subgraph_algorithm: str | None = Field(default=None,
        description="Algorithm to run on the projected subgraph (any existing graph type)")
    subgraph_params: dict[str, Any] = Field(default_factory=dict,
        description="Params for the algorithm running on the projected subgraph")

    # ── General traversal ──
    traverse_direction: str = Field(default="outgoing", pattern="^(outgoing|incoming|both)$",
        description="Traversal direction: outgoing (references), incoming (citations), both")
    traverse_predicate: dict[str, Any] = Field(default_factory=dict,
        description="Filter predicate for nodes during traversal: {min_citations, max_citations, categories, primary_category, has_github, date_from, date_to}")
    traverse_until: dict[str, Any] = Field(default_factory=dict,
        description="Stop condition: {max_nodes, max_depth, min_citations, category}")
    collect_edges: bool = Field(default=True,
        description="Whether to include edges in traversal result")

    # ── Graph set operations ──
    set_queries: list[dict[str, Any]] = Field(default_factory=list,
        description="Two sub-queries whose results to union/intersect. Each dict is a full graph query.")

    # ── Aggregation ──
    aggregations: list[Aggregation] | None = Field(default=None,
        description="Aggregation functions to compute over results (like Cypher RETURN count/avg/sum)")

    # ── Path filtering ──
    path_filter: PathFilter | None = Field(default=None,
        description="Filter paths by length or node properties (path variable binding)")

    # Pagination
    limit: int = Field(default=50, ge=1, le=10000)


class GraphNode(BaseModel):
    id: str
    label: str
    type: str  # "paper", "author", "category"
    properties: dict[str, Any] = Field(default_factory=dict)


class GraphEdge(BaseModel):
    source: str
    target: str
    relation: str  # "co_authored", "in_category", "publishes_in", "co_occurs"
    weight: float | None = None


class GraphResponse(BaseModel):
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    total: int
    took_ms: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class GraphSearchRequest(BaseModel):
    """Combined request: graph query + optional search filters."""

    graph: GraphQuery

    # All standard search filters (optional — applied as graph context)
    query: str | None = Field(default=None, max_length=2000)
    title_query: str | None = Field(default=None, max_length=500)
    abstract_query: str | None = Field(default=None, max_length=2000)
    semantic: SemanticQuery | list[SemanticQuery] | None = None

    @field_validator("semantic", mode="before")
    @classmethod
    def _normalize_semantic(cls, v: Any) -> list[SemanticQuery] | None:
        if v is None:
            return None
        if isinstance(v, dict):
            return [v]
        if isinstance(v, SemanticQuery):
            return [v]
        if isinstance(v, list):
            return v
        return v

    fuzzy: str | None = Field(default=None, max_length=500)
    fuzzy_fuzziness: int = Field(default=2, ge=0, le=3)
    title_regex: str | None = Field(default=None, max_length=200)
    abstract_regex: str | None = Field(default=None, max_length=200)
    author_regex: str | None = Field(default=None, max_length=200)
    author: str | None = None
    first_author: str | None = None
    min_h_index: int | None = Field(default=None, ge=0)
    max_h_index: int | None = Field(default=None, ge=0)
    min_first_author_h_index: int | None = Field(default=None, ge=0)
    min_median_h_index_citing: float | None = Field(default=None, ge=0)
    min_citations: int | None = Field(default=None, ge=0)
    max_citations: int | None = Field(default=None, ge=0)
    min_references: int | None = Field(default=None, ge=0)
    categories: list[str] | None = Field(default=None, max_length=50)
    primary_category: str | None = None
    exclude_categories: list[str] | None = Field(default=None, max_length=50)
    submitted_date: DateRange | None = None
    updated_date: DateRange | None = None
    has_github: bool | None = None
    min_page_count: int | None = Field(default=None, ge=0)
    max_page_count: int | None = Field(default=None, ge=0)
    has_doi: bool | None = None
    has_journal_ref: bool | None = None
    minimum_should_match: str | None = Field(
        default=None,
        pattern=r"^\d+%?$",
        description="ES minimum_should_match spec, e.g. '2', '75%'"
    )
    operator: str = Field(default="or", pattern="^(and|or)$")

    @field_validator("title_regex", "abstract_regex", "author_regex")
    @classmethod
    def validate_regex_safety(cls, v: str | None) -> str | None:
        if v is None:
            return v
        import re
        dangerous = [
            r"(.+)+",
            r"(.*)*",
            r"([a-zA-Z]+)*",
            r"(a|a)+",
        ]
        for pat in dangerous:
            if pat in v:
                raise ValueError(f"Potentially dangerous regex pattern blocked: {pat}")
        import re as _re
        if _re.search(r'\([^)]*[+*][^)]*\)[+*]', v):
            raise ValueError("Potentially dangerous regex pattern (nested quantifiers)")
        try:
            re.compile(v)
        except re.error as exc:
            raise ValueError(f"Invalid regex: {exc}") from exc
        return v

    def to_search_request(self) -> SearchRequest:
        """Extract the search filter portion as a SearchRequest."""
        data = self.model_dump(exclude={"graph"}, exclude_none=True)
        return SearchRequest(**data)
