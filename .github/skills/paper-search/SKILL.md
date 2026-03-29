---
name: paper-search
description: "Search the ArXiv paper database (~3M papers). Use when: finding papers by author/topic/category/date, looking up specific papers, exploring research trends, filtering by citations or GitHub availability, fuzzy/regex search, combining multiple filters, graph queries (55 types: co-authorship, citation networks, author influence, temporal evolution, paper similarity, domain collaboration, topic evolution, GitHub landscape, bibliographic coupling, co-citation, interdisciplinary detection, category flows, author bridges, multihop citation, shortest citation path, PageRank, community detection, citation patterns, connected components, weighted shortest path, betweenness centrality, closeness centrality, strongly connected components, topological sort, link prediction, Louvain community detection, degree centrality, eigenvector centrality, k-core decomposition, articulation points, influence maximization, HITS, harmonic centrality, Katz centrality, all shortest paths, k-shortest paths, random walk, triangle count, graph diameter, Leiden community, bridge edges, min-cut, minimum spanning tree, node similarity, bipartite projection, Adamic-Adar index, pattern matching, pipeline composition, subgraph projection, general traversal, graph union, graph intersection). Invoked by /paper-search."
---

# PaperPilot — ArXiv Search Skill

> **If the API is unreachable**, run `bash scripts/start_tunnel.sh` on the server to restart the tunnel.

Search a live Elasticsearch index of ~2.99 million ArXiv papers (all categories, 2005–present, updated daily).

## API Endpoints

**Base**: `https://arxiv-paperpilot.serveousercontent.com`  
**Auth**: `X-API-Key: changeme-key-1`

| Method | Endpoint | Auth | Purpose |
|--------|----------|------|---------|
| POST | `/search` | Yes | Full-featured search |
| POST | `/graph` | Yes | Graph queries (55 types: citation networks, co-authorship, similarity, temporal, influence, domain collaboration, graph-DB algorithms, pattern matching, pipeline composition, subgraph projection, general traversal, graph set operations) |
| GET | `/stats` | Yes | Database statistics |
| GET | `/paper/{arxiv_id}` | Yes | Single paper lookup |
| GET | `/health` | No | Health check |

## Search Request Schema

POST `/search` with JSON body. All fields are optional — omit any you don't need.

```jsonc
{
  // ── Text search (OR by default) ──
  "query": "string",              // full-text across title+abstract+authors
  "title_query": "string",        // title only
  "abstract_query": "string",     // abstract only
  "operator": "and" | "or",       // default: "or"
  "minimum_should_match": "3" | "75%",  // how many terms must match

  // ── Fuzzy search (typo-tolerant) ──
  "fuzzy": "string",              // fuzzy full-text
  "fuzzy_fuzziness": 0-3,         // edit distance (default: 2)

  // ── Regex search ──
  "title_regex": ".*[Tt]ransformer.*",
  "abstract_regex": ".*pattern.*",
  "author_regex": ".*pattern.*",

  // ── Author filters ──
  "author": "name",               // nested match on any author
  "first_author": "name",         // match first author only
  "min_h_index": 0,
  "max_h_index": 100,
  "min_first_author_h_index": 0,

  // ── Citation filters ──
  "min_citations": 0,
  "max_citations": 1000,
  "min_references": 0,

  // ── Category filters ──
  "categories": ["cs.AI", "cs.LG"],        // include (OR)
  "primary_category": "cs.AI",              // exact primary
  "exclude_categories": ["cs.CV", "cs.CL"], // exclude

  // ── Date filters ──
  "submitted_date": {"gte": "2025-01-01T00:00:00+00:00", "lte": "2025-12-31T23:59:59+00:00"},
  "updated_date":   {"gte": "2025-01-01T00:00:00+00:00"},

  // ── Metadata filters ──
  "has_github": true | false,
  "min_page_count": 10,
  "max_page_count": 50,
  "has_doi": true,
  "has_journal_ref": true,

  // ── Sorting ──
  "sort_by": "relevance" | "date" | "citations" | "h_index" | "page_count" | "updated",
  "sort_order": "asc" | "desc",

  // ── Semantic similarity ──
  // Single object OR array of objects. Each has:
  //   text (string), level ("title"|"abstract"), weight (0-10), mode ("boost"|"exclude")
  // "boost" (default): rank similar papers higher
  // "exclude": penalize papers similar to the text (continuous — proportional to similarity)
  "semantic": {"text": "topic description", "level": "abstract", "weight": 1.0},
  // or multiple:
  "semantic": [
    {"text": "boost this topic", "level": "abstract", "weight": 1.0, "mode": "boost"},
    {"text": "exclude this topic", "level": "abstract", "weight": 0.5, "mode": "exclude"}
  ],

  // ── Pagination ──
  "offset": 0,       // 0-9999
  "limit": 20,       // 1-200

  // ── Highlights ──
  "highlight": true   // returns <em>matched</em> fragments
}
```

## Complex Query Examples

### 1. Find recent transformer papers in NLP with code on GitHub

```bash
curl -s https://arxiv-paperpilot.serveousercontent.com/search \
  -H "X-API-Key: changeme-key-1" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "transformer attention mechanism",
    "operator": "and",
    "categories": ["cs.CL", "cs.AI"],
    "submitted_date": {"gte": "2025-01-01T00:00:00+00:00"},
    "has_github": true,
    "sort_by": "date",
    "sort_order": "desc",
    "limit": 10,
    "highlight": true
  }'
```

### 2. Author search combined with topic and date

```bash
curl -s https://arxiv-paperpilot.serveousercontent.com/search \
  -H "X-API-Key: changeme-key-1" \
  -H "Content-Type: application/json" \
  -d '{
    "author": "Yann LeCun",
    "query": "self-supervised learning",
    "submitted_date": {"gte": "2023-01-01T00:00:00+00:00"},
    "sort_by": "date",
    "sort_order": "desc"
  }'
```

### 3. First-author search for specific researcher

```bash
curl -s https://arxiv-paperpilot.serveousercontent.com/search \
  -H "X-API-Key: changeme-key-1" \
  -H "Content-Type: application/json" \
  -d '{
    "first_author": "Vaswani",
    "categories": ["cs.CL"],
    "sort_by": "date",
    "sort_order": "desc"
  }'
```

### 4. Fuzzy search (handles typos)

```bash
curl -s https://arxiv-paperpilot.serveousercontent.com/search \
  -H "X-API-Key: changeme-key-1" \
  -H "Content-Type: application/json" \
  -d '{
    "fuzzy": "reinformcent lerning",
    "fuzzy_fuzziness": 2,
    "categories": ["cs.AI", "cs.LG"],
    "has_github": true,
    "limit": 20
  }'
```

### 5. Regex title search

```bash
curl -s https://arxiv-paperpilot.serveousercontent.com/search \
  -H "X-API-Key: changeme-key-1" \
  -H "Content-Type: application/json" \
  -d '{
    "title_regex": ".*[Ll]arge [Ll]anguage [Mm]odel.*",
    "submitted_date": {"gte": "2024-01-01T00:00:00+00:00"},
    "sort_by": "date",
    "sort_order": "desc",
    "limit": 50
  }'
```

### 6. Survey finder (long papers with "survey" in title)

```bash
curl -s https://arxiv-paperpilot.serveousercontent.com/search \
  -H "X-API-Key: changeme-key-1" \
  -H "Content-Type: application/json" \
  -d '{
    "title_query": "survey",
    "abstract_query": "comprehensive overview",
    "min_page_count": 20,
    "submitted_date": {"gte": "2024-01-01T00:00:00+00:00"},
    "sort_by": "date",
    "sort_order": "desc",
    "limit": 20
  }'
```

### 7. Cross-category search with exclusions

```bash
curl -s https://arxiv-paperpilot.serveousercontent.com/search \
  -H "X-API-Key: changeme-key-1" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "optimization convergence",
    "categories": ["cs.LG", "stat.ML", "math.OC"],
    "exclude_categories": ["cs.CV"],
    "operator": "and",
    "sort_by": "relevance",
    "limit": 30
  }'
```

### 8. Maximum filter combo (query + fuzzy + author + category + date + github + pages + sort)

```bash
curl -s https://arxiv-paperpilot.serveousercontent.com/search \
  -H "X-API-Key: changeme-key-1" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "graph neural network",
    "fuzzy": "molecluar property predction",
    "fuzzy_fuzziness": 2,
    "author": "Kipf",
    "categories": ["cs.LG", "cs.AI"],
    "submitted_date": {"gte": "2023-01-01T00:00:00+00:00"},
    "has_github": true,
    "min_page_count": 8,
    "sort_by": "date",
    "sort_order": "desc",
    "highlight": true,
    "limit": 10
  }'
```

### 9. Browse by category and date (no text query)

```bash
curl -s https://arxiv-paperpilot.serveousercontent.com/search \
  -H "X-API-Key: changeme-key-1" \
  -H "Content-Type: application/json" \
  -d '{
    "primary_category": "quant-ph",
    "submitted_date": {
      "gte": "2026-03-01T00:00:00+00:00",
      "lte": "2026-03-25T23:59:59+00:00"
    },
    "sort_by": "date",
    "sort_order": "desc",
    "limit": 50
  }'
```

### 10. Published/peer-reviewed papers only

```bash
curl -s https://arxiv-paperpilot.serveousercontent.com/search \
  -H "X-API-Key: changeme-key-1" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "protein structure prediction",
    "has_doi": true,
    "has_journal_ref": true,
    "sort_by": "citations",
    "sort_order": "desc",
    "limit": 20
  }'
```

### 11. Strict phrase matching with minimum_should_match

```bash
curl -s https://arxiv-paperpilot.serveousercontent.com/search \
  -H "X-API-Key: changeme-key-1" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "federated learning differential privacy heterogeneous",
    "minimum_should_match": "4",
    "categories": ["cs.CR", "cs.LG"],
    "limit": 20
  }'
```

### 12. Paginating through results

```bash
# Page 1
curl -s https://arxiv-paperpilot.serveousercontent.com/search \
  -H "X-API-Key: changeme-key-1" \
  -H "Content-Type: application/json" \
  -d '{"categories": ["cs.AI"], "sort_by": "date", "sort_order": "desc", "offset": 0, "limit": 50}'

# Page 2
curl -s https://arxiv-paperpilot.serveousercontent.com/search \
  -H "X-API-Key: changeme-key-1" \
  -H "Content-Type: application/json" \
  -d '{"categories": ["cs.AI"], "sort_by": "date", "sort_order": "desc", "offset": 50, "limit": 50}'
```

### 13. Single paper lookup

```bash
# New-format ID
curl -s https://arxiv-paperpilot.serveousercontent.com/paper/2301.00001 -H "X-API-Key: changeme-key-1"

# Old-format ID
curl -s https://arxiv-paperpilot.serveousercontent.com/paper/hep-th/0508186 -H "X-API-Key: changeme-key-1"
```

### 14. Database statistics

```bash
curl -s https://arxiv-paperpilot.serveousercontent.com/stats -H "X-API-Key: changeme-key-1"
# Returns: total_papers, categories (with counts), date_range, papers_with_github, avg_pages, avg_citations
```

### 15. Aggregation queries via the API

Use `/search` with targeted filters and pagination to build aggregations. The `/stats` endpoint already provides top categories with counts.

```bash
# Top categories by paper count (use /stats)
curl -s https://arxiv-paperpilot.serveousercontent.com/stats -H "X-API-Key: changeme-key-1"

# Monthly paper count for cs.AI in 2025 — paginate by month
curl -s https://arxiv-paperpilot.serveousercontent.com/search -H "X-API-Key: changeme-key-1" -H "Content-Type: application/json" -d '{
  "primary_category": "cs.AI",
  "submitted_date": {"gte": "2025-01-01T00:00:00+00:00", "lte": "2025-01-31T23:59:59+00:00"},
  "limit": 1
}'
# → use "total" from the response; repeat for each month

# Recent papers per category
curl -s https://arxiv-paperpilot.serveousercontent.com/search -H "X-API-Key: changeme-key-1" -H "Content-Type: application/json" -d '{
  "primary_category": "cs.LG",
  "submitted_date": {"gte": "2025-01-01T00:00:00+00:00"},
  "limit": 1
}'
# → "total" gives the count for that category+date range
```

## Response Format

```jsonc
{
  "total": 42567,         // total matching papers
  "hits": [
    {
      "arxiv_id": "2401.12345",
      "title": "Paper Title",
      "abstract": "Full abstract text...",
      "authors": [{"name": "Alice Smith", "is_first_author": true}],
      "categories": ["cs.AI", "cs.LG"],
      "primary_category": "cs.AI",
      "submitted_date": "2024-01-15T00:00:00+00:00",
      "has_github": true,
      "page_count": 12,
      "score": 15.7,
      "highlights": {
        "title": ["<em>Transformer</em> Architecture..."],
        "abstract": ["...using <em>attention</em>..."]
      }
    }
  ],
  "took_ms": 42,
  "offset": 0,
  "limit": 20
}
```

## Graph Query Schema

POST `/graph` with JSON body. The `graph` field is required; all other fields are standard search filters applied as context.

```jsonc
{
  "graph": {
    // ── Query type (required) ──
    "type": "category_diversity"       // papers spanning many subcategories
          | "coauthor_network"         // collaboration graph around an author
          | "author_bridge"            // authors publishing across disjoint fields
          | "cross_category_flow"      // category co-occurrence graph
          | "interdisciplinary"        // papers with rare category combos (scored)
          | "rising_interdisciplinary" // recent breakout papers cited across fields
          | "citation_traversal"       // follow refs/citers, aggregate by category/author/year
          | "paper_citation_network"   // direct paper→paper citation graph
          | "author_influence"         // author→author influence via citations
          | "temporal_evolution"       // category publication volume over time
          | "paper_similarity"         // semantic similarity network (embeddings)
          | "domain_collaboration"     // domain-level co-occurrence (cs, math, physics…)
          | "author_topic_evolution"   // how an author's topics shift over time
          | "github_landscape"         // code-availability patterns by category/time
          | "bibliographic_coupling"   // papers sharing many references
          | "cocitation"               // papers frequently cited together
          | "pattern_match"           // declarative structural pattern matching (like Cypher MATCH)
          | "pipeline"                // chain multiple graph algorithms sequentially
          | "subgraph_projection"    // define precise subgraph, then run any algorithm on it
          | "traverse"               // general BFS traversal with predicates and stop conditions
          | "graph_union"            // union of two sub-query results
          | "graph_intersection",    // intersection of two sub-query results

    // ── Common parameters ──
    "seed_author": "Author Name",       // for coauthor_network, author_influence, author_topic_evolution
    "depth": 1,                          // coauthor_network depth (1-5)
    "min_categories": 3,                 // for category_diversity, interdisciplinary, author_bridge
    "source_categories": ["cs.AI"],      // for cross_category_flow, author_bridge, domain_collaboration
    "target_categories": ["q-bio.NC"],   // for cross_category_flow, author_bridge, domain_collaboration
    "direction": "references"|"cited_by", // for citation_traversal, paper_citation_network, author_influence
    "aggregate_by": "category"|"author"|"year", // for citation_traversal

    // ── Seed papers ──
    "seed_arxiv_id": "2301.00001",       // single paper seed (citation_traversal, paper_citation_network, bibliographic_coupling, cocitation)
    "seed_arxiv_ids": ["2301.00001","2301.00002"], // multiple paper seeds

    // ── Rising interdisciplinary ──
    "citation_percentile": 90.0,         // top 10% = 90 (default)
    "recency_months": 6,                 // paper must be < 6 months old
    "citation_window_years": 2,          // percentile computed over last 2 years
    "min_citing_categories": 3,          // min distinct fields among citers

    // ── Temporal queries ──
    "time_interval": "year"|"quarter"|"month", // for temporal_evolution, author_topic_evolution, github_landscape

    // ── Similarity ──
    "similarity_threshold": 0.5,         // min cosine similarity for paper_similarity edges (0.0-1.0)

    // ── Graph-DB algorithms (types 17-28) ──
    "max_hops": 3,                       // max traversal depth (multihop_citation, shortest_citation_path, weighted_shortest_path)
    "target_arxiv_id": "2405.04233",    // destination for shortest-path algorithms
    "damping_factor": 0.85,              // PageRank damping (0.1-0.99)
    "iterations": 20,                    // PageRank / community detection iterations (1-500)
    "pattern": "mutual",                 // citation_patterns: mutual|star|chain|triangle
    "weight_field": "citations",        // weighted_shortest_path: "citations" or "uniform"
    "prediction_method": "common_neighbors", // link_prediction: common_neighbors|jaccard|adamic_adar|preferential_attachment
    "influence_seeds": 5,                // influence_maximization: number of seed papers to select (1-200)
    "degree_mode": "total",             // degree_centrality: "in" | "out" | "total"

    // ── Graph-DB algorithms (types 35-49) ──
    "k_paths": 3,                        // k_shortest_paths: number of paths to find (1-20)
    "walk_length": 10,                   // random_walk: steps per walk (1-100)
    "num_walks": 100,                    // random_walk: number of walks (1-1000)
    "teleport_prob": 0.15,              // random_walk: restart probability (0.0-1.0)
    "similarity_method": "jaccard",      // node_similarity: jaccard|overlap|cosine
    "projection_side": "papers",         // bipartite_projection: papers|categories|authors

    // ── Pattern matching (type 50) ──
    "pattern_nodes": [                    // declare pattern nodes with filters
      {"alias": "a", "type": "paper", "filters": {"categories": ["cs.CL"], "date_from": "2024-01-01"}},
      {"alias": "b", "type": "paper", "filters": {"categories": ["cs.LG"]}}
    ],
    "pattern_edges": [                    // declare pattern edges with relations
      {"source": "a", "target": "b", "relation": "cites"}  // cites|cited_by|co_authored|same_category
    ],

    // ── Pipeline composition (type 51) ──
    "pipeline_steps": [                   // ordered list of algorithm steps (2-5)
      {"type": "pagerank", "limit": 50, "params": {"damping_factor": 0.85}},
      {"type": "community_detection", "limit": 20,
       "filter_property": "pagerank", "filter_min": 0.001}  // optional inter-step filter
    ],

    // ── Subgraph projection (type 52) ──
    "subgraph_filter": {                  // define the exact subgraph
      "categories": ["cs.AI"],
      "exclude_categories": ["cs.CV"],
      "date_from": "2024-01-01",
      "date_to": "2025-01-01",
      "min_citations": 10,
      "max_citations": 1000,
      "has_github": true,
      "authors": ["Author Name"],
      "seed_arxiv_ids": ["2301.00001"],
      "direction": "both",               // references|cited_by|both
      "max_nodes": 500                    // 10-2000
    },
    "subgraph_algorithm": "pagerank",     // any of the 49 base graph algorithms
    "subgraph_params": {"damping_factor": 0.85},  // params for the target algorithm

    // ── General traversal (type 53) ──
    "traverse_direction": "outgoing",     // outgoing (references), incoming (citations), both
    "traverse_predicate": {               // filter nodes during expansion
      "min_citations": 10,
      "categories": ["cs.AI"],
      "has_github": true,
      "date_from": "2024-01-01",
      "date_to": "2025-01-01"
    },
    "traverse_until": {                   // stop conditions
      "max_nodes": 100,                   // max nodes to collect (default: limit*3)
      "max_depth": 5,                     // max BFS depth (default: max_hops)
      "category": "cs.CL",               // stop when reaching this category
      "min_citations": 50                 // stop when reaching a paper with >= N citations
    },
    "collect_edges": true,                // include edges in result (default: true)

    // ── Graph set operations (types 54-55) ──
    "set_queries": [                      // two sub-queries to union/intersect
      {"type": "pagerank", "limit": 50},
      {"type": "community_detection", "limit": 50}
    ],

    "limit": 50                          // max results (1-200)
  },

  // ── ALL 30 search filters compose with every graph type ──
  // Text search
  "query": "transformer attention",
  "title_query": "large language model",
  "abstract_query": "protein folding",
  "operator": "and" | "or",
  "minimum_should_match": "75%",
  // Fuzzy & regex
  "fuzzy": "tansformer",
  "fuzzy_fuzziness": 2,
  "title_regex": ".*[Tt]ransformer.*",
  "abstract_regex": ".*pattern.*",
  "author_regex": ".*LeCun.*",
  // Author filters
  "author": "Yann LeCun",
  "first_author": "Yann LeCun",
  // H-index filters
  "min_h_index": 10,
  "max_h_index": 100,
  "min_first_author_h_index": 20,
  "min_median_h_index_citing": 15.0,
  // Citation & reference filters
  "min_citations": 50,
  "max_citations": 1000,
  "min_references": 10,
  // Category filters
  "categories": ["cs.LG", "cs.AI"],
  "primary_category": "cs.LG",
  "exclude_categories": ["cs.CV"],
  // Date range filters
  "submitted_date": {"gte": "2024-01-01T00:00:00"},
  "updated_date": {"gte": "2024-01-01T00:00:00", "lte": "2024-12-31T00:00:00"},
  // Boolean filters
  "has_github": true,
  "has_doi": true,
  "has_journal_ref": true,
  // Page count filters
  "min_page_count": 5,
  "max_page_count": 50,
  // Semantic similarity (boost/exclude — works with all graph types)
  "semantic": [
    {"text": "neural architecture search", "level": "abstract", "weight": 1.0, "mode": "boost"},
    {"text": "computer vision", "level": "abstract", "weight": 0.5, "mode": "exclude"}
  ]
}
```

### Graph Response Format

```jsonc
{
  "nodes": [
    {"id": "2411.00813", "label": "Paper Title", "type": "paper",
     "properties": {"categories": [...], "interdisciplinary_score": 0.95, ...}},
    {"id": "cs.AI", "label": "cs.AI", "type": "category", "properties": {...}},
    {"id": "Author Name", "label": "Author Name", "type": "author", "properties": {...}},
    {"id": "cs", "label": "cs", "type": "domain", "properties": {"paper_count": 899338}},
    {"id": "2024-01-01T00:00:00.000Z", "label": "2024-01-01T00:00:00.000Z", "type": "time", "properties": {...}}
  ],
  "edges": [
    {"source": "2411.00813", "target": "cs.AI", "relation": "in_category", "weight": null},
    {"source": "Author A", "target": "Author B", "relation": "co_authored", "weight": 5},
    {"source": "paper1", "target": "paper2", "relation": "cites|similar|shared_references|co_cited", "weight": 3}
  ],
  "total": 42,
  "took_ms": 150,
  "metadata": {"min_categories": 3, "papers_scored": 42}
}
```

### Graph Query Types Explained

| Type | What it does | Key params | Nodes | Edges |
|------|-------------|------------|-------|-------|
| `category_diversity` | Papers tagged in many subcategories | `min_categories` | papers + categories | paper→category |
| `coauthor_network` | Collaboration ego-graph around an author | `seed_author`, `depth` | authors | author↔author (weighted by co-papers) |
| `author_bridge` | Authors publishing across disjoint fields | `min_categories`, `source_categories`, `target_categories` | authors + categories | author→category |
| `cross_category_flow` | How categories co-occur on papers | `source_categories`, `target_categories` | categories | category↔category (weighted) |
| `interdisciplinary` | Papers with unusually rare category combos | `min_categories` | papers + categories | paper→category (with rarity score) |
| `rising_interdisciplinary` | Recent top-cited papers with diverse citing fields | `recency_months`, `citation_percentile`, `citation_window_years`, `min_citing_categories` | papers + categories | paper→category + category→paper (citing flow) |
| `citation_traversal` | Follow references/citers of seed papers and aggregate the linked set | `direction`, `aggregate_by`, `seed_arxiv_id` | papers (seeds) + aggregated nodes | seed→aggregate (weighted) |
| `paper_citation_network` | Direct paper→paper citation graph (not aggregated) | `direction`, `seed_arxiv_id`/`seed_arxiv_ids` | papers (seed + linked) | paper→paper (cites) |
| `author_influence` | Author→author influence via citation paths | `seed_author`, `direction` | authors | author→author (weighted by papers) |
| `temporal_evolution` | Category publication volume over time | `time_interval` | categories + time periods | category→time (weighted by count) |
| `paper_similarity` | Semantic similarity network using embeddings | `similarity_threshold` (requires `semantic` boost) | papers | paper↔paper (weighted by cosine sim) |
| `domain_collaboration` | Domain-level (cs, math, physics…) co-occurrence | `source_categories`, `target_categories` | domains | domain↔domain (weighted) |
| `author_topic_evolution` | How an author's topics shift over time | `seed_author`, `time_interval` | author + time + categories | author→time→category |
| `github_landscape` | Code-availability patterns by category/domain/time | `time_interval` | categories + domains + time | category→domain (weighted) |
| `bibliographic_coupling` | Papers sharing many references | `seed_arxiv_id`/`seed_arxiv_ids` | papers | paper↔paper (weighted by shared refs) |
| `cocitation` | Papers frequently cited together by other papers | `seed_arxiv_id`/`seed_arxiv_ids` | papers | paper↔paper (weighted by co-citers) |
| `multihop_citation` | Multi-hop traversal of citation graph from seed paper | `seed_arxiv_id`, `max_hops`, `direction` | papers | paper→paper (cites) |
| `shortest_citation_path` | BFS shortest path between two papers | `seed_arxiv_id`, `target_arxiv_id`, `max_hops` | papers on path | paper→paper (cites) |
| `pagerank` | PageRank centrality on citation subgraph | `damping_factor`, `iterations` | papers | paper→paper (cites) |
| `community_detection` | Label Propagation community detection on citation/co-authorship graph | `iterations` | communities + papers | community→paper (contains) |
| `citation_patterns` | Detect structural motifs in citation graph | `pattern` (mutual\|star\|chain\|triangle) | papers | paper→paper (pattern edges) |
| `connected_components` | Find connected components in the citation graph | — | components + papers | component→paper (contains) |
| `weighted_shortest_path` | Dijkstra weighted shortest path between two papers | `seed_arxiv_id`, `target_arxiv_id`, `max_hops`, `weight_field` | papers on path | paper→paper (weighted) |
| `betweenness_centrality` | Brandes' betweenness centrality on citation subgraph | `limit` | top-N papers by betweenness | paper→paper (cites) |
| `closeness_centrality` | BFS-based closeness centrality with Wasserman-Faust normalization | `limit` | top-N papers by closeness | paper→paper (cites) |
| `strongly_connected_components` | Tarjan's SCC detection on directed citation graph | `limit` | SCC clusters + papers | SCC→paper (contains) + inter-SCC |
| `topological_sort` | Kahn's topological ordering of citation DAG | `limit` | papers in topo order | paper→paper (cites) |
| `link_prediction` | Predict likely future citations using graph heuristics | `prediction_method`, `limit` | papers | paper→paper (predicted_link with score) |
| `louvain_community` | Louvain modularity-optimization community detection | `iterations`, `limit` | communities + papers | community→paper (contains) + inter-community |
| `degree_centrality` | In/out/total degree ranking of papers | `degree_mode`, `limit` | top-N papers by degree | paper→paper (cites) |
| `eigenvector_centrality` | Power-iteration eigenvector centrality (recursive prestige) | `iterations`, `limit` | top-N papers by eigenvector | paper→paper (cites) |
| `kcore_decomposition` | K-core peeling to find densest subgraph | `limit` | papers labeled with coreness | paper→paper (cites) |
| `articulation_points` | Cut vertices whose removal disconnects the citation graph | `limit` | articulation points + bridged neighbors | AP→neighbor (bridges) |
| `influence_maximization` | Greedy influence spread (Independent Cascade model) | `influence_seeds`, `limit` | seed papers + influenced papers | cites + influences |
| `hits` | HITS algorithm — hub & authority scores (Kleinberg) | `iterations`, `limit` | papers with hub/authority scores | paper→paper (cites) |
| `harmonic_centrality` | Harmonic centrality (handles disconnected graphs, sum of inverse distances) | `limit` | top-N papers by harmonic centrality | paper→paper (cites) |
| `katz_centrality` | Katz centrality (counts all paths weighted by attenuation) | `damping_factor`, `iterations`, `limit` | top-N papers by Katz score | paper→paper (cites) |
| `all_shortest_paths` | Find ALL shortest paths between two papers (not just one) | `seed_arxiv_id`, `target_arxiv_id`, `max_hops` | papers on paths | paper→paper (shortest_path) |
| `k_shortest_paths` | Yen's K-shortest simple paths between two papers | `seed_arxiv_id`, `target_arxiv_id`, `k_paths`, `max_hops` | papers on K paths | paper→paper (path_step) |
| `random_walk` | Random walk with restart (Personalized PageRank) from seed paper | `seed_arxiv_id`, `walk_length`, `num_walks`, `teleport_prob`, `limit` | papers ranked by visit frequency | walk_transition (weighted) |
| `triangle_count` | Count triangles and local/global clustering coefficients | `limit` | papers with triangle count + CC | paper→paper (cites) |
| `graph_diameter` | Graph diameter, radius, eccentricity, center nodes | `limit` | papers with eccentricity | paper→paper (cites) |
| `leiden_community` | Leiden algorithm (improved Louvain with refinement) | `iterations`, `limit` | communities + papers | community→paper + inter-community |
| `bridge_edges` | Bridge edges whose removal disconnects the graph (Tarjan's) | `limit` | bridge endpoints | bridge (weighted) |
| `min_cut` | Minimum cut / max-flow between two papers (Edmonds-Karp) | `seed_arxiv_id`, `target_arxiv_id` | source/target side papers + cut edges | min_cut_edge |
| `minimum_spanning_tree` | Kruskal's MST on citation subgraph (weight = 1/(1+shared)) | `limit` | MST edge endpoints | mst_edge (weighted) |
| `node_similarity` | Structural node similarity (Jaccard/Overlap/Cosine on neighborhoods) | `similarity_method`, `limit` | paper pairs by similarity | similar (weighted) |
| `bipartite_projection` | Bipartite paper↔category/author graph projected onto one side | `projection_side`, `limit` | projected nodes | co_occur/shared_category/co_authored |
| `adamic_adar_index` | Adamic-Adar similarity (weighted common neighbors by 1/log(deg)) | `limit` | paper pairs by AA score | adamic_adar (weighted) |
| `pattern_match` | Declarative structural pattern matching (like Cypher MATCH) | `pattern_nodes`, `pattern_edges`, `limit` | papers matching pattern | relation per pattern edge |
| `pipeline` | Chain multiple graph algorithms sequentially | `pipeline_steps`, `limit` | final step output nodes | final step output edges |
| `subgraph_projection` | Define precise subgraph then run any algorithm on it | `subgraph_filter`, `subgraph_algorithm`, `subgraph_params`, `limit` | algorithm output nodes | algorithm output edges |
| `traverse` | General BFS traversal with predicates and stop conditions | `seed_arxiv_id`, `traverse_direction`, `traverse_predicate`, `traverse_until`, `collect_edges`, `max_hops` | papers (with depth) | paper→paper (cites) |
| `graph_union` | Union of two sub-query results (all nodes/edges from both) | `set_queries` (2 sub-queries) | merged nodes | merged edges |
| `graph_intersection` | Intersection of two sub-query results (only shared nodes) | `set_queries` (2 sub-queries) | shared nodes | shared edges |

## Graph Query Examples

### 16. Find interdisciplinary papers (rare category combinations)

```bash
curl -s https://arxiv-paperpilot.serveousercontent.com/graph \
  -H "X-API-Key: changeme-key-1" \
  -H "Content-Type: application/json" \
  -d '{
    "graph": {"type": "interdisciplinary", "min_categories": 4, "limit": 10}
  }'
```

### 17. Interdisciplinary LLM papers from 2024+ with code

```bash
curl -s https://arxiv-paperpilot.serveousercontent.com/graph \
  -H "X-API-Key: changeme-key-1" \
  -H "Content-Type: application/json" \
  -d '{
    "graph": {"type": "interdisciplinary", "min_categories": 3, "limit": 10},
    "title_query": "large language model",
    "submitted_date": {"gte": "2024-01-01T00:00:00"},
    "has_github": true
  }'
```

### 18. Co-authorship network of Yann LeCun (2-hop)

```bash
curl -s https://arxiv-paperpilot.serveousercontent.com/graph \
  -H "X-API-Key: changeme-key-1" \
  -H "Content-Type: application/json" \
  -d '{
    "graph": {"type": "coauthor_network", "seed_author": "Yann LeCun", "depth": 2, "limit": 50}
  }'
```

### 19. Authors bridging cs.AI and biology

```bash
curl -s https://arxiv-paperpilot.serveousercontent.com/graph \
  -H "X-API-Key: changeme-key-1" \
  -H "Content-Type: application/json" \
  -d '{
    "graph": {
      "type": "author_bridge",
      "source_categories": ["cs.AI", "cs.LG"],
      "target_categories": ["q-bio.NC", "q-bio.QM", "q-bio.BM"],
      "min_categories": 3,
      "limit": 20
    }
  }'
```

### 20. Category flow: how cs.LG cross-pollinates with other fields

```bash
curl -s https://arxiv-paperpilot.serveousercontent.com/graph \
  -H "X-API-Key: changeme-key-1" \
  -H "Content-Type: application/json" \
  -d '{
    "graph": {
      "type": "cross_category_flow",
      "source_categories": ["cs.LG"],
      "limit": 20
    }
  }'
```

### 21. Papers spanning 5+ categories in physics

```bash
curl -s https://arxiv-paperpilot.serveousercontent.com/graph \
  -H "X-API-Key: changeme-key-1" \
  -H "Content-Type: application/json" \
  -d '{
    "graph": {"type": "category_diversity", "min_categories": 5, "limit": 10},
    "categories": ["physics.comp-ph", "physics.flu-dyn", "physics.class-ph", "hep-th"]
  }'
```

### 22. Co-authorship network scoped to reinforcement learning papers only

```bash
curl -s https://arxiv-paperpilot.serveousercontent.com/graph \
  -H "X-API-Key: changeme-key-1" \
  -H "Content-Type: application/json" \
  -d '{
    "graph": {"type": "coauthor_network", "seed_author": "Sergey Levine", "depth": 1, "limit": 30},
    "query": "reinforcement learning",
    "categories": ["cs.LG", "cs.AI", "cs.RO"]
  }'
```

### 23. Authors bridging math and CS who publish code

```bash
curl -s https://arxiv-paperpilot.serveousercontent.com/graph \
  -H "X-API-Key: changeme-key-1" \
  -H "Content-Type: application/json" \
  -d '{
    "graph": {
      "type": "author_bridge",
      "source_categories": ["math.OC", "math.ST", "stat.ML"],
      "target_categories": ["cs.LG", "cs.AI"],
      "min_categories": 4,
      "limit": 15
    },
    "has_github": true,
    "submitted_date": {"gte": "2023-01-01T00:00:00"}
  }'
```

### 24. Recent papers in the top 10% most cited, with interdisciplinary citations

This finds breakout papers: less than 6 months old, citation count in the top 10% of the last 2 years, and cited by papers from 3+ distinct research fields.

```bash
curl -s https://arxiv-paperpilot.serveousercontent.com/graph \
  -H "X-API-Key: changeme-key-1" \
  -H "Content-Type: application/json" \
  -d '{
    "graph": {
      "type": "rising_interdisciplinary",
      "recency_months": 6,
      "citation_percentile": 90,
      "citation_window_years": 2,
      "min_citing_categories": 3,
      "limit": 10
    }
  }'
```

### 25. Rising interdisciplinary papers in ML/AI with code

```bash
curl -s https://arxiv-paperpilot.serveousercontent.com/graph \
  -H "X-API-Key: changeme-key-1" \
  -H "Content-Type: application/json" \
  -d '{
    "graph": {
      "type": "rising_interdisciplinary",
      "recency_months": 12,
      "citation_percentile": 80,
      "citation_window_years": 2,
      "min_citing_categories": 3,
      "limit": 10
    },
    "categories": ["cs.LG", "cs.AI"],
    "has_github": true
  }'
```

### 26. Citation traversal: what categories do transformer papers cite?

```bash
curl -s https://arxiv-paperpilot.serveousercontent.com/graph \
  -H "X-API-Key: changeme-key-1" \
  -H "Content-Type: application/json" \
  -d '{
    "graph": {
      "type": "citation_traversal",
      "direction": "references",
      "aggregate_by": "category",
      "limit": 15
    },
    "query": "transformer attention",
    "categories": ["cs.LG"],
    "submitted_date": {"gte": "2024-01-01T00:00:00"}
  }'
```

### 27. Citation traversal: who are the most-referenced authors by a single paper?

```bash
curl -s https://arxiv-paperpilot.serveousercontent.com/graph \
  -H "X-API-Key: changeme-key-1" \
  -H "Content-Type: application/json" \
  -d '{
    "graph": {
      "type": "citation_traversal",
      "direction": "references",
      "aggregate_by": "author",
      "seed_arxiv_id": "2602.21169",
      "limit": 10
    }
  }'
```

### 28. Paper citation network: direct paper→paper citation graph

```bash
curl -s https://arxiv-paperpilot.serveousercontent.com/graph \
  -H "X-API-Key: changeme-key-1" \
  -H "Content-Type: application/json" \
  -d '{
    "graph": {
      "type": "paper_citation_network",
      "direction": "references",
      "limit": 20
    },
    "query": "large language model",
    "categories": ["cs.CL"]
  }'
```

### 29. Author influence: who cites Yann LeCun the most?

```bash
curl -s https://arxiv-paperpilot.serveousercontent.com/graph \
  -H "X-API-Key: changeme-key-1" \
  -H "Content-Type: application/json" \
  -d '{
    "graph": {
      "type": "author_influence",
      "seed_author": "Yann LeCun",
      "direction": "cited_by",
      "limit": 20
    }
  }'
```

### 30. Temporal evolution: how cs.AI publication volume changes over time

```bash
curl -s https://arxiv-paperpilot.serveousercontent.com/graph \
  -H "X-API-Key: changeme-key-1" \
  -H "Content-Type: application/json" \
  -d '{
    "graph": {
      "type": "temporal_evolution",
      "time_interval": "year",
      "limit": 30
    },
    "categories": ["cs.AI"]
  }'
```

### 31. Paper similarity: semantic similarity network for protein folding papers

```bash
curl -s https://arxiv-paperpilot.serveousercontent.com/graph \
  -H "X-API-Key: changeme-key-1" \
  -H "Content-Type: application/json" \
  -d '{
    "graph": {
      "type": "paper_similarity",
      "similarity_threshold": 0.6,
      "limit": 20
    },
    "semantic": [
      {"text": "protein folding prediction", "level": "abstract", "weight": 1.0, "mode": "boost"}
    ]
  }'
```

### 32. Domain collaboration: how top-level ArXiv domains co-occur

```bash
curl -s https://arxiv-paperpilot.serveousercontent.com/graph \
  -H "X-API-Key: changeme-key-1" \
  -H "Content-Type: application/json" \
  -d '{
    "graph": {
      "type": "domain_collaboration",
      "limit": 50
    }
  }'
```

### 33. Author topic evolution: how Yann LeCun's research topics shifted over time

```bash
curl -s https://arxiv-paperpilot.serveousercontent.com/graph \
  -H "X-API-Key: changeme-key-1" \
  -H "Content-Type: application/json" \
  -d '{
    "graph": {
      "type": "author_topic_evolution",
      "seed_author": "Yann LeCun",
      "time_interval": "year"
    }
  }'
```

### 34. GitHub landscape: which categories have the most code?

```bash
curl -s https://arxiv-paperpilot.serveousercontent.com/graph \
  -H "X-API-Key: changeme-key-1" \
  -H "Content-Type: application/json" \
  -d '{
    "graph": {
      "type": "github_landscape",
      "time_interval": "year",
      "limit": 30
    }
  }'
```

### 35. Bibliographic coupling: papers with shared references in a topic

```bash
curl -s https://arxiv-paperpilot.serveousercontent.com/graph \
  -H "X-API-Key: changeme-key-1" \
  -H "Content-Type: application/json" \
  -d '{
    "graph": {
      "type": "bibliographic_coupling",
      "limit": 20
    },
    "query": "large language model",
    "categories": ["cs.CL"]
  }'
```

### 36. Co-citation: papers cited together in the same research area

```bash
curl -s https://arxiv-paperpilot.serveousercontent.com/graph \
  -H "X-API-Key: changeme-key-1" \
  -H "Content-Type: application/json" \
  -d '{
    "graph": {
      "type": "cocitation",
      "limit": 20
    },
    "query": "deep reinforcement learning",
    "categories": ["cs.LG"]
  }'
```

### 37. Weighted shortest path between two papers (Dijkstra)

```bash
curl -s https://arxiv-paperpilot.serveousercontent.com/graph \
  -H "X-API-Key: changeme-key-1" \
  -H "Content-Type: application/json" \
  -d '{
    "graph": {
      "type": "weighted_shortest_path",
      "seed_arxiv_id": "2406.03736",
      "target_arxiv_id": "2405.04233",
      "max_hops": 5,
      "weight_field": "citations"
    }
  }'
```

### 38. Betweenness centrality: find bridge papers in transformer research

```bash
curl -s https://arxiv-paperpilot.serveousercontent.com/graph \
  -H "X-API-Key: changeme-key-1" \
  -H "Content-Type: application/json" \
  -d '{
    "graph": {
      "type": "betweenness_centrality",
      "limit": 10
    },
    "query": "transformer"
  }'
```

### 39. Closeness centrality: most-connected papers in a field

```bash
curl -s https://arxiv-paperpilot.serveousercontent.com/graph \
  -H "X-API-Key: changeme-key-1" \
  -H "Content-Type: application/json" \
  -d '{
    "graph": {
      "type": "closeness_centrality",
      "limit": 10
    },
    "query": "neural network",
    "categories": ["cs.LG"]
  }'
```

### 40. Strongly connected components: mutual citation clusters

```bash
curl -s https://arxiv-paperpilot.serveousercontent.com/graph \
  -H "X-API-Key: changeme-key-1" \
  -H "Content-Type: application/json" \
  -d '{
    "graph": {
      "type": "strongly_connected_components",
      "limit": 10
    },
    "query": "machine learning"
  }'
```

### 41. Topological sort: intellectual lineage ordering

```bash
curl -s https://arxiv-paperpilot.serveousercontent.com/graph \
  -H "X-API-Key: changeme-key-1" \
  -H "Content-Type: application/json" \
  -d '{
    "graph": {
      "type": "topological_sort",
      "limit": 20
    },
    "query": "attention mechanism"
  }'
```

### 42. Link prediction: predict future citations

```bash
curl -s https://arxiv-paperpilot.serveousercontent.com/graph \
  -H "X-API-Key: changeme-key-1" \
  -H "Content-Type: application/json" \
  -d '{
    "graph": {
      "type": "link_prediction",
      "prediction_method": "adamic_adar",
      "limit": 10
    },
    "query": "deep learning",
    "categories": ["cs.LG"]
  }'
```

### 43. Louvain community detection: find research communities

```bash
curl -s https://arxiv-paperpilot.serveousercontent.com/graph \
  -H "X-API-Key: changeme-key-1" \
  -H "Content-Type: application/json" \
  -d '{
    "graph": {
      "type": "louvain_community",
      "iterations": 20,
      "limit": 15
    },
    "query": "transformer",
    "categories": ["cs.CL"]
  }'
```

### 44. Degree centrality: most-connected papers by in/out/total degree

```bash
curl -s https://arxiv-paperpilot.serveousercontent.com/graph \
  -H "X-API-Key: changeme-key-1" \
  -H "Content-Type: application/json" \
  -d '{
    "graph": {
      "type": "degree_centrality",
      "degree_mode": "in",
      "limit": 10
    },
    "query": "attention mechanism"
  }'
```

### 45. Eigenvector centrality: recursively prestigious papers

```bash
curl -s https://arxiv-paperpilot.serveousercontent.com/graph \
  -H "X-API-Key: changeme-key-1" \
  -H "Content-Type: application/json" \
  -d '{
    "graph": {
      "type": "eigenvector_centrality",
      "iterations": 30,
      "limit": 10
    },
    "query": "neural network",
    "categories": ["cs.LG"]
  }'
```

### 46. K-core decomposition: find the densest research clusters

```bash
curl -s https://arxiv-paperpilot.serveousercontent.com/graph \
  -H "X-API-Key: changeme-key-1" \
  -H "Content-Type: application/json" \
  -d '{
    "graph": {
      "type": "kcore_decomposition",
      "limit": 20
    },
    "query": "machine learning"
  }'
```

### 47. Articulation points: critical bridge papers

```bash
curl -s https://arxiv-paperpilot.serveousercontent.com/graph \
  -H "X-API-Key: changeme-key-1" \
  -H "Content-Type: application/json" \
  -d '{
    "graph": {
      "type": "articulation_points",
      "limit": 10
    },
    "query": "reinforcement learning"
  }'
```

### 48. Influence maximization: find the k most impactful seed papers

```bash
curl -s https://arxiv-paperpilot.serveousercontent.com/graph \
  -H "X-API-Key: changeme-key-1" \
  -H "Content-Type: application/json" \
  -d '{
    "graph": {
      "type": "influence_maximization",
      "influence_seeds": 3,
      "limit": 20
    },
    "query": "large language model"
  }'
```

### 49. Pattern matching: find NLP papers that cite ML papers (like Cypher MATCH)

```bash
curl -s https://arxiv-paperpilot.serveousercontent.com/graph \
  -H "X-API-Key: changeme-key-1" \
  -H "Content-Type: application/json" \
  -d '{
    "graph": {
      "type": "pattern_match",
      "pattern_nodes": [
        {"alias": "nlp", "type": "paper", "filters": {"categories": ["cs.CL"]}},
        {"alias": "ml", "type": "paper", "filters": {"categories": ["cs.LG"]}}
      ],
      "pattern_edges": [
        {"source": "nlp", "target": "ml", "relation": "cites"}
      ],
      "limit": 10
    },
    "query": "language model"
  }'
```

### 50. Pipeline: PageRank → filter top results → community detection

```bash
curl -s https://arxiv-paperpilot.serveousercontent.com/graph \
  -H "X-API-Key: changeme-key-1" \
  -H "Content-Type: application/json" \
  -d '{
    "graph": {
      "type": "pipeline",
      "pipeline_steps": [
        {"type": "pagerank", "limit": 50, "params": {"damping_factor": 0.85},
         "filter_property": "pagerank", "filter_min": 0.001},
        {"type": "community_detection", "limit": 20}
      ],
      "limit": 15
    },
    "query": "transformer attention"
  }'
```

### 51. Subgraph projection: PageRank on only cs.AI papers from 2024 with 5+ citations

```bash
curl -s https://arxiv-paperpilot.serveousercontent.com/graph \
  -H "X-API-Key: changeme-key-1" \
  -H "Content-Type: application/json" \
  -d '{
    "graph": {
      "type": "subgraph_projection",
      "subgraph_filter": {
        "categories": ["cs.AI"],
        "date_from": "2024-01-01",
        "min_citations": 5,
        "direction": "both",
        "max_nodes": 500
      },
      "subgraph_algorithm": "pagerank",
      "subgraph_params": {"damping_factor": 0.85},
      "limit": 10
    },
    "query": ""
  }'
```

### 52. General traversal: BFS from a paper following cited_by with filters

```bash
curl -s https://arxiv-paperpilot.serveousercontent.com/graph \
  -H "X-API-Key: changeme-key-1" \
  -H "Content-Type: application/json" \
  -d '{
    "graph": {
      "type": "traverse",
      "seed_arxiv_id": "2301.00001",
      "traverse_direction": "outgoing",
      "max_hops": 5,
      "traverse_predicate": {
        "categories": ["cs.AI"],
        "min_citations": 5
      },
      "traverse_until": {
        "max_nodes": 100,
        "max_depth": 4
      },
      "collect_edges": true,
      "limit": 50
    }
  }'
```

### 53. Graph union: combine PageRank and HITS top results

```bash
curl -s https://arxiv-paperpilot.serveousercontent.com/graph \
  -H "X-API-Key: changeme-key-1" \
  -H "Content-Type: application/json" \
  -d '{
    "graph": {
      "type": "graph_union",
      "set_queries": [
        {"type": "pagerank", "limit": 20},
        {"type": "hits", "limit": 20}
      ],
      "limit": 30
    },
    "query": "large language model"
  }'
```

### 54. Graph intersection: papers ranked highly by BOTH betweenness and closeness

```bash
curl -s https://arxiv-paperpilot.serveousercontent.com/graph \
  -H "X-API-Key: changeme-key-1" \
  -H "Content-Type: application/json" \
  -d '{
    "graph": {
      "type": "graph_intersection",
      "set_queries": [
        {"type": "betweenness_centrality", "limit": 50},
        {"type": "closeness_centrality", "limit": 50}
      ],
      "limit": 20
    },
    "query": "neural network"
  }'
```

### Example 22 — Graph + Semantic Similarity (co-authors working on topics similar to "diffusion models for protein folding")

```bash
curl -s https://arxiv-paperpilot.serveousercontent.com/graph \
  -H "X-API-Key: changeme-key-1" \
  -H "Content-Type: application/json" \
  -d '{
    "graph": {
      "type": "coauthor_network",
      "seed_author": "David Baker",
      "depth": 1,
      "limit": 30
    },
    "semantic": {
      "text": "diffusion models for protein structure prediction",
      "level": "abstract",
      "weight": 0.6
    },
    "filters": {
      "date_from": "2023-01-01"
    }
  }'
```

> **Note:** Semantic similarity (`semantic` field) can be combined with **any** graph query type.
> This restricts the graph to papers that are semantically close to the given text,
> giving you e.g. "co-authors of X who work on topics similar to Y" or
> "interdisciplinary papers similar to Z."

### Example 23 — Multi-semantic: papers at the intersection of two topics

```bash
curl -s https://arxiv-paperpilot.serveousercontent.com/search \
  -H "X-API-Key: changeme-key-1" \
  -H "Content-Type: application/json" \
  -d '{
    "limit": 10,
    "semantic": [
      {"text": "machine learning neural networks", "level": "abstract", "weight": 1.0, "mode": "boost"},
      {"text": "protein structure prediction biology", "level": "abstract", "weight": 1.0, "mode": "boost"}
    ]
  }'
```

### Example 24 — Semantic exclude: find ML papers but NOT about computer vision

```bash
curl -s https://arxiv-paperpilot.serveousercontent.com/search \
  -H "X-API-Key: changeme-key-1" \
  -H "Content-Type: application/json" \
  -d '{
    "limit": 10,
    "semantic": [
      {"text": "deep learning machine learning", "level": "abstract", "weight": 1.0, "mode": "boost"},
      {"text": "computer vision image recognition object detection", "level": "abstract", "weight": 0.5, "mode": "exclude"}
    ]
  }'
```

### Example 25 — Text query + semantic exclude: RL papers but NOT game-playing

```bash
curl -s https://arxiv-paperpilot.serveousercontent.com/search \
  -H "X-API-Key: changeme-key-1" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "reinforcement learning",
    "limit": 10,
    "semantic": [
      {"text": "Atari game playing arcade", "level": "abstract", "weight": 0.5, "mode": "exclude"}
    ]
  }'
```

## Output Guidelines

When presenting results to the user:
- Link papers: `https://arxiv.org/abs/{arxiv_id}`
- Highlight first author with bold
- Show categories as tags
- Include abstract snippet only when relevant to the question
- Use tables for multi-paper results
- Mention total count and whether pagination is needed
- For graph results: describe the network structure (node count, edge count, key hubs)
- For interdisciplinary papers: highlight the `interdisciplinary_score` and category spread
- For co-authorship: mention strongest collaborators (highest edge weights)
- For author bridges: list the categories they span
- For citation traversal: describe seeds count, traversed papers, and top aggregates (categories/authors/years)
- For paper_citation_network: show the citation chain and connected papers
- For author_influence: highlight top authors by citation weight
- For temporal_evolution: describe growth/decline trends per category
- For paper_similarity: mention similarity scores and cluster patterns
- For domain_collaboration: highlight strongest domain pairings
- For author_topic_evolution: describe how the author's focus shifted over time
- For github_landscape: compare github adoption rates across categories
- For bibliographic_coupling / cocitation: describe paper relationships and shared-reference counts
- For multihop_citation / shortest_citation_path: show the citation chain and hop distances
- For pagerank: highlight top-ranked papers by PageRank score
- For community_detection: describe discovered communities, sizes, and dominant categories
- For citation_patterns: describe detected motifs (mutual citations, star hubs, chains, triangles)
- For connected_components: describe component sizes and isolated papers
- For weighted_shortest_path: show the path, total cost, and weight method used
- For betweenness_centrality: highlight bridge papers with highest betweenness scores
- For closeness_centrality: highlight most central papers by closeness score
- For strongly_connected_components: describe SCC sizes and mutual citation patterns (note: most citation graphs are DAG-like, so nontrivial SCCs are rare)
- For topological_sort: present papers in foundational→derivative order with topo depth
- For link_prediction: list predicted future citations with scores and prediction method
- For louvain_community: describe communities by size and dominant categories, report modularity score, compare with label propagation if relevant
- For degree_centrality: highlight most-connected papers by the chosen mode (in/out/total), show degree distribution
- For eigenvector_centrality: highlight papers with recursive prestige, note convergence status
- For kcore_decomposition: describe core levels, highlight papers in the densest k-core, show core distribution
- For articulation_points: identify critical bridge papers, describe what communities they connect
- For influence_maximization: list the selected seed papers by influence spread, describe the cascade reach
- For pattern_match: describe the matched structural patterns, number of matches found, and how nodes relate via the declared pattern edges
- For pipeline: describe each step's contribution, how the dataset narrowed at each stage, and the final output
- For subgraph_projection: describe the projected subgraph (size, filters applied, edge direction) and the algorithm results within it
- For traverse: describe the BFS traversal path, depth reached, stop condition triggered, and node/edge counts
- For graph_union: describe what each sub-query contributed and the combined result
- For graph_intersection: describe the overlap between sub-queries and what the shared nodes have in common

## Constraints

- DO NOT fabricate papers — only return what the database contains
- Always use the API or ES `_source` filter — never fetch embedding fields
- ArXiv ID formats: `YYMM.NNNNN` (new) or `category/NNNNNNN` (old, pre-2007)
- Max 200 results per page, max offset 9,999
- Regex patterns max 200 chars, no nested quantifiers
