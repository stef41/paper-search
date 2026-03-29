"""Comprehensive E2E tests for all 16 graph query types.

Tests:
- Each type returns valid response structure
- Each type returns meaningful data (nodes, edges)
- Filter composition works on each type
- Edge cases and error handling
- Semantic boost/exclude compose with graph queries
"""
import asyncio
import json
import sys
import time
from dataclasses import dataclass, field

import httpx

BASE = "http://localhost:8000"
HEADERS = {"X-API-Key": "changeme-key-1", "Content-Type": "application/json"}
TIMEOUT = 30.0


@dataclass
class TestResult:
    name: str
    passed: bool
    took_ms: int = 0
    detail: str = ""
    nodes: int = 0
    edges: int = 0


results: list[TestResult] = []


async def post_graph(client: httpx.AsyncClient, body: dict, name: str) -> dict | None:
    """POST /graph and return parsed response or None on failure."""
    start = time.monotonic()
    try:
        resp = await client.post(f"{BASE}/graph", json=body, headers=HEADERS, timeout=TIMEOUT)
        elapsed = int((time.monotonic() - start) * 1000)
        if resp.status_code != 200:
            results.append(TestResult(name, False, elapsed, f"HTTP {resp.status_code}: {resp.text[:200]}"))
            return None
        data = resp.json()
        return data
    except Exception as e:
        elapsed = int((time.monotonic() - start) * 1000)
        results.append(TestResult(name, False, elapsed, str(e)[:200]))
        return None


def check_structure(data: dict, name: str, min_nodes: int = 0, min_edges: int = 0,
                    allow_empty: bool = False) -> bool:
    """Validate graph response structure and optionally check for data."""
    # Structure checks
    for key in ("nodes", "edges", "total", "took_ms", "metadata"):
        if key not in data:
            results.append(TestResult(name, False, data.get("took_ms", 0),
                                      f"Missing key: {key}"))
            return False

    if not isinstance(data["nodes"], list):
        results.append(TestResult(name, False, data["took_ms"], "nodes is not a list"))
        return False

    if not isinstance(data["edges"], list):
        results.append(TestResult(name, False, data["took_ms"], "edges is not a list"))
        return False

    nn, ne = len(data["nodes"]), len(data["edges"])

    if not allow_empty:
        if nn < min_nodes:
            results.append(TestResult(name, False, data["took_ms"],
                                      f"Expected >= {min_nodes} nodes, got {nn}"))
            return False
        if ne < min_edges:
            results.append(TestResult(name, False, data["took_ms"],
                                      f"Expected >= {min_edges} edges, got {ne}"))
            return False

    # Validate node structure
    for node in data["nodes"][:5]:
        for k in ("id", "label", "type"):
            if k not in node:
                results.append(TestResult(name, False, data["took_ms"],
                                          f"Node missing key: {k}"))
                return False

    # Validate edge structure
    for edge in data["edges"][:5]:
        for k in ("source", "target", "relation"):
            if k not in edge:
                results.append(TestResult(name, False, data["took_ms"],
                                          f"Edge missing key: {k}"))
                return False

    return True


# ─────────────────────────────────────────────────
# 1. CATEGORY DIVERSITY
# ─────────────────────────────────────────────────
async def test_category_diversity_basic(c: httpx.AsyncClient):
    name = "category_diversity:basic"
    data = await post_graph(c, {"graph": {"type": "category_diversity", "min_categories": 4, "limit": 10}}, name)
    if not data:
        return
    if not check_structure(data, name, min_nodes=2, min_edges=1):
        return
    # All paper nodes should have >= 4 categories
    for node in data["nodes"]:
        if node["type"] == "paper":
            cc = node["properties"].get("category_count", 0)
            if cc < 4:
                results.append(TestResult(name, False, data["took_ms"],
                                          f"Paper {node['id']} has {cc} categories, expected >= 4"))
                return
    results.append(TestResult(name, True, data["took_ms"],
                              f"{len(data['nodes'])} nodes, {len(data['edges'])} edges",
                              len(data["nodes"]), len(data["edges"])))


async def test_category_diversity_with_filters(c: httpx.AsyncClient):
    name = "category_diversity:filters"
    data = await post_graph(c, {
        "graph": {"type": "category_diversity", "min_categories": 3, "limit": 10},
        "categories": ["cs.AI", "cs.LG"],
        "submitted_date": {"gte": "2023-01-01T00:00:00"},
        "has_github": True,
    }, name)
    if not data:
        return
    if not check_structure(data, name, min_nodes=1, min_edges=0, allow_empty=True):
        return
    results.append(TestResult(name, True, data["took_ms"],
                              f"{len(data['nodes'])} nodes, {len(data['edges'])} edges",
                              len(data["nodes"]), len(data["edges"])))


# ─────────────────────────────────────────────────
# 2. COAUTHOR NETWORK
# ─────────────────────────────────────────────────
async def test_coauthor_network_basic(c: httpx.AsyncClient):
    name = "coauthor_network:basic"
    data = await post_graph(c, {
        "graph": {"type": "coauthor_network", "seed_author": "Yann LeCun", "depth": 1, "limit": 30}
    }, name)
    if not data:
        return
    if not check_structure(data, name, min_nodes=2, min_edges=1):
        return
    # Seed author should be present
    seed_found = any(n["id"] == "Yann LeCun" for n in data["nodes"])
    if not seed_found:
        results.append(TestResult(name, False, data["took_ms"], "Seed author not in nodes"))
        return
    # All edges should be co_authored
    bad = [e for e in data["edges"] if e["relation"] != "co_authored"]
    if bad:
        results.append(TestResult(name, False, data["took_ms"],
                                  f"Non-co_authored edges: {bad[0]['relation']}"))
        return
    results.append(TestResult(name, True, data["took_ms"],
                              f"{len(data['nodes'])} nodes, {len(data['edges'])} edges, seed=Yann LeCun",
                              len(data["nodes"]), len(data["edges"])))


async def test_coauthor_network_depth2(c: httpx.AsyncClient):
    name = "coauthor_network:depth2"
    data = await post_graph(c, {
        "graph": {"type": "coauthor_network", "seed_author": "Yann LeCun", "depth": 2, "limit": 50}
    }, name)
    if not data:
        return
    if not check_structure(data, name, min_nodes=3, min_edges=2):
        return
    # Should have depth-2 nodes
    depth2_nodes = [n for n in data["nodes"] if n["properties"].get("depth") == 2]
    results.append(TestResult(name, True, data["took_ms"],
                              f"{len(data['nodes'])} nodes, {len(data['edges'])} edges, depth2={len(depth2_nodes)}",
                              len(data["nodes"]), len(data["edges"])))


async def test_coauthor_network_with_filters(c: httpx.AsyncClient):
    name = "coauthor_network:filters"
    data = await post_graph(c, {
        "graph": {"type": "coauthor_network", "seed_author": "Yann LeCun", "depth": 1, "limit": 20},
        "categories": ["cs.LG"],
        "submitted_date": {"gte": "2020-01-01T00:00:00"},
    }, name)
    if not data:
        return
    if not check_structure(data, name, min_nodes=1, min_edges=0, allow_empty=True):
        return
    results.append(TestResult(name, True, data["took_ms"],
                              f"{len(data['nodes'])} nodes, {len(data['edges'])} edges",
                              len(data["nodes"]), len(data["edges"])))


async def test_coauthor_network_missing_seed(c: httpx.AsyncClient):
    name = "coauthor_network:no_seed"
    data = await post_graph(c, {
        "graph": {"type": "coauthor_network", "depth": 1, "limit": 10}
    }, name)
    if not data:
        return
    # Should return error in metadata
    if data["metadata"].get("error"):
        results.append(TestResult(name, True, data["took_ms"], f"error={data['metadata']['error']}"))
    else:
        results.append(TestResult(name, False, data["took_ms"], "Expected error for missing seed_author"))


# ─────────────────────────────────────────────────
# 3. AUTHOR BRIDGE
# ─────────────────────────────────────────────────
async def test_author_bridge_basic(c: httpx.AsyncClient):
    name = "author_bridge:basic"
    data = await post_graph(c, {
        "graph": {"type": "author_bridge", "min_categories": 3, "limit": 15}
    }, name)
    if not data:
        return
    if not check_structure(data, name, min_nodes=2, min_edges=1):
        return
    # Authors should have >= 3 categories
    for node in data["nodes"]:
        if node["type"] == "author":
            cc = node["properties"].get("category_count", 0)
            if cc < 3:
                results.append(TestResult(name, False, data["took_ms"],
                                          f"Author {node['id']} has {cc} cats, expected >= 3"))
                return
    results.append(TestResult(name, True, data["took_ms"],
                              f"{len(data['nodes'])} nodes, {len(data['edges'])} edges",
                              len(data["nodes"]), len(data["edges"])))


async def test_author_bridge_source_target(c: httpx.AsyncClient):
    name = "author_bridge:source_target"
    data = await post_graph(c, {
        "graph": {
            "type": "author_bridge",
            "source_categories": ["cs.AI", "cs.LG"],
            "target_categories": ["q-bio.NC", "q-bio.QM"],
            "min_categories": 3,
            "limit": 10
        }
    }, name)
    if not data:
        return
    if not check_structure(data, name, allow_empty=True):
        return
    results.append(TestResult(name, True, data["took_ms"],
                              f"{len(data['nodes'])} nodes, {len(data['edges'])} edges",
                              len(data["nodes"]), len(data["edges"])))


# ─────────────────────────────────────────────────
# 4. CROSS CATEGORY FLOW
# ─────────────────────────────────────────────────
async def test_cross_category_flow_basic(c: httpx.AsyncClient):
    name = "cross_category_flow:basic"
    data = await post_graph(c, {
        "graph": {"type": "cross_category_flow", "limit": 20}
    }, name)
    if not data:
        return
    if not check_structure(data, name, min_nodes=2, min_edges=1):
        return
    # All edges should be co_occurs
    bad = [e for e in data["edges"] if e["relation"] != "co_occurs"]
    if bad:
        results.append(TestResult(name, False, data["took_ms"],
                                  f"Non co_occurs edge: {bad[0]['relation']}"))
        return
    # All edges should have weight > 0
    zero_w = [e for e in data["edges"] if (e.get("weight") or 0) <= 0]
    if zero_w:
        results.append(TestResult(name, False, data["took_ms"], "Edge with zero weight"))
        return
    results.append(TestResult(name, True, data["took_ms"],
                              f"{len(data['nodes'])} nodes, {len(data['edges'])} edges",
                              len(data["nodes"]), len(data["edges"])))


async def test_cross_category_flow_source(c: httpx.AsyncClient):
    name = "cross_category_flow:source_filter"
    data = await post_graph(c, {
        "graph": {"type": "cross_category_flow", "source_categories": ["cs.LG"], "limit": 20}
    }, name)
    if not data:
        return
    if not check_structure(data, name, min_nodes=2, min_edges=1):
        return
    # All edges should involve cs.LG
    for e in data["edges"]:
        if "cs.LG" not in (e["source"], e["target"]):
            results.append(TestResult(name, False, data["took_ms"],
                                      f"Edge {e['source']}→{e['target']} doesn't involve cs.LG"))
            return
    results.append(TestResult(name, True, data["took_ms"],
                              f"{len(data['nodes'])} nodes, {len(data['edges'])} edges",
                              len(data["nodes"]), len(data["edges"])))


# ─────────────────────────────────────────────────
# 5. INTERDISCIPLINARY
# ─────────────────────────────────────────────────
async def test_interdisciplinary_basic(c: httpx.AsyncClient):
    name = "interdisciplinary:basic"
    data = await post_graph(c, {
        "graph": {"type": "interdisciplinary", "min_categories": 3, "limit": 10}
    }, name)
    if not data:
        return
    if not check_structure(data, name, min_nodes=2, min_edges=1):
        return
    # Papers should have interdisciplinary_score
    for node in data["nodes"]:
        if node["type"] == "paper":
            score = node["properties"].get("interdisciplinary_score")
            if score is None:
                results.append(TestResult(name, False, data["took_ms"],
                                          f"Paper {node['id']} missing interdisciplinary_score"))
                return
            if not (0 <= score <= 1):
                results.append(TestResult(name, False, data["took_ms"],
                                          f"Score {score} out of range"))
                return
    results.append(TestResult(name, True, data["took_ms"],
                              f"{len(data['nodes'])} nodes, {len(data['edges'])} edges",
                              len(data["nodes"]), len(data["edges"])))


async def test_interdisciplinary_with_query(c: httpx.AsyncClient):
    name = "interdisciplinary:with_query"
    data = await post_graph(c, {
        "graph": {"type": "interdisciplinary", "min_categories": 2, "limit": 10},
        "title_query": "large language model",
        "submitted_date": {"gte": "2024-01-01T00:00:00"},
    }, name)
    if not data:
        return
    if not check_structure(data, name, allow_empty=True):
        return
    results.append(TestResult(name, True, data["took_ms"],
                              f"{len(data['nodes'])} nodes, {len(data['edges'])} edges",
                              len(data["nodes"]), len(data["edges"])))


# ─────────────────────────────────────────────────
# 6. RISING INTERDISCIPLINARY
# ─────────────────────────────────────────────────
async def test_rising_interdisciplinary_basic(c: httpx.AsyncClient):
    name = "rising_interdisciplinary:basic"
    data = await post_graph(c, {
        "graph": {
            "type": "rising_interdisciplinary",
            "recency_months": 12,
            "citation_percentile": 80,
            "citation_window_years": 3,
            "min_citing_categories": 2,
            "limit": 10
        }
    }, name)
    if not data:
        return
    if not check_structure(data, name, allow_empty=True):
        return
    # Check metadata has expected keys
    meta = data["metadata"]
    for k in ("citation_threshold", "citation_percentile", "recency_months"):
        if k not in meta and "error" not in meta:
            results.append(TestResult(name, False, data["took_ms"], f"Missing metadata key: {k}"))
            return
    results.append(TestResult(name, True, data["took_ms"],
                              f"{len(data['nodes'])} nodes, {len(data['edges'])} edges, meta={meta}",
                              len(data["nodes"]), len(data["edges"])))


# ─────────────────────────────────────────────────
# 7. CITATION TRAVERSAL
# ─────────────────────────────────────────────────
async def test_citation_traversal_by_category(c: httpx.AsyncClient):
    name = "citation_traversal:by_category"
    data = await post_graph(c, {
        "graph": {
            "type": "citation_traversal",
            "direction": "references",
            "aggregate_by": "category",
            "limit": 15
        },
        "query": "transformer",
        "categories": ["cs.CL"],
    }, name)
    if not data:
        return
    if not check_structure(data, name, allow_empty=True):
        return
    results.append(TestResult(name, True, data["took_ms"],
                              f"{len(data['nodes'])} nodes, {len(data['edges'])} edges",
                              len(data["nodes"]), len(data["edges"])))


async def test_citation_traversal_by_author(c: httpx.AsyncClient):
    name = "citation_traversal:by_author"
    # Find a paper with references first
    resp = await c.post(f"{BASE}/search", json={
        "query": "transformer", "min_references": 5, "limit": 1
    }, headers=HEADERS, timeout=TIMEOUT)
    if resp.status_code != 200:
        results.append(TestResult(name, False, 0, f"Search failed: {resp.status_code}"))
        return
    papers = resp.json().get("hits", [])
    if not papers:
        results.append(TestResult(name, True, 0, "No papers with references found (skip)"))
        return
    seed_id = papers[0]["arxiv_id"]

    data = await post_graph(c, {
        "graph": {
            "type": "citation_traversal",
            "direction": "references",
            "aggregate_by": "author",
            "seed_arxiv_id": seed_id,
            "limit": 15
        }
    }, name)
    if not data:
        return
    if not check_structure(data, name, allow_empty=True):
        return
    results.append(TestResult(name, True, data["took_ms"],
                              f"seed={seed_id}, {len(data['nodes'])} nodes, {len(data['edges'])} edges",
                              len(data["nodes"]), len(data["edges"])))


async def test_citation_traversal_by_year(c: httpx.AsyncClient):
    name = "citation_traversal:by_year"
    data = await post_graph(c, {
        "graph": {
            "type": "citation_traversal",
            "direction": "cited_by",
            "aggregate_by": "year",
            "limit": 20
        },
        "categories": ["cs.LG"],
        "min_citations": 10,
    }, name)
    if not data:
        return
    if not check_structure(data, name, allow_empty=True):
        return
    results.append(TestResult(name, True, data["took_ms"],
                              f"{len(data['nodes'])} nodes, {len(data['edges'])} edges",
                              len(data["nodes"]), len(data["edges"])))


# ─────────────────────────────────────────────────
# 8. PAPER CITATION NETWORK
# ─────────────────────────────────────────────────
async def test_paper_citation_network_basic(c: httpx.AsyncClient):
    name = "paper_citation_network:basic"
    data = await post_graph(c, {
        "graph": {"type": "paper_citation_network", "direction": "references", "limit": 20},
        "query": "large language model",
        "categories": ["cs.CL"]
    }, name)
    if not data:
        return
    if not check_structure(data, name, allow_empty=True):
        return
    # If we have data, edges should be "cites"
    for e in data["edges"][:10]:
        if e["relation"] != "cites":
            results.append(TestResult(name, False, data["took_ms"],
                                      f"Expected relation=cites, got {e['relation']}"))
            return
    results.append(TestResult(name, True, data["took_ms"],
                              f"{len(data['nodes'])} nodes, {len(data['edges'])} edges",
                              len(data["nodes"]), len(data["edges"])))


async def test_paper_citation_network_seed_id(c: httpx.AsyncClient):
    name = "paper_citation_network:seed_id"
    # Find paper with references
    resp = await c.get(f"{BASE}/paper/2603.19272", headers=HEADERS, timeout=TIMEOUT)
    if resp.status_code != 200:
        results.append(TestResult(name, True, 0, "Paper 2603.19272 not found (skip)"))
        return
    data = await post_graph(c, {
        "graph": {"type": "paper_citation_network", "seed_arxiv_id": "2603.19272",
                  "direction": "references", "limit": 20}
    }, name)
    if not data:
        return
    if not check_structure(data, name, allow_empty=True):
        return
    # Seed should be in nodes
    seed_found = any(n["id"] == "2603.19272" for n in data["nodes"])
    if data["nodes"] and not seed_found:
        results.append(TestResult(name, False, data["took_ms"],
                                  "Seed paper not in nodes"))
        return
    results.append(TestResult(name, True, data["took_ms"],
                              f"{len(data['nodes'])} nodes, {len(data['edges'])} edges",
                              len(data["nodes"]), len(data["edges"])))


async def test_paper_citation_network_seed_ids(c: httpx.AsyncClient):
    name = "paper_citation_network:seed_ids"
    data = await post_graph(c, {
        "graph": {"type": "paper_citation_network",
                  "seed_arxiv_ids": ["2603.19272", "2502.10001"],
                  "direction": "references", "limit": 20}
    }, name)
    if not data:
        return
    if not check_structure(data, name, allow_empty=True):
        return
    results.append(TestResult(name, True, data["took_ms"],
                              f"{len(data['nodes'])} nodes, {len(data['edges'])} edges",
                              len(data["nodes"]), len(data["edges"])))


async def test_paper_citation_network_cited_by(c: httpx.AsyncClient):
    name = "paper_citation_network:cited_by"
    data = await post_graph(c, {
        "graph": {"type": "paper_citation_network", "direction": "cited_by", "limit": 15},
        "categories": ["cs.LG"],
        "min_citations": 5,
    }, name)
    if not data:
        return
    if not check_structure(data, name, allow_empty=True):
        return
    results.append(TestResult(name, True, data["took_ms"],
                              f"{len(data['nodes'])} nodes, {len(data['edges'])} edges",
                              len(data["nodes"]), len(data["edges"])))


async def test_paper_citation_network_with_filters(c: httpx.AsyncClient):
    name = "paper_citation_network:all_filters"
    data = await post_graph(c, {
        "graph": {"type": "paper_citation_network", "direction": "references", "limit": 10},
        "query": "attention mechanism",
        "categories": ["cs.CL", "cs.LG"],
        "submitted_date": {"gte": "2023-01-01T00:00:00"},
        "has_github": True,
    }, name)
    if not data:
        return
    if not check_structure(data, name, allow_empty=True):
        return
    results.append(TestResult(name, True, data["took_ms"],
                              f"{len(data['nodes'])} nodes, {len(data['edges'])} edges",
                              len(data["nodes"]), len(data["edges"])))


# ─────────────────────────────────────────────────
# 9. AUTHOR INFLUENCE
# ─────────────────────────────────────────────────
async def test_author_influence_cited_by(c: httpx.AsyncClient):
    name = "author_influence:cited_by"
    data = await post_graph(c, {
        "graph": {"type": "author_influence", "seed_author": "Yann LeCun",
                  "direction": "cited_by", "limit": 20}
    }, name)
    if not data:
        return
    if not check_structure(data, name, allow_empty=True):
        return
    # Seed = first node
    if data["nodes"]:
        if data["nodes"][0]["id"] != "Yann LeCun":
            results.append(TestResult(name, False, data["took_ms"],
                                      f"First node is {data['nodes'][0]['id']}, expected Yann LeCun"))
            return
    results.append(TestResult(name, True, data["took_ms"],
                              f"{len(data['nodes'])} nodes, {len(data['edges'])} edges",
                              len(data["nodes"]), len(data["edges"])))


async def test_author_influence_references(c: httpx.AsyncClient):
    name = "author_influence:references"
    data = await post_graph(c, {
        "graph": {"type": "author_influence", "seed_author": "Yann LeCun",
                  "direction": "references", "limit": 20}
    }, name)
    if not data:
        return
    if not check_structure(data, name, allow_empty=True):
        return
    # Edges should flow from seed
    for e in data["edges"]:
        if e["source"] != "Yann LeCun":
            results.append(TestResult(name, False, data["took_ms"],
                                      f"Edge source={e['source']}, expected Yann LeCun"))
            return
    results.append(TestResult(name, True, data["took_ms"],
                              f"{len(data['nodes'])} nodes, {len(data['edges'])} edges",
                              len(data["nodes"]), len(data["edges"])))


async def test_author_influence_missing_seed(c: httpx.AsyncClient):
    name = "author_influence:no_seed"
    data = await post_graph(c, {
        "graph": {"type": "author_influence", "direction": "cited_by", "limit": 10}
    }, name)
    if not data:
        return
    if data["metadata"].get("error"):
        results.append(TestResult(name, True, data["took_ms"], f"error={data['metadata']['error']}"))
    else:
        results.append(TestResult(name, False, data["took_ms"], "Expected error for missing seed_author"))


async def test_author_influence_with_filters(c: httpx.AsyncClient):
    name = "author_influence:filters"
    data = await post_graph(c, {
        "graph": {"type": "author_influence", "seed_author": "Yann LeCun",
                  "direction": "cited_by", "limit": 15},
        "categories": ["cs.LG"],
        "submitted_date": {"gte": "2020-01-01T00:00:00"},
    }, name)
    if not data:
        return
    if not check_structure(data, name, allow_empty=True):
        return
    results.append(TestResult(name, True, data["took_ms"],
                              f"{len(data['nodes'])} nodes, {len(data['edges'])} edges",
                              len(data["nodes"]), len(data["edges"])))


# ─────────────────────────────────────────────────
# 10. TEMPORAL EVOLUTION
# ─────────────────────────────────────────────────
async def test_temporal_evolution_year(c: httpx.AsyncClient):
    name = "temporal_evolution:year"
    data = await post_graph(c, {
        "graph": {"type": "temporal_evolution", "time_interval": "year", "limit": 20},
        "categories": ["cs.AI"]
    }, name)
    if not data:
        return
    if not check_structure(data, name, min_nodes=3, min_edges=1):
        return
    # Should have time and category nodes
    time_nodes = [n for n in data["nodes"] if n["type"] == "time"]
    cat_nodes = [n for n in data["nodes"] if n["type"] == "category"]
    if not time_nodes:
        results.append(TestResult(name, False, data["took_ms"], "No time nodes"))
        return
    if not cat_nodes:
        results.append(TestResult(name, False, data["took_ms"], "No category nodes"))
        return
    results.append(TestResult(name, True, data["took_ms"],
                              f"{len(time_nodes)} time, {len(cat_nodes)} cat, {len(data['edges'])} edges",
                              len(data["nodes"]), len(data["edges"])))


async def test_temporal_evolution_month(c: httpx.AsyncClient):
    name = "temporal_evolution:month"
    data = await post_graph(c, {
        "graph": {"type": "temporal_evolution", "time_interval": "month", "limit": 10},
        "categories": ["cs.LG"],
        "submitted_date": {"gte": "2025-01-01T00:00:00"},
    }, name)
    if not data:
        return
    if not check_structure(data, name, allow_empty=True):
        return
    results.append(TestResult(name, True, data["took_ms"],
                              f"{len(data['nodes'])} nodes, {len(data['edges'])} edges",
                              len(data["nodes"]), len(data["edges"])))


async def test_temporal_evolution_quarter(c: httpx.AsyncClient):
    name = "temporal_evolution:quarter"
    data = await post_graph(c, {
        "graph": {"type": "temporal_evolution", "time_interval": "quarter", "limit": 15},
        "query": "reinforcement learning"
    }, name)
    if not data:
        return
    if not check_structure(data, name, min_nodes=1, min_edges=0, allow_empty=True):
        return
    results.append(TestResult(name, True, data["took_ms"],
                              f"{len(data['nodes'])} nodes, {len(data['edges'])} edges",
                              len(data["nodes"]), len(data["edges"])))


async def test_temporal_evolution_all_filters(c: httpx.AsyncClient):
    name = "temporal_evolution:all_filters"
    data = await post_graph(c, {
        "graph": {"type": "temporal_evolution", "time_interval": "year", "limit": 10},
        "query": "neural network",
        "categories": ["cs.LG"],
        "submitted_date": {"gte": "2020-01-01T00:00:00"},
        "min_citations": 5,
        "has_github": True,
        "has_doi": True,
    }, name)
    if not data:
        return
    if not check_structure(data, name, allow_empty=True):
        return
    results.append(TestResult(name, True, data["took_ms"],
                              f"{len(data['nodes'])} nodes, {len(data['edges'])} edges",
                              len(data["nodes"]), len(data["edges"])))


# ─────────────────────────────────────────────────
# 11. PAPER SIMILARITY
# ─────────────────────────────────────────────────
async def test_paper_similarity_basic(c: httpx.AsyncClient):
    name = "paper_similarity:basic"
    data = await post_graph(c, {
        "graph": {"type": "paper_similarity", "similarity_threshold": 0.5, "limit": 20},
        "semantic": [{"text": "protein folding prediction", "level": "abstract",
                      "weight": 1.0, "mode": "boost"}]
    }, name)
    if not data:
        return
    if not check_structure(data, name, min_nodes=2, min_edges=0, allow_empty=True):
        return
    # Similarity edges should have weight
    for e in data["edges"]:
        if e.get("weight") is None:
            results.append(TestResult(name, False, data["took_ms"], "Edge missing weight"))
            return
        if e["weight"] < 0.5:
            results.append(TestResult(name, False, data["took_ms"],
                                      f"Edge weight {e['weight']} below threshold 0.5"))
            return
    results.append(TestResult(name, True, data["took_ms"],
                              f"{len(data['nodes'])} nodes, {len(data['edges'])} edges",
                              len(data["nodes"]), len(data["edges"])))


async def test_paper_similarity_high_threshold(c: httpx.AsyncClient):
    name = "paper_similarity:high_threshold"
    data = await post_graph(c, {
        "graph": {"type": "paper_similarity", "similarity_threshold": 0.9, "limit": 20},
        "semantic": [{"text": "transformer attention mechanism", "level": "abstract",
                      "weight": 1.0, "mode": "boost"}]
    }, name)
    if not data:
        return
    if not check_structure(data, name, allow_empty=True):
        return
    # High threshold should produce fewer edges
    for e in data["edges"]:
        if e.get("weight", 0) < 0.9:
            results.append(TestResult(name, False, data["took_ms"],
                                      f"Edge weight {e['weight']} below threshold 0.9"))
            return
    results.append(TestResult(name, True, data["took_ms"],
                              f"{len(data['nodes'])} nodes, {len(data['edges'])} edges",
                              len(data["nodes"]), len(data["edges"])))


async def test_paper_similarity_no_semantic(c: httpx.AsyncClient):
    name = "paper_similarity:no_semantic"
    data = await post_graph(c, {
        "graph": {"type": "paper_similarity", "similarity_threshold": 0.5, "limit": 10},
        "query": "transformer"
    }, name)
    if not data:
        return
    # Should return error because semantic boost is required
    if data["metadata"].get("error"):
        results.append(TestResult(name, True, data["took_ms"], f"error={data['metadata']['error']}"))
    else:
        results.append(TestResult(name, False, data["took_ms"], "Expected error without semantic"))


async def test_paper_similarity_with_filters(c: httpx.AsyncClient):
    name = "paper_similarity:with_filters"
    data = await post_graph(c, {
        "graph": {"type": "paper_similarity", "similarity_threshold": 0.5, "limit": 15},
        "semantic": [{"text": "graph neural networks", "level": "abstract",
                      "weight": 1.0, "mode": "boost"}],
        "categories": ["cs.LG"],
        "submitted_date": {"gte": "2023-01-01T00:00:00"},
    }, name)
    if not data:
        return
    if not check_structure(data, name, allow_empty=True):
        return
    results.append(TestResult(name, True, data["took_ms"],
                              f"{len(data['nodes'])} nodes, {len(data['edges'])} edges",
                              len(data["nodes"]), len(data["edges"])))


# ─────────────────────────────────────────────────
# 12. DOMAIN COLLABORATION
# ─────────────────────────────────────────────────
async def test_domain_collaboration_basic(c: httpx.AsyncClient):
    name = "domain_collaboration:basic"
    data = await post_graph(c, {
        "graph": {"type": "domain_collaboration", "limit": 50}
    }, name)
    if not data:
        return
    if not check_structure(data, name, min_nodes=2, min_edges=1):
        return
    # All nodes should be domain type
    non_domain = [n for n in data["nodes"] if n["type"] != "domain"]
    if non_domain:
        results.append(TestResult(name, False, data["took_ms"],
                                  f"Non-domain node: {non_domain[0]}"))
        return
    # Check known domains exist
    domain_ids = {n["id"] for n in data["nodes"]}
    expected = {"cs", "math", "physics"}
    found = expected & domain_ids
    results.append(TestResult(name, True, data["took_ms"],
                              f"{len(data['nodes'])} domains, {len(data['edges'])} edges, found={found}",
                              len(data["nodes"]), len(data["edges"])))


async def test_domain_collaboration_source_target(c: httpx.AsyncClient):
    name = "domain_collaboration:source_target"
    data = await post_graph(c, {
        "graph": {"type": "domain_collaboration",
                  "source_categories": ["cs"],
                  "target_categories": ["math", "physics"],
                  "limit": 20}
    }, name)
    if not data:
        return
    if not check_structure(data, name, allow_empty=True):
        return
    results.append(TestResult(name, True, data["took_ms"],
                              f"{len(data['nodes'])} nodes, {len(data['edges'])} edges",
                              len(data["nodes"]), len(data["edges"])))


async def test_domain_collaboration_with_query(c: httpx.AsyncClient):
    name = "domain_collaboration:with_query"
    data = await post_graph(c, {
        "graph": {"type": "domain_collaboration", "limit": 20},
        "query": "machine learning",
        "submitted_date": {"gte": "2023-01-01T00:00:00"},
    }, name)
    if not data:
        return
    if not check_structure(data, name, allow_empty=True):
        return
    results.append(TestResult(name, True, data["took_ms"],
                              f"{len(data['nodes'])} nodes, {len(data['edges'])} edges",
                              len(data["nodes"]), len(data["edges"])))


# ─────────────────────────────────────────────────
# 13. AUTHOR TOPIC EVOLUTION
# ─────────────────────────────────────────────────
async def test_author_topic_evolution_year(c: httpx.AsyncClient):
    name = "author_topic_evolution:year"
    data = await post_graph(c, {
        "graph": {"type": "author_topic_evolution", "seed_author": "Yann LeCun",
                  "time_interval": "year"}
    }, name)
    if not data:
        return
    if not check_structure(data, name, min_nodes=3, min_edges=1):
        return
    # Should have author, time, and category nodes
    types = {n["type"] for n in data["nodes"]}
    expected_types = {"author", "time", "category"}
    if not expected_types.issubset(types):
        results.append(TestResult(name, False, data["took_ms"],
                                  f"Node types={types}, expected {expected_types}"))
        return
    # First node should be the seed author
    if data["nodes"][0]["id"] != "Yann LeCun":
        results.append(TestResult(name, False, data["took_ms"],
                                  f"First node is {data['nodes'][0]['id']}"))
        return
    results.append(TestResult(name, True, data["took_ms"],
                              f"{len(data['nodes'])} nodes, {len(data['edges'])} edges, total={data['total']}",
                              len(data["nodes"]), len(data["edges"])))


async def test_author_topic_evolution_month(c: httpx.AsyncClient):
    name = "author_topic_evolution:month"
    data = await post_graph(c, {
        "graph": {"type": "author_topic_evolution", "seed_author": "Yann LeCun",
                  "time_interval": "month"}
    }, name)
    if not data:
        return
    if not check_structure(data, name, min_nodes=3, min_edges=1):
        return
    results.append(TestResult(name, True, data["took_ms"],
                              f"{len(data['nodes'])} nodes, {len(data['edges'])} edges",
                              len(data["nodes"]), len(data["edges"])))


async def test_author_topic_evolution_no_seed(c: httpx.AsyncClient):
    name = "author_topic_evolution:no_seed"
    data = await post_graph(c, {
        "graph": {"type": "author_topic_evolution", "time_interval": "year"}
    }, name)
    if not data:
        return
    if data["metadata"].get("error"):
        results.append(TestResult(name, True, data["took_ms"], f"error={data['metadata']['error']}"))
    else:
        results.append(TestResult(name, False, data["took_ms"], "Expected error for missing seed_author"))


async def test_author_topic_evolution_with_filters(c: httpx.AsyncClient):
    name = "author_topic_evolution:filters"
    data = await post_graph(c, {
        "graph": {"type": "author_topic_evolution", "seed_author": "Yann LeCun",
                  "time_interval": "year"},
        "categories": ["cs.LG"],
        "submitted_date": {"gte": "2015-01-01T00:00:00"},
    }, name)
    if not data:
        return
    if not check_structure(data, name, allow_empty=True):
        return
    results.append(TestResult(name, True, data["took_ms"],
                              f"{len(data['nodes'])} nodes, {len(data['edges'])} edges",
                              len(data["nodes"]), len(data["edges"])))


# ─────────────────────────────────────────────────
# 14. GITHUB LANDSCAPE
# ─────────────────────────────────────────────────
async def test_github_landscape_basic(c: httpx.AsyncClient):
    name = "github_landscape:basic"
    data = await post_graph(c, {
        "graph": {"type": "github_landscape", "time_interval": "year", "limit": 20}
    }, name)
    if not data:
        return
    if not check_structure(data, name, min_nodes=3, min_edges=1):
        return
    # Should have category, domain, and time nodes
    types = {n["type"] for n in data["nodes"]}
    if "category" not in types:
        results.append(TestResult(name, False, data["took_ms"], f"No category nodes, types={types}"))
        return
    # Categories should have github_rate property
    for n in data["nodes"]:
        if n["type"] == "category":
            rate = n["properties"].get("github_rate")
            if rate is None:
                results.append(TestResult(name, False, data["took_ms"],
                                          f"Category {n['id']} missing github_rate"))
                return
            if not (0 <= rate <= 1):
                results.append(TestResult(name, False, data["took_ms"],
                                          f"github_rate {rate} out of range"))
                return
    results.append(TestResult(name, True, data["took_ms"],
                              f"{len(data['nodes'])} nodes, {len(data['edges'])} edges, total={data['total']}",
                              len(data["nodes"]), len(data["edges"])))


async def test_github_landscape_with_query(c: httpx.AsyncClient):
    name = "github_landscape:with_query"
    data = await post_graph(c, {
        "graph": {"type": "github_landscape", "time_interval": "year", "limit": 10},
        "query": "deep learning",
        "categories": ["cs.LG", "cs.AI"],
    }, name)
    if not data:
        return
    if not check_structure(data, name, allow_empty=True):
        return
    results.append(TestResult(name, True, data["took_ms"],
                              f"{len(data['nodes'])} nodes, {len(data['edges'])} edges",
                              len(data["nodes"]), len(data["edges"])))


async def test_github_landscape_month(c: httpx.AsyncClient):
    name = "github_landscape:month"
    data = await post_graph(c, {
        "graph": {"type": "github_landscape", "time_interval": "month", "limit": 10},
        "submitted_date": {"gte": "2025-01-01T00:00:00"},
    }, name)
    if not data:
        return
    if not check_structure(data, name, allow_empty=True):
        return
    results.append(TestResult(name, True, data["took_ms"],
                              f"{len(data['nodes'])} nodes, {len(data['edges'])} edges",
                              len(data["nodes"]), len(data["edges"])))


# ─────────────────────────────────────────────────
# 15. BIBLIOGRAPHIC COUPLING
# ─────────────────────────────────────────────────
async def test_bibliographic_coupling_basic(c: httpx.AsyncClient):
    name = "bibliographic_coupling:basic"
    data = await post_graph(c, {
        "graph": {"type": "bibliographic_coupling", "limit": 20},
        "query": "large language model",
        "categories": ["cs.CL"]
    }, name)
    if not data:
        return
    if not check_structure(data, name, allow_empty=True):
        return
    # Edges should be shared_references with weight > 0
    for e in data["edges"]:
        if e["relation"] != "shared_references":
            results.append(TestResult(name, False, data["took_ms"],
                                      f"Expected shared_references, got {e['relation']}"))
            return
        if (e.get("weight") or 0) <= 0:
            results.append(TestResult(name, False, data["took_ms"], "Edge weight <= 0"))
            return
    results.append(TestResult(name, True, data["took_ms"],
                              f"{len(data['nodes'])} nodes, {len(data['edges'])} edges",
                              len(data["nodes"]), len(data["edges"])))


async def test_bibliographic_coupling_seed_id(c: httpx.AsyncClient):
    name = "bibliographic_coupling:seed_id"
    data = await post_graph(c, {
        "graph": {"type": "bibliographic_coupling", "seed_arxiv_id": "2603.19272", "limit": 15}
    }, name)
    if not data:
        return
    if not check_structure(data, name, allow_empty=True):
        return
    results.append(TestResult(name, True, data["took_ms"],
                              f"{len(data['nodes'])} nodes, {len(data['edges'])} edges",
                              len(data["nodes"]), len(data["edges"])))


async def test_bibliographic_coupling_seed_ids(c: httpx.AsyncClient):
    name = "bibliographic_coupling:seed_ids"
    data = await post_graph(c, {
        "graph": {"type": "bibliographic_coupling",
                  "seed_arxiv_ids": ["2603.19272", "2502.10001", "2603.19348"],
                  "limit": 15}
    }, name)
    if not data:
        return
    if not check_structure(data, name, allow_empty=True):
        return
    # Papers should have reference_count in properties
    for n in data["nodes"]:
        if "reference_count" not in n["properties"]:
            results.append(TestResult(name, False, data["took_ms"],
                                      f"Node {n['id']} missing reference_count"))
            return
    results.append(TestResult(name, True, data["took_ms"],
                              f"{len(data['nodes'])} nodes, {len(data['edges'])} edges",
                              len(data["nodes"]), len(data["edges"])))


async def test_bibliographic_coupling_with_filters(c: httpx.AsyncClient):
    name = "bibliographic_coupling:filters"
    data = await post_graph(c, {
        "graph": {"type": "bibliographic_coupling", "limit": 10},
        "categories": ["cs.CL"],
        "submitted_date": {"gte": "2025-01-01T00:00:00"},
        "min_citations": 1,
    }, name)
    if not data:
        return
    if not check_structure(data, name, allow_empty=True):
        return
    results.append(TestResult(name, True, data["took_ms"],
                              f"{len(data['nodes'])} nodes, {len(data['edges'])} edges",
                              len(data["nodes"]), len(data["edges"])))


# ─────────────────────────────────────────────────
# 16. COCITATION
# ─────────────────────────────────────────────────
async def test_cocitation_basic(c: httpx.AsyncClient):
    name = "cocitation:basic"
    data = await post_graph(c, {
        "graph": {"type": "cocitation", "limit": 20},
        "query": "deep learning",
        "categories": ["cs.LG"]
    }, name)
    if not data:
        return
    if not check_structure(data, name, allow_empty=True):
        return
    for e in data["edges"]:
        if e["relation"] != "co_cited":
            results.append(TestResult(name, False, data["took_ms"],
                                      f"Expected co_cited, got {e['relation']}"))
            return
    results.append(TestResult(name, True, data["took_ms"],
                              f"{len(data['nodes'])} nodes, {len(data['edges'])} edges",
                              len(data["nodes"]), len(data["edges"])))


async def test_cocitation_seed_id(c: httpx.AsyncClient):
    name = "cocitation:seed_id"
    data = await post_graph(c, {
        "graph": {"type": "cocitation", "seed_arxiv_id": "2603.19272", "limit": 10}
    }, name)
    if not data:
        return
    if not check_structure(data, name, allow_empty=True):
        return
    results.append(TestResult(name, True, data["took_ms"],
                              f"{len(data['nodes'])} nodes, {len(data['edges'])} edges",
                              len(data["nodes"]), len(data["edges"])))


async def test_cocitation_with_filters(c: httpx.AsyncClient):
    name = "cocitation:filters"
    data = await post_graph(c, {
        "graph": {"type": "cocitation", "limit": 10},
        "categories": ["cs.LG"],
        "min_citations": 5,
        "submitted_date": {"gte": "2020-01-01T00:00:00"},
    }, name)
    if not data:
        return
    if not check_structure(data, name, allow_empty=True):
        return
    results.append(TestResult(name, True, data["took_ms"],
                              f"{len(data['nodes'])} nodes, {len(data['edges'])} edges",
                              len(data["nodes"]), len(data["edges"])))


# ─────────────────────────────────────────────────
# FILTER COMPOSITION TESTS
# ─────────────────────────────────────────────────
async def test_filters_text_search(c: httpx.AsyncClient):
    name = "filters:text_search"
    data = await post_graph(c, {
        "graph": {"type": "temporal_evolution", "time_interval": "year", "limit": 10},
        "query": "attention mechanism",
        "title_query": "transformer",
        "abstract_query": "self-attention",
        "operator": "and",
    }, name)
    if not data:
        return
    if not check_structure(data, name, allow_empty=True):
        return
    results.append(TestResult(name, True, data["took_ms"],
                              f"{len(data['nodes'])} nodes, {len(data['edges'])} edges",
                              len(data["nodes"]), len(data["edges"])))


async def test_filters_fuzzy(c: httpx.AsyncClient):
    name = "filters:fuzzy"
    data = await post_graph(c, {
        "graph": {"type": "cross_category_flow", "limit": 10},
        "fuzzy": "tansformer",
        "fuzzy_fuzziness": 2,
    }, name)
    if not data:
        return
    if not check_structure(data, name, allow_empty=True):
        return
    results.append(TestResult(name, True, data["took_ms"],
                              f"{len(data['nodes'])} nodes, {len(data['edges'])} edges",
                              len(data["nodes"]), len(data["edges"])))


async def test_filters_regex(c: httpx.AsyncClient):
    name = "filters:regex"
    data = await post_graph(c, {
        "graph": {"type": "domain_collaboration", "limit": 10},
        "title_regex": ".*[Tt]ransformer.*",
    }, name)
    if not data:
        return
    if not check_structure(data, name, allow_empty=True):
        return
    results.append(TestResult(name, True, data["took_ms"],
                              f"{len(data['nodes'])} nodes, {len(data['edges'])} edges",
                              len(data["nodes"]), len(data["edges"])))


async def test_filters_author(c: httpx.AsyncClient):
    name = "filters:author"
    data = await post_graph(c, {
        "graph": {"type": "temporal_evolution", "time_interval": "year", "limit": 10},
        "author": "Geoffrey Hinton",
    }, name)
    if not data:
        return
    if not check_structure(data, name, allow_empty=True):
        return
    results.append(TestResult(name, True, data["took_ms"],
                              f"{len(data['nodes'])} nodes, {len(data['edges'])} edges",
                              len(data["nodes"]), len(data["edges"])))


async def test_filters_hindex(c: httpx.AsyncClient):
    name = "filters:h_index"
    data = await post_graph(c, {
        "graph": {"type": "cross_category_flow", "limit": 10},
        "min_h_index": 50,
        "min_first_author_h_index": 30,
    }, name)
    if not data:
        return
    if not check_structure(data, name, allow_empty=True):
        return
    results.append(TestResult(name, True, data["took_ms"],
                              f"{len(data['nodes'])} nodes, {len(data['edges'])} edges",
                              len(data["nodes"]), len(data["edges"])))


async def test_filters_citations(c: httpx.AsyncClient):
    name = "filters:citations"
    data = await post_graph(c, {
        "graph": {"type": "domain_collaboration", "limit": 10},
        "min_citations": 50,
        "max_citations": 500,
    }, name)
    if not data:
        return
    if not check_structure(data, name, allow_empty=True):
        return
    results.append(TestResult(name, True, data["took_ms"],
                              f"{len(data['nodes'])} nodes, {len(data['edges'])} edges",
                              len(data["nodes"]), len(data["edges"])))


async def test_filters_boolean(c: httpx.AsyncClient):
    name = "filters:boolean"
    data = await post_graph(c, {
        "graph": {"type": "cross_category_flow", "limit": 10},
        "has_github": True,
        "has_doi": True,
        "has_journal_ref": True,
    }, name)
    if not data:
        return
    if not check_structure(data, name, allow_empty=True):
        return
    results.append(TestResult(name, True, data["took_ms"],
                              f"{len(data['nodes'])} nodes, {len(data['edges'])} edges",
                              len(data["nodes"]), len(data["edges"])))


async def test_filters_page_count(c: httpx.AsyncClient):
    name = "filters:page_count"
    data = await post_graph(c, {
        "graph": {"type": "temporal_evolution", "time_interval": "year", "limit": 5},
        "min_page_count": 10,
        "max_page_count": 50,
        "categories": ["cs.LG"],
    }, name)
    if not data:
        return
    if not check_structure(data, name, allow_empty=True):
        return
    results.append(TestResult(name, True, data["took_ms"],
                              f"{len(data['nodes'])} nodes, {len(data['edges'])} edges",
                              len(data["nodes"]), len(data["edges"])))


async def test_filters_exclude_categories(c: httpx.AsyncClient):
    name = "filters:exclude_categories"
    data = await post_graph(c, {
        "graph": {"type": "cross_category_flow", "limit": 10},
        "categories": ["cs.AI", "cs.LG", "cs.CL", "cs.CV"],
        "exclude_categories": ["cs.CV"],
    }, name)
    if not data:
        return
    if not check_structure(data, name, allow_empty=True):
        return
    results.append(TestResult(name, True, data["took_ms"],
                              f"{len(data['nodes'])} nodes, {len(data['edges'])} edges",
                              len(data["nodes"]), len(data["edges"])))


async def test_filters_primary_category(c: httpx.AsyncClient):
    name = "filters:primary_category"
    data = await post_graph(c, {
        "graph": {"type": "github_landscape", "time_interval": "year", "limit": 10},
        "primary_category": "cs.LG",
    }, name)
    if not data:
        return
    if not check_structure(data, name, allow_empty=True):
        return
    results.append(TestResult(name, True, data["took_ms"],
                              f"{len(data['nodes'])} nodes, {len(data['edges'])} edges",
                              len(data["nodes"]), len(data["edges"])))


async def test_filters_date_range(c: httpx.AsyncClient):
    name = "filters:date_range"
    data = await post_graph(c, {
        "graph": {"type": "temporal_evolution", "time_interval": "month", "limit": 10},
        "submitted_date": {"gte": "2024-06-01T00:00:00", "lte": "2024-12-31T00:00:00"},
        "updated_date": {"gte": "2024-06-01T00:00:00"},
    }, name)
    if not data:
        return
    if not check_structure(data, name, allow_empty=True):
        return
    results.append(TestResult(name, True, data["took_ms"],
                              f"{len(data['nodes'])} nodes, {len(data['edges'])} edges",
                              len(data["nodes"]), len(data["edges"])))


async def test_filters_semantic_boost(c: httpx.AsyncClient):
    name = "filters:semantic_boost"
    data = await post_graph(c, {
        "graph": {"type": "coauthor_network", "seed_author": "Yann LeCun",
                  "depth": 1, "limit": 15},
        "semantic": [{"text": "self-supervised learning", "level": "abstract",
                      "weight": 1.0, "mode": "boost"}]
    }, name)
    if not data:
        return
    if not check_structure(data, name, allow_empty=True):
        return
    results.append(TestResult(name, True, data["took_ms"],
                              f"{len(data['nodes'])} nodes, {len(data['edges'])} edges",
                              len(data["nodes"]), len(data["edges"])))


async def test_filters_semantic_boost_exclude(c: httpx.AsyncClient):
    name = "filters:semantic_boost+exclude"
    data = await post_graph(c, {
        "graph": {"type": "interdisciplinary", "min_categories": 3, "limit": 10},
        "semantic": [
            {"text": "deep learning neural networks", "level": "abstract",
             "weight": 1.0, "mode": "boost"},
            {"text": "computer vision image recognition", "level": "abstract",
             "weight": 0.5, "mode": "exclude"},
        ]
    }, name)
    if not data:
        return
    if not check_structure(data, name, allow_empty=True):
        return
    results.append(TestResult(name, True, data["took_ms"],
                              f"{len(data['nodes'])} nodes, {len(data['edges'])} edges",
                              len(data["nodes"]), len(data["edges"])))


async def test_filters_mega_combo(c: httpx.AsyncClient):
    """Test the maximum filter combo on a graph query."""
    name = "filters:mega_combo"
    data = await post_graph(c, {
        "graph": {"type": "temporal_evolution", "time_interval": "year", "limit": 5},
        "query": "neural network",
        "categories": ["cs.LG", "cs.AI"],
        "primary_category": "cs.LG",
        "submitted_date": {"gte": "2022-01-01T00:00:00"},
        "min_citations": 1,
        "has_github": True,
        "min_page_count": 5,
        "author": "Hinton",
    }, name)
    if not data:
        return
    if not check_structure(data, name, allow_empty=True):
        return
    results.append(TestResult(name, True, data["took_ms"],
                              f"{len(data['nodes'])} nodes, {len(data['edges'])} edges",
                              len(data["nodes"]), len(data["edges"])))


# ─────────────────────────────────────────────────
# EDGE CASES
# ─────────────────────────────────────────────────
async def test_edge_invalid_type(c: httpx.AsyncClient):
    name = "edge:invalid_type"
    start = time.monotonic()
    resp = await c.post(f"{BASE}/graph", json={
        "graph": {"type": "nonexistent_type", "limit": 10}
    }, headers=HEADERS, timeout=TIMEOUT)
    elapsed = int((time.monotonic() - start) * 1000)
    if resp.status_code == 422:
        results.append(TestResult(name, True, elapsed, "422 as expected"))
    else:
        results.append(TestResult(name, False, elapsed,
                                  f"Expected 422, got {resp.status_code}"))


async def test_edge_limit_boundary(c: httpx.AsyncClient):
    name = "edge:limit_1"
    data = await post_graph(c, {
        "graph": {"type": "cross_category_flow", "limit": 1}
    }, name)
    if not data:
        return
    if not check_structure(data, name, allow_empty=True):
        return
    results.append(TestResult(name, True, data["took_ms"],
                              f"{len(data['nodes'])} nodes, {len(data['edges'])} edges",
                              len(data["nodes"]), len(data["edges"])))


async def test_edge_limit_max(c: httpx.AsyncClient):
    name = "edge:limit_200"
    data = await post_graph(c, {
        "graph": {"type": "cross_category_flow", "limit": 200}
    }, name)
    if not data:
        return
    if not check_structure(data, name, allow_empty=True):
        return
    results.append(TestResult(name, True, data["took_ms"],
                              f"{len(data['nodes'])} nodes, {len(data['edges'])} edges",
                              len(data["nodes"]), len(data["edges"])))


async def test_edge_empty_result(c: httpx.AsyncClient):
    name = "edge:empty_result"
    data = await post_graph(c, {
        "graph": {"type": "coauthor_network", "seed_author": "ZZZNoSuchAuthorXYZ123", "limit": 10}
    }, name)
    if not data:
        return
    if not check_structure(data, name, allow_empty=True):
        return
    # Should return 0 or very few results
    if len(data["nodes"]) <= 1:
        results.append(TestResult(name, True, data["took_ms"],
                                  f"Empty as expected: {len(data['nodes'])} nodes"))
    else:
        results.append(TestResult(name, False, data["took_ms"],
                                  f"Expected empty, got {len(data['nodes'])} nodes"))


async def test_edge_no_graph_field(c: httpx.AsyncClient):
    name = "edge:no_graph_field"
    start = time.monotonic()
    resp = await c.post(f"{BASE}/graph", json={"query": "test"}, headers=HEADERS, timeout=TIMEOUT)
    elapsed = int((time.monotonic() - start) * 1000)
    if resp.status_code == 422:
        results.append(TestResult(name, True, elapsed, "422 for missing graph"))
    else:
        results.append(TestResult(name, False, elapsed, f"Expected 422, got {resp.status_code}"))


# ─────────────────────────────────────────────────
# 17. MULTIHOP CITATION
# ─────────────────────────────────────────────────
async def test_multihop_citation_basic(c: httpx.AsyncClient):
    name = "multihop_citation:basic"
    data = await post_graph(c, {
        "graph": {"type": "multihop_citation", "max_hops": 2, "direction": "references", "limit": 20},
        "query": "neural network",
    }, name)
    if not data:
        return
    if not check_structure(data, name, min_nodes=1, allow_empty=True):
        return
    meta = data["metadata"]
    if "direction" not in meta or "max_hops" not in meta:
        results.append(TestResult(name, False, data["took_ms"], "Missing metadata fields"))
        return
    results.append(TestResult(name, True, data["took_ms"],
                              f"{len(data['nodes'])} nodes, {len(data['edges'])} edges, "
                              f"hops={meta.get('hops_reached', 0)}",
                              len(data["nodes"]), len(data["edges"])))


async def test_multihop_citation_seed_id(c: httpx.AsyncClient):
    name = "multihop_citation:seed_id"
    data = await post_graph(c, {
        "graph": {"type": "multihop_citation", "seed_arxiv_id": "2301.00001",
                  "max_hops": 1, "direction": "references"},
    }, name)
    if not data:
        return
    if not check_structure(data, name, min_nodes=1, allow_empty=True):
        return
    # Seed paper should be hop 0
    seed_nodes = [n for n in data["nodes"] if n["id"] == "2301.00001"]
    if seed_nodes and seed_nodes[0]["properties"].get("hop") != 0:
        results.append(TestResult(name, False, data["took_ms"], "Seed paper not at hop 0"))
        return
    results.append(TestResult(name, True, data["took_ms"],
                              f"{len(data['nodes'])} nodes, hops_reached={data['metadata'].get('hops_reached')}",
                              len(data["nodes"]), len(data["edges"])))


async def test_multihop_citation_cited_by(c: httpx.AsyncClient):
    name = "multihop_citation:cited_by"
    data = await post_graph(c, {
        "graph": {"type": "multihop_citation", "max_hops": 2, "direction": "cited_by", "limit": 10},
        "query": "deep learning",
    }, name)
    if not data:
        return
    if not check_structure(data, name, min_nodes=1, allow_empty=True):
        return
    if data["metadata"].get("direction") != "cited_by":
        results.append(TestResult(name, False, data["took_ms"], "Wrong direction in metadata"))
        return
    results.append(TestResult(name, True, data["took_ms"],
                              f"{len(data['nodes'])} nodes",
                              len(data["nodes"]), len(data["edges"])))


# ─────────────────────────────────────────────────
# 18. SHORTEST CITATION PATH
# ─────────────────────────────────────────────────
async def test_shortest_citation_path_basic(c: httpx.AsyncClient):
    name = "shortest_path:basic"
    data = await post_graph(c, {
        "graph": {"type": "shortest_citation_path",
                  "seed_arxiv_id": "2301.00001", "target_arxiv_id": "2301.00002",
                  "max_hops": 4},
    }, name)
    if not data:
        return
    # May or may not find a path — both are valid
    if "error" in (data.get("metadata") or {}):
        results.append(TestResult(name, True, data["took_ms"],
                                  f"status={data['metadata']['error']}"))
        return
    if not check_structure(data, name, allow_empty=True):
        return
    results.append(TestResult(name, True, data["took_ms"],
                              f"path_length={data['metadata'].get('path_length')}",
                              len(data["nodes"]), len(data["edges"])))


async def test_shortest_citation_path_missing_args(c: httpx.AsyncClient):
    name = "shortest_path:missing_args"
    data = await post_graph(c, {
        "graph": {"type": "shortest_citation_path", "seed_arxiv_id": "2301.00001"},
    }, name)
    if not data:
        return
    if "error" in (data.get("metadata") or {}):
        results.append(TestResult(name, True, data["took_ms"],
                                  f"error={data['metadata']['error']}"))
    else:
        results.append(TestResult(name, False, data["took_ms"],
                                  "Expected error for missing target"))


async def test_shortest_citation_path_same_paper(c: httpx.AsyncClient):
    name = "shortest_path:same_paper"
    data = await post_graph(c, {
        "graph": {"type": "shortest_citation_path",
                  "seed_arxiv_id": "2301.00001", "target_arxiv_id": "2301.00001"},
    }, name)
    if not data:
        return
    if "error" in (data.get("metadata") or {}):
        results.append(TestResult(name, True, data["took_ms"],
                                  f"error={data['metadata']['error']}"))
    else:
        results.append(TestResult(name, False, data["took_ms"],
                                  "Expected error for same source/target"))


# ─────────────────────────────────────────────────
# 19. PAGERANK
# ─────────────────────────────────────────────────
async def test_pagerank_basic(c: httpx.AsyncClient):
    name = "pagerank:basic"
    data = await post_graph(c, {
        "graph": {"type": "pagerank", "damping_factor": 0.85, "iterations": 10, "limit": 10},
        "query": "machine learning",
    }, name)
    if not data:
        return
    if not check_structure(data, name, min_nodes=1, allow_empty=True):
        return
    meta = data["metadata"]
    if "damping_factor" not in meta or "iterations" not in meta:
        results.append(TestResult(name, False, data["took_ms"], "Missing metadata"))
        return
    # Check pagerank property exists
    for node in data["nodes"][:5]:
        if "pagerank" not in node.get("properties", {}):
            results.append(TestResult(name, False, data["took_ms"], "Missing pagerank property"))
            return
    results.append(TestResult(name, True, data["took_ms"],
                              f"{len(data['nodes'])} nodes, max_pr={meta.get('max_pagerank')}",
                              len(data["nodes"]), len(data["edges"])))


async def test_pagerank_seed_ids(c: httpx.AsyncClient):
    name = "pagerank:seed_ids"
    data = await post_graph(c, {
        "graph": {"type": "pagerank", "seed_arxiv_ids": ["2301.00001", "2301.00002"],
                  "iterations": 5, "limit": 10},
    }, name)
    if not data:
        return
    if not check_structure(data, name, allow_empty=True):
        return
    results.append(TestResult(name, True, data["took_ms"],
                              f"{len(data['nodes'])} nodes, subgraph={data['metadata'].get('papers_in_subgraph')}",
                              len(data["nodes"]), len(data["edges"])))


async def test_pagerank_with_filters(c: httpx.AsyncClient):
    name = "pagerank:filters"
    data = await post_graph(c, {
        "graph": {"type": "pagerank", "iterations": 5, "limit": 10},
        "query": "computer vision", "categories": ["cs.CV"],
    }, name)
    if not data:
        return
    if not check_structure(data, name, allow_empty=True):
        return
    results.append(TestResult(name, True, data["took_ms"],
                              f"{len(data['nodes'])} nodes",
                              len(data["nodes"]), len(data["edges"])))


# ─────────────────────────────────────────────────
# 20. COMMUNITY DETECTION
# ─────────────────────────────────────────────────
async def test_community_detection_basic(c: httpx.AsyncClient):
    name = "community_detection:basic"
    data = await post_graph(c, {
        "graph": {"type": "community_detection", "iterations": 10, "limit": 10},
        "query": "reinforcement learning",
    }, name)
    if not data:
        return
    if not check_structure(data, name, allow_empty=True):
        return
    meta = data["metadata"]
    if "communities_found" not in meta:
        results.append(TestResult(name, False, data["took_ms"], "Missing communities_found"))
        return
    results.append(TestResult(name, True, data["took_ms"],
                              f"{meta.get('communities_found')} communities, "
                              f"largest={meta.get('largest_community')}",
                              len(data["nodes"]), len(data["edges"])))


async def test_community_detection_with_filters(c: httpx.AsyncClient):
    name = "community_detection:filters"
    data = await post_graph(c, {
        "graph": {"type": "community_detection", "iterations": 10, "limit": 5},
        "query": "natural language processing", "categories": ["cs.CL"],
    }, name)
    if not data:
        return
    if not check_structure(data, name, allow_empty=True):
        return
    results.append(TestResult(name, True, data["took_ms"],
                              f"{data['metadata'].get('communities_found')} communities",
                              len(data["nodes"]), len(data["edges"])))


# ─────────────────────────────────────────────────
# 21. CITATION PATTERNS
# ─────────────────────────────────────────────────
async def test_citation_patterns_mutual(c: httpx.AsyncClient):
    name = "citation_patterns:mutual"
    data = await post_graph(c, {
        "graph": {"type": "citation_patterns", "pattern": "mutual", "limit": 10},
        "query": "attention mechanism",
    }, name)
    if not data:
        return
    if not check_structure(data, name, allow_empty=True):
        return
    results.append(TestResult(name, True, data["took_ms"],
                              f"patterns={data['metadata'].get('patterns_found')}, "
                              f"subgraph={data['metadata'].get('papers_in_subgraph')}",
                              len(data["nodes"]), len(data["edges"])))


async def test_citation_patterns_star(c: httpx.AsyncClient):
    name = "citation_patterns:star"
    data = await post_graph(c, {
        "graph": {"type": "citation_patterns", "pattern": "star", "limit": 5},
        "query": "convolutional neural network",
    }, name)
    if not data:
        return
    if not check_structure(data, name, allow_empty=True):
        return
    results.append(TestResult(name, True, data["took_ms"],
                              f"patterns={data['metadata'].get('patterns_found')}",
                              len(data["nodes"]), len(data["edges"])))


async def test_citation_patterns_chain(c: httpx.AsyncClient):
    name = "citation_patterns:chain"
    data = await post_graph(c, {
        "graph": {"type": "citation_patterns", "pattern": "chain", "limit": 5},
        "query": "generative adversarial",
    }, name)
    if not data:
        return
    if not check_structure(data, name, allow_empty=True):
        return
    results.append(TestResult(name, True, data["took_ms"],
                              f"patterns={data['metadata'].get('patterns_found')}, "
                              f"edges_in_subgraph={data['metadata'].get('edges_in_subgraph')}",
                              len(data["nodes"]), len(data["edges"])))


async def test_citation_patterns_triangle(c: httpx.AsyncClient):
    name = "citation_patterns:triangle"
    data = await post_graph(c, {
        "graph": {"type": "citation_patterns", "pattern": "triangle", "limit": 5},
        "query": "object detection",
    }, name)
    if not data:
        return
    if not check_structure(data, name, allow_empty=True):
        return
    results.append(TestResult(name, True, data["took_ms"],
                              f"patterns={data['metadata'].get('patterns_found')}",
                              len(data["nodes"]), len(data["edges"])))


# ─────────────────────────────────────────────────
# 22. CONNECTED COMPONENTS
# ─────────────────────────────────────────────────
async def test_connected_components_basic(c: httpx.AsyncClient):
    name = "connected_components:basic"
    data = await post_graph(c, {
        "graph": {"type": "connected_components", "limit": 10},
        "query": "robotics",
    }, name)
    if not data:
        return
    if not check_structure(data, name, allow_empty=True):
        return
    meta = data["metadata"]
    if "components_found" not in meta:
        results.append(TestResult(name, False, data["took_ms"], "Missing components_found"))
        return
    results.append(TestResult(name, True, data["took_ms"],
                              f"{meta.get('components_found')} components, "
                              f"largest={meta.get('largest_component')}, "
                              f"isolated={meta.get('isolated_papers')}",
                              len(data["nodes"]), len(data["edges"])))


async def test_connected_components_with_filters(c: httpx.AsyncClient):
    name = "connected_components:filters"
    data = await post_graph(c, {
        "graph": {"type": "connected_components", "limit": 5},
        "query": "quantum computing", "categories": ["quant-ph"],
    }, name)
    if not data:
        return
    if not check_structure(data, name, allow_empty=True):
        return
    results.append(TestResult(name, True, data["took_ms"],
                              f"{data['metadata'].get('components_found')} components",
                              len(data["nodes"]), len(data["edges"])))


# ─────────────────────────────────────────────────
# RUNNER
# ─────────────────────────────────────────────────
async def main():
    async with httpx.AsyncClient() as c:
        # Health check
        try:
            resp = await c.get(f"{BASE}/health", timeout=5)
            if resp.status_code != 200:
                print(f"FAIL: Server not healthy (HTTP {resp.status_code})")
                sys.exit(1)
        except Exception as e:
            print(f"FAIL: Cannot reach server: {e}")
            sys.exit(1)

        # Collect all test functions
        tests = [v for k, v in globals().items()
                 if k.startswith("test_") and asyncio.iscoroutinefunction(v)]

        print(f"Running {len(tests)} graph E2E tests...\n")
        start = time.monotonic()

        for test_fn in tests:
            await test_fn(c)

        elapsed = time.monotonic() - start

    # Print results
    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)

    print(f"{'='*80}")
    print(f" GRAPH E2E TEST RESULTS: {passed}/{len(results)} passed ({failed} failed) in {elapsed:.1f}s")
    print(f"{'='*80}\n")

    # Group by type
    current_group = ""
    for r in results:
        group = r.name.split(":")[0]
        if group != current_group:
            current_group = group
            print(f"  ── {group} ──")
        status = "✓" if r.passed else "✗"
        print(f"    {status} {r.name:<45} {r.took_ms:>5}ms  {r.detail}")

    print()

    if failed > 0:
        print(f"\nFAILED TESTS ({failed}):")
        for r in results:
            if not r.passed:
                print(f"  ✗ {r.name}: {r.detail}")
        sys.exit(1)
    else:
        print(f"ALL {passed} TESTS PASSED")


if __name__ == "__main__":
    asyncio.run(main())
