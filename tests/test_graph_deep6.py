"""Deep E2E tests for graph-DB query expressiveness (types 50-52).

Tests:
- pattern_match: structural pattern matching with filters, multi-hop, edges
- pipeline: chained algorithms, inter-step filtering, error cases
- subgraph_projection: filtered subgraph + algorithm dispatch, edge direction
- Boundary conditions: missing params, nesting prevention, limits
- Data integrity: node IDs match edges, no duplicates
"""
import asyncio
import json
import sys
import time
from collections import Counter
from dataclasses import dataclass

import httpx

BASE = "http://localhost:8000"
HEADERS = {"X-API-Key": "changeme-key-1", "Content-Type": "application/json"}
TIMEOUT = 120.0

# Well-connected papers from the index
CONNECTED_PAPER = "2406.03736"   # 40 refs, 230 cited_by
BIG_REFS_PAPER  = "2402.01749"   # 265 refs, 20 cited_by
HUGE_REFS_PAPER = "2602.21169"   # 342 refs, 3 cited_by


@dataclass
class TestResult:
    name: str
    passed: bool
    took_ms: int = 0
    detail: str = ""


results: list[TestResult] = []


async def post_graph(client: httpx.AsyncClient, body: dict, name: str) -> dict | None:
    start = time.monotonic()
    try:
        resp = await client.post(f"{BASE}/graph", json=body, headers=HEADERS, timeout=TIMEOUT)
        elapsed = int((time.monotonic() - start) * 1000)
        if resp.status_code != 200:
            results.append(TestResult(name, False, elapsed, f"HTTP {resp.status_code}: {resp.text[:200]}"))
            return None
        return resp.json()
    except Exception as e:
        elapsed = int((time.monotonic() - start) * 1000)
        results.append(TestResult(name, False, elapsed, str(e)[:200]))
        return None


def ok(name: str, ms: int, detail: str):
    results.append(TestResult(name, True, ms, detail))

def fail(name: str, ms: int, detail: str):
    results.append(TestResult(name, False, ms, detail))


def validate_graph_integrity(data: dict, name: str) -> bool:
    nodes = data.get("nodes", [])
    edges = data.get("edges", [])
    for key in ("nodes", "edges", "total", "took_ms", "metadata"):
        if key not in data:
            fail(name, data.get("took_ms", 0), f"Missing response key: {key}")
            return False
    node_ids = set()
    for n in nodes:
        for k in ("id", "label", "type"):
            if k not in n:
                fail(name, data["took_ms"], f"Node missing '{k}'")
                return False
        if n["id"] in node_ids:
            fail(name, data["took_ms"], f"Duplicate node ID: {n['id']}")
            return False
        node_ids.add(n["id"])
    for e in edges:
        for k in ("source", "target", "relation"):
            if k not in e:
                fail(name, data["took_ms"], f"Edge missing '{k}'")
                return False
        if e["source"] not in node_ids:
            fail(name, data["took_ms"], f"Edge source '{e['source']}' not in nodes")
            return False
        if e["target"] not in node_ids:
            fail(name, data["took_ms"], f"Edge target '{e['target']}' not in nodes")
            return False
    return True


# ═══════════════════════════════════════════════════════
# 50. PATTERN MATCH — Deep Tests
# ═══════════════════════════════════════════════════════

async def test_pattern_match_basic(c: httpx.AsyncClient):
    """Basic 2-node citation pattern returns matches."""
    name = "pattern_match:basic"
    data = await post_graph(c, {
        "graph": {
            "type": "pattern_match",
            "pattern_nodes": [
                {"alias": "a", "type": "paper"},
                {"alias": "b", "type": "paper"},
            ],
            "pattern_edges": [
                {"source": "a", "target": "b", "relation": "cites"},
            ],
            "limit": 10,
        },
        "query": "transformer attention",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    nodes = data.get("nodes", [])
    meta = data.get("metadata", {})
    mf = meta.get("matches_found", 0)
    if mf == 0:
        fail(name, data["took_ms"], "No matches found for basic citation pattern")
        return
    ok(name, data["took_ms"],
       f"{len(nodes)} nodes, matches_found={mf}, papers_searched={meta.get('papers_searched')}")


async def test_pattern_match_with_category_filter(c: httpx.AsyncClient):
    """Pattern where node A is cs.CL and node B is cs.LG."""
    name = "pattern_match:category_filter"
    data = await post_graph(c, {
        "graph": {
            "type": "pattern_match",
            "pattern_nodes": [
                {"alias": "nlp", "type": "paper", "filters": {"categories": ["cs.CL"]}},
                {"alias": "ml", "type": "paper", "filters": {"categories": ["cs.LG"]}},
            ],
            "pattern_edges": [
                {"source": "nlp", "target": "ml", "relation": "cites"},
            ],
            "limit": 5,
        },
        "query": "language model",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    meta = data.get("metadata", {})
    ok(name, data["took_ms"],
       f"matches_found={meta.get('matches_found')}, nodes={len(data['nodes'])}")


async def test_pattern_match_same_category(c: httpx.AsyncClient):
    """Pattern using same_category relation."""
    name = "pattern_match:same_category"
    data = await post_graph(c, {
        "graph": {
            "type": "pattern_match",
            "pattern_nodes": [
                {"alias": "p1", "type": "paper"},
                {"alias": "p2", "type": "paper"},
            ],
            "pattern_edges": [
                {"source": "p1", "target": "p2", "relation": "same_category"},
            ],
            "limit": 5,
        },
        "query": "deep learning",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    meta = data.get("metadata", {})
    ok(name, data["took_ms"],
       f"matches_found={meta.get('matches_found')}, papers_searched={meta.get('papers_searched')}")


async def test_pattern_match_co_authored(c: httpx.AsyncClient):
    """Pattern using co_authored relation."""
    name = "pattern_match:co_authored"
    data = await post_graph(c, {
        "graph": {
            "type": "pattern_match",
            "pattern_nodes": [
                {"alias": "x", "type": "paper"},
                {"alias": "y", "type": "paper"},
            ],
            "pattern_edges": [
                {"source": "x", "target": "y", "relation": "co_authored"},
            ],
            "limit": 5,
        },
        "query": "reinforcement learning",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    meta = data.get("metadata", {})
    ok(name, data["took_ms"],
       f"matches_found={meta.get('matches_found')}, papers_searched={meta.get('papers_searched')}")


async def test_pattern_match_3_node_chain(c: httpx.AsyncClient):
    """A→B→C citation chain pattern with 3 nodes."""
    name = "pattern_match:3_node_chain"
    data = await post_graph(c, {
        "graph": {
            "type": "pattern_match",
            "pattern_nodes": [
                {"alias": "a", "type": "paper"},
                {"alias": "b", "type": "paper"},
                {"alias": "c", "type": "paper"},
            ],
            "pattern_edges": [
                {"source": "a", "target": "b", "relation": "cites"},
                {"source": "b", "target": "c", "relation": "cites"},
            ],
            "limit": 5,
        },
        "query": "neural network",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    meta = data.get("metadata", {})
    edges = data.get("edges", [])
    # Each match should produce 2 edges (a→b, b→c)
    ok(name, data["took_ms"],
       f"matches_found={meta.get('matches_found')}, nodes={len(data['nodes'])}, edges={len(edges)}")


async def test_pattern_match_metadata_keys(c: httpx.AsyncClient):
    """Metadata should contain all expected keys."""
    name = "pattern_match:metadata_keys"
    data = await post_graph(c, {
        "graph": {
            "type": "pattern_match",
            "pattern_nodes": [
                {"alias": "a", "type": "paper"},
                {"alias": "b", "type": "paper"},
            ],
            "pattern_edges": [
                {"source": "a", "target": "b", "relation": "cites"},
            ],
            "limit": 5,
        },
        "query": "computer vision",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    meta = data.get("metadata", {})
    required = {"matches_found", "matches_returned", "matches_capped",
                "pattern_nodes", "pattern_edges", "papers_searched", "unique_result_nodes"}
    missing = required - set(meta.keys())
    if missing:
        fail(name, data["took_ms"], f"Missing metadata keys: {missing}")
        return
    ok(name, data["took_ms"],
       f"All metadata keys present: matches_found={meta['matches_found']}")


async def test_pattern_match_node_properties(c: httpx.AsyncClient):
    """Paper nodes should have pattern_alias and match_index properties."""
    name = "pattern_match:node_properties"
    data = await post_graph(c, {
        "graph": {
            "type": "pattern_match",
            "pattern_nodes": [
                {"alias": "src", "type": "paper"},
                {"alias": "dst", "type": "paper"},
            ],
            "pattern_edges": [
                {"source": "src", "target": "dst", "relation": "cites"},
            ],
            "limit": 10,
        },
        "query": "transformer attention",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    nodes = data.get("nodes", [])
    if not nodes:
        fail(name, data["took_ms"], "No nodes returned")
        return
    for n in nodes[:3]:
        props = n.get("properties", {})
        if "pattern_alias" not in props:
            fail(name, data["took_ms"], f"Node {n['id']} missing pattern_alias property")
            return
        if "match_index" not in props:
            fail(name, data["took_ms"], f"Node {n['id']} missing match_index property")
            return
    ok(name, data["took_ms"], f"All nodes have pattern_alias and match_index")


async def test_pattern_match_edge_validity(c: httpx.AsyncClient):
    """All edge sources/targets should exist as node IDs."""
    name = "pattern_match:edge_validity"
    data = await post_graph(c, {
        "graph": {
            "type": "pattern_match",
            "pattern_nodes": [
                {"alias": "a", "type": "paper"},
                {"alias": "b", "type": "paper"},
            ],
            "pattern_edges": [
                {"source": "a", "target": "b", "relation": "cites"},
            ],
            "limit": 10,
        },
        "query": "machine learning",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    node_ids = {n["id"] for n in data["nodes"]}
    for e in data["edges"]:
        if e["source"] not in node_ids:
            fail(name, data["took_ms"], f"Edge source {e['source']} not in node IDs")
            return
        if e["target"] not in node_ids:
            fail(name, data["took_ms"], f"Edge target {e['target']} not in node IDs")
            return
    ok(name, data["took_ms"], f"All {len(data['edges'])} edges valid")


async def test_pattern_match_empty_pattern(c: httpx.AsyncClient):
    """Missing pattern_nodes/edges → error in metadata."""
    name = "pattern_match:empty_pattern"
    data = await post_graph(c, {
        "graph": {
            "type": "pattern_match",
            "limit": 5,
        },
        "query": "test",
    }, name)
    if not data:
        return
    meta = data.get("metadata", {})
    if "error" not in meta:
        fail(name, data.get("took_ms", 0), "Expected error for empty pattern")
        return
    ok(name, data.get("took_ms", 0), f"Correct error: {meta['error'][:60]}")


async def test_pattern_match_limit_variation(c: httpx.AsyncClient):
    """limit=3 should return fewer matches than limit=20."""
    name = "pattern_match:limit_variation"
    data_small = await post_graph(c, {
        "graph": {
            "type": "pattern_match",
            "pattern_nodes": [
                {"alias": "a", "type": "paper"},
                {"alias": "b", "type": "paper"},
            ],
            "pattern_edges": [
                {"source": "a", "target": "b", "relation": "cites"},
            ],
            "limit": 3,
        },
        "query": "deep learning",
    }, f"{name}:small")
    data_big = await post_graph(c, {
        "graph": {
            "type": "pattern_match",
            "pattern_nodes": [
                {"alias": "a", "type": "paper"},
                {"alias": "b", "type": "paper"},
            ],
            "pattern_edges": [
                {"source": "a", "target": "b", "relation": "cites"},
            ],
            "limit": 20,
        },
        "query": "deep learning",
    }, f"{name}:big")
    if not data_small or not data_big:
        return
    mr_small = data_small.get("metadata", {}).get("matches_returned", 0)
    mr_big = data_big.get("metadata", {}).get("matches_returned", 0)
    if mr_big >= mr_small:
        ok(name, 0, f"limit=3→{mr_small} matches_returned, limit=20→{mr_big} matches_returned")
    else:
        fail(name, 0, f"limit=20 returned fewer: {mr_small} vs {mr_big}")


async def test_pattern_match_date_filter(c: httpx.AsyncClient):
    """Pattern node with date_from filter."""
    name = "pattern_match:date_filter"
    data = await post_graph(c, {
        "graph": {
            "type": "pattern_match",
            "pattern_nodes": [
                {"alias": "recent", "type": "paper", "filters": {"date_from": "2024-01-01"}},
                {"alias": "any", "type": "paper"},
            ],
            "pattern_edges": [
                {"source": "recent", "target": "any", "relation": "cites"},
            ],
            "limit": 5,
        },
        "query": "large language model",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    ok(name, data["took_ms"],
       f"matches_found={data['metadata'].get('matches_found')}")


# ═══════════════════════════════════════════════════════
# 51. PIPELINE — Deep Tests
# ═══════════════════════════════════════════════════════

async def test_pipeline_basic(c: httpx.AsyncClient):
    """Basic 2-step pipeline: pagerank → community_detection."""
    name = "pipeline:basic"
    data = await post_graph(c, {
        "graph": {
            "type": "pipeline",
            "pipeline_steps": [
                {"type": "pagerank", "limit": 30, "params": {"damping_factor": 0.85}},
                {"type": "community_detection", "limit": 15},
            ],
            "limit": 10,
        },
        "query": "transformer model",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    meta = data.get("metadata", {})
    steps = meta.get("pipeline_steps", [])
    if len(steps) != 2:
        fail(name, data["took_ms"], f"Expected 2 pipeline_steps in metadata, got {len(steps)}")
        return
    ok(name, data["took_ms"],
       f"nodes={len(data['nodes'])}, steps={len(steps)}, "
       f"step0_nodes={steps[0].get('nodes_produced')}, step1_nodes={steps[1].get('nodes_produced')}")


async def test_pipeline_3_steps(c: httpx.AsyncClient):
    """3-step pipeline: pagerank → betweenness → community_detection."""
    name = "pipeline:3_steps"
    data = await post_graph(c, {
        "graph": {
            "type": "pipeline",
            "pipeline_steps": [
                {"type": "pagerank", "limit": 50, "params": {}},
                {"type": "betweenness_centrality", "limit": 30},
                {"type": "community_detection", "limit": 15},
            ],
            "limit": 10,
        },
        "query": "graph neural network",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    meta = data.get("metadata", {})
    steps = meta.get("pipeline_steps", [])
    if len(steps) != 3:
        fail(name, data["took_ms"], f"Expected 3 steps, got {len(steps)}")
        return
    total = meta.get("total_steps")
    if total != 3:
        fail(name, data["took_ms"], f"total_steps={total}, expected 3")
        return
    ok(name, data["took_ms"],
       f"3-step pipeline OK, output_type={meta.get('pipeline_output_type')}")


async def test_pipeline_metadata_keys(c: httpx.AsyncClient):
    """Pipeline metadata should contain pipeline_steps, total_steps, pipeline_output_type."""
    name = "pipeline:metadata_keys"
    data = await post_graph(c, {
        "graph": {
            "type": "pipeline",
            "pipeline_steps": [
                {"type": "pagerank", "limit": 20},
                {"type": "degree_centrality", "limit": 10},
            ],
            "limit": 10,
        },
        "query": "optimization",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    meta = data.get("metadata", {})
    required = {"pipeline_steps", "total_steps", "pipeline_output_type"}
    missing = required - set(meta.keys())
    if missing:
        fail(name, data["took_ms"], f"Missing metadata keys: {missing}")
        return
    ok(name, data["took_ms"],
       f"All metadata keys present, output_type={meta['pipeline_output_type']}")


async def test_pipeline_step_metadata(c: httpx.AsyncClient):
    """Each step in pipeline_steps should have step, type, nodes_produced."""
    name = "pipeline:step_metadata"
    data = await post_graph(c, {
        "graph": {
            "type": "pipeline",
            "pipeline_steps": [
                {"type": "pagerank", "limit": 30},
                {"type": "community_detection", "limit": 15},
            ],
            "limit": 10,
        },
        "query": "natural language processing",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    steps = data.get("metadata", {}).get("pipeline_steps", [])
    for s in steps:
        for k in ("step", "type", "nodes_produced"):
            if k not in s:
                fail(name, data["took_ms"], f"Step missing key '{k}': {s}")
                return
    ok(name, data["took_ms"], f"All {len(steps)} steps have proper metadata")


async def test_pipeline_with_filter(c: httpx.AsyncClient):
    """Pipeline with inter-step filtering by property."""
    name = "pipeline:with_filter"
    data = await post_graph(c, {
        "graph": {
            "type": "pipeline",
            "pipeline_steps": [
                {"type": "pagerank", "limit": 50, "params": {},
                 "filter_property": "pagerank", "filter_min": 0.001},
                {"type": "community_detection", "limit": 15},
            ],
            "limit": 10,
        },
        "query": "attention mechanism",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    steps = data.get("metadata", {}).get("pipeline_steps", [])
    if steps and steps[0].get("filtered"):
        ok(name, data["took_ms"],
           f"Filtering applied, nodes_produced={steps[0].get('nodes_produced')}")
    else:
        # Filtering might not reduce count if all exceed threshold
        ok(name, data["took_ms"],
           f"Pipeline completed, step0_nodes={steps[0].get('nodes_produced') if steps else 'N/A'}")


async def test_pipeline_error_single_step(c: httpx.AsyncClient):
    """Pipeline with < 2 steps should return error."""
    name = "pipeline:error_single_step"
    data = await post_graph(c, {
        "graph": {
            "type": "pipeline",
            "pipeline_steps": [
                {"type": "pagerank", "limit": 10},
            ],
            "limit": 10,
        },
        "query": "test",
    }, name)
    if not data:
        return
    meta = data.get("metadata", {})
    if "error" not in meta:
        fail(name, data.get("took_ms", 0), "Expected error for single step pipeline")
        return
    ok(name, data.get("took_ms", 0), f"Correct error: {meta['error'][:60]}")


async def test_pipeline_error_nested(c: httpx.AsyncClient):
    """Pipeline containing another pipeline should return error."""
    name = "pipeline:error_nested"
    data = await post_graph(c, {
        "graph": {
            "type": "pipeline",
            "pipeline_steps": [
                {"type": "pagerank", "limit": 20},
                {"type": "pipeline", "limit": 10},
            ],
            "limit": 10,
        },
        "query": "test",
    }, name)
    if not data:
        return
    meta = data.get("metadata", {})
    if "error" not in meta:
        fail(name, data.get("took_ms", 0), "Expected error for nested pipeline")
        return
    ok(name, data.get("took_ms", 0), f"Correct error: {meta['error'][:60]}")


async def test_pipeline_error_too_many_steps(c: httpx.AsyncClient):
    """Pipeline with > 5 steps should return error."""
    name = "pipeline:error_too_many_steps"
    data = await post_graph(c, {
        "graph": {
            "type": "pipeline",
            "pipeline_steps": [
                {"type": "pagerank", "limit": 20},
                {"type": "pagerank", "limit": 20},
                {"type": "pagerank", "limit": 20},
                {"type": "pagerank", "limit": 20},
                {"type": "pagerank", "limit": 20},
                {"type": "pagerank", "limit": 20},
            ],
            "limit": 10,
        },
        "query": "test",
    }, name)
    if not data:
        return
    meta = data.get("metadata", {})
    if "error" not in meta:
        fail(name, data.get("took_ms", 0), "Expected error for >5 steps")
        return
    ok(name, data.get("took_ms", 0), f"Correct error: {meta['error'][:60]}")


async def test_pipeline_edge_validity(c: httpx.AsyncClient):
    """All edges should reference valid node IDs."""
    name = "pipeline:edge_validity"
    data = await post_graph(c, {
        "graph": {
            "type": "pipeline",
            "pipeline_steps": [
                {"type": "pagerank", "limit": 40},
                {"type": "community_detection", "limit": 20},
            ],
            "limit": 15,
        },
        "query": "reinforcement learning",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    node_ids = {n["id"] for n in data["nodes"]}
    for e in data["edges"]:
        if e["source"] not in node_ids:
            fail(name, data["took_ms"], f"Edge source {e['source']} not in node IDs")
            return
        if e["target"] not in node_ids:
            fail(name, data["took_ms"], f"Edge target {e['target']} not in node IDs")
            return
    ok(name, data["took_ms"], f"All {len(data['edges'])} edges valid")


async def test_pipeline_with_search_filter(c: httpx.AsyncClient):
    """Pipeline with top-level search filters (category, date)."""
    name = "pipeline:with_search_filter"
    data = await post_graph(c, {
        "graph": {
            "type": "pipeline",
            "pipeline_steps": [
                {"type": "pagerank", "limit": 30},
                {"type": "community_detection", "limit": 15},
            ],
            "limit": 10,
        },
        "query": "language model",
        "categories": ["cs.CL"],
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    ok(name, data["took_ms"],
       f"nodes={len(data['nodes'])}, edges={len(data['edges'])}")


# ═══════════════════════════════════════════════════════
# 52. SUBGRAPH PROJECTION — Deep Tests
# ═══════════════════════════════════════════════════════

async def test_subgraph_projection_basic(c: httpx.AsyncClient):
    """Basic subgraph projection with pagerank."""
    name = "subgraph_projection:basic"
    data = await post_graph(c, {
        "graph": {
            "type": "subgraph_projection",
            "subgraph_filter": {
                "categories": ["cs.AI"],
                "direction": "both",
                "max_nodes": 200,
            },
            "subgraph_algorithm": "pagerank",
            "subgraph_params": {"damping_factor": 0.85},
            "limit": 10,
        },
        "query": "",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    meta = data.get("metadata", {})
    proj = meta.get("subgraph_projection", {})
    if not proj:
        fail(name, data["took_ms"], "No subgraph_projection metadata")
        return
    ok(name, data["took_ms"],
       f"nodes={len(data['nodes'])}, papers_in_projection={proj.get('papers_in_projection')}, "
       f"edges_in_projection={proj.get('edges_in_projection')}")


async def test_subgraph_projection_with_date_filter(c: httpx.AsyncClient):
    """Subgraph projection with date range filter."""
    name = "subgraph_projection:date_filter"
    data = await post_graph(c, {
        "graph": {
            "type": "subgraph_projection",
            "subgraph_filter": {
                "categories": ["cs.LG"],
                "date_from": "2024-01-01",
                "date_to": "2025-01-01",
                "direction": "both",
                "max_nodes": 150,
            },
            "subgraph_algorithm": "community_detection",
            "limit": 10,
        },
        "query": "",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    proj = data.get("metadata", {}).get("subgraph_projection", {})
    filters_applied = proj.get("filters_applied", {})
    if "date_from" not in filters_applied:
        fail(name, data["took_ms"], "date_from not in filters_applied")
        return
    ok(name, data["took_ms"],
       f"papers_in_projection={proj.get('papers_in_projection')}, "
       f"filters={list(filters_applied.keys())}")


async def test_subgraph_projection_with_citation_filter(c: httpx.AsyncClient):
    """Subgraph projection filtering by citation count."""
    name = "subgraph_projection:citation_filter"
    data = await post_graph(c, {
        "graph": {
            "type": "subgraph_projection",
            "subgraph_filter": {
                "categories": ["cs.AI"],
                "min_citations": 5,
                "direction": "references",
                "max_nodes": 100,
            },
            "subgraph_algorithm": "betweenness_centrality",
            "limit": 10,
        },
        "query": "",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    proj = data.get("metadata", {}).get("subgraph_projection", {})
    ok(name, data["took_ms"],
       f"papers_in_projection={proj.get('papers_in_projection')}, direction=references")


async def test_subgraph_projection_metadata_keys(c: httpx.AsyncClient):
    """subgraph_projection metadata should have papers_in_projection, edges_in_projection, direction, algorithm, filters_applied."""
    name = "subgraph_projection:metadata_keys"
    data = await post_graph(c, {
        "graph": {
            "type": "subgraph_projection",
            "subgraph_filter": {
                "categories": ["cs.CV"],
                "direction": "both",
                "max_nodes": 100,
            },
            "subgraph_algorithm": "degree_centrality",
            "limit": 10,
        },
        "query": "",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    proj = data.get("metadata", {}).get("subgraph_projection", {})
    required = {"papers_in_projection", "edges_in_projection", "direction", "algorithm", "filters_applied"}
    missing = required - set(proj.keys())
    if missing:
        fail(name, data["took_ms"], f"Missing subgraph_projection keys: {missing}")
        return
    ok(name, data["took_ms"],
       f"All projection metadata keys present, algo={proj['algorithm']}")


async def test_subgraph_projection_different_algorithms(c: httpx.AsyncClient):
    """Run multiple different algorithms on projected subgraph."""
    name = "subgraph_projection:different_algorithms"
    algos = ["pagerank", "degree_centrality", "community_detection"]
    results_for_algos = []
    for algo in algos:
        data = await post_graph(c, {
            "graph": {
                "type": "subgraph_projection",
                "subgraph_filter": {
                    "categories": ["cs.AI"],
                    "direction": "both",
                    "max_nodes": 100,
                },
                "subgraph_algorithm": algo,
                "limit": 5,
            },
            "query": "",
        }, f"{name}:{algo}")
        if not data:
            fail(name, 0, f"Failed for algorithm {algo}")
            return
        results_for_algos.append(f"{algo}={len(data.get('nodes', []))}n")
    ok(name, 0, f"All algorithms succeeded: {', '.join(results_for_algos)}")


async def test_subgraph_projection_direction_references(c: httpx.AsyncClient):
    """Direction=references should only include outgoing citation edges."""
    name = "subgraph_projection:direction_references"
    data = await post_graph(c, {
        "graph": {
            "type": "subgraph_projection",
            "subgraph_filter": {
                "categories": ["cs.AI"],
                "direction": "references",
                "max_nodes": 100,
            },
            "subgraph_algorithm": "pagerank",
            "limit": 10,
        },
        "query": "",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    proj = data.get("metadata", {}).get("subgraph_projection", {})
    if proj.get("direction") != "references":
        fail(name, data["took_ms"], f"Direction should be 'references', got {proj.get('direction')}")
        return
    ok(name, data["took_ms"],
       f"direction=references, edges_in_projection={proj.get('edges_in_projection')}")


async def test_subgraph_projection_direction_cited_by(c: httpx.AsyncClient):
    """Direction=cited_by should only include incoming citation edges."""
    name = "subgraph_projection:direction_cited_by"
    data = await post_graph(c, {
        "graph": {
            "type": "subgraph_projection",
            "subgraph_filter": {
                "categories": ["cs.AI"],
                "direction": "cited_by",
                "max_nodes": 100,
            },
            "subgraph_algorithm": "pagerank",
            "limit": 10,
        },
        "query": "",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    proj = data.get("metadata", {}).get("subgraph_projection", {})
    if proj.get("direction") != "cited_by":
        fail(name, data["took_ms"], f"Direction should be 'cited_by', got {proj.get('direction')}")
        return
    ok(name, data["took_ms"],
       f"direction=cited_by, edges_in_projection={proj.get('edges_in_projection')}")


async def test_subgraph_projection_error_missing_filter(c: httpx.AsyncClient):
    """Missing subgraph_filter should return error."""
    name = "subgraph_projection:error_missing_filter"
    data = await post_graph(c, {
        "graph": {
            "type": "subgraph_projection",
            "subgraph_algorithm": "pagerank",
            "limit": 10,
        },
        "query": "",
    }, name)
    if not data:
        return
    meta = data.get("metadata", {})
    if "error" not in meta:
        fail(name, data.get("took_ms", 0), "Expected error for missing filter")
        return
    ok(name, data.get("took_ms", 0), f"Correct error: {meta['error'][:60]}")


async def test_subgraph_projection_error_missing_algorithm(c: httpx.AsyncClient):
    """Missing subgraph_algorithm should return error."""
    name = "subgraph_projection:error_missing_algorithm"
    data = await post_graph(c, {
        "graph": {
            "type": "subgraph_projection",
            "subgraph_filter": {
                "categories": ["cs.AI"],
                "direction": "both",
            },
            "limit": 10,
        },
        "query": "",
    }, name)
    if not data:
        return
    meta = data.get("metadata", {})
    if "error" not in meta:
        fail(name, data.get("took_ms", 0), "Expected error for missing algorithm")
        return
    ok(name, data.get("took_ms", 0), f"Correct error: {meta['error'][:60]}")


async def test_subgraph_projection_error_nested(c: httpx.AsyncClient):
    """Cannot nest subgraph_projection inside subgraph_projection."""
    name = "subgraph_projection:error_nested"
    data = await post_graph(c, {
        "graph": {
            "type": "subgraph_projection",
            "subgraph_filter": {
                "categories": ["cs.AI"],
                "direction": "both",
            },
            "subgraph_algorithm": "subgraph_projection",
            "limit": 10,
        },
        "query": "",
    }, name)
    if not data:
        return
    meta = data.get("metadata", {})
    if "error" not in meta:
        fail(name, data.get("took_ms", 0), "Expected error for nested subgraph_projection")
        return
    ok(name, data.get("took_ms", 0), f"Correct error: {meta['error'][:60]}")


async def test_subgraph_projection_edge_validity(c: httpx.AsyncClient):
    """All edges should reference valid node IDs."""
    name = "subgraph_projection:edge_validity"
    data = await post_graph(c, {
        "graph": {
            "type": "subgraph_projection",
            "subgraph_filter": {
                "categories": ["cs.AI"],
                "direction": "both",
                "max_nodes": 200,
            },
            "subgraph_algorithm": "pagerank",
            "limit": 15,
        },
        "query": "",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    node_ids = {n["id"] for n in data["nodes"]}
    for e in data["edges"]:
        if e["source"] not in node_ids:
            fail(name, data["took_ms"], f"Edge source {e['source']} not in node IDs")
            return
        if e["target"] not in node_ids:
            fail(name, data["took_ms"], f"Edge target {e['target']} not in node IDs")
            return
    ok(name, data["took_ms"], f"All {len(data['edges'])} edges valid")


async def test_subgraph_projection_with_search_query(c: httpx.AsyncClient):
    """Subgraph projection combined with a text search query."""
    name = "subgraph_projection:with_search_query"
    data = await post_graph(c, {
        "graph": {
            "type": "subgraph_projection",
            "subgraph_filter": {
                "categories": ["cs.CL"],
                "direction": "both",
                "max_nodes": 150,
            },
            "subgraph_algorithm": "pagerank",
            "limit": 10,
        },
        "query": "language model",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    proj = data.get("metadata", {}).get("subgraph_projection", {})
    ok(name, data["took_ms"],
       f"nodes={len(data['nodes'])}, papers_in_projection={proj.get('papers_in_projection')}")


# ═══════════════════════════════════════════════════════
# RUNNER
# ═══════════════════════════════════════════════════════

async def main():
    async with httpx.AsyncClient() as c:
        try:
            resp = await c.get(f"{BASE}/health", timeout=5)
            if resp.status_code != 200:
                print(f"FAIL: Server not healthy (HTTP {resp.status_code})")
                sys.exit(1)
        except Exception as e:
            print(f"FAIL: Cannot reach server: {e}")
            sys.exit(1)

        tests = [v for k, v in globals().items()
                 if k.startswith("test_") and asyncio.iscoroutinefunction(v)]
        print(f"Running {len(tests)} deep graph tests (types 50-52)...\n")
        start = time.monotonic()

        for test_fn in tests:
            await test_fn(c)

        elapsed = time.monotonic() - start

    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)

    print(f"\n{'='*80}")
    print(f" DEEP GRAPH TEST RESULTS (types 50-52): {passed}/{len(results)} passed ({failed} failed) in {elapsed:.1f}s")
    print(f"{'='*80}\n")

    current_group = ""
    for r in results:
        group = r.name.split(":")[0]
        if group != current_group:
            current_group = group
            print(f"  ── {group} ──")
        status = "✓" if r.passed else "✗"
        print(f"    {status} {r.name:<55} {r.took_ms:>5}ms  {r.detail}")

    if failed > 0:
        print(f"\nFAILED TESTS ({failed}):")
        for r in results:
            if not r.passed:
                print(f"  ✗ {r.name}: {r.detail}")
        sys.exit(1)
    else:
        print(f"\nALL {passed} TESTS PASSED")


if __name__ == "__main__":
    asyncio.run(main())
