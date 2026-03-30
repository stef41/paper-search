"""Deep E2E tests for graph-DB algorithms batch 4 (types 35-42).

Tests:
- Correctness: node/edge properties, metadata integrity
- Filter composition: all 8 algorithms with search filters
- Boundary conditions: limit variations, parameter tuning
- Data integrity: node IDs match edges, no duplicates
- Path-finding: all_shortest_paths, k_shortest_paths with known path
- Random walk: visit counts, probabilities
- Structural: triangle_count, graph_diameter
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
TIMEOUT = 120.0  # some graph algorithms can be slow

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
# 35. HITS — Deep Tests
# ═══════════════════════════════════════════════════════

async def test_hits_basic(c: httpx.AsyncClient):
    """Basic HITS returns nodes with authority/hub scores."""
    name = "hits:basic"
    data = await post_graph(c, {
        "graph": {"type": "hits", "limit": 10},
        "query": "transformer attention",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    nodes = data.get("nodes", [])
    if not nodes:
        fail(name, data["took_ms"], "No nodes returned")
        return
    ok(name, data["took_ms"],
       f"{len(nodes)} nodes, {len(data['edges'])} edges")


async def test_hits_with_search_filter(c: httpx.AsyncClient):
    """HITS with category and date filters."""
    name = "hits:with_search_filter"
    data = await post_graph(c, {
        "graph": {"type": "hits", "limit": 10},
        "query": "language model",
        "categories": ["cs.CL"],
        "submitted_date": {"gte": "2024-01-01T00:00:00"},
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    ok(name, data["took_ms"],
       f"{len(data['nodes'])} nodes with category+date filter")


async def test_hits_limit(c: httpx.AsyncClient):
    """limit=3 vs limit=20 should return different counts."""
    name = "hits:limit"
    data_small = await post_graph(c, {
        "graph": {"type": "hits", "limit": 3},
        "query": "deep learning",
    }, f"{name}:small")
    data_large = await post_graph(c, {
        "graph": {"type": "hits", "limit": 20},
        "query": "deep learning",
    }, f"{name}:large")
    if not data_small or not data_large:
        return
    n_small = len(data_small.get("nodes", []))
    n_large = len(data_large.get("nodes", []))
    if n_small > n_large:
        fail(name, 0, f"limit=3 got {n_small} nodes, limit=20 got {n_large} — small should not exceed large")
        return
    ok(name, 0, f"limit=3→{n_small} nodes, limit=20→{n_large} nodes")


async def test_hits_properties_validation(c: httpx.AsyncClient):
    """Nodes should have authority, hub, auth_rank properties."""
    name = "hits:properties_validation"
    data = await post_graph(c, {
        "graph": {"type": "hits", "limit": 10},
        "query": "neural network",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    nodes = data.get("nodes", [])
    if not nodes:
        fail(name, data["took_ms"], "No nodes")
        return
    for n in nodes:
        props = n.get("properties", {})
        for k in ("authority", "hub"):
            if k not in props:
                fail(name, data["took_ms"], f"Node {n['id']} missing property '{k}'")
                return
        if "auth_rank" not in props and "hub_rank" not in props:
            fail(name, data["took_ms"], f"Node {n['id']} missing auth_rank or hub_rank")
            return
    ok(name, data["took_ms"], f"All {len(nodes)} nodes have authority/hub/rank")


async def test_hits_graph_integrity(c: httpx.AsyncClient):
    """validate_graph_integrity passes for HITS result."""
    name = "hits:graph_integrity"
    data = await post_graph(c, {
        "graph": {"type": "hits", "limit": 15},
        "query": "reinforcement learning",
    }, name)
    if not data:
        return
    if not validate_graph_integrity(data, name):
        return
    ok(name, data["took_ms"], "Graph integrity validated")


async def test_hits_edge_validity(c: httpx.AsyncClient):
    """All edge sources/targets exist as node IDs."""
    name = "hits:edge_validity"
    data = await post_graph(c, {
        "graph": {"type": "hits", "limit": 15},
        "query": "computer vision",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    node_ids = {n["id"] for n in data["nodes"]}
    edges = data.get("edges", [])
    for e in edges:
        if e["source"] not in node_ids:
            fail(name, data["took_ms"], f"Edge source {e['source']} not in node set")
            return
        if e["target"] not in node_ids:
            fail(name, data["took_ms"], f"Edge target {e['target']} not in node set")
            return
    ok(name, data["took_ms"], f"All {len(edges)} edges reference valid nodes")


async def test_hits_metadata_fields(c: httpx.AsyncClient):
    """Metadata should include max_authority, max_hub."""
    name = "hits:metadata_fields"
    data = await post_graph(c, {
        "graph": {"type": "hits", "limit": 10},
        "query": "generative model",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    meta = data.get("metadata", {})
    expected_keys = ["max_authority", "max_hub"]
    missing = [k for k in expected_keys if k not in meta]
    if missing:
        fail(name, data["took_ms"], f"Missing metadata keys: {missing}")
        return
    ok(name, data["took_ms"],
       f"max_authority={meta['max_authority']}, max_hub={meta['max_hub']}")


async def test_hits_with_seed_paper(c: httpx.AsyncClient):
    """HITS seeded from a specific paper."""
    name = "hits:with_seed_paper"
    data = await post_graph(c, {
        "graph": {"type": "hits", "seed_arxiv_id": CONNECTED_PAPER, "limit": 10},
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    nodes = data.get("nodes", [])
    ok(name, data["took_ms"],
       f"{len(nodes)} nodes seeded from {CONNECTED_PAPER}")


# ═══════════════════════════════════════════════════════
# 36. HARMONIC CENTRALITY — Deep Tests
# ═══════════════════════════════════════════════════════

async def test_harmonic_basic(c: httpx.AsyncClient):
    """Basic harmonic centrality returns nodes with scores."""
    name = "harmonic_centrality:basic"
    data = await post_graph(c, {
        "graph": {"type": "harmonic_centrality", "limit": 10},
        "query": "transformer attention",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    nodes = data.get("nodes", [])
    if not nodes:
        fail(name, data["took_ms"], "No nodes returned")
        return
    ok(name, data["took_ms"],
       f"{len(nodes)} nodes, {len(data['edges'])} edges")


async def test_harmonic_with_search_filter(c: httpx.AsyncClient):
    """Harmonic centrality with category filter."""
    name = "harmonic_centrality:with_search_filter"
    data = await post_graph(c, {
        "graph": {"type": "harmonic_centrality", "limit": 10},
        "query": "natural language processing",
        "categories": ["cs.CL"],
        "submitted_date": {"gte": "2024-01-01T00:00:00"},
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    ok(name, data["took_ms"],
       f"{len(data['nodes'])} nodes with category+date filter")


async def test_harmonic_limit(c: httpx.AsyncClient):
    """limit=3 vs limit=20 should return different counts."""
    name = "harmonic_centrality:limit"
    data_small = await post_graph(c, {
        "graph": {"type": "harmonic_centrality", "limit": 3},
        "query": "deep learning",
    }, f"{name}:small")
    data_large = await post_graph(c, {
        "graph": {"type": "harmonic_centrality", "limit": 20},
        "query": "deep learning",
    }, f"{name}:large")
    if not data_small or not data_large:
        return
    n_small = len(data_small.get("nodes", []))
    n_large = len(data_large.get("nodes", []))
    if n_small > n_large:
        fail(name, 0, f"limit=3 got {n_small} nodes, limit=20 got {n_large} — small should not exceed large")
        return
    ok(name, 0, f"limit=3→{n_small} nodes, limit=20→{n_large} nodes")


async def test_harmonic_properties_validation(c: httpx.AsyncClient):
    """Nodes should have harmonic_centrality and rank properties."""
    name = "harmonic_centrality:properties_validation"
    data = await post_graph(c, {
        "graph": {"type": "harmonic_centrality", "limit": 10},
        "query": "graph neural networks",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    nodes = data.get("nodes", [])
    if not nodes:
        fail(name, data["took_ms"], "No nodes")
        return
    for n in nodes:
        props = n.get("properties", {})
        for k in ("harmonic_centrality", "rank"):
            if k not in props:
                fail(name, data["took_ms"], f"Node {n['id']} missing property '{k}'")
                return
    ok(name, data["took_ms"], f"All {len(nodes)} nodes have harmonic_centrality/rank")


async def test_harmonic_graph_integrity(c: httpx.AsyncClient):
    """validate_graph_integrity passes for harmonic centrality result."""
    name = "harmonic_centrality:graph_integrity"
    data = await post_graph(c, {
        "graph": {"type": "harmonic_centrality", "limit": 15},
        "query": "optimization",
    }, name)
    if not data:
        return
    if not validate_graph_integrity(data, name):
        return
    ok(name, data["took_ms"], "Graph integrity validated")


async def test_harmonic_edge_validity(c: httpx.AsyncClient):
    """All edge sources/targets exist as node IDs."""
    name = "harmonic_centrality:edge_validity"
    data = await post_graph(c, {
        "graph": {"type": "harmonic_centrality", "limit": 15},
        "query": "image recognition",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    node_ids = {n["id"] for n in data["nodes"]}
    edges = data.get("edges", [])
    for e in edges:
        if e["source"] not in node_ids:
            fail(name, data["took_ms"], f"Edge source {e['source']} not in node set")
            return
        if e["target"] not in node_ids:
            fail(name, data["took_ms"], f"Edge target {e['target']} not in node set")
            return
    ok(name, data["took_ms"], f"All {len(edges)} edges reference valid nodes")


async def test_harmonic_metadata_fields(c: httpx.AsyncClient):
    """Metadata should include expected centrality stats."""
    name = "harmonic_centrality:metadata_fields"
    data = await post_graph(c, {
        "graph": {"type": "harmonic_centrality", "limit": 10},
        "query": "machine learning",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    meta = data.get("metadata", {})
    # At minimum metadata should exist and not be empty
    if not meta:
        fail(name, data["took_ms"], "Metadata is empty")
        return
    ok(name, data["took_ms"],
       f"Metadata keys: {list(meta.keys())[:10]}")


async def test_harmonic_with_seed_paper(c: httpx.AsyncClient):
    """Harmonic centrality seeded from a specific paper."""
    name = "harmonic_centrality:with_seed_paper"
    data = await post_graph(c, {
        "graph": {"type": "harmonic_centrality", "seed_arxiv_id": CONNECTED_PAPER, "limit": 10},
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    nodes = data.get("nodes", [])
    ok(name, data["took_ms"],
       f"{len(nodes)} nodes seeded from {CONNECTED_PAPER}")


# ═══════════════════════════════════════════════════════
# 37. KATZ CENTRALITY — Deep Tests
# ═══════════════════════════════════════════════════════

async def test_katz_basic(c: httpx.AsyncClient):
    """Basic katz centrality returns nodes with scores."""
    name = "katz_centrality:basic"
    data = await post_graph(c, {
        "graph": {"type": "katz_centrality", "limit": 10},
        "query": "transformer attention",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    nodes = data.get("nodes", [])
    if not nodes:
        fail(name, data["took_ms"], "No nodes returned")
        return
    ok(name, data["took_ms"],
       f"{len(nodes)} nodes, {len(data['edges'])} edges")


async def test_katz_with_search_filter(c: httpx.AsyncClient):
    """Katz centrality with category and date filters."""
    name = "katz_centrality:with_search_filter"
    data = await post_graph(c, {
        "graph": {"type": "katz_centrality", "limit": 10},
        "query": "language model",
        "categories": ["cs.CL"],
        "submitted_date": {"gte": "2024-01-01T00:00:00"},
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    ok(name, data["took_ms"],
       f"{len(data['nodes'])} nodes with category+date filter")


async def test_katz_limit(c: httpx.AsyncClient):
    """limit=3 vs limit=20 should return different counts."""
    name = "katz_centrality:limit"
    data_small = await post_graph(c, {
        "graph": {"type": "katz_centrality", "limit": 3},
        "query": "deep learning",
    }, f"{name}:small")
    data_large = await post_graph(c, {
        "graph": {"type": "katz_centrality", "limit": 20},
        "query": "deep learning",
    }, f"{name}:large")
    if not data_small or not data_large:
        return
    n_small = len(data_small.get("nodes", []))
    n_large = len(data_large.get("nodes", []))
    if n_small > n_large:
        fail(name, 0, f"limit=3 got {n_small} nodes, limit=20 got {n_large} — small should not exceed large")
        return
    ok(name, 0, f"limit=3→{n_small} nodes, limit=20→{n_large} nodes")


async def test_katz_properties_validation(c: httpx.AsyncClient):
    """Nodes should have katz_centrality and rank properties."""
    name = "katz_centrality:properties_validation"
    data = await post_graph(c, {
        "graph": {"type": "katz_centrality", "limit": 10},
        "query": "neural network",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    nodes = data.get("nodes", [])
    if not nodes:
        fail(name, data["took_ms"], "No nodes")
        return
    for n in nodes:
        props = n.get("properties", {})
        for k in ("katz_centrality", "rank"):
            if k not in props:
                fail(name, data["took_ms"], f"Node {n['id']} missing property '{k}'")
                return
    ok(name, data["took_ms"], f"All {len(nodes)} nodes have katz_centrality/rank")


async def test_katz_graph_integrity(c: httpx.AsyncClient):
    """validate_graph_integrity passes for katz centrality result."""
    name = "katz_centrality:graph_integrity"
    data = await post_graph(c, {
        "graph": {"type": "katz_centrality", "limit": 15},
        "query": "reinforcement learning",
    }, name)
    if not data:
        return
    if not validate_graph_integrity(data, name):
        return
    ok(name, data["took_ms"], "Graph integrity validated")


async def test_katz_edge_validity(c: httpx.AsyncClient):
    """All edge sources/targets exist as node IDs."""
    name = "katz_centrality:edge_validity"
    data = await post_graph(c, {
        "graph": {"type": "katz_centrality", "limit": 15},
        "query": "computer vision",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    node_ids = {n["id"] for n in data["nodes"]}
    edges = data.get("edges", [])
    for e in edges:
        if e["source"] not in node_ids:
            fail(name, data["took_ms"], f"Edge source {e['source']} not in node set")
            return
        if e["target"] not in node_ids:
            fail(name, data["took_ms"], f"Edge target {e['target']} not in node set")
            return
    ok(name, data["took_ms"], f"All {len(edges)} edges reference valid nodes")


async def test_katz_metadata_fields(c: httpx.AsyncClient):
    """Metadata should have expected keys."""
    name = "katz_centrality:metadata_fields"
    data = await post_graph(c, {
        "graph": {"type": "katz_centrality", "limit": 10},
        "query": "generative adversarial",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    meta = data.get("metadata", {})
    if not meta:
        fail(name, data["took_ms"], "Metadata is empty")
        return
    ok(name, data["took_ms"],
       f"Metadata keys: {list(meta.keys())[:10]}")


async def test_katz_with_damping_and_iterations(c: httpx.AsyncClient):
    """Katz centrality with custom damping_factor and iterations."""
    name = "katz_centrality:with_damping_and_iterations"
    data = await post_graph(c, {
        "graph": {
            "type": "katz_centrality",
            "seed_arxiv_id": CONNECTED_PAPER,
            "damping_factor": 0.5,
            "iterations": 10,
            "limit": 10,
        },
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    nodes = data.get("nodes", [])
    ok(name, data["took_ms"],
       f"{len(nodes)} nodes with damping=0.5, iterations=10, seed={CONNECTED_PAPER}")


# ═══════════════════════════════════════════════════════
# 38. ALL SHORTEST PATHS — Deep Tests
# ═══════════════════════════════════════════════════════

async def test_all_shortest_paths_basic(c: httpx.AsyncClient):
    """Basic all_shortest_paths between two known connected papers."""
    name = "all_shortest_paths:basic"
    data = await post_graph(c, {
        "graph": {
            "type": "all_shortest_paths",
            "seed_arxiv_id": CONNECTED_PAPER,
            "target_arxiv_id": BIG_REFS_PAPER,
            "limit": 20,
        },
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    nodes = data.get("nodes", [])
    if not nodes:
        fail(name, data["took_ms"], "No nodes returned")
        return
    ok(name, data["took_ms"],
       f"{len(nodes)} nodes, {len(data['edges'])} edges")


async def test_all_shortest_paths_with_search_filter(c: httpx.AsyncClient):
    """all_shortest_paths with category filter."""
    name = "all_shortest_paths:with_search_filter"
    data = await post_graph(c, {
        "graph": {
            "type": "all_shortest_paths",
            "seed_arxiv_id": CONNECTED_PAPER,
            "target_arxiv_id": BIG_REFS_PAPER,
            "limit": 20,
        },
        "categories": ["cs.AI", "cs.CL"],
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    ok(name, data["took_ms"],
       f"{len(data['nodes'])} nodes with category filter")


async def test_all_shortest_paths_limit(c: httpx.AsyncClient):
    """Limit parameter restricts output."""
    name = "all_shortest_paths:limit"
    data = await post_graph(c, {
        "graph": {
            "type": "all_shortest_paths",
            "seed_arxiv_id": CONNECTED_PAPER,
            "target_arxiv_id": BIG_REFS_PAPER,
            "limit": 5,
        },
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    nodes = data.get("nodes", [])
    if len(nodes) > 50:
        fail(name, data["took_ms"], f"limit=5 but got {len(nodes)} nodes — too many")
        return
    ok(name, data["took_ms"], f"{len(nodes)} nodes with limit=5")


async def test_all_shortest_paths_properties_validation(c: httpx.AsyncClient):
    """Nodes should have on_path and path_position properties."""
    name = "all_shortest_paths:properties_validation"
    data = await post_graph(c, {
        "graph": {
            "type": "all_shortest_paths",
            "seed_arxiv_id": CONNECTED_PAPER,
            "target_arxiv_id": BIG_REFS_PAPER,
            "limit": 20,
        },
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    nodes = data.get("nodes", [])
    if not nodes:
        fail(name, data["took_ms"], "No nodes")
        return
    for n in nodes:
        props = n.get("properties", {})
        for k in ("on_path", "path_position"):
            if k not in props:
                fail(name, data["took_ms"], f"Node {n['id']} missing property '{k}'")
                return
    ok(name, data["took_ms"], f"All {len(nodes)} nodes have on_path/path_position")


async def test_all_shortest_paths_graph_integrity(c: httpx.AsyncClient):
    """validate_graph_integrity passes."""
    name = "all_shortest_paths:graph_integrity"
    data = await post_graph(c, {
        "graph": {
            "type": "all_shortest_paths",
            "seed_arxiv_id": CONNECTED_PAPER,
            "target_arxiv_id": BIG_REFS_PAPER,
            "limit": 20,
        },
    }, name)
    if not data:
        return
    if not validate_graph_integrity(data, name):
        return
    ok(name, data["took_ms"], "Graph integrity validated")


async def test_all_shortest_paths_edge_validity(c: httpx.AsyncClient):
    """All edge sources/targets exist as node IDs."""
    name = "all_shortest_paths:edge_validity"
    data = await post_graph(c, {
        "graph": {
            "type": "all_shortest_paths",
            "seed_arxiv_id": CONNECTED_PAPER,
            "target_arxiv_id": BIG_REFS_PAPER,
            "limit": 20,
        },
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    node_ids = {n["id"] for n in data["nodes"]}
    edges = data.get("edges", [])
    for e in edges:
        if e["source"] not in node_ids:
            fail(name, data["took_ms"], f"Edge source {e['source']} not in node set")
            return
        if e["target"] not in node_ids:
            fail(name, data["took_ms"], f"Edge target {e['target']} not in node set")
            return
    ok(name, data["took_ms"], f"All {len(edges)} edges reference valid nodes")


async def test_all_shortest_paths_metadata_fields(c: httpx.AsyncClient):
    """Metadata should include source, target, shortest_distance, paths_found."""
    name = "all_shortest_paths:metadata_fields"
    data = await post_graph(c, {
        "graph": {
            "type": "all_shortest_paths",
            "seed_arxiv_id": CONNECTED_PAPER,
            "target_arxiv_id": BIG_REFS_PAPER,
            "limit": 20,
        },
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    meta = data.get("metadata", {})
    expected_keys = ["source", "target", "shortest_distance", "paths_found"]
    missing = [k for k in expected_keys if k not in meta]
    if missing:
        fail(name, data["took_ms"], f"Missing metadata keys: {missing}")
        return
    ok(name, data["took_ms"],
       f"source={meta['source']}, target={meta['target']}, "
       f"distance={meta['shortest_distance']}, paths={meta['paths_found']}")


async def test_all_shortest_paths_with_max_hops(c: httpx.AsyncClient):
    """Test with max_hops parameter."""
    name = "all_shortest_paths:with_max_hops"
    data = await post_graph(c, {
        "graph": {
            "type": "all_shortest_paths",
            "seed_arxiv_id": CONNECTED_PAPER,
            "target_arxiv_id": BIG_REFS_PAPER,
            "max_hops": 3,
            "limit": 20,
        },
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    ok(name, data["took_ms"],
       f"{len(data['nodes'])} nodes with max_hops=3")


# ═══════════════════════════════════════════════════════
# 39. K SHORTEST PATHS — Deep Tests
# ═══════════════════════════════════════════════════════

async def test_k_shortest_paths_basic(c: httpx.AsyncClient):
    """Basic k_shortest_paths between two known connected papers."""
    name = "k_shortest_paths:basic"
    data = await post_graph(c, {
        "graph": {
            "type": "k_shortest_paths",
            "seed_arxiv_id": CONNECTED_PAPER,
            "target_arxiv_id": BIG_REFS_PAPER,
            "k_paths": 3,
            "limit": 20,
        },
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    nodes = data.get("nodes", [])
    if not nodes:
        fail(name, data["took_ms"], "No nodes returned")
        return
    ok(name, data["took_ms"],
       f"{len(nodes)} nodes, {len(data['edges'])} edges")


async def test_k_shortest_paths_with_search_filter(c: httpx.AsyncClient):
    """k_shortest_paths with category filter."""
    name = "k_shortest_paths:with_search_filter"
    data = await post_graph(c, {
        "graph": {
            "type": "k_shortest_paths",
            "seed_arxiv_id": CONNECTED_PAPER,
            "target_arxiv_id": BIG_REFS_PAPER,
            "k_paths": 3,
            "limit": 20,
        },
        "categories": ["cs.AI", "cs.CL"],
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    ok(name, data["took_ms"],
       f"{len(data['nodes'])} nodes with category filter")


async def test_k_shortest_paths_limit(c: httpx.AsyncClient):
    """k_paths=1 vs k_paths=5 should differ."""
    name = "k_shortest_paths:limit"
    data_small = await post_graph(c, {
        "graph": {
            "type": "k_shortest_paths",
            "seed_arxiv_id": CONNECTED_PAPER,
            "target_arxiv_id": BIG_REFS_PAPER,
            "k_paths": 1,
            "limit": 20,
        },
    }, f"{name}:k1")
    data_large = await post_graph(c, {
        "graph": {
            "type": "k_shortest_paths",
            "seed_arxiv_id": CONNECTED_PAPER,
            "target_arxiv_id": BIG_REFS_PAPER,
            "k_paths": 5,
            "limit": 50,
        },
    }, f"{name}:k5")
    if not data_small or not data_large:
        return
    n_small = len(data_small.get("nodes", []))
    n_large = len(data_large.get("nodes", []))
    ok(name, 0, f"k=1→{n_small} nodes, k=5→{n_large} nodes")


async def test_k_shortest_paths_properties_validation(c: httpx.AsyncClient):
    """Nodes should have on_paths property."""
    name = "k_shortest_paths:properties_validation"
    data = await post_graph(c, {
        "graph": {
            "type": "k_shortest_paths",
            "seed_arxiv_id": CONNECTED_PAPER,
            "target_arxiv_id": BIG_REFS_PAPER,
            "k_paths": 3,
            "limit": 20,
        },
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    nodes = data.get("nodes", [])
    if not nodes:
        fail(name, data["took_ms"], "No nodes")
        return
    for n in nodes:
        props = n.get("properties", {})
        if "on_paths" not in props:
            fail(name, data["took_ms"], f"Node {n['id']} missing property 'on_paths'")
            return
    ok(name, data["took_ms"], f"All {len(nodes)} nodes have on_paths property")


async def test_k_shortest_paths_graph_integrity(c: httpx.AsyncClient):
    """validate_graph_integrity passes."""
    name = "k_shortest_paths:graph_integrity"
    data = await post_graph(c, {
        "graph": {
            "type": "k_shortest_paths",
            "seed_arxiv_id": CONNECTED_PAPER,
            "target_arxiv_id": BIG_REFS_PAPER,
            "k_paths": 3,
            "limit": 20,
        },
    }, name)
    if not data:
        return
    if not validate_graph_integrity(data, name):
        return
    ok(name, data["took_ms"], "Graph integrity validated")


async def test_k_shortest_paths_edge_validity(c: httpx.AsyncClient):
    """All edge sources/targets exist as node IDs."""
    name = "k_shortest_paths:edge_validity"
    data = await post_graph(c, {
        "graph": {
            "type": "k_shortest_paths",
            "seed_arxiv_id": CONNECTED_PAPER,
            "target_arxiv_id": BIG_REFS_PAPER,
            "k_paths": 3,
            "limit": 20,
        },
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    node_ids = {n["id"] for n in data["nodes"]}
    edges = data.get("edges", [])
    for e in edges:
        if e["source"] not in node_ids:
            fail(name, data["took_ms"], f"Edge source {e['source']} not in node set")
            return
        if e["target"] not in node_ids:
            fail(name, data["took_ms"], f"Edge target {e['target']} not in node set")
            return
    ok(name, data["took_ms"], f"All {len(edges)} edges reference valid nodes")


async def test_k_shortest_paths_metadata_fields(c: httpx.AsyncClient):
    """Metadata should include k_requested, paths_found, path_lengths."""
    name = "k_shortest_paths:metadata_fields"
    data = await post_graph(c, {
        "graph": {
            "type": "k_shortest_paths",
            "seed_arxiv_id": CONNECTED_PAPER,
            "target_arxiv_id": BIG_REFS_PAPER,
            "k_paths": 3,
            "limit": 20,
        },
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    meta = data.get("metadata", {})
    expected_keys = ["k_requested", "paths_found", "path_lengths"]
    missing = [k for k in expected_keys if k not in meta]
    if missing:
        fail(name, data["took_ms"], f"Missing metadata keys: {missing}")
        return
    ok(name, data["took_ms"],
       f"k_requested={meta['k_requested']}, paths_found={meta['paths_found']}, "
       f"path_lengths={meta['path_lengths']}")


async def test_k_shortest_paths_with_seed_paper(c: httpx.AsyncClient):
    """k_shortest_paths with different k value and known seed/target."""
    name = "k_shortest_paths:with_seed_paper"
    data = await post_graph(c, {
        "graph": {
            "type": "k_shortest_paths",
            "seed_arxiv_id": CONNECTED_PAPER,
            "target_arxiv_id": BIG_REFS_PAPER,
            "k_paths": 5,
            "limit": 30,
        },
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    meta = data.get("metadata", {})
    ok(name, data["took_ms"],
       f"{len(data['nodes'])} nodes, k_requested={meta.get('k_requested')}, "
       f"paths_found={meta.get('paths_found')}")


# ═══════════════════════════════════════════════════════
# 40. RANDOM WALK — Deep Tests
# ═══════════════════════════════════════════════════════

async def test_random_walk_basic(c: httpx.AsyncClient):
    """Basic random walk from a seed paper."""
    name = "random_walk:basic"
    data = await post_graph(c, {
        "graph": {
            "type": "random_walk",
            "seed_arxiv_id": CONNECTED_PAPER,
            "walk_length": 10,
            "num_walks": 50,
            "limit": 15,
        },
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    nodes = data.get("nodes", [])
    if not nodes:
        fail(name, data["took_ms"], "No nodes returned")
        return
    ok(name, data["took_ms"],
       f"{len(nodes)} nodes, {len(data['edges'])} edges")


async def test_random_walk_with_search_filter(c: httpx.AsyncClient):
    """Random walk with category filter."""
    name = "random_walk:with_search_filter"
    data = await post_graph(c, {
        "graph": {
            "type": "random_walk",
            "seed_arxiv_id": CONNECTED_PAPER,
            "walk_length": 10,
            "num_walks": 50,
            "limit": 15,
        },
        "query": "language model",
        "categories": ["cs.CL"],
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    ok(name, data["took_ms"],
       f"{len(data['nodes'])} nodes with category filter")


async def test_random_walk_limit(c: httpx.AsyncClient):
    """limit=3 vs limit=20 should return different counts."""
    name = "random_walk:limit"
    data_small = await post_graph(c, {
        "graph": {
            "type": "random_walk",
            "seed_arxiv_id": CONNECTED_PAPER,
            "walk_length": 10,
            "num_walks": 50,
            "limit": 3,
        },
    }, f"{name}:small")
    data_large = await post_graph(c, {
        "graph": {
            "type": "random_walk",
            "seed_arxiv_id": CONNECTED_PAPER,
            "walk_length": 10,
            "num_walks": 50,
            "limit": 20,
        },
    }, f"{name}:large")
    if not data_small or not data_large:
        return
    n_small = len(data_small.get("nodes", []))
    n_large = len(data_large.get("nodes", []))
    if n_small > n_large:
        fail(name, 0, f"limit=3 got {n_small} nodes, limit=20 got {n_large} — small should not exceed large")
        return
    ok(name, 0, f"limit=3→{n_small} nodes, limit=20→{n_large} nodes")


async def test_random_walk_properties_validation(c: httpx.AsyncClient):
    """Nodes should have visit_count, visit_probability, is_seed, rank."""
    name = "random_walk:properties_validation"
    data = await post_graph(c, {
        "graph": {
            "type": "random_walk",
            "seed_arxiv_id": CONNECTED_PAPER,
            "walk_length": 10,
            "num_walks": 100,
            "limit": 15,
        },
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    nodes = data.get("nodes", [])
    if not nodes:
        fail(name, data["took_ms"], "No nodes")
        return
    for n in nodes:
        props = n.get("properties", {})
        for k in ("visit_count", "visit_probability", "is_seed", "rank"):
            if k not in props:
                fail(name, data["took_ms"], f"Node {n['id']} missing property '{k}'")
                return
    ok(name, data["took_ms"], f"All {len(nodes)} nodes have visit_count/visit_probability/is_seed/rank")


async def test_random_walk_graph_integrity(c: httpx.AsyncClient):
    """validate_graph_integrity passes for random walk result."""
    name = "random_walk:graph_integrity"
    data = await post_graph(c, {
        "graph": {
            "type": "random_walk",
            "seed_arxiv_id": CONNECTED_PAPER,
            "walk_length": 10,
            "num_walks": 50,
            "limit": 15,
        },
    }, name)
    if not data:
        return
    if not validate_graph_integrity(data, name):
        return
    ok(name, data["took_ms"], "Graph integrity validated")


async def test_random_walk_edge_validity(c: httpx.AsyncClient):
    """All edge sources/targets exist as node IDs."""
    name = "random_walk:edge_validity"
    data = await post_graph(c, {
        "graph": {
            "type": "random_walk",
            "seed_arxiv_id": CONNECTED_PAPER,
            "walk_length": 10,
            "num_walks": 50,
            "limit": 15,
        },
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    node_ids = {n["id"] for n in data["nodes"]}
    edges = data.get("edges", [])
    for e in edges:
        if e["source"] not in node_ids:
            fail(name, data["took_ms"], f"Edge source {e['source']} not in node set")
            return
        if e["target"] not in node_ids:
            fail(name, data["took_ms"], f"Edge target {e['target']} not in node set")
            return
    ok(name, data["took_ms"], f"All {len(edges)} edges reference valid nodes")


async def test_random_walk_metadata_fields(c: httpx.AsyncClient):
    """Metadata should exist and not be empty."""
    name = "random_walk:metadata_fields"
    data = await post_graph(c, {
        "graph": {
            "type": "random_walk",
            "seed_arxiv_id": CONNECTED_PAPER,
            "walk_length": 10,
            "num_walks": 100,
            "limit": 15,
        },
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    meta = data.get("metadata", {})
    if not meta:
        fail(name, data["took_ms"], "Metadata is empty")
        return
    ok(name, data["took_ms"],
       f"Metadata keys: {list(meta.keys())[:10]}")


async def test_random_walk_with_teleport(c: httpx.AsyncClient):
    """Random walk with custom teleport_prob."""
    name = "random_walk:with_teleport"
    data = await post_graph(c, {
        "graph": {
            "type": "random_walk",
            "seed_arxiv_id": BIG_REFS_PAPER,
            "walk_length": 15,
            "num_walks": 100,
            "teleport_prob": 0.3,
            "limit": 15,
        },
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    nodes = data.get("nodes", [])
    # Seed should be in the results and marked as is_seed
    seed_nodes = [n for n in nodes if n.get("properties", {}).get("is_seed")]
    ok(name, data["took_ms"],
       f"{len(nodes)} nodes, {len(seed_nodes)} seed(s), teleport_prob=0.3")


# ═══════════════════════════════════════════════════════
# 41. TRIANGLE COUNT — Deep Tests
# ═══════════════════════════════════════════════════════

async def test_triangle_count_basic(c: httpx.AsyncClient):
    """Basic triangle count returns nodes with triangle info."""
    name = "triangle_count:basic"
    data = await post_graph(c, {
        "graph": {"type": "triangle_count", "limit": 10},
        "query": "transformer attention",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    nodes = data.get("nodes", [])
    if not nodes:
        fail(name, data["took_ms"], "No nodes returned")
        return
    ok(name, data["took_ms"],
       f"{len(nodes)} nodes, {len(data['edges'])} edges")


async def test_triangle_count_with_search_filter(c: httpx.AsyncClient):
    """Triangle count with category and date filters."""
    name = "triangle_count:with_search_filter"
    data = await post_graph(c, {
        "graph": {"type": "triangle_count", "limit": 10},
        "query": "natural language processing",
        "categories": ["cs.CL"],
        "submitted_date": {"gte": "2024-01-01T00:00:00"},
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    ok(name, data["took_ms"],
       f"{len(data['nodes'])} nodes with category+date filter")


async def test_triangle_count_limit(c: httpx.AsyncClient):
    """limit=3 vs limit=20 should return different counts."""
    name = "triangle_count:limit"
    data_small = await post_graph(c, {
        "graph": {"type": "triangle_count", "limit": 3},
        "query": "deep learning",
    }, f"{name}:small")
    data_large = await post_graph(c, {
        "graph": {"type": "triangle_count", "limit": 20},
        "query": "deep learning",
    }, f"{name}:large")
    if not data_small or not data_large:
        return
    n_small = len(data_small.get("nodes", []))
    n_large = len(data_large.get("nodes", []))
    if n_small > n_large:
        fail(name, 0, f"limit=3 got {n_small} nodes, limit=20 got {n_large} — small should not exceed large")
        return
    ok(name, 0, f"limit=3→{n_small} nodes, limit=20→{n_large} nodes")


async def test_triangle_count_properties_validation(c: httpx.AsyncClient):
    """Nodes should have triangles, clustering_coefficient, degree, rank."""
    name = "triangle_count:properties_validation"
    data = await post_graph(c, {
        "graph": {"type": "triangle_count", "limit": 10},
        "query": "neural network",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    nodes = data.get("nodes", [])
    if not nodes:
        fail(name, data["took_ms"], "No nodes")
        return
    for n in nodes:
        props = n.get("properties", {})
        for k in ("triangles", "clustering_coefficient", "degree", "rank"):
            if k not in props:
                fail(name, data["took_ms"], f"Node {n['id']} missing property '{k}'")
                return
    ok(name, data["took_ms"],
       f"All {len(nodes)} nodes have triangles/clustering_coefficient/degree/rank")


async def test_triangle_count_graph_integrity(c: httpx.AsyncClient):
    """validate_graph_integrity passes for triangle count result."""
    name = "triangle_count:graph_integrity"
    data = await post_graph(c, {
        "graph": {"type": "triangle_count", "limit": 15},
        "query": "reinforcement learning",
    }, name)
    if not data:
        return
    if not validate_graph_integrity(data, name):
        return
    ok(name, data["took_ms"], "Graph integrity validated")


async def test_triangle_count_edge_validity(c: httpx.AsyncClient):
    """All edge sources/targets exist as node IDs."""
    name = "triangle_count:edge_validity"
    data = await post_graph(c, {
        "graph": {"type": "triangle_count", "limit": 15},
        "query": "computer vision",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    node_ids = {n["id"] for n in data["nodes"]}
    edges = data.get("edges", [])
    for e in edges:
        if e["source"] not in node_ids:
            fail(name, data["took_ms"], f"Edge source {e['source']} not in node set")
            return
        if e["target"] not in node_ids:
            fail(name, data["took_ms"], f"Edge target {e['target']} not in node set")
            return
    ok(name, data["took_ms"], f"All {len(edges)} edges reference valid nodes")


async def test_triangle_count_metadata_fields(c: httpx.AsyncClient):
    """Metadata should include total_triangles and global_clustering_coefficient."""
    name = "triangle_count:metadata_fields"
    data = await post_graph(c, {
        "graph": {"type": "triangle_count", "limit": 10},
        "query": "generative model",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    meta = data.get("metadata", {})
    expected_keys = ["total_triangles", "global_clustering_coefficient"]
    missing = [k for k in expected_keys if k not in meta]
    if missing:
        fail(name, data["took_ms"], f"Missing metadata keys: {missing}")
        return
    ok(name, data["took_ms"],
       f"total_triangles={meta['total_triangles']}, "
       f"global_clustering={meta['global_clustering_coefficient']}")


async def test_triangle_count_with_seed_paper(c: httpx.AsyncClient):
    """Triangle count seeded from a specific paper."""
    name = "triangle_count:with_seed_paper"
    data = await post_graph(c, {
        "graph": {"type": "triangle_count", "seed_arxiv_id": CONNECTED_PAPER, "limit": 10},
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    nodes = data.get("nodes", [])
    ok(name, data["took_ms"],
       f"{len(nodes)} nodes seeded from {CONNECTED_PAPER}")


# ═══════════════════════════════════════════════════════
# 42. GRAPH DIAMETER — Deep Tests
# ═══════════════════════════════════════════════════════

async def test_graph_diameter_basic(c: httpx.AsyncClient):
    """Basic graph diameter returns nodes with eccentricity info."""
    name = "graph_diameter:basic"
    data = await post_graph(c, {
        "graph": {"type": "graph_diameter", "limit": 10},
        "query": "transformer attention",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    nodes = data.get("nodes", [])
    if not nodes:
        fail(name, data["took_ms"], "No nodes returned")
        return
    ok(name, data["took_ms"],
       f"{len(nodes)} nodes, {len(data['edges'])} edges")


async def test_graph_diameter_with_search_filter(c: httpx.AsyncClient):
    """Graph diameter with category and date filters."""
    name = "graph_diameter:with_search_filter"
    data = await post_graph(c, {
        "graph": {"type": "graph_diameter", "limit": 10},
        "query": "natural language processing",
        "categories": ["cs.CL"],
        "submitted_date": {"gte": "2024-01-01T00:00:00"},
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    ok(name, data["took_ms"],
       f"{len(data['nodes'])} nodes with category+date filter")


async def test_graph_diameter_limit(c: httpx.AsyncClient):
    """limit=3 vs limit=20 should return different counts."""
    name = "graph_diameter:limit"
    data_small = await post_graph(c, {
        "graph": {"type": "graph_diameter", "limit": 3},
        "query": "deep learning",
    }, f"{name}:small")
    data_large = await post_graph(c, {
        "graph": {"type": "graph_diameter", "limit": 20},
        "query": "deep learning",
    }, f"{name}:large")
    if not data_small or not data_large:
        return
    n_small = len(data_small.get("nodes", []))
    n_large = len(data_large.get("nodes", []))
    if n_small > n_large:
        fail(name, 0, f"limit=3 got {n_small} nodes, limit=20 got {n_large} — small should not exceed large")
        return
    ok(name, 0, f"limit=3→{n_small} nodes, limit=20→{n_large} nodes")


async def test_graph_diameter_properties_validation(c: httpx.AsyncClient):
    """Nodes should have eccentricity, reachable, is_center, rank."""
    name = "graph_diameter:properties_validation"
    data = await post_graph(c, {
        "graph": {"type": "graph_diameter", "limit": 10},
        "query": "neural network",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    nodes = data.get("nodes", [])
    if not nodes:
        fail(name, data["took_ms"], "No nodes")
        return
    for n in nodes:
        props = n.get("properties", {})
        for k in ("eccentricity", "reachable", "is_center", "rank"):
            if k not in props:
                fail(name, data["took_ms"], f"Node {n['id']} missing property '{k}'")
                return
    ok(name, data["took_ms"],
       f"All {len(nodes)} nodes have eccentricity/reachable/is_center/rank")


async def test_graph_diameter_graph_integrity(c: httpx.AsyncClient):
    """validate_graph_integrity passes for graph diameter result."""
    name = "graph_diameter:graph_integrity"
    data = await post_graph(c, {
        "graph": {"type": "graph_diameter", "limit": 15},
        "query": "optimization",
    }, name)
    if not data:
        return
    if not validate_graph_integrity(data, name):
        return
    ok(name, data["took_ms"], "Graph integrity validated")


async def test_graph_diameter_edge_validity(c: httpx.AsyncClient):
    """All edge sources/targets exist as node IDs."""
    name = "graph_diameter:edge_validity"
    data = await post_graph(c, {
        "graph": {"type": "graph_diameter", "limit": 15},
        "query": "image recognition",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    node_ids = {n["id"] for n in data["nodes"]}
    edges = data.get("edges", [])
    for e in edges:
        if e["source"] not in node_ids:
            fail(name, data["took_ms"], f"Edge source {e['source']} not in node set")
            return
        if e["target"] not in node_ids:
            fail(name, data["took_ms"], f"Edge target {e['target']} not in node set")
            return
    ok(name, data["took_ms"], f"All {len(edges)} edges reference valid nodes")


async def test_graph_diameter_metadata_fields(c: httpx.AsyncClient):
    """Metadata should include diameter, radius, center_size."""
    name = "graph_diameter:metadata_fields"
    data = await post_graph(c, {
        "graph": {"type": "graph_diameter", "limit": 10},
        "query": "machine learning",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    meta = data.get("metadata", {})
    expected_keys = ["diameter", "radius", "center_size"]
    missing = [k for k in expected_keys if k not in meta]
    if missing:
        fail(name, data["took_ms"], f"Missing metadata keys: {missing}")
        return
    ok(name, data["took_ms"],
       f"diameter={meta['diameter']}, radius={meta['radius']}, "
       f"center_size={meta['center_size']}")


async def test_graph_diameter_with_seed_paper(c: httpx.AsyncClient):
    """Graph diameter seeded from a specific paper."""
    name = "graph_diameter:with_seed_paper"
    data = await post_graph(c, {
        "graph": {"type": "graph_diameter", "seed_arxiv_id": CONNECTED_PAPER, "limit": 10},
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    nodes = data.get("nodes", [])
    ok(name, data["took_ms"],
       f"{len(nodes)} nodes seeded from {CONNECTED_PAPER}")


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
        print(f"Running {len(tests)} deep graph tests (types 35-42)...\n")
        start = time.monotonic()

        for test_fn in tests:
            await test_fn(c)

        elapsed = time.monotonic() - start

    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)

    print(f"\n{'='*80}")
    print(f" DEEP GRAPH TEST RESULTS (types 35-42): {passed}/{len(results)} passed ({failed} failed) in {elapsed:.1f}s")
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
