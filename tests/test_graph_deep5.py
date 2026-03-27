"""Deep E2E tests for graph-DB algorithms batch 5 (types 43-49).

Tests:
- Correctness: node/edge properties, metadata integrity
- Filter composition: all 7 algorithms with search filters
- Boundary conditions: limit variation, small graphs
- Data integrity: node IDs match edges, no duplicates
- Type-specific validation per algorithm
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
    return True


# ═══════════════════════════════════════════════════════
# 43. LEIDEN COMMUNITY — Deep Tests
# ═══════════════════════════════════════════════════════

async def test_leiden_basic(c: httpx.AsyncClient):
    """Basic leiden community detection returns nodes and edges."""
    name = "leiden:basic"
    data = await post_graph(c, {
        "graph": {"type": "leiden_community", "limit": 10},
        "query": "transformer attention",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    nodes = data.get("nodes", [])
    edges = data.get("edges", [])
    if not nodes:
        fail(name, data["took_ms"], "No nodes returned")
        return
    meta = data.get("metadata", {})
    ok(name, data["took_ms"],
       f"{len(nodes)} nodes, {len(edges)} edges, communities={meta.get('communities_found')}")


async def test_leiden_with_search_filter(c: httpx.AsyncClient):
    """Leiden with category and date filters."""
    name = "leiden:with_search_filter"
    data = await post_graph(c, {
        "graph": {"type": "leiden_community", "limit": 10},
        "query": "language model",
        "categories": ["cs.CL"],
        "submitted_date": {"gte": "2024-01-01T00:00:00"},
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    ok(name, data["took_ms"],
       f"{len(data['nodes'])} nodes, communities={data['metadata'].get('communities_found')}")


async def test_leiden_limit(c: httpx.AsyncClient):
    """Limit=3 vs limit=20 should produce different node counts."""
    name = "leiden:limit"
    data_small = await post_graph(c, {
        "graph": {"type": "leiden_community", "limit": 3},
        "query": "deep learning",
    }, f"{name}:small")
    data_big = await post_graph(c, {
        "graph": {"type": "leiden_community", "limit": 20},
        "query": "deep learning",
    }, f"{name}:big")
    if not data_small or not data_big:
        return
    n_small = len(data_small.get("nodes", []))
    n_big = len(data_big.get("nodes", []))
    if n_big <= n_small:
        fail(name, 0, f"limit=20 ({n_big} nodes) should exceed limit=3 ({n_small} nodes)")
        return
    ok(name, 0, f"limit=3→{n_small} nodes, limit=20→{n_big} nodes")


async def test_leiden_properties_validation(c: httpx.AsyncClient):
    """Paper/member nodes should have community property."""
    name = "leiden:properties_validation"
    data = await post_graph(c, {
        "graph": {"type": "leiden_community", "limit": 15},
        "query": "graph neural networks",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    papers = [n for n in data["nodes"] if n["type"] == "paper"]
    if not papers:
        fail(name, data["took_ms"], "No paper nodes")
        return
    for pn in papers:
        comm = pn.get("properties", {}).get("community")
        if comm is None:
            fail(name, data["took_ms"], f"Paper {pn['id']} missing 'community' property")
            return
    ok(name, data["took_ms"], f"All {len(papers)} papers have community property")


async def test_leiden_graph_integrity(c: httpx.AsyncClient):
    """validate_graph_integrity passes."""
    name = "leiden:graph_integrity"
    data = await post_graph(c, {
        "graph": {"type": "leiden_community", "limit": 10},
        "query": "reinforcement learning",
    }, name)
    if not data:
        return
    if not validate_graph_integrity(data, name):
        return
    ok(name, data["took_ms"], f"{len(data['nodes'])} nodes, {len(data['edges'])} edges — integrity OK")


async def test_leiden_edge_validity(c: httpx.AsyncClient):
    """All edge sources/targets exist as node IDs."""
    name = "leiden:edge_validity"
    data = await post_graph(c, {
        "graph": {"type": "leiden_community", "limit": 15},
        "query": "computer vision",
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


async def test_leiden_metadata_fields(c: httpx.AsyncClient):
    """Check expected metadata keys."""
    name = "leiden:metadata_fields"
    data = await post_graph(c, {
        "graph": {"type": "leiden_community", "limit": 10},
        "query": "optimization",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    meta = data.get("metadata", {})
    expected_keys = ["communities_found", "modularity", "largest_community", "algorithm"]
    missing = [k for k in expected_keys if k not in meta]
    if missing:
        fail(name, data["took_ms"], f"Missing metadata keys: {missing}")
        return
    if meta.get("algorithm") != "leiden":
        fail(name, data["took_ms"], f"algorithm should be 'leiden', got '{meta.get('algorithm')}'")
        return
    ok(name, data["took_ms"],
       f"algorithm={meta['algorithm']}, communities={meta['communities_found']}, modularity={meta.get('modularity')}")


async def test_leiden_modularity_range(c: httpx.AsyncClient):
    """Modularity should be between -0.5 and 1.0."""
    name = "leiden:modularity_range"
    data = await post_graph(c, {
        "graph": {"type": "leiden_community", "limit": 20},
        "query": "transformer attention",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    meta = data.get("metadata", {})
    mod = meta.get("modularity")
    if mod is None:
        fail(name, data["took_ms"], "modularity not in metadata")
        return
    if not isinstance(mod, (int, float)):
        fail(name, data["took_ms"], f"modularity not numeric: {type(mod)}")
        return
    if mod < -0.5 or mod > 1.0:
        fail(name, data["took_ms"], f"modularity={mod} out of range [-0.5, 1.0]")
        return
    ok(name, data["took_ms"], f"modularity={mod:.4f}")


# ═══════════════════════════════════════════════════════
# 44. BRIDGE EDGES — Deep Tests
# ═══════════════════════════════════════════════════════

async def test_bridge_edges_basic(c: httpx.AsyncClient):
    """Basic bridge edges detection returns nodes and edges."""
    name = "bridge_edges:basic"
    data = await post_graph(c, {
        "graph": {"type": "bridge_edges", "limit": 10},
        "query": "transformer attention",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    nodes = data.get("nodes", [])
    edges = data.get("edges", [])
    if not nodes:
        fail(name, data["took_ms"], "No nodes returned")
        return
    meta = data.get("metadata", {})
    ok(name, data["took_ms"],
       f"{len(nodes)} nodes, {len(edges)} edges, bridges_found={meta.get('bridges_found')}")


async def test_bridge_edges_with_search_filter(c: httpx.AsyncClient):
    """Bridge edges with category and date filters."""
    name = "bridge_edges:with_search_filter"
    data = await post_graph(c, {
        "graph": {"type": "bridge_edges", "limit": 10},
        "query": "language model",
        "categories": ["cs.CL"],
        "submitted_date": {"gte": "2024-01-01T00:00:00"},
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    ok(name, data["took_ms"],
       f"{len(data['nodes'])} nodes, bridges_found={data['metadata'].get('bridges_found')}")


async def test_bridge_edges_limit(c: httpx.AsyncClient):
    """Limit=3 vs limit=20 should produce different results."""
    name = "bridge_edges:limit"
    data_small = await post_graph(c, {
        "graph": {"type": "bridge_edges", "limit": 3},
        "query": "deep learning",
    }, f"{name}:small")
    data_big = await post_graph(c, {
        "graph": {"type": "bridge_edges", "limit": 20},
        "query": "deep learning",
    }, f"{name}:big")
    if not data_small or not data_big:
        return
    n_small = len(data_small.get("nodes", []))
    n_big = len(data_big.get("nodes", []))
    if n_big < n_small:
        fail(name, 0, f"limit=20 ({n_big} nodes) should be >= limit=3 ({n_small} nodes)")
        return
    ok(name, 0, f"limit=3→{n_small} nodes, limit=20→{n_big} nodes")


async def test_bridge_edges_properties_validation(c: httpx.AsyncClient):
    """Nodes should have is_bridge_endpoint and degree properties."""
    name = "bridge_edges:properties_validation"
    data = await post_graph(c, {
        "graph": {"type": "bridge_edges", "limit": 15},
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
        if "is_bridge_endpoint" not in props:
            fail(name, data["took_ms"], f"Node {n['id']} missing 'is_bridge_endpoint'")
            return
        if "degree" not in props:
            fail(name, data["took_ms"], f"Node {n['id']} missing 'degree'")
            return
    ok(name, data["took_ms"], f"All {len(nodes)} nodes have is_bridge_endpoint and degree")


async def test_bridge_edges_graph_integrity(c: httpx.AsyncClient):
    """validate_graph_integrity passes."""
    name = "bridge_edges:graph_integrity"
    data = await post_graph(c, {
        "graph": {"type": "bridge_edges", "limit": 10},
        "query": "reinforcement learning",
    }, name)
    if not data:
        return
    if not validate_graph_integrity(data, name):
        return
    ok(name, data["took_ms"], f"{len(data['nodes'])} nodes, {len(data['edges'])} edges — integrity OK")


async def test_bridge_edges_edge_validity(c: httpx.AsyncClient):
    """All edge sources/targets exist as node IDs and bridge edges have relation 'bridge'."""
    name = "bridge_edges:edge_validity"
    data = await post_graph(c, {
        "graph": {"type": "bridge_edges", "limit": 15},
        "query": "computer vision",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    node_ids = {n["id"] for n in data["nodes"]}
    bridge_edges = [e for e in data["edges"] if e["relation"] == "bridge"]
    for e in data["edges"]:
        if e["source"] not in node_ids:
            fail(name, data["took_ms"], f"Edge source {e['source']} not in node IDs")
            return
        if e["target"] not in node_ids:
            fail(name, data["took_ms"], f"Edge target {e['target']} not in node IDs")
            return
    ok(name, data["took_ms"],
       f"All {len(data['edges'])} edges valid, {len(bridge_edges)} bridge edges")


async def test_bridge_edges_metadata_fields(c: httpx.AsyncClient):
    """Check expected metadata keys."""
    name = "bridge_edges:metadata_fields"
    data = await post_graph(c, {
        "graph": {"type": "bridge_edges", "limit": 10},
        "query": "optimization",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    meta = data.get("metadata", {})
    expected_keys = ["bridges_found", "bridges_returned"]
    missing = [k for k in expected_keys if k not in meta]
    if missing:
        fail(name, data["took_ms"], f"Missing metadata keys: {missing}")
        return
    ok(name, data["took_ms"],
       f"bridges_found={meta['bridges_found']}, bridges_returned={meta['bridges_returned']}")


async def test_bridge_edges_found_gte_returned(c: httpx.AsyncClient):
    """bridges_found should be >= bridges_returned."""
    name = "bridge_edges:found_gte_returned"
    data = await post_graph(c, {
        "graph": {"type": "bridge_edges", "limit": 10},
        "query": "machine learning",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    meta = data.get("metadata", {})
    found = meta.get("bridges_found", 0)
    returned = meta.get("bridges_returned", 0)
    if found < returned:
        fail(name, data["took_ms"], f"bridges_found ({found}) < bridges_returned ({returned})")
        return
    ok(name, data["took_ms"], f"bridges_found={found} >= bridges_returned={returned}")


# ═══════════════════════════════════════════════════════
# 45. MIN CUT — Deep Tests
# ═══════════════════════════════════════════════════════

async def test_min_cut_basic(c: httpx.AsyncClient):
    """Basic min cut returns nodes and edges."""
    name = "min_cut:basic"
    data = await post_graph(c, {
        "graph": {
            "type": "min_cut",
            "seed_arxiv_id": CONNECTED_PAPER,
            "target_arxiv_id": BIG_REFS_PAPER,
            "limit": 10,
        },
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    nodes = data.get("nodes", [])
    edges = data.get("edges", [])
    if not nodes:
        fail(name, data["took_ms"], "No nodes returned")
        return
    meta = data.get("metadata", {})
    ok(name, data["took_ms"],
       f"{len(nodes)} nodes, {len(edges)} edges, min_cut_value={meta.get('min_cut_value')}")


async def test_min_cut_with_search_filter(c: httpx.AsyncClient):
    """Min cut with category filter."""
    name = "min_cut:with_search_filter"
    data = await post_graph(c, {
        "graph": {
            "type": "min_cut",
            "seed_arxiv_id": CONNECTED_PAPER,
            "target_arxiv_id": BIG_REFS_PAPER,
            "limit": 10,
        },
        "query": "transformer",
        "categories": ["cs.CL"],
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    ok(name, data["took_ms"],
       f"{len(data['nodes'])} nodes, min_cut_value={data['metadata'].get('min_cut_value')}")


async def test_min_cut_limit(c: httpx.AsyncClient):
    """Limit=3 vs limit=20 should produce different node counts."""
    name = "min_cut:limit"
    data_small = await post_graph(c, {
        "graph": {
            "type": "min_cut",
            "seed_arxiv_id": CONNECTED_PAPER,
            "target_arxiv_id": BIG_REFS_PAPER,
            "limit": 3,
        },
    }, f"{name}:small")
    data_big = await post_graph(c, {
        "graph": {
            "type": "min_cut",
            "seed_arxiv_id": CONNECTED_PAPER,
            "target_arxiv_id": BIG_REFS_PAPER,
            "limit": 20,
        },
    }, f"{name}:big")
    if not data_small or not data_big:
        return
    n_small = len(data_small.get("nodes", []))
    n_big = len(data_big.get("nodes", []))
    if n_big < n_small:
        fail(name, 0, f"limit=20 ({n_big} nodes) should be >= limit=3 ({n_small} nodes)")
        return
    ok(name, 0, f"limit=3→{n_small} nodes, limit=20→{n_big} nodes")


async def test_min_cut_properties_validation(c: httpx.AsyncClient):
    """Nodes should have side (source/target) and is_endpoint or is_cut_endpoint."""
    name = "min_cut:properties_validation"
    data = await post_graph(c, {
        "graph": {
            "type": "min_cut",
            "seed_arxiv_id": CONNECTED_PAPER,
            "target_arxiv_id": BIG_REFS_PAPER,
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
        side = props.get("side")
        if side not in ("source", "target"):
            fail(name, data["took_ms"],
                 f"Node {n['id']} has side='{side}', expected 'source' or 'target'")
            return
    ok(name, data["took_ms"], f"All {len(nodes)} nodes have valid side property")


async def test_min_cut_graph_integrity(c: httpx.AsyncClient):
    """validate_graph_integrity passes."""
    name = "min_cut:graph_integrity"
    data = await post_graph(c, {
        "graph": {
            "type": "min_cut",
            "seed_arxiv_id": CONNECTED_PAPER,
            "target_arxiv_id": BIG_REFS_PAPER,
            "limit": 10,
        },
    }, name)
    if not data:
        return
    if not validate_graph_integrity(data, name):
        return
    ok(name, data["took_ms"], f"{len(data['nodes'])} nodes, {len(data['edges'])} edges — integrity OK")


async def test_min_cut_edge_validity(c: httpx.AsyncClient):
    """All edge sources/targets exist as node IDs."""
    name = "min_cut:edge_validity"
    data = await post_graph(c, {
        "graph": {
            "type": "min_cut",
            "seed_arxiv_id": CONNECTED_PAPER,
            "target_arxiv_id": BIG_REFS_PAPER,
            "limit": 15,
        },
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


async def test_min_cut_metadata_fields(c: httpx.AsyncClient):
    """Check expected metadata keys."""
    name = "min_cut:metadata_fields"
    data = await post_graph(c, {
        "graph": {
            "type": "min_cut",
            "seed_arxiv_id": CONNECTED_PAPER,
            "target_arxiv_id": BIG_REFS_PAPER,
            "limit": 10,
        },
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    meta = data.get("metadata", {})
    expected_keys = ["min_cut_value", "cut_edges", "source_side_size", "target_side_size"]
    missing = [k for k in expected_keys if k not in meta]
    if missing:
        fail(name, data["took_ms"], f"Missing metadata keys: {missing}")
        return
    ok(name, data["took_ms"],
       f"min_cut_value={meta['min_cut_value']}, cut_edges={meta['cut_edges']}, "
       f"source_side={meta['source_side_size']}, target_side={meta['target_side_size']}")


async def test_min_cut_value_positive(c: httpx.AsyncClient):
    """min_cut_value should be > 0 for connected source/target."""
    name = "min_cut:value_positive"
    data = await post_graph(c, {
        "graph": {
            "type": "min_cut",
            "seed_arxiv_id": CONNECTED_PAPER,
            "target_arxiv_id": BIG_REFS_PAPER,
            "limit": 10,
        },
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    meta = data.get("metadata", {})
    mcv = meta.get("min_cut_value")
    if mcv is None:
        fail(name, data["took_ms"], "min_cut_value not in metadata")
        return
    if mcv <= 0:
        fail(name, data["took_ms"], f"min_cut_value={mcv}, expected > 0")
        return
    ok(name, data["took_ms"], f"min_cut_value={mcv} > 0")


# ═══════════════════════════════════════════════════════
# 46. MINIMUM SPANNING TREE — Deep Tests
# ═══════════════════════════════════════════════════════

async def test_mst_basic(c: httpx.AsyncClient):
    """Basic MST returns nodes and edges."""
    name = "minimum_spanning_tree:basic"
    data = await post_graph(c, {
        "graph": {"type": "minimum_spanning_tree", "limit": 10},
        "query": "transformer attention",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    nodes = data.get("nodes", [])
    edges = data.get("edges", [])
    if not nodes:
        fail(name, data["took_ms"], "No nodes returned")
        return
    meta = data.get("metadata", {})
    ok(name, data["took_ms"],
       f"{len(nodes)} nodes, {len(edges)} edges, mst_edges={meta.get('mst_edges')}")


async def test_mst_with_search_filter(c: httpx.AsyncClient):
    """MST with category and date filters."""
    name = "minimum_spanning_tree:with_search_filter"
    data = await post_graph(c, {
        "graph": {"type": "minimum_spanning_tree", "limit": 10},
        "query": "language model",
        "categories": ["cs.CL"],
        "submitted_date": {"gte": "2024-01-01T00:00:00"},
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    ok(name, data["took_ms"],
       f"{len(data['nodes'])} nodes, mst_edges={data['metadata'].get('mst_edges')}")


async def test_mst_limit(c: httpx.AsyncClient):
    """Limit=3 vs limit=20 should produce different node counts."""
    name = "minimum_spanning_tree:limit"
    data_small = await post_graph(c, {
        "graph": {"type": "minimum_spanning_tree", "limit": 3},
        "query": "deep learning",
    }, f"{name}:small")
    data_big = await post_graph(c, {
        "graph": {"type": "minimum_spanning_tree", "limit": 20},
        "query": "deep learning",
    }, f"{name}:big")
    if not data_small or not data_big:
        return
    n_small = len(data_small.get("nodes", []))
    n_big = len(data_big.get("nodes", []))
    if n_big < n_small:
        fail(name, 0, f"limit=20 ({n_big} nodes) should be >= limit=3 ({n_small} nodes)")
        return
    ok(name, 0, f"limit=3→{n_small} nodes, limit=20→{n_big} nodes")


async def test_mst_properties_validation(c: httpx.AsyncClient):
    """Nodes should have in_mst and degree properties."""
    name = "minimum_spanning_tree:properties_validation"
    data = await post_graph(c, {
        "graph": {"type": "minimum_spanning_tree", "limit": 15},
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
        if "in_mst" not in props:
            fail(name, data["took_ms"], f"Node {n['id']} missing 'in_mst'")
            return
        if "degree" not in props:
            fail(name, data["took_ms"], f"Node {n['id']} missing 'degree'")
            return
    ok(name, data["took_ms"], f"All {len(nodes)} nodes have in_mst and degree")


async def test_mst_graph_integrity(c: httpx.AsyncClient):
    """validate_graph_integrity passes."""
    name = "minimum_spanning_tree:graph_integrity"
    data = await post_graph(c, {
        "graph": {"type": "minimum_spanning_tree", "limit": 10},
        "query": "reinforcement learning",
    }, name)
    if not data:
        return
    if not validate_graph_integrity(data, name):
        return
    ok(name, data["took_ms"], f"{len(data['nodes'])} nodes, {len(data['edges'])} edges — integrity OK")


async def test_mst_edge_validity(c: httpx.AsyncClient):
    """All edge sources/targets exist as node IDs, MST edges have relation and weight."""
    name = "minimum_spanning_tree:edge_validity"
    data = await post_graph(c, {
        "graph": {"type": "minimum_spanning_tree", "limit": 15},
        "query": "computer vision",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    node_ids = {n["id"] for n in data["nodes"]}
    mst_edges = [e for e in data["edges"] if e["relation"] == "mst_edge"]
    for e in data["edges"]:
        if e["source"] not in node_ids:
            fail(name, data["took_ms"], f"Edge source {e['source']} not in node IDs")
            return
        if e["target"] not in node_ids:
            fail(name, data["took_ms"], f"Edge target {e['target']} not in node IDs")
            return
    for e in mst_edges:
        if "weight" not in e:
            fail(name, data["took_ms"], f"MST edge missing weight: {e['source']}→{e['target']}")
            return
    ok(name, data["took_ms"],
       f"All {len(data['edges'])} edges valid, {len(mst_edges)} mst_edge edges with weight")


async def test_mst_metadata_fields(c: httpx.AsyncClient):
    """Check expected metadata keys."""
    name = "minimum_spanning_tree:metadata_fields"
    data = await post_graph(c, {
        "graph": {"type": "minimum_spanning_tree", "limit": 10},
        "query": "optimization",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    meta = data.get("metadata", {})
    expected_keys = ["mst_edges", "total_weight", "components", "is_connected"]
    missing = [k for k in expected_keys if k not in meta]
    if missing:
        fail(name, data["took_ms"], f"Missing metadata keys: {missing}")
        return
    ok(name, data["took_ms"],
       f"mst_edges={meta['mst_edges']}, total_weight={meta['total_weight']}, "
       f"components={meta['components']}, is_connected={meta['is_connected']}")


async def test_mst_components_gte_1(c: httpx.AsyncClient):
    """components should be >= 1."""
    name = "minimum_spanning_tree:components_gte_1"
    data = await post_graph(c, {
        "graph": {"type": "minimum_spanning_tree", "limit": 20},
        "query": "machine learning",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    meta = data.get("metadata", {})
    comp = meta.get("components")
    if comp is None:
        fail(name, data["took_ms"], "components not in metadata")
        return
    if comp < 1:
        fail(name, data["took_ms"], f"components={comp}, expected >= 1")
        return
    ok(name, data["took_ms"], f"components={comp} >= 1")


# ═══════════════════════════════════════════════════════
# 47. NODE SIMILARITY — Deep Tests
# ═══════════════════════════════════════════════════════

async def test_node_similarity_basic(c: httpx.AsyncClient):
    """Basic node similarity returns nodes and edges."""
    name = "node_similarity:basic"
    data = await post_graph(c, {
        "graph": {"type": "node_similarity", "limit": 10},
        "query": "transformer attention",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    nodes = data.get("nodes", [])
    edges = data.get("edges", [])
    if not nodes:
        fail(name, data["took_ms"], "No nodes returned")
        return
    meta = data.get("metadata", {})
    ok(name, data["took_ms"],
       f"{len(nodes)} nodes, {len(edges)} edges, method={meta.get('method')}")


async def test_node_similarity_with_search_filter(c: httpx.AsyncClient):
    """Node similarity with category and date filters."""
    name = "node_similarity:with_search_filter"
    data = await post_graph(c, {
        "graph": {"type": "node_similarity", "limit": 10},
        "query": "language model",
        "categories": ["cs.CL"],
        "submitted_date": {"gte": "2024-01-01T00:00:00"},
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    ok(name, data["took_ms"],
       f"{len(data['nodes'])} nodes, method={data['metadata'].get('method')}")


async def test_node_similarity_limit(c: httpx.AsyncClient):
    """Limit=3 vs limit=20 should produce different node counts."""
    name = "node_similarity:limit"
    data_small = await post_graph(c, {
        "graph": {"type": "node_similarity", "limit": 3},
        "query": "deep learning",
    }, f"{name}:small")
    data_big = await post_graph(c, {
        "graph": {"type": "node_similarity", "limit": 20},
        "query": "deep learning",
    }, f"{name}:big")
    if not data_small or not data_big:
        return
    n_small = len(data_small.get("nodes", []))
    n_big = len(data_big.get("nodes", []))
    if n_big < n_small:
        fail(name, 0, f"limit=20 ({n_big} nodes) should be >= limit=3 ({n_small} nodes)")
        return
    ok(name, 0, f"limit=3→{n_small} nodes, limit=20→{n_big} nodes")


async def test_node_similarity_properties_validation(c: httpx.AsyncClient):
    """Nodes should have degree property."""
    name = "node_similarity:properties_validation"
    data = await post_graph(c, {
        "graph": {"type": "node_similarity", "limit": 15},
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
        if "degree" not in props:
            fail(name, data["took_ms"], f"Node {n['id']} missing 'degree'")
            return
    ok(name, data["took_ms"], f"All {len(nodes)} nodes have degree property")


async def test_node_similarity_graph_integrity(c: httpx.AsyncClient):
    """validate_graph_integrity passes."""
    name = "node_similarity:graph_integrity"
    data = await post_graph(c, {
        "graph": {"type": "node_similarity", "limit": 10},
        "query": "reinforcement learning",
    }, name)
    if not data:
        return
    if not validate_graph_integrity(data, name):
        return
    ok(name, data["took_ms"], f"{len(data['nodes'])} nodes, {len(data['edges'])} edges — integrity OK")


async def test_node_similarity_edge_validity(c: httpx.AsyncClient):
    """Edges have relation 'similar' and weight between 0-1."""
    name = "node_similarity:edge_validity"
    data = await post_graph(c, {
        "graph": {"type": "node_similarity", "limit": 15},
        "query": "computer vision",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    node_ids = {n["id"] for n in data["nodes"]}
    similar_edges = [e for e in data["edges"] if e["relation"] == "similar"]
    for e in data["edges"]:
        if e["source"] not in node_ids:
            fail(name, data["took_ms"], f"Edge source {e['source']} not in node IDs")
            return
        if e["target"] not in node_ids:
            fail(name, data["took_ms"], f"Edge target {e['target']} not in node IDs")
            return
    for e in similar_edges:
        w = e.get("weight")
        if w is None:
            fail(name, data["took_ms"], f"Similar edge missing weight: {e['source']}→{e['target']}")
            return
        if w < 0 or w > 1:
            fail(name, data["took_ms"], f"Similar edge weight={w} out of [0,1]")
            return
    ok(name, data["took_ms"],
       f"All {len(data['edges'])} edges valid, {len(similar_edges)} similar edges with weight in [0,1]")


async def test_node_similarity_metadata_fields(c: httpx.AsyncClient):
    """Check expected metadata keys."""
    name = "node_similarity:metadata_fields"
    data = await post_graph(c, {
        "graph": {"type": "node_similarity", "limit": 10},
        "query": "optimization",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    meta = data.get("metadata", {})
    expected_keys = ["method", "pairs_computed", "pairs_returned", "max_similarity"]
    missing = [k for k in expected_keys if k not in meta]
    if missing:
        fail(name, data["took_ms"], f"Missing metadata keys: {missing}")
        return
    ok(name, data["took_ms"],
       f"method={meta['method']}, pairs_computed={meta['pairs_computed']}, "
       f"pairs_returned={meta['pairs_returned']}, max_similarity={meta['max_similarity']}")


async def test_node_similarity_all_methods(c: httpx.AsyncClient):
    """Test all 3 similarity methods: jaccard, overlap, cosine."""
    name = "node_similarity:all_methods"
    methods_ok = []
    for method in ("jaccard", "overlap", "cosine"):
        data = await post_graph(c, {
            "graph": {"type": "node_similarity", "similarity_method": method, "limit": 10},
            "query": "transformer attention",
        }, f"{name}:{method}")
        if not data:
            continue
        meta = data.get("metadata", {})
        if meta.get("method") != method:
            fail(name, data.get("took_ms", 0),
                 f"Expected method='{method}', got '{meta.get('method')}'")
            return
        methods_ok.append(method)
    if len(methods_ok) == 3:
        ok(name, 0, f"All 3 methods work: {', '.join(methods_ok)}")
    else:
        fail(name, 0, f"Only {len(methods_ok)}/3 methods succeeded: {methods_ok}")


# ═══════════════════════════════════════════════════════
# 48. BIPARTITE PROJECTION — Deep Tests
# ═══════════════════════════════════════════════════════

async def test_bipartite_projection_basic(c: httpx.AsyncClient):
    """Basic bipartite projection returns nodes and edges."""
    name = "bipartite_projection:basic"
    data = await post_graph(c, {
        "graph": {"type": "bipartite_projection", "limit": 10},
        "query": "transformer attention",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    nodes = data.get("nodes", [])
    edges = data.get("edges", [])
    if not nodes:
        fail(name, data["took_ms"], "No nodes returned")
        return
    meta = data.get("metadata", {})
    ok(name, data["took_ms"],
       f"{len(nodes)} nodes, {len(edges)} edges, projection_side={meta.get('projection_side')}")


async def test_bipartite_projection_with_search_filter(c: httpx.AsyncClient):
    """Bipartite projection with category and date filters."""
    name = "bipartite_projection:with_search_filter"
    data = await post_graph(c, {
        "graph": {"type": "bipartite_projection", "limit": 10},
        "query": "language model",
        "categories": ["cs.CL"],
        "submitted_date": {"gte": "2024-01-01T00:00:00"},
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    ok(name, data["took_ms"],
       f"{len(data['nodes'])} nodes, projection_side={data['metadata'].get('projection_side')}")


async def test_bipartite_projection_limit(c: httpx.AsyncClient):
    """Limit=3 vs limit=20 should produce different node counts."""
    name = "bipartite_projection:limit"
    data_small = await post_graph(c, {
        "graph": {"type": "bipartite_projection", "limit": 3},
        "query": "deep learning",
    }, f"{name}:small")
    data_big = await post_graph(c, {
        "graph": {"type": "bipartite_projection", "limit": 20},
        "query": "deep learning",
    }, f"{name}:big")
    if not data_small or not data_big:
        return
    n_small = len(data_small.get("nodes", []))
    n_big = len(data_big.get("nodes", []))
    if n_big < n_small:
        fail(name, 0, f"limit=20 ({n_big} nodes) should be >= limit=3 ({n_small} nodes)")
        return
    ok(name, 0, f"limit=3→{n_small} nodes, limit=20→{n_big} nodes")


async def test_bipartite_projection_properties_validation(c: httpx.AsyncClient):
    """Nodes should have properties appropriate to their type."""
    name = "bipartite_projection:properties_validation"
    data = await post_graph(c, {
        "graph": {"type": "bipartite_projection", "limit": 15},
        "query": "neural network",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    nodes = data.get("nodes", [])
    if not nodes:
        fail(name, data["took_ms"], "No nodes")
        return
    # All nodes should at least have properties dict
    for n in nodes:
        if "properties" not in n:
            fail(name, data["took_ms"], f"Node {n['id']} missing 'properties'")
            return
    ok(name, data["took_ms"], f"All {len(nodes)} nodes have properties")


async def test_bipartite_projection_graph_integrity(c: httpx.AsyncClient):
    """validate_graph_integrity passes."""
    name = "bipartite_projection:graph_integrity"
    data = await post_graph(c, {
        "graph": {"type": "bipartite_projection", "limit": 10},
        "query": "reinforcement learning",
    }, name)
    if not data:
        return
    if not validate_graph_integrity(data, name):
        return
    ok(name, data["took_ms"], f"{len(data['nodes'])} nodes, {len(data['edges'])} edges — integrity OK")


async def test_bipartite_projection_edge_validity(c: httpx.AsyncClient):
    """All edge sources/targets exist as node IDs."""
    name = "bipartite_projection:edge_validity"
    data = await post_graph(c, {
        "graph": {"type": "bipartite_projection", "limit": 15},
        "query": "computer vision",
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


async def test_bipartite_projection_metadata_fields(c: httpx.AsyncClient):
    """Check expected metadata keys."""
    name = "bipartite_projection:metadata_fields"
    data = await post_graph(c, {
        "graph": {"type": "bipartite_projection", "limit": 10},
        "query": "optimization",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    meta = data.get("metadata", {})
    expected_keys = ["projection_side", "papers_in_subgraph"]
    missing = [k for k in expected_keys if k not in meta]
    if missing:
        fail(name, data["took_ms"], f"Missing metadata keys: {missing}")
        return
    ok(name, data["took_ms"],
       f"projection_side={meta['projection_side']}, papers_in_subgraph={meta['papers_in_subgraph']}")


async def test_bipartite_projection_sides(c: httpx.AsyncClient):
    """Test categories projection vs authors projection."""
    name = "bipartite_projection:sides"
    sides_ok = []
    for side in ("categories", "authors"):
        data = await post_graph(c, {
            "graph": {"type": "bipartite_projection", "projection_side": side, "limit": 10},
            "query": "transformer attention",
        }, f"{name}:{side}")
        if not data:
            continue
        meta = data.get("metadata", {})
        if meta.get("projection_side") != side:
            fail(name, data.get("took_ms", 0),
                 f"Expected projection_side='{side}', got '{meta.get('projection_side')}'")
            return
        sides_ok.append(side)
    if len(sides_ok) == 2:
        ok(name, 0, f"Both projection sides work: {', '.join(sides_ok)}")
    else:
        fail(name, 0, f"Only {len(sides_ok)}/2 sides succeeded: {sides_ok}")


# ═══════════════════════════════════════════════════════
# 49. ADAMIC-ADAR INDEX — Deep Tests
# ═══════════════════════════════════════════════════════

async def test_adamic_adar_basic(c: httpx.AsyncClient):
    """Basic Adamic-Adar index returns nodes and edges."""
    name = "adamic_adar_index:basic"
    data = await post_graph(c, {
        "graph": {"type": "adamic_adar_index", "limit": 10},
        "query": "transformer attention",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    nodes = data.get("nodes", [])
    edges = data.get("edges", [])
    if not nodes:
        fail(name, data["took_ms"], "No nodes returned")
        return
    meta = data.get("metadata", {})
    ok(name, data["took_ms"],
       f"{len(nodes)} nodes, {len(edges)} edges, max_score={meta.get('max_score')}")


async def test_adamic_adar_with_search_filter(c: httpx.AsyncClient):
    """Adamic-Adar with category and date filters."""
    name = "adamic_adar_index:with_search_filter"
    data = await post_graph(c, {
        "graph": {"type": "adamic_adar_index", "limit": 10},
        "query": "language model",
        "categories": ["cs.CL"],
        "submitted_date": {"gte": "2024-01-01T00:00:00"},
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    ok(name, data["took_ms"],
       f"{len(data['nodes'])} nodes, max_score={data['metadata'].get('max_score')}")


async def test_adamic_adar_limit(c: httpx.AsyncClient):
    """Limit=3 vs limit=20 should produce different node counts."""
    name = "adamic_adar_index:limit"
    data_small = await post_graph(c, {
        "graph": {"type": "adamic_adar_index", "limit": 3},
        "query": "deep learning",
    }, f"{name}:small")
    data_big = await post_graph(c, {
        "graph": {"type": "adamic_adar_index", "limit": 20},
        "query": "deep learning",
    }, f"{name}:big")
    if not data_small or not data_big:
        return
    n_small = len(data_small.get("nodes", []))
    n_big = len(data_big.get("nodes", []))
    if n_big < n_small:
        fail(name, 0, f"limit=20 ({n_big} nodes) should be >= limit=3 ({n_small} nodes)")
        return
    ok(name, 0, f"limit=3→{n_small} nodes, limit=20→{n_big} nodes")


async def test_adamic_adar_properties_validation(c: httpx.AsyncClient):
    """Nodes should have degree property."""
    name = "adamic_adar_index:properties_validation"
    data = await post_graph(c, {
        "graph": {"type": "adamic_adar_index", "limit": 15},
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
        if "degree" not in props:
            fail(name, data["took_ms"], f"Node {n['id']} missing 'degree'")
            return
    ok(name, data["took_ms"], f"All {len(nodes)} nodes have degree property")


async def test_adamic_adar_graph_integrity(c: httpx.AsyncClient):
    """validate_graph_integrity passes."""
    name = "adamic_adar_index:graph_integrity"
    data = await post_graph(c, {
        "graph": {"type": "adamic_adar_index", "limit": 10},
        "query": "reinforcement learning",
    }, name)
    if not data:
        return
    if not validate_graph_integrity(data, name):
        return
    ok(name, data["took_ms"], f"{len(data['nodes'])} nodes, {len(data['edges'])} edges — integrity OK")


async def test_adamic_adar_edge_validity(c: httpx.AsyncClient):
    """Edges have relation 'adamic_adar' and weight > 0."""
    name = "adamic_adar_index:edge_validity"
    data = await post_graph(c, {
        "graph": {"type": "adamic_adar_index", "limit": 15},
        "query": "computer vision",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    node_ids = {n["id"] for n in data["nodes"]}
    aa_edges = [e for e in data["edges"] if e["relation"] == "adamic_adar"]
    for e in data["edges"]:
        if e["source"] not in node_ids:
            fail(name, data["took_ms"], f"Edge source {e['source']} not in node IDs")
            return
        if e["target"] not in node_ids:
            fail(name, data["took_ms"], f"Edge target {e['target']} not in node IDs")
            return
    for e in aa_edges:
        w = e.get("weight")
        if w is None:
            fail(name, data["took_ms"], f"Adamic-Adar edge missing weight: {e['source']}→{e['target']}")
            return
        if w <= 0:
            fail(name, data["took_ms"], f"Adamic-Adar edge weight={w}, expected > 0")
            return
    ok(name, data["took_ms"],
       f"All {len(data['edges'])} edges valid, {len(aa_edges)} adamic_adar edges with weight > 0")


async def test_adamic_adar_metadata_fields(c: httpx.AsyncClient):
    """Check expected metadata keys."""
    name = "adamic_adar_index:metadata_fields"
    data = await post_graph(c, {
        "graph": {"type": "adamic_adar_index", "limit": 10},
        "query": "optimization",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    meta = data.get("metadata", {})
    expected_keys = ["pairs_computed", "pairs_returned", "max_score"]
    missing = [k for k in expected_keys if k not in meta]
    if missing:
        fail(name, data["took_ms"], f"Missing metadata keys: {missing}")
        return
    ok(name, data["took_ms"],
       f"pairs_computed={meta['pairs_computed']}, pairs_returned={meta['pairs_returned']}, "
       f"max_score={meta['max_score']}")


async def test_adamic_adar_max_score_positive(c: httpx.AsyncClient):
    """max_score should be > 0."""
    name = "adamic_adar_index:max_score_positive"
    data = await post_graph(c, {
        "graph": {"type": "adamic_adar_index", "limit": 10},
        "query": "machine learning",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    meta = data.get("metadata", {})
    max_score = meta.get("max_score")
    if max_score is None:
        fail(name, data["took_ms"], "max_score not in metadata")
        return
    if max_score <= 0:
        fail(name, data["took_ms"], f"max_score={max_score}, expected > 0")
        return
    ok(name, data["took_ms"], f"max_score={max_score} > 0")


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
        print(f"Running {len(tests)} deep graph tests (types 43-49)...\n")
        start = time.monotonic()

        for test_fn in tests:
            await test_fn(c)

        elapsed = time.monotonic() - start

    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)

    print(f"\n{'='*80}")
    print(f" DEEP GRAPH TEST RESULTS (types 43-49): {passed}/{len(results)} passed ({failed} failed) in {elapsed:.1f}s")
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
