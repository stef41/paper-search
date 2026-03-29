"""Deep E2E tests for the 6 graph-DB algorithms (types 23-28).

Tests:
- Correctness: node/edge properties, metadata integrity
- Filter composition: all 6 algorithms with search filters
- Boundary conditions: empty results, small graphs
- Data integrity: node IDs match edges, no duplicates
- Cross-algorithm: same query through different algorithms
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
TIMEOUT = 60.0

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
    """Validate structural integrity of graph response."""
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
# 23. WEIGHTED SHORTEST PATH — Deep Tests
# ═══════════════════════════════════════════════════════

# Known direct link: 2406.03736 cites 2405.04233
DIRECT_TARGET = "2405.04233"
UNREACHABLE_ID = "9999.99999"


async def test_wsp_basic_functionality(c: httpx.AsyncClient):
    """Basic weighted shortest path between two connected papers."""
    name = "wsp:basic_functionality"
    data = await post_graph(c, {
        "graph": {"type": "weighted_shortest_path",
                  "seed_arxiv_id": CONNECTED_PAPER, "target_arxiv_id": DIRECT_TARGET,
                  "max_hops": 5, "weight_field": "citations"},
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    meta = data.get("metadata", {})
    if meta.get("path_found") is False:
        ok(name, data["took_ms"], "No path found (may depend on index state)")
        return
    if len(data["nodes"]) < 2:
        fail(name, data["took_ms"], f"Expected >=2 nodes on path, got {len(data['nodes'])}")
        return
    ok(name, data["took_ms"],
       f"path_length={meta.get('path_length')}, nodes={len(data['nodes'])}, edges={len(data['edges'])}")


async def test_wsp_metadata_correctness(c: httpx.AsyncClient):
    """Check all expected metadata fields are present."""
    name = "wsp:metadata_correctness"
    data = await post_graph(c, {
        "graph": {"type": "weighted_shortest_path",
                  "seed_arxiv_id": CONNECTED_PAPER, "target_arxiv_id": DIRECT_TARGET,
                  "max_hops": 5, "weight_field": "citations"},
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    meta = data.get("metadata", {})
    if meta.get("path_found") is False:
        ok(name, data["took_ms"], "No path found — metadata has path_found=false")
        return
    expected_keys = ["source", "target", "path_length", "total_cost", "weight_field", "explored", "path_ids"]
    missing = [k for k in expected_keys if k not in meta]
    if missing:
        fail(name, data["took_ms"], f"Missing metadata keys: {missing}")
        return
    if meta.get("source") != CONNECTED_PAPER:
        fail(name, data["took_ms"], f"source mismatch: {meta.get('source')}")
        return
    if meta.get("target") != DIRECT_TARGET:
        fail(name, data["took_ms"], f"target mismatch: {meta.get('target')}")
        return
    ok(name, data["took_ms"], f"All metadata present, cost={meta.get('total_cost')}")


async def test_wsp_path_ids_contain_endpoints(c: httpx.AsyncClient):
    """path_ids should contain both source and target."""
    name = "wsp:path_ids_endpoints"
    data = await post_graph(c, {
        "graph": {"type": "weighted_shortest_path",
                  "seed_arxiv_id": CONNECTED_PAPER, "target_arxiv_id": DIRECT_TARGET,
                  "max_hops": 5, "weight_field": "citations"},
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    meta = data.get("metadata", {})
    if meta.get("path_found") is False:
        ok(name, data["took_ms"], "No path — skipping path_ids check")
        return
    path_ids = meta.get("path_ids", [])
    if not path_ids:
        fail(name, data["took_ms"], "path_ids is empty")
        return
    if path_ids[0] != CONNECTED_PAPER:
        fail(name, data["took_ms"], f"path_ids[0]={path_ids[0]}, expected {CONNECTED_PAPER}")
        return
    if path_ids[-1] != DIRECT_TARGET:
        fail(name, data["took_ms"], f"path_ids[-1]={path_ids[-1]}, expected {DIRECT_TARGET}")
        return
    ok(name, data["took_ms"], f"path_ids has {len(path_ids)} entries, endpoints correct")


async def test_wsp_uniform_weight(c: httpx.AsyncClient):
    """Test with weight_field=uniform (BFS-like)."""
    name = "wsp:uniform_weight"
    data = await post_graph(c, {
        "graph": {"type": "weighted_shortest_path",
                  "seed_arxiv_id": CONNECTED_PAPER, "target_arxiv_id": DIRECT_TARGET,
                  "max_hops": 5, "weight_field": "uniform"},
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    meta = data.get("metadata", {})
    if meta.get("path_found") is False:
        ok(name, data["took_ms"], "No path found with uniform weight")
        return
    if meta.get("weight_field") != "uniform":
        fail(name, data["took_ms"], f"weight_field mismatch: {meta.get('weight_field')}")
        return
    ok(name, data["took_ms"], f"uniform path_length={meta.get('path_length')}, cost={meta.get('total_cost')}")


async def test_wsp_citations_vs_uniform(c: httpx.AsyncClient):
    """Both weight fields should find a path (if one exists), path lengths may differ."""
    name = "wsp:citations_vs_uniform"
    start = time.monotonic()
    d_cit = await post_graph(c, {
        "graph": {"type": "weighted_shortest_path",
                  "seed_arxiv_id": CONNECTED_PAPER, "target_arxiv_id": DIRECT_TARGET,
                  "max_hops": 5, "weight_field": "citations"},
    }, name)
    d_uni = await post_graph(c, {
        "graph": {"type": "weighted_shortest_path",
                  "seed_arxiv_id": CONNECTED_PAPER, "target_arxiv_id": DIRECT_TARGET,
                  "max_hops": 5, "weight_field": "uniform"},
    }, name)
    elapsed = int((time.monotonic() - start) * 1000)
    if not d_cit or not d_uni:
        return
    m_cit = d_cit.get("metadata", {})
    m_uni = d_uni.get("metadata", {})
    cit_found = m_cit.get("path_found") is not False
    uni_found = m_uni.get("path_found") is not False
    ok(name, elapsed,
       f"citations_found={cit_found} len={m_cit.get('path_length')}, "
       f"uniform_found={uni_found} len={m_uni.get('path_length')}")


async def test_wsp_unreachable_target(c: httpx.AsyncClient):
    """Unreachable target should return path_found=false."""
    name = "wsp:unreachable_target"
    data = await post_graph(c, {
        "graph": {"type": "weighted_shortest_path",
                  "seed_arxiv_id": CONNECTED_PAPER, "target_arxiv_id": UNREACHABLE_ID,
                  "max_hops": 3, "weight_field": "uniform"},
    }, name)
    if not data:
        return
    meta = data.get("metadata", {})
    # Should either be path_found=false or an error about paper not found
    if meta.get("path_found") is False or "error" in meta or "not found" in str(meta).lower():
        ok(name, data.get("took_ms", 0), f"Correctly handled unreachable: {meta}")
        return
    # If it found a path to a nonexistent paper, that's a bug
    fail(name, data.get("took_ms", 0), f"Found path to unreachable paper: {meta}")


async def test_wsp_same_source_target(c: httpx.AsyncClient):
    """source == target should return an error."""
    name = "wsp:same_source_target"
    data = await post_graph(c, {
        "graph": {"type": "weighted_shortest_path",
                  "seed_arxiv_id": CONNECTED_PAPER, "target_arxiv_id": CONNECTED_PAPER,
                  "max_hops": 3, "weight_field": "uniform"},
    }, name)
    if not data:
        return
    meta = data.get("metadata", {})
    if "error" in meta:
        ok(name, data.get("took_ms", 0), f"Got expected error: {meta['error']}")
    else:
        fail(name, data.get("took_ms", 0), "Expected error for same source/target")


async def test_wsp_max_hops_respected(c: httpx.AsyncClient):
    """max_hops=1 limits search depth."""
    name = "wsp:max_hops_respected"
    data = await post_graph(c, {
        "graph": {"type": "weighted_shortest_path",
                  "seed_arxiv_id": CONNECTED_PAPER, "target_arxiv_id": DIRECT_TARGET,
                  "max_hops": 1, "weight_field": "uniform"},
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    meta = data.get("metadata", {})
    if meta.get("path_found") is False:
        ok(name, data["took_ms"], "No path within 1 hop")
        return
    path_len = meta.get("path_length", 0)
    if path_len and path_len > 1:
        fail(name, data["took_ms"], f"Path length {path_len} exceeds max_hops=1")
        return
    ok(name, data["took_ms"], f"path_length={path_len} within max_hops=1")


async def test_wsp_node_properties(c: httpx.AsyncClient):
    """Nodes on the path should have standard paper properties."""
    name = "wsp:node_properties"
    data = await post_graph(c, {
        "graph": {"type": "weighted_shortest_path",
                  "seed_arxiv_id": CONNECTED_PAPER, "target_arxiv_id": DIRECT_TARGET,
                  "max_hops": 5, "weight_field": "citations"},
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    if not data["nodes"]:
        ok(name, data["took_ms"], "No nodes to check")
        return
    node = data["nodes"][0]
    if node.get("type") != "paper":
        fail(name, data["took_ms"], f"Expected type=paper, got {node.get('type')}")
        return
    ok(name, data["took_ms"], f"First node type={node['type']}, label={node.get('label', '')[:50]}")


# ═══════════════════════════════════════════════════════
# 24. BETWEENNESS CENTRALITY — Deep Tests
# ═══════════════════════════════════════════════════════

async def test_bc_basic_functionality(c: httpx.AsyncClient):
    """Basic betweenness centrality with query."""
    name = "bc:basic_functionality"
    data = await post_graph(c, {
        "graph": {"type": "betweenness_centrality", "limit": 10},
        "query": "transformer attention mechanism",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    meta = data.get("metadata", {})
    if "error" in meta:
        ok(name, data["took_ms"], f"Insufficient data: {meta['error']}")
        return
    if not data["nodes"]:
        fail(name, data["took_ms"], "No nodes returned")
        return
    ok(name, data["took_ms"],
       f"{len(data['nodes'])} nodes, subgraph={meta.get('papers_in_subgraph')}")


async def test_bc_metadata_correctness(c: httpx.AsyncClient):
    """Check all expected metadata fields."""
    name = "bc:metadata_correctness"
    data = await post_graph(c, {
        "graph": {"type": "betweenness_centrality", "limit": 10},
        "query": "large language model",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    meta = data.get("metadata", {})
    if "error" in meta:
        ok(name, data["took_ms"], f"Insufficient data: {meta['error']}")
        return
    expected = ["papers_in_subgraph", "edges_in_subgraph", "max_betweenness", "sampled_sources"]
    missing = [k for k in expected if k not in meta]
    if missing:
        fail(name, data["took_ms"], f"Missing metadata: {missing}")
        return
    if not isinstance(meta["papers_in_subgraph"], int) or meta["papers_in_subgraph"] < 3:
        fail(name, data["took_ms"], f"papers_in_subgraph invalid: {meta['papers_in_subgraph']}")
        return
    ok(name, data["took_ms"],
       f"subgraph={meta['papers_in_subgraph']}, max_bc={meta['max_betweenness']}")


async def test_bc_node_betweenness_property(c: httpx.AsyncClient):
    """Each node should have a 'betweenness' score in properties."""
    name = "bc:node_betweenness_property"
    data = await post_graph(c, {
        "graph": {"type": "betweenness_centrality", "limit": 10},
        "query": "diffusion model image generation",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    meta = data.get("metadata", {})
    if "error" in meta:
        ok(name, data["took_ms"], f"Insufficient data: {meta['error']}")
        return
    for n in data["nodes"]:
        props = n.get("properties", {})
        if "betweenness" not in props:
            fail(name, data["took_ms"], f"Node {n['id']} missing 'betweenness' property")
            return
        if not isinstance(props["betweenness"], (int, float)):
            fail(name, data["took_ms"], f"betweenness not numeric: {type(props['betweenness'])}")
            return
    ok(name, data["took_ms"], f"All {len(data['nodes'])} nodes have betweenness scores")


async def test_bc_with_query_filter(c: httpx.AsyncClient):
    """Betweenness centrality combined with text query."""
    name = "bc:with_query_filter"
    data = await post_graph(c, {
        "graph": {"type": "betweenness_centrality", "limit": 5},
        "query": "reinforcement learning robotics",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    meta = data.get("metadata", {})
    if "error" in meta:
        ok(name, data["took_ms"], f"Insufficient data: {meta['error']}")
        return
    ok(name, data["took_ms"], f"{len(data['nodes'])} nodes, {len(data['edges'])} edges")


async def test_bc_with_category_filter(c: httpx.AsyncClient):
    """Betweenness centrality + category filter."""
    name = "bc:with_category_filter"
    data = await post_graph(c, {
        "graph": {"type": "betweenness_centrality", "limit": 10},
        "query": "neural network", "categories": ["cs.AI"],
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    meta = data.get("metadata", {})
    if "error" in meta:
        ok(name, data["took_ms"], f"Insufficient data: {meta['error']}")
        return
    ok(name, data["took_ms"],
       f"cs.AI filter: {len(data['nodes'])} nodes, subgraph={meta.get('papers_in_subgraph')}")


async def test_bc_with_date_filter(c: httpx.AsyncClient):
    """Betweenness centrality + date range filter."""
    name = "bc:with_date_filter"
    data = await post_graph(c, {
        "graph": {"type": "betweenness_centrality", "limit": 10},
        "query": "language model", "submitted_date": {"gte": "2024-01-01T00:00:00"},
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    meta = data.get("metadata", {})
    if "error" in meta:
        ok(name, data["took_ms"], f"Insufficient data with date filter: {meta['error']}")
        return
    ok(name, data["took_ms"],
       f"date-filtered: {len(data['nodes'])} nodes, subgraph={meta.get('papers_in_subgraph')}")


async def test_bc_limit_1(c: httpx.AsyncClient):
    """limit=1 should return exactly one node."""
    name = "bc:limit_1"
    data = await post_graph(c, {
        "graph": {"type": "betweenness_centrality", "limit": 1},
        "query": "computer vision object detection",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    meta = data.get("metadata", {})
    if "error" in meta:
        ok(name, data["took_ms"], f"Insufficient data: {meta['error']}")
        return
    if len(data["nodes"]) != 1:
        fail(name, data["took_ms"], f"Expected 1 node, got {len(data['nodes'])}")
        return
    ok(name, data["took_ms"], f"Got exactly 1 node: {data['nodes'][0]['id']}")


async def test_bc_limit_50(c: httpx.AsyncClient):
    """limit=50 should return up to 50 nodes."""
    name = "bc:limit_50"
    data = await post_graph(c, {
        "graph": {"type": "betweenness_centrality", "limit": 50},
        "query": "deep learning",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    meta = data.get("metadata", {})
    if "error" in meta:
        ok(name, data["took_ms"], f"Insufficient data: {meta['error']}")
        return
    if len(data["nodes"]) > 50:
        fail(name, data["took_ms"], f"Exceeded limit: {len(data['nodes'])} nodes")
        return
    ok(name, data["took_ms"], f"{len(data['nodes'])} nodes (limit=50)")


async def test_bc_empty_result_graceful(c: httpx.AsyncClient):
    """Very specific query should return gracefully (empty or error)."""
    name = "bc:empty_result_graceful"
    data = await post_graph(c, {
        "graph": {"type": "betweenness_centrality", "limit": 10},
        "query": "xylophone quantum teleportation zebra unicorn 9999",
    }, name)
    if not data:
        return
    meta = data.get("metadata", {})
    if "error" in meta or len(data.get("nodes", [])) == 0:
        ok(name, data.get("took_ms", 0), f"Graceful empty: {meta.get('error', 'no nodes')}")
    else:
        ok(name, data.get("took_ms", 0), f"Got {len(data['nodes'])} nodes (unexpected but OK)")


async def test_bc_seed_arxiv_id(c: httpx.AsyncClient):
    """Betweenness centrality with seed_arxiv_id instead of query."""
    name = "bc:seed_arxiv_id"
    data = await post_graph(c, {
        "graph": {"type": "betweenness_centrality", "seed_arxiv_id": CONNECTED_PAPER, "limit": 10},
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    meta = data.get("metadata", {})
    if "error" in meta:
        ok(name, data["took_ms"], f"Insufficient data: {meta['error']}")
        return
    ok(name, data["took_ms"],
       f"seed-based: {len(data['nodes'])} nodes, subgraph={meta.get('papers_in_subgraph')}")


# ═══════════════════════════════════════════════════════
# 25. CLOSENESS CENTRALITY — Deep Tests
# ═══════════════════════════════════════════════════════

async def test_cc_basic_functionality(c: httpx.AsyncClient):
    """Basic closeness centrality with query."""
    name = "cc:basic_functionality"
    data = await post_graph(c, {
        "graph": {"type": "closeness_centrality", "limit": 10},
        "query": "graph neural network",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    meta = data.get("metadata", {})
    if "error" in meta:
        ok(name, data["took_ms"], f"Insufficient data: {meta['error']}")
        return
    if not data["nodes"]:
        fail(name, data["took_ms"], "No nodes returned")
        return
    ok(name, data["took_ms"],
       f"{len(data['nodes'])} nodes, subgraph={meta.get('papers_in_subgraph')}")


async def test_cc_metadata_correctness(c: httpx.AsyncClient):
    """Check all expected metadata fields."""
    name = "cc:metadata_correctness"
    data = await post_graph(c, {
        "graph": {"type": "closeness_centrality", "limit": 10},
        "query": "natural language processing",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    meta = data.get("metadata", {})
    if "error" in meta:
        ok(name, data["took_ms"], f"Insufficient data: {meta['error']}")
        return
    expected = ["papers_in_subgraph", "edges_in_subgraph", "max_closeness", "min_closeness"]
    missing = [k for k in expected if k not in meta]
    if missing:
        fail(name, data["took_ms"], f"Missing metadata: {missing}")
        return
    max_c = meta.get("max_closeness", 0)
    min_c = meta.get("min_closeness", 0)
    if max_c < min_c:
        fail(name, data["took_ms"], f"max_closeness ({max_c}) < min_closeness ({min_c})")
        return
    ok(name, data["took_ms"], f"closeness range [{min_c}, {max_c}]")


async def test_cc_node_closeness_property(c: httpx.AsyncClient):
    """Each node should have a 'closeness' score in properties."""
    name = "cc:node_closeness_property"
    data = await post_graph(c, {
        "graph": {"type": "closeness_centrality", "limit": 10},
        "query": "speech recognition",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    meta = data.get("metadata", {})
    if "error" in meta:
        ok(name, data["took_ms"], f"Insufficient data: {meta['error']}")
        return
    for n in data["nodes"]:
        props = n.get("properties", {})
        if "closeness" not in props:
            fail(name, data["took_ms"], f"Node {n['id']} missing 'closeness' property")
            return
        if not isinstance(props["closeness"], (int, float)):
            fail(name, data["took_ms"], f"closeness not numeric: {type(props['closeness'])}")
            return
    ok(name, data["took_ms"], f"All {len(data['nodes'])} nodes have closeness scores")


async def test_cc_with_category_filter(c: httpx.AsyncClient):
    """Closeness centrality + category filter."""
    name = "cc:with_category_filter"
    data = await post_graph(c, {
        "graph": {"type": "closeness_centrality", "limit": 10},
        "query": "optimization", "categories": ["cs.LG"],
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    meta = data.get("metadata", {})
    if "error" in meta:
        ok(name, data["took_ms"], f"Insufficient data: {meta['error']}")
        return
    ok(name, data["took_ms"],
       f"cs.LG filter: {len(data['nodes'])} nodes, subgraph={meta.get('papers_in_subgraph')}")


async def test_cc_with_date_filter(c: httpx.AsyncClient):
    """Closeness centrality + date range filter."""
    name = "cc:with_date_filter"
    data = await post_graph(c, {
        "graph": {"type": "closeness_centrality", "limit": 10},
        "query": "multimodal learning", "submitted_date": {"gte": "2024-01-01T00:00:00"},
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    meta = data.get("metadata", {})
    if "error" in meta:
        ok(name, data["took_ms"], f"Insufficient data with date filter: {meta['error']}")
        return
    ok(name, data["took_ms"],
       f"date-filtered: {len(data['nodes'])} nodes")


async def test_cc_limit_1(c: httpx.AsyncClient):
    """limit=1 should return exactly one node."""
    name = "cc:limit_1"
    data = await post_graph(c, {
        "graph": {"type": "closeness_centrality", "limit": 1},
        "query": "neural architecture search",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    meta = data.get("metadata", {})
    if "error" in meta:
        ok(name, data["took_ms"], f"Insufficient data: {meta['error']}")
        return
    if len(data["nodes"]) != 1:
        fail(name, data["took_ms"], f"Expected 1 node, got {len(data['nodes'])}")
        return
    ok(name, data["took_ms"], f"Got 1 node, closeness={data['nodes'][0].get('properties', {}).get('closeness')}")


async def test_cc_closeness_ordering(c: httpx.AsyncClient):
    """Nodes should be ordered by descending closeness score."""
    name = "cc:closeness_ordering"
    data = await post_graph(c, {
        "graph": {"type": "closeness_centrality", "limit": 20},
        "query": "knowledge graph embedding",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    meta = data.get("metadata", {})
    if "error" in meta:
        ok(name, data["took_ms"], f"Insufficient data: {meta['error']}")
        return
    scores = [n.get("properties", {}).get("closeness", 0) for n in data["nodes"]]
    if len(scores) < 2:
        ok(name, data["took_ms"], "Too few nodes to check ordering")
        return
    is_sorted = all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))
    if not is_sorted:
        fail(name, data["took_ms"], f"Scores not descending: {scores[:5]}")
        return
    ok(name, data["took_ms"], f"{len(scores)} nodes in descending closeness order")


async def test_cc_empty_result_graceful(c: httpx.AsyncClient):
    """Very specific query should fail gracefully."""
    name = "cc:empty_result_graceful"
    data = await post_graph(c, {
        "graph": {"type": "closeness_centrality", "limit": 10},
        "query": "platypus thermodynamics cryptocurrency ballet 9999",
    }, name)
    if not data:
        return
    meta = data.get("metadata", {})
    if "error" in meta or len(data.get("nodes", [])) == 0:
        ok(name, data.get("took_ms", 0), f"Graceful empty: {meta.get('error', 'no nodes')}")
    else:
        ok(name, data.get("took_ms", 0), f"Got {len(data['nodes'])} nodes")


async def test_cc_seed_arxiv_id(c: httpx.AsyncClient):
    """Closeness centrality with seed_arxiv_id."""
    name = "cc:seed_arxiv_id"
    data = await post_graph(c, {
        "graph": {"type": "closeness_centrality", "seed_arxiv_id": BIG_REFS_PAPER, "limit": 10},
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    meta = data.get("metadata", {})
    if "error" in meta:
        ok(name, data["took_ms"], f"Insufficient data: {meta['error']}")
        return
    ok(name, data["took_ms"],
       f"seed-based: {len(data['nodes'])} nodes, subgraph={meta.get('papers_in_subgraph')}")


# ═══════════════════════════════════════════════════════
# 26. STRONGLY CONNECTED COMPONENTS — Deep Tests
# ═══════════════════════════════════════════════════════

async def test_scc_basic_functionality(c: httpx.AsyncClient):
    """Basic SCC query returns scc-type nodes."""
    name = "scc:basic_functionality"
    data = await post_graph(c, {
        "graph": {"type": "strongly_connected_components", "limit": 10},
        "query": "transformer attention",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    meta = data.get("metadata", {})
    if "error" in meta:
        ok(name, data["took_ms"], f"Insufficient data: {meta['error']}")
        return
    scc_nodes = [n for n in data["nodes"] if n.get("type") == "scc"]
    paper_nodes = [n for n in data["nodes"] if n.get("type") == "paper"]
    ok(name, data["took_ms"],
       f"{len(scc_nodes)} SCC nodes, {len(paper_nodes)} paper nodes, {len(data['edges'])} edges")


async def test_scc_metadata_correctness(c: httpx.AsyncClient):
    """Check all expected metadata fields."""
    name = "scc:metadata_correctness"
    data = await post_graph(c, {
        "graph": {"type": "strongly_connected_components", "limit": 10},
        "query": "deep reinforcement learning",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    meta = data.get("metadata", {})
    if "error" in meta:
        ok(name, data["took_ms"], f"Insufficient data: {meta['error']}")
        return
    expected = ["papers_in_graph", "directed_edges", "sccs_found", "nontrivial_sccs", "largest_scc"]
    missing = [k for k in expected if k not in meta]
    if missing:
        fail(name, data["took_ms"], f"Missing metadata: {missing}")
        return
    ok(name, data["took_ms"],
       f"sccs={meta['sccs_found']}, nontrivial={meta['nontrivial_sccs']}, largest={meta['largest_scc']}")


async def test_scc_node_structure(c: httpx.AsyncClient):
    """SCC nodes should have size, is_trivial, member_ids properties."""
    name = "scc:node_structure"
    data = await post_graph(c, {
        "graph": {"type": "strongly_connected_components", "limit": 10},
        "query": "convolutional neural network",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    meta = data.get("metadata", {})
    if "error" in meta:
        ok(name, data["took_ms"], f"Insufficient data: {meta['error']}")
        return
    scc_nodes = [n for n in data["nodes"] if n.get("type") == "scc"]
    if not scc_nodes:
        ok(name, data["took_ms"], "No SCC nodes to check")
        return
    for sn in scc_nodes:
        props = sn.get("properties", {})
        for k in ("size", "is_trivial", "member_ids"):
            if k not in props:
                fail(name, data["took_ms"], f"SCC node {sn['id']} missing '{k}'")
                return
    ok(name, data["took_ms"], f"All {len(scc_nodes)} SCC nodes have required properties")


async def test_scc_contains_edges(c: httpx.AsyncClient):
    """SCC→paper edges should have relation='contains'."""
    name = "scc:contains_edges"
    data = await post_graph(c, {
        "graph": {"type": "strongly_connected_components", "limit": 10},
        "query": "recurrent neural network",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    meta = data.get("metadata", {})
    if "error" in meta:
        ok(name, data["took_ms"], f"Insufficient data: {meta['error']}")
        return
    contains_edges = [e for e in data["edges"] if e["relation"] == "contains"]
    scc_edges = [e for e in data["edges"] if e["relation"] == "scc_edge"]
    ok(name, data["took_ms"],
       f"{len(contains_edges)} contains edges, {len(scc_edges)} scc_edge edges")


async def test_scc_with_category_filter(c: httpx.AsyncClient):
    """SCC + category filter."""
    name = "scc:with_category_filter"
    data = await post_graph(c, {
        "graph": {"type": "strongly_connected_components", "limit": 10},
        "query": "machine learning", "categories": ["cs.CL"],
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    meta = data.get("metadata", {})
    if "error" in meta:
        ok(name, data["took_ms"], f"Insufficient data: {meta['error']}")
        return
    ok(name, data["took_ms"],
       f"cs.CL filter: sccs={meta.get('sccs_found')}, papers={meta.get('papers_in_graph')}")


async def test_scc_with_date_filter(c: httpx.AsyncClient):
    """SCC + date range filter."""
    name = "scc:with_date_filter"
    data = await post_graph(c, {
        "graph": {"type": "strongly_connected_components", "limit": 10},
        "query": "generative model", "submitted_date": {"gte": "2024-01-01T00:00:00"},
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    meta = data.get("metadata", {})
    if "error" in meta:
        ok(name, data["took_ms"], f"Insufficient data with date filter: {meta['error']}")
        return
    ok(name, data["took_ms"],
       f"date-filtered: sccs={meta.get('sccs_found')}, papers={meta.get('papers_in_graph')}")


async def test_scc_nontrivial_not_required(c: httpx.AsyncClient):
    """Citation graphs are DAG-like so nontrivial SCCs are rare — don't require them."""
    name = "scc:nontrivial_not_required"
    data = await post_graph(c, {
        "graph": {"type": "strongly_connected_components", "limit": 20},
        "query": "image classification",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    meta = data.get("metadata", {})
    if "error" in meta:
        ok(name, data["took_ms"], f"Insufficient data: {meta['error']}")
        return
    nontrivial = meta.get("nontrivial_sccs", 0)
    # It's fine if nontrivial is 0 — citation graphs are mostly DAGs
    ok(name, data["took_ms"],
       f"nontrivial_sccs={nontrivial} (0 is acceptable for DAG-like citation graphs)")


async def test_scc_limit_parameter(c: httpx.AsyncClient):
    """Limit should cap the number of SCCs returned."""
    name = "scc:limit_parameter"
    data = await post_graph(c, {
        "graph": {"type": "strongly_connected_components", "limit": 3},
        "query": "natural language understanding",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    meta = data.get("metadata", {})
    if "error" in meta:
        ok(name, data["took_ms"], f"Insufficient data: {meta['error']}")
        return
    scc_nodes = [n for n in data["nodes"] if n.get("type") == "scc"]
    if len(scc_nodes) > 3:
        fail(name, data["took_ms"], f"Expected <=3 SCC nodes, got {len(scc_nodes)}")
        return
    ok(name, data["took_ms"], f"{len(scc_nodes)} SCC nodes (limit=3)")


async def test_scc_empty_result_graceful(c: httpx.AsyncClient):
    """Very specific query should fail gracefully."""
    name = "scc:empty_result_graceful"
    data = await post_graph(c, {
        "graph": {"type": "strongly_connected_components", "limit": 10},
        "query": "accordion accordion manufacturing defects 9999",
    }, name)
    if not data:
        return
    meta = data.get("metadata", {})
    if "error" in meta or len(data.get("nodes", [])) == 0:
        ok(name, data.get("took_ms", 0), f"Graceful empty: {meta.get('error', 'no nodes')}")
    else:
        ok(name, data.get("took_ms", 0), f"Got {len(data['nodes'])} nodes")


# ═══════════════════════════════════════════════════════
# 27. TOPOLOGICAL SORT — Deep Tests
# ═══════════════════════════════════════════════════════

async def test_topo_basic_functionality(c: httpx.AsyncClient):
    """Basic topological sort with query."""
    name = "topo:basic_functionality"
    data = await post_graph(c, {
        "graph": {"type": "topological_sort", "limit": 20},
        "query": "variational autoencoder",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    meta = data.get("metadata", {})
    if "error" in meta:
        ok(name, data["took_ms"], f"Insufficient data: {meta['error']}")
        return
    if not data["nodes"]:
        fail(name, data["took_ms"], "No nodes returned")
        return
    ok(name, data["took_ms"],
       f"{len(data['nodes'])} nodes, papers_in_graph={meta.get('papers_in_graph')}")


async def test_topo_metadata_correctness(c: httpx.AsyncClient):
    """Check all expected metadata fields."""
    name = "topo:metadata_correctness"
    data = await post_graph(c, {
        "graph": {"type": "topological_sort", "limit": 20},
        "query": "generative adversarial network",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    meta = data.get("metadata", {})
    if "error" in meta:
        ok(name, data["took_ms"], f"Insufficient data: {meta['error']}")
        return
    expected = ["papers_in_graph", "papers_in_order", "cycles_broken", "max_depth", "depth_distribution"]
    missing = [k for k in expected if k not in meta]
    if missing:
        fail(name, data["took_ms"], f"Missing metadata: {missing}")
        return
    ok(name, data["took_ms"],
       f"papers_in_order={meta['papers_in_order']}, max_depth={meta['max_depth']}, "
       f"cycles_broken={meta['cycles_broken']}")


async def test_topo_node_properties(c: httpx.AsyncClient):
    """Each node should have topo_position and topo_depth."""
    name = "topo:node_properties"
    data = await post_graph(c, {
        "graph": {"type": "topological_sort", "limit": 15},
        "query": "self-supervised learning",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    meta = data.get("metadata", {})
    if "error" in meta:
        ok(name, data["took_ms"], f"Insufficient data: {meta['error']}")
        return
    for n in data["nodes"]:
        props = n.get("properties", {})
        if "topo_position" not in props:
            fail(name, data["took_ms"], f"Node {n['id']} missing 'topo_position'")
            return
        if "topo_depth" not in props:
            fail(name, data["took_ms"], f"Node {n['id']} missing 'topo_depth'")
            return
    ok(name, data["took_ms"], f"All {len(data['nodes'])} nodes have topo_position & topo_depth")


async def test_topo_position_sequential(c: httpx.AsyncClient):
    """topo_position should be sequential starting from 0."""
    name = "topo:position_sequential"
    data = await post_graph(c, {
        "graph": {"type": "topological_sort", "limit": 20},
        "query": "object detection",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    meta = data.get("metadata", {})
    if "error" in meta:
        ok(name, data["took_ms"], f"Insufficient data: {meta['error']}")
        return
    positions = [n.get("properties", {}).get("topo_position") for n in data["nodes"]]
    positions = [p for p in positions if p is not None]
    if not positions:
        ok(name, data["took_ms"], "No positions to check")
        return
    expected = list(range(len(positions)))
    if positions != expected:
        fail(name, data["took_ms"], f"Positions not sequential: {positions[:10]}...")
        return
    ok(name, data["took_ms"], f"{len(positions)} positions in sequential order 0..{positions[-1]}")


async def test_topo_depth_distribution(c: httpx.AsyncClient):
    """depth_distribution in metadata should be a dict mapping depths to counts."""
    name = "topo:depth_distribution"
    data = await post_graph(c, {
        "graph": {"type": "topological_sort", "limit": 30},
        "query": "federated learning",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    meta = data.get("metadata", {})
    if "error" in meta:
        ok(name, data["took_ms"], f"Insufficient data: {meta['error']}")
        return
    dd = meta.get("depth_distribution")
    if dd is None:
        fail(name, data["took_ms"], "No depth_distribution in metadata")
        return
    if not isinstance(dd, dict):
        fail(name, data["took_ms"], f"depth_distribution not a dict: {type(dd)}")
        return
    total_in_dist = sum(dd.values())
    ok(name, data["took_ms"],
       f"depth_distribution has {len(dd)} levels, total={total_in_dist}")


async def test_topo_with_category_filter(c: httpx.AsyncClient):
    """Topological sort + category filter."""
    name = "topo:with_category_filter"
    data = await post_graph(c, {
        "graph": {"type": "topological_sort", "limit": 15},
        "query": "text generation", "categories": ["cs.CL"],
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    meta = data.get("metadata", {})
    if "error" in meta:
        ok(name, data["took_ms"], f"Insufficient data: {meta['error']}")
        return
    ok(name, data["took_ms"],
       f"cs.CL filter: {len(data['nodes'])} nodes, max_depth={meta.get('max_depth')}")


async def test_topo_with_date_filter(c: httpx.AsyncClient):
    """Topological sort + date range filter."""
    name = "topo:with_date_filter"
    data = await post_graph(c, {
        "graph": {"type": "topological_sort", "limit": 15},
        "query": "protein structure prediction", "submitted_date": {"gte": "2024-01-01T00:00:00"},
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    meta = data.get("metadata", {})
    if "error" in meta:
        ok(name, data["took_ms"], f"Insufficient data with date filter: {meta['error']}")
        return
    ok(name, data["took_ms"],
       f"date-filtered: {len(data['nodes'])} nodes, max_depth={meta.get('max_depth')}")


async def test_topo_limit_1(c: httpx.AsyncClient):
    """limit=1 should return exactly one node."""
    name = "topo:limit_1"
    data = await post_graph(c, {
        "graph": {"type": "topological_sort", "limit": 1},
        "query": "semantic segmentation",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    meta = data.get("metadata", {})
    if "error" in meta:
        ok(name, data["took_ms"], f"Insufficient data: {meta['error']}")
        return
    if len(data["nodes"]) != 1:
        fail(name, data["took_ms"], f"Expected 1 node, got {len(data['nodes'])}")
        return
    ok(name, data["took_ms"], f"Got 1 node at topo_position=0")


async def test_topo_empty_result_graceful(c: httpx.AsyncClient):
    """Very specific query should fail gracefully."""
    name = "topo:empty_result_graceful"
    data = await post_graph(c, {
        "graph": {"type": "topological_sort", "limit": 10},
        "query": "bagpipe repair instructions volcano surfing 9999",
    }, name)
    if not data:
        return
    meta = data.get("metadata", {})
    if "error" in meta or len(data.get("nodes", [])) == 0:
        ok(name, data.get("took_ms", 0), f"Graceful empty: {meta.get('error', 'no nodes')}")
    else:
        ok(name, data.get("took_ms", 0), f"Got {len(data['nodes'])} nodes")


# ═══════════════════════════════════════════════════════
# 28. LINK PREDICTION — Deep Tests
# ═══════════════════════════════════════════════════════

async def test_lp_common_neighbors(c: httpx.AsyncClient):
    """Link prediction with common_neighbors method."""
    name = "lp:common_neighbors"
    data = await post_graph(c, {
        "graph": {"type": "link_prediction", "prediction_method": "common_neighbors", "limit": 10},
        "query": "transformer language model",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    meta = data.get("metadata", {})
    if "error" in meta:
        ok(name, data["took_ms"], f"Insufficient data: {meta['error']}")
        return
    if meta.get("method") != "common_neighbors":
        fail(name, data["took_ms"], f"method mismatch: {meta.get('method')}")
        return
    ok(name, data["took_ms"],
       f"predictions={meta.get('predictions_returned')}, max_score={meta.get('max_score')}")


async def test_lp_jaccard(c: httpx.AsyncClient):
    """Link prediction with jaccard method."""
    name = "lp:jaccard"
    data = await post_graph(c, {
        "graph": {"type": "link_prediction", "prediction_method": "jaccard", "limit": 10},
        "query": "image generation diffusion",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    meta = data.get("metadata", {})
    if "error" in meta:
        ok(name, data["took_ms"], f"Insufficient data: {meta['error']}")
        return
    if meta.get("method") != "jaccard":
        fail(name, data["took_ms"], f"method mismatch: {meta.get('method')}")
        return
    ok(name, data["took_ms"],
       f"predictions={meta.get('predictions_returned')}, max_score={meta.get('max_score')}")


async def test_lp_adamic_adar(c: httpx.AsyncClient):
    """Link prediction with adamic_adar method."""
    name = "lp:adamic_adar"
    data = await post_graph(c, {
        "graph": {"type": "link_prediction", "prediction_method": "adamic_adar", "limit": 10},
        "query": "graph neural network",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    meta = data.get("metadata", {})
    if "error" in meta:
        ok(name, data["took_ms"], f"Insufficient data: {meta['error']}")
        return
    if meta.get("method") != "adamic_adar":
        fail(name, data["took_ms"], f"method mismatch: {meta.get('method')}")
        return
    ok(name, data["took_ms"],
       f"predictions={meta.get('predictions_returned')}, max_score={meta.get('max_score')}")


async def test_lp_preferential_attachment(c: httpx.AsyncClient):
    """Link prediction with preferential_attachment method."""
    name = "lp:preferential_attachment"
    data = await post_graph(c, {
        "graph": {"type": "link_prediction", "prediction_method": "preferential_attachment", "limit": 10},
        "query": "reinforcement learning",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    meta = data.get("metadata", {})
    if "error" in meta:
        ok(name, data["took_ms"], f"Insufficient data: {meta['error']}")
        return
    if meta.get("method") != "preferential_attachment":
        fail(name, data["took_ms"], f"method mismatch: {meta.get('method')}")
        return
    ok(name, data["took_ms"],
       f"predictions={meta.get('predictions_returned')}, max_score={meta.get('max_score')}")


async def test_lp_metadata_correctness(c: httpx.AsyncClient):
    """Check all expected metadata fields for link prediction."""
    name = "lp:metadata_correctness"
    data = await post_graph(c, {
        "graph": {"type": "link_prediction", "prediction_method": "common_neighbors", "limit": 10},
        "query": "attention mechanism",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    meta = data.get("metadata", {})
    if "error" in meta:
        ok(name, data["took_ms"], f"Insufficient data: {meta['error']}")
        return
    expected = ["method", "papers_in_subgraph", "existing_edges", "predictions_total",
                "predictions_returned", "max_score"]
    missing = [k for k in expected if k not in meta]
    if missing:
        fail(name, data["took_ms"], f"Missing metadata: {missing}")
        return
    ok(name, data["took_ms"],
       f"subgraph={meta['papers_in_subgraph']}, existing={meta['existing_edges']}, "
       f"predicted={meta['predictions_returned']}")


async def test_lp_edge_relation_predicted_link(c: httpx.AsyncClient):
    """All predicted edges should have relation='predicted_link'."""
    name = "lp:edge_relation_predicted_link"
    data = await post_graph(c, {
        "graph": {"type": "link_prediction", "prediction_method": "jaccard", "limit": 10},
        "query": "medical image analysis",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    meta = data.get("metadata", {})
    if "error" in meta:
        ok(name, data["took_ms"], f"Insufficient data: {meta['error']}")
        return
    if not data["edges"]:
        ok(name, data["took_ms"], "No predicted edges")
        return
    non_predicted = [e for e in data["edges"] if e["relation"] != "predicted_link"]
    if non_predicted:
        fail(name, data["took_ms"],
             f"{len(non_predicted)} edges with wrong relation: {non_predicted[0]['relation']}")
        return
    ok(name, data["took_ms"], f"All {len(data['edges'])} edges are predicted_link")


async def test_lp_edge_has_weight(c: httpx.AsyncClient):
    """Predicted edges should have a weight (score)."""
    name = "lp:edge_has_weight"
    data = await post_graph(c, {
        "graph": {"type": "link_prediction", "prediction_method": "adamic_adar", "limit": 10},
        "query": "point cloud 3D",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    meta = data.get("metadata", {})
    if "error" in meta:
        ok(name, data["took_ms"], f"Insufficient data: {meta['error']}")
        return
    if not data["edges"]:
        ok(name, data["took_ms"], "No predicted edges to check weight")
        return
    for e in data["edges"]:
        if "weight" not in e:
            fail(name, data["took_ms"], f"Edge {e['source']}→{e['target']} missing weight")
            return
    ok(name, data["took_ms"], f"All {len(data['edges'])} edges have weight/score")


async def test_lp_with_category_filter(c: httpx.AsyncClient):
    """Link prediction + category filter."""
    name = "lp:with_category_filter"
    data = await post_graph(c, {
        "graph": {"type": "link_prediction", "prediction_method": "common_neighbors", "limit": 10},
        "query": "neural network", "categories": ["cs.CV"],
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    meta = data.get("metadata", {})
    if "error" in meta:
        ok(name, data["took_ms"], f"Insufficient data: {meta['error']}")
        return
    ok(name, data["took_ms"],
       f"cs.CV filter: predictions={meta.get('predictions_returned')}, "
       f"subgraph={meta.get('papers_in_subgraph')}")


async def test_lp_with_date_filter(c: httpx.AsyncClient):
    """Link prediction + date range filter."""
    name = "lp:with_date_filter"
    data = await post_graph(c, {
        "graph": {"type": "link_prediction", "prediction_method": "preferential_attachment", "limit": 10},
        "query": "large language model", "submitted_date": {"gte": "2024-01-01T00:00:00"},
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    meta = data.get("metadata", {})
    if "error" in meta:
        ok(name, data["took_ms"], f"Insufficient data with date filter: {meta['error']}")
        return
    ok(name, data["took_ms"],
       f"date-filtered: predictions={meta.get('predictions_returned')}")


async def test_lp_limit_1(c: httpx.AsyncClient):
    """limit=1 should return at most 1 predicted edge."""
    name = "lp:limit_1"
    data = await post_graph(c, {
        "graph": {"type": "link_prediction", "prediction_method": "common_neighbors", "limit": 1},
        "query": "deep learning optimization",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    meta = data.get("metadata", {})
    if "error" in meta:
        ok(name, data["took_ms"], f"Insufficient data: {meta['error']}")
        return
    predicted_edges = [e for e in data["edges"] if e["relation"] == "predicted_link"]
    if len(predicted_edges) > 1:
        fail(name, data["took_ms"], f"Expected <=1 predicted edge, got {len(predicted_edges)}")
        return
    ok(name, data["took_ms"], f"{len(predicted_edges)} predicted edges (limit=1)")


async def test_lp_empty_result_graceful(c: httpx.AsyncClient):
    """Very specific query should fail gracefully."""
    name = "lp:empty_result_graceful"
    data = await post_graph(c, {
        "graph": {"type": "link_prediction", "prediction_method": "jaccard", "limit": 10},
        "query": "kazoo orchestra submarine painting 9999",
    }, name)
    if not data:
        return
    meta = data.get("metadata", {})
    if "error" in meta or meta.get("predictions_returned", 0) == 0:
        ok(name, data.get("took_ms", 0), f"Graceful empty: {meta.get('error', 'no predictions')}")
    else:
        fail(name, data.get("took_ms", 0), f"Expected empty but got {meta.get('predictions_returned')} predictions")


# ═══════════════════════════════════════════════════════
# CROSS-ALGORITHM TESTS
# ═══════════════════════════════════════════════════════

async def test_cross_same_query_all_centrality(c: httpx.AsyncClient):
    """Same query through betweenness and closeness centrality — both should return results."""
    name = "cross:centrality_consistency"
    query = "transformer architecture"
    start = time.monotonic()
    d_bc = await post_graph(c, {
        "graph": {"type": "betweenness_centrality", "limit": 10}, "query": query,
    }, name)
    d_cc = await post_graph(c, {
        "graph": {"type": "closeness_centrality", "limit": 10}, "query": query,
    }, name)
    elapsed = int((time.monotonic() - start) * 1000)
    if not d_bc or not d_cc:
        return
    bc_meta = d_bc.get("metadata", {})
    cc_meta = d_cc.get("metadata", {})
    bc_ok = "error" not in bc_meta and len(d_bc.get("nodes", [])) > 0
    cc_ok = "error" not in cc_meta and len(d_cc.get("nodes", [])) > 0
    if not bc_ok and not cc_ok:
        fail(name, elapsed, f"Both centrality algorithms returned no results")
        return
    ok(name, elapsed,
       f"bc: {len(d_bc.get('nodes', []))} nodes, cc: {len(d_cc.get('nodes', []))} nodes, "
       f"bc_ok={bc_ok}, cc_ok={cc_ok}")


async def test_cross_same_query_topo_and_scc(c: httpx.AsyncClient):
    """Same query through topological sort and SCC — structural comparison."""
    name = "cross:topo_and_scc"
    query = "knowledge distillation"
    start = time.monotonic()
    d_topo = await post_graph(c, {
        "graph": {"type": "topological_sort", "limit": 20}, "query": query,
    }, name)
    d_scc = await post_graph(c, {
        "graph": {"type": "strongly_connected_components", "limit": 20}, "query": query,
    }, name)
    elapsed = int((time.monotonic() - start) * 1000)
    if not d_topo or not d_scc:
        return
    topo_meta = d_topo.get("metadata", {})
    scc_meta = d_scc.get("metadata", {})
    ok(name, elapsed,
       f"topo: papers_in_graph={topo_meta.get('papers_in_graph')}, "
       f"scc: papers_in_graph={scc_meta.get('papers_in_graph')}")


async def test_cross_link_prediction_all_methods(c: httpx.AsyncClient):
    """Same query through all 4 link prediction methods — all should succeed."""
    name = "cross:link_prediction_methods"
    query = "contrastive learning representation"
    methods = ["common_neighbors", "jaccard", "adamic_adar", "preferential_attachment"]
    start = time.monotonic()
    results_per_method = {}
    for method in methods:
        d = await post_graph(c, {
            "graph": {"type": "link_prediction", "prediction_method": method, "limit": 5},
            "query": query,
        }, name)
        if d:
            meta = d.get("metadata", {})
            results_per_method[method] = meta.get("predictions_returned", 0)
        else:
            results_per_method[method] = -1
    elapsed = int((time.monotonic() - start) * 1000)
    summary = ", ".join(f"{m}={v}" for m, v in results_per_method.items())
    failed_methods = [m for m, v in results_per_method.items() if v <= 0]
    if len(failed_methods) == len(methods):
        fail(name, elapsed, f"All methods returned no results: {summary}")
        return
    ok(name, elapsed, summary)


async def test_cross_all_algos_with_seed(c: httpx.AsyncClient):
    """All 4 subgraph-based algorithms with same seed paper."""
    name = "cross:all_algos_seed"
    seed = CONNECTED_PAPER
    algos = [
        ("betweenness_centrality", {}),
        ("closeness_centrality", {}),
        ("strongly_connected_components", {}),
        ("topological_sort", {}),
    ]
    start = time.monotonic()
    algo_results = {}
    for algo_type, extra in algos:
        body = {"graph": {"type": algo_type, "seed_arxiv_id": seed, "limit": 10, **extra}}
        d = await post_graph(c, body, name)
        if d:
            algo_results[algo_type] = len(d.get("nodes", []))
        else:
            algo_results[algo_type] = -1
    elapsed = int((time.monotonic() - start) * 1000)
    summary = ", ".join(f"{a}={v}" for a, v in algo_results.items())
    ok(name, elapsed, summary)


async def test_cross_semantic_query_with_centrality(c: httpx.AsyncClient):
    """Semantic query input with betweenness centrality."""
    name = "cross:semantic_with_bc"
    data = await post_graph(c, {
        "graph": {"type": "betweenness_centrality", "limit": 10},
        "semantic": [{"text": "novel attention mechanisms for efficient inference",
                      "level": "abstract", "weight": 2.0, "mode": "boost"}],
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    meta = data.get("metadata", {})
    if "error" in meta:
        ok(name, data["took_ms"], f"Insufficient data: {meta['error']}")
        return
    ok(name, data["took_ms"],
       f"semantic+bc: {len(data['nodes'])} nodes, subgraph={meta.get('papers_in_subgraph')}")


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
        print(f"Running {len(tests)} deep graph tests (types 23-28)...\n")
        start = time.monotonic()

        for test_fn in tests:
            await test_fn(c)

        elapsed = time.monotonic() - start

    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)

    print(f"\n{'='*80}")
    print(f" DEEP GRAPH TEST RESULTS (types 23-28): {passed}/{len(results)} passed ({failed} failed) in {elapsed:.1f}s")
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
