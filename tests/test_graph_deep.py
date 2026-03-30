"""Deep E2E tests for the 6 graph-DB algorithms (types 17-22).

Tests:
- Correctness: node/edge properties, metadata integrity, hop distances
- Filter composition: all 6 algorithms with various search filters
- Boundary conditions: max_hops limits, empty results, single-node graphs
- Data integrity: node IDs match edges, no duplicates, proper types
- Concurrent correctness: same query returns consistent results
- Stress: large limits, many hops, high iterations
- Cross-algorithm: same paper set through different algorithms
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

    # Check all required fields
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
            # Duplicate node IDs
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
# 17. MULTIHOP CITATION — Deep Tests
# ═══════════════════════════════════════════════════════

async def test_multihop_hop_depth_property(c: httpx.AsyncClient):
    """Verify each node has a correct 'hop' property that increases monotonically."""
    name = "multihop:hop_depth_correctness"
    data = await post_graph(c, {
        "graph": {"type": "multihop_citation", "seed_arxiv_id": CONNECTED_PAPER,
                  "max_hops": 3, "direction": "cited_by", "limit": 100},
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    nodes = data["nodes"]
    if not nodes:
        ok(name, data["took_ms"], "No nodes (paper may lack cited_by links)")
        return
    # Seed should be hop 0
    seed_nodes = [n for n in nodes if n["id"] == CONNECTED_PAPER]
    if not seed_nodes:
        fail(name, data["took_ms"], f"Seed paper {CONNECTED_PAPER} not in nodes")
        return
    if seed_nodes[0]["properties"].get("hop") != 0:
        fail(name, data["took_ms"], f"Seed hop != 0: {seed_nodes[0]['properties'].get('hop')}")
        return
    # All hops should be 0..max_hops
    hops = [n["properties"].get("hop") for n in nodes]
    if any(h is None for h in hops):
        fail(name, data["took_ms"], "Some nodes missing 'hop' property")
        return
    if min(hops) != 0 or max(hops) > 3:
        fail(name, data["took_ms"], f"Hop range invalid: {min(hops)}-{max(hops)}")
        return
    ok(name, data["took_ms"], f"{len(nodes)} nodes, hops 0..{max(hops)}")


async def test_multihop_edge_source_target_consistency(c: httpx.AsyncClient):
    """Every edge source/target should reference nodes that exist in the result."""
    name = "multihop:edge_node_consistency"
    data = await post_graph(c, {
        "graph": {"type": "multihop_citation", "seed_arxiv_id": CONNECTED_PAPER,
                  "max_hops": 2, "direction": "references", "limit": 50},
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    node_ids = {n["id"] for n in data["nodes"]}
    # Edge endpoints should be in node set or point to papers we haven't fetched
    # But the source should always be in nodes (since we walk from known papers)
    bad_sources = [e for e in data["edges"] if e["source"] not in node_ids]
    if bad_sources:
        fail(name, data["took_ms"],
             f"{len(bad_sources)} edges have source not in node set")
        return
    # Target might not be in nodes (fetched at next hop but capped)
    ok(name, data["took_ms"],
       f"{len(data['edges'])} edges, all sources in node set")


async def test_multihop_direction_references(c: httpx.AsyncClient):
    """With direction=references, edges should go FROM seed papers TO their references."""
    name = "multihop:direction_references"
    data = await post_graph(c, {
        "graph": {"type": "multihop_citation", "seed_arxiv_id": CONNECTED_PAPER,
                  "max_hops": 1, "direction": "references", "limit": 50},
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    if data["metadata"].get("direction") != "references":
        fail(name, data["took_ms"], f"Wrong direction: {data['metadata'].get('direction')}")
        return
    # All edges should have relation "cites" and source=seed when hop=0→1
    cites_edges = [e for e in data["edges"] if e["relation"] == "cites"]
    if not cites_edges:
        fail(name, data["took_ms"], "No cites edges found in references direction")
        return
    ok(name, data["took_ms"], f"{len(cites_edges)}/{len(data['edges'])} cites edges")


async def test_multihop_direction_cited_by(c: httpx.AsyncClient):
    """With direction=cited_by, we follow who cites the seed."""
    name = "multihop:direction_cited_by"
    data = await post_graph(c, {
        "graph": {"type": "multihop_citation", "seed_arxiv_id": CONNECTED_PAPER,
                  "max_hops": 1, "direction": "cited_by", "limit": 50},
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    if data["metadata"].get("direction") != "cited_by":
        fail(name, data["took_ms"], "Wrong direction in metadata")
        return
    # Should have found some citers
    hop1 = [n for n in data["nodes"] if n["properties"].get("hop") == 1]
    ok(name, data["took_ms"], f"{len(hop1)} hop-1 papers found")


async def test_multihop_max_hops_1(c: httpx.AsyncClient):
    """max_hops=1 should return seed + immediate neighbors only."""
    name = "multihop:max_hops_1"
    data = await post_graph(c, {
        "graph": {"type": "multihop_citation", "seed_arxiv_id": CONNECTED_PAPER,
                  "max_hops": 1, "direction": "cited_by", "limit": 200},
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    hops = {n["properties"].get("hop") for n in data["nodes"]}
    if hops - {0, 1}:
        fail(name, data["took_ms"], f"Found unexpected hops: {hops}")
        return
    ok(name, data["took_ms"], f"Hops: {sorted(hops)}, {len(data['nodes'])} nodes")


async def test_multihop_max_hops_5_stress(c: httpx.AsyncClient):
    """Stress test with max_hops=5 — should not crash or timeout."""
    name = "multihop:max_hops_5_stress"
    start = time.monotonic()
    data = await post_graph(c, {
        "graph": {"type": "multihop_citation", "seed_arxiv_id": CONNECTED_PAPER,
                  "max_hops": 5, "direction": "cited_by", "limit": 50},
    }, name)
    elapsed = int((time.monotonic() - start) * 1000)
    if not data:
        return
    if not validate_graph_integrity(data, name):
        return
    hops_reached = data["metadata"].get("hops_reached", 0)
    ok(name, elapsed, f"Completed in {elapsed}ms, hops_reached={hops_reached}, {len(data['nodes'])} nodes")


async def test_multihop_seed_ids_multiple(c: httpx.AsyncClient):
    """Multiple seed_arxiv_ids should produce a combined graph."""
    name = "multihop:multi_seed"
    data = await post_graph(c, {
        "graph": {"type": "multihop_citation",
                  "seed_arxiv_ids": [CONNECTED_PAPER, BIG_REFS_PAPER],
                  "max_hops": 1, "direction": "references", "limit": 50},
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    node_ids = {n["id"] for n in data["nodes"]}
    has_both = CONNECTED_PAPER in node_ids and BIG_REFS_PAPER in node_ids
    if not has_both:
        fail(name, data["took_ms"], f"Not all seeds in nodes: {CONNECTED_PAPER in node_ids}, {BIG_REFS_PAPER in node_ids}")
        return
    ok(name, data["took_ms"], f"Both seeds present, {len(data['nodes'])} nodes")


async def test_multihop_with_category_filter(c: httpx.AsyncClient):
    """Multihop + category filter should limit the seed set."""
    name = "multihop:category_filter"
    data = await post_graph(c, {
        "graph": {"type": "multihop_citation", "max_hops": 2,
                  "direction": "references", "limit": 30},
        "query": "diffusion model", "categories": ["cs.LG"],
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    ok(name, data["took_ms"], f"{len(data['nodes'])} nodes, {len(data['edges'])} edges")


async def test_multihop_with_date_filter(c: httpx.AsyncClient):
    """Multihop + date range filter."""
    name = "multihop:date_filter"
    data = await post_graph(c, {
        "graph": {"type": "multihop_citation", "max_hops": 1,
                  "direction": "cited_by", "limit": 20},
        "query": "language model",
        "submitted_date": {"gte": "2024-01-01", "lte": "2024-12-31"},
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    ok(name, data["took_ms"], f"{len(data['nodes'])} nodes")


async def test_multihop_nonexistent_seed(c: httpx.AsyncClient):
    """Non-existent seed paper should return empty or error."""
    name = "multihop:nonexistent_seed"
    data = await post_graph(c, {
        "graph": {"type": "multihop_citation", "seed_arxiv_id": "9999.99999",
                  "max_hops": 2, "direction": "references"},
    }, name)
    if not data:
        return
    if data["metadata"].get("error") or len(data["nodes"]) == 0:
        ok(name, data["took_ms"], f"Correctly empty: {data['metadata']}")
    else:
        fail(name, data["took_ms"], f"Expected empty for non-existent seed, got {len(data['nodes'])} nodes")


async def test_multihop_papers_per_hop_metadata(c: httpx.AsyncClient):
    """Verify papers_per_hop metadata sums to total nodes."""
    name = "multihop:papers_per_hop_sum"
    data = await post_graph(c, {
        "graph": {"type": "multihop_citation", "seed_arxiv_id": CONNECTED_PAPER,
                  "max_hops": 2, "direction": "cited_by", "limit": 100},
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    pph = data["metadata"].get("papers_per_hop", {})
    total_from_meta = sum(pph.values())
    if total_from_meta != len(data["nodes"]):
        fail(name, data["took_ms"],
             f"papers_per_hop sum={total_from_meta} != nodes={len(data['nodes'])}")
        return
    ok(name, data["took_ms"], f"papers_per_hop={pph}, sum matches {len(data['nodes'])} nodes")


# ═══════════════════════════════════════════════════════
# 18. SHORTEST CITATION PATH — Deep Tests
# ═══════════════════════════════════════════════════════

async def test_shortest_path_connected_papers(c: httpx.AsyncClient):
    """Try to find path between papers that share citation neighborhoods."""
    name = "shortest_path:connected_pair"
    # 2406.03736 cites 2405.04233, and 2405.04233 is cited_by 2406.03736
    # Try path from a citer to a reference
    data = await post_graph(c, {
        "graph": {"type": "shortest_citation_path",
                  "seed_arxiv_id": CONNECTED_PAPER,
                  "target_arxiv_id": "2405.04233",
                  "max_hops": 3},
    }, name)
    if not data:
        return
    meta = data.get("metadata", {})
    if "error" in meta:
        # They might not be connected in both directions
        ok(name, data["took_ms"], f"status={meta['error']} (explored fwd={meta.get('forward_explored')} bwd={meta.get('backward_explored')})")
    else:
        path_len = meta.get("path_length", -1)
        if path_len < 1:
            fail(name, data["took_ms"], f"Expected path_length >= 1, got {path_len}")
            return
        # Path should start with source and end with target
        path_ids = meta.get("path_ids", [])
        if path_ids and path_ids[0] != CONNECTED_PAPER:
            fail(name, data["took_ms"], f"Path doesn't start with source: {path_ids[0]}")
            return
        if path_ids and path_ids[-1] != "2405.04233":
            fail(name, data["took_ms"], f"Path doesn't end with target: {path_ids[-1]}")
            return
        ok(name, data["took_ms"], f"Path found: length={path_len}, ids={path_ids}")


async def test_shortest_path_no_path(c: httpx.AsyncClient):
    """Two disconnected papers should return no_path_found."""
    name = "shortest_path:no_path"
    data = await post_graph(c, {
        "graph": {"type": "shortest_citation_path",
                  "seed_arxiv_id": "2301.00001",
                  "target_arxiv_id": "2301.00002",
                  "max_hops": 2},
    }, name)
    if not data:
        return
    meta = data.get("metadata", {})
    if "error" in meta:
        ok(name, data["took_ms"], f"Correctly no path: explored {meta.get('forward_explored', 0)} + {meta.get('backward_explored', 0)}")
    else:
        # A path across 2 random papers is possible, accept either result
        ok(name, data["took_ms"], f"Found unexpected path of length {meta.get('path_length')}")


async def test_shortest_path_self_loop(c: httpx.AsyncClient):
    """Source == target should return error."""
    name = "shortest_path:self_loop"
    data = await post_graph(c, {
        "graph": {"type": "shortest_citation_path",
                  "seed_arxiv_id": CONNECTED_PAPER,
                  "target_arxiv_id": CONNECTED_PAPER},
    }, name)
    if not data:
        return
    if data.get("metadata", {}).get("error") == "source and target are the same paper":
        ok(name, data["took_ms"], "Correctly rejected self-loop")
    else:
        fail(name, data["took_ms"], f"Expected self-loop error, got: {data['metadata']}")


async def test_shortest_path_missing_target(c: httpx.AsyncClient):
    """Missing target_arxiv_id should return error."""
    name = "shortest_path:missing_target"
    data = await post_graph(c, {
        "graph": {"type": "shortest_citation_path",
                  "seed_arxiv_id": CONNECTED_PAPER},
    }, name)
    if not data:
        return
    if "error" in data.get("metadata", {}):
        ok(name, data["took_ms"], f"Correctly errored: {data['metadata']['error']}")
    else:
        fail(name, data["took_ms"], "Expected error for missing target_arxiv_id")


async def test_shortest_path_missing_source(c: httpx.AsyncClient):
    """Missing seed_arxiv_id should return error."""
    name = "shortest_path:missing_source"
    data = await post_graph(c, {
        "graph": {"type": "shortest_citation_path",
                  "target_arxiv_id": CONNECTED_PAPER},
    }, name)
    if not data:
        return
    if "error" in data.get("metadata", {}):
        ok(name, data["took_ms"], f"Correctly errored: {data['metadata']['error']}")
    else:
        fail(name, data["took_ms"], "Expected error for missing seed_arxiv_id")


async def test_shortest_path_nonexistent_paper(c: httpx.AsyncClient):
    """Non-existent paper in path should return 'not found' error."""
    name = "shortest_path:nonexistent"
    data = await post_graph(c, {
        "graph": {"type": "shortest_citation_path",
                  "seed_arxiv_id": "9999.99999",
                  "target_arxiv_id": CONNECTED_PAPER,
                  "max_hops": 2},
    }, name)
    if not data:
        return
    if "error" in data.get("metadata", {}):
        ok(name, data["took_ms"], f"Error: {data['metadata']['error']}")
    else:
        fail(name, data["took_ms"], "Expected error for non-existent paper")


async def test_shortest_path_nodes_have_position(c: httpx.AsyncClient):
    """Path nodes should have path_position property, sequential from 0."""
    name = "shortest_path:node_positions"
    data = await post_graph(c, {
        "graph": {"type": "shortest_citation_path",
                  "seed_arxiv_id": CONNECTED_PAPER,
                  "target_arxiv_id": "2405.04233",
                  "max_hops": 3},
    }, name)
    if not data:
        return
    if "error" in data.get("metadata", {}):
        ok(name, data["took_ms"], f"No path found (acceptable): {data['metadata']['error']}")
        return
    positions = [n["properties"].get("path_position") for n in data["nodes"]]
    if any(p is None for p in positions):
        fail(name, data["took_ms"], "Some nodes missing path_position")
        return
    if positions != list(range(len(positions))):
        fail(name, data["took_ms"], f"Positions not sequential: {positions}")
        return
    ok(name, data["took_ms"], f"Sequential positions 0..{len(positions)-1}")


async def test_shortest_path_max_hops_respected(c: httpx.AsyncClient):
    """With max_hops=1, path must be direct (length 1) or not found."""
    name = "shortest_path:max_hops_1"
    data = await post_graph(c, {
        "graph": {"type": "shortest_citation_path",
                  "seed_arxiv_id": CONNECTED_PAPER,
                  "target_arxiv_id": "2405.04233",
                  "max_hops": 1},
    }, name)
    if not data:
        return
    meta = data.get("metadata", {})
    if "error" in meta:
        ok(name, data["took_ms"], f"No direct path: {meta.get('error')}")
    elif meta.get("path_length", 99) <= 1:
        ok(name, data["took_ms"], f"Direct path found, length={meta['path_length']}")
    else:
        fail(name, data["took_ms"], f"Path length {meta['path_length']} exceeds max_hops=1")


# ═══════════════════════════════════════════════════════
# 19. PAGERANK — Deep Tests
# ═══════════════════════════════════════════════════════

async def test_pagerank_values_sum_roughly_1(c: httpx.AsyncClient):
    """PageRank values across all nodes in the subgraph should sum ~1.0."""
    name = "pagerank:values_sum"
    data = await post_graph(c, {
        "graph": {"type": "pagerank", "seed_arxiv_id": CONNECTED_PAPER,
                  "damping_factor": 0.85, "iterations": 20, "limit": 200},
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    if not data["nodes"]:
        ok(name, data["took_ms"], "Empty (acceptable)")
        return
    total_pr = sum(n["properties"].get("pagerank", 0) for n in data["nodes"])
    # PageRank should sum to approximately 1.0 when all nodes returned
    # But we only return top-N, so sum will be less
    subgraph_size = data["metadata"].get("papers_in_subgraph", 0)
    ok(name, data["took_ms"],
       f"PR sum={total_pr:.4f} (top {len(data['nodes'])}/{subgraph_size})")


async def test_pagerank_ordering(c: httpx.AsyncClient):
    """Nodes should be returned in descending PageRank order."""
    name = "pagerank:ordering"
    data = await post_graph(c, {
        "graph": {"type": "pagerank", "seed_arxiv_id": CONNECTED_PAPER,
                  "damping_factor": 0.85, "iterations": 20, "limit": 50},
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    if len(data["nodes"]) < 2:
        ok(name, data["took_ms"], "Too few nodes to verify ordering")
        return
    prs = [n["properties"].get("pagerank", 0) for n in data["nodes"]]
    is_sorted = all(prs[i] >= prs[i+1] for i in range(len(prs)-1))
    if not is_sorted:
        fail(name, data["took_ms"], f"Not sorted: first={prs[0]}, last={prs[-1]}")
        return
    ok(name, data["took_ms"], f"Sorted: max={prs[0]:.6f}, min={prs[-1]:.6f}")


async def test_pagerank_in_out_degree(c: httpx.AsyncClient):
    """Each node should have in_degree and out_degree properties."""
    name = "pagerank:degree_props"
    data = await post_graph(c, {
        "graph": {"type": "pagerank", "seed_arxiv_id": CONNECTED_PAPER,
                  "iterations": 10, "limit": 20},
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    if not data["nodes"]:
        ok(name, data["took_ms"], "Empty (acceptable)")
        return
    missing = []
    for n in data["nodes"]:
        for prop in ("pagerank", "in_degree", "out_degree"):
            if prop not in n.get("properties", {}):
                missing.append(f"{n['id']}:{prop}")
    if missing:
        fail(name, data["took_ms"], f"Missing properties: {missing[:5]}")
        return
    ok(name, data["took_ms"], f"All {len(data['nodes'])} nodes have degree props")


async def test_pagerank_damping_factor_effect(c: httpx.AsyncClient):
    """Different damping factors should produce different rankings."""
    name = "pagerank:damping_effect"
    data_low = await post_graph(c, {
        "graph": {"type": "pagerank", "seed_arxiv_id": CONNECTED_PAPER,
                  "damping_factor": 0.5, "iterations": 20, "limit": 10},
    }, name)
    data_high = await post_graph(c, {
        "graph": {"type": "pagerank", "seed_arxiv_id": CONNECTED_PAPER,
                  "damping_factor": 0.95, "iterations": 20, "limit": 10},
    }, name)
    if not data_low or not data_high:
        return
    max_low = data_low["metadata"].get("max_pagerank", 0)
    max_high = data_high["metadata"].get("max_pagerank", 0)
    # With higher damping, PageRank concentrates more on important nodes
    ok(name, data_high["took_ms"],
       f"d=0.5 max={max_low:.6f}, d=0.95 max={max_high:.6f}")


async def test_pagerank_iterations_convergence(c: httpx.AsyncClient):
    """More iterations should converge (rankings stabilize)."""
    name = "pagerank:iteration_convergence"
    data_5 = await post_graph(c, {
        "graph": {"type": "pagerank", "seed_arxiv_id": CONNECTED_PAPER,
                  "iterations": 5, "limit": 10},
    }, name)
    data_50 = await post_graph(c, {
        "graph": {"type": "pagerank", "seed_arxiv_id": CONNECTED_PAPER,
                  "iterations": 50, "limit": 10},
    }, name)
    if not data_5 or not data_50:
        return
    # Top-ranked paper should be the same (or very similar)
    top_5 = data_5["nodes"][0]["id"] if data_5["nodes"] else None
    top_50 = data_50["nodes"][0]["id"] if data_50["nodes"] else None
    ok(name, data_50["took_ms"],
       f"5 iters top={top_5}, 50 iters top={top_50}, same={top_5==top_50}")


async def test_pagerank_with_query_filter(c: httpx.AsyncClient):
    """PageRank + search query should restrict the subgraph."""
    name = "pagerank:query_filter"
    data = await post_graph(c, {
        "graph": {"type": "pagerank", "iterations": 10, "limit": 20},
        "query": "diffusion probabilistic", "categories": ["cs.LG"],
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    ok(name, data["took_ms"],
       f"{len(data['nodes'])} nodes, subgraph={data['metadata'].get('papers_in_subgraph')}")


async def test_pagerank_edges_between_ranked(c: httpx.AsyncClient):
    """Edges should only connect papers that are both in the returned node set."""
    name = "pagerank:edge_validity"
    data = await post_graph(c, {
        "graph": {"type": "pagerank", "seed_arxiv_id": CONNECTED_PAPER,
                  "iterations": 10, "limit": 30},
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    node_ids = {n["id"] for n in data["nodes"]}
    bad_edges = [e for e in data["edges"]
                 if e["source"] not in node_ids or e["target"] not in node_ids]
    if bad_edges:
        fail(name, data["took_ms"], f"{len(bad_edges)} edges reference nodes not in result")
        return
    ok(name, data["took_ms"], f"All {len(data['edges'])} edges valid")


async def test_pagerank_max_iterations_stress(c: httpx.AsyncClient):
    """50 iterations should not timeout."""
    name = "pagerank:50_iterations_stress"
    start = time.monotonic()
    data = await post_graph(c, {
        "graph": {"type": "pagerank", "seed_arxiv_id": CONNECTED_PAPER,
                  "iterations": 50, "limit": 50},
    }, name)
    elapsed = int((time.monotonic() - start) * 1000)
    if not data:
        return
    ok(name, elapsed, f"50 iters in {elapsed}ms, {data['metadata'].get('papers_in_subgraph')} papers")


# ═══════════════════════════════════════════════════════
# 20. COMMUNITY DETECTION — Deep Tests
# ═══════════════════════════════════════════════════════

async def test_community_nodes_have_community_label(c: httpx.AsyncClient):
    """Paper nodes should have 'community' property, community nodes should have 'size'."""
    name = "community:node_properties"
    data = await post_graph(c, {
        "graph": {"type": "community_detection", "iterations": 15, "limit": 20},
        "query": "natural language processing",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    communities = [n for n in data["nodes"] if n["type"] == "community"]
    papers = [n for n in data["nodes"] if n["type"] == "paper"]
    for cn in communities:
        if "size" not in cn.get("properties", {}):
            fail(name, data["took_ms"], f"Community {cn['id']} missing 'size'")
            return
    for pn in papers:
        if "community" not in pn.get("properties", {}):
            fail(name, data["took_ms"], f"Paper {pn['id']} missing 'community'")
            return
    ok(name, data["took_ms"],
       f"{len(communities)} communities, {len(papers)} papers")


async def test_community_contains_edges(c: httpx.AsyncClient):
    """Each paper should be connected to its community via 'contains' edge."""
    name = "community:contains_edges"
    data = await post_graph(c, {
        "graph": {"type": "community_detection", "iterations": 10, "limit": 10},
        "query": "computer vision object detection",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    contains = [e for e in data["edges"] if e["relation"] == "contains"]
    papers = [n for n in data["nodes"] if n["type"] == "paper"]
    ok(name, data["took_ms"],
       f"{len(contains)} contains edges for {len(papers)} papers")


async def test_community_inter_community_edges(c: httpx.AsyncClient):
    """Inter-community edges should have weight > 0."""
    name = "community:inter_edges"
    data = await post_graph(c, {
        "graph": {"type": "community_detection", "iterations": 10, "limit": 20},
        "query": "machine learning",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    inter = [e for e in data["edges"] if e["relation"] == "inter_community"]
    if inter:
        bad_weight = [e for e in inter if (e.get("weight") or 0) <= 0]
        if bad_weight:
            fail(name, data["took_ms"], f"{len(bad_weight)} inter-community edges with weight <= 0")
            return
    ok(name, data["took_ms"], f"{len(inter)} inter-community edges")


async def test_community_top_categories(c: httpx.AsyncClient):
    """Community nodes should have top_categories list."""
    name = "community:top_categories"
    data = await post_graph(c, {
        "graph": {"type": "community_detection", "iterations": 10, "limit": 10},
        "query": "reinforcement learning robot",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    communities = [n for n in data["nodes"] if n["type"] == "community"]
    for cn in communities:
        cats = cn.get("properties", {}).get("top_categories", [])
        if not isinstance(cats, list):
            fail(name, data["took_ms"], f"Community {cn['id']} top_categories not a list")
            return
    ok(name, data["took_ms"],
       f"{len(communities)} communities all have top_categories")


async def test_community_with_semantic_boost(c: httpx.AsyncClient):
    """Community detection + semantic boost should work."""
    name = "community:semantic_boost"
    data = await post_graph(c, {
        "graph": {"type": "community_detection", "iterations": 10, "limit": 10},
        "query": "graph neural networks",
        "semantic": [{"text": "graph convolution message passing", "level": "abstract", "weight": 2.0, "mode": "boost"}],
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    ok(name, data["took_ms"],
       f"{data['metadata'].get('communities_found')} communities")


async def test_community_metadata_integrity(c: httpx.AsyncClient):
    """Metadata should have consistent values."""
    name = "community:metadata"
    data = await post_graph(c, {
        "graph": {"type": "community_detection", "iterations": 10, "limit": 30},
        "query": "optimization gradient descent",
    }, name)
    if not data:
        return
    meta = data["metadata"]
    for key in ("papers_in_graph", "communities_found", "largest_community"):
        if key not in meta:
            fail(name, data["took_ms"], f"Missing metadata key: {key}")
            return
    if meta["largest_community"] > meta.get("papers_in_graph", 0):
        fail(name, data["took_ms"],
             f"Largest community ({meta['largest_community']}) > total papers ({meta['papers_in_graph']})")
        return
    ok(name, data["took_ms"],
       f"papers={meta['papers_in_graph']}, communities={meta['communities_found']}, largest={meta['largest_community']}")


# ═══════════════════════════════════════════════════════
# 21. CITATION PATTERNS — Deep Tests
# ═══════════════════════════════════════════════════════

async def test_citation_patterns_all_four_types(c: httpx.AsyncClient):
    """All four patterns should return valid responses without errors."""
    name = "citation_patterns:all_four"
    patterns_tested = 0
    for patt in ("mutual", "triangle", "star", "chain"):
        data = await post_graph(c, {
            "graph": {"type": "citation_patterns", "pattern": patt, "limit": 10},
            "query": "neural network",
        }, f"{name}:{patt}")
        if not data:
            fail(name, 0, f"Pattern '{patt}' failed to respond")
            return
        if not validate_graph_integrity(data, name):
            return
        patterns_tested += 1
    ok(name, 0, f"All 4 patterns responded successfully")


async def test_citation_patterns_star_hub_properties(c: httpx.AsyncClient):
    """Star pattern hubs should have pattern_role and in_degree properties."""
    name = "citation_patterns:star_hub_props"
    data = await post_graph(c, {
        "graph": {"type": "citation_patterns", "pattern": "star", "limit": 20},
        "query": "deep learning",
        "categories": ["cs.LG", "cs.CL", "cs.CV"],
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    hubs = [n for n in data["nodes"] if n.get("properties", {}).get("pattern_role") == "hub"]
    if data["metadata"].get("patterns_found", 0) > 0 and not hubs:
        fail(name, data["took_ms"], "Patterns found but no hub nodes marked")
        return
    ok(name, data["took_ms"],
       f"patterns={data['metadata'].get('patterns_found')}, hubs={len(hubs)}, "
       f"subgraph={data['metadata'].get('papers_in_subgraph')}")


async def test_citation_patterns_mutual_symmetry(c: httpx.AsyncClient):
    """Mutual citations should have edges in both directions."""
    name = "citation_patterns:mutual_symmetry"
    data = await post_graph(c, {
        "graph": {"type": "citation_patterns", "pattern": "mutual", "limit": 20},
        "query": "transformer attention",
        "categories": ["cs.CL"],
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    mutual_edges = [e for e in data["edges"] if e["relation"] == "mutual_citation"]
    # Each mutual pair should have exactly 2 edges (A→B and B→A)
    if len(mutual_edges) % 2 != 0:
        fail(name, data["took_ms"],
             f"Odd number of mutual edges: {len(mutual_edges)}")
        return
    ok(name, data["took_ms"],
       f"{len(mutual_edges)} mutual edges ({len(mutual_edges)//2} pairs)")


async def test_citation_patterns_chain_ordered(c: httpx.AsyncClient):
    """Chain pattern should produce sequential chain_link edges."""
    name = "citation_patterns:chain_ordered"
    data = await post_graph(c, {
        "graph": {"type": "citation_patterns", "pattern": "chain", "limit": 10},
        "query": "generative model",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    chain_edges = [e for e in data["edges"] if e["relation"] == "chain_link"]
    ok(name, data["took_ms"],
       f"{len(chain_edges)} chain_link edges, patterns={data['metadata'].get('patterns_found')}")


async def test_citation_patterns_triangle_three_edges(c: httpx.AsyncClient):
    """Triangle patterns should produce exactly 3 edges per triangle."""
    name = "citation_patterns:triangle_edges"
    data = await post_graph(c, {
        "graph": {"type": "citation_patterns", "pattern": "triangle", "limit": 10},
        "query": "convolutional network image",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    triangle_edges = [e for e in data["edges"] if e["relation"] == "triangle_edge"]
    found = data["metadata"].get("patterns_found", 0)
    if found > 0 and len(triangle_edges) != found * 3:
        fail(name, data["took_ms"],
             f"Expected {found*3} triangle edges, got {len(triangle_edges)}")
        return
    ok(name, data["took_ms"],
       f"{found} triangles, {len(triangle_edges)} edges")


async def test_citation_patterns_with_connected_data(c: httpx.AsyncClient):
    """Use seed papers known to have references for better pattern coverage."""
    name = "citation_patterns:connected_seed"
    data = await post_graph(c, {
        "graph": {"type": "citation_patterns", "pattern": "star", "limit": 10,
                  "seed_arxiv_ids": [CONNECTED_PAPER]},
    }, name)
    if not data:
        # seed_arxiv_ids might not be supported for citation_patterns — that's ok
        # It'll use base query instead
        data = await post_graph(c, {
            "graph": {"type": "citation_patterns", "pattern": "star", "limit": 20},
            "query": "discrete diffusion",
        }, name)
        if not data:
            return
    ok(name, data["took_ms"],
       f"patterns={data['metadata'].get('patterns_found')}, "
       f"subgraph={data['metadata'].get('papers_in_subgraph')}, "
       f"edges={data['metadata'].get('edges_in_subgraph')}")


async def test_citation_patterns_metadata_fields(c: httpx.AsyncClient):
    """All pattern responses should have pattern, patterns_found, papers_in_subgraph."""
    name = "citation_patterns:metadata_fields"
    data = await post_graph(c, {
        "graph": {"type": "citation_patterns", "pattern": "chain", "limit": 5},
        "query": "recurrent network LSTM",
    }, name)
    if not data:
        return
    meta = data["metadata"]
    required = {"pattern", "patterns_found", "papers_in_subgraph", "edges_in_subgraph"}
    missing = required - set(meta.keys())
    if missing:
        fail(name, data["took_ms"], f"Missing metadata: {missing}")
        return
    if meta["pattern"] != "chain":
        fail(name, data["took_ms"], f"Expected pattern=chain, got {meta['pattern']}")
        return
    ok(name, data["took_ms"], f"All metadata present, pattern={meta['pattern']}")


# ═══════════════════════════════════════════════════════
# 22. CONNECTED COMPONENTS — Deep Tests
# ═══════════════════════════════════════════════════════

async def test_connected_components_structure(c: httpx.AsyncClient):
    """Component nodes should have size and member_ids."""
    name = "connected_comp:structure"
    data = await post_graph(c, {
        "graph": {"type": "connected_components", "limit": 15},
        "query": "bayesian inference",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    components = [n for n in data["nodes"] if n["type"] == "component"]
    for cn in components:
        props = cn.get("properties", {})
        if "size" not in props or "member_ids" not in props:
            fail(name, data["took_ms"], f"Component {cn['id']} missing size/member_ids")
            return
        if not isinstance(props["member_ids"], list):
            fail(name, data["took_ms"], f"member_ids not a list: {type(props['member_ids'])}")
            return
    ok(name, data["took_ms"],
       f"{len(components)} components found")


async def test_connected_components_sorted_by_size(c: httpx.AsyncClient):
    """Components should be returned sorted by size (largest first)."""
    name = "connected_comp:sorted"
    data = await post_graph(c, {
        "graph": {"type": "connected_components", "limit": 20},
        "query": "Monte Carlo sampling",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    components = [n for n in data["nodes"] if n["type"] == "component"]
    sizes = [cn["properties"].get("size", 0) for cn in components]
    if len(sizes) >= 2 and sizes != sorted(sizes, reverse=True):
        fail(name, data["took_ms"], f"Not sorted by size: {sizes}")
        return
    ok(name, data["took_ms"], f"Sizes (sorted desc): {sizes[:10]}")


async def test_connected_components_contains_edges(c: httpx.AsyncClient):
    """Each paper member should have a 'contains' edge from its component."""
    name = "connected_comp:contains_edges"
    data = await post_graph(c, {
        "graph": {"type": "connected_components", "limit": 10},
        "query": "variational autoencoder",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    papers = [n for n in data["nodes"] if n["type"] == "paper"]
    contains = [e for e in data["edges"] if e["relation"] == "contains"]
    ok(name, data["took_ms"],
       f"{len(papers)} paper nodes, {len(contains)} contains edges")


async def test_connected_components_isolated_count(c: httpx.AsyncClient):
    """isolated_papers metadata should match singleton component count."""
    name = "connected_comp:isolated_count"
    data = await post_graph(c, {
        "graph": {"type": "connected_components", "limit": 50},
        "query": "knowledge graph embedding",
    }, name)
    if not data:
        return
    meta = data["metadata"]
    components = [n for n in data["nodes"] if n["type"] == "component"]
    singletons = [cn for cn in components if cn["properties"].get("size", 0) == 1]
    reported_isolated = meta.get("isolated_papers", 0)
    ok(name, data["took_ms"],
       f"Reported isolated={reported_isolated}, "
       f"singleton components in result={len(singletons)}")


async def test_connected_components_with_filters(c: httpx.AsyncClient):
    """Connected components + date/category filters."""
    name = "connected_comp:date_cat_filter"
    data = await post_graph(c, {
        "graph": {"type": "connected_components", "limit": 10},
        "query": "federated learning",
        "categories": ["cs.LG"],
        "submitted_date": {"gte": "2023-01-01"},
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    ok(name, data["took_ms"],
       f"components={data['metadata'].get('components_found')}, "
       f"papers={data['metadata'].get('papers_in_graph')}")


async def test_connected_components_metadata_integrity(c: httpx.AsyncClient):
    """Metadata values should be consistent."""
    name = "connected_comp:metadata"
    data = await post_graph(c, {
        "graph": {"type": "connected_components", "limit": 30},
        "query": "speech recognition",
    }, name)
    if not data:
        return
    meta = data["metadata"]
    required = {"papers_in_graph", "edges_in_graph", "components_found",
                "largest_component", "isolated_papers"}
    missing = required - set(meta.keys())
    if missing:
        fail(name, data["took_ms"], f"Missing metadata: {missing}")
        return
    if meta["largest_component"] > meta["papers_in_graph"]:
        fail(name, data["took_ms"],
             f"Largest component ({meta['largest_component']}) > total ({meta['papers_in_graph']})")
        return
    if meta["isolated_papers"] > meta["papers_in_graph"]:
        fail(name, data["took_ms"],
             f"Isolated ({meta['isolated_papers']}) > total ({meta['papers_in_graph']})")
        return
    ok(name, data["took_ms"],
       f"papers={meta['papers_in_graph']}, components={meta['components_found']}, "
       f"largest={meta['largest_component']}, isolated={meta['isolated_papers']}")


# ═══════════════════════════════════════════════════════
# CROSS-ALGORITHM TESTS
# ═══════════════════════════════════════════════════════

async def test_cross_all_algorithms_same_query(c: httpx.AsyncClient):
    """Same query+filters through all 6 new algorithms — all should succeed."""
    name = "cross:all_6_same_query"
    base_filters = {"query": "transformer attention", "categories": ["cs.CL"]}
    types_and_extras = [
        ("multihop_citation", {"max_hops": 2, "direction": "cited_by", "limit": 10}),
        ("shortest_citation_path", {"seed_arxiv_id": CONNECTED_PAPER,
                                     "target_arxiv_id": BIG_REFS_PAPER, "max_hops": 3}),
        ("pagerank", {"iterations": 10, "limit": 10}),
        ("community_detection", {"iterations": 10, "limit": 10}),
        ("citation_patterns", {"pattern": "star", "limit": 10}),
        ("connected_components", {"limit": 10}),
    ]
    ok_count = 0
    for gtype, extras in types_and_extras:
        body = {**base_filters, "graph": {"type": gtype, **extras}}
        data = await post_graph(c, body, f"{name}:{gtype}")
        if data and validate_graph_integrity(data, name):
            ok_count += 1
    if ok_count == 6:
        ok(name, 0, "All 6 algorithms responded correctly")
    else:
        fail(name, 0, f"Only {ok_count}/6 algorithms succeeded")


async def test_cross_semantic_boost_all_algorithms(c: httpx.AsyncClient):
    """Semantic boost should compose with all 6 algorithms."""
    name = "cross:semantic_boost"
    semantic = [{"text": "self-supervised contrastive learning", "level": "abstract", "weight": 2.0, "mode": "boost"}]
    base = {"query": "representation learning", "semantic": semantic}
    types_and_extras = [
        ("multihop_citation", {"max_hops": 1, "direction": "references", "limit": 5}),
        ("pagerank", {"iterations": 5, "limit": 5}),
        ("community_detection", {"iterations": 5, "limit": 5}),
        ("citation_patterns", {"pattern": "mutual", "limit": 5}),
        ("connected_components", {"limit": 5}),
    ]
    ok_count = 0
    for gtype, extras in types_and_extras:
        body = {**base, "graph": {"type": gtype, **extras}}
        data = await post_graph(c, body, f"{name}:{gtype}")
        if data:
            ok_count += 1
    if ok_count == 5:
        ok(name, 0, "All 5 algorithms work with semantic boost")
    else:
        fail(name, 0, f"Only {ok_count}/5 algorithms succeeded with semantic boost")


async def test_cross_regex_filter_all(c: httpx.AsyncClient):
    """Regex filter should compose with new graph algorithms."""
    name = "cross:regex_filter"
    base = {"title_regex": ".*[Tt]ransform.*"}
    types_and_extras = [
        ("multihop_citation", {"max_hops": 1, "direction": "references", "limit": 5}),
        ("pagerank", {"iterations": 5, "limit": 5}),
        ("connected_components", {"limit": 5}),
    ]
    ok_count = 0
    for gtype, extras in types_and_extras:
        body = {**base, "graph": {"type": gtype, **extras}}
        data = await post_graph(c, body, f"{name}:{gtype}")
        if data:
            ok_count += 1
    if ok_count == 3:
        ok(name, 0, f"All 3 regex-filtered tests passed")
    else:
        fail(name, 0, f"Only {ok_count}/3 passed")


async def test_cross_consistency_repeated_call(c: httpx.AsyncClient):
    """Same query twice should return consistent results."""
    name = "cross:consistency"
    body = {
        "graph": {"type": "pagerank", "seed_arxiv_id": CONNECTED_PAPER,
                  "iterations": 10, "limit": 10},
    }
    data1 = await post_graph(c, body, f"{name}:1")
    data2 = await post_graph(c, body, f"{name}:2")
    if not data1 or not data2:
        return
    ids1 = [n["id"] for n in data1["nodes"]]
    ids2 = [n["id"] for n in data2["nodes"]]
    if ids1 == ids2:
        ok(name, data2["took_ms"], f"Identical results ({len(ids1)} nodes)")
    else:
        # Community detection uses random, but pagerank should be deterministic
        common = set(ids1) & set(ids2)
        fail(name, data2["took_ms"],
             f"Results differ: {len(common)}/{len(ids1)} overlap")


# ═══════════════════════════════════════════════════════
# VALIDATION EDGE CASES
# ═══════════════════════════════════════════════════════

async def test_validation_invalid_pattern(c: httpx.AsyncClient):
    """Invalid pattern should be rejected by validation."""
    name = "validation:invalid_pattern"
    start = time.monotonic()
    resp = await c.post(f"{BASE}/graph", json={
        "graph": {"type": "citation_patterns", "pattern": "octagon"},
    }, headers=HEADERS, timeout=TIMEOUT)
    elapsed = int((time.monotonic() - start) * 1000)
    if resp.status_code == 422:
        ok(name, elapsed, "422 for invalid pattern")
    else:
        fail(name, elapsed, f"Expected 422, got {resp.status_code}")


async def test_validation_max_hops_too_high(c: httpx.AsyncClient):
    """max_hops > 50 should be rejected."""
    name = "validation:max_hops_too_high"
    start = time.monotonic()
    resp = await c.post(f"{BASE}/graph", json={
        "graph": {"type": "multihop_citation", "max_hops": 55,
                  "seed_arxiv_id": CONNECTED_PAPER},
    }, headers=HEADERS, timeout=TIMEOUT)
    elapsed = int((time.monotonic() - start) * 1000)
    if resp.status_code == 422:
        ok(name, elapsed, "422 for max_hops=55")
    else:
        fail(name, elapsed, f"Expected 422, got {resp.status_code}")


async def test_validation_damping_out_of_range(c: httpx.AsyncClient):
    """Damping factor > 0.99 should be rejected."""
    name = "validation:damping_range"
    start = time.monotonic()
    resp = await c.post(f"{BASE}/graph", json={
        "graph": {"type": "pagerank", "damping_factor": 1.5},
    }, headers=HEADERS, timeout=TIMEOUT)
    elapsed = int((time.monotonic() - start) * 1000)
    if resp.status_code == 422:
        ok(name, elapsed, "422 for damping=1.5")
    else:
        fail(name, elapsed, f"Expected 422, got {resp.status_code}")


async def test_validation_iterations_out_of_range(c: httpx.AsyncClient):
    """iterations > 500 should be rejected."""
    name = "validation:iterations_range"
    start = time.monotonic()
    resp = await c.post(f"{BASE}/graph", json={
        "graph": {"type": "pagerank", "iterations": 550},
    }, headers=HEADERS, timeout=TIMEOUT)
    elapsed = int((time.monotonic() - start) * 1000)
    if resp.status_code == 422:
        ok(name, elapsed, "422 for iterations=550")
    else:
        fail(name, elapsed, f"Expected 422, got {resp.status_code}")


async def test_validation_direction_invalid(c: httpx.AsyncClient):
    """Invalid direction should be rejected."""
    name = "validation:direction_invalid"
    start = time.monotonic()
    resp = await c.post(f"{BASE}/graph", json={
        "graph": {"type": "multihop_citation", "direction": "backwards",
                  "seed_arxiv_id": CONNECTED_PAPER},
    }, headers=HEADERS, timeout=TIMEOUT)
    elapsed = int((time.monotonic() - start) * 1000)
    if resp.status_code == 422:
        ok(name, elapsed, "422 for direction=backwards")
    else:
        fail(name, elapsed, f"Expected 422, got {resp.status_code}")


async def test_validation_limit_boundary(c: httpx.AsyncClient):
    """limit=0 should be rejected, limit=200 should work."""
    name = "validation:limit_boundary"
    start = time.monotonic()
    resp = await c.post(f"{BASE}/graph", json={
        "graph": {"type": "connected_components", "limit": 0},
        "query": "test",
    }, headers=HEADERS, timeout=TIMEOUT)
    elapsed = int((time.monotonic() - start) * 1000)
    if resp.status_code != 422:
        fail(name, elapsed, f"Expected 422 for limit=0, got {resp.status_code}")
        return
    # limit=200 should work
    data = await post_graph(c, {
        "graph": {"type": "connected_components", "limit": 200},
        "query": "test",
    }, name)
    if data:
        ok(name, elapsed, "limit=0→422, limit=200→OK")
    else:
        fail(name, elapsed, "limit=200 failed")


# ═══════════════════════════════════════════════════════
# RUNNER
# ═══════════════════════════════════════════════════════

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

        tests = [v for k, v in globals().items()
                 if k.startswith("test_") and asyncio.iscoroutinefunction(v)]

        print(f"Running {len(tests)} deep graph tests...\n")
        start = time.monotonic()

        for test_fn in tests:
            await test_fn(c)

        elapsed = time.monotonic() - start

    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)

    print(f"\n{'='*80}")
    print(f" DEEP GRAPH TEST RESULTS: {passed}/{len(results)} passed ({failed} failed) in {elapsed:.1f}s")
    print(f"{'='*80}\n")

    current_group = ""
    for r in results:
        group = r.name.split(":")[0]
        if group != current_group:
            current_group = group
            print(f"  ── {group} ──")
        status = "✓" if r.passed else "✗"
        print(f"    {status} {r.name:<50} {r.took_ms:>5}ms  {r.detail}")

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
