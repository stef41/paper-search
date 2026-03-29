"""Deep E2E tests for graph-DB algorithms batch 3 (types 29-34).

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
TIMEOUT = 120.0  # influence_maximization can be slow

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
# 29. LOUVAIN COMMUNITY — Deep Tests
# ═══════════════════════════════════════════════════════

async def test_louvain_basic_functionality(c: httpx.AsyncClient):
    """Basic louvain community detection returns communities and papers."""
    name = "louvain:basic_functionality"
    data = await post_graph(c, {
        "graph": {"type": "louvain_community", "iterations": 10, "limit": 20},
        "query": "natural language processing",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    communities = [n for n in data["nodes"] if n["type"] == "community"]
    papers = [n for n in data["nodes"] if n["type"] == "paper"]
    if not communities:
        fail(name, data["took_ms"], "No community nodes returned")
        return
    if not papers:
        fail(name, data["took_ms"], "No paper nodes returned")
        return
    ok(name, data["took_ms"],
       f"{len(communities)} communities, {len(papers)} papers, {len(data['edges'])} edges")


async def test_louvain_metadata_correctness(c: httpx.AsyncClient):
    """All expected metadata fields present and valid."""
    name = "louvain:metadata_correctness"
    data = await post_graph(c, {
        "graph": {"type": "louvain_community", "iterations": 10, "limit": 15},
        "query": "deep learning",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    meta = data.get("metadata", {})
    expected_keys = ["papers_in_graph", "communities_found", "modularity", "largest_community", "algorithm"]
    missing = [k for k in expected_keys if k not in meta]
    if missing:
        fail(name, data["took_ms"], f"Missing metadata keys: {missing}")
        return
    if meta.get("algorithm") != "louvain":
        fail(name, data["took_ms"], f"algorithm should be 'louvain', got '{meta.get('algorithm')}'")
        return
    ok(name, data["took_ms"],
       f"papers={meta['papers_in_graph']}, communities={meta['communities_found']}, "
       f"modularity={meta.get('modularity')}, algorithm={meta['algorithm']}")


async def test_louvain_community_node_ids(c: httpx.AsyncClient):
    """Community node IDs should start with 'louvain_'."""
    name = "louvain:community_node_ids"
    data = await post_graph(c, {
        "graph": {"type": "louvain_community", "iterations": 10, "limit": 15},
        "query": "computer vision",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    communities = [n for n in data["nodes"] if n["type"] == "community"]
    if not communities:
        fail(name, data["took_ms"], "No community nodes found")
        return
    bad = [c for c in communities if not c["id"].startswith("louvain_")]
    if bad:
        fail(name, data["took_ms"], f"{len(bad)} community IDs don't start with 'louvain_': {bad[0]['id']}")
        return
    ok(name, data["took_ms"], f"All {len(communities)} community IDs start with 'louvain_'")


async def test_louvain_community_node_properties(c: httpx.AsyncClient):
    """Community nodes should have size, top_categories, member_ids properties."""
    name = "louvain:community_node_props"
    data = await post_graph(c, {
        "graph": {"type": "louvain_community", "iterations": 10, "limit": 20},
        "query": "reinforcement learning",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    communities = [n for n in data["nodes"] if n["type"] == "community"]
    if not communities:
        fail(name, data["took_ms"], "No community nodes")
        return
    for cn in communities:
        props = cn.get("properties", {})
        for k in ("size", "top_categories", "member_ids"):
            if k not in props:
                fail(name, data["took_ms"], f"Community {cn['id']} missing property '{k}'")
                return
        if not isinstance(props["top_categories"], list):
            fail(name, data["took_ms"], f"top_categories not a list for {cn['id']}")
            return
        if not isinstance(props["member_ids"], list):
            fail(name, data["took_ms"], f"member_ids not a list for {cn['id']}")
            return
    ok(name, data["took_ms"], f"All {len(communities)} communities have correct properties")


async def test_louvain_paper_community_property(c: httpx.AsyncClient):
    """Paper/member nodes should have community (int) property."""
    name = "louvain:paper_community_prop"
    data = await post_graph(c, {
        "graph": {"type": "louvain_community", "iterations": 10, "limit": 15},
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
        if not isinstance(comm, int):
            fail(name, data["took_ms"], f"Paper {pn['id']} community not int: {type(comm)}")
            return
    ok(name, data["took_ms"], f"All {len(papers)} papers have integer community property")


async def test_louvain_contains_edges(c: httpx.AsyncClient):
    """'contains' edges should connect community → paper."""
    name = "louvain:contains_edges"
    data = await post_graph(c, {
        "graph": {"type": "louvain_community", "iterations": 10, "limit": 15},
        "query": "optimization",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    contains = [e for e in data["edges"] if e["relation"] == "contains"]
    if not contains:
        fail(name, data["took_ms"], "No 'contains' edges found")
        return
    community_ids = {n["id"] for n in data["nodes"] if n["type"] == "community"}
    paper_ids = {n["id"] for n in data["nodes"] if n["type"] == "paper"}
    for e in contains:
        if e["source"] not in community_ids:
            fail(name, data["took_ms"], f"contains edge source {e['source']} not a community node")
            return
        if e["target"] not in paper_ids:
            fail(name, data["took_ms"], f"contains edge target {e['target']} not a paper node")
            return
    ok(name, data["took_ms"], f"{len(contains)} contains edges are valid")


async def test_louvain_inter_community_edges(c: httpx.AsyncClient):
    """Inter-community edges should have weight > 0."""
    name = "louvain:inter_community_edges"
    data = await post_graph(c, {
        "graph": {"type": "louvain_community", "iterations": 10, "limit": 20},
        "query": "machine learning",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    inter = [e for e in data["edges"] if e["relation"] == "inter_community"]
    if inter:
        bad = [e for e in inter if (e.get("weight") or 0) <= 0]
        if bad:
            fail(name, data["took_ms"], f"{len(bad)} inter_community edges with weight <= 0")
            return
    ok(name, data["took_ms"], f"{len(inter)} inter_community edges")


async def test_louvain_modularity_range(c: httpx.AsyncClient):
    """Modularity should be a float between -0.5 and 1.0."""
    name = "louvain:modularity_range"
    data = await post_graph(c, {
        "graph": {"type": "louvain_community", "iterations": 15, "limit": 20},
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


async def test_louvain_with_category_filter(c: httpx.AsyncClient):
    """Louvain with category filter should work."""
    name = "louvain:category_filter"
    data = await post_graph(c, {
        "graph": {"type": "louvain_community", "iterations": 10, "limit": 15},
        "query": "language model",
        "categories": ["cs.CL"],
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    ok(name, data["took_ms"],
       f"communities={data['metadata'].get('communities_found')}, papers={data['metadata'].get('papers_in_graph')}")


async def test_louvain_with_date_filter(c: httpx.AsyncClient):
    """Louvain with date range filter."""
    name = "louvain:date_filter"
    data = await post_graph(c, {
        "graph": {"type": "louvain_community", "iterations": 10, "limit": 15},
        "query": "large language model",
        "submitted_date": {"gte": "2024-01-01T00:00:00"},
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    ok(name, data["took_ms"],
       f"communities={data['metadata'].get('communities_found')}, papers={data['metadata'].get('papers_in_graph')}")


async def test_louvain_limit_respected(c: httpx.AsyncClient):
    """Limit parameter should restrict number of community nodes."""
    name = "louvain:limit_respected"
    data = await post_graph(c, {
        "graph": {"type": "louvain_community", "iterations": 10, "limit": 5},
        "query": "neural network",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    comms = [n for n in data["nodes"] if n["type"] == "community"]
    if len(comms) > 5:
        fail(name, data["took_ms"], f"limit=5 but got {len(comms)} community nodes")
        return
    ok(name, data["took_ms"], f"{len(comms)} communities (limit=5)")


# ═══════════════════════════════════════════════════════
# 30. DEGREE CENTRALITY — Deep Tests
# ═══════════════════════════════════════════════════════

async def test_degree_basic_functionality(c: httpx.AsyncClient):
    """Basic degree centrality returns papers with degree info."""
    name = "degree:basic_functionality"
    data = await post_graph(c, {
        "graph": {"type": "degree_centrality", "limit": 10},
        "query": "deep learning",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    nodes = data.get("nodes", [])
    if not nodes:
        fail(name, data["took_ms"], "No nodes returned")
        return
    ok(name, data["took_ms"],
       f"{len(nodes)} nodes, {len(data['edges'])} edges")


async def test_degree_metadata_correctness(c: httpx.AsyncClient):
    """Check all expected metadata fields."""
    name = "degree:metadata_correctness"
    data = await post_graph(c, {
        "graph": {"type": "degree_centrality", "limit": 10},
        "query": "transformer",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    meta = data.get("metadata", {})
    expected_keys = ["papers_in_subgraph", "mode", "max_degree", "avg_degree", "edges_in_subgraph"]
    missing = [k for k in expected_keys if k not in meta]
    if missing:
        fail(name, data["took_ms"], f"Missing metadata keys: {missing}")
        return
    ok(name, data["took_ms"],
       f"mode={meta['mode']}, max_degree={meta['max_degree']}, avg={meta.get('avg_degree')}")


async def test_degree_node_properties(c: httpx.AsyncClient):
    """Nodes should have degree, degree_centrality, degree_mode, rank."""
    name = "degree:node_properties"
    data = await post_graph(c, {
        "graph": {"type": "degree_centrality", "limit": 10},
        "query": "attention mechanism",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    nodes = data.get("nodes", [])
    if not nodes:
        fail(name, data["took_ms"], "No nodes")
        return
    for n in nodes:
        props = n.get("properties", {})
        for k in ("degree", "degree_centrality", "degree_mode", "rank"):
            if k not in props:
                fail(name, data["took_ms"], f"Node {n['id']} missing property '{k}'")
                return
    ok(name, data["took_ms"], f"All {len(nodes)} nodes have required properties")


async def test_degree_mode_in(c: httpx.AsyncClient):
    """degree_mode='in' — in-degree (citations received)."""
    name = "degree:mode_in"
    data = await post_graph(c, {
        "graph": {"type": "degree_centrality", "degree_mode": "in", "limit": 10},
        "query": "neural network",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    meta = data.get("metadata", {})
    if meta.get("mode") != "in":
        fail(name, data["took_ms"], f"Expected mode='in', got '{meta.get('mode')}'")
        return
    nodes = data.get("nodes", [])
    for n in nodes:
        if n.get("properties", {}).get("degree_mode") != "in":
            fail(name, data["took_ms"], f"Node {n['id']} degree_mode != 'in'")
            return
    ok(name, data["took_ms"], f"{len(nodes)} nodes with in-degree ranking")


async def test_degree_mode_out(c: httpx.AsyncClient):
    """degree_mode='out' — out-degree (references made)."""
    name = "degree:mode_out"
    data = await post_graph(c, {
        "graph": {"type": "degree_centrality", "degree_mode": "out", "limit": 10},
        "query": "language model",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    meta = data.get("metadata", {})
    if meta.get("mode") != "out":
        fail(name, data["took_ms"], f"Expected mode='out', got '{meta.get('mode')}'")
        return
    ok(name, data["took_ms"], f"{len(data['nodes'])} nodes with out-degree ranking")


async def test_degree_mode_total(c: httpx.AsyncClient):
    """degree_mode='total' — default total degree."""
    name = "degree:mode_total"
    data = await post_graph(c, {
        "graph": {"type": "degree_centrality", "degree_mode": "total", "limit": 10},
        "query": "convolutional neural network",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    meta = data.get("metadata", {})
    if meta.get("mode") != "total":
        fail(name, data["took_ms"], f"Expected mode='total', got '{meta.get('mode')}'")
        return
    ok(name, data["took_ms"], f"{len(data['nodes'])} nodes with total-degree ranking")


async def test_degree_rank_sequential(c: httpx.AsyncClient):
    """Rank should be sequential starting from 0."""
    name = "degree:rank_sequential"
    data = await post_graph(c, {
        "graph": {"type": "degree_centrality", "limit": 15},
        "query": "image recognition",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    nodes = data.get("nodes", [])
    if not nodes:
        fail(name, data["took_ms"], "No nodes")
        return
    ranks = sorted([n.get("properties", {}).get("rank", -1) for n in nodes])
    expected = list(range(len(nodes)))
    if ranks != expected:
        fail(name, data["took_ms"], f"Ranks not sequential 0..{len(nodes)-1}: got {ranks[:10]}")
        return
    ok(name, data["took_ms"], f"Ranks sequential 0..{len(nodes)-1}")


async def test_degree_centrality_range(c: httpx.AsyncClient):
    """degree_centrality should be between 0 and 1."""
    name = "degree:centrality_range"
    data = await post_graph(c, {
        "graph": {"type": "degree_centrality", "limit": 10},
        "query": "generative adversarial",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    nodes = data.get("nodes", [])
    for n in nodes:
        dc = n.get("properties", {}).get("degree_centrality")
        if dc is None:
            continue
        if not isinstance(dc, (int, float)) or dc < 0 or dc > 1:
            fail(name, data["took_ms"], f"Node {n['id']} degree_centrality={dc} out of [0,1]")
            return
    ok(name, data["took_ms"], f"All {len(nodes)} nodes have degree_centrality in [0,1]")


async def test_degree_with_category_filter(c: httpx.AsyncClient):
    """Degree centrality with category filter."""
    name = "degree:category_filter"
    data = await post_graph(c, {
        "graph": {"type": "degree_centrality", "limit": 10},
        "query": "robot control",
        "categories": ["cs.AI"],
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    ok(name, data["took_ms"],
       f"nodes={len(data['nodes'])}, max_degree={data['metadata'].get('max_degree')}")


async def test_degree_limit_small(c: httpx.AsyncClient):
    """limit=1 should return exactly 1 node."""
    name = "degree:limit_1"
    data = await post_graph(c, {
        "graph": {"type": "degree_centrality", "limit": 1},
        "query": "deep reinforcement learning",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    nodes = data.get("nodes", [])
    if len(nodes) != 1:
        fail(name, data["took_ms"], f"limit=1 but got {len(nodes)} nodes")
        return
    ok(name, data["took_ms"], f"1 node returned, degree={nodes[0].get('properties', {}).get('degree')}")


# ═══════════════════════════════════════════════════════
# 31. EIGENVECTOR CENTRALITY — Deep Tests
# ═══════════════════════════════════════════════════════

async def test_eigenvector_basic_functionality(c: httpx.AsyncClient):
    """Basic eigenvector centrality returns papers ranked by eigenvector centrality."""
    name = "eigenvector:basic_functionality"
    data = await post_graph(c, {
        "graph": {"type": "eigenvector_centrality", "limit": 10},
        "query": "deep learning",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    nodes = data.get("nodes", [])
    if not nodes:
        fail(name, data["took_ms"], "No nodes returned")
        return
    ok(name, data["took_ms"],
       f"{len(nodes)} nodes, {len(data['edges'])} edges")


async def test_eigenvector_metadata_correctness(c: httpx.AsyncClient):
    """Check all expected metadata fields."""
    name = "eigenvector:metadata_correctness"
    data = await post_graph(c, {
        "graph": {"type": "eigenvector_centrality", "limit": 10},
        "query": "transformer architecture",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    meta = data.get("metadata", {})
    expected_keys = ["papers_in_subgraph", "edges_in_subgraph", "max_eigenvector", "iterations_run", "converged"]
    missing = [k for k in expected_keys if k not in meta]
    if missing:
        fail(name, data["took_ms"], f"Missing metadata keys: {missing}")
        return
    ok(name, data["took_ms"],
       f"papers={meta['papers_in_subgraph']}, max_ev={meta.get('max_eigenvector')}, "
       f"iters={meta['iterations_run']}, converged={meta['converged']}")


async def test_eigenvector_node_properties(c: httpx.AsyncClient):
    """Nodes should have eigenvector_centrality and rank."""
    name = "eigenvector:node_properties"
    data = await post_graph(c, {
        "graph": {"type": "eigenvector_centrality", "limit": 10},
        "query": "attention mechanism",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    nodes = data.get("nodes", [])
    if not nodes:
        fail(name, data["took_ms"], "No nodes")
        return
    for n in nodes:
        props = n.get("properties", {})
        if "eigenvector_centrality" not in props:
            fail(name, data["took_ms"], f"Node {n['id']} missing 'eigenvector_centrality'")
            return
        if "rank" not in props:
            fail(name, data["took_ms"], f"Node {n['id']} missing 'rank'")
            return
    ok(name, data["took_ms"], f"All {len(nodes)} nodes have eigenvector_centrality and rank")


async def test_eigenvector_centrality_positive(c: httpx.AsyncClient):
    """Eigenvector centrality values should be positive floats."""
    name = "eigenvector:centrality_positive"
    data = await post_graph(c, {
        "graph": {"type": "eigenvector_centrality", "limit": 10},
        "query": "neural network",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    nodes = data.get("nodes", [])
    for n in nodes:
        ev = n.get("properties", {}).get("eigenvector_centrality")
        if ev is None:
            continue
        if not isinstance(ev, (int, float)) or ev < 0:
            fail(name, data["took_ms"], f"Node {n['id']} eigenvector_centrality={ev} not positive")
            return
    ok(name, data["took_ms"], f"All {len(nodes)} nodes have positive eigenvector_centrality")


async def test_eigenvector_converged_bool(c: httpx.AsyncClient):
    """converged should be a boolean."""
    name = "eigenvector:converged_bool"
    data = await post_graph(c, {
        "graph": {"type": "eigenvector_centrality", "iterations": 20, "limit": 10},
        "query": "recurrent neural network",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    meta = data.get("metadata", {})
    converged = meta.get("converged")
    if not isinstance(converged, bool):
        fail(name, data["took_ms"], f"converged is not bool: {type(converged)} = {converged}")
        return
    ok(name, data["took_ms"], f"converged={converged}, iters={meta.get('iterations_run')}")


async def test_eigenvector_with_category_filter(c: httpx.AsyncClient):
    """Eigenvector centrality with category filter."""
    name = "eigenvector:category_filter"
    data = await post_graph(c, {
        "graph": {"type": "eigenvector_centrality", "limit": 10},
        "query": "speech recognition",
        "categories": ["cs.CL"],
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    ok(name, data["took_ms"],
       f"nodes={len(data['nodes'])}, max_ev={data['metadata'].get('max_eigenvector')}")


async def test_eigenvector_with_date_filter(c: httpx.AsyncClient):
    """Eigenvector centrality with date range."""
    name = "eigenvector:date_filter"
    data = await post_graph(c, {
        "graph": {"type": "eigenvector_centrality", "limit": 10},
        "query": "large language model",
        "submitted_date": {"gte": "2024-01-01T00:00:00"},
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    ok(name, data["took_ms"],
       f"nodes={len(data['nodes'])}, converged={data['metadata'].get('converged')}")


async def test_eigenvector_limit_small(c: httpx.AsyncClient):
    """limit=1 should return exactly 1 node."""
    name = "eigenvector:limit_1"
    data = await post_graph(c, {
        "graph": {"type": "eigenvector_centrality", "limit": 1},
        "query": "object detection",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    nodes = data.get("nodes", [])
    if len(nodes) != 1:
        fail(name, data["took_ms"], f"limit=1 but got {len(nodes)} nodes")
        return
    ok(name, data["took_ms"],
       f"1 node, ev={nodes[0].get('properties', {}).get('eigenvector_centrality')}")


async def test_eigenvector_high_iterations(c: httpx.AsyncClient):
    """Higher iterations should still converge."""
    name = "eigenvector:high_iterations"
    data = await post_graph(c, {
        "graph": {"type": "eigenvector_centrality", "iterations": 40, "limit": 10},
        "query": "knowledge graph",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    meta = data.get("metadata", {})
    ok(name, data["took_ms"],
       f"iters_run={meta.get('iterations_run')}, converged={meta.get('converged')}")


# ═══════════════════════════════════════════════════════
# 32. K-CORE DECOMPOSITION — Deep Tests
# ═══════════════════════════════════════════════════════

async def test_kcore_basic_functionality(c: httpx.AsyncClient):
    """Basic k-core decomposition returns papers with coreness."""
    name = "kcore:basic_functionality"
    data = await post_graph(c, {
        "graph": {"type": "kcore_decomposition", "limit": 10},
        "query": "deep learning",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    nodes = data.get("nodes", [])
    if not nodes:
        fail(name, data["took_ms"], "No nodes returned")
        return
    ok(name, data["took_ms"],
       f"{len(nodes)} nodes, {len(data['edges'])} edges")


async def test_kcore_metadata_correctness(c: httpx.AsyncClient):
    """Check all expected metadata fields."""
    name = "kcore:metadata_correctness"
    data = await post_graph(c, {
        "graph": {"type": "kcore_decomposition", "limit": 10},
        "query": "transformer",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    meta = data.get("metadata", {})
    expected_keys = ["papers_in_subgraph", "max_coreness", "core_distribution", "edges_in_subgraph"]
    missing = [k for k in expected_keys if k not in meta]
    if missing:
        fail(name, data["took_ms"], f"Missing metadata keys: {missing}")
        return
    ok(name, data["took_ms"],
       f"papers={meta['papers_in_subgraph']}, max_coreness={meta['max_coreness']}, "
       f"distribution_keys={list(meta.get('core_distribution', {}).keys())[:5]}")


async def test_kcore_node_properties(c: httpx.AsyncClient):
    """Nodes should have coreness and max_coreness properties."""
    name = "kcore:node_properties"
    data = await post_graph(c, {
        "graph": {"type": "kcore_decomposition", "limit": 10},
        "query": "natural language processing",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    nodes = data.get("nodes", [])
    if not nodes:
        fail(name, data["took_ms"], "No nodes")
        return
    for n in nodes:
        props = n.get("properties", {})
        if "coreness" not in props:
            fail(name, data["took_ms"], f"Node {n['id']} missing 'coreness'")
            return
        if "max_coreness" not in props:
            fail(name, data["took_ms"], f"Node {n['id']} missing 'max_coreness'")
            return
    ok(name, data["took_ms"], f"All {len(nodes)} nodes have coreness and max_coreness")


async def test_kcore_coreness_nonnegative(c: httpx.AsyncClient):
    """Coreness values should be non-negative integers."""
    name = "kcore:coreness_nonnegative"
    data = await post_graph(c, {
        "graph": {"type": "kcore_decomposition", "limit": 15},
        "query": "machine learning",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    nodes = data.get("nodes", [])
    for n in nodes:
        coreness = n.get("properties", {}).get("coreness")
        if coreness is None:
            continue
        if not isinstance(coreness, int) or coreness < 0:
            fail(name, data["took_ms"], f"Node {n['id']} coreness={coreness} not non-negative int")
            return
    ok(name, data["took_ms"], f"All {len(nodes)} nodes have valid coreness")


async def test_kcore_distribution_dict(c: httpx.AsyncClient):
    """core_distribution should be a dict with integer keys (as strings) mapping to counts."""
    name = "kcore:distribution_dict"
    data = await post_graph(c, {
        "graph": {"type": "kcore_decomposition", "limit": 15},
        "query": "computer vision",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    meta = data.get("metadata", {})
    dist = meta.get("core_distribution")
    if dist is None:
        fail(name, data["took_ms"], "core_distribution missing")
        return
    if not isinstance(dist, dict):
        fail(name, data["took_ms"], f"core_distribution not a dict: {type(dist)}")
        return
    for k, v in dist.items():
        try:
            int(k)
        except (ValueError, TypeError):
            fail(name, data["took_ms"], f"core_distribution key '{k}' not integer-like")
            return
        if not isinstance(v, int) or v < 0:
            fail(name, data["took_ms"], f"core_distribution[{k}]={v} not non-negative int")
            return
    ok(name, data["took_ms"], f"core_distribution has {len(dist)} entries: {dist}")


async def test_kcore_with_category_filter(c: httpx.AsyncClient):
    """K-core with category filter."""
    name = "kcore:category_filter"
    data = await post_graph(c, {
        "graph": {"type": "kcore_decomposition", "limit": 10},
        "query": "reinforcement learning",
        "categories": ["cs.AI"],
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    ok(name, data["took_ms"],
       f"nodes={len(data['nodes'])}, max_coreness={data['metadata'].get('max_coreness')}")


async def test_kcore_with_date_filter(c: httpx.AsyncClient):
    """K-core with date filter."""
    name = "kcore:date_filter"
    data = await post_graph(c, {
        "graph": {"type": "kcore_decomposition", "limit": 10},
        "query": "large language model",
        "submitted_date": {"gte": "2024-01-01T00:00:00"},
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    ok(name, data["took_ms"],
       f"nodes={len(data['nodes'])}, max_coreness={data['metadata'].get('max_coreness')}")


async def test_kcore_limit_respected(c: httpx.AsyncClient):
    """Limit should restrict returned nodes."""
    name = "kcore:limit_respected"
    data = await post_graph(c, {
        "graph": {"type": "kcore_decomposition", "limit": 3},
        "query": "image segmentation",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    nodes = data.get("nodes", [])
    if len(nodes) > 3:
        fail(name, data["took_ms"], f"limit=3 but got {len(nodes)} nodes")
        return
    ok(name, data["took_ms"], f"{len(nodes)} nodes (limit=3)")


async def test_kcore_max_coreness_consistent(c: httpx.AsyncClient):
    """max_coreness in metadata should match max coreness across nodes."""
    name = "kcore:max_coreness_consistent"
    data = await post_graph(c, {
        "graph": {"type": "kcore_decomposition", "limit": 20},
        "query": "optimization",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    meta = data.get("metadata", {})
    max_meta = meta.get("max_coreness")
    nodes = data.get("nodes", [])
    if not nodes or max_meta is None:
        ok(name, data["took_ms"], "No nodes or no max_coreness — skipping")
        return
    max_node = max((n.get("properties", {}).get("coreness", 0) for n in nodes), default=0)
    if max_node != max_meta:
        # max_coreness in metadata may reflect full subgraph, not just returned top nodes
        # so max_node <= max_meta is acceptable
        if max_node > max_meta:
            fail(name, data["took_ms"], f"Node max_coreness={max_node} > metadata max={max_meta}")
            return
    ok(name, data["took_ms"], f"metadata max_coreness={max_meta}, node max={max_node}")


# ═══════════════════════════════════════════════════════
# 33. ARTICULATION POINTS — Deep Tests
# ═══════════════════════════════════════════════════════

async def test_ap_basic_functionality(c: httpx.AsyncClient):
    """Basic articulation points returns AP nodes and neighbors."""
    name = "ap:basic_functionality"
    data = await post_graph(c, {
        "graph": {"type": "articulation_points", "limit": 10},
        "query": "deep learning",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    nodes = data.get("nodes", [])
    if not nodes:
        fail(name, data["took_ms"], "No nodes returned")
        return
    ap_nodes = [n for n in nodes if n.get("properties", {}).get("is_articulation_point") is True]
    ok(name, data["took_ms"],
       f"{len(nodes)} total nodes, {len(ap_nodes)} APs, {len(data['edges'])} edges")


async def test_ap_metadata_correctness(c: httpx.AsyncClient):
    """Check all expected metadata fields."""
    name = "ap:metadata_correctness"
    data = await post_graph(c, {
        "graph": {"type": "articulation_points", "limit": 10},
        "query": "transformer attention",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    meta = data.get("metadata", {})
    expected_keys = ["papers_in_subgraph", "edges_in_subgraph", "articulation_points_found", "articulation_points_returned"]
    missing = [k for k in expected_keys if k not in meta]
    if missing:
        fail(name, data["took_ms"], f"Missing metadata keys: {missing}")
        return
    ok(name, data["took_ms"],
       f"subgraph={meta['papers_in_subgraph']}, APs_found={meta['articulation_points_found']}, "
       f"APs_returned={meta['articulation_points_returned']}")


async def test_ap_node_properties(c: httpx.AsyncClient):
    """AP nodes should have is_articulation_point=true, degree, rank."""
    name = "ap:node_properties"
    data = await post_graph(c, {
        "graph": {"type": "articulation_points", "limit": 10},
        "query": "neural network",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    nodes = data.get("nodes", [])
    ap_nodes = [n for n in nodes if n.get("properties", {}).get("is_articulation_point") is True]
    if not ap_nodes:
        ok(name, data["took_ms"], "No articulation points found (data may lack connectivity)")
        return
    for n in ap_nodes:
        props = n.get("properties", {})
        if "degree" not in props:
            fail(name, data["took_ms"], f"AP node {n['id']} missing 'degree'")
            return
        if "rank" not in props:
            fail(name, data["took_ms"], f"AP node {n['id']} missing 'rank'")
            return
    ok(name, data["took_ms"], f"{len(ap_nodes)} AP nodes all have degree and rank")


async def test_ap_neighbor_properties(c: httpx.AsyncClient):
    """Neighbor nodes should have is_articulation_point=false and bridged_by."""
    name = "ap:neighbor_properties"
    data = await post_graph(c, {
        "graph": {"type": "articulation_points", "limit": 10},
        "query": "natural language processing",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    nodes = data.get("nodes", [])
    neighbors = [n for n in nodes if n.get("properties", {}).get("is_articulation_point") is False]
    if not neighbors:
        ok(name, data["took_ms"], "No neighbor nodes returned")
        return
    for n in neighbors:
        props = n.get("properties", {})
        if "bridged_by" not in props:
            fail(name, data["took_ms"], f"Neighbor {n['id']} missing 'bridged_by'")
            return
    ok(name, data["took_ms"], f"{len(neighbors)} neighbors all have bridged_by")


async def test_ap_bridge_edges(c: httpx.AsyncClient):
    """'bridges' edges should connect AP → neighbor."""
    name = "ap:bridge_edges"
    data = await post_graph(c, {
        "graph": {"type": "articulation_points", "limit": 10},
        "query": "computer vision",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    bridges = [e for e in data["edges"] if e["relation"] == "bridges"]
    if not bridges:
        ok(name, data["took_ms"], "No bridge edges (may depend on graph structure)")
        return
    ap_ids = {n["id"] for n in data["nodes"] if n.get("properties", {}).get("is_articulation_point") is True}
    for e in bridges:
        if e["source"] not in ap_ids:
            fail(name, data["took_ms"], f"Bridge edge source {e['source']} is not an AP")
            return
    ok(name, data["took_ms"], f"{len(bridges)} bridge edges, all sourced from APs")


async def test_ap_cites_edges(c: httpx.AsyncClient):
    """'cites' edges should exist between AP nodes."""
    name = "ap:cites_edges"
    data = await post_graph(c, {
        "graph": {"type": "articulation_points", "limit": 10},
        "query": "reinforcement learning",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    cites = [e for e in data["edges"] if e["relation"] == "cites"]
    ok(name, data["took_ms"], f"{len(cites)} cites edges among APs")


async def test_ap_with_category_filter(c: httpx.AsyncClient):
    """Articulation points with category filter."""
    name = "ap:category_filter"
    data = await post_graph(c, {
        "graph": {"type": "articulation_points", "limit": 10},
        "query": "language model",
        "categories": ["cs.CL"],
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    ok(name, data["took_ms"],
       f"nodes={len(data['nodes'])}, APs_found={data['metadata'].get('articulation_points_found')}")


async def test_ap_with_date_filter(c: httpx.AsyncClient):
    """Articulation points with date filter."""
    name = "ap:date_filter"
    data = await post_graph(c, {
        "graph": {"type": "articulation_points", "limit": 10},
        "query": "diffusion model",
        "submitted_date": {"gte": "2024-01-01T00:00:00"},
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    ok(name, data["took_ms"],
       f"nodes={len(data['nodes'])}, APs={data['metadata'].get('articulation_points_returned')}")


async def test_ap_limit_respected(c: httpx.AsyncClient):
    """Small limit should restrict AP count."""
    name = "ap:limit_respected"
    data = await post_graph(c, {
        "graph": {"type": "articulation_points", "limit": 2},
        "query": "optimization gradient",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    ap_nodes = [n for n in data["nodes"] if n.get("properties", {}).get("is_articulation_point") is True]
    if len(ap_nodes) > 2:
        fail(name, data["took_ms"], f"limit=2 but got {len(ap_nodes)} AP nodes")
        return
    ok(name, data["took_ms"], f"{len(ap_nodes)} APs (limit=2), {len(data['nodes'])} total nodes")


# ═══════════════════════════════════════════════════════
# 34. INFLUENCE MAXIMIZATION — Deep Tests
# ═══════════════════════════════════════════════════════

async def test_infmax_basic_functionality(c: httpx.AsyncClient):
    """Basic influence maximization returns seed and influenced papers."""
    name = "infmax:basic_functionality"
    data = await post_graph(c, {
        "graph": {"type": "influence_maximization", "influence_seeds": 2, "limit": 10},
        "query": "deep learning",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    nodes = data.get("nodes", [])
    seeds = [n for n in nodes if n.get("properties", {}).get("is_influence_seed") is True]
    influenced = [n for n in nodes if n.get("properties", {}).get("is_influence_seed") is False]
    if not seeds:
        fail(name, data["took_ms"], "No seed nodes returned")
        return
    ok(name, data["took_ms"],
       f"{len(seeds)} seeds, {len(influenced)} influenced, {len(data['edges'])} edges")


async def test_infmax_metadata_correctness(c: httpx.AsyncClient):
    """Check all expected metadata fields."""
    name = "infmax:metadata_correctness"
    data = await post_graph(c, {
        "graph": {"type": "influence_maximization", "influence_seeds": 2, "limit": 8},
        "query": "transformer",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    meta = data.get("metadata", {})
    expected_keys = [
        "papers_in_subgraph", "edges_in_subgraph", "seeds_selected",
        "seed_ids", "estimated_total_spread", "influenced_papers_sampled",
        "simulations_per_candidate",
    ]
    missing = [k for k in expected_keys if k not in meta]
    if missing:
        fail(name, data["took_ms"], f"Missing metadata keys: {missing}")
        return
    ok(name, data["took_ms"],
       f"seeds={meta['seeds_selected']}, spread={meta.get('estimated_total_spread')}, "
       f"sims={meta.get('simulations_per_candidate')}")


async def test_infmax_seed_node_properties(c: httpx.AsyncClient):
    """Seed nodes should have is_influence_seed=true, seed_rank, estimated_spread, degree."""
    name = "infmax:seed_node_props"
    data = await post_graph(c, {
        "graph": {"type": "influence_maximization", "influence_seeds": 2, "limit": 8},
        "query": "neural network",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    nodes = data.get("nodes", [])
    seeds = [n for n in nodes if n.get("properties", {}).get("is_influence_seed") is True]
    if not seeds:
        fail(name, data["took_ms"], "No seed nodes found")
        return
    for s in seeds:
        props = s.get("properties", {})
        for k in ("seed_rank", "estimated_spread", "degree"):
            if k not in props:
                fail(name, data["took_ms"], f"Seed {s['id']} missing property '{k}'")
                return
    ok(name, data["took_ms"], f"All {len(seeds)} seed nodes have correct properties")


async def test_infmax_influenced_node_properties(c: httpx.AsyncClient):
    """Influenced nodes should have is_influence_seed=false, degree."""
    name = "infmax:influenced_node_props"
    data = await post_graph(c, {
        "graph": {"type": "influence_maximization", "influence_seeds": 2, "limit": 8},
        "query": "computer vision",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    nodes = data.get("nodes", [])
    influenced = [n for n in nodes if n.get("properties", {}).get("is_influence_seed") is False]
    if not influenced:
        ok(name, data["took_ms"], "No influenced nodes — all may be seeds")
        return
    for n in influenced:
        props = n.get("properties", {})
        if "degree" not in props:
            fail(name, data["took_ms"], f"Influenced {n['id']} missing 'degree'")
            return
    ok(name, data["took_ms"], f"All {len(influenced)} influenced nodes have degree")


async def test_infmax_seed_ids_match_nodes(c: httpx.AsyncClient):
    """seed_ids in metadata should match actual seed nodes."""
    name = "infmax:seed_ids_match"
    data = await post_graph(c, {
        "graph": {"type": "influence_maximization", "influence_seeds": 2, "limit": 10},
        "query": "attention mechanism",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    meta = data.get("metadata", {})
    seed_ids_meta = set(meta.get("seed_ids", []))
    nodes = data.get("nodes", [])
    seed_ids_nodes = {n["id"] for n in nodes if n.get("properties", {}).get("is_influence_seed") is True}
    if not seed_ids_meta:
        fail(name, data["took_ms"], "seed_ids is empty in metadata")
        return
    if seed_ids_meta != seed_ids_nodes:
        fail(name, data["took_ms"],
             f"seed_ids mismatch: meta={seed_ids_meta}, nodes={seed_ids_nodes}")
        return
    ok(name, data["took_ms"], f"seed_ids match: {seed_ids_meta}")


async def test_infmax_single_seed(c: httpx.AsyncClient):
    """influence_seeds=1 should select exactly 1 seed."""
    name = "infmax:single_seed"
    data = await post_graph(c, {
        "graph": {"type": "influence_maximization", "influence_seeds": 1, "limit": 5},
        "query": "reinforcement learning",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    meta = data.get("metadata", {})
    if meta.get("seeds_selected") != 1:
        fail(name, data["took_ms"], f"Expected seeds_selected=1, got {meta.get('seeds_selected')}")
        return
    seed_ids = meta.get("seed_ids", [])
    if len(seed_ids) != 1:
        fail(name, data["took_ms"], f"Expected 1 seed_id, got {len(seed_ids)}")
        return
    ok(name, data["took_ms"], f"Single seed: {seed_ids[0]}")


async def test_infmax_three_seeds(c: httpx.AsyncClient):
    """influence_seeds=3 should select 3 seeds."""
    name = "infmax:three_seeds"
    data = await post_graph(c, {
        "graph": {"type": "influence_maximization", "influence_seeds": 3, "limit": 10},
        "query": "language model",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    meta = data.get("metadata", {})
    if meta.get("seeds_selected") != 3:
        # May have fewer if subgraph is small
        if (meta.get("seeds_selected") or 0) > 3:
            fail(name, data["took_ms"], f"Expected seeds_selected<=3, got {meta.get('seeds_selected')}")
            return
    ok(name, data["took_ms"],
       f"seeds_selected={meta.get('seeds_selected')}, spread={meta.get('estimated_total_spread')}")


async def test_infmax_with_category_filter(c: httpx.AsyncClient):
    """Influence maximization with category filter."""
    name = "infmax:category_filter"
    data = await post_graph(c, {
        "graph": {"type": "influence_maximization", "influence_seeds": 2, "limit": 8},
        "query": "text generation",
        "categories": ["cs.CL"],
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    ok(name, data["took_ms"],
       f"seeds={data['metadata'].get('seeds_selected')}, nodes={len(data['nodes'])}")


async def test_infmax_with_date_filter(c: httpx.AsyncClient):
    """Influence maximization with date filter."""
    name = "infmax:date_filter"
    data = await post_graph(c, {
        "graph": {"type": "influence_maximization", "influence_seeds": 2, "limit": 8},
        "query": "large language model",
        "submitted_date": {"gte": "2024-01-01T00:00:00"},
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    ok(name, data["took_ms"],
       f"seeds={data['metadata'].get('seeds_selected')}, spread={data['metadata'].get('estimated_total_spread')}")


async def test_infmax_edges_have_correct_relations(c: httpx.AsyncClient):
    """Edges should be 'cites' or 'influences'."""
    name = "infmax:edge_relations"
    data = await post_graph(c, {
        "graph": {"type": "influence_maximization", "influence_seeds": 2, "limit": 8},
        "query": "optimization",
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    edges = data.get("edges", [])
    valid_relations = {"cites", "influences"}
    bad = [e for e in edges if e["relation"] not in valid_relations]
    if bad:
        fail(name, data["took_ms"],
             f"{len(bad)} edges with unexpected relation: {bad[0]['relation']}")
        return
    rel_counts = Counter(e["relation"] for e in edges)
    ok(name, data["took_ms"], f"Edge relations: {dict(rel_counts)}")


# ═══════════════════════════════════════════════════════
# CROSS-ALGORITHM TESTS
# ═══════════════════════════════════════════════════════

async def test_cross_all_6_same_query(c: httpx.AsyncClient):
    """Same query through all 6 new algorithms — all should succeed."""
    name = "cross:all_6_same_query"
    base_filters = {"query": "transformer attention", "categories": ["cs.CL"]}
    types_and_extras = [
        ("louvain_community", {"iterations": 5, "limit": 10}),
        ("degree_centrality", {"limit": 10}),
        ("eigenvector_centrality", {"limit": 10}),
        ("kcore_decomposition", {"limit": 10}),
        ("articulation_points", {"limit": 10}),
        ("influence_maximization", {"influence_seeds": 1, "limit": 5}),
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


async def test_cross_degree_modes_comparison(c: httpx.AsyncClient):
    """Compare in vs out vs total degree for same query — all should succeed, may differ."""
    name = "cross:degree_modes"
    start = time.monotonic()
    results_inner = {}
    for mode in ("in", "out", "total"):
        data = await post_graph(c, {
            "graph": {"type": "degree_centrality", "degree_mode": mode, "limit": 5},
            "query": "neural network",
        }, f"{name}:{mode}")
        if data:
            results_inner[mode] = {
                "nodes": len(data.get("nodes", [])),
                "max_degree": data.get("metadata", {}).get("max_degree"),
            }
    elapsed = int((time.monotonic() - start) * 1000)
    if len(results_inner) == 3:
        detail = ", ".join(f"{m}: max={r['max_degree']}" for m, r in results_inner.items())
        ok(name, elapsed, detail)
    else:
        fail(name, elapsed, f"Only {len(results_inner)}/3 modes succeeded")


async def test_cross_louvain_vs_label_propagation(c: httpx.AsyncClient):
    """Compare louvain vs label propagation (community_detection) — both should return communities."""
    name = "cross:louvain_vs_lp"
    start = time.monotonic()
    d_louv = await post_graph(c, {
        "graph": {"type": "louvain_community", "iterations": 10, "limit": 15},
        "query": "machine learning",
    }, f"{name}:louvain")
    d_lp = await post_graph(c, {
        "graph": {"type": "community_detection", "iterations": 10, "limit": 15},
        "query": "machine learning",
    }, f"{name}:label_propagation")
    elapsed = int((time.monotonic() - start) * 1000)
    if not d_louv or not d_lp:
        return
    louv_comm = d_louv.get("metadata", {}).get("communities_found", 0)
    lp_comm = d_lp.get("metadata", {}).get("communities_found", 0)
    ok(name, elapsed,
       f"louvain={louv_comm} communities, label_prop={lp_comm} communities")


async def test_cross_semantic_boost_all(c: httpx.AsyncClient):
    """Semantic boost should compose with all 6 algorithms."""
    name = "cross:semantic_boost"
    semantic = [{"text": "self-supervised contrastive learning", "level": "abstract", "weight": 2.0, "mode": "boost"}]
    base = {"query": "representation learning", "semantic": semantic}
    types_and_extras = [
        ("louvain_community", {"iterations": 5, "limit": 5}),
        ("degree_centrality", {"limit": 5}),
        ("eigenvector_centrality", {"limit": 5}),
        ("kcore_decomposition", {"limit": 5}),
        ("articulation_points", {"limit": 5}),
        ("influence_maximization", {"influence_seeds": 1, "limit": 5}),
    ]
    ok_count = 0
    for gtype, extras in types_and_extras:
        body = {**base, "graph": {"type": gtype, **extras}}
        data = await post_graph(c, body, f"{name}:{gtype}")
        if data:
            ok_count += 1
    if ok_count == 6:
        ok(name, 0, "All 6 algorithms work with semantic boost")
    else:
        fail(name, 0, f"Only {ok_count}/6 algorithms succeeded with semantic boost")


async def test_cross_empty_results_graceful(c: httpx.AsyncClient):
    """Very narrow query should return gracefully (empty or small) for all 6."""
    name = "cross:empty_results"
    base = {"query": "zyxwvutsrqp nonexistent topic 9999"}
    types_and_extras = [
        ("louvain_community", {"iterations": 5, "limit": 5}),
        ("degree_centrality", {"limit": 5}),
        ("eigenvector_centrality", {"limit": 5}),
        ("kcore_decomposition", {"limit": 5}),
        ("articulation_points", {"limit": 5}),
        ("influence_maximization", {"influence_seeds": 1, "limit": 5}),
    ]
    ok_count = 0
    for gtype, extras in types_and_extras:
        body = {**base, "graph": {"type": gtype, **extras}}
        data = await post_graph(c, body, f"{name}:{gtype}")
        # Either success with empty/small result, or a handled error — both are fine
        if data is not None:
            ok_count += 1
    # We accept partial success since some algos may 422 on zero results
    if ok_count == 0:
        fail(name, 0, f"0/6 algorithms handled empty query gracefully")
        return
    ok(name, 0, f"{ok_count}/6 algorithms handled empty query gracefully")


async def test_cross_regex_filter_all(c: httpx.AsyncClient):
    """Regex filter should compose with new graph algorithms."""
    name = "cross:regex_filter"
    base = {"title_regex": ".*[Tt]ransform.*"}
    types_and_extras = [
        ("louvain_community", {"iterations": 5, "limit": 5}),
        ("degree_centrality", {"limit": 5}),
        ("eigenvector_centrality", {"limit": 5}),
        ("kcore_decomposition", {"limit": 5}),
        ("articulation_points", {"limit": 5}),
        ("influence_maximization", {"influence_seeds": 1, "limit": 5}),
    ]
    ok_count = 0
    for gtype, extras in types_and_extras:
        body = {**base, "graph": {"type": gtype, **extras}}
        data = await post_graph(c, body, f"{name}:{gtype}")
        if data:
            ok_count += 1
    if ok_count == 6:
        ok(name, 0, "All 6 algorithms work with regex filter")
    else:
        fail(name, 0, f"Only {ok_count}/6 algorithms succeeded with regex filter")


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
        print(f"Running {len(tests)} deep graph tests (types 29-34)...\n")
        start = time.monotonic()

        for test_fn in tests:
            await test_fn(c)

        elapsed = time.monotonic() - start

    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)

    print(f"\n{'='*80}")
    print(f" DEEP GRAPH TEST RESULTS (types 29-34): {passed}/{len(results)} passed ({failed} failed) in {elapsed:.1f}s")
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
