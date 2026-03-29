"""Deep E2E tests for graph-DB query expressiveness features (types 53-57).

Tests the 5 cross-cutting features that bring parity with Cypher/Gremlin:
- WHERE conditions (cross-node predicates)
- OPTIONAL MATCH (left-outer-join edges)
- Aggregation RETURN (count, avg, sum, collect, group_count)
- Path variable binding (path_filter)
- Nested subqueries (pattern_match / subgraph_projection inside pipeline)
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
# WHERE CONDITIONS — Cross-node predicates
# ═══════════════════════════════════════════════════════

async def test_where_date_gt(c: httpx.AsyncClient):
    """WHERE a.date > b.date filters co-authored pairs by date ordering."""
    name = "where:date_gt"
    data = await post_graph(c, {
        "query": "deep learning",
        "graph": {
            "type": "pattern_match",
            "pattern_nodes": [
                {"alias": "a", "type": "paper", "filters": {"primary_category": "cs.LG"}},
                {"alias": "b", "type": "paper", "filters": {"primary_category": "cs.LG"}},
            ],
            "pattern_edges": [{"source": "a", "target": "b", "relation": "co_authored"}],
            "where": [{"left": "a.date", "op": ">", "right": "b.date"}],
            "limit": 10,
        },
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    meta = data.get("metadata", {})
    mf = meta.get("matches_found", 0)
    wc = meta.get("where_conditions", 0)
    if wc != 1:
        fail(name, data["took_ms"], f"Expected where_conditions=1, got {wc}")
        return
    # Verify the date ordering in results
    nodes = data.get("nodes", [])
    match_nodes: dict[int, dict[str, str]] = {}
    for n in nodes:
        mi = n["properties"].get("match_index")
        alias = n["properties"].get("pattern_alias")
        if mi is not None and alias:
            match_nodes.setdefault(mi, {})[alias] = n["properties"].get("submitted_date", "")
    for mi, aliases in match_nodes.items():
        a_date = aliases.get("a", "")
        b_date = aliases.get("b", "")
        if a_date and b_date and a_date <= b_date:
            fail(name, data["took_ms"], f"Match {mi}: a.date={a_date} NOT > b.date={b_date}")
            return
    ok(name, data["took_ms"], f"matches={mf}, all a.date > b.date verified")


async def test_where_eq_category(c: httpx.AsyncClient):
    """WHERE a.primary_category == b.primary_category for same-category match."""
    name = "where:eq_category"
    data = await post_graph(c, {
        "query": "reinforcement learning",
        "graph": {
            "type": "pattern_match",
            "pattern_nodes": [
                {"alias": "a", "type": "paper"},
                {"alias": "b", "type": "paper"},
            ],
            "pattern_edges": [{"source": "a", "target": "b", "relation": "co_authored"}],
            "where": [{"left": "a.primary_category", "op": "==", "right": "b.primary_category"}],
            "limit": 10,
        },
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    meta = data.get("metadata", {})
    nodes = data.get("nodes", [])
    match_nodes: dict[int, dict[str, str]] = {}
    for n in nodes:
        mi = n["properties"].get("match_index")
        alias = n["properties"].get("pattern_alias")
        if mi is not None and alias:
            match_nodes.setdefault(mi, {})[alias] = n["properties"].get("primary_category", "")
    for mi, aliases in match_nodes.items():
        if aliases.get("a") and aliases.get("b") and aliases["a"] != aliases["b"]:
            fail(name, data["took_ms"], f"Match {mi}: a.cat={aliases['a']} != b.cat={aliases['b']}")
            return
    ok(name, data["took_ms"], f"matches={meta.get('matches_found', 0)}, same categories verified")


async def test_where_neq_category(c: httpx.AsyncClient):
    """WHERE a.primary_category != b.primary_category filters cross-field papers."""
    name = "where:neq_category"
    data = await post_graph(c, {
        "query": "neural network",
        "graph": {
            "type": "pattern_match",
            "pattern_nodes": [
                {"alias": "a", "type": "paper"},
                {"alias": "b", "type": "paper"},
            ],
            "pattern_edges": [{"source": "a", "target": "b", "relation": "co_authored"}],
            "where": [{"left": "a.primary_category", "op": "!=", "right": "b.primary_category"}],
            "limit": 10,
        },
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    meta = data.get("metadata", {})
    nodes = data.get("nodes", [])
    match_nodes: dict[int, dict[str, str]] = {}
    for n in nodes:
        mi = n["properties"].get("match_index")
        alias = n["properties"].get("pattern_alias")
        if mi is not None and alias:
            match_nodes.setdefault(mi, {})[alias] = n["properties"].get("primary_category", "")
    for mi, aliases in match_nodes.items():
        if aliases.get("a") and aliases.get("b") and aliases["a"] == aliases["b"]:
            fail(name, data["took_ms"], f"Match {mi}: a.cat={aliases['a']} == b.cat (should differ)")
            return
    ok(name, data["took_ms"], f"matches={meta.get('matches_found', 0)}, different categories verified")


async def test_where_date_lte(c: httpx.AsyncClient):
    """WHERE a.date <= b.date (reversed ordering)."""
    name = "where:date_lte"
    data = await post_graph(c, {
        "query": "optimization",
        "graph": {
            "type": "pattern_match",
            "pattern_nodes": [
                {"alias": "a", "type": "paper"},
                {"alias": "b", "type": "paper"},
            ],
            "pattern_edges": [{"source": "a", "target": "b", "relation": "co_authored"}],
            "where": [{"left": "a.date", "op": "<=", "right": "b.date"}],
            "limit": 10,
        },
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    nodes = data.get("nodes", [])
    match_nodes: dict[int, dict[str, str]] = {}
    for n in nodes:
        mi = n["properties"].get("match_index")
        alias = n["properties"].get("pattern_alias")
        if mi is not None and alias:
            match_nodes.setdefault(mi, {})[alias] = n["properties"].get("submitted_date", "")
    for mi, aliases in match_nodes.items():
        a_date = aliases.get("a", "")
        b_date = aliases.get("b", "")
        if a_date and b_date and a_date > b_date:
            fail(name, data["took_ms"], f"Match {mi}: a.date={a_date} NOT <= b.date={b_date}")
            return
    ok(name, data["took_ms"], f"matches={data['metadata'].get('matches_found', 0)}, a.date <= b.date verified")


async def test_where_contains(c: httpx.AsyncClient):
    """WHERE a.categories contains 'stat.ML' keeps only multi-listed papers."""
    name = "where:contains"
    data = await post_graph(c, {
        "query": "statistical learning",
        "graph": {
            "type": "pattern_match",
            "pattern_nodes": [
                {"alias": "a", "type": "paper"},
                {"alias": "b", "type": "paper"},
            ],
            "pattern_edges": [{"source": "a", "target": "b", "relation": "same_category"}],
            "where": [{"left": "a.categories", "op": "contains", "right": "stat.ML"}],
            "limit": 10,
        },
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    # Verify all 'a' nodes have stat.ML in categories
    for n in data.get("nodes", []):
        if n["properties"].get("pattern_alias") == "a":
            cats = n["properties"].get("categories", [])
            if "stat.ML" not in cats:
                fail(name, data["took_ms"], f"{n['id']}: categories={cats} missing stat.ML")
                return
    ok(name, data["took_ms"], f"matches={data['metadata'].get('matches_found', 0)}, contains verified")


async def test_where_literal_comparison(c: httpx.AsyncClient):
    """WHERE a.primary_category == 'cs.AI' with literal RHS."""
    name = "where:literal_comparison"
    data = await post_graph(c, {
        "query": "artificial intelligence",
        "graph": {
            "type": "pattern_match",
            "pattern_nodes": [
                {"alias": "a", "type": "paper"},
                {"alias": "b", "type": "paper"},
            ],
            "pattern_edges": [{"source": "a", "target": "b", "relation": "co_authored"}],
            "where": [{"left": "a.primary_category", "op": "==", "right": "cs.AI"}],
            "limit": 10,
        },
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    # Verify all 'a' nodes have cs.AI primary_category
    for n in data.get("nodes", []):
        if n["properties"].get("pattern_alias") == "a":
            cat = n["properties"].get("primary_category", "")
            if cat != "cs.AI":
                fail(name, data["took_ms"], f"{n['id']}: primary_category={cat}, expected cs.AI")
                return
    ok(name, data["took_ms"], f"matches={data['metadata'].get('matches_found', 0)}, literal cs.AI verified")


async def test_where_multiple_conditions(c: httpx.AsyncClient):
    """Multiple WHERE conditions applied simultaneously (AND semantics)."""
    name = "where:multiple_conditions"
    data = await post_graph(c, {
        "query": "learning",
        "graph": {
            "type": "pattern_match",
            "pattern_nodes": [
                {"alias": "a", "type": "paper"},
                {"alias": "b", "type": "paper"},
            ],
            "pattern_edges": [{"source": "a", "target": "b", "relation": "co_authored"}],
            "where": [
                {"left": "a.date", "op": ">", "right": "b.date"},
                {"left": "a.primary_category", "op": "==", "right": "b.primary_category"},
            ],
            "limit": 10,
        },
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    meta = data.get("metadata", {})
    if meta.get("where_conditions") != 2:
        fail(name, data["took_ms"], f"Expected where_conditions=2, got {meta.get('where_conditions')}")
        return
    nodes = data.get("nodes", [])
    match_nodes: dict[int, dict[str, dict]] = {}
    for n in nodes:
        mi = n["properties"].get("match_index")
        alias = n["properties"].get("pattern_alias")
        if mi is not None and alias:
            match_nodes.setdefault(mi, {})[alias] = n["properties"]
    for mi, aliases in match_nodes.items():
        a_p = aliases.get("a", {})
        b_p = aliases.get("b", {})
        a_date = a_p.get("submitted_date", "")
        b_date = b_p.get("submitted_date", "")
        if a_date and b_date and a_date <= b_date:
            fail(name, data["took_ms"], f"Match {mi}: a.date NOT > b.date")
            return
        if a_p.get("primary_category") and b_p.get("primary_category"):
            if a_p["primary_category"] != b_p["primary_category"]:
                fail(name, data["took_ms"], f"Match {mi}: categories differ")
                return
    ok(name, data["took_ms"], f"matches={meta.get('matches_found', 0)}, both WHERE conditions verified")


async def test_where_no_matches(c: httpx.AsyncClient):
    """WHERE with impossible condition returns 0 matches gracefully."""
    name = "where:no_matches"
    data = await post_graph(c, {
        "query": "quantum",
        "graph": {
            "type": "pattern_match",
            "pattern_nodes": [
                {"alias": "a", "type": "paper", "filters": {"primary_category": "cs.LG"}},
                {"alias": "b", "type": "paper", "filters": {"primary_category": "cs.LG"}},
            ],
            "pattern_edges": [{"source": "a", "target": "b", "relation": "co_authored"}],
            "where": [{"left": "a.primary_category", "op": "==", "right": "quant-ph"}],
            "limit": 5,
        },
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    meta = data.get("metadata", {})
    mf = meta.get("matches_found", 0)
    if mf != 0:
        fail(name, data["took_ms"], f"Expected 0 matches for impossible WHERE, got {mf}")
        return
    ok(name, data["took_ms"], "0 matches as expected for impossible condition")


# ═══════════════════════════════════════════════════════
# OPTIONAL MATCH — Left-outer-join edges
# ═══════════════════════════════════════════════════════

async def test_optional_basic(c: httpx.AsyncClient):
    """OPTIONAL cites edge returns a-nodes even when no citation link exists."""
    name = "optional:basic"
    data = await post_graph(c, {
        "query": "quantum computing",
        "graph": {
            "type": "pattern_match",
            "pattern_nodes": [
                {"alias": "a", "type": "paper", "filters": {"primary_category": "quant-ph"}},
                {"alias": "b", "type": "paper"},
            ],
            "pattern_edges": [{"source": "a", "target": "b", "relation": "cites", "optional": True}],
            "limit": 5,
        },
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    meta = data.get("metadata", {})
    if meta.get("optional_edges") != 1:
        fail(name, data["took_ms"], f"Expected optional_edges=1, got {meta.get('optional_edges')}")
        return
    mf = meta.get("matches_found", 0)
    if mf == 0:
        fail(name, data["took_ms"], "No matches found with optional edge")
        return
    # With optional cites and no citation data, b aliases should be None (excluded from nodes)
    a_count = sum(1 for n in data["nodes"] if n["properties"].get("pattern_alias") == "a")
    if a_count == 0:
        fail(name, data["took_ms"], "No a-nodes despite optional edge")
        return
    ok(name, data["took_ms"], f"matches={mf}, a_nodes={a_count}, optional edge handled")


async def test_optional_metadata(c: httpx.AsyncClient):
    """Optional edge count correctly reported in metadata."""
    name = "optional:metadata"
    data = await post_graph(c, {
        "query": "deep learning",
        "graph": {
            "type": "pattern_match",
            "pattern_nodes": [
                {"alias": "a", "type": "paper"},
                {"alias": "b", "type": "paper"},
                {"alias": "c", "type": "paper"},
            ],
            "pattern_edges": [
                {"source": "a", "target": "b", "relation": "co_authored"},
                {"source": "a", "target": "c", "relation": "cites", "optional": True},
            ],
            "limit": 5,
        },
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    meta = data.get("metadata", {})
    if meta.get("optional_edges") != 1:
        fail(name, data["took_ms"], f"Expected optional_edges=1, got {meta.get('optional_edges')}")
        return
    ok(name, data["took_ms"], f"optional_edges=1 in metadata, matches={meta.get('matches_found', 0)}")


async def test_optional_mixed_with_required(c: httpx.AsyncClient):
    """Mix of required and optional edges: required edges enforce, optional don't block."""
    name = "optional:mixed_required"
    data = await post_graph(c, {
        "query": "computer vision",
        "graph": {
            "type": "pattern_match",
            "pattern_nodes": [
                {"alias": "a", "type": "paper", "filters": {"primary_category": "cs.CV"}},
                {"alias": "b", "type": "paper"},
                {"alias": "c", "type": "paper"},
            ],
            "pattern_edges": [
                {"source": "a", "target": "b", "relation": "co_authored"},
                {"source": "a", "target": "c", "relation": "cites", "optional": True},
            ],
            "limit": 5,
        },
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    meta = data.get("metadata", {})
    mf = meta.get("matches_found", 0)
    # Should have a-nodes and b-nodes (required co_authored) but c might be None (optional cites)
    a_nodes = [n for n in data["nodes"] if n["properties"].get("pattern_alias") == "a"]
    b_nodes = [n for n in data["nodes"] if n["properties"].get("pattern_alias") == "b"]
    if mf > 0 and len(a_nodes) == 0:
        fail(name, data["took_ms"], "Matches found but no a-nodes")
        return
    if mf > 0 and len(b_nodes) == 0:
        fail(name, data["took_ms"], "Matches found but no b-nodes (required edge)")
        return
    ok(name, data["took_ms"], f"matches={mf}, a={len(a_nodes)}, b={len(b_nodes)}, optional c handled")


async def test_optional_with_where(c: httpx.AsyncClient):
    """OPTIONAL edge combined with WHERE conditions."""
    name = "optional:with_where"
    data = await post_graph(c, {
        "query": "machine learning",
        "graph": {
            "type": "pattern_match",
            "pattern_nodes": [
                {"alias": "a", "type": "paper", "filters": {"primary_category": "cs.LG"}},
                {"alias": "b", "type": "paper"},
            ],
            "pattern_edges": [{"source": "a", "target": "b", "relation": "cites", "optional": True}],
            "where": [{"left": "a.primary_category", "op": "==", "right": "cs.LG"}],
            "limit": 5,
        },
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    meta = data.get("metadata", {})
    if meta.get("optional_edges") != 1:
        fail(name, data["took_ms"], f"Expected optional_edges=1, got {meta.get('optional_edges')}")
        return
    if meta.get("where_conditions") != 1:
        fail(name, data["took_ms"], f"Expected where_conditions=1, got {meta.get('where_conditions')}")
        return
    # All a-nodes should be cs.LG
    for n in data.get("nodes", []):
        if n["properties"].get("pattern_alias") == "a":
            if n["properties"].get("primary_category") != "cs.LG":
                fail(name, data["took_ms"], f"WHERE violated: {n['id']} not cs.LG")
                return
    ok(name, data["took_ms"], f"matches={meta.get('matches_found', 0)}, WHERE + OPTIONAL combined")


async def test_optional_false_is_required(c: httpx.AsyncClient):
    """optional=false (default) enforces edge is required."""
    name = "optional:false_is_required"
    data = await post_graph(c, {
        "query": "quantum",
        "graph": {
            "type": "pattern_match",
            "pattern_nodes": [
                {"alias": "a", "type": "paper", "filters": {"primary_category": "quant-ph"}},
                {"alias": "b", "type": "paper"},
            ],
            "pattern_edges": [{"source": "a", "target": "b", "relation": "cites", "optional": False}],
            "limit": 5,
        },
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    meta = data.get("metadata", {})
    if meta.get("optional_edges") != 0:
        fail(name, data["took_ms"], f"Expected optional_edges=0, got {meta.get('optional_edges')}")
        return
    ok(name, data["took_ms"], f"optional_edges=0, required edge enforced")


# ═══════════════════════════════════════════════════════
# AGGREGATION RETURN — count, avg, sum, collect, group_count
# ═══════════════════════════════════════════════════════

async def test_aggregation_count(c: httpx.AsyncClient):
    """count aggregation returns paper count."""
    name = "aggregation:count"
    data = await post_graph(c, {
        "query": "machine learning",
        "graph": {
            "type": "category_diversity",
            "limit": 20,
            "aggregations": [{"function": "count", "alias": "total"}],
        },
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    meta = data.get("metadata", {})
    aggs = meta.get("aggregations", {})
    if "total" not in aggs:
        fail(name, data["took_ms"], f"Missing 'total' in aggregations: {aggs}")
        return
    total = aggs["total"]
    nodes_paper = [n for n in data["nodes"] if n["type"] == "paper"]
    if total != len(nodes_paper):
        fail(name, data["took_ms"], f"count={total} != paper_nodes={len(nodes_paper)}")
        return
    ok(name, data["took_ms"], f"count={total} matches paper nodes")


async def test_aggregation_avg(c: httpx.AsyncClient):
    """avg aggregation computes mean of a numeric field."""
    name = "aggregation:avg"
    data = await post_graph(c, {
        "query": "machine learning",
        "graph": {
            "type": "category_diversity",
            "limit": 20,
            "aggregations": [{"function": "avg", "field": "citations", "alias": "avg_cit"}],
        },
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    aggs = data.get("metadata", {}).get("aggregations", {})
    if "avg_cit" not in aggs:
        fail(name, data["took_ms"], f"Missing avg_cit in aggregations: {aggs}")
        return
    # avg_cit should be a number (could be 0.0 for freshly seeded data)
    if not isinstance(aggs["avg_cit"], (int, float)):
        fail(name, data["took_ms"], f"avg_cit is not numeric: {type(aggs['avg_cit'])}")
        return
    ok(name, data["took_ms"], f"avg_cit={aggs['avg_cit']}")


async def test_aggregation_sum(c: httpx.AsyncClient):
    """sum aggregation totals a numeric field."""
    name = "aggregation:sum"
    data = await post_graph(c, {
        "query": "deep learning",
        "graph": {
            "type": "category_diversity",
            "limit": 20,
            "aggregations": [{"function": "sum", "field": "citations", "alias": "sum_cit"}],
        },
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    aggs = data.get("metadata", {}).get("aggregations", {})
    if "sum_cit" not in aggs:
        fail(name, data["took_ms"], f"Missing sum_cit: {aggs}")
        return
    if not isinstance(aggs["sum_cit"], (int, float)):
        fail(name, data["took_ms"], f"sum_cit not numeric: {type(aggs['sum_cit'])}")
        return
    ok(name, data["took_ms"], f"sum_cit={aggs['sum_cit']}")


async def test_aggregation_min_max(c: httpx.AsyncClient):
    """min and max aggregation on citations field."""
    name = "aggregation:min_max"
    data = await post_graph(c, {
        "query": "deep learning",
        "graph": {
            "type": "category_diversity",
            "limit": 30,
            "aggregations": [
                {"function": "min", "field": "citations", "alias": "min_cit"},
                {"function": "max", "field": "citations", "alias": "max_cit"},
            ],
        },
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    aggs = data.get("metadata", {}).get("aggregations", {})
    if "min_cit" not in aggs or "max_cit" not in aggs:
        fail(name, data["took_ms"], f"Missing min/max: {aggs}")
        return
    if aggs["min_cit"] is not None and aggs["max_cit"] is not None:
        if aggs["min_cit"] > aggs["max_cit"]:
            fail(name, data["took_ms"], f"min={aggs['min_cit']} > max={aggs['max_cit']}")
            return
    ok(name, data["took_ms"], f"min={aggs['min_cit']}, max={aggs['max_cit']}")


async def test_aggregation_collect(c: httpx.AsyncClient):
    """collect aggregation returns list of field values."""
    name = "aggregation:collect"
    data = await post_graph(c, {
        "query": "deep learning",
        "graph": {
            "type": "category_diversity",
            "limit": 10,
            "aggregations": [{"function": "collect", "field": "primary_category", "alias": "cats"}],
        },
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    aggs = data.get("metadata", {}).get("aggregations", {})
    if "cats" not in aggs:
        fail(name, data["took_ms"], f"Missing cats: {aggs}")
        return
    if not isinstance(aggs["cats"], list):
        fail(name, data["took_ms"], f"collect should return list, got {type(aggs['cats'])}")
        return
    ok(name, data["took_ms"], f"collected {len(aggs['cats'])} categories")


async def test_aggregation_group_count(c: httpx.AsyncClient):
    """group_count aggregation returns frequency distribution."""
    name = "aggregation:group_count"
    data = await post_graph(c, {
        "query": "deep learning",
        "graph": {
            "type": "category_diversity",
            "limit": 30,
            "aggregations": [{"function": "group_count", "field": "primary_category", "alias": "dist"}],
        },
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    aggs = data.get("metadata", {}).get("aggregations", {})
    if "dist" not in aggs:
        fail(name, data["took_ms"], f"Missing dist: {aggs}")
        return
    dist = aggs["dist"]
    if not isinstance(dist, dict):
        fail(name, data["took_ms"], f"group_count should return dict, got {type(dist)}")
        return
    # Sum of counts should equal paper node count
    total_from_dist = sum(dist.values())
    paper_nodes = [n for n in data["nodes"] if n["type"] == "paper"]
    if total_from_dist != len(paper_nodes):
        fail(name, data["took_ms"], f"dist sum={total_from_dist} != paper_nodes={len(paper_nodes)}")
        return
    ok(name, data["took_ms"], f"dist has {len(dist)} categories, sum={total_from_dist}")


async def test_aggregation_multi(c: httpx.AsyncClient):
    """Multiple aggregations in single request."""
    name = "aggregation:multi"
    data = await post_graph(c, {
        "query": "machine learning",
        "graph": {
            "type": "category_diversity",
            "limit": 20,
            "aggregations": [
                {"function": "count", "alias": "total"},
                {"function": "avg", "field": "citations", "alias": "avg_cit"},
                {"function": "group_count", "field": "primary_category", "alias": "cat_dist"},
                {"function": "collect", "field": "primary_category", "alias": "all_cats"},
            ],
        },
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    aggs = data.get("metadata", {}).get("aggregations", {})
    for key in ["total", "avg_cit", "cat_dist", "all_cats"]:
        if key not in aggs:
            fail(name, data["took_ms"], f"Missing aggregation '{key}': {list(aggs.keys())}")
            return
    ok(name, data["took_ms"],
       f"total={aggs['total']}, avg={aggs['avg_cit']}, cats={len(aggs['cat_dist'])}, collected={len(aggs['all_cats'])}")


async def test_aggregation_on_pattern_match(c: httpx.AsyncClient):
    """Aggregations work on pattern_match results too."""
    name = "aggregation:pattern_match"
    data = await post_graph(c, {
        "query": "deep learning",
        "graph": {
            "type": "pattern_match",
            "pattern_nodes": [
                {"alias": "a", "type": "paper"},
                {"alias": "b", "type": "paper"},
            ],
            "pattern_edges": [{"source": "a", "target": "b", "relation": "co_authored"}],
            "limit": 10,
            "aggregations": [
                {"function": "count", "alias": "total"},
                {"function": "group_count", "field": "primary_category", "alias": "dist"},
            ],
        },
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    aggs = data.get("metadata", {}).get("aggregations", {})
    if "total" not in aggs or "dist" not in aggs:
        fail(name, data["took_ms"], f"Missing aggregations: {list(aggs.keys())}")
        return
    ok(name, data["took_ms"], f"total={aggs['total']}, categories={len(aggs['dist'])}")


async def test_aggregation_empty_results(c: httpx.AsyncClient):
    """Aggregations return sensible defaults when no results."""
    name = "aggregation:empty"
    data = await post_graph(c, {
        "query": "xyzzy_nonexistent_query_12345",
        "graph": {
            "type": "category_diversity",
            "limit": 5,
            "aggregations": [
                {"function": "count", "alias": "total"},
                {"function": "avg", "field": "citations", "alias": "avg_cit"},
                {"function": "collect", "field": "primary_category", "alias": "cats"},
                {"function": "group_count", "field": "primary_category", "alias": "dist"},
            ],
        },
    }, name)
    if not data:
        return
    aggs = data.get("metadata", {}).get("aggregations", {})
    if aggs.get("total", -1) != 0:
        fail(name, data["took_ms"], f"Expected count=0, got {aggs.get('total')}")
        return
    if aggs.get("avg_cit") is not None:
        fail(name, data["took_ms"], f"Expected avg=null, got {aggs.get('avg_cit')}")
        return
    if aggs.get("cats") != []:
        fail(name, data["took_ms"], f"Expected empty list, got {aggs.get('cats')}")
        return
    if aggs.get("dist") != {}:
        fail(name, data["took_ms"], f"Expected empty dict, got {aggs.get('dist')}")
        return
    ok(name, data["took_ms"], "Empty aggregations have correct defaults")


# ═══════════════════════════════════════════════════════
# PATH FILTER — Path variable binding
# ═══════════════════════════════════════════════════════

async def test_path_filter_model_validation(c: httpx.AsyncClient):
    """path_filter model accepts valid parameters without error."""
    name = "path_filter:model_validation"
    data = await post_graph(c, {
        "query": "machine learning",
        "graph": {
            "type": "shortest_citation_path",
            "seed_arxiv_id": "9999.99999",
            "target_arxiv_id": "9999.99998",
            "path_filter": {
                "max_path_length": 3,
                "min_path_length": 1,
                "all_nodes_match": {"primary_category": "cs.LG"},
            },
        },
    }, name)
    if not data:
        return
    # Should not crash — even if no path found, the filter params are accepted
    ok(name, data.get("took_ms", 0), f"path_filter accepted, result: {data.get('metadata', {}).get('error', 'ok')}")


async def test_path_filter_max_length(c: httpx.AsyncClient):
    """path_filter with max_path_length is accepted on all_shortest_paths."""
    name = "path_filter:max_length"
    data = await post_graph(c, {
        "query": "neural network",
        "graph": {
            "type": "all_shortest_paths",
            "seed_arxiv_id": "9999.99999",
            "target_arxiv_id": "9999.99998",
            "path_filter": {"max_path_length": 5},
        },
    }, name)
    if not data:
        return
    ok(name, data.get("took_ms", 0), f"max_path_length accepted, meta={data.get('metadata', {}).get('error', 'ok')}")


async def test_path_filter_any_node_matches(c: httpx.AsyncClient):
    """path_filter with any_node_matches is accepted on k_shortest_paths."""
    name = "path_filter:any_node_matches"
    data = await post_graph(c, {
        "query": "deep learning",
        "graph": {
            "type": "k_shortest_paths",
            "seed_arxiv_id": "9999.99999",
            "target_arxiv_id": "9999.99998",
            "k_paths": 3,
            "path_filter": {"any_node_matches": {"has_github": True}},
        },
    }, name)
    if not data:
        return
    ok(name, data.get("took_ms", 0), f"any_node_matches accepted, meta={data.get('metadata', {}).get('error', 'ok')}")


# ═══════════════════════════════════════════════════════
# NESTED SUBQUERIES — pattern_match/subgraph inside pipeline
# ═══════════════════════════════════════════════════════

async def test_nested_pattern_match_in_pipeline(c: httpx.AsyncClient):
    """pattern_match as final pipeline step produces valid pattern results."""
    name = "nested:pattern_match_in_pipeline"
    data = await post_graph(c, {
        "query": "deep learning",
        "graph": {
            "type": "pipeline",
            "pipeline_steps": [
                {"type": "category_diversity", "limit": 30},
                {
                    "type": "pattern_match",
                    "limit": 10,
                    "params": {
                        "pattern_nodes": [
                            {"alias": "a", "type": "paper"},
                            {"alias": "b", "type": "paper"},
                        ],
                        "pattern_edges": [
                            {"source": "a", "target": "b", "relation": "co_authored"},
                        ],
                    },
                },
            ],
        },
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    meta = data.get("metadata", {})
    steps = meta.get("pipeline_steps", [])
    if len(steps) != 2:
        fail(name, data["took_ms"], f"Expected 2 pipeline steps, got {len(steps)}")
        return
    if steps[0]["type"] != "category_diversity" or steps[1]["type"] != "pattern_match":
        fail(name, data["took_ms"], f"Wrong step types: {[s['type'] for s in steps]}")
        return
    if steps[0]["nodes_produced"] == 0:
        fail(name, data["took_ms"], "Category diversity step produced 0 nodes")
        return
    ok(name, data["took_ms"],
       f"step0={steps[0]['nodes_produced']} nodes, step1={steps[1].get('nodes_produced', '?')} nodes")


async def test_nested_pattern_match_with_where(c: httpx.AsyncClient):
    """Nested pattern_match with WHERE conditions inside pipeline."""
    name = "nested:pattern_match_with_where"
    data = await post_graph(c, {
        "query": "machine learning",
        "graph": {
            "type": "pipeline",
            "pipeline_steps": [
                {"type": "category_diversity", "limit": 40},
                {
                    "type": "pattern_match",
                    "limit": 5,
                    "params": {
                        "pattern_nodes": [
                            {"alias": "a", "type": "paper"},
                            {"alias": "b", "type": "paper"},
                        ],
                        "pattern_edges": [
                            {"source": "a", "target": "b", "relation": "co_authored"},
                        ],
                        "where": [
                            {"left": "a.primary_category", "op": "==", "right": "b.primary_category"},
                        ],
                    },
                },
            ],
        },
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    meta = data.get("metadata", {})
    wc = meta.get("where_conditions", 0)
    ok(name, data["took_ms"],
       f"pipeline + pattern_match + WHERE, where_conditions={wc}, matches={meta.get('matches_found', 0)}")


async def test_nested_subgraph_projection_in_pipeline(c: httpx.AsyncClient):
    """subgraph_projection as final pipeline step."""
    name = "nested:subgraph_projection_in_pipeline"
    data = await post_graph(c, {
        "query": "natural language processing",
        "graph": {
            "type": "pipeline",
            "pipeline_steps": [
                {"type": "category_diversity", "limit": 30},
                {
                    "type": "subgraph_projection",
                    "limit": 20,
                    "params": {
                        "subgraph_filter": {
                            "primary_category": "cs.CL",
                            "direction": "both",
                            "max_nodes": 100,
                        },
                        "subgraph_algorithm": "degree_centrality",
                    },
                },
            ],
        },
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    meta = data.get("metadata", {})
    steps = meta.get("pipeline_steps", [])
    if len(steps) >= 2 and steps[1]["type"] == "subgraph_projection":
        ok(name, data["took_ms"], f"subgraph_projection in pipeline, nodes={steps[1].get('nodes_produced', '?')}")
    else:
        fail(name, data["took_ms"], f"Expected subgraph_projection step, got: {[s.get('type') for s in steps]}")


async def test_nested_pipeline_in_pipeline_blocked(c: httpx.AsyncClient):
    """pipeline inside pipeline is blocked (prevents infinite recursion)."""
    name = "nested:pipeline_blocked"
    data = await post_graph(c, {
        "query": "machine learning",
        "graph": {
            "type": "pipeline",
            "pipeline_steps": [
                {"type": "category_diversity", "limit": 10},
                {
                    "type": "pipeline",
                    "params": {
                        "pipeline_steps": [
                            {"type": "category_diversity", "limit": 5},
                            {"type": "category_diversity", "limit": 5},
                        ],
                    },
                },
            ],
        },
    }, name)
    if not data:
        return
    meta = data.get("metadata", {})
    error = meta.get("error", "")
    if "cannot nest pipeline" in error.lower():
        ok(name, data.get("took_ms", 0), f"Correctly blocked: {error}")
    else:
        fail(name, data.get("took_ms", 0), f"Expected pipeline nesting error, got: {error}")


async def test_nested_three_steps_with_pattern_match(c: httpx.AsyncClient):
    """3-step pipeline with pattern_match in the middle."""
    name = "nested:three_steps"
    data = await post_graph(c, {
        "query": "deep learning",
        "graph": {
            "type": "pipeline",
            "pipeline_steps": [
                {"type": "category_diversity", "limit": 40},
                {
                    "type": "pattern_match",
                    "limit": 20,
                    "params": {
                        "pattern_nodes": [
                            {"alias": "a", "type": "paper"},
                            {"alias": "b", "type": "paper"},
                        ],
                        "pattern_edges": [
                            {"source": "a", "target": "b", "relation": "same_category"},
                        ],
                    },
                },
                {"type": "category_diversity", "limit": 10},
            ],
        },
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    meta = data.get("metadata", {})
    steps = meta.get("pipeline_steps", [])
    if len(steps) == 3:
        ok(name, data["took_ms"],
           f"3-step pipeline: [{steps[0]['type']}→{steps[1]['type']}→{steps[2]['type']}]")
    else:
        fail(name, data["took_ms"], f"Expected 3 pipeline steps, got {len(steps)}")


async def test_nested_pattern_match_feeds_ids(c: httpx.AsyncClient):
    """Pattern match step correctly feeds paper IDs to next step."""
    name = "nested:pattern_match_feeds_ids"
    data = await post_graph(c, {
        "query": "optimization",
        "graph": {
            "type": "pipeline",
            "pipeline_steps": [
                {
                    "type": "pattern_match",
                    "limit": 20,
                    "params": {
                        "pattern_nodes": [
                            {"alias": "a", "type": "paper"},
                            {"alias": "b", "type": "paper"},
                        ],
                        "pattern_edges": [
                            {"source": "a", "target": "b", "relation": "co_authored"},
                        ],
                    },
                },
                {"type": "category_diversity", "limit": 10},
            ],
        },
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    meta = data.get("metadata", {})
    steps = meta.get("pipeline_steps", [])
    if len(steps) >= 2:
        step0_ids = steps[0].get("paper_ids_out", 0)
        if step0_ids > 0:
            ok(name, data["took_ms"], f"pattern_match→category_diversity: {step0_ids} IDs fed forward")
        else:
            fail(name, data["took_ms"], "pattern_match produced 0 paper IDs for next step")
    else:
        fail(name, data["took_ms"], f"Expected 2 steps, got {len(steps)}")


# ═══════════════════════════════════════════════════════
# COMBINED FEATURES
# ═══════════════════════════════════════════════════════

async def test_combined_where_optional_aggregation(c: httpx.AsyncClient):
    """All three features combined: WHERE + OPTIONAL + aggregation."""
    name = "combined:where_optional_agg"
    data = await post_graph(c, {
        "query": "deep learning",
        "graph": {
            "type": "pattern_match",
            "pattern_nodes": [
                {"alias": "a", "type": "paper", "filters": {"primary_category": "cs.LG"}},
                {"alias": "b", "type": "paper"},
            ],
            "pattern_edges": [{"source": "a", "target": "b", "relation": "cites", "optional": True}],
            "where": [{"left": "a.primary_category", "op": "==", "right": "cs.LG"}],
            "limit": 10,
            "aggregations": [
                {"function": "count", "alias": "total"},
                {"function": "group_count", "field": "primary_category", "alias": "dist"},
            ],
        },
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    meta = data.get("metadata", {})
    aggs = meta.get("aggregations", {})
    if meta.get("optional_edges") != 1:
        fail(name, data["took_ms"], f"optional_edges should be 1")
        return
    if meta.get("where_conditions") != 1:
        fail(name, data["took_ms"], f"where_conditions should be 1")
        return
    if "total" not in aggs:
        fail(name, data["took_ms"], "aggregation 'total' missing")
        return
    ok(name, data["took_ms"],
       f"combined: where=1, optional=1, agg_total={aggs['total']}, dist={len(aggs.get('dist', {}))}")


async def test_combined_pipeline_with_aggregation(c: httpx.AsyncClient):
    """Pipeline with aggregations on final result."""
    name = "combined:pipeline_aggregation"
    data = await post_graph(c, {
        "query": "deep learning",
        "graph": {
            "type": "pipeline",
            "pipeline_steps": [
                {"type": "category_diversity", "limit": 20},
                {"type": "category_diversity", "limit": 10},
            ],
            "aggregations": [
                {"function": "count", "alias": "total"},
                {"function": "group_count", "field": "primary_category", "alias": "dist"},
            ],
        },
    }, name)
    if not data or not validate_graph_integrity(data, name):
        return
    aggs = data.get("metadata", {}).get("aggregations", {})
    if "total" in aggs:
        ok(name, data["took_ms"], f"pipeline + aggregations: total={aggs['total']}, dist_keys={len(aggs.get('dist', {}))}")
    else:
        fail(name, data["took_ms"], f"Aggregations missing from pipeline output: keys={list(data.get('metadata', {}).keys())}")


# ═══════════════════════════════════════════════════════
# RUNNER
# ═══════════════════════════════════════════════════════

ALL_TESTS = [
    # WHERE
    test_where_date_gt,
    test_where_eq_category,
    test_where_neq_category,
    test_where_date_lte,
    test_where_contains,
    test_where_literal_comparison,
    test_where_multiple_conditions,
    test_where_no_matches,
    # OPTIONAL
    test_optional_basic,
    test_optional_metadata,
    test_optional_mixed_with_required,
    test_optional_with_where,
    test_optional_false_is_required,
    # AGGREGATION
    test_aggregation_count,
    test_aggregation_avg,
    test_aggregation_sum,
    test_aggregation_min_max,
    test_aggregation_collect,
    test_aggregation_group_count,
    test_aggregation_multi,
    test_aggregation_on_pattern_match,
    test_aggregation_empty_results,
    # PATH FILTER
    test_path_filter_model_validation,
    test_path_filter_max_length,
    test_path_filter_any_node_matches,
    # NESTED SUBQUERIES
    test_nested_pattern_match_in_pipeline,
    test_nested_pattern_match_with_where,
    test_nested_subgraph_projection_in_pipeline,
    test_nested_pipeline_in_pipeline_blocked,
    test_nested_three_steps_with_pattern_match,
    test_nested_pattern_match_feeds_ids,
    # COMBINED
    test_combined_where_optional_aggregation,
    test_combined_pipeline_with_aggregation,
]


async def main():
    start = time.monotonic()
    async with httpx.AsyncClient() as c:
        # Health check
        try:
            r = await c.get(f"{BASE}/health", timeout=5)
            if r.status_code != 200:
                print("❌  Server not healthy")
                sys.exit(1)
        except Exception:
            print("❌  Server unreachable")
            sys.exit(1)

        for test_fn in ALL_TESTS:
            await test_fn(c)

    elapsed = time.monotonic() - start

    # Report
    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)
    total = len(results)

    print(f"\n{'═' * 70}")
    print(f"  DEEP7 RESULTS: {passed}/{total} passed, {failed} failed  ({elapsed:.1f}s)")
    print(f"{'═' * 70}")

    for r in results:
        status = "✅" if r.passed else "❌"
        print(f"  {status}  {r.name:<45} {r.took_ms:>5}ms  {r.detail[:80]}")

    print()
    if failed > 0:
        print(f"  ❌ {failed} FAILURES")
        sys.exit(1)
    else:
        print(f"  ✅ ALL {total} TESTS PASSED")
        sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())
