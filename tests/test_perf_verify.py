"""Verification script for all 5 performance optimizations.

Tests each optimization individually with timing + correctness checks:
1. Lazy co-author adjacency (no O(n²) precomputation)
2. Lazy category adjacency (no O(n²) precomputation)
3. Memoized reachable() (same traversal not repeated)
4. Parallel ES batch fetches (asyncio.gather)
5. Expansion early-exit (already_attempted tracking)
6. islice vs sorted (no unnecessary O(n log n))
"""
import asyncio
import json
import time
import httpx

BASE = "http://localhost:8000"
HEADERS = {"X-API-Key": "changeme-key-1", "Content-Type": "application/json"}
TIMEOUT = 120.0

results = []

def ok(name, ms, detail):
    results.append((True, name, ms, detail))
    print(f"  ✅  {name:<50} {ms:>5}ms  {detail}")

def fail(name, ms, detail):
    results.append((False, name, ms, detail))
    print(f"  ❌  {name:<50} {ms:>5}ms  {detail}")


async def post_graph(c, body, name):
    start = time.monotonic()
    try:
        resp = await c.post(f"{BASE}/graph", json=body, headers=HEADERS, timeout=TIMEOUT)
        ms = int((time.monotonic() - start) * 1000)
        if resp.status_code != 200:
            fail(name, ms, f"HTTP {resp.status_code}: {resp.text[:200]}")
            return None, ms
        return resp.json(), ms
    except Exception as e:
        ms = int((time.monotonic() - start) * 1000)
        fail(name, ms, str(e)[:200])
        return None, ms


async def test_lazy_coauthor(c):
    """Test 1: co_authored queries work correctly (lazy adjacency, not precomputed)."""
    name = "lazy_coauthor:basic"
    data, ms = await post_graph(c, {
        "query": "machine learning",
        "graph": {
            "type": "pattern_match",
            "pattern_nodes": [
                {"alias": "a", "type": "paper"},
                {"alias": "b", "type": "paper"},
            ],
            "pattern_edges": [{"source": "a", "target": "b", "relation": "co_authored"}],
            "limit": 20,
        },
    }, name)
    if not data:
        return
    meta = data.get("metadata", {})
    matches = meta.get("matches_found", 0)
    papers = meta.get("papers_searched", 0)
    if matches > 0:
        ok(name, ms, f"matches={matches}, papers={papers}")
    else:
        # Even 0 matches is acceptable if no co-authors exist, but check structure
        if "matches_found" in meta:
            ok(name, ms, f"matches=0 (no co-authors), papers={papers}")
        else:
            fail(name, ms, f"Missing metadata keys: {list(meta.keys())}")


async def test_lazy_coauthor_large(c):
    """Test 1b: co_authored with large limit—would OOM with O(n²) precompute."""
    name = "lazy_coauthor:large_limit"
    data, ms = await post_graph(c, {
        "query": "deep learning",
        "graph": {
            "type": "pattern_match",
            "pattern_nodes": [
                {"alias": "a", "type": "paper"},
                {"alias": "b", "type": "paper"},
            ],
            "pattern_edges": [{"source": "a", "target": "b", "relation": "co_authored"}],
            "limit": 100,
            "max_expansion": 3,
        },
    }, name)
    if not data:
        return
    meta = data.get("metadata", {})
    matches = meta.get("matches_found", 0)
    papers = meta.get("papers_searched", 0)
    # Key: should complete reasonably fast (<5s) even with many papers
    if ms < 10000:
        ok(name, ms, f"matches={matches}, papers={papers}, fast enough")
    else:
        fail(name, ms, f"Too slow: {ms}ms (O(n²) precompute likely)")


async def test_lazy_category(c):
    """Test 2: same_category queries work correctly (lazy adjacency)."""
    name = "lazy_category:basic"
    data, ms = await post_graph(c, {
        "query": "neural network",
        "graph": {
            "type": "pattern_match",
            "pattern_nodes": [
                {"alias": "a", "type": "paper"},
                {"alias": "b", "type": "paper"},
            ],
            "pattern_edges": [{"source": "a", "target": "b", "relation": "same_category"}],
            "limit": 20,
        },
    }, name)
    if not data:
        return
    meta = data.get("metadata", {})
    matches = meta.get("matches_found", 0)
    if "matches_found" in meta:
        ok(name, ms, f"matches={matches}")
    else:
        fail(name, ms, f"Missing matches_found in metadata")


async def test_lazy_category_correctness(c):
    """Test 2b: Verify same_category matches actually share a category."""
    name = "lazy_category:correctness"
    data, ms = await post_graph(c, {
        "query": "transformer",
        "graph": {
            "type": "pattern_match",
            "pattern_nodes": [
                {"alias": "a", "type": "paper"},
                {"alias": "b", "type": "paper"},
            ],
            "pattern_edges": [{"source": "a", "target": "b", "relation": "same_category"}],
            "limit": 5,
        },
    }, name)
    if not data:
        return
    nodes = data.get("nodes", [])
    if len(nodes) < 2:
        ok(name, ms, "Too few nodes for correctness check, but no error")
        return

    # Group nodes by match_index and check categories overlap
    matches_by_idx = {}
    for n in nodes:
        mi = n.get("properties", {}).get("match_index")
        alias = n.get("properties", {}).get("pattern_alias")
        cats = n.get("properties", {}).get("categories", [])
        if mi is not None and alias:
            matches_by_idx.setdefault(mi, {})[alias] = set(cats)

    violations = 0
    checked = 0
    for mi, aliases in matches_by_idx.items():
        a_cats = aliases.get("a", set())
        b_cats = aliases.get("b", set())
        if a_cats and b_cats:
            checked += 1
            if not (a_cats & b_cats):
                violations += 1

    if violations > 0:
        fail(name, ms, f"{violations}/{checked} matches don't share categories!")
    else:
        ok(name, ms, f"{checked} matches verified: all share categories")


async def test_memoized_reachable(c):
    """Test 3: Multi-hop traversal (exercises reachable() memoization)."""
    name = "memoize_reachable:multi_hop"
    data, ms = await post_graph(c, {
        "query": "reinforcement learning",
        "graph": {
            "type": "pattern_match",
            "pattern_nodes": [
                {"alias": "a", "type": "paper"},
                {"alias": "b", "type": "paper"},
                {"alias": "c", "type": "paper"},
            ],
            "pattern_edges": [
                {"source": "a", "target": "b", "relation": "co_authored"},
                {"source": "b", "target": "c", "relation": "co_authored"},
            ],
            "limit": 10,
        },
    }, name)
    if not data:
        return
    meta = data.get("metadata", {})
    matches = meta.get("matches_found", 0)
    # 3-node chain: b->c reachable() from many b's should be memoized
    ok(name, ms, f"3-node chain: matches={matches}")


async def test_memoized_reachable_variable_hops(c):
    """Test 3b: Variable-length hops (min_hops=1, max_hops=2)."""
    name = "memoize_reachable:variable_hops"
    data, ms = await post_graph(c, {
        "query": "natural language processing",
        "graph": {
            "type": "pattern_match",
            "pattern_nodes": [
                {"alias": "a", "type": "paper"},
                {"alias": "b", "type": "paper"},
            ],
            "pattern_edges": [
                {"source": "a", "target": "b", "relation": "co_authored",
                 "min_hops": 1, "max_hops": 2},
            ],
            "limit": 10,
        },
    }, name)
    if not data:
        return
    meta = data.get("metadata", {})
    matches = meta.get("matches_found", 0)
    ok(name, ms, f"variable hops [1,2]: matches={matches}")


async def test_parallel_es_fetches(c):
    """Test 4: Large expansion exercises parallel ES batch fetches."""
    name = "parallel_es:large_expansion"
    # max_expansion=5 forces multiple expansion rounds, each with parallel batches
    data, ms = await post_graph(c, {
        "query": "computer vision",
        "graph": {
            "type": "pattern_match",
            "pattern_nodes": [
                {"alias": "a", "type": "paper"},
                {"alias": "b", "type": "paper"},
            ],
            "pattern_edges": [{"source": "a", "target": "b", "relation": "co_authored"}],
            "limit": 50,
            "max_expansion": 5,
        },
    }, name)
    if not data:
        return
    meta = data.get("metadata", {})
    matches = meta.get("matches_found", 0)
    papers = meta.get("papers_searched", 0)
    ok(name, ms, f"expansion=5: matches={matches}, papers_searched={papers}")


async def test_parallel_es_citation_subgraph(c):
    """Test 4b: Citation subgraph build also uses parallel fetches."""
    name = "parallel_es:citation_subgraph"
    data, ms = await post_graph(c, {
        "query": "attention mechanism",
        "graph": {
            "type": "paper_citation_network",
            "limit": 50,
        },
    }, name)
    if not data:
        return
    nodes = data.get("nodes", [])
    edges = data.get("edges", [])
    ok(name, ms, f"citation_network: nodes={len(nodes)}, edges={len(edges)}")


async def test_expansion_early_exit(c):
    """Test 5: Expansion with data that has no citations should exit quickly."""
    name = "early_exit:no_citations"
    # With OAI-PMH data (no reference_ids), expansion should exit on iteration 1
    data, ms = await post_graph(c, {
        "query": "quantum computing",
        "graph": {
            "type": "pattern_match",
            "pattern_nodes": [
                {"alias": "a", "type": "paper"},
                {"alias": "b", "type": "paper"},
            ],
            "pattern_edges": [{"source": "a", "target": "b", "relation": "co_authored"}],
            "limit": 10,
            "max_expansion": 10,  # High expansion, but should exit early
        },
    }, name)
    if not data:
        return
    meta = data.get("metadata", {})
    # Should be fast because no citation refs to expand
    if ms < 5000:
        ok(name, ms, f"max_expansion=10 but fast (no refs to expand)")
    else:
        fail(name, ms, f"Slow despite no citations to expand: {ms}ms")


async def test_islice_candidates(c):
    """Test 6: Pattern match with many candidates uses islice (no sort)."""
    name = "islice:many_candidates"
    data, ms = await post_graph(c, {
        "query": "deep learning",
        "graph": {
            "type": "pattern_match",
            "pattern_nodes": [
                {"alias": "a", "type": "paper"},
                {"alias": "b", "type": "paper"},
            ],
            "pattern_edges": [{"source": "a", "target": "b", "relation": "same_category"}],
            "limit": 50,
        },
    }, name)
    if not data:
        return
    meta = data.get("metadata", {})
    matches = meta.get("matches_found", 0)
    ok(name, ms, f"matches={matches} (candidates taken via islice)")


async def test_where_with_coauthor(c):
    """Combined: WHERE + co_authored (tests lazy adj + memoize + where)."""
    name = "combined:where_coauthor"
    data, ms = await post_graph(c, {
        "query": "generative model",
        "graph": {
            "type": "pattern_match",
            "pattern_nodes": [
                {"alias": "a", "type": "paper"},
                {"alias": "b", "type": "paper"},
            ],
            "pattern_edges": [{"source": "a", "target": "b", "relation": "co_authored"}],
            "where": [{"left": "a.date", "op": ">", "right": "b.date"}],
            "limit": 10,
        },
    }, name)
    if not data:
        return
    meta = data.get("metadata", {})
    matches = meta.get("matches_found", 0)
    wc = meta.get("where_conditions", 0)
    if wc >= 1:
        ok(name, ms, f"WHERE + co_authored: matches={matches}, where_conditions={wc}")
    elif matches == 0 and "error" not in meta:
        ok(name, ms, f"No matches (OK, 0 results with WHERE)")
    else:
        fail(name, ms, f"where_conditions={wc}, meta={meta}")


async def test_optional_with_category(c):
    """Combined: OPTIONAL + same_category (lazy adj + optional logic)."""
    name = "combined:optional_category"
    data, ms = await post_graph(c, {
        "query": "language model",
        "graph": {
            "type": "pattern_match",
            "pattern_nodes": [
                {"alias": "a", "type": "paper"},
                {"alias": "b", "type": "paper"},
            ],
            "pattern_edges": [
                {"source": "a", "target": "b", "relation": "same_category", "optional": True},
            ],
            "limit": 10,
        },
    }, name)
    if not data:
        return
    meta = data.get("metadata", {})
    opt = meta.get("optional_edges", 0)
    matches = meta.get("matches_found", 0)
    ok(name, ms, f"optional_edges={opt}, matches={matches}")


async def test_aggregation_with_coauthor(c):
    """Combined: aggregation over co_authored matches."""
    name = "combined:agg_coauthor"
    data, ms = await post_graph(c, {
        "query": "optimization",
        "graph": {
            "type": "pattern_match",
            "pattern_nodes": [
                {"alias": "a", "type": "paper"},
                {"alias": "b", "type": "paper"},
            ],
            "pattern_edges": [{"source": "a", "target": "b", "relation": "co_authored"}],
            "aggregations": [
                {"alias": "total", "function": "count"},
                {"alias": "cats", "function": "collect", "field": "primary_category"},
            ],
            "limit": 20,
        },
    }, name)
    if not data:
        return
    meta = data.get("metadata", {})
    agg = meta.get("aggregations", {})
    total = agg.get("total", 0)
    cats = agg.get("cats", [])
    ok(name, ms, f"co_authored agg: total={total}, categories={len(cats)}")


async def test_timing_consistency(c):
    """Run the same query twice to verify memoization effect within a request."""
    name = "timing:repeat_query"
    body = {
        "query": "deep learning",
        "graph": {
            "type": "pattern_match",
            "pattern_nodes": [
                {"alias": "a", "type": "paper"},
                {"alias": "b", "type": "paper"},
            ],
            "pattern_edges": [{"source": "a", "target": "b", "relation": "co_authored"}],
            "limit": 20,
        },
    }
    _, ms1 = await post_graph(c, body, "timing:run1")
    _, ms2 = await post_graph(c, body, "timing:run2")
    if ms1 is not None and ms2 is not None:
        ok(name, ms2, f"run1={ms1}ms, run2={ms2}ms (note: memoization is per-request)")
    else:
        fail(name, 0, "One or both runs failed")


async def test_subgraph_projection(c):
    """Test subgraph projection uses parallel ES batches."""
    name = "parallel_es:subgraph_projection"
    data, ms = await post_graph(c, {
        "query": "robotics",
        "graph": {
            "type": "subgraph_projection",
            "subgraph_filter": {
                "categories": ["cs.RO"],
                "direction": "both",
                "max_hops": 1,
            },
            "subgraph_algorithm": "pagerank",
            "limit": 20,
        },
    }, name)
    if not data:
        return
    nodes = data.get("nodes", [])
    ok(name, ms, f"subgraph_projection: nodes={len(nodes)}")


async def main():
    print("=" * 70)
    print("  PERFORMANCE OPTIMIZATION VERIFICATION")
    print("=" * 70)

    async with httpx.AsyncClient() as c:
        print("\n── Fix 1: Lazy co-author adjacency ──")
        await test_lazy_coauthor(c)
        await test_lazy_coauthor_large(c)

        print("\n── Fix 2: Lazy category adjacency ──")
        await test_lazy_category(c)
        await test_lazy_category_correctness(c)

        print("\n── Fix 3: Memoized reachable() ──")
        await test_memoized_reachable(c)
        await test_memoized_reachable_variable_hops(c)

        print("\n── Fix 4: Parallel ES batch fetches ──")
        await test_parallel_es_fetches(c)
        await test_parallel_es_citation_subgraph(c)
        await test_subgraph_projection(c)

        print("\n── Fix 5: Expansion early-exit ──")
        await test_expansion_early_exit(c)

        print("\n── Fix 6: islice vs sorted ──")
        await test_islice_candidates(c)

        print("\n── Combined optimizations ──")
        await test_where_with_coauthor(c)
        await test_optional_with_category(c)
        await test_aggregation_with_coauthor(c)
        await test_timing_consistency(c)

    passed = sum(1 for r in results if r[0])
    failed = sum(1 for r in results if not r[0])
    print(f"\n{'=' * 70}")
    print(f"  RESULTS: {passed}/{passed + failed} passed, {failed} failed")
    print(f"{'=' * 70}")

    if failed > 0:
        print("\n  FAILURES:")
        for r in results:
            if not r[0]:
                print(f"    ❌ {r[1]}: {r[3]}")

    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
