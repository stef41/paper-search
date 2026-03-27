"""Verify ALL 52 graph query types return valid responses (no 500s, proper structure)."""
import asyncio
import json
import time
import httpx

BASE = "http://localhost:8000"
HEADERS = {"X-API-Key": "changeme-key-1", "Content-Type": "application/json"}
TIMEOUT = 60.0

results: list[tuple[bool, str, int, str]] = []

def ok(name, ms, detail):
    results.append((True, name, ms, detail))

def fail(name, ms, detail):
    results.append((False, name, ms, detail))


async def test_type(c: httpx.AsyncClient, name: str, body: dict):
    start = time.monotonic()
    try:
        resp = await c.post(f"{BASE}/graph", json=body, headers=HEADERS, timeout=TIMEOUT)
        ms = int((time.monotonic() - start) * 1000)
    except Exception as e:
        ms = int((time.monotonic() - start) * 1000)
        fail(name, ms, f"EXCEPTION: {e}")
        return

    if resp.status_code == 500:
        fail(name, ms, f"HTTP 500: {resp.text[:200]}")
        return
    if resp.status_code == 422:
        fail(name, ms, f"HTTP 422 (validation): {resp.text[:200]}")
        return
    if resp.status_code != 200:
        fail(name, ms, f"HTTP {resp.status_code}: {resp.text[:150]}")
        return

    data = resp.json()
    # Check response structure
    for key in ("nodes", "edges", "total", "took_ms"):
        if key not in data:
            fail(name, ms, f"Missing response key: {key}")
            return

    nodes = data["nodes"]
    edges = data["edges"]

    # Check node structure
    for n in nodes[:3]:
        for k in ("id", "label", "type"):
            if k not in n:
                fail(name, ms, f"Node missing '{k}': {list(n.keys())}")
                return

    # Check edge structure
    for e in edges[:3]:
        for k in ("source", "target", "relation"):
            if k not in e:
                fail(name, ms, f"Edge missing '{k}': {list(e.keys())}")
                return

    meta = data.get("metadata", {})
    err = meta.get("error", "")
    detail = f"n={len(nodes)}, e={len(edges)}"
    if err:
        detail += f", meta_err={err[:80]}"

    ok(name, ms, detail)


async def main():
    print("=" * 78)
    print("  ALL 52 GRAPH QUERY TYPES — VERIFICATION")
    print("=" * 78)

    # First get a sample arxiv_id from the index for seed-based queries
    async with httpx.AsyncClient() as c:
        resp = await c.post(f"{BASE}/search", json={"query": "machine learning", "limit": 5},
                            headers=HEADERS, timeout=30)
        papers = resp.json().get("hits", []) or resp.json().get("results", [])
        sample_ids = [p["arxiv_id"] for p in papers[:5]] if papers else []
        sid1 = sample_ids[0] if len(sample_ids) > 0 else "2501.00001"
        sid2 = sample_ids[1] if len(sample_ids) > 1 else "2501.00002"

    queries = {
        # ── Types 1-16: Foundation ──
        "01_category_diversity": {
            "query": "deep learning",
            "graph": {"type": "category_diversity", "limit": 20},
        },
        "02_coauthor_network": {
            "query": "deep learning",
            "graph": {"type": "coauthor_network", "seed_author": "Yann LeCun", "limit": 20},
        },
        "03_author_bridge": {
            "query": "deep learning",
            "graph": {"type": "author_bridge", "limit": 20},
        },
        "04_cross_category_flow": {
            "query": "deep learning",
            "graph": {"type": "cross_category_flow", "limit": 20},
        },
        "05_interdisciplinary": {
            "query": "deep learning",
            "graph": {"type": "interdisciplinary", "limit": 20},
        },
        "06_rising_interdisciplinary": {
            "query": "deep learning",
            "graph": {"type": "rising_interdisciplinary", "limit": 20},
        },
        "07_citation_traversal": {
            "query": "transformer",
            "graph": {"type": "citation_traversal", "limit": 20},
        },
        "08_paper_citation_network": {
            "query": "attention mechanism",
            "graph": {"type": "paper_citation_network", "limit": 20},
        },
        "09_author_influence": {
            "query": "neural network",
            "graph": {"type": "author_influence", "limit": 20},
        },
        "10_temporal_evolution": {
            "query": "large language model",
            "graph": {"type": "temporal_evolution", "limit": 20},
        },
        "11_paper_similarity": {
            "query": "reinforcement learning",
            "graph": {"type": "paper_similarity", "limit": 20},
        },
        "12_domain_collaboration": {
            "query": "computer vision",
            "graph": {"type": "domain_collaboration", "limit": 20},
        },
        "13_author_topic_evolution": {
            "query": "deep learning",
            "graph": {"type": "author_topic_evolution", "seed_author": "Yann LeCun", "limit": 20},
        },
        "14_github_landscape": {
            "query": "deep learning",
            "graph": {"type": "github_landscape", "limit": 20},
        },
        "15_bibliographic_coupling": {
            "query": "deep learning",
            "graph": {"type": "bibliographic_coupling", "limit": 20},
        },
        "16_cocitation": {
            "query": "deep learning",
            "graph": {"type": "cocitation", "limit": 20},
        },

        # ── Types 17-28: Graph algorithms batch 1 ──
        "17_multihop_citation": {
            "query": "transformer",
            "graph": {"type": "multihop_citation", "seed_arxiv_id": sid1, "max_hops": 2, "limit": 20},
        },
        "18_shortest_citation_path": {
            "query": "deep learning",
            "graph": {"type": "shortest_citation_path", "seed_arxiv_id": sid1,
                       "target_arxiv_id": sid2, "max_hops": 3, "limit": 20},
        },
        "19_pagerank": {
            "query": "machine learning",
            "graph": {"type": "pagerank", "limit": 20},
        },
        "20_community_detection": {
            "query": "neural network",
            "graph": {"type": "community_detection", "limit": 20},
        },
        "21_citation_patterns": {
            "query": "deep learning",
            "graph": {"type": "citation_patterns", "limit": 20},
        },
        "22_connected_components": {
            "query": "transformer",
            "graph": {"type": "connected_components", "limit": 20},
        },
        "23_weighted_shortest_path": {
            "query": "deep learning",
            "graph": {"type": "weighted_shortest_path", "seed_arxiv_id": sid1,
                       "target_arxiv_id": sid2, "max_hops": 3, "limit": 20},
        },
        "24_betweenness_centrality": {
            "query": "machine learning",
            "graph": {"type": "betweenness_centrality", "limit": 20},
        },
        "25_closeness_centrality": {
            "query": "deep learning",
            "graph": {"type": "closeness_centrality", "limit": 20},
        },
        "26_strongly_connected": {
            "query": "neural network",
            "graph": {"type": "strongly_connected_components", "limit": 20},
        },
        "27_topological_sort": {
            "query": "deep learning",
            "graph": {"type": "topological_sort", "limit": 20},
        },
        "28_link_prediction": {
            "query": "machine learning",
            "graph": {"type": "link_prediction", "limit": 20},
        },

        # ── Types 29-34: Graph algorithms batch 2 ──
        "29_louvain_community": {
            "query": "deep learning",
            "graph": {"type": "louvain_community", "limit": 20},
        },
        "30_degree_centrality": {
            "query": "neural network",
            "graph": {"type": "degree_centrality", "limit": 20},
        },
        "31_eigenvector_centrality": {
            "query": "machine learning",
            "graph": {"type": "eigenvector_centrality", "limit": 20},
        },
        "32_kcore_decomposition": {
            "query": "deep learning",
            "graph": {"type": "kcore_decomposition", "limit": 20},
        },
        "33_articulation_points": {
            "query": "transformer",
            "graph": {"type": "articulation_points", "limit": 20},
        },
        "34_influence_maximization": {
            "query": "deep learning",
            "graph": {"type": "influence_maximization", "influence_seeds": 3, "limit": 20},
        },

        # ── Types 35-49: Graph algorithms batch 3 ──
        "35_hits": {
            "query": "machine learning",
            "graph": {"type": "hits", "limit": 20},
        },
        "36_harmonic_centrality": {
            "query": "deep learning",
            "graph": {"type": "harmonic_centrality", "limit": 20},
        },
        "37_katz_centrality": {
            "query": "neural network",
            "graph": {"type": "katz_centrality", "limit": 20},
        },
        "38_all_shortest_paths": {
            "query": "deep learning",
            "graph": {"type": "all_shortest_paths", "seed_arxiv_id": sid1,
                       "target_arxiv_id": sid2, "max_hops": 3, "limit": 20},
        },
        "39_k_shortest_paths": {
            "query": "deep learning",
            "graph": {"type": "k_shortest_paths", "seed_arxiv_id": sid1,
                       "target_arxiv_id": sid2, "k_paths": 3, "max_hops": 3, "limit": 20},
        },
        "40_random_walk": {
            "query": "deep learning",
            "graph": {"type": "random_walk", "seed_arxiv_id": sid1,
                       "walk_length": 5, "num_walks": 3, "limit": 20},
        },
        "41_triangle_count": {
            "query": "machine learning",
            "graph": {"type": "triangle_count", "limit": 20},
        },
        "42_graph_diameter": {
            "query": "deep learning",
            "graph": {"type": "graph_diameter", "limit": 20},
        },
        "43_leiden_community": {
            "query": "neural network",
            "graph": {"type": "leiden_community", "limit": 20},
        },
        "44_bridge_edges": {
            "query": "deep learning",
            "graph": {"type": "bridge_edges", "limit": 20},
        },
        "45_min_cut": {
            "query": "deep learning",
            "graph": {"type": "min_cut", "seed_arxiv_id": sid1,
                       "target_arxiv_id": sid2, "limit": 20},
        },
        "46_minimum_spanning_tree": {
            "query": "machine learning",
            "graph": {"type": "minimum_spanning_tree", "limit": 20},
        },
        "47_node_similarity": {
            "query": "deep learning",
            "graph": {"type": "node_similarity", "limit": 20},
        },
        "48_bipartite_projection": {
            "query": "neural network",
            "graph": {"type": "bipartite_projection", "limit": 20},
        },
        "49_adamic_adar_index": {
            "query": "machine learning",
            "graph": {"type": "adamic_adar_index", "limit": 20},
        },

        # ── Types 50-52: Expressiveness ──
        "50_pattern_match": {
            "query": "deep learning",
            "graph": {
                "type": "pattern_match",
                "pattern_nodes": [
                    {"alias": "a", "type": "paper"},
                    {"alias": "b", "type": "paper"},
                ],
                "pattern_edges": [
                    {"source": "a", "target": "b", "relation": "co_authored"},
                ],
                "limit": 20,
            },
        },
        "51_pipeline": {
            "query": "deep learning",
            "graph": {
                "type": "pipeline",
                "pipeline_steps": [
                    {"type": "category_diversity", "limit": 30},
                    {"type": "pattern_match",
                     "pattern_nodes": [
                         {"alias": "a", "type": "paper"},
                         {"alias": "b", "type": "paper"},
                     ],
                     "pattern_edges": [
                         {"source": "a", "target": "b", "relation": "co_authored"},
                     ],
                     "limit": 20,
                    },
                ],
                "limit": 20,
            },
        },
        "52_subgraph_projection": {
            "query": "deep learning",
            "graph": {
                "type": "subgraph_projection",
                "subgraph_filter": {
                    "categories": ["cs.LG"],
                    "direction": "both",
                    "max_hops": 1,
                },
                "subgraph_algorithm": "pagerank",
                "limit": 20,
            },
        },
    }

    async with httpx.AsyncClient() as c:
        print(f"\nUsing sample IDs: {sid1}, {sid2}")
        print(f"Testing {len(queries)} query types...\n")

        for name, body in queries.items():
            await test_type(c, name, body)
            # Print as we go
            r = results[-1]
            icon = "✅" if r[0] else "❌"
            print(f"  {icon}  {r[1]:<42} {r[2]:>5}ms  {r[3]}")

    passed = sum(1 for r in results if r[0])
    failed = sum(1 for r in results if not r[0])
    total = len(results)
    print(f"\n{'=' * 78}")
    print(f"  RESULTS: {passed}/{total} passed, {failed} failed")
    print(f"{'=' * 78}")

    if failed > 0:
        print("\n  FAILURES:")
        for r in results:
            if not r[0]:
                print(f"    ❌ {r[1]}: {r[3]}")

    return failed == 0

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
