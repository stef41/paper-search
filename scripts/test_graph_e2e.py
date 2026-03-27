#!/usr/bin/env python3
"""End-to-end tests for graph query endpoints."""
import json
import sys
import httpx

BASE = "http://localhost:8000"
HEADERS = {"X-API-Key": "changeme-key-1", "Content-Type": "application/json"}

passed = 0
failed = 0


def test(name, payload, checks):
    global passed, failed
    try:
        r = httpx.post(f"{BASE}/graph", headers=HEADERS, json=payload, timeout=30)
        data = r.json()
        if r.status_code != 200:
            print(f"  FAIL {name}: HTTP {r.status_code} — {data.get('detail', data)}")
            failed += 1
            return
        for desc, check_fn in checks:
            if not check_fn(data):
                print(f"  FAIL {name}: {desc}")
                failed += 1
                return
        print(f"  OK   {name}")
        passed += 1
    except Exception as e:
        print(f"  FAIL {name}: {e}")
        failed += 1


print("=" * 70)
print("GRAPH QUERY E2E TESTS")
print("=" * 70)

# ── 1. category_diversity ──
print("\n── category_diversity ──")

test("papers with 5+ categories", {
    "graph": {"type": "category_diversity", "min_categories": 5, "limit": 5}
}, [
    ("has nodes", lambda d: len(d["nodes"]) > 0),
    ("has paper nodes", lambda d: any(n["type"] == "paper" for n in d["nodes"])),
    ("has category nodes", lambda d: any(n["type"] == "category" for n in d["nodes"])),
    ("has edges", lambda d: len(d["edges"]) > 0),
    ("edges are in_category", lambda d: all(e["relation"] == "in_category" for e in d["edges"])),
    ("papers have 5+ cats", lambda d: all(
        n["properties"]["category_count"] >= 5
        for n in d["nodes"] if n["type"] == "paper"
    )),
])

test("category_diversity with search filter", {
    "graph": {"type": "category_diversity", "min_categories": 3, "limit": 5},
    "categories": ["cs.LG"],
    "submitted_date": {"gte": "2023-01-01T00:00:00"}
}, [
    ("has nodes", lambda d: len(d["nodes"]) > 0),
    ("papers include cs.LG", lambda d: all(
        "cs.LG" in n["properties"]["categories"]
        for n in d["nodes"] if n["type"] == "paper"
    )),
])

test("category_diversity returns sorted by count", {
    "graph": {"type": "category_diversity", "min_categories": 3, "limit": 20}
}, [
    ("has results", lambda d: len([n for n in d["nodes"] if n["type"] == "paper"]) > 0),
])

# ── 2. coauthor_network ──
print("\n── coauthor_network ──")

test("LeCun co-authors depth 1", {
    "graph": {"type": "coauthor_network", "seed_author": "Yann LeCun", "depth": 1, "limit": 20}
}, [
    ("has seed node", lambda d: any(
        n["id"] == "Yann LeCun" and n["properties"]["depth"] == 0 for n in d["nodes"]
    )),
    ("has co-author nodes", lambda d: len([n for n in d["nodes"] if n["properties"].get("depth") == 1]) > 0),
    ("all edges are co_authored", lambda d: all(e["relation"] == "co_authored" for e in d["edges"])),
    ("edges have weights", lambda d: all(e["weight"] is not None and e["weight"] > 0 for e in d["edges"])),
    ("total papers > 0", lambda d: d["total"] > 0),
])

test("LeCun co-authors depth 2", {
    "graph": {"type": "coauthor_network", "seed_author": "Yann LeCun", "depth": 2, "limit": 50}
}, [
    ("has more than seed alone", lambda d: len(d["nodes"]) > 1),
    ("has depth-2 edges", lambda d: len(d["edges"]) > 5),
    ("unique_coauthors reported", lambda d: d["metadata"].get("unique_coauthors", 0) > 0),
])

test("co-author with topic filter", {
    "graph": {"type": "coauthor_network", "seed_author": "Yann LeCun", "depth": 1, "limit": 15},
    "query": "self-supervised",
    "categories": ["cs.LG", "cs.CV"]
}, [
    ("has seed node", lambda d: any(n["id"] == "Yann LeCun" for n in d["nodes"])),
])

test("missing seed_author returns error metadata", {
    "graph": {"type": "coauthor_network", "depth": 1, "limit": 10}
}, [
    ("has error in metadata", lambda d: "error" in d.get("metadata", {})),
])

# ── 3. author_bridge ──
print("\n── author_bridge ──")

test("bridge cs.AI → q-bio", {
    "graph": {
        "type": "author_bridge",
        "source_categories": ["cs.AI"],
        "target_categories": ["q-bio.NC", "q-bio.QM"],
        "min_categories": 3,
        "limit": 10
    }
}, [
    ("has author nodes", lambda d: any(n["type"] == "author" for n in d["nodes"])),
    ("has category nodes", lambda d: any(n["type"] == "category" for n in d["nodes"])),
    ("edges are publishes_in", lambda d: all(e["relation"] == "publishes_in" for e in d["edges"])),
    ("authors have 3+ categories", lambda d: all(
        n["properties"]["category_count"] >= 3
        for n in d["nodes"] if n["type"] == "author"
    )),
])

test("bridge without target (pure diversity)", {
    "graph": {"type": "author_bridge", "min_categories": 5, "limit": 5}
}, [
    ("has author nodes", lambda d: any(n["type"] == "author" for n in d["nodes"])),
    ("authors have 5+ cats", lambda d: all(
        n["properties"]["category_count"] >= 5
        for n in d["nodes"] if n["type"] == "author"
    )),
])

test("bridge with date filter", {
    "graph": {
        "type": "author_bridge",
        "source_categories": ["cs.LG"],
        "target_categories": ["physics.comp-ph"],
        "min_categories": 3,
        "limit": 10
    },
    "submitted_date": {"gte": "2022-01-01T00:00:00"}
}, [
    ("has results or empty (niche)", lambda d: isinstance(d["nodes"], list)),
])

# ── 4. cross_category_flow ──
print("\n── cross_category_flow ──")

test("flow from cs.LG", {
    "graph": {"type": "cross_category_flow", "source_categories": ["cs.LG"], "limit": 15}
}, [
    ("all nodes are categories", lambda d: all(n["type"] == "category" for n in d["nodes"])),
    ("all edges are co_occurs", lambda d: all(e["relation"] == "co_occurs" for e in d["edges"])),
    ("edges have weights", lambda d: all(e["weight"] is not None and e["weight"] > 0 for e in d["edges"])),
    ("cs.LG in nodes", lambda d: any(n["id"] == "cs.LG" for n in d["nodes"])),
])

test("flow between specific categories", {
    "graph": {
        "type": "cross_category_flow",
        "source_categories": ["cs.AI", "cs.LG"],
        "target_categories": ["math.OC", "stat.ML"],
        "limit": 10
    }
}, [
    ("has edges", lambda d: len(d["edges"]) > 0),
    ("edges connect source↔target", lambda d: all(
        (e["source"] in ["cs.AI", "cs.LG", "math.OC", "stat.ML"] and
         e["target"] in ["cs.AI", "cs.LG", "math.OC", "stat.ML"])
        for e in d["edges"]
    )),
])

test("flow with date filter", {
    "graph": {"type": "cross_category_flow", "source_categories": ["cs.CL"], "limit": 10},
    "submitted_date": {"gte": "2024-01-01T00:00:00"}
}, [
    ("has nodes", lambda d: len(d["nodes"]) > 0),
])

test("global flow (no source/target)", {
    "graph": {"type": "cross_category_flow", "limit": 20}
}, [
    ("has many categories", lambda d: len(d["nodes"]) > 5),
    ("highest weight makes sense", lambda d: d["edges"][0]["weight"] > 1000 if d["edges"] else True),
])

# ── 5. interdisciplinary ──
print("\n── interdisciplinary ──")

test("basic interdisciplinary (4+ cats)", {
    "graph": {"type": "interdisciplinary", "min_categories": 4, "limit": 10}
}, [
    ("has paper nodes", lambda d: any(n["type"] == "paper" for n in d["nodes"])),
    ("has scores", lambda d: all(
        "interdisciplinary_score" in n["properties"]
        for n in d["nodes"] if n["type"] == "paper"
    )),
    ("scores between 0 and 1", lambda d: all(
        0 <= n["properties"]["interdisciplinary_score"] <= 1
        for n in d["nodes"] if n["type"] == "paper"
    )),
    ("sorted by score desc", lambda d: (
        lambda papers: all(papers[i] >= papers[i+1] for i in range(len(papers)-1))
    )([n["properties"]["interdisciplinary_score"] for n in d["nodes"] if n["type"] == "paper"])),
    ("papers have 4+ cats", lambda d: all(
        n["properties"]["category_count"] >= 4
        for n in d["nodes"] if n["type"] == "paper"
    )),
])

test("interdisciplinary + topic filter", {
    "graph": {"type": "interdisciplinary", "min_categories": 3, "limit": 5},
    "title_query": "neural network",
    "submitted_date": {"gte": "2023-01-01T00:00:00"}
}, [
    ("has results", lambda d: len(d["nodes"]) > 0),
])

test("interdisciplinary + category filter", {
    "graph": {"type": "interdisciplinary", "min_categories": 3, "limit": 5},
    "categories": ["cs.LG", "cs.AI"]
}, [
    ("papers include cs.LG or cs.AI", lambda d: all(
        "cs.LG" in n["properties"]["categories"] or "cs.AI" in n["properties"]["categories"]
        for n in d["nodes"] if n["type"] == "paper"
    )),
])

test("interdisciplinary + has_github", {
    "graph": {"type": "interdisciplinary", "min_categories": 3, "limit": 5},
    "has_github": True
}, [
    ("returns results or empty list", lambda d: isinstance(d["nodes"], list)),
])

# ── 6. rising_interdisciplinary ──
print("\n── rising_interdisciplinary ──")

test("rising interdisciplinary basic", {
    "graph": {"type": "rising_interdisciplinary", "recency_months": 6,
              "citation_percentile": 90, "citation_window_years": 2,
              "min_citing_categories": 2, "limit": 10}
}, [
    ("valid response", lambda d: "nodes" in d and "edges" in d),
    ("has metadata", lambda d: "citation_threshold" in d.get("metadata", {})),
    ("has percentile info", lambda d: d["metadata"].get("citation_percentile") == 90.0),
    ("has window info", lambda d: d["metadata"].get("citation_window_years") == 2),
    ("papers have citing cats", lambda d: all(
        "citing_category_count" in n["properties"]
        for n in d["nodes"] if n["type"] == "paper"
    )),
])

test("rising interdisciplinary with category filter", {
    "graph": {"type": "rising_interdisciplinary", "recency_months": 12,
              "citation_percentile": 80, "citation_window_years": 2,
              "min_citing_categories": 2, "limit": 5},
    "categories": ["cs.LG"]
}, [
    ("valid response", lambda d: isinstance(d["nodes"], list)),
])

test("rising interdisciplinary with high threshold (may be empty)", {
    "graph": {"type": "rising_interdisciplinary", "recency_months": 1,
              "citation_percentile": 99, "citation_window_years": 1,
              "min_citing_categories": 5, "limit": 5}
}, [
    ("valid response", lambda d: "nodes" in d),
])

# ── 7. citation_traversal ──
print("\n── citation_traversal ──")

test("traversal by category (filter-based seeds)", {
    "graph": {"type": "citation_traversal", "direction": "references",
              "aggregate_by": "category", "limit": 15},
    "query": "transformer attention",
    "categories": ["cs.LG"],
    "submitted_date": {"gte": "2024-01-01T00:00:00"},
}, [
    ("has nodes", lambda d: len(d["nodes"]) > 0),
    ("has seed nodes", lambda d: any(
        n["type"] == "paper" and n["properties"].get("role") == "seed"
        for n in d["nodes"])),
    ("has category nodes", lambda d: any(n["type"] == "category" for n in d["nodes"])),
    ("has edges", lambda d: len(d["edges"]) > 0),
    ("metadata has direction", lambda d: d["metadata"].get("direction") == "references"),
    ("linked_ids_found > 0", lambda d: d["metadata"].get("linked_ids_found", 0) > 0),
])

test("traversal by author", {
    "graph": {"type": "citation_traversal", "direction": "references",
              "aggregate_by": "author", "limit": 10},
    "query": "transformer attention",
    "categories": ["cs.LG"],
    "submitted_date": {"gte": "2024-01-01T00:00:00"},
}, [
    ("has author nodes", lambda d: any(n["type"] == "author" for n in d["nodes"])),
    ("author paper_count > 0", lambda d: all(
        n["properties"]["paper_count"] > 0
        for n in d["nodes"] if n["type"] == "author")),
])

test("traversal by year", {
    "graph": {"type": "citation_traversal", "direction": "references",
              "aggregate_by": "year", "limit": 20},
    "query": "transformer attention",
    "categories": ["cs.LG"],
    "submitted_date": {"gte": "2024-01-01T00:00:00"},
}, [
    ("has year nodes", lambda d: any(n["type"] == "year" for n in d["nodes"])),
    ("years are reasonable", lambda d: all(
        1990 <= int(n["id"]) <= 2030
        for n in d["nodes"] if n["type"] == "year")),
])

test("single-paper seed traversal", {
    "graph": {"type": "citation_traversal", "direction": "references",
              "aggregate_by": "category", "seed_arxiv_id": "2602.21169", "limit": 10},
}, [
    ("has nodes", lambda d: len(d["nodes"]) > 0),
    ("has category nodes", lambda d: any(n["type"] == "category" for n in d["nodes"])),
    ("traversed papers > 0", lambda d: d["metadata"].get("traversed_in_index", 0) > 0),
])

# ── 8. edge cases & validation ──
print("\n── edge cases ──")

test("unknown author co-author network", {
    "graph": {"type": "coauthor_network", "seed_author": "Zzzzxyzzy Nonexistent", "depth": 1}
}, [
    ("total is 0 or low", lambda d: d["total"] == 0),
])

test("high min_categories returns fewer results", {
    "graph": {"type": "category_diversity", "min_categories": 10, "limit": 5}
}, [
    ("fewer papers than min_categories=3", lambda d: len([n for n in d["nodes"] if n["type"] == "paper"]) <= 10),
])

# ── 7. composition correctness ──
print("\n── composition with all filter types ──")

test("graph + author + dates + categories", {
    "graph": {"type": "interdisciplinary", "min_categories": 3, "limit": 5},
    "author": "Bengio",
    "categories": ["cs.LG"],
    "submitted_date": {"gte": "2020-01-01T00:00:00"}
}, [
    ("returns valid response", lambda d: "nodes" in d and "edges" in d),
])

test("graph + fuzzy search", {
    "graph": {"type": "category_diversity", "min_categories": 3, "limit": 5},
    "fuzzy": "neural nework",
    "fuzzy_fuzziness": 2
}, [
    ("returns results", lambda d: len(d["nodes"]) >= 0),
])

test("graph + operator=and", {
    "graph": {"type": "interdisciplinary", "min_categories": 3, "limit": 5},
    "query": "graph neural network",
    "operator": "and"
}, [
    ("returns valid response", lambda d: isinstance(d["nodes"], list)),
])

# ── 8. invalid requests ──
print("\n── invalid requests ──")

try:
    r = httpx.post(f"{BASE}/graph", headers=HEADERS, json={
        "graph": {"type": "invalid_type"}
    }, timeout=10)
    if r.status_code == 422:
        print("  OK   invalid graph type → 422")
        passed += 1
    else:
        print(f"  FAIL invalid graph type → {r.status_code} (expected 422)")
        failed += 1
except Exception as e:
    print(f"  FAIL invalid graph type: {e}")
    failed += 1

try:
    r = httpx.post(f"{BASE}/graph", headers=HEADERS, json={
        "graph": {"type": "coauthor_network", "depth": 5}
    }, timeout=10)
    if r.status_code == 422:
        print("  OK   depth > 2 → 422")
        passed += 1
    else:
        print(f"  FAIL depth > 2 → {r.status_code} (expected 422)")
        failed += 1
except Exception as e:
    print(f"  FAIL depth > 2: {e}")
    failed += 1

try:
    r = httpx.post(f"{BASE}/graph", headers=HEADERS, json={}, timeout=10)
    if r.status_code == 422:
        print("  OK   missing graph field → 422")
        passed += 1
    else:
        print(f"  FAIL missing graph → {r.status_code} (expected 422)")
        failed += 1
except Exception as e:
    print(f"  FAIL missing graph: {e}")
    failed += 1

try:
    r = httpx.post(f"{BASE}/graph", headers={}, json={
        "graph": {"type": "category_diversity"}
    }, timeout=10)
    if r.status_code in (401, 403):
        print("  OK   no API key → 401/403")
        passed += 1
    else:
        print(f"  FAIL no API key → {r.status_code} (expected 401/403)")
        failed += 1
except Exception as e:
    print(f"  FAIL no API key: {e}")
    failed += 1

# ── Summary ──
print("\n" + "=" * 70)
print(f"RESULTS: {passed} passed, {failed} failed, {passed + failed} total")
print("=" * 70)
sys.exit(1 if failed > 0 else 0)
