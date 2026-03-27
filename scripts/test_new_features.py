#!/usr/bin/env python3
"""Comprehensive test suite for new features:
  1. Semantic search (/search + semantic)
  2. Graph + semantic (all 7 graph types)
  3. Local-only citation counts (no external)
  4. Embeddings populated correctly
"""
import json
import sys
import time
import httpx

API = "http://localhost:8000"
ES = "http://localhost:9200"
KEY = "changeme-key-1"
HEADERS = {"X-API-Key": KEY, "Content-Type": "application/json"}

passed = 0
failed = 0
errors = []


def test(name: str, result: bool, detail: str = ""):
    global passed, failed
    if result:
        passed += 1
        print(f"  ✓ {name}")
    else:
        failed += 1
        errors.append(f"{name}: {detail}")
        print(f"  ✗ {name} — {detail}")


def api(path: str, body: dict) -> dict:
    with httpx.Client(timeout=30) as c:
        r = c.post(f"{API}{path}", json=body, headers=HEADERS)
        return r.json()


def es(path: str, body: dict) -> dict:
    with httpx.Client(timeout=30) as c:
        r = c.post(f"{ES}{path}", json=body, headers={"Content-Type": "application/json"})
        return r.json()


# ════════════════════════════════════════════════
# 1. SEMANTIC SEARCH
# ════════════════════════════════════════════════
print("\n" + "=" * 60)
print("1. SEMANTIC SEARCH")
print("=" * 60)

# 1a. Basic semantic search returns results
d = api("/search", {
    "query": "machine learning",
    "limit": 5,
    "semantic": {"text": "deep learning for image classification", "level": "abstract", "weight": 0.5}
})
test("Semantic search returns hits", len(d.get("hits", [])) > 0, f"got {len(d.get('hits', []))}")

# 1b. Semantic with different topics gives different top results
d1 = api("/search", {
    "limit": 10,
    "semantic": {"text": "protein folding structure prediction", "level": "abstract", "weight": 1.0}
})
d2 = api("/search", {
    "limit": 10,
    "semantic": {"text": "gradient descent convergence rates", "level": "abstract", "weight": 1.0}
})
ids1 = {h["arxiv_id"] for h in d1.get("hits", [])}
ids2 = {h["arxiv_id"] for h in d2.get("hits", [])}
test("Different semantic texts give different results", ids1 != ids2,
     f"both returned same {len(ids1)} papers")

# 1c. Semantic with title level works
d = api("/search", {
    "query": "transformers",
    "limit": 5,
    "semantic": {"text": "attention mechanism", "level": "title", "weight": 0.6}
})
test("Title-level semantic works", len(d.get("hits", [])) > 0)

# 1d. Semantic-only (no text query) works
d = api("/search", {
    "limit": 5,
    "semantic": {"text": "quantum error correction", "level": "abstract", "weight": 1.0}
})
test("Semantic-only search (no query text)", d.get("total", 0) > 0, f"total={d.get('total', 0)}")

# 1e. Semantic with filters
d = api("/search", {
    "limit": 5,
    "categories": ["cs.CV"],
    "semantic": {"text": "object detection in autonomous driving", "level": "abstract", "weight": 0.7}
})
hits = d.get("hits", [])
all_cv = all("cs.CV" in h.get("categories", []) for h in hits) if hits else False
test("Semantic + category filter respected", all_cv, f"non-cs.CV paper found" if not all_cv else "")

# 1f. Semantic with date filter
d = api("/search", {
    "limit": 5,
    "submitted_date": {"gte": "2024-01-01"},
    "semantic": {"text": "large language models", "level": "abstract", "weight": 0.6}
})
hits = d.get("hits", [])
recent = all(h.get("submitted_date", "2024-01-01") >= "2024-01-01" for h in hits) if hits else True
all_ok = recent and len(hits) > 0
test("Semantic + date filter respected", all_ok,
     f"hits={len(hits)}, dates={[h.get('submitted_date','?')[:10] for h in hits[:3]]}")


# ════════════════════════════════════════════════
# 2. GRAPH + SEMANTIC (all 7 graph types)
# ════════════════════════════════════════════════
print("\n" + "=" * 60)
print("2. GRAPH + SEMANTIC (all 7 types)")
print("=" * 60)

semantic = {"text": "reinforcement learning for robotics", "level": "abstract", "weight": 0.6}

# 2a. category_diversity + semantic
d = api("/graph", {
    "graph": {"type": "category_diversity", "min_categories": 3, "limit": 10},
    "semantic": semantic,
})
paper_nodes = [n for n in d.get("nodes", []) if n["type"] == "paper"]
test("category_diversity + semantic", len(paper_nodes) > 0, f"papers={len(paper_nodes)}")

# 2b. coauthor_network + semantic
d = api("/graph", {
    "graph": {"type": "coauthor_network", "seed_author": "Yann LeCun", "depth": 1, "limit": 10},
    "semantic": semantic,
})
test("coauthor_network + semantic",
     len(d.get("nodes", [])) > 0 and d.get("metadata", {}).get("seed_author") == "Yann LeCun",
     f"nodes={len(d.get('nodes', []))}")

# 2c. author_bridge + semantic
d = api("/graph", {
    "graph": {"type": "author_bridge", "min_categories": 3, "limit": 10},
    "semantic": semantic,
})
test("author_bridge + semantic", len(d.get("nodes", [])) > 0, f"nodes={len(d.get('nodes', []))}")

# 2d. cross_category_flow + semantic
d = api("/graph", {
    "graph": {"type": "cross_category_flow", "limit": 20},
    "semantic": semantic,
})
edges = d.get("edges", [])
test("cross_category_flow + semantic", len(edges) > 0, f"edges={len(edges)}")

# 2e. interdisciplinary + semantic
d = api("/graph", {
    "graph": {"type": "interdisciplinary", "min_categories": 3, "limit": 10},
    "semantic": semantic,
})
papers = [n for n in d.get("nodes", []) if n["type"] == "paper"]
test("interdisciplinary + semantic", len(papers) > 0, f"papers={len(papers)}")

# 2f. rising_interdisciplinary + semantic
d = api("/graph", {
    "graph": {"type": "rising_interdisciplinary", "limit": 10},
    "semantic": semantic,
})
# May return 0 if insufficient citation data — that's ok, just no error
test("rising_interdisciplinary + semantic (no error)",
     "nodes" in d and "metadata" in d,
     str(d.get("detail", d.get("metadata", {}).get("error", ""))))

# 2g. citation_traversal + semantic
d = api("/graph", {
    "graph": {"type": "citation_traversal", "direction": "cited_by", "aggregate_by": "category", "limit": 10},
    "semantic": semantic,
})
test("citation_traversal + semantic (no error)",
     "nodes" in d and "metadata" in d,
     str(d.get("detail", d.get("metadata", {}).get("error", ""))))


# 2h. Graph without semantic still works (regression check)
d = api("/graph", {
    "graph": {"type": "coauthor_network", "seed_author": "Yann LeCun", "depth": 1, "limit": 10},
})
test("Graph without semantic (regression)",
     len(d.get("nodes", [])) > 0,
     f"nodes={len(d.get('nodes', []))}")


# ════════════════════════════════════════════════
# 3. LOCAL-ONLY CITATION COUNTS
# ════════════════════════════════════════════════
print("\n" + "=" * 60)
print("3. LOCAL-ONLY CITATION COUNTS")
print("=" * 60)

# 3a. Papers with citations > 0 ALL have cited_by_ids
r = es("/arxiv_papers/_count", {
    "query": {"bool": {
        "must": [{"range": {"citation_stats.total_citations": {"gt": 0}}}],
        "must_not": [{"exists": {"field": "cited_by_ids"}}]
    }}
})
orphan_count = r.get("count", -1)
test("No citation counts without cited_by_ids (local only)", orphan_count == 0,
     f"found {orphan_count} papers with external counts")

# 3b. cited_by_ids count matches total_citations for a sample
r = es("/arxiv_papers/_search", {
    "query": {"exists": {"field": "cited_by_ids"}},
    "size": 20,
    "_source": ["arxiv_id", "cited_by_ids", "citation_stats.total_citations"]
})
mismatches = 0
for h in r["hits"]["hits"]:
    src = h["_source"]
    ids_len = len(src.get("cited_by_ids", []))
    count = src.get("citation_stats", {}).get("total_citations", 0)
    if ids_len != count:
        mismatches += 1
test("cited_by_ids length == total_citations (20 samples)", mismatches == 0,
     f"{mismatches}/20 mismatches")

# 3c. Papers with reference_ids exist
r = es("/arxiv_papers/_count", {"query": {"exists": {"field": "reference_ids"}}})
ref_count = r.get("count", 0)
test("Papers with reference_ids exist", ref_count > 0, f"count={ref_count}")

# 3d. Cited papers were created by inverting reference graph
# Pick a paper with cited_by_ids and verify a citer references it
r = es("/arxiv_papers/_search", {
    "query": {"exists": {"field": "cited_by_ids"}},
    "size": 1,
    "_source": ["arxiv_id", "cited_by_ids"]
})
if r["hits"]["hits"]:
    cited_paper = r["hits"]["hits"][0]["_source"]["arxiv_id"]
    citer_id = r["hits"]["hits"][0]["_source"]["cited_by_ids"][0]
    # Verify the citer has this paper in its reference_ids
    r2 = es("/arxiv_papers/_search", {
        "query": {"term": {"arxiv_id": citer_id}},
        "size": 1,
        "_source": ["reference_ids"]
    })
    if r2["hits"]["hits"]:
        refs = r2["hits"]["hits"][0]["_source"].get("reference_ids", [])
        test("Citation graph consistency (A cites B → B in A.reference_ids)",
             cited_paper in refs, f"{citer_id} doesn't reference {cited_paper}")
    else:
        test("Citation graph consistency", False, f"citer {citer_id} not found")
else:
    test("Citation graph consistency", False, "no papers with cited_by_ids")


# ════════════════════════════════════════════════
# 4. EMBEDDINGS
# ════════════════════════════════════════════════
print("\n" + "=" * 60)
print("4. EMBEDDINGS")
print("=" * 60)

# 4a. Embeddings are being stored
r = es("/arxiv_papers/_count", {"query": {"exists": {"field": "title_embedding"}}})
title_count = r.get("count", 0)
r2 = es("/arxiv_papers/_count", {"query": {"exists": {"field": "abstract_embedding"}}})
abstract_count = r2.get("count", 0)
test("Title embeddings being stored", title_count > 0, f"count={title_count}")
test("Abstract embeddings being stored", abstract_count > 0, f"count={abstract_count}")
test("Title and abstract counts match", title_count == abstract_count,
     f"title={title_count}, abstract={abstract_count}")

# 4b. Embedding dimension is correct (384)
r = es("/arxiv_papers/_search", {
    "query": {"exists": {"field": "title_embedding"}},
    "size": 1,
    "_source": ["title_embedding"]
})
if r["hits"]["hits"]:
    emb = r["hits"]["hits"][0]["_source"]["title_embedding"]
    test("Embedding dimension is 384", len(emb) == 384, f"got dim={len(emb)}")
else:
    test("Embedding dimension is 384", False, "no embeddings found")

# 4c. Embedding values are normalized (L2 norm ≈ 1.0)
if r["hits"]["hits"]:
    import math
    norm = math.sqrt(sum(x * x for x in emb))
    test("Embedding is normalized (L2 norm ≈ 1.0)", abs(norm - 1.0) < 0.01,
         f"norm={norm:.4f}")


# ════════════════════════════════════════════════
# 5. SEMANTIC RELEVANCE QUALITY
# ════════════════════════════════════════════════
print("\n" + "=" * 60)
print("5. SEMANTIC RELEVANCE QUALITY")
print("=" * 60)

# 5a. Semantic search for "GAN" returns generative adversarial papers
d = api("/search", {
    "limit": 5,
    "semantic": {"text": "generative adversarial networks image synthesis", "level": "abstract", "weight": 1.0}
})
hits = d.get("hits", [])
gan_related = sum(1 for h in hits if any(
    kw in (h.get("title", "") + " " + h.get("abstract", "")).lower()
    for kw in ["generative", "adversarial", "gan", "discriminator", "generator"]
))
test("Semantic 'GAN' returns relevant papers", gan_related >= 2,
     f"only {gan_related}/5 GAN-related")

# 5b. Semantic search for physics vs CS gives different categories
d_physics = api("/search", {
    "limit": 10,
    "semantic": {"text": "dark matter particle detection astrophysics", "level": "abstract", "weight": 1.0}
})
d_cs = api("/search", {
    "limit": 10,
    "semantic": {"text": "compiler optimization code generation", "level": "abstract", "weight": 1.0}
})
phys_cats = set()
for h in d_physics.get("hits", []):
    phys_cats.update(h.get("categories", []))
cs_cats = set()
for h in d_cs.get("hits", []):
    cs_cats.update(h.get("categories", []))
# These should have little overlap
overlap = phys_cats & cs_cats
test("Physics vs CS semantic give different categories",
     len(overlap) < max(len(phys_cats), len(cs_cats)) / 2,
     f"overlap={overlap}")

# 5c. Graph interdisciplinary + semantic narrows to topic
d_no_sem = api("/graph", {
    "graph": {"type": "interdisciplinary", "min_categories": 3, "limit": 10},
})
d_with_sem = api("/graph", {
    "graph": {"type": "interdisciplinary", "min_categories": 3, "limit": 10},
    "semantic": {"text": "natural language processing transformers", "level": "abstract", "weight": 0.7}
})
papers_no = {n["id"] for n in d_no_sem.get("nodes", []) if n["type"] == "paper"}
papers_with = {n["id"] for n in d_with_sem.get("nodes", []) if n["type"] == "paper"}
test("Semantic changes interdisciplinary results",
     papers_no != papers_with or len(papers_with) == 0,  # diff or graceful fallback
     "same results with and without semantic")


# ════════════════════════════════════════════════
# 6. EDGE CASES & ERROR HANDLING
# ════════════════════════════════════════════════
print("\n" + "=" * 60)
print("6. EDGE CASES & ERROR HANDLING")
print("=" * 60)

# 6a. Empty semantic text
d = api("/search", {"query": "test", "limit": 3, "semantic": {"text": "", "level": "abstract", "weight": 0.5}})
test("Empty semantic text doesn't crash", "hits" in d or "detail" in d)

# 6b. Very long semantic text
long_text = "quantum computing " * 200
d = api("/search", {"limit": 3, "semantic": {"text": long_text, "level": "abstract", "weight": 0.5}})
test("Very long semantic text handled", "hits" in d or "detail" in d)

# 6c. Semantic weight = 0
d = api("/search", {"query": "neural", "limit": 3, "semantic": {"text": "ignored text", "level": "abstract", "weight": 0.0}})
test("Semantic weight=0 still returns results", len(d.get("hits", [])) > 0)

# 6d. Graph seed_arxiv_id + semantic (citation_traversal)
d = api("/graph", {
    "graph": {"type": "citation_traversal", "seed_arxiv_id": "2301.00001", "direction": "references", "aggregate_by": "category", "limit": 10},
    "semantic": {"text": "something", "level": "abstract", "weight": 0.5}
})
test("citation_traversal with seed_arxiv_id + semantic",
     "nodes" in d and "metadata" in d)


# ════════════════════════════════════════════════
# SUMMARY
# ════════════════════════════════════════════════
print("\n" + "=" * 60)
total = passed + failed
print(f"RESULTS: {passed}/{total} passed, {failed} failed")
print("=" * 60)

if errors:
    print("\nFailed tests:")
    for e in errors:
        print(f"  ✗ {e}")

sys.exit(0 if failed == 0 else 1)
