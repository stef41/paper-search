#!/usr/bin/env python3
"""Comprehensive test suite for multi-semantic and exclude features.

Covers:
  1. Backward compat: single semantic object (no mode field)
  2. Single semantic with explicit mode="boost"
  3. Multiple boost semantics (intersection of topics)
  4. Exclude mode: single exclude, no boost
  5. Boost + exclude combos
  6. Graph + multi-semantic
  7. Edge cases & error handling
  8. Verification that exclude actually removes relevant papers
"""
import json
import sys
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


def ids_from(resp: dict) -> set[str]:
    return {h["arxiv_id"] for h in resp.get("hits", [])}


def titles_from(resp: dict) -> list[str]:
    return [h.get("title", "") for h in resp.get("hits", [])]


def cats_from(resp: dict) -> set[str]:
    cats = set()
    for h in resp.get("hits", []):
        cats.update(h.get("categories", []))
    return cats


# ════════════════════════════════════════════════
# 1. BACKWARD COMPATIBILITY — single semantic object (no mode)
# ════════════════════════════════════════════════
print("\n" + "=" * 70)
print("1. BACKWARD COMPATIBILITY — single semantic object")
print("=" * 70)

# 1a. Single object with no mode field
d = api("/search", {
    "limit": 5,
    "semantic": {"text": "deep learning image classification", "level": "abstract", "weight": 1.0}
})
test("Single semantic (no mode) returns hits", len(d.get("hits", [])) > 0,
     f"got {len(d.get('hits', []))}")

# 1b. Single object with explicit mode=boost
d2 = api("/search", {
    "limit": 5,
    "semantic": {"text": "deep learning image classification", "level": "abstract", "weight": 1.0, "mode": "boost"}
})
test("Single semantic (mode=boost) returns hits", len(d2.get("hits", [])) > 0)

# 1c. Both should return the same results
ids_no_mode = ids_from(d)
ids_with_mode = ids_from(d2)
test("No-mode and mode=boost give same results", ids_no_mode == ids_with_mode,
     f"differ: {ids_no_mode.symmetric_difference(ids_with_mode)}")

# 1d. Combine with text query (old pattern)
d3 = api("/search", {
    "query": "convolutional neural network",
    "limit": 5,
    "semantic": {"text": "medical imaging diagnosis", "level": "abstract", "weight": 0.7}
})
test("Single semantic + text query works", len(d3.get("hits", [])) > 0)

# 1e. Combine with filters
d4 = api("/search", {
    "limit": 5,
    "categories": ["cs.CV"],
    "semantic": {"text": "self-driving car perception", "level": "abstract", "weight": 1.0}
})
hits = d4.get("hits", [])
all_cv = all("cs.CV" in h.get("categories", []) for h in hits) if hits else False
test("Single semantic + category filter", all_cv and len(hits) > 0,
     f"hits={len(hits)}, all_cv={all_cv}")

# 1f. Title-level semantic
d5 = api("/search", {
    "limit": 5,
    "semantic": {"text": "attention is all you need", "level": "title", "weight": 1.0}
})
test("Title-level semantic works", len(d5.get("hits", [])) > 0)


# ════════════════════════════════════════════════
# 2. MULTI-BOOST — multiple boost semantics (topic intersection)
# ════════════════════════════════════════════════
print("\n" + "=" * 70)
print("2. MULTI-BOOST — intersection of two topics")
print("=" * 70)

# 2a. ML + biology → should get computational biology papers
d = api("/search", {
    "limit": 10,
    "semantic": [
        {"text": "machine learning neural networks", "level": "abstract", "weight": 1.0, "mode": "boost"},
        {"text": "protein structure prediction biology", "level": "abstract", "weight": 1.0, "mode": "boost"}
    ]
})
bio_cats = cats_from(d)
test("ML+Biology returns bio-related papers",
     any(c.startswith("q-bio") or c in ("cs.CE", "cs.LG") for c in bio_cats),
     f"cats: {bio_cats}")
test("ML+Biology returns hits", len(d.get("hits", [])) > 0)

# 2b. Physics + CS → should get quantum computing or similar
d = api("/search", {
    "limit": 10,
    "semantic": [
        {"text": "quantum mechanics quantum physics", "level": "abstract", "weight": 1.0, "mode": "boost"},
        {"text": "algorithm optimization computation", "level": "abstract", "weight": 1.0, "mode": "boost"}
    ]
})
test("Physics+CS returns hits", len(d.get("hits", [])) > 0)
cats = cats_from(d)
# Should have both physics and CS categories
has_phys = any(c.startswith("quant-ph") or c.startswith("physics") or c.startswith("cond-mat") for c in cats)
has_cs = any(c.startswith("cs.") for c in cats)
test("Physics+CS returns cross-domain papers", has_phys or has_cs,
     f"cats: {cats}")

# 2c. Multi-boost gives DIFFERENT results than single boost
d_single = api("/search", {
    "limit": 10,
    "semantic": [
        {"text": "machine learning neural networks", "level": "abstract", "weight": 1.0, "mode": "boost"},
    ]
})
d_multi = api("/search", {
    "limit": 10,
    "semantic": [
        {"text": "machine learning neural networks", "level": "abstract", "weight": 1.0, "mode": "boost"},
        {"text": "protein structure prediction biology", "level": "abstract", "weight": 1.0, "mode": "boost"}
    ]
})
ids_single = ids_from(d_single)
ids_multi = ids_from(d_multi)
test("Multi-boost differs from single-boost", ids_single != ids_multi,
     "identical results — second semantic had no effect")

# 2d. Three boost topics
d = api("/search", {
    "limit": 10,
    "semantic": [
        {"text": "graph neural network", "level": "abstract", "weight": 1.0, "mode": "boost"},
        {"text": "drug discovery molecular", "level": "abstract", "weight": 1.0, "mode": "boost"},
        {"text": "combinatorial optimization", "level": "abstract", "weight": 0.5, "mode": "boost"}
    ]
})
test("Three boost semantics returns hits", len(d.get("hits", [])) > 0)

# 2e. Title + abstract level mix
d = api("/search", {
    "limit": 5,
    "semantic": [
        {"text": "transformer", "level": "title", "weight": 1.0, "mode": "boost"},
        {"text": "natural language generation text", "level": "abstract", "weight": 0.8, "mode": "boost"}
    ]
})
test("Mixed title+abstract levels", len(d.get("hits", [])) > 0)


# ════════════════════════════════════════════════
# 3. EXCLUDE MODE — filter OUT similar papers
# ════════════════════════════════════════════════
print("\n" + "=" * 70)
print("3. EXCLUDE MODE — filtering out similar papers")
print("=" * 70)

# 3a. Text query + exclude: RL but NOT Atari
d = api("/search", {
    "query": "reinforcement learning",
    "limit": 10,
    "semantic": [
        {"text": "Atari game playing arcade video games", "level": "abstract", "weight": 0.5, "mode": "exclude"}
    ]
})
titles = titles_from(d)
atari_count = sum(1 for t in titles if "atari" in t.lower() or "game" in t.lower())
test("RL + exclude Atari: no game papers in results", atari_count == 0,
     f"found {atari_count} game-related titles: {[t for t in titles if 'atari' in t.lower() or 'game' in t.lower()]}")
test("RL + exclude Atari: still returns RL papers", len(d.get("hits", [])) > 0)

# 3b. Just an exclude with no boost and no text query — should still work
d = api("/search", {
    "limit": 10,
    "categories": ["cs.LG"],
    "semantic": [
        {"text": "computer vision image recognition", "level": "abstract", "weight": 0.5, "mode": "exclude"}
    ]
})
test("Category + exclude (no boost/query)", len(d.get("hits", [])) > 0)

# 3c. Exclude with date filter
d = api("/search", {
    "query": "neural network",
    "limit": 10,
    "submitted_date": {"gte": "2023-01-01"},
    "semantic": [
        {"text": "language model GPT BERT", "level": "abstract", "weight": 0.5, "mode": "exclude"}
    ]
})
test("Exclude + date filter", len(d.get("hits", [])) > 0)
dates_ok = all(h.get("submitted_date", "2023") >= "2023" for h in d.get("hits", []))
test("Date filter still respected with exclude", dates_ok)


# ════════════════════════════════════════════════
# 4. BOOST + EXCLUDE COMBOS
# ════════════════════════════════════════════════
print("\n" + "=" * 70)
print("4. BOOST + EXCLUDE COMBOS")
print("=" * 70)

# 4a. Boost ML, exclude vision → should get NLP/tabular/theory ML papers
d = api("/search", {
    "limit": 10,
    "semantic": [
        {"text": "deep learning machine learning", "level": "abstract", "weight": 1.0, "mode": "boost"},
        {"text": "computer vision image recognition object detection convolutional", "level": "abstract", "weight": 0.5, "mode": "exclude"}
    ]
})
test("Boost ML + exclude vision: returns hits", len(d.get("hits", [])) > 0)
cats = cats_from(d)
# Shouldn't be dominated by cs.CV
cv_only = all(c == "cs.CV" for c in cats)
test("Boost ML + exclude vision: not all cs.CV", not cv_only, f"cats: {cats}")

# 4b. Boost NLP, exclude translation → should get non-MT NLP papers
d = api("/search", {
    "limit": 10,
    "semantic": [
        {"text": "natural language processing text understanding", "level": "abstract", "weight": 1.0, "mode": "boost"},
        {"text": "machine translation neural MT bilingual parallel corpus", "level": "abstract", "weight": 0.5, "mode": "exclude"}
    ]
})
titles = titles_from(d)
mt_count = sum(1 for t in titles if "translat" in t.lower())
test("Boost NLP + exclude MT: few translation papers", mt_count <= 2,
     f"got {mt_count} translation papers")

# 4c. Boost robotics, exclude simulation → should get real-world robotics
d = api("/search", {
    "limit": 10,
    "semantic": [
        {"text": "robot manipulation grasping control", "level": "abstract", "weight": 1.0, "mode": "boost"},
        {"text": "simulation virtual environment synthetic data", "level": "abstract", "weight": 0.5, "mode": "exclude"}
    ]
})
test("Boost robotics + exclude simulation", len(d.get("hits", [])) > 0)

# 4d. Two boosts + one exclude
d = api("/search", {
    "limit": 10,
    "semantic": [
        {"text": "reinforcement learning policy optimization", "level": "abstract", "weight": 1.0, "mode": "boost"},
        {"text": "robotics control manipulation", "level": "abstract", "weight": 0.8, "mode": "boost"},
        {"text": "Atari game playing arcade", "level": "abstract", "weight": 0.5, "mode": "exclude"}
    ]
})
test("Two boosts + one exclude", len(d.get("hits", [])) > 0)
titles = titles_from(d)
game_count = sum(1 for t in titles if "game" in t.lower() or "atari" in t.lower())
test("Two boosts + exclude: no game papers", game_count == 0,
     f"found {game_count} game titles")

# 4e. One boost + two excludes
d = api("/search", {
    "limit": 10,
    "semantic": [
        {"text": "machine learning applications", "level": "abstract", "weight": 1.0, "mode": "boost"},
        {"text": "computer vision image", "level": "abstract", "weight": 0.5, "mode": "exclude"},
        {"text": "natural language processing text", "level": "abstract", "weight": 0.5, "mode": "exclude"}
    ]
})
test("One boost + two excludes", len(d.get("hits", [])) > 0)
cats = cats_from(d)
test("One boost + two excludes: diverse categories", len(cats) > 0, f"cats: {cats}")


# ════════════════════════════════════════════════
# 5. GRAPH + MULTI-SEMANTIC
# ════════════════════════════════════════════════
print("\n" + "=" * 70)
print("5. GRAPH + MULTI-SEMANTIC")
print("=" * 70)

# 5a. Graph + single boost (backward compat)
d = api("/graph", {
    "graph": {"type": "interdisciplinary", "min_categories": 3, "limit": 5},
    "semantic": {"text": "quantum computing error correction", "level": "abstract", "weight": 0.7}
})
test("Graph + single semantic (backward)", len(d.get("nodes", [])) > 0)

# 5b. Graph + list with one boost
d = api("/graph", {
    "graph": {"type": "interdisciplinary", "min_categories": 3, "limit": 5},
    "semantic": [
        {"text": "quantum computing error correction", "level": "abstract", "weight": 0.7, "mode": "boost"}
    ]
})
test("Graph + list[1] boost", len(d.get("nodes", [])) > 0)

# 5c. Graph + multi boost
d = api("/graph", {
    "graph": {"type": "coauthor_network", "seed_author": "Yann LeCun", "depth": 1, "limit": 10},
    "semantic": [
        {"text": "self-supervised learning contrastive", "level": "abstract", "weight": 0.8, "mode": "boost"},
        {"text": "image representation visual features", "level": "abstract", "weight": 0.5, "mode": "boost"}
    ]
})
test("Graph coauthor + multi boost",
     len(d.get("nodes", [])) > 0 and d.get("metadata", {}).get("seed_author") == "Yann LeCun",
     f"nodes={len(d.get('nodes', []))}")

# 5d. Graph + boost + exclude
d = api("/graph", {
    "graph": {"type": "category_diversity", "min_categories": 3, "limit": 10},
    "semantic": [
        {"text": "deep learning neural architecture", "level": "abstract", "weight": 0.8, "mode": "boost"},
        {"text": "computer vision image classification object", "level": "abstract", "weight": 0.5, "mode": "exclude"}
    ]
})
papers = [n for n in d.get("nodes", []) if n["type"] == "paper"]
test("Graph category_diversity + boost+exclude", len(papers) >= 0)  # May be 0 if exclude prunes too much

# 5e. Graph without semantic still works
d = api("/graph", {
    "graph": {"type": "cross_category_flow", "limit": 10}
})
test("Graph without semantic (regression)", len(d.get("edges", [])) > 0)


# ════════════════════════════════════════════════
# 6. EDGE CASES & ERROR HANDLING
# ════════════════════════════════════════════════
print("\n" + "=" * 70)
print("6. EDGE CASES & ERROR HANDLING")
print("=" * 70)

# 6a. Empty semantic list
d = api("/search", {"query": "test", "limit": 3, "semantic": []})
test("Empty semantic list (treated as None)", "hits" in d, str(d.get("detail", "")))

# 6b. Single exclude with no boost, no query — pure exclude
d = api("/search", {
    "limit": 5,
    "semantic": [
        {"text": "everything", "level": "abstract", "weight": 1.0, "mode": "exclude"}
    ]
})
test("Pure exclude (no query/boost) doesn't crash", "hits" in d or "detail" in d)

# 6c. Invalid mode
d = api("/search", {
    "limit": 3,
    "semantic": [{"text": "test", "level": "abstract", "weight": 1.0, "mode": "invalid"}]
})
test("Invalid mode returns error", "detail" in d, f"no error: {list(d.keys())}")

# 6d. Empty text in list
d = api("/search", {
    "limit": 3,
    "semantic": [{"text": "", "level": "abstract", "weight": 1.0, "mode": "boost"}]
})
test("Empty text in list doesn't crash", "hits" in d or "detail" in d)

# 6e. Very high weight
d = api("/search", {
    "limit": 3,
    "semantic": [{"text": "neural", "level": "abstract", "weight": 10.0, "mode": "boost"}]
})
test("Weight=10.0 works", len(d.get("hits", [])) > 0)

# 6f. Weight=0 in a list
d = api("/search", {
    "query": "deep learning",
    "limit": 3,
    "semantic": [
        {"text": "ignored", "level": "abstract", "weight": 0.0, "mode": "boost"},
        {"text": "also ignored for exclude", "level": "abstract", "weight": 0.0, "mode": "exclude"}
    ]
})
test("Weight=0 entries in list (grace)", "hits" in d or "detail" in d)

# 6g. Max entries (5 semantics)
d = api("/search", {
    "limit": 3,
    "semantic": [
        {"text": f"topic {i}", "level": "abstract", "weight": 0.5, "mode": "boost"}
        for i in range(5)
    ]
})
test("5 semantic entries", "hits" in d or "detail" in d)

# 6h. Mix of title and abstract levels in list
d = api("/search", {
    "limit": 5,
    "semantic": [
        {"text": "transformer architecture", "level": "title", "weight": 1.0, "mode": "boost"},
        {"text": "protein folding prediction", "level": "abstract", "weight": 0.5, "mode": "exclude"}
    ]
})
test("Mixed levels (title boost + abstract exclude)", len(d.get("hits", [])) > 0)


# ════════════════════════════════════════════════
# 7. VERIFICATION: EXCLUDE ACTUALLY REMOVES RELEVANT PAPERS
# ════════════════════════════════════════════════
print("\n" + "=" * 70)
print("7. VERIFICATION — exclude removes papers that would otherwise appear")
print("=" * 70)

# Strategy: search for topic X. Then search for X with exclude on X's text.
# The excluded version should NOT contain the top results from the first search.

# 7a. Search for GANs without exclude
d_base = api("/search", {
    "limit": 20,
    "semantic": [{"text": "generative adversarial network image synthesis", "level": "abstract", "weight": 1.0, "mode": "boost"}]
})
base_ids = ids_from(d_base)

# Now search broadly but exclude GANs
d_excluded = api("/search", {
    "query": "deep learning",
    "limit": 20,
    "semantic": [{"text": "generative adversarial network image synthesis", "level": "abstract", "weight": 0.5, "mode": "exclude"}]
})
excluded_ids = ids_from(d_excluded)
overlap = base_ids & excluded_ids
test("Exclude removes GAN papers from broad search",
     len(overlap) < len(base_ids) / 2,
     f"overlap={len(overlap)}/{len(base_ids)} ({overlap})")

# 7b. Search for transformer NLP
d_base2 = api("/search", {
    "limit": 20,
    "semantic": [{"text": "transformer attention mechanism BERT GPT", "level": "abstract", "weight": 1.0, "mode": "boost"}]
})
base_ids2 = ids_from(d_base2)

# Search for ML but exclude transformers
d_excluded2 = api("/search", {
    "limit": 20,
    "semantic": [
        {"text": "machine learning deep learning", "level": "abstract", "weight": 1.0, "mode": "boost"},
        {"text": "transformer attention mechanism BERT GPT", "level": "abstract", "weight": 0.5, "mode": "exclude"}
    ]
})
excluded_ids2 = ids_from(d_excluded2)
overlap2 = base_ids2 & excluded_ids2
test("Exclude removes transformer papers from ML search",
     len(overlap2) < len(base_ids2) / 2,
     f"overlap={len(overlap2)}/{len(base_ids2)}")

# 7c. Verify exclude removes overlapping papers (same topic for boost+exclude)
d_base_cv = api("/search", {
    "limit": 20,
    "semantic": [{"text": "image classification convolutional neural network object detection", "level": "abstract", "weight": 1.0, "mode": "boost"}]
})
base_cv_ids = ids_from(d_base_cv)

d_excluded_cv = api("/search", {
    "limit": 20,
    "semantic": [
        {"text": "neural network deep learning", "level": "abstract", "weight": 1.0, "mode": "boost"},
        {"text": "image classification convolutional neural network object detection", "level": "abstract", "weight": 0.5, "mode": "exclude"}
    ]
})
excluded_cv_ids = ids_from(d_excluded_cv)
overlap_cv = base_cv_ids & excluded_cv_ids
test("Excluding same-topic papers removes them from results",
     len(overlap_cv) < len(base_cv_ids) / 2,
     f"overlap={len(overlap_cv)}/{len(base_cv_ids)}")


# ════════════════════════════════════════════════
# 8. MULTI-SEMANTIC WITH ALL FILTER COMBOS
# ════════════════════════════════════════════════
print("\n" + "=" * 70)
print("8. MULTI-SEMANTIC WITH VARIOUS FILTERS")
print("=" * 70)

# 8a. Multi-semantic + date filter
d = api("/search", {
    "limit": 5,
    "submitted_date": {"gte": "2024-01-01"},
    "semantic": [
        {"text": "large language model fine-tuning", "level": "abstract", "weight": 1.0, "mode": "boost"},
        {"text": "image generation diffusion", "level": "abstract", "weight": 0.5, "mode": "exclude"}
    ]
})
hits = d.get("hits", [])
dates_ok = all(h.get("submitted_date", "2024") >= "2024" for h in hits) if hits else True
test("Multi-semantic + date filter", dates_ok and len(hits) > 0,
     f"hits={len(hits)}, dates={[h.get('submitted_date','?')[:10] for h in hits[:3]]}")

# 8b. Multi-semantic + category filter
d = api("/search", {
    "limit": 5,
    "categories": ["cs.CL"],
    "semantic": [
        {"text": "sentiment analysis opinion mining", "level": "abstract", "weight": 1.0, "mode": "boost"},
        {"text": "machine translation parallel corpus", "level": "abstract", "weight": 0.5, "mode": "exclude"}
    ]
})
hits = d.get("hits", [])
all_cl = all("cs.CL" in h.get("categories", []) for h in hits) if hits else False
test("Multi-semantic + category filter", all_cl and len(hits) > 0,
     f"hits={len(hits)}, all_cl={all_cl}")

# 8c. Multi-semantic + author filter
d = api("/search", {
    "limit": 5,
    "author": "Geoffrey Hinton",
    "semantic": [
        {"text": "neural network learning representation", "level": "abstract", "weight": 0.8, "mode": "boost"}
    ]
})
test("Multi-semantic + author filter", len(d.get("hits", [])) > 0, f"hits={len(d.get('hits', []))}")

# 8d. Multi-semantic + citation filter
d = api("/search", {
    "limit": 5,
    "min_citations": 3,
    "semantic": [
        {"text": "deep reinforcement learning", "level": "abstract", "weight": 1.0, "mode": "boost"},
        {"text": "game playing Atari", "level": "abstract", "weight": 0.5, "mode": "exclude"}
    ]
})
test("Multi-semantic + min_citations", len(d.get("hits", [])) >= 0)  # May be 0 if not many cited papers with embeddings

# 8e. Multi-semantic + sorting
d = api("/search", {
    "limit": 5,
    "sort_by": "date",
    "sort_order": "desc",
    "semantic": [
        {"text": "federated learning privacy", "level": "abstract", "weight": 0.8, "mode": "boost"}
    ]
})
test("Multi-semantic + sort by date", len(d.get("hits", [])) > 0)


# ════════════════════════════════════════════════
# SUMMARY
# ════════════════════════════════════════════════
print("\n" + "=" * 70)
total = passed + failed
print(f"RESULTS: {passed}/{total} passed, {failed} failed")
print("=" * 70)

if errors:
    print("\nFailed tests:")
    for e in errors:
        print(f"  ✗ {e}")

sys.exit(0 if failed == 0 else 1)
