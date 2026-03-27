#!/usr/bin/env python3
"""
Comprehensive query type and performance test suite.
Tests every search feature against the live ~3M paper database.
"""
import os, sys, time, json

os.environ["ES_HOST"] = "localhost"
os.environ["REDIS_HOST"] = "localhost"
os.environ["API_KEYS"] = "perf-test-key"
os.environ["RATE_LIMIT_PER_MINUTE"] = "99999"
os.environ["RATE_LIMIT_BURST"] = "9999"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.core.config import reset_settings
reset_settings()

from fastapi.testclient import TestClient
from src.api.main import create_app

app = create_app()
H = {"X-API-Key": "perf-test-key"}

passed = 0
failed = 0
errors = []
perf_results = []

_cm = TestClient(app)
client = _cm.__enter__()


def test(name, fn, max_ms=None):
    global passed, failed
    try:
        t0 = time.monotonic()
        fn()
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        if max_ms and elapsed_ms > max_ms:
            failed += 1
            msg = f"TOO SLOW: {elapsed_ms}ms > {max_ms}ms"
            errors.append((name, msg))
            perf_results.append((name, elapsed_ms, "SLOW"))
            print(f"  ✗ {name} [{elapsed_ms}ms]: {msg}")
        else:
            passed += 1
            perf_results.append((name, elapsed_ms, "OK"))
            print(f"  ✓ {name} [{elapsed_ms}ms]")
    except AssertionError as e:
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        failed += 1
        errors.append((name, str(e)))
        perf_results.append((name, elapsed_ms, "FAIL"))
        print(f"  ✗ {name} [{elapsed_ms}ms]: {e}")
    except Exception as e:
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        failed += 1
        errors.append((name, f"EXCEPTION: {e}"))
        perf_results.append((name, elapsed_ms, "ERROR"))
        print(f"  ✗ {name} [{elapsed_ms}ms]: EXCEPTION {e}")


# ════════════════════════════════════════════════════════
# 1. FULL-TEXT SEARCH
# ════════════════════════════════════════════════════════
print("\n═══ 1. FULL-TEXT SEARCH ═══")

def test_ft_single_word():
    r = client.post("/search", json={"query": "transformer"}, headers=H)
    d = r.json()
    assert d["total"] > 10000, f"Expected >10k results, got {d['total']}"
    assert all("score" in h for h in d["hits"])
test("Single word: 'transformer'", test_ft_single_word, max_ms=3000)

def test_ft_multi_word():
    r = client.post("/search", json={"query": "graph neural network"}, headers=H)
    d = r.json()
    assert d["total"] > 5000
test("Multi-word: 'graph neural network'", test_ft_multi_word, max_ms=3000)

def test_ft_phrase_and():
    r = client.post("/search", json={"query": "quantum error correction", "operator": "and"}, headers=H)
    d = r.json()
    assert d["total"] > 0
    assert d["total"] < 50000, "AND should be stricter"
test("AND operator: 'quantum error correction'", test_ft_phrase_and, max_ms=3000)

def test_ft_phrase_or():
    r_and = client.post("/search", json={"query": "quantum error correction", "operator": "and"}, headers=H)
    r_or = client.post("/search", json={"query": "quantum error correction", "operator": "or"}, headers=H)
    assert r_or.json()["total"] >= r_and.json()["total"], "OR should return >= AND"
test("OR returns >= AND results", test_ft_phrase_or, max_ms=5000)

def test_ft_title_only():
    r = client.post("/search", json={"title_query": "attention is all you need"}, headers=H)
    d = r.json()
    assert d["total"] > 0
test("Title query: 'attention is all you need'", test_ft_title_only, max_ms=3000)

def test_ft_abstract_only():
    r = client.post("/search", json={"abstract_query": "we propose a novel method for training"}, headers=H)
    d = r.json()
    assert d["total"] > 100
test("Abstract query: 'we propose a novel method'", test_ft_abstract_only, max_ms=3000)

def test_ft_title_plus_abstract():
    r = client.post("/search", json={
        "title_query": "reinforcement learning",
        "abstract_query": "multi-agent",
    }, headers=H)
    d = r.json()
    assert d["total"] > 0
test("Title + abstract combined", test_ft_title_plus_abstract, max_ms=3000)

def test_ft_highlights():
    r = client.post("/search", json={"query": "BERT language model", "highlight": True}, headers=H)
    d = r.json()
    found_hl = False
    for h in d["hits"]:
        if h.get("highlights"):
            found_hl = True
            # Highlights should contain <em> tags
            for field, frags in h["highlights"].items():
                for frag in frags:
                    assert "<em>" in frag, f"Missing highlight tags in {frag[:50]}"
            break
    assert found_hl, "No highlights found"
test("Highlights with <em> tags", test_ft_highlights, max_ms=3000)

def test_ft_relevance_scoring():
    r = client.post("/search", json={"query": "convolutional neural network image classification"}, headers=H)
    d = r.json()
    scores = [h["score"] for h in d["hits"]]
    for i in range(len(scores) - 1):
        assert scores[i] >= scores[i+1], f"Score not descending: {scores[i]} < {scores[i+1]}"
test("Relevance scores descending", test_ft_relevance_scoring, max_ms=3000)

def test_ft_no_results():
    r = client.post("/search", json={"query": "xyzzyplugh42foobarbaz"}, headers=H)
    assert r.json()["total"] == 0
    assert len(r.json()["hits"]) == 0
test("Nonsense query → 0 results", test_ft_no_results, max_ms=2000)

def test_ft_msm_number():
    r = client.post("/search", json={
        "query": "deep reinforcement learning robot navigation",
        "minimum_should_match": "4",
    }, headers=H)
    assert r.status_code == 200
    d = r.json()
    assert d["total"] >= 0
test("minimum_should_match=4", test_ft_msm_number, max_ms=3000)

def test_ft_msm_percent():
    r100 = client.post("/search", json={
        "query": "neural network training optimization convergence",
        "minimum_should_match": "100%",
    }, headers=H)
    r50 = client.post("/search", json={
        "query": "neural network training optimization convergence",
        "minimum_should_match": "50%",
    }, headers=H)
    assert r50.json()["total"] >= r100.json()["total"], "50% should match >= 100%"
test("MSM 50% >= 100% results", test_ft_msm_percent, max_ms=5000)

def test_ft_empty_query_matchall():
    r = client.post("/search", json={}, headers=H)
    d = r.json()
    assert d["total"] > 2900000, f"Empty query should match all ~3M papers, got {d['total']}"
test("Empty query → match_all (~3M)", test_ft_empty_query_matchall, max_ms=3000)


# ════════════════════════════════════════════════════════
# 2. FUZZY SEARCH
# ════════════════════════════════════════════════════════
print("\n═══ 2. FUZZY SEARCH ═══")

def test_fz_typo1():
    r = client.post("/search", json={"fuzzy": "tansformer", "fuzzy_fuzziness": 1}, headers=H)
    d = r.json()
    assert d["total"] > 0, "Should match 'transformer' with 1 edit"
test("Fuzzy 1-edit: 'tansformer'", test_fz_typo1, max_ms=3000)

def test_fz_typo2():
    r = client.post("/search", json={"fuzzy": "nueral neetwork", "fuzzy_fuzziness": 2}, headers=H)
    d = r.json()
    assert d["total"] > 0
test("Fuzzy 2-edit: 'nueral neetwork'", test_fz_typo2, max_ms=3000)

def test_fz_exact():
    r = client.post("/search", json={"fuzzy": "xyznonexist", "fuzzy_fuzziness": 0}, headers=H)
    assert r.json()["total"] == 0
test("Fuzzy 0-edit: strict match", test_fz_exact, max_ms=2000)

def test_fz_auto():
    r = client.post("/search", json={"fuzzy": "reinformcent lerning"}, headers=H)
    d = r.json()
    assert d["total"] > 0, "Auto fuzziness should handle typos"
test("Fuzzy auto: 'reinformcent lerning'", test_fz_auto, max_ms=3000)

def test_fz_combined_with_query():
    r = client.post("/search", json={
        "query": "deep learning",
        "fuzzy": "convluton",
    }, headers=H)
    assert r.status_code == 200
test("Fuzzy + full-text combined", test_fz_combined_with_query, max_ms=3000)


# ════════════════════════════════════════════════════════
# 3. REGEX SEARCH
# ════════════════════════════════════════════════════════
print("\n═══ 3. REGEX SEARCH ═══")

def test_rx_title():
    r = client.post("/search", json={"title_regex": ".*[Tt]ransformer.*"}, headers=H)
    assert r.status_code == 200
    d = r.json()
    assert d["total"] > 0
test("Title regex: .*[Tt]ransformer.*", test_rx_title, max_ms=10000)

def test_rx_case_sensitive():
    r_upper = client.post("/search", json={"title_regex": ".*BERT.*"}, headers=H)
    assert r_upper.status_code == 200
test("Title regex: .*BERT.*", test_rx_case_sensitive, max_ms=10000)

def test_rx_invalid():
    r = client.post("/search", json={"title_regex": "[unclosed"}, headers=H)
    assert r.status_code == 422
test("Invalid regex → 422", test_rx_invalid, max_ms=1000)

def test_rx_too_long():
    r = client.post("/search", json={"title_regex": "a" * 201}, headers=H)
    assert r.status_code == 422
test("Regex >200 chars → 422", test_rx_too_long, max_ms=1000)

def test_rx_redos():
    r = client.post("/search", json={"title_regex": "(a+)+"}, headers=H)
    assert r.status_code == 422
test("ReDoS pattern → 422", test_rx_redos, max_ms=1000)


# ════════════════════════════════════════════════════════
# 4. AUTHOR SEARCH
# ════════════════════════════════════════════════════════
print("\n═══ 4. AUTHOR SEARCH ═══")

def test_au_name():
    r = client.post("/search", json={"author": "Yann LeCun"}, headers=H)
    d = r.json()
    assert d["total"] > 0
    found = False
    for h in d["hits"]:
        for a in h["authors"]:
            if "lecun" in a["name"].lower():
                found = True
    assert found, "LeCun not found in results"
test("Author: 'Yann LeCun'", test_au_name, max_ms=3000)

def test_au_first_author():
    r = client.post("/search", json={"first_author": "Hinton"}, headers=H)
    d = r.json()
    assert d["total"] > 0
    for h in d["hits"]:
        first = h["authors"][0] if h["authors"] else {}
        assert "hinton" in first.get("name", "").lower() or any(
            a.get("is_first_author") and "hinton" in a["name"].lower()
            for a in h["authors"]
        )
test("First author: 'Hinton'", test_au_first_author, max_ms=3000)

def test_au_combined_with_topic():
    r = client.post("/search", json={"author": "Bengio", "query": "attention"}, headers=H)
    d = r.json()
    assert d["total"] > 0
test("Author Bengio + query 'attention'", test_au_combined_with_topic, max_ms=3000)

def test_au_zacharie():
    r = client.post("/search", json={"author": "Zacharie Bugaud"}, headers=H)
    d = r.json()
    assert d["total"] >= 3, f"Expected >=3 papers by Zacharie Bugaud, got {d['total']}"
test("Author: 'Zacharie Bugaud' → papers found", test_au_zacharie, max_ms=3000)


# ════════════════════════════════════════════════════════
# 5. CATEGORY FILTERS
# ════════════════════════════════════════════════════════
print("\n═══ 5. CATEGORY FILTERS ═══")

def test_cat_single():
    r = client.post("/search", json={"categories": ["cs.AI"]}, headers=H)
    d = r.json()
    assert d["total"] > 0
    for h in d["hits"]:
        assert "cs.AI" in h["categories"]
test("Category: cs.AI", test_cat_single, max_ms=3000)

def test_cat_multi():
    r = client.post("/search", json={"categories": ["hep-th", "hep-ph"]}, headers=H)
    d = r.json()
    assert d["total"] > 0
    for h in d["hits"]:
        assert "hep-th" in h["categories"] or "hep-ph" in h["categories"]
test("Categories: hep-th OR hep-ph", test_cat_multi, max_ms=3000)

def test_cat_primary():
    r = client.post("/search", json={"primary_category": "quant-ph"}, headers=H)
    d = r.json()
    assert d["total"] > 0
    for h in d["hits"]:
        assert h["primary_category"] == "quant-ph"
test("Primary category: quant-ph", test_cat_primary, max_ms=3000)

def test_cat_exclude():
    r_all = client.post("/search", json={"categories": ["cs.LG"]}, headers=H)
    r_excl = client.post("/search", json={"categories": ["cs.LG"], "exclude_categories": ["cs.CV"]}, headers=H)
    assert r_excl.json()["total"] <= r_all.json()["total"]
test("Exclude cs.CV from cs.LG", test_cat_exclude, max_ms=5000)

def test_cat_plus_query():
    r = client.post("/search", json={"categories": ["cs.CL"], "query": "machine translation"}, headers=H)
    d = r.json()
    assert d["total"] > 0
    for h in d["hits"]:
        assert "cs.CL" in h["categories"]
test("Category cs.CL + query 'machine translation'", test_cat_plus_query, max_ms=3000)

def test_cat_rare():
    r = client.post("/search", json={"categories": ["cs.ET"]}, headers=H)
    d = r.json()
    assert d["total"] >= 0  # may be rare
test("Rare category: cs.ET", test_cat_rare, max_ms=3000)


# ════════════════════════════════════════════════════════
# 6. DATE FILTERS
# ════════════════════════════════════════════════════════
print("\n═══ 6. DATE FILTERS ═══")

def test_date_gte():
    r = client.post("/search", json={"submitted_date": {"gte": "2026-01-01T00:00:00+00:00"}}, headers=H)
    d = r.json()
    assert d["total"] > 0
    for h in d["hits"]:
        assert h["submitted_date"] >= "2026-01-01"
test("Date >= 2026-01-01", test_date_gte, max_ms=3000)

def test_date_lte():
    r = client.post("/search", json={"submitted_date": {"lte": "2006-12-31T23:59:59+00:00"}}, headers=H)
    d = r.json()
    assert d["total"] > 0
test("Date <= 2006-12-31", test_date_lte, max_ms=3000)

def test_date_range():
    r = client.post("/search", json={
        "submitted_date": {"gte": "2023-06-01T00:00:00+00:00", "lte": "2023-06-30T23:59:59+00:00"}
    }, headers=H)
    d = r.json()
    assert d["total"] > 0
    for h in d["hits"]:
        assert h["submitted_date"][:7] == "2023-06"
test("Date range: June 2023", test_date_range, max_ms=3000)

def test_date_single_day():
    r = client.post("/search", json={
        "submitted_date": {"gte": "2025-01-15T00:00:00+00:00", "lte": "2025-01-15T23:59:59+00:00"}
    }, headers=H)
    assert r.status_code == 200
test("Date: single day 2025-01-15", test_date_single_day, max_ms=3000)

def test_date_future():
    r = client.post("/search", json={"submitted_date": {"gte": "2030-01-01T00:00:00+00:00"}}, headers=H)
    assert r.json()["total"] == 0
test("Future date → 0 results", test_date_future, max_ms=2000)

def test_date_plus_category():
    r = client.post("/search", json={
        "categories": ["cs.AI"],
        "submitted_date": {"gte": "2025-01-01T00:00:00+00:00"},
    }, headers=H)
    d = r.json()
    assert d["total"] > 0
    for h in d["hits"]:
        assert "cs.AI" in h["categories"]
test("Date 2025+ + category cs.AI", test_date_plus_category, max_ms=3000)


# ════════════════════════════════════════════════════════
# 7. METADATA FILTERS
# ════════════════════════════════════════════════════════
print("\n═══ 7. METADATA FILTERS ═══")

def test_meta_github_true():
    r = client.post("/search", json={"has_github": True}, headers=H)
    d = r.json()
    assert d["total"] > 0
    for h in d["hits"]:
        assert h["has_github"] is True
test("has_github=True", test_meta_github_true, max_ms=3000)

def test_meta_github_false():
    r = client.post("/search", json={"has_github": False, "limit": 5}, headers=H)
    d = r.json()
    assert d["total"] > 0
    for h in d["hits"]:
        assert h["has_github"] is False
test("has_github=False", test_meta_github_false, max_ms=3000)

def test_meta_pages_min():
    r = client.post("/search", json={"min_page_count": 50}, headers=H)
    d = r.json()
    assert d["total"] > 0
    for h in d["hits"]:
        if h["page_count"] is not None:
            assert h["page_count"] >= 50
test("min_page_count=50", test_meta_pages_min, max_ms=3000)

def test_meta_pages_range():
    r = client.post("/search", json={"min_page_count": 10, "max_page_count": 15}, headers=H)
    d = r.json()
    assert d["total"] > 0
    for h in d["hits"]:
        if h["page_count"] is not None:
            assert 10 <= h["page_count"] <= 15
test("Page count range 10-15", test_meta_pages_range, max_ms=3000)

def test_meta_github_plus_query():
    r = client.post("/search", json={
        "has_github": True,
        "query": "deep learning framework",
        "categories": ["cs.LG"],
    }, headers=H)
    d = r.json()
    for h in d["hits"]:
        assert h["has_github"] is True
        assert "cs.LG" in h["categories"]
test("GitHub + query + category combined", test_meta_github_plus_query, max_ms=3000)


# ════════════════════════════════════════════════════════
# 8. SORTING
# ════════════════════════════════════════════════════════
print("\n═══ 8. SORTING ═══")

def test_sort_date_desc():
    r = client.post("/search", json={"sort_by": "date", "sort_order": "desc", "limit": 50}, headers=H)
    dates = [h["submitted_date"] for h in r.json()["hits"] if h["submitted_date"]]
    for i in range(len(dates) - 1):
        assert dates[i] >= dates[i+1], f"Not desc: {dates[i]} < {dates[i+1]}"
test("Sort by date desc", test_sort_date_desc, max_ms=3000)

def test_sort_date_asc():
    r = client.post("/search", json={"sort_by": "date", "sort_order": "asc", "limit": 50}, headers=H)
    dates = [h["submitted_date"] for h in r.json()["hits"] if h["submitted_date"]]
    for i in range(len(dates) - 1):
        assert dates[i] <= dates[i+1], f"Not asc: {dates[i]} > {dates[i+1]}"
test("Sort by date asc", test_sort_date_asc, max_ms=3000)

def test_sort_relevance():
    r = client.post("/search", json={"query": "deep learning", "sort_by": "relevance", "limit": 50}, headers=H)
    scores = [h["score"] for h in r.json()["hits"] if h["score"] is not None]
    for i in range(len(scores) - 1):
        assert scores[i] >= scores[i+1], f"Not desc: {scores[i]} < {scores[i+1]}"
test("Sort by relevance (score desc)", test_sort_relevance, max_ms=3000)

def test_sort_citations():
    r = client.post("/search", json={"sort_by": "citations", "sort_order": "desc"}, headers=H)
    assert r.status_code == 200
test("Sort by citations", test_sort_citations, max_ms=3000)

def test_sort_page_count():
    r = client.post("/search", json={"sort_by": "page_count", "sort_order": "desc", "limit": 20}, headers=H)
    assert r.status_code == 200
    pages = [h["page_count"] for h in r.json()["hits"] if h["page_count"] is not None]
    if len(pages) > 1:
        for i in range(len(pages) - 1):
            assert pages[i] >= pages[i+1]
test("Sort by page_count desc", test_sort_page_count, max_ms=3000)


# ════════════════════════════════════════════════════════
# 9. PAGINATION
# ════════════════════════════════════════════════════════
print("\n═══ 9. PAGINATION ═══")

def test_pag_first_page():
    r = client.post("/search", json={"offset": 0, "limit": 10, "sort_by": "date", "sort_order": "desc"}, headers=H)
    assert len(r.json()["hits"]) == 10
    assert r.json()["offset"] == 0
test("Page 1: offset=0, limit=10", test_pag_first_page, max_ms=3000)

def test_pag_second_page():
    r = client.post("/search", json={"offset": 10, "limit": 10, "sort_by": "date", "sort_order": "desc"}, headers=H)
    assert len(r.json()["hits"]) == 10
    assert r.json()["offset"] == 10
test("Page 2: offset=10", test_pag_second_page, max_ms=3000)

def test_pag_no_overlap():
    body = {"sort_by": "date", "sort_order": "desc"}
    r1 = client.post("/search", json={**body, "offset": 0, "limit": 20}, headers=H)
    r2 = client.post("/search", json={**body, "offset": 20, "limit": 20}, headers=H)
    ids1 = {h["arxiv_id"] for h in r1.json()["hits"]}
    ids2 = {h["arxiv_id"] for h in r2.json()["hits"]}
    assert ids1.isdisjoint(ids2), f"Overlap: {ids1 & ids2}"
test("No overlap between pages", test_pag_no_overlap, max_ms=5000)

def test_pag_large_limit():
    r = client.post("/search", json={"limit": 200}, headers=H)
    assert len(r.json()["hits"]) == 200
test("Limit=200 (max allowed)", test_pag_large_limit, max_ms=5000)

def test_pag_deep():
    r = client.post("/search", json={"offset": 9900, "limit": 100}, headers=H)
    assert r.status_code == 200
test("Deep pagination offset=9900", test_pag_deep, max_ms=5000)

def test_pag_beyond_limit():
    r = client.post("/search", json={"offset": 9999, "limit": 100}, headers=H)
    assert r.status_code == 400
test("offset+limit > 10000 → 400", test_pag_beyond_limit, max_ms=1000)


# ════════════════════════════════════════════════════════
# 10. COMPLEX COMBINED QUERIES
# ════════════════════════════════════════════════════════
print("\n═══ 10. COMPLEX COMBINED QUERIES ═══")

def test_cmb_full():
    r = client.post("/search", json={
        "query": "language model",
        "categories": ["cs.CL", "cs.AI"],
        "submitted_date": {"gte": "2024-01-01T00:00:00+00:00"},
        "has_github": True,
        "sort_by": "date",
        "sort_order": "desc",
        "limit": 10,
        "highlight": True,
    }, headers=H)
    d = r.json()
    assert d["total"] > 0
    for h in d["hits"]:
        assert h["has_github"] is True
        assert "cs.CL" in h["categories"] or "cs.AI" in h["categories"]
        assert h["submitted_date"] >= "2024-01-01"
test("Full combo: query+cat+date+github+sort+highlight", test_cmb_full, max_ms=5000)

def test_cmb_author_cat_date():
    r = client.post("/search", json={
        "author": "Vaswani",
        "categories": ["cs.CL"],
        "submitted_date": {"gte": "2017-01-01T00:00:00+00:00"},
    }, headers=H)
    assert r.status_code == 200
test("Author + category + date", test_cmb_author_cat_date, max_ms=3000)

def test_cmb_fuzzy_cat_github():
    r = client.post("/search", json={
        "fuzzy": "rienforcement lerning",
        "categories": ["cs.AI"],
        "has_github": True,
    }, headers=H)
    assert r.status_code == 200
test("Fuzzy + category + github", test_cmb_fuzzy_cat_github, max_ms=3000)

def test_cmb_title_abstract_date_pages():
    r = client.post("/search", json={
        "title_query": "survey",
        "abstract_query": "comprehensive overview",
        "min_page_count": 20,
        "submitted_date": {"gte": "2023-01-01T00:00:00+00:00"},
    }, headers=H)
    assert r.status_code == 200
test("Title+abstract+pages+date", test_cmb_title_abstract_date_pages, max_ms=3000)

def test_cmb_exclude_plus_include():
    r = client.post("/search", json={
        "categories": ["cs.LG"],
        "exclude_categories": ["cs.CV", "cs.CL"],
        "query": "optimization",
    }, headers=H)
    d = r.json()
    for h in d["hits"]:
        assert "cs.LG" in h["categories"]
        assert "cs.CV" not in h["categories"]
        assert "cs.CL" not in h["categories"]
test("Include cs.LG, exclude cs.CV+cs.CL", test_cmb_exclude_plus_include, max_ms=3000)


# ════════════════════════════════════════════════════════
# 11. STATS ENDPOINT
# ════════════════════════════════════════════════════════
print("\n═══ 11. STATS ENDPOINT ═══")

def test_stats():
    r = client.get("/stats", headers=H)
    assert r.status_code == 200
    d = r.json()
    assert d["total_papers"] > 2900000
    assert len(d["categories"]) > 50
    assert d["papers_with_github"] > 50000
    assert d["date_range"]["min"] is not None
    assert d["date_range"]["max"] is not None
test("Stats: totals, categories, github, dates", test_stats, max_ms=5000)


# ════════════════════════════════════════════════════════
# 12. SINGLE PAPER LOOKUP
# ════════════════════════════════════════════════════════
print("\n═══ 12. SINGLE PAPER LOOKUP ═══")

def test_paper_new_id():
    r = client.post("/search", json={"limit": 1, "sort_by": "date", "sort_order": "desc"}, headers=H)
    aid = r.json()["hits"][0]["arxiv_id"]
    r2 = client.get(f"/paper/{aid}", headers=H)
    assert r2.status_code == 200
    assert r2.json()["arxiv_id"] == aid
    assert r2.json()["title"]
    assert r2.json()["abstract"]
    assert len(r2.json()["authors"]) > 0
test("Get paper by new-format ID", test_paper_new_id, max_ms=3000)

def test_paper_old_id():
    # Old format like hep-th/0508186
    r = client.get("/paper/hep-th/0508186", headers=H)
    assert r.status_code == 200
    assert "hep-th" in r.json().get("categories", []) or "hep-th" in r.json().get("arxiv_id", "")
test("Get paper by old-format ID", test_paper_old_id, max_ms=3000)

def test_paper_404():
    r = client.get("/paper/9999.99999", headers=H)
    assert r.status_code == 404
test("Nonexistent paper → 404", test_paper_404, max_ms=2000)


# ════════════════════════════════════════════════════════
# 13. INPUT VALIDATION & SECURITY
# ════════════════════════════════════════════════════════
print("\n═══ 13. INPUT VALIDATION & SECURITY ═══")

def test_val_no_key():
    r = client.post("/search", json={"query": "test"})
    assert r.status_code == 401
test("No API key → 401", test_val_no_key, max_ms=1000)

def test_val_bad_key():
    r = client.post("/search", json={"query": "test"}, headers={"X-API-Key": "wrong"})
    assert r.status_code == 403
test("Bad API key → 403", test_val_bad_key, max_ms=1000)

def test_val_query_length():
    r = client.post("/search", json={"query": "a" * 2500}, headers=H)
    assert r.status_code == 422
test("Query >2000 chars → 422", test_val_query_length, max_ms=1000)

def test_val_limit_too_big():
    r = client.post("/search", json={"limit": 500}, headers=H)
    assert r.status_code == 422
test("Limit=500 → 422", test_val_limit_too_big, max_ms=1000)

def test_val_negative_offset():
    r = client.post("/search", json={"offset": -1}, headers=H)
    assert r.status_code == 422
test("Negative offset → 422", test_val_negative_offset, max_ms=1000)

def test_val_xss():
    r = client.post("/search", json={"query": "<script>alert(1)</script>"}, headers=H)
    assert r.status_code == 200
    assert "<script>" not in r.text
test("XSS in query → safe", test_val_xss, max_ms=2000)

def test_val_sql_injection():
    r = client.post("/search", json={"query": "'; DROP TABLE papers; --"}, headers=H)
    assert r.status_code == 200
test("SQL injection → harmless", test_val_sql_injection, max_ms=2000)

def test_val_security_headers():
    r = client.get("/health")
    assert r.headers["X-Content-Type-Options"] == "nosniff"
    assert r.headers["X-Frame-Options"] == "DENY"
    assert "max-age" in r.headers["Strict-Transport-Security"]
    assert r.headers["Cache-Control"] == "no-store"
test("Security headers on every response", test_val_security_headers, max_ms=1000)

def test_val_embeddings_not_leaked():
    r = client.post("/search", json={"limit": 1}, headers=H)
    h = r.json()["hits"][0]
    assert "title_embedding" not in h
    assert "abstract_embedding" not in h
test("Embeddings excluded from response", test_val_embeddings_not_leaked, max_ms=2000)


# ════════════════════════════════════════════════════════
# 14. EDGE CASES
# ════════════════════════════════════════════════════════
print("\n═══ 14. EDGE CASES ═══")

def test_edge_unicode():
    r = client.post("/search", json={"query": "Schrödinger equation réseau neuronal"}, headers=H)
    assert r.status_code == 200
test("Unicode: Schrödinger, réseau", test_edge_unicode, max_ms=3000)

def test_edge_cjk():
    r = client.post("/search", json={"query": "量子力学 深度学习"}, headers=H)
    assert r.status_code == 200
test("CJK characters", test_edge_cjk, max_ms=3000)

def test_edge_latex():
    r = client.post("/search", json={"query": "$\\alpha$ convergence \\mathbb{R}^n"}, headers=H)
    assert r.status_code == 200
test("LaTeX in query", test_edge_latex, max_ms=3000)

def test_edge_whitespace():
    r = client.post("/search", json={"query": "   neural   network   "}, headers=H)
    assert r.status_code == 200
    assert r.json()["total"] > 0
test("Extra whitespace normalized", test_edge_whitespace, max_ms=3000)

def test_edge_special_chars():
    r = client.post("/search", json={"query": "O(n log n) bounds"}, headers=H)
    assert r.status_code in (200, 400)
test("Special chars: O(n log n)", test_edge_special_chars, max_ms=3000)

def test_edge_impossible_combo():
    r = client.post("/search", json={
        "categories": ["quant-ph"],
        "has_github": True,
        "min_page_count": 500,
    }, headers=H)
    assert r.json()["total"] == 0
test("Impossible combo → 0", test_edge_impossible_combo, max_ms=3000)

def test_edge_deterministic():
    body = {"query": "neural network", "sort_by": "date", "sort_order": "desc", "limit": 20}
    r1 = client.post("/search", json=body, headers=H)
    r2 = client.post("/search", json=body, headers=H)
    ids1 = [h["arxiv_id"] for h in r1.json()["hits"]]
    ids2 = [h["arxiv_id"] for h in r2.json()["hits"]]
    assert ids1 == ids2
test("Same query → deterministic results", test_edge_deterministic, max_ms=5000)


# ════════════════════════════════════════════════════════
# 15. PERFORMANCE BENCHMARKS (3M scale)
# ════════════════════════════════════════════════════════
print("\n═══ 15. PERFORMANCE BENCHMARKS ═══")

def test_perf_empty():
    r = client.post("/search", json={}, headers=H)
    assert r.json()["took_ms"] < 500
test("match_all < 500ms", test_perf_empty, max_ms=1000)

def test_perf_simple():
    r = client.post("/search", json={"query": "neural network"}, headers=H)
    assert r.json()["took_ms"] < 1000
test("Simple query < 1s", test_perf_simple, max_ms=2000)

def test_perf_complex():
    r = client.post("/search", json={
        "query": "deep learning",
        "categories": ["cs.AI", "cs.LG"],
        "submitted_date": {"gte": "2024-01-01T00:00:00+00:00"},
        "has_github": True,
        "sort_by": "date",
        "highlight": True,
        "limit": 50,
    }, headers=H)
    assert r.json()["took_ms"] < 3000
test("Complex multi-filter < 3s", test_perf_complex, max_ms=5000)

def test_perf_fuzzy():
    r = client.post("/search", json={"fuzzy": "reinforcment lerning"}, headers=H)
    assert r.json()["took_ms"] < 2000
test("Fuzzy search < 2s", test_perf_fuzzy, max_ms=3000)

def test_perf_regex():
    r = client.post("/search", json={"title_regex": ".*[Nn]eural.*"}, headers=H)
    assert r.json()["took_ms"] < 5000
test("Regex search < 5s", test_perf_regex, max_ms=8000)

def test_perf_author_nested():
    r = client.post("/search", json={"author": "Goodfellow"}, headers=H)
    assert r.json()["took_ms"] < 2000
test("Nested author query < 2s", test_perf_author_nested, max_ms=3000)

def test_perf_stats():
    r = client.get("/stats", headers=H)
    # Stats involves aggregations over 3M docs
test("Stats aggregation over 3M", test_perf_stats, max_ms=5000)

def test_perf_pagination_deep():
    r = client.post("/search", json={"offset": 5000, "limit": 100, "sort_by": "date", "sort_order": "desc"}, headers=H)
    assert r.json()["took_ms"] < 3000
test("Deep pagination (offset=5000) < 3s", test_perf_pagination_deep, max_ms=5000)


# ════════════════════════════════════════════════════════
# CLEANUP & RESULTS
# ════════════════════════════════════════════════════════
_cm.__exit__(None, None, None)

print("\n" + "═" * 60)
print(f"  RESULTS: {passed} passed, {failed} failed out of {passed + failed} tests")
print("═" * 60)

if errors:
    print("\n  FAILURES:")
    for name, err in errors:
        print(f"    ✗ {name}: {err}")

# Performance summary
print("\n  PERFORMANCE SUMMARY:")
slow = [(n, ms) for n, ms, s in perf_results if s == "SLOW"]
fast = [(n, ms) for n, ms, s in perf_results if s == "OK"]
if fast:
    avg_ms = sum(ms for _, ms in fast) / len(fast)
    max_test = max(fast, key=lambda x: x[1])
    min_test = min(fast, key=lambda x: x[1])
    print(f"    Avg response: {avg_ms:.0f}ms")
    print(f"    Fastest: {min_test[1]}ms ({min_test[0]})")
    print(f"    Slowest: {max_test[1]}ms ({max_test[0]})")
if slow:
    print(f"    SLOW tests ({len(slow)}):")
    for n, ms in slow:
        print(f"      {ms}ms - {n}")

print()
if failed == 0:
    print("  🟢 ALL TESTS PASSED")
else:
    print(f"  🔴 {failed} TESTS FAILED")

sys.exit(1 if failed else 0)
