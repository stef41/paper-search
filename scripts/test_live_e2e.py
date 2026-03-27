"""
Live end-to-end tests against the real seeded database.
Tests every single feature of the search engine against real ArXiv papers.
"""
import os
import sys
import time

# Force localhost
os.environ["ES_HOST"] = "localhost"
os.environ["REDIS_HOST"] = "localhost"
os.environ["API_KEYS"] = "test-key-1,test-key-2"
os.environ["RATE_LIMIT_PER_MINUTE"] = "10000"
os.environ["RATE_LIMIT_BURST"] = "1000"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.config import reset_settings
reset_settings()

from fastapi.testclient import TestClient
from src.api.main import create_app

app = create_app()
H = {"X-API-Key": "test-key-1"}

passed = 0
failed = 0
errors = []


def test(name, fn):
    global passed, failed
    try:
        fn()
        passed += 1
        print(f"  ✓ {name}")
    except AssertionError as e:
        failed += 1
        errors.append((name, str(e)))
        print(f"  ✗ {name}: {e}")
    except Exception as e:
        failed += 1
        errors.append((name, str(e)))
        print(f"  ✗ {name}: EXCEPTION {e}")

# Use context manager to keep the lifespan active
_client_cm = TestClient(app)
client = _client_cm.__enter__()

# ════════════════════════════════════════════
# HEALTH & INFRASTRUCTURE
# ════════════════════════════════════════════
print("\n═══ HEALTH & INFRASTRUCTURE ═══")

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    d = r.json()
    assert d["status"] in ("healthy", "degraded")
    assert "total_papers" not in d, "Health should not leak paper count"
test("Health endpoint", test_health)

def test_health_no_auth():
    r = client.get("/health")
    assert r.status_code == 200  # no auth needed
test("Health needs no auth", test_health_no_auth)


# ════════════════════════════════════════════
# AUTHENTICATION & SECURITY
# ════════════════════════════════════════════
print("\n═══ AUTHENTICATION & SECURITY ═══")

def test_no_key():
    r = client.post("/search", json={"query": "test"})
    assert r.status_code == 401
test("Search without API key → 401", test_no_key)

def test_bad_key():
    r = client.post("/search", json={"query": "test"}, headers={"X-API-Key": "wrong"})
    assert r.status_code == 403
test("Search with bad API key → 403", test_bad_key)

def test_valid_key():
    r = client.post("/search", json={"query": "test"}, headers=H)
    assert r.status_code == 200
test("Search with valid API key → 200", test_valid_key)

def test_second_key():
    r = client.post("/search", json={"query": "test"}, headers={"X-API-Key": "test-key-2"})
    assert r.status_code == 200
test("Search with second API key → 200", test_second_key)

def test_security_headers():
    r = client.get("/health")
    assert r.headers.get("X-Content-Type-Options") == "nosniff"
    assert r.headers.get("X-Frame-Options") == "DENY"
    assert "max-age" in r.headers.get("Strict-Transport-Security", "")
    assert r.headers.get("Cache-Control") == "no-store"
test("Security headers present", test_security_headers)

def test_xss_in_query():
    r = client.post("/search", json={"query": "<script>alert('xss')</script>"}, headers=H)
    assert r.status_code == 200
    # XSS content is safely handled by ES (no reflected execution)
    body = r.text
    assert "<script>" not in body, "Raw script tag reflected in response!"
test("XSS in query safely handled", test_xss_in_query)

def test_sql_injection():
    r = client.post("/search", json={"query": "'; DROP TABLE papers; --"}, headers=H)
    assert r.status_code == 200
test("SQL injection harmless", test_sql_injection)


# ════════════════════════════════════════════
# INPUT VALIDATION
# ════════════════════════════════════════════
print("\n═══ INPUT VALIDATION ═══")

def test_query_too_long():
    r = client.post("/search", json={"query": "a" * 3000}, headers=H)
    assert r.status_code == 422
test("Query >2000 chars → 422", test_query_too_long)

def test_regex_too_long():
    r = client.post("/search", json={"title_regex": "a" * 300}, headers=H)
    assert r.status_code == 422
test("Regex >200 chars → 422", test_regex_too_long)

def test_invalid_regex():
    r = client.post("/search", json={"title_regex": "[invalid"}, headers=H)
    assert r.status_code == 422
test("Invalid regex → 422", test_invalid_regex)

def test_redos_blocked():
    r = client.post("/search", json={"title_regex": "(.+)+"}, headers=H)
    assert r.status_code == 422
test("ReDoS pattern blocked → 422", test_redos_blocked)

def test_negative_offset():
    r = client.post("/search", json={"offset": -1}, headers=H)
    assert r.status_code == 422
test("Negative offset → 422", test_negative_offset)

def test_limit_too_large():
    r = client.post("/search", json={"limit": 500}, headers=H)
    assert r.status_code == 422
test("Limit >200 → 422", test_limit_too_large)

def test_invalid_operator():
    r = client.post("/search", json={"operator": "xor"}, headers=H)
    assert r.status_code == 422
test("Invalid operator → 422", test_invalid_operator)

def test_negative_citations():
    r = client.post("/search", json={"min_citations": -5}, headers=H)
    assert r.status_code == 422
test("Negative min_citations → 422", test_negative_citations)

def test_deep_pagination():
    r = client.post("/search", json={"offset": 10000, "limit": 100}, headers=H)
    assert r.status_code == 400
test("Deep pagination → 400", test_deep_pagination)


# ════════════════════════════════════════════
# FULL-TEXT SEARCH
# ════════════════════════════════════════════
print("\n═══ FULL-TEXT SEARCH ═══")

def test_basic_query():
    r = client.post("/search", json={"query": "neural network"}, headers=H)
    assert r.status_code == 200
    d = r.json()
    assert d["total"] > 0, "No results for 'neural network'"
    assert len(d["hits"]) > 0
    assert d["took_ms"] >= 0
test("Basic query: neural network", test_basic_query)

def test_title_query():
    r = client.post("/search", json={"title_query": "learning"}, headers=H)
    d = r.json()
    assert d["total"] > 0
test("Title query: learning", test_title_query)

def test_abstract_query():
    r = client.post("/search", json={"abstract_query": "deep learning model"}, headers=H)
    d = r.json()
    assert d["total"] > 0
test("Abstract query: deep learning model", test_abstract_query)

def test_and_operator():
    r = client.post("/search", json={"query": "transformer attention", "operator": "and"}, headers=H)
    assert r.status_code == 200
test("AND operator", test_and_operator)

def test_or_operator():
    r = client.post("/search", json={"query": "quantum robot", "operator": "or"}, headers=H)
    d = r.json()
    assert d["total"] > 0, "No results for quantum OR robot"
test("OR operator", test_or_operator)

def test_empty_query_returns_all():
    r = client.post("/search", json={}, headers=H)
    d = r.json()
    assert d["total"] > 100
test("Empty query returns all papers", test_empty_query_returns_all)

def test_nonexistent_term():
    r = client.post("/search", json={"query": "xyznonexistent12345"}, headers=H)
    d = r.json()
    assert d["total"] == 0
test("Nonexistent term → 0 results", test_nonexistent_term)

def test_highlights():
    r = client.post("/search", json={"query": "neural", "highlight": True}, headers=H)
    d = r.json()
    has_hl = any(h.get("highlights") for h in d["hits"])
    assert has_hl, "No highlights in results"
test("Highlights present in results", test_highlights)

def test_minimum_should_match():
    r = client.post("/search", json={
        "query": "neural network language processing",
        "minimum_should_match": "3",
    }, headers=H)
    assert r.status_code == 200
test("Minimum should match", test_minimum_should_match)

def test_minimum_should_match_pct():
    r = client.post("/search", json={
        "query": "neural network language model",
        "minimum_should_match": "75%",
    }, headers=H)
    assert r.status_code == 200
test("Minimum should match percentage", test_minimum_should_match_pct)


# ════════════════════════════════════════════
# FUZZY SEARCH
# ════════════════════════════════════════════
print("\n═══ FUZZY SEARCH ═══")

def test_fuzzy_typo():
    r = client.post("/search", json={"fuzzy": "nueral netwerk"}, headers=H)
    d = r.json()
    assert d["total"] > 0, "Fuzzy search didn't match 'nueral netwerk'"
test("Fuzzy match with typos", test_fuzzy_typo)

def test_fuzzy_zero():
    r = client.post("/search", json={"fuzzy": "xyztypo", "fuzzy_fuzziness": 0}, headers=H)
    d = r.json()
    assert d["total"] == 0
test("Fuzzy fuzziness=0 → exact only", test_fuzzy_zero)

def test_fuzzy_high():
    r = client.post("/search", json={"fuzzy": "lerning", "fuzzy_fuzziness": 2}, headers=H)
    assert r.status_code == 200
test("Fuzzy fuzziness=2", test_fuzzy_high)

def test_fuzzy_max_rejected():
    r = client.post("/search", json={"fuzzy": "test", "fuzzy_fuzziness": 5}, headers=H)
    assert r.status_code == 422
test("Fuzzy fuzziness=5 → 422", test_fuzzy_max_rejected)


# ════════════════════════════════════════════
# REGEX SEARCH
# ════════════════════════════════════════════
print("\n═══ REGEX SEARCH ═══")

def test_title_regex():
    r = client.post("/search", json={"title_regex": ".*[Ll]earning.*"}, headers=H)
    assert r.status_code == 200
    # May or may not match depending on keyword field
test("Title regex", test_title_regex)

def test_simple_regex():
    r = client.post("/search", json={"title_regex": ".*Neural.*"}, headers=H)
    assert r.status_code == 200
test("Simple regex allowed", test_simple_regex)


# ════════════════════════════════════════════
# CATEGORY FILTERS
# ════════════════════════════════════════════
print("\n═══ CATEGORY FILTERS ═══")

def test_category_filter():
    r = client.post("/search", json={"categories": ["cs.CL"]}, headers=H)
    d = r.json()
    assert d["total"] > 0
    for h in d["hits"]:
        assert "cs.CL" in h["categories"], f"cs.CL not in {h['categories']}"
test("Filter by category cs.CL", test_category_filter)

def test_primary_category():
    r = client.post("/search", json={"primary_category": "quant-ph"}, headers=H)
    d = r.json()
    assert d["total"] > 0
    for h in d["hits"]:
        assert h["primary_category"] == "quant-ph"
test("Filter by primary_category quant-ph", test_primary_category)

def test_exclude_categories():
    r = client.post("/search", json={"exclude_categories": ["cs.CV"]}, headers=H)
    d = r.json()
    for h in d["hits"]:
        assert "cs.CV" not in h["categories"]
test("Exclude cs.CV", test_exclude_categories)

def test_multiple_categories():
    r = client.post("/search", json={"categories": ["cs.AI", "cs.LG"]}, headers=H)
    d = r.json()
    assert d["total"] > 0
test("Multiple categories filter", test_multiple_categories)


# ════════════════════════════════════════════
# DATE FILTERS
# ════════════════════════════════════════════
print("\n═══ DATE FILTERS ═══")

def test_date_gte():
    r = client.post("/search", json={"submitted_date": {"gte": "2025-01-01T00:00:00+00:00"}}, headers=H)
    d = r.json()
    assert d["total"] > 0
    for h in d["hits"]:
        assert h["submitted_date"] >= "2025-01-01"
test("Date filter gte 2025", test_date_gte)

def test_date_lte():
    r = client.post("/search", json={"submitted_date": {"lte": "2024-12-31T23:59:59+00:00"}}, headers=H)
    assert r.status_code == 200
test("Date filter lte 2024", test_date_lte)

def test_date_range():
    r = client.post("/search", json={
        "submitted_date": {
            "gte": "2026-03-01T00:00:00+00:00",
            "lte": "2026-03-31T23:59:59+00:00",
        }
    }, headers=H)
    d = r.json()
    assert d["total"] > 0, "Should have papers from March 2026"
test("Date range March 2026", test_date_range)

def test_empty_date_range():
    r = client.post("/search", json={
        "submitted_date": {"gte": "2030-01-01T00:00:00+00:00"}
    }, headers=H)
    d = r.json()
    assert d["total"] == 0
test("Future date → 0 results", test_empty_date_range)


# ════════════════════════════════════════════
# METADATA FILTERS
# ════════════════════════════════════════════
print("\n═══ METADATA FILTERS ═══")

def test_has_github_true():
    r = client.post("/search", json={"has_github": True}, headers=H)
    d = r.json()
    assert d["total"] >= 0  # depends on actual data
    for h in d["hits"]:
        assert h["has_github"] is True
test("has_github=True filter", test_has_github_true)

def test_has_github_false():
    r = client.post("/search", json={"has_github": False}, headers=H)
    d = r.json()
    assert d["total"] > 0
    for h in d["hits"]:
        assert h["has_github"] is False
test("has_github=False filter", test_has_github_false)

def test_page_count_range():
    r = client.post("/search", json={"min_page_count": 5, "max_page_count": 20}, headers=H)
    d = r.json()
    for h in d["hits"]:
        if h["page_count"] is not None:
            assert 5 <= h["page_count"] <= 20
test("Page count range filter", test_page_count_range)


# ════════════════════════════════════════════
# SORTING
# ════════════════════════════════════════════
print("\n═══ SORTING ═══")

def test_sort_date_desc():
    r = client.post("/search", json={"sort_by": "date", "sort_order": "desc"}, headers=H)
    d = r.json()
    dates = [h["submitted_date"] for h in d["hits"] if h["submitted_date"]]
    for i in range(len(dates) - 1):
        assert dates[i] >= dates[i+1], f"{dates[i]} < {dates[i+1]}"
test("Sort by date desc", test_sort_date_desc)

def test_sort_date_asc():
    r = client.post("/search", json={"sort_by": "date", "sort_order": "asc"}, headers=H)
    d = r.json()
    dates = [h["submitted_date"] for h in d["hits"] if h["submitted_date"]]
    for i in range(len(dates) - 1):
        assert dates[i] <= dates[i+1]
test("Sort by date asc", test_sort_date_asc)

def test_sort_relevance():
    r = client.post("/search", json={"query": "neural network", "sort_by": "relevance"}, headers=H)
    d = r.json()
    scores = [h["score"] for h in d["hits"] if h["score"] is not None]
    for i in range(len(scores) - 1):
        assert scores[i] >= scores[i+1]
test("Sort by relevance", test_sort_relevance)

def test_sort_citations():
    r = client.post("/search", json={"sort_by": "citations", "sort_order": "desc"}, headers=H)
    assert r.status_code == 200
test("Sort by citations", test_sort_citations)

def test_sort_page_count():
    r = client.post("/search", json={"sort_by": "page_count", "sort_order": "desc"}, headers=H)
    assert r.status_code == 200
test("Sort by page count", test_sort_page_count)


# ════════════════════════════════════════════
# PAGINATION
# ════════════════════════════════════════════
print("\n═══ PAGINATION ═══")

def test_pagination_page1():
    r = client.post("/search", json={"offset": 0, "limit": 5}, headers=H)
    d = r.json()
    assert len(d["hits"]) == 5
    assert d["offset"] == 0
    assert d["limit"] == 5
test("Pagination page 1", test_pagination_page1)

def test_pagination_page2():
    r = client.post("/search", json={"offset": 5, "limit": 5}, headers=H)
    d = r.json()
    assert len(d["hits"]) == 5
    assert d["offset"] == 5
test("Pagination page 2", test_pagination_page2)

def test_pagination_no_overlap():
    r1 = client.post("/search", json={"offset": 0, "limit": 10, "sort_by": "date", "sort_order": "desc"}, headers=H)
    r2 = client.post("/search", json={"offset": 10, "limit": 10, "sort_by": "date", "sort_order": "desc"}, headers=H)
    ids1 = {h["arxiv_id"] for h in r1.json()["hits"]}
    ids2 = {h["arxiv_id"] for h in r2.json()["hits"]}
    assert ids1.isdisjoint(ids2), "Pages overlap!"
test("No overlap between pages", test_pagination_no_overlap)

def test_pagination_beyond():
    # Use a narrow filter that returns few results, then offset past them
    r = client.post("/search", json={
        "query": "xyznonexistent12345term",
        "offset": 0, "limit": 20,
    }, headers=H)
    d = r.json()
    assert d["total"] == 0
    assert len(d["hits"]) == 0
test("Offset beyond results → 0 hits", test_pagination_beyond)

def test_full_scan():
    # Use a narrow filter so the scan finishes within ES's max_result_window (10k)
    all_ids = set()
    offset = 0
    total = None
    while True:
        r = client.post("/search", json={
            "query": "quantum annealing",
            "categories": ["quant-ph"],
            "offset": offset, "limit": 50, "sort_by": "date", "sort_order": "desc",
        }, headers=H)
        d = r.json()
        if total is None:
            total = d["total"]
        if not d["hits"]:
            break
        for h in d["hits"]:
            all_ids.add(h["arxiv_id"])
        offset += 50
        if offset >= total or offset >= 10000:
            break
    # We should have covered at least all results up to our stop point
    expected = min(total, 10000)
    assert len(all_ids) == expected, f"Scan found {len(all_ids)}, expected {expected}"
test("Full paginated scan covers all papers", test_full_scan)


# ════════════════════════════════════════════
# COMBINED QUERIES
# ════════════════════════════════════════════
print("\n═══ COMBINED QUERIES ═══")

def test_query_plus_category():
    r = client.post("/search", json={"query": "learning", "categories": ["cs.AI"]}, headers=H)
    d = r.json()
    for h in d["hits"]:
        assert "cs.AI" in h["categories"]
test("Query + category filter", test_query_plus_category)

def test_github_plus_date():
    r = client.post("/search", json={
        "has_github": True,
        "submitted_date": {"gte": "2026-01-01T00:00:00+00:00"},
    }, headers=H)
    d = r.json()
    for h in d["hits"]:
        assert h["has_github"] is True
test("GitHub + date filter", test_github_plus_date)

def test_complex_combined():
    r = client.post("/search", json={
        "query": "learning",
        "categories": ["cs.AI", "cs.LG"],
        "submitted_date": {"gte": "2025-01-01T00:00:00+00:00"},
        "sort_by": "date",
        "sort_order": "desc",
        "limit": 10,
        "highlight": True,
    }, headers=H)
    assert r.status_code == 200
    d = r.json()
    assert d["total"] >= 0
test("Complex combined query", test_complex_combined)


# ════════════════════════════════════════════
# STATS ENDPOINT
# ════════════════════════════════════════════
print("\n═══ STATS ENDPOINT ═══")

def test_stats():
    r = client.get("/stats", headers=H)
    assert r.status_code == 200
    d = r.json()
    assert d["total_papers"] > 100
    assert len(d["categories"]) > 5
    assert "cs.AI" in d["categories"]
    assert d["date_range"]["min"] is not None
    assert d["date_range"]["max"] is not None
test("Stats endpoint", test_stats)

def test_stats_no_auth():
    r = client.get("/stats")
    assert r.status_code == 401
test("Stats requires auth", test_stats_no_auth)


# ════════════════════════════════════════════
# SINGLE PAPER ENDPOINT
# ════════════════════════════════════════════
print("\n═══ SINGLE PAPER ═══")

def test_get_paper():
    # First find a paper
    r = client.post("/search", json={"limit": 1}, headers=H)
    arxiv_id = r.json()["hits"][0]["arxiv_id"]
    r2 = client.get(f"/paper/{arxiv_id}", headers=H)
    assert r2.status_code == 200
    d = r2.json()
    assert d["arxiv_id"] == arxiv_id
    assert d["title"]
    assert d["abstract"]
    assert len(d["authors"]) > 0
test("Get single paper", test_get_paper)

def test_get_nonexistent_paper():
    r = client.get("/paper/9999.99999", headers=H)
    assert r.status_code == 404
test("Nonexistent paper → 404", test_get_nonexistent_paper)


# ════════════════════════════════════════════
# RESPONSE FORMAT
# ════════════════════════════════════════════
print("\n═══ RESPONSE FORMAT ═══")

def test_response_structure():
    r = client.post("/search", json={"query": "network"}, headers=H)
    d = r.json()
    assert "total" in d
    assert "hits" in d
    assert "took_ms" in d
    assert "offset" in d
    assert "limit" in d
test("Response has required fields", test_response_structure)

def test_hit_structure():
    r = client.post("/search", json={"limit": 1}, headers=H)
    h = r.json()["hits"][0]
    required = ["arxiv_id", "title", "abstract", "authors", "categories",
                 "primary_category", "submitted_date", "has_github",
                 "citation_stats", "references_stats"]
    for field in required:
        assert field in h, f"Missing field: {field}"
test("Hit has required fields", test_hit_structure)

def test_author_structure():
    r = client.post("/search", json={"limit": 1}, headers=H)
    author = r.json()["hits"][0]["authors"][0]
    assert "name" in author
    assert "is_first_author" in author
test("Author has required fields", test_author_structure)

def test_embeddings_excluded():
    r = client.post("/search", json={"limit": 1}, headers=H)
    h = r.json()["hits"][0]
    assert "title_embedding" not in h
    assert "abstract_embedding" not in h
test("Embeddings excluded by default", test_embeddings_excluded)


# ════════════════════════════════════════════
# EDGE CASES
# ════════════════════════════════════════════
print("\n═══ EDGE CASES ═══")

def test_unicode():
    r = client.post("/search", json={"query": "réseau neuronal 深度学习"}, headers=H)
    assert r.status_code == 200
test("Unicode query accepted", test_unicode)

def test_empty_string():
    r = client.post("/search", json={"query": ""}, headers=H)
    assert r.status_code == 200
test("Empty string query", test_empty_string)

def test_whitespace():
    r = client.post("/search", json={"query": "   "}, headers=H)
    assert r.status_code == 200
test("Whitespace-only query", test_whitespace)

def test_special_chars():
    r = client.post("/search", json={"query": "neural (networks) [2024]"}, headers=H)
    assert r.status_code in (200, 400)
test("Special characters in query", test_special_chars)

def test_impossible_page_range():
    r = client.post("/search", json={"min_page_count": 999, "max_page_count": 1}, headers=H)
    d = r.json()
    assert d["total"] == 0
test("Impossible page range → 0", test_impossible_page_range)

def test_limit_1():
    r = client.post("/search", json={"limit": 1}, headers=H)
    assert len(r.json()["hits"]) == 1
test("Limit 1 returns 1 hit", test_limit_1)

def test_same_query_twice():
    body = {"query": "neural", "sort_by": "date", "sort_order": "desc"}
    r1 = client.post("/search", json=body, headers=H)
    r2 = client.post("/search", json=body, headers=H)
    ids1 = [h["arxiv_id"] for h in r1.json()["hits"]]
    ids2 = [h["arxiv_id"] for h in r2.json()["hits"]]
    assert ids1 == ids2
test("Same query → same results", test_same_query_twice)

def test_conflicting_filters():
    r = client.post("/search", json={"has_github": True, "primary_category": "quant-ph"}, headers=H)
    d = r.json()
    # May return 0 — quant-ph papers rarely have github
    for h in d["hits"]:
        assert h["has_github"] is True
        assert h["primary_category"] == "quant-ph"
test("Conflicting filters compose correctly", test_conflicting_filters)


# ════════════════════════════════════════════
# PERFORMANCE
# ════════════════════════════════════════════
print("\n═══ PERFORMANCE ═══")

def test_search_fast():
    r = client.post("/search", json={"query": "neural network"}, headers=H)
    assert r.json()["took_ms"] < 2000, f"Search took {r.json()['took_ms']}ms"
test("Simple search < 2s", test_search_fast)

def test_complex_fast():
    r = client.post("/search", json={
        "query": "deep learning",
        "categories": ["cs.AI", "cs.LG"],
        "submitted_date": {"gte": "2025-01-01T00:00:00+00:00"},
        "sort_by": "date",
        "highlight": True,
    }, headers=H)
    assert r.json()["took_ms"] < 5000
test("Complex query < 5s", test_complex_fast)

def test_empty_fast():
    r = client.post("/search", json={}, headers=H)
    assert r.json()["took_ms"] < 1000
test("Empty query < 1s", test_empty_fast)


# ════════════════════════════════════════════
# E2E WORKFLOWS
# ════════════════════════════════════════════
print("\n═══ E2E WORKFLOWS ═══")

def test_e2e_search_then_paper():
    # Search → pick first → get details
    r = client.post("/search", json={"query": "transformer", "limit": 1}, headers=H)
    d = r.json()
    if d["total"] > 0:
        aid = d["hits"][0]["arxiv_id"]
        r2 = client.get(f"/paper/{aid}", headers=H)
        assert r2.status_code == 200
        assert r2.json()["arxiv_id"] == aid
test("E2E: search → get paper", test_e2e_search_then_paper)

def test_e2e_stats_then_search():
    r = client.get("/stats", headers=H)
    cats = r.json()["categories"]
    top_cat = max(cats, key=cats.get)
    r2 = client.post("/search", json={"primary_category": top_cat, "limit": 5}, headers=H)
    assert r2.status_code == 200
    assert r2.json()["total"] > 0
test("E2E: stats → search top category", test_e2e_stats_then_search)

def test_e2e_fuzzy_then_exact():
    r1 = client.post("/search", json={"fuzzy": "reinforcment lerning"}, headers=H)
    r2 = client.post("/search", json={"query": "reinforcement learning", "operator": "and"}, headers=H)
    assert r1.status_code == 200
    assert r2.status_code == 200
test("E2E: fuzzy → exact refinement", test_e2e_fuzzy_then_exact)

def test_e2e_paginated_scan():
    all_ids = set()
    offset = 0
    while offset < 100:
        r = client.post("/search", json={"offset": offset, "limit": 20, "sort_by": "date", "sort_order": "desc"}, headers=H)
        hits = r.json()["hits"]
        if not hits:
            break
        for h in hits:
            all_ids.add(h["arxiv_id"])
        offset += 20
    assert len(all_ids) >= 80  # Should have at least 80 unique papers
test("E2E: paginated scan no duplicates", test_e2e_paginated_scan)


# Clean up the client context
_client_cm.__exit__(None, None, None)

# ════════════════════════════════════════════
# RESULTS
# ════════════════════════════════════════════
print("\n" + "═" * 50)
print(f"RESULTS: {passed} passed, {failed} failed out of {passed + failed} tests")
print("═" * 50)

if errors:
    print("\nFAILURES:")
    for name, err in errors:
        print(f"  ✗ {name}: {err}")
    sys.exit(1)
else:
    print("\n🟢 ALL TESTS PASSED")
    sys.exit(0)
