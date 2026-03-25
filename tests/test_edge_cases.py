"""Edge case and stress tests."""
from __future__ import annotations

import pytest
from tests.conftest import auth_headers


class TestEdgeCases:
    """Test unusual inputs and boundary conditions."""

    @pytest.mark.edge
    @pytest.mark.integration
    def test_empty_string_query(self, client):
        resp = client.post(
            "/search",
            json={"query": ""},
            headers=auth_headers(),
        )
        assert resp.status_code == 200

    @pytest.mark.edge
    @pytest.mark.integration
    def test_whitespace_only_query(self, client):
        resp = client.post(
            "/search",
            json={"query": "   "},
            headers=auth_headers(),
        )
        assert resp.status_code == 200

    @pytest.mark.edge
    @pytest.mark.integration
    def test_special_characters_in_query(self, client):
        resp = client.post(
            "/search",
            json={"query": "neural (networks) [2024] {test}"},
            headers=auth_headers(),
        )
        # Should not crash
        assert resp.status_code in (200, 400)

    @pytest.mark.edge
    @pytest.mark.integration
    def test_unicode_query(self, client):
        resp = client.post(
            "/search",
            json={"query": "réseau neuronal 深度学习 ニューラル"},
            headers=auth_headers(),
        )
        assert resp.status_code == 200

    @pytest.mark.edge
    @pytest.mark.integration
    def test_html_in_query(self, client):
        resp = client.post(
            "/search",
            json={"query": "<script>alert('xss')</script>"},
            headers=auth_headers(),
        )
        assert resp.status_code == 200
        data = resp.json()
        # Should not return anything that echoes back HTML
        assert data["total"] == 0

    @pytest.mark.edge
    @pytest.mark.integration
    def test_sql_injection_in_query(self, client):
        resp = client.post(
            "/search",
            json={"query": "'; DROP TABLE papers; --"},
            headers=auth_headers(),
        )
        assert resp.status_code == 200
        assert resp.json()["total"] == 0

    @pytest.mark.edge
    @pytest.mark.integration
    def test_very_long_author_name(self, client):
        resp = client.post(
            "/search",
            json={"author": "A" * 200},
            headers=auth_headers(),
        )
        assert resp.status_code == 200
        assert resp.json()["total"] == 0

    @pytest.mark.edge
    @pytest.mark.integration
    def test_limit_one(self, client):
        resp = client.post(
            "/search",
            json={"limit": 1},
            headers=auth_headers(),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["hits"]) == 1

    @pytest.mark.edge
    @pytest.mark.integration
    def test_limit_max(self, client):
        resp = client.post(
            "/search",
            json={"limit": 200},
            headers=auth_headers(),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["hits"]) <= 200

    @pytest.mark.edge
    @pytest.mark.integration
    def test_offset_zero_limit_one(self, client):
        resp = client.post(
            "/search",
            json={"offset": 0, "limit": 1, "sort_by": "date", "sort_order": "desc"},
            headers=auth_headers(),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["hits"]) == 1

    @pytest.mark.edge
    @pytest.mark.integration
    def test_conflicting_has_github_with_github_false(self, client):
        """Query for github=True + category that has no github papers."""
        resp = client.post(
            "/search",
            json={"has_github": True, "primary_category": "quant-ph"},
            headers=auth_headers(),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0

    @pytest.mark.edge
    @pytest.mark.integration
    def test_impossible_citation_range(self, client):
        resp = client.post(
            "/search",
            json={"min_citations": 1000000},
            headers=auth_headers(),
        )
        assert resp.status_code == 200
        assert resp.json()["total"] == 0

    @pytest.mark.edge
    @pytest.mark.integration
    def test_impossible_page_range(self, client):
        """min > max should return no results."""
        resp = client.post(
            "/search",
            json={"min_page_count": 100, "max_page_count": 5},
            headers=auth_headers(),
        )
        assert resp.status_code == 200
        assert resp.json()["total"] == 0

    @pytest.mark.edge
    @pytest.mark.integration
    def test_empty_categories_list(self, client):
        resp = client.post(
            "/search",
            json={"categories": []},
            headers=auth_headers(),
        )
        # Empty list should be treated as no filter
        assert resp.status_code == 200

    @pytest.mark.edge
    @pytest.mark.integration
    def test_both_title_and_abstract_query(self, client):
        resp = client.post(
            "/search",
            json={
                "title_query": "neural",
                "abstract_query": "deep learning",
            },
            headers=auth_headers(),
        )
        assert resp.status_code == 200

    @pytest.mark.edge
    @pytest.mark.integration
    def test_multiple_regex_filters(self, client):
        resp = client.post(
            "/search",
            json={
                "title_regex": ".*Neural.*",
                "author_regex": ".*Smith.*",
            },
            headers=auth_headers(),
        )
        assert resp.status_code == 200

    @pytest.mark.edge
    @pytest.mark.integration
    def test_sort_with_no_results(self, client):
        resp = client.post(
            "/search",
            json={
                "query": "xyznonexistent",
                "sort_by": "citations",
                "sort_order": "desc",
            },
            headers=auth_headers(),
        )
        assert resp.status_code == 200
        assert resp.json()["total"] == 0

    @pytest.mark.edge
    @pytest.mark.integration
    def test_repeated_same_query(self, client):
        """Same query twice should return same results."""
        body = {"query": "neural networks", "sort_by": "date", "sort_order": "desc"}
        resp1 = client.post("/search", json=body, headers=auth_headers())
        resp2 = client.post("/search", json=body, headers=auth_headers())
        data1 = resp1.json()
        data2 = resp2.json()
        assert data1["total"] == data2["total"]
        assert [h["arxiv_id"] for h in data1["hits"]] == [h["arxiv_id"] for h in data2["hits"]]


class TestPerformance:
    """Basic performance tests."""

    @pytest.mark.integration
    def test_search_is_fast(self, client):
        """Simple search should respond in < 2 seconds."""
        resp = client.post(
            "/search",
            json={"query": "neural networks"},
            headers=auth_headers(),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["took_ms"] < 2000

    @pytest.mark.integration
    def test_complex_query_is_fast(self, client):
        """Complex multi-filter query should respond in < 5 seconds."""
        resp = client.post(
            "/search",
            json={
                "query": "deep learning",
                "categories": ["cs.AI", "cs.LG"],
                "has_github": True,
                "min_citations": 10,
                "min_page_count": 5,
                "submitted_date": {"gte": "2023-01-01T00:00:00+00:00"},
                "sort_by": "citations",
                "sort_order": "desc",
                "highlight": True,
            },
            headers=auth_headers(),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["took_ms"] < 5000

    @pytest.mark.integration
    def test_empty_search_is_fast(self, client):
        resp = client.post("/search", json={}, headers=auth_headers())
        assert resp.status_code == 200
        data = resp.json()
        assert data["took_ms"] < 1000
