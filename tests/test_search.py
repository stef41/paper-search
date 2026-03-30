"""Tests for full-text, fuzzy, and regex search."""
from __future__ import annotations

import pytest
from tests.conftest import auth_headers


class TestFullTextSearch:
    """Test full-text search capabilities."""

    @pytest.mark.integration
    def test_basic_query(self, client):
        resp = client.post("/search", json={"query": "neural networks"}, headers=auth_headers())
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] >= 1
        assert len(data["hits"]) >= 1
        assert data["took_ms"] >= 0

    @pytest.mark.integration
    def test_title_query(self, client):
        resp = client.post(
            "/search",
            json={"title_query": "Transformer Architecture"},
            headers=auth_headers(),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] >= 1
        titles = [h["title"] for h in data["hits"]]
        assert any("Transformer" in t for t in titles)

    @pytest.mark.integration
    def test_abstract_query(self, client):
        resp = client.post(
            "/search",
            json={"abstract_query": "reinforcement learning human feedback"},
            headers=auth_headers(),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] >= 1

    @pytest.mark.integration
    def test_query_with_and_operator(self, client):
        resp = client.post(
            "/search",
            json={"query": "neural language", "operator": "and"},
            headers=auth_headers(),
        )
        assert resp.status_code == 200
        data = resp.json()
        # Should match only papers with BOTH neural AND language
        assert data["total"] >= 1

    @pytest.mark.integration
    def test_query_with_or_operator(self, client):
        resp = client.post(
            "/search",
            json={"query": "quantum transformer", "operator": "or"},
            headers=auth_headers(),
        )
        assert resp.status_code == 200
        data = resp.json()
        # Should match papers with quantum OR transformer
        assert data["total"] >= 2

    @pytest.mark.integration
    def test_empty_query_returns_all(self, client):
        resp = client.post("/search", json={}, headers=auth_headers())
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] >= 6

    @pytest.mark.integration
    def test_no_results_query(self, client):
        resp = client.post(
            "/search",
            json={"query": "xyznonexistentterm12345"},
            headers=auth_headers(),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0
        assert len(data["hits"]) == 0

    @pytest.mark.integration
    def test_highlights_in_results(self, client):
        resp = client.post(
            "/search",
            json={"query": "neural networks", "highlight": True},
            headers=auth_headers(),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] >= 1
        # Check at least one hit has highlights
        has_highlight = any(h.get("highlights") for h in data["hits"])
        assert has_highlight

    @pytest.mark.integration
    def test_minimum_should_match(self, client):
        resp = client.post(
            "/search",
            json={
                "query": "neural network language processing reinforcement",
                "minimum_should_match": "3",
            },
            headers=auth_headers(),
        )
        assert resp.status_code == 200

    @pytest.mark.integration
    def test_minimum_should_match_percentage(self, client):
        resp = client.post(
            "/search",
            json={
                "query": "neural network language processing",
                "minimum_should_match": "75%",
            },
            headers=auth_headers(),
        )
        assert resp.status_code == 200


class TestFuzzySearch:
    """Test fuzzy matching functionality."""

    @pytest.mark.integration
    def test_fuzzy_match_typo(self, client):
        # "nueral" is a typo for "neural"
        resp = client.post(
            "/search",
            json={"fuzzy": "nueral networks"},
            headers=auth_headers(),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] >= 1

    @pytest.mark.integration
    def test_fuzzy_match_custom_fuzziness(self, client):
        resp = client.post(
            "/search",
            json={"fuzzy": "transformr", "fuzzy_fuzziness": 2},
            headers=auth_headers(),
        )
        assert resp.status_code == 200

    @pytest.mark.integration
    def test_fuzzy_zero_fuzziness(self, client):
        resp = client.post(
            "/search",
            json={"fuzzy": "nueral", "fuzzy_fuzziness": 0},
            headers=auth_headers(),
        )
        assert resp.status_code == 200
        # Zero fuzziness = exact match, so "nueral" should return 0
        data = resp.json()
        assert data["total"] == 0

    @pytest.mark.integration
    def test_fuzzy_fuzziness_validation(self, client):
        resp = client.post(
            "/search",
            json={"fuzzy": "test", "fuzzy_fuzziness": 5},
            headers=auth_headers(),
        )
        assert resp.status_code == 422  # Max fuzziness is 2 (ES limit)


class TestRegexSearch:
    """Test regex search functionality."""

    @pytest.mark.integration
    def test_title_regex(self, client):
        resp = client.post(
            "/search",
            json={"title_regex": ".*[Nn]eural.*"},
            headers=auth_headers(),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] >= 1

    @pytest.mark.integration
    def test_author_regex(self, client):
        resp = client.post(
            "/search",
            json={"author_regex": ".*Smith.*"},
            headers=auth_headers(),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] >= 1

    @pytest.mark.integration
    def test_invalid_regex_rejected(self, client):
        resp = client.post(
            "/search",
            json={"title_regex": "[unclosed"},
            headers=auth_headers(),
        )
        assert resp.status_code == 422

    @pytest.mark.integration
    def test_redos_pattern_blocked(self, client):
        resp = client.post(
            "/search",
            json={"title_regex": "(.+)+"},
            headers=auth_headers(),
        )
        assert resp.status_code == 422

    @pytest.mark.integration
    def test_simple_regex_allowed(self, client):
        resp = client.post(
            "/search",
            json={"title_regex": "Neural.*"},
            headers=auth_headers(),
        )
        assert resp.status_code == 200
