"""End-to-end tests for the full workflow."""
from __future__ import annotations

import pytest
from tests.conftest import auth_headers, SAMPLE_PAPERS


class TestE2ESearchWorkflows:
    """Full end-to-end search workflows."""

    @pytest.mark.e2e
    @pytest.mark.integration
    def test_e2e_find_highly_cited_ml_papers_with_code(self, client):
        """Agent scenario: find well-cited ML papers with code."""
        resp = client.post(
            "/search",
            json={
                "categories": ["cs.LG"],
                "has_github": True,
                "min_citations": 100,
                "sort_by": "citations",
                "sort_order": "desc",
            },
            headers=auth_headers(),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] >= 1
        # Most cited should be first
        if len(data["hits"]) > 1:
            assert (
                data["hits"][0]["citation_stats"]["total_citations"]
                >= data["hits"][1]["citation_stats"]["total_citations"]
            )

    @pytest.mark.e2e
    @pytest.mark.integration
    def test_e2e_find_papers_by_prolific_author(self, client):
        """Agent scenario: find papers by high h-index first authors."""
        resp = client.post(
            "/search",
            json={
                "min_first_author_h_index": 50,
                "sort_by": "h_index",
                "sort_order": "desc",
            },
            headers=auth_headers(),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] >= 1
        for hit in data["hits"]:
            assert hit["first_author_h_index"] >= 50

    @pytest.mark.e2e
    @pytest.mark.integration
    def test_e2e_recent_nlp_papers(self, client):
        """Agent scenario: find recent NLP papers."""
        resp = client.post(
            "/search",
            json={
                "primary_category": "cs.CL",
                "submitted_date": {"gte": "2024-01-01T00:00:00+00:00"},
                "sort_by": "date",
                "sort_order": "desc",
            },
            headers=auth_headers(),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] >= 1
        for hit in data["hits"]:
            assert hit["primary_category"] == "cs.CL"

    @pytest.mark.e2e
    @pytest.mark.integration
    def test_e2e_short_papers_without_github(self, client):
        """Agent scenario: find short papers without code."""
        resp = client.post(
            "/search",
            json={
                "has_github": False,
                "max_page_count": 15,
                "sort_by": "date",
                "sort_order": "desc",
            },
            headers=auth_headers(),
        )
        assert resp.status_code == 200
        data = resp.json()
        for hit in data["hits"]:
            assert hit["has_github"] is False
            assert hit["page_count"] <= 15

    @pytest.mark.e2e
    @pytest.mark.integration
    def test_e2e_search_then_get_paper(self, client):
        """Agent scenario: search, then fetch specific paper details."""
        # Step 1: Search
        resp = client.post(
            "/search",
            json={"query": "transformer", "limit": 1},
            headers=auth_headers(),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] >= 1
        arxiv_id = data["hits"][0]["arxiv_id"]

        # Step 2: Get full paper
        resp2 = client.get(f"/paper/{arxiv_id}", headers=auth_headers())
        assert resp2.status_code == 200
        paper = resp2.json()
        assert paper["arxiv_id"] == arxiv_id

    @pytest.mark.e2e
    @pytest.mark.integration
    def test_e2e_check_stats_then_search(self, client):
        """Agent scenario: check stats, then search based on findings."""
        # Step 1: Get stats
        resp = client.get("/stats", headers=auth_headers())
        assert resp.status_code == 200
        stats = resp.json()
        assert stats["total_papers"] >= 6

        # Step 2: Search in popular category
        if "cs.AI" in stats["categories"]:
            resp2 = client.post(
                "/search",
                json={"categories": ["cs.AI"]},
                headers=auth_headers(),
            )
            assert resp2.status_code == 200

    @pytest.mark.e2e
    @pytest.mark.integration
    def test_e2e_paginated_full_scan(self, client):
        """Agent scenario: paginate through all results."""
        all_ids = []
        offset = 0
        limit = 2

        while True:
            resp = client.post(
                "/search",
                json={
                    "offset": offset,
                    "limit": limit,
                    "sort_by": "date",
                    "sort_order": "desc",
                },
                headers=auth_headers(),
            )
            assert resp.status_code == 200
            data = resp.json()
            if not data["hits"]:
                break
            all_ids.extend([h["arxiv_id"] for h in data["hits"]])
            offset += limit
            if offset >= data["total"]:
                break

        assert len(all_ids) == len(SAMPLE_PAPERS)
        assert len(set(all_ids)) == len(SAMPLE_PAPERS)  # No duplicates

    @pytest.mark.e2e
    @pytest.mark.integration
    def test_e2e_fuzzy_then_exact(self, client):
        """Agent scenario: first try fuzzy, then narrow with exact."""
        # Step 1: Fuzzy search with typo
        resp = client.post(
            "/search",
            json={"fuzzy": "reinforcment lerning"},
            headers=auth_headers(),
        )
        assert resp.status_code == 200
        data1 = resp.json()

        # Step 2: Exact search with correct spelling
        resp2 = client.post(
            "/search",
            json={"query": "reinforcement learning", "operator": "and"},
            headers=auth_headers(),
        )
        assert resp2.status_code == 200
        data2 = resp2.json()
        assert data2["total"] >= 1

    @pytest.mark.e2e
    @pytest.mark.integration
    def test_e2e_exclude_and_include_categories(self, client):
        """Agent scenario: include some categories, exclude others."""
        resp = client.post(
            "/search",
            json={
                "categories": ["cs.AI"],
                "exclude_categories": ["cs.CV"],
            },
            headers=auth_headers(),
        )
        assert resp.status_code == 200
        data = resp.json()
        for hit in data["hits"]:
            assert "cs.CV" not in hit["categories"]

    @pytest.mark.e2e
    @pytest.mark.integration
    def test_e2e_health_before_operations(self, client):
        """Agent scenario: check health first, then proceed."""
        resp = client.get("/health")
        assert resp.status_code == 200
        health = resp.json()
        assert health["status"] in ("healthy", "degraded")

        if health["status"] == "healthy":
            resp2 = client.post(
                "/search",
                json={"query": "neural"},
                headers=auth_headers(),
            )
            assert resp2.status_code == 200
