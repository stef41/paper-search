"""Tests for sorting, pagination, combined queries, and response format."""
from __future__ import annotations

import pytest
from tests.conftest import auth_headers


class TestSorting:
    """Test sort-by functionality."""

    @pytest.mark.integration
    def test_sort_by_date_desc(self, client):
        resp = client.post(
            "/search",
            json={"sort_by": "date", "sort_order": "desc"},
            headers=auth_headers(),
        )
        assert resp.status_code == 200
        data = resp.json()
        dates = [h["submitted_date"] for h in data["hits"] if h["submitted_date"]]
        assert len(dates) >= 2, "Need at least 2 papers with dates to verify sort order"
        for i in range(len(dates) - 1):
            assert dates[i] >= dates[i + 1]

    @pytest.mark.integration
    def test_sort_by_date_asc(self, client):
        resp = client.post(
            "/search",
            json={"sort_by": "date", "sort_order": "asc"},
            headers=auth_headers(),
        )
        assert resp.status_code == 200
        data = resp.json()
        dates = [h["submitted_date"] for h in data["hits"] if h["submitted_date"]]
        assert len(dates) >= 2, "Need at least 2 papers with dates to verify sort order"
        for i in range(len(dates) - 1):
            assert dates[i] <= dates[i + 1]

    @pytest.mark.integration
    def test_sort_by_citations(self, client):
        resp = client.post(
            "/search",
            json={"sort_by": "citations", "sort_order": "desc"},
            headers=auth_headers(),
        )
        assert resp.status_code == 200
        data = resp.json()
        citations = [h["citation_stats"]["total_citations"] for h in data["hits"]]
        assert len(citations) >= 2, "Need at least 2 papers to verify sort order"
        for i in range(len(citations) - 1):
            assert citations[i] >= citations[i + 1]

    @pytest.mark.integration
    def test_sort_by_h_index(self, client):
        resp = client.post(
            "/search",
            json={"sort_by": "h_index", "sort_order": "desc"},
            headers=auth_headers(),
        )
        assert resp.status_code == 200
        data = resp.json()
        h_indices = [
            h["first_author_h_index"] for h in data["hits"]
            if h.get("first_author_h_index") is not None
        ]
        assert len(h_indices) >= 2, "Need at least 2 papers with h_index to verify sort order"
        for i in range(len(h_indices) - 1):
            assert h_indices[i] >= h_indices[i + 1]

    @pytest.mark.integration
    def test_sort_by_page_count(self, client):
        resp = client.post(
            "/search",
            json={"sort_by": "page_count", "sort_order": "desc"},
            headers=auth_headers(),
        )
        assert resp.status_code == 200
        data = resp.json()
        pages = [h["page_count"] for h in data["hits"] if h["page_count"]]
        assert len(pages) >= 2, "Need at least 2 papers with page_count to verify sort order"
        for i in range(len(pages) - 1):
            assert pages[i] >= pages[i + 1]

    @pytest.mark.integration
    def test_sort_by_relevance(self, client):
        resp = client.post(
            "/search",
            json={"query": "networks", "sort_by": "relevance"},
            headers=auth_headers(),
        )
        assert resp.status_code == 200
        data = resp.json()
        scores = [h["score"] for h in data["hits"] if h["score"] is not None]
        assert len(scores) >= 2, "Need at least 2 papers with scores to verify sort order"
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1]

    @pytest.mark.integration
    def test_sort_by_updated(self, client):
        resp = client.post(
            "/search",
            json={"sort_by": "updated", "sort_order": "desc"},
            headers=auth_headers(),
        )
        assert resp.status_code == 200
        data = resp.json()
        dates = [h["updated_date"] for h in data["hits"] if h.get("updated_date")]
        assert len(dates) >= 2, "Need at least 2 papers with updated_date to verify sort order"
        for i in range(len(dates) - 1):
            assert dates[i] >= dates[i + 1]


class TestPagination:
    """Test pagination."""

    @pytest.mark.integration
    def test_pagination_first_page(self, client):
        resp = client.post(
            "/search",
            json={"offset": 0, "limit": 2},
            headers=auth_headers(),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["offset"] == 0
        assert data["limit"] == 2
        assert len(data["hits"]) == 2

    @pytest.mark.integration
    def test_pagination_second_page(self, client):
        resp = client.post(
            "/search",
            json={"offset": 2, "limit": 2},
            headers=auth_headers(),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["offset"] == 2
        assert len(data["hits"]) == 2

    @pytest.mark.integration
    def test_pagination_beyond_results(self, client):
        resp = client.post(
            "/search",
            json={"offset": 1000, "limit": 20},
            headers=auth_headers(),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["hits"]) == 0

    @pytest.mark.integration
    def test_pagination_consistency(self, client):
        """All papers from paginated results should cover total results."""
        # First, get total paper count
        first_resp = client.post(
            "/search",
            json={"offset": 0, "limit": 1, "sort_by": "date", "sort_order": "desc"},
            headers=auth_headers(),
        )
        total_papers = first_resp.json()["total"]

        all_ids = set()
        for offset in range(0, total_papers + 2, 2):
            resp = client.post(
                "/search",
                json={"offset": offset, "limit": 2, "sort_by": "date", "sort_order": "desc"},
                headers=auth_headers(),
            )
            data = resp.json()
            for hit in data["hits"]:
                all_ids.add(hit["arxiv_id"])

        assert len(all_ids) == total_papers


class TestCombinedQueries:
    """Test combining multiple filters."""

    @pytest.mark.integration
    def test_query_with_category_and_date(self, client):
        resp = client.post(
            "/search",
            json={
                "query": "neural",
                "categories": ["cs.CL"],
                "submitted_date": {"gte": "2024-01-01T00:00:00+00:00"},
            },
            headers=auth_headers(),
        )
        assert resp.status_code == 200
        data = resp.json()
        for hit in data["hits"]:
            assert "cs.CL" in hit["categories"]

    @pytest.mark.integration
    def test_query_with_github_and_citations(self, client):
        resp = client.post(
            "/search",
            json={
                "has_github": True,
                "min_citations": 100,
            },
            headers=auth_headers(),
        )
        assert resp.status_code == 200
        data = resp.json()
        for hit in data["hits"]:
            assert hit["has_github"] is True
            assert hit["citation_stats"]["total_citations"] >= 100

    @pytest.mark.integration
    def test_query_with_author_and_h_index(self, client):
        resp = client.post(
            "/search",
            json={
                "min_h_index": 40,
                "categories": ["cs.LG"],
            },
            headers=auth_headers(),
        )
        assert resp.status_code == 200

    @pytest.mark.integration
    def test_complex_combined_query(self, client):
        resp = client.post(
            "/search",
            json={
                "query": "learning",
                "categories": ["cs.LG", "cs.AI"],
                "has_github": True,
                "min_citations": 50,
                "submitted_date": {"gte": "2024-01-01T00:00:00+00:00"},
                "sort_by": "citations",
                "sort_order": "desc",
                "limit": 10,
            },
            headers=auth_headers(),
        )
        assert resp.status_code == 200
        data = resp.json()
        for hit in data["hits"]:
            assert hit["has_github"] is True
            assert hit["citation_stats"]["total_citations"] >= 50

    @pytest.mark.integration
    def test_all_filters_simultaneously(self, client):
        """Apply many filters at once to ensure they compose correctly."""
        resp = client.post(
            "/search",
            json={
                "query": "network",
                "categories": ["cs.AI"],
                "has_github": True,
                "min_page_count": 5,
                "max_page_count": 100,
                "min_citations": 1,
                "submitted_date": {"gte": "2023-01-01T00:00:00+00:00"},
                "sort_by": "date",
                "sort_order": "desc",
                "offset": 0,
                "limit": 50,
                "highlight": True,
            },
            headers=auth_headers(),
        )
        assert resp.status_code == 200


class TestResponseFormat:
    """Test response structure and format."""

    @pytest.mark.integration
    def test_response_structure(self, client):
        resp = client.post("/search", json={"query": "neural"}, headers=auth_headers())
        data = resp.json()
        assert "total" in data
        assert "hits" in data
        assert "took_ms" in data
        assert "offset" in data
        assert "limit" in data

    @pytest.mark.integration
    def test_hit_structure(self, client):
        resp = client.post("/search", json={}, headers=auth_headers())
        data = resp.json()
        assert len(data["hits"]) > 0
        hit = data["hits"][0]
        assert "arxiv_id" in hit
        assert "title" in hit
        assert "abstract" in hit
        assert "authors" in hit
        assert "categories" in hit
        assert "submitted_date" in hit
        assert "citation_stats" in hit
        assert "references_stats" in hit
        assert "has_github" in hit

    @pytest.mark.integration
    def test_author_structure(self, client):
        resp = client.post("/search", json={}, headers=auth_headers())
        data = resp.json()
        for hit in data["hits"]:
            for author in hit["authors"]:
                assert "name" in author
                assert "is_first_author" in author

    @pytest.mark.integration
    def test_embeddings_excluded_by_default(self, client):
        resp = client.post("/search", json={"query": "neural"}, headers=auth_headers())
        data = resp.json()
        for hit in data["hits"]:
            assert "title_embedding" not in hit
            assert "abstract_embedding" not in hit

    @pytest.mark.integration
    def test_citation_stats_structure(self, client):
        resp = client.post("/search", json={}, headers=auth_headers())
        data = resp.json()
        for hit in data["hits"]:
            cs = hit["citation_stats"]
            assert "total_citations" in cs
            assert "top_citing_categories" in cs

    @pytest.mark.integration
    def test_github_urls_included(self, client):
        resp = client.post(
            "/search",
            json={"has_github": True},
            headers=auth_headers(),
        )
        data = resp.json()
        for hit in data["hits"]:
            assert len(hit["github_urls"]) > 0
