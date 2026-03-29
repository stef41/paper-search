"""Tests for filtering: authors, citations, categories, dates, metadata."""
from __future__ import annotations

import pytest
from tests.conftest import auth_headers


class TestAuthorFilters:
    """Test author-related filtering."""

    @pytest.mark.integration
    def test_filter_by_author(self, client):
        resp = client.post(
            "/search",
            json={"author": "John Smith"},
            headers=auth_headers(),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] >= 1
        # Verify John Smith is in results
        for hit in data["hits"]:
            author_names = [a["name"] for a in hit["authors"]]
            assert any("Smith" in n for n in author_names)

    @pytest.mark.integration
    def test_filter_by_first_author(self, client):
        resp = client.post(
            "/search",
            json={"first_author": "Alice Johnson"},
            headers=auth_headers(),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] >= 1
        for hit in data["hits"]:
            first_authors = [a["name"] for a in hit["authors"] if a["is_first_author"]]
            assert any("Johnson" in n for n in first_authors)

    @pytest.mark.integration
    def test_filter_by_min_h_index(self, client):
        resp = client.post(
            "/search",
            json={"min_h_index": 50},
            headers=auth_headers(),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] >= 1
        # Papers should have at least one author with h_index >= 50
        for hit in data["hits"]:
            h_indices = [a.get("h_index") for a in hit["authors"] if a.get("h_index") is not None]
            assert any(h >= 50 for h in h_indices), f"No author with h_index >= 50 in {hit['arxiv_id']}"

    @pytest.mark.integration
    def test_filter_by_max_h_index(self, client):
        resp = client.post(
            "/search",
            json={"max_h_index": 10},
            headers=auth_headers(),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] >= 1
        for hit in data["hits"]:
            h_indices = [a.get("h_index") for a in hit["authors"] if a.get("h_index") is not None]
            assert any(h <= 10 for h in h_indices), f"No author with h_index <= 10 in {hit['arxiv_id']}"

    @pytest.mark.integration
    def test_filter_by_first_author_h_index(self, client):
        resp = client.post(
            "/search",
            json={"min_first_author_h_index": 60},
            headers=auth_headers(),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] >= 1    # Alice (60) and Grace (72)

    @pytest.mark.integration
    def test_filter_nonexistent_author(self, client):
        resp = client.post(
            "/search",
            json={"author": "Nonexistent Person XYZ"},
            headers=auth_headers(),
        )
        assert resp.status_code == 200
        assert resp.json()["total"] == 0


class TestCitationFilters:
    """Test citation-related filtering."""

    @pytest.mark.integration
    def test_min_citations(self, client):
        resp = client.post(
            "/search",
            json={"min_citations": 100},
            headers=auth_headers(),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] >= 1
        for hit in data["hits"]:
            assert hit["citation_stats"]["total_citations"] >= 100

    @pytest.mark.integration
    def test_max_citations(self, client):
        resp = client.post(
            "/search",
            json={"max_citations": 50},
            headers=auth_headers(),
        )
        assert resp.status_code == 200
        data = resp.json()
        for hit in data["hits"]:
            assert hit["citation_stats"]["total_citations"] <= 50

    @pytest.mark.integration
    def test_citation_range(self, client):
        resp = client.post(
            "/search",
            json={"min_citations": 10, "max_citations": 200},
            headers=auth_headers(),
        )
        assert resp.status_code == 200
        data = resp.json()
        for hit in data["hits"]:
            c = hit["citation_stats"]["total_citations"]
            assert 10 <= c <= 200

    @pytest.mark.integration
    def test_min_references(self, client):
        resp = client.post(
            "/search",
            json={"min_references": 60},
            headers=auth_headers(),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] >= 1

    @pytest.mark.integration
    def test_median_h_index_citing_filter(self, client):
        resp = client.post(
            "/search",
            json={"min_median_h_index_citing": 30.0},
            headers=auth_headers(),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] >= 1

    @pytest.mark.integration
    def test_zero_citations_filter(self, client):
        resp = client.post(
            "/search",
            json={"max_citations": 0},
            headers=auth_headers(),
        )
        assert resp.status_code == 200
        data = resp.json()
        # Paper 2301.99999 has 0 citations
        assert data["total"] >= 1


class TestCategoryFilters:
    """Test category-related filtering."""

    @pytest.mark.integration
    def test_filter_by_categories(self, client):
        resp = client.post(
            "/search",
            json={"categories": ["cs.CL"]},
            headers=auth_headers(),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] >= 1
        for hit in data["hits"]:
            assert "cs.CL" in hit["categories"]

    @pytest.mark.integration
    def test_filter_by_primary_category(self, client):
        resp = client.post(
            "/search",
            json={"primary_category": "quant-ph"},
            headers=auth_headers(),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] >= 1
        for hit in data["hits"]:
            assert hit["primary_category"] == "quant-ph"

    @pytest.mark.integration
    def test_exclude_categories(self, client):
        resp = client.post(
            "/search",
            json={"exclude_categories": ["cs.CV"]},
            headers=auth_headers(),
        )
        assert resp.status_code == 200
        data = resp.json()
        for hit in data["hits"]:
            assert "cs.CV" not in hit["categories"]

    @pytest.mark.integration
    def test_multiple_categories_filter(self, client):
        resp = client.post(
            "/search",
            json={"categories": ["cs.LG", "cs.CL"]},
            headers=auth_headers(),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] >= 1


class TestDateFilters:
    """Test date-based filtering."""

    @pytest.mark.integration
    def test_submitted_date_gte(self, client):
        resp = client.post(
            "/search",
            json={"submitted_date": {"gte": "2024-03-01T00:00:00+00:00"}},
            headers=auth_headers(),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] >= 1
        for hit in data["hits"]:
            assert hit["submitted_date"] >= "2024-03-01"

    @pytest.mark.integration
    def test_submitted_date_lte(self, client):
        resp = client.post(
            "/search",
            json={"submitted_date": {"lte": "2023-12-31T23:59:59+00:00"}},
            headers=auth_headers(),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] >= 1

    @pytest.mark.integration
    def test_submitted_date_range(self, client):
        resp = client.post(
            "/search",
            json={
                "submitted_date": {
                    "gte": "2024-01-01T00:00:00+00:00",
                    "lte": "2024-02-28T23:59:59+00:00",
                }
            },
            headers=auth_headers(),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] >= 1

    @pytest.mark.integration
    def test_empty_date_range(self, client):
        resp = client.post(
            "/search",
            json={
                "submitted_date": {
                    "gte": "2025-01-01T00:00:00+00:00",
                    "lte": "2025-12-31T23:59:59+00:00",
                }
            },
            headers=auth_headers(),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0

    @pytest.mark.integration
    def test_updated_date_filter(self, client):
        resp = client.post(
            "/search",
            json={"updated_date": {"gte": "2024-01-01T00:00:00+00:00"}},
            headers=auth_headers(),
        )
        assert resp.status_code == 200


class TestMetadataFilters:
    """Test metadata-based filtering."""

    @pytest.mark.integration
    def test_has_github_true(self, client):
        resp = client.post(
            "/search",
            json={"has_github": True},
            headers=auth_headers(),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] >= 1
        for hit in data["hits"]:
            assert hit["has_github"] is True

    @pytest.mark.integration
    def test_has_github_false(self, client):
        resp = client.post(
            "/search",
            json={"has_github": False},
            headers=auth_headers(),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] >= 1
        for hit in data["hits"]:
            assert hit["has_github"] is False

    @pytest.mark.integration
    def test_min_page_count(self, client):
        resp = client.post(
            "/search",
            json={"min_page_count": 20},
            headers=auth_headers(),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] >= 1
        for hit in data["hits"]:
            assert hit["page_count"] >= 20

    @pytest.mark.integration
    def test_max_page_count(self, client):
        resp = client.post(
            "/search",
            json={"max_page_count": 10},
            headers=auth_headers(),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] >= 1
        for hit in data["hits"]:
            assert hit["page_count"] <= 10

    @pytest.mark.integration
    def test_page_count_range(self, client):
        resp = client.post(
            "/search",
            json={"min_page_count": 10, "max_page_count": 30},
            headers=auth_headers(),
        )
        assert resp.status_code == 200
        data = resp.json()
        for hit in data["hits"]:
            assert 10 <= hit["page_count"] <= 30

    @pytest.mark.integration
    def test_has_doi_true(self, client):
        resp = client.post(
            "/search",
            json={"has_doi": True},
            headers=auth_headers(),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] >= 1
        for hit in data["hits"]:
            assert hit["doi"] is not None

    @pytest.mark.integration
    def test_has_doi_false(self, client):
        resp = client.post(
            "/search",
            json={"has_doi": False},
            headers=auth_headers(),
        )
        assert resp.status_code == 200
        data = resp.json()
        for hit in data["hits"]:
            assert hit["doi"] is None

    @pytest.mark.integration
    def test_has_journal_ref(self, client):
        resp = client.post(
            "/search",
            json={"has_journal_ref": True},
            headers=auth_headers(),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] >= 1
        for hit in data["hits"]:
            assert hit["journal_ref"] is not None
