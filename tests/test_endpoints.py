"""Tests for health, stats, and single paper endpoints."""
from __future__ import annotations

import pytest
from tests.conftest import auth_headers


class TestHealthEndpoint:
    """Test the /health endpoint."""

    @pytest.mark.integration
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    @pytest.mark.integration
    def test_health_response_structure(self, client):
        resp = client.get("/health")
        data = resp.json()
        assert "status" in data
        # Hardened: only exposes aggregate status, not component details
        assert data["status"] in ("healthy", "degraded")

    @pytest.mark.integration
    def test_health_no_info_leak(self, client):
        resp = client.get("/health")
        data = resp.json()
        # Security: health must NOT leak internal component details
        assert "elasticsearch" not in data
        assert "redis" not in data
        assert "total_papers" not in data

    @pytest.mark.integration
    def test_health_healthy_status(self, client):
        resp = client.get("/health")
        data = resp.json()
        assert data["status"] == "healthy"


class TestStatsEndpoint:
    """Test the /stats endpoint."""

    @pytest.mark.integration
    def test_stats_returns_200(self, client):
        resp = client.get("/stats", headers=auth_headers())
        assert resp.status_code == 200

    @pytest.mark.integration
    def test_stats_response_structure(self, client):
        resp = client.get("/stats", headers=auth_headers())
        data = resp.json()
        assert "total_papers" in data
        assert "categories" in data
        assert "date_range" in data
        assert "papers_with_github" in data
        assert "avg_page_count" in data
        assert "avg_citations" in data

    @pytest.mark.integration
    def test_stats_correct_counts(self, client):
        resp = client.get("/stats", headers=auth_headers())
        data = resp.json()
        assert data["total_papers"] >= 6
        assert data["papers_with_github"] >= 3

    @pytest.mark.integration
    def test_stats_categories(self, client):
        resp = client.get("/stats", headers=auth_headers())
        data = resp.json()
        assert len(data["categories"]) > 0
        assert "cs.AI" in data["categories"]


class TestPaperEndpoint:
    """Test the /paper/{arxiv_id} endpoint."""

    @pytest.mark.integration
    def test_get_existing_paper(self, client):
        resp = client.get("/paper/2401.00001", headers=auth_headers())
        assert resp.status_code == 200
        data = resp.json()
        assert data["arxiv_id"] == "2401.00001"
        assert data["title"] == "Neural Networks for Natural Language Processing"

    @pytest.mark.integration
    def test_get_nonexistent_paper(self, client):
        resp = client.get("/paper/9999.99999", headers=auth_headers())
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()

    @pytest.mark.integration
    def test_get_paper_full_data(self, client):
        resp = client.get("/paper/2401.00002", headers=auth_headers())
        assert resp.status_code == 200
        data = resp.json()
        assert data["arxiv_id"] == "2401.00002"
        assert len(data["authors"]) == 2
        assert data["has_github"] is True
        assert len(data["categories"]) == 2
