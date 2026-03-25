"""Tests for authentication & security."""
from __future__ import annotations

import pytest
from tests.conftest import auth_headers, VALID_API_KEY, VALID_API_KEY_2, INVALID_API_KEY


class TestAuthentication:
    """API key authentication tests."""

    @pytest.mark.integration
    def test_search_without_api_key(self, client):
        resp = client.post("/search", json={"query": "test"})
        assert resp.status_code == 401
        assert "Missing API key" in resp.json()["detail"]

    @pytest.mark.integration
    def test_search_with_invalid_api_key(self, client):
        resp = client.post("/search", json={"query": "test"}, headers=auth_headers(INVALID_API_KEY))
        assert resp.status_code == 403
        assert "Invalid API key" in resp.json()["detail"]

    @pytest.mark.integration
    def test_search_with_valid_api_key(self, client):
        resp = client.post("/search", json={"query": "test"}, headers=auth_headers(VALID_API_KEY))
        assert resp.status_code == 200

    @pytest.mark.integration
    def test_search_with_second_valid_api_key(self, client):
        resp = client.post("/search", json={"query": "test"}, headers=auth_headers(VALID_API_KEY_2))
        assert resp.status_code == 200

    @pytest.mark.integration
    def test_stats_without_api_key(self, client):
        resp = client.get("/stats")
        assert resp.status_code == 401

    @pytest.mark.integration
    def test_paper_without_api_key(self, client):
        resp = client.get("/paper/2401.00001")
        assert resp.status_code == 401

    @pytest.mark.integration
    def test_health_no_auth_required(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200


class TestSecurityHeaders:
    """Test security response headers."""

    @pytest.mark.integration
    def test_security_headers_present(self, client):
        resp = client.get("/health")
        assert resp.headers.get("X-Content-Type-Options") == "nosniff"
        assert resp.headers.get("X-Frame-Options") == "DENY"
        assert resp.headers.get("X-XSS-Protection") == "1; mode=block"
        assert "max-age" in resp.headers.get("Strict-Transport-Security", "")
        assert resp.headers.get("Cache-Control") == "no-store"


class TestInputValidation:
    """Test input validation and sanitization."""

    @pytest.mark.integration
    def test_query_too_long(self, client):
        resp = client.post(
            "/search",
            json={"query": "a" * 3000},
            headers=auth_headers(),
        )
        assert resp.status_code == 422  # Pydantic validation

    @pytest.mark.integration
    def test_regex_too_long(self, client):
        resp = client.post(
            "/search",
            json={"title_regex": "a" * 300},
            headers=auth_headers(),
        )
        assert resp.status_code == 422

    @pytest.mark.integration
    def test_invalid_regex_syntax(self, client):
        resp = client.post(
            "/search",
            json={"title_regex": "[invalid"},
            headers=auth_headers(),
        )
        assert resp.status_code == 422

    @pytest.mark.integration
    def test_dangerous_regex_blocked(self, client):
        resp = client.post(
            "/search",
            json={"title_regex": "(.+)+"},
            headers=auth_headers(),
        )
        assert resp.status_code == 422

    @pytest.mark.integration
    def test_negative_offset(self, client):
        resp = client.post(
            "/search",
            json={"query": "test", "offset": -1},
            headers=auth_headers(),
        )
        assert resp.status_code == 422

    @pytest.mark.integration
    def test_limit_too_large(self, client):
        resp = client.post(
            "/search",
            json={"query": "test", "limit": 500},
            headers=auth_headers(),
        )
        assert resp.status_code == 422

    @pytest.mark.integration
    def test_invalid_operator(self, client):
        resp = client.post(
            "/search",
            json={"query": "test", "operator": "xor"},
            headers=auth_headers(),
        )
        assert resp.status_code == 422

    @pytest.mark.integration
    def test_negative_min_citations(self, client):
        resp = client.post(
            "/search",
            json={"min_citations": -5},
            headers=auth_headers(),
        )
        assert resp.status_code == 422

    @pytest.mark.integration
    def test_pagination_beyond_limit(self, client):
        resp = client.post(
            "/search",
            json={"query": "test", "offset": 50000, "limit": 100},
            headers=auth_headers(),
        )
        assert resp.status_code == 400

    @pytest.mark.integration
    def test_null_bytes_in_query(self, client):
        # Null bytes should be handled gracefully
        resp = client.post(
            "/search",
            json={"query": "neural\x00network"},
            headers=auth_headers(),
        )
        # Should not crash
        assert resp.status_code in (200, 422)
