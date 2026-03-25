"""Unit tests for ingestion worker parsing logic."""
from __future__ import annotations

import pytest
from src.ingestion.worker import (
    extract_github_urls,
    estimate_page_count,
)


class TestGitHubUrlExtraction:
    """Test GitHub URL detection from text."""

    @pytest.mark.unit
    def test_extract_github_url_from_abstract(self):
        text = "Code is available at https://github.com/user/repo"
        urls = extract_github_urls(text)
        assert urls == ["https://github.com/user/repo"]

    @pytest.mark.unit
    def test_extract_multiple_github_urls(self):
        text = "See https://github.com/a/b and https://github.com/c/d"
        urls = extract_github_urls(text)
        assert len(urls) == 2

    @pytest.mark.unit
    def test_no_github_urls(self):
        text = "This paper has no code."
        urls = extract_github_urls(text)
        assert urls == []

    @pytest.mark.unit
    def test_empty_text(self):
        assert extract_github_urls("") == []

    @pytest.mark.unit
    def test_none_text(self):
        assert extract_github_urls(None) == []

    @pytest.mark.unit
    def test_github_url_with_dashes(self):
        text = "https://github.com/my-org/my-project"
        urls = extract_github_urls(text)
        assert len(urls) == 1

    @pytest.mark.unit
    def test_github_url_dedup(self):
        text = "https://github.com/a/b https://github.com/a/b"
        urls = extract_github_urls(text)
        assert len(urls) == 1


class TestPageCountEstimation:
    """Test page count extraction from comments."""

    @pytest.mark.unit
    def test_standard_pages(self):
        assert estimate_page_count("12 pages, 5 figures") == 12

    @pytest.mark.unit
    def test_single_page(self):
        assert estimate_page_count("1 page") == 1

    @pytest.mark.unit
    def test_pages_with_extra_text(self):
        assert estimate_page_count("Accepted at ICML, 25 pages including appendix") == 25

    @pytest.mark.unit
    def test_no_page_info(self):
        assert estimate_page_count("Accepted at NeurIPS 2024") is None

    @pytest.mark.unit
    def test_none_comments(self):
        assert estimate_page_count(None) is None

    @pytest.mark.unit
    def test_empty_comments(self):
        assert estimate_page_count("") is None
