"""Unit tests for ingestion worker parsing logic and data pipeline."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from src.ingestion.worker import (
    extract_github_urls,
    estimate_page_count,
    parse_oai_record,
)
from src.ingestion.seed import parse_api_entry
from src.ingestion.bulk_import import parse_kaggle_record, _parse_date


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


class TestKaggleRecordParsing:
    """Test parsing of Kaggle ArXiv JSON records."""

    @pytest.mark.unit
    def test_parse_standard_record(self):
        record = {
            "id": "2401.12345",
            "title": "  Test Paper   Title  ",
            "abstract": "  This is the abstract.  ",
            "categories": "cs.AI cs.LG",
            "authors": "Smith, John and Doe, Jane",
            "authors_parsed": [["Smith", "John", ""], ["Doe", "Jane", ""]],
            "versions": [
                {"version": "v1", "created": "Mon, 15 Jan 2024 00:00:00 GMT"},
                {"version": "v2", "created": "Fri, 19 Jan 2024 00:00:00 GMT"},
            ],
            "doi": "10.1234/test",
            "journal-ref": "NeurIPS 2024",
            "comments": "12 pages, 5 figures. Code at https://github.com/test/repo",
        }
        result = parse_kaggle_record(record)
        assert result is not None
        assert result["arxiv_id"] == "2401.12345"
        assert result["title"] == "Test Paper Title"  # whitespace normalized
        assert result["primary_category"] == "cs.AI"
        assert result["categories"] == ["cs.AI", "cs.LG"]
        assert len(result["authors"]) == 2
        assert result["authors"][0]["name"] == "John Smith"
        assert result["authors"][0]["is_first_author"] is True
        assert result["authors"][1]["is_first_author"] is False
        assert result["page_count"] == 12
        assert result["has_github"] is True
        assert result["doi"] == "10.1234/test"
        assert result["journal_ref"] == "NeurIPS 2024"

    @pytest.mark.unit
    def test_parse_record_missing_fields(self):
        record = {
            "id": "2301.00001",
            "title": "Minimal Paper",
            "abstract": "Short abstract.",
            "categories": "math.CO",
        }
        result = parse_kaggle_record(record)
        assert result is not None
        assert result["arxiv_id"] == "2301.00001"
        assert result["has_github"] is False
        assert result["page_count"] is None
        assert result["doi"] is None

    @pytest.mark.unit
    def test_parse_record_empty_id(self):
        record = {"id": "", "title": "Test", "abstract": "Test"}
        assert parse_kaggle_record(record) is None

    @pytest.mark.unit
    def test_parse_record_no_title(self):
        record = {"id": "1234", "title": "", "abstract": "Test"}
        assert parse_kaggle_record(record) is None

    @pytest.mark.unit
    def test_parse_record_no_abstract(self):
        record = {"id": "1234", "title": "Test", "abstract": ""}
        assert parse_kaggle_record(record) is None

    @pytest.mark.unit
    def test_parse_record_github_in_comments(self):
        record = {
            "id": "2401.00001",
            "title": "Paper",
            "abstract": "No code here.",
            "categories": "cs.AI",
            "comments": "See https://github.com/org/project",
        }
        result = parse_kaggle_record(record)
        assert result["has_github"] is True
        assert "https://github.com/org/project" in result["github_urls"]

    @pytest.mark.unit
    def test_parse_record_single_author_parsed(self):
        record = {
            "id": "2401.00001",
            "title": "Solo Paper",
            "abstract": "I did it alone.",
            "categories": "cs.AI",
            "authors_parsed": [["Einstein", "Albert", ""]],
        }
        result = parse_kaggle_record(record)
        assert len(result["authors"]) == 1
        assert result["authors"][0]["name"] == "Albert Einstein"
        assert result["first_author"] == "Albert Einstein"


class TestDateParsing:
    """Test date format parsing."""

    @pytest.mark.unit
    def test_parse_oai_date(self):
        d = _parse_date("Mon, 15 Jan 2024 00:00:00 GMT")
        assert "2024-01-15" in d

    @pytest.mark.unit
    def test_parse_iso_date(self):
        d = _parse_date("2024-01-15")
        assert "2024-01-15" in d

    @pytest.mark.unit
    def test_parse_iso_datetime(self):
        d = _parse_date("2024-01-15T12:30:00")
        assert "2024-01-15" in d

    @pytest.mark.unit
    def test_parse_invalid_date(self):
        with pytest.raises(ValueError):
            _parse_date("not-a-date")


class TestArxivApiParsing:
    """Test parsing of ArXiv API Atom entries."""

    @pytest.mark.unit
    def test_parse_api_entry_returns_none_for_empty(self):
        from xml.etree.ElementTree import fromstring
        xml_str = '<entry xmlns="http://www.w3.org/2005/Atom"></entry>'
        entry = fromstring(xml_str)
        result = parse_api_entry(entry)
        assert result is None  # No id element

    @pytest.mark.unit
    def test_parse_api_entry_with_data(self):
        from xml.etree.ElementTree import fromstring
        xml_str = '''<entry xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
            <id>http://arxiv.org/abs/2401.00001v1</id>
            <title>Test Neural Network Paper</title>
            <summary>We study neural networks for NLP tasks.</summary>
            <published>2024-01-15T00:00:00Z</published>
            <updated>2024-01-20T00:00:00Z</updated>
            <author><name>John Smith</name></author>
            <author><name>Jane Doe</name></author>
            <arxiv:primary_category term="cs.CL"/>
            <category term="cs.CL"/>
            <category term="cs.AI"/>
            <arxiv:comment>10 pages, https://github.com/test/repo</arxiv:comment>
            <arxiv:doi>10.1234/test</arxiv:doi>
            <link title="pdf" href="https://arxiv.org/pdf/2401.00001v1"/>
        </entry>'''
        entry = fromstring(xml_str)
        result = parse_api_entry(entry)
        assert result is not None
        assert result["arxiv_id"] == "2401.00001"  # version stripped
        assert result["title"] == "Test Neural Network Paper"
        assert len(result["authors"]) == 2
        assert result["authors"][0]["name"] == "John Smith"
        assert result["authors"][0]["is_first_author"] is True
        assert result["primary_category"] == "cs.CL"
        assert "cs.AI" in result["categories"]
        assert result["page_count"] == 10
        assert result["has_github"] is True
        assert result["doi"] == "10.1234/test"
        assert result["submitted_date"] is not None
        assert result["updated_date"] is not None


class TestBulkImportStream:
    """Test the streaming JSON parser."""

    @pytest.mark.unit
    def test_stream_filters_by_category(self):
        from src.ingestion.bulk_import import stream_kaggle_file

        records = [
            {"id": "1", "title": "AI Paper", "abstract": "About AI", "categories": "cs.AI"},
            {"id": "2", "title": "Physics", "abstract": "About physics", "categories": "hep-ph"},
            {"id": "3", "title": "ML Paper", "abstract": "About ML", "categories": "cs.LG cs.AI"},
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
            f.flush()

            results = list(stream_kaggle_file(f.name, filter_categories=["cs.AI"]))
            assert len(results) == 2  # records 1 and 3

    @pytest.mark.unit
    def test_stream_max_papers(self):
        from src.ingestion.bulk_import import stream_kaggle_file

        records = [
            {"id": str(i), "title": f"Paper {i}", "abstract": f"Abstract {i}", "categories": "cs.AI"}
            for i in range(10)
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
            f.flush()

            results = list(stream_kaggle_file(f.name, max_papers=3))
            assert len(results) == 3

    @pytest.mark.unit
    def test_stream_handles_bad_json(self):
        from src.ingestion.bulk_import import stream_kaggle_file

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write('{"id": "1", "title": "Good", "abstract": "Good", "categories": "cs.AI"}\n')
            f.write("not valid json\n")
            f.write('{"id": "2", "title": "Also Good", "abstract": "Also Good", "categories": "cs.AI"}\n')
            f.flush()

            results = list(stream_kaggle_file(f.name))
            assert len(results) == 2  # skips bad line

    @pytest.mark.unit
    def test_stream_skips_known_ids(self):
        from src.ingestion.bulk_import import stream_kaggle_file

        records = [
            {"id": "1", "title": "Paper 1", "abstract": "Abs 1", "categories": "cs.AI"},
            {"id": "2", "title": "Paper 2", "abstract": "Abs 2", "categories": "cs.AI"},
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
            f.flush()

            results = list(stream_kaggle_file(f.name, skip_ids={"1"}))
            assert len(results) == 1
            assert results[0]["arxiv_id"] == "2"


class TestEnrichmentComputation:
    """Test Semantic Scholar enrichment computation."""

    @pytest.mark.unit
    def test_compute_enrichment_basic(self):
        from src.ingestion.enrich import compute_enrichment

        s2_data = {
            "citationCount": 150,
            "authors": [
                {"hIndex": 45, "citationCount": 12000},
                {"hIndex": 32, "citationCount": 8000},
            ],
            "citations": [
                {"citationCount": 100, "fieldsOfStudy": ["Computer Science"]},
                {"citationCount": 50, "fieldsOfStudy": ["Computer Science", "Mathematics"]},
            ],
            "references": [
                {"fieldsOfStudy": ["Computer Science"]},
                {"fieldsOfStudy": ["Mathematics"]},
            ],
        }

        result = compute_enrichment(s2_data)
        assert result["citation_stats"]["total_citations"] == 150
        assert result["citation_stats"]["median_h_index_citing_authors"] is not None
        assert result["references_stats"]["total_references"] == 2
        assert result["first_author_h_index"] == 45

    @pytest.mark.unit
    def test_compute_enrichment_empty(self):
        from src.ingestion.enrich import compute_enrichment

        s2_data = {
            "citationCount": 0,
            "authors": [],
            "citations": [],
            "references": [],
        }

        result = compute_enrichment(s2_data)
        assert result["citation_stats"]["total_citations"] == 0
        assert result["references_stats"]["total_references"] == 0

    @pytest.mark.unit
    def test_compute_enrichment_none_fields(self):
        from src.ingestion.enrich import compute_enrichment

        s2_data = {
            "citationCount": None,
            "authors": None,
            "citations": None,
            "references": None,
        }

        result = compute_enrichment(s2_data)
        assert result["citation_stats"]["total_citations"] == 0
