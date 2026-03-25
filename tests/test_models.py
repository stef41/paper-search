"""Unit tests for models and config."""
from __future__ import annotations

import os
import pytest
from pydantic import ValidationError

from src.core.models import SearchRequest, SimilarityLevel, SortField, SortOrder, DateRange, SemanticQuery


class TestSearchRequestValidation:
    """Test SearchRequest Pydantic model validation."""

    @pytest.mark.unit
    def test_default_values(self):
        req = SearchRequest()
        assert req.query is None
        assert req.offset == 0
        assert req.limit == 20
        assert req.operator == "or"
        assert req.sort_by == SortField.RELEVANCE
        assert req.sort_order == SortOrder.DESC

    @pytest.mark.unit
    def test_query_max_length(self):
        with pytest.raises(ValidationError):
            SearchRequest(query="a" * 2001)

    @pytest.mark.unit
    def test_valid_query(self):
        req = SearchRequest(query="neural networks")
        assert req.query == "neural networks"

    @pytest.mark.unit
    def test_negative_offset_rejected(self):
        with pytest.raises(ValidationError):
            SearchRequest(offset=-1)

    @pytest.mark.unit
    def test_limit_too_large_rejected(self):
        with pytest.raises(ValidationError):
            SearchRequest(limit=500)

    @pytest.mark.unit
    def test_limit_zero_rejected(self):
        with pytest.raises(ValidationError):
            SearchRequest(limit=0)

    @pytest.mark.unit
    def test_invalid_operator(self):
        with pytest.raises(ValidationError):
            SearchRequest(operator="xor")

    @pytest.mark.unit
    def test_valid_operator_and(self):
        req = SearchRequest(operator="and")
        assert req.operator == "and"

    @pytest.mark.unit
    def test_valid_operator_or(self):
        req = SearchRequest(operator="or")
        assert req.operator == "or"

    @pytest.mark.unit
    def test_negative_min_citations(self):
        with pytest.raises(ValidationError):
            SearchRequest(min_citations=-1)

    @pytest.mark.unit
    def test_negative_min_h_index(self):
        with pytest.raises(ValidationError):
            SearchRequest(min_h_index=-1)

    @pytest.mark.unit
    def test_fuzziness_too_high(self):
        with pytest.raises(ValidationError):
            SearchRequest(fuzzy_fuzziness=5)

    @pytest.mark.unit
    def test_dangerous_regex_blocked(self):
        with pytest.raises(ValidationError):
            SearchRequest(title_regex="(.+)+")

    @pytest.mark.unit
    def test_invalid_regex_blocked(self):
        with pytest.raises(ValidationError):
            SearchRequest(title_regex="[invalid")

    @pytest.mark.unit
    def test_valid_regex(self):
        req = SearchRequest(title_regex=".*neural.*")
        assert req.title_regex == ".*neural.*"

    @pytest.mark.unit
    def test_regex_max_length(self):
        with pytest.raises(ValidationError):
            SearchRequest(title_regex="a" * 201)

    @pytest.mark.unit
    def test_semantic_query(self):
        req = SearchRequest(
            semantic=SemanticQuery(
                text="neural networks",
                level=SimilarityLevel.TITLE,
                weight=2.0,
            )
        )
        assert req.semantic.level == SimilarityLevel.TITLE
        assert req.semantic.weight == 2.0

    @pytest.mark.unit
    def test_date_range(self):
        req = SearchRequest(
            submitted_date=DateRange(
                gte="2024-01-01T00:00:00+00:00",
                lte="2024-12-31T23:59:59+00:00",
            )
        )
        assert req.submitted_date.gte is not None
        assert req.submitted_date.lte is not None

    @pytest.mark.unit
    def test_categories_list(self):
        req = SearchRequest(categories=["cs.AI", "cs.LG"])
        assert len(req.categories) == 2

    @pytest.mark.unit
    def test_all_sort_fields(self):
        for f in SortField:
            req = SearchRequest(sort_by=f)
            assert req.sort_by == f

    @pytest.mark.unit
    def test_all_sort_orders(self):
        for o in SortOrder:
            req = SearchRequest(sort_order=o)
            assert req.sort_order == o

    @pytest.mark.unit
    def test_all_similarity_levels(self):
        for lvl in SimilarityLevel:
            sq = SemanticQuery(text="test", level=lvl)
            assert sq.level == lvl

    @pytest.mark.unit
    def test_semantic_weight_boundaries(self):
        SemanticQuery(text="test", weight=0.0)
        SemanticQuery(text="test", weight=10.0)
        with pytest.raises(ValidationError):
            SemanticQuery(text="test", weight=-1.0)
        with pytest.raises(ValidationError):
            SemanticQuery(text="test", weight=11.0)
