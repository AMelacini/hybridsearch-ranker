from mock import AsyncMock, Mock, patch

from app.engine.retriever import HybridRetriever, _reciprocal_rank_fusion


def test_reciprocal_rank_fusion_ranks_shared_hits_highest() -> None:
    left = [
        {"content": "alpha", "metadata": {"source": "doc-a", "chunk_index": 0}},
        {"content": "beta", "metadata": {"source": "doc-b", "chunk_index": 1}},
    ]
    right = [
        {"content": "beta", "metadata": {"source": "doc-b", "chunk_index": 1}},
        {"content": "gamma", "metadata": {"source": "doc-c", "chunk_index": 0}},
    ]

    fused = _reciprocal_rank_fusion([left, right], k=60)

    assert fused[0]["content"] == "beta"
    assert "rrf_score" in fused[0]
    assert fused[0]["rrf_score"] > fused[1]["rrf_score"]


def test_use_cache() -> None:
    indexer = Mock()
    retriever = HybridRetriever(indexer)
    retriever._query_cache = Mock()
    retriever._query_cache.get.return_value = [{"content": "cached", "metadata": {}}]

    results = retriever.search("hello", top_k=3, vector_search_weight=0.0)

    assert results[0]["content"] == "cached"
    indexer.search_bm25.assert_not_called()
    retriever._query_cache.get.assert_called_once_with("hello", 3, 0.0)


def test_query_keyword_indexer_on_empty_cache() -> None:
    indexer = Mock()
    indexer.search_bm25.return_value = [{"content": "bm25-hit", "metadata": {"source": "doc.md"}}]
    retriever = HybridRetriever(indexer)
    retriever._query_cache = Mock()
    retriever._query_cache.get.return_value = None

    results = retriever.search("hello", top_k=2, vector_search_weight=0.0)

    assert results[0]["content"] == "bm25-hit"
    indexer.search_bm25.assert_called_once_with("hello", top_k=2)
    retriever._query_cache.set.assert_called_once()


def test_semantic_mode() -> None:
    indexer = Mock()
    indexer.search_vector.return_value = [{"content": "vector-hit", "metadata": {"source": "doc.md"}}]
    retriever = HybridRetriever(indexer)
    retriever._query_cache = Mock()
    retriever._query_cache.get.return_value = None

    results = retriever.search("hello", top_k=4, vector_search_weight=1.0)

    assert results[0]["content"] == "vector-hit"
    indexer.search_vector.assert_called_once_with("hello", top_k=4)
    retriever._query_cache.set.assert_called_once_with("hello", 4, 1.0, results)
