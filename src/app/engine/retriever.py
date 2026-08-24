"""
Hybrid retriever combining BM25 keyword search and ChromaDB vector search.

Uses Reciprocal Rank Fusion (RRF) to merge results from both sources.
"""

import logging
from typing import Any, Literal, cast

from app.engine.cache import QueryCache
from app.engine.indexer import DocumentIndexer

logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)

SearchMode = Literal["hybrid", "semantic", "keyword"]


def get_search_mode(vector_search_weight: float) -> SearchMode:
    if vector_search_weight >= 1.0:
        return "semantic"

    if vector_search_weight <= 0.0:
        return "keyword"

    return "hybrid"


def _reciprocal_rank_fusion(
    result_lists: list[list[dict[str, Any]]],
    k: int = 60,
) -> list[dict[str, Any]]:
    """
    Merge multiple ranked result lists using Reciprocal Rank Fusion.

    Each result dict must have a "content" and "metadata" key.
    A unique key is constructed from (source, chunk_index) metadata.

    Args:
        result_lists: List of ranked result lists to fuse.
        k: RRF constant (default 60 per the original paper).

    Returns:
        Merged and re-ranked list of result dicts with an "rrf_score" key.
    """
    scores: dict[str, float] = {}
    result_map: dict[str, dict[str, Any]] = {}

    for results in result_lists:
        for rank, result in enumerate(results):
            meta = result.get("metadata", {})
            key = f"{meta.get('source', '')}::{meta.get('chunk_index', '')}"
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            if key not in result_map:
                result_map[key] = result

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    out = []
    for key, score in ranked:
        entry = result_map[key].copy()
        entry["rrf_score"] = score
        out.append(entry)
    return out


class HybridRetriever:
    """
    Retriever that combines BM25 keyword search and ChromaDB vector search.

    Modes:
      - "hybrid"   — Reciprocal Rank Fusion of BM25 + vector results (default).
      - "semantic"  — Vector search only.
      - "keyword"   — BM25 keyword search only.

    Usage:
        retriever = HybridRetriever(indexer)
        results = retriever.search("authentication flow", top_k=5)
    """

    def __init__(self, indexer: DocumentIndexer):
        self._indexer = indexer
        self._query_cache = QueryCache()

    def search(
        self, query: str, top_k: int = 5, vector_search_weight: float = 0.5, rrf_fold: int = 3, rrf_k: int = 60
    ) -> list[dict[str, Any]]:
        """
        Search the index and return the top-k most relevant chunks.

        Args:
            query: Natural language search query.
            top_k: Maximum number of results.
            vector_search_weight: combination of keyword and vectoer search
                0.0: Keyword Search
                0.0 < weight < 1.0: Hybrid Search
                1.0: fully Semantic Search
            rrf_fold: Multiplier to preserve fuse resolution in RRF (Hybrid ONLY)
            rrf_k: K parameter in RRF canonical formula (Hybrid ONLY)

        Returns:
            List of dicts with keys: content, metadata, score/rrf_score.
        """
        mode: SearchMode = get_search_mode(vector_search_weight=vector_search_weight)

        # Check cache first
        if mode == "hybrid":
            cached = self._query_cache.get(query, top_k, vector_search_weight, rrf_fold=rrf_fold, rrf_k=rrf_k)
        else:
            cached = self._query_cache.get(query, top_k, vector_search_weight)
        if cached is not None:
            logger.debug("Query cache hit: %s", query[:60])
            return cached

        if mode == "semantic":
            results = self._indexer.search_vector(query, top_k=top_k)
            dbg_mgs = f"query={query}, top_k={top_k}, mode={mode}, v_weight={vector_search_weight}"
        elif mode == "keyword":
            results = self._indexer.search_bm25(query, top_k=top_k)
            dbg_mgs = f"query={query}, top_k={top_k}, mode={mode}, v_weight={vector_search_weight}"
        else:
            # Hybrid: fetch more from each source, then fuse
            fetch_k = top_k * rrf_fold
            vec_results = self._indexer.search_vector(query, top_k=fetch_k)
            bm25_results = self._indexer.search_bm25(query, top_k=fetch_k)

            num_total_hits = len(vec_results) + len(bm25_results)  # NOTE: might be < 2*fetch_k
            num_vector_hits = int(vector_search_weight * num_total_hits + 1)
            num_keyword_hits = int((1.0 - vector_search_weight) * num_total_hits + 1)
            results = _reciprocal_rank_fusion([vec_results[:num_vector_hits], bm25_results[:num_keyword_hits]], rrf_k)[
                :top_k
            ]
            dbg_mgs = f"query={query}, top_k={top_k}, mode={mode}, v_weight={vector_search_weight}, rrf_fold={rrf_fold}, rrf_k={rrf_k}"

        logger.debug(dbg_mgs)

        # Cache the results
        if mode == "hybrid":
            self._query_cache.set(query, top_k, vector_search_weight, results, rrf_fold=rrf_fold, rrf_k=rrf_k)
        else:
            self._query_cache.set(query, top_k, vector_search_weight, results)

        return results

    def clear_cache(self) -> None:
        """Clear the query result cache."""
        cast(Any, self._query_cache).clear()
