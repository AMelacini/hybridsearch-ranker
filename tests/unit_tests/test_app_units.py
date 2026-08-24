from pathlib import Path

import pytest
from fastapi import Response

# from unittest.mock import Mock
from mock import Mock

from app.engine.indexer import load_file, tokenize
from app.engine.loader_registry import LoaderRegistry
from app.engine.retriever import HybridRetriever, _reciprocal_rank_fusion
from app.models import FileType, FoundDocuments, SearchModel
from app.routers import create_admin_endpoints, create_document_endpoints, create_search_endpoints


def test_tokenize_normalizes_and_splits_words() -> None:
    assert tokenize("Hello, world! 123") == ["hello", "world", "123"]


def test_loader_registry_uses_filetype_keys() -> None:
    registry = LoaderRegistry()
    loader = Mock()

    registry.set_loader(FileType.MD, loader)

    assert registry.get_loader(FileType.MD) is loader
    assert registry.get_loader(FileType.PDF) is None


def test_loader_registry_builds_from_enum_members() -> None:
    registry = LoaderRegistry.from_file_types({FileType.MD: Mock(), FileType.PDF: Mock()})

    assert registry.get_loader(FileType.MD) is not None
    assert registry.get_loader(FileType.PDF) is not None
    assert registry.get_loader(FileType.TXT) is None


def test_loader_registry_resolves_extension_aliases() -> None:
    loader = Mock()
    registry = LoaderRegistry.from_file_types({FileType.MD: loader})

    assert registry.get_loader_for_extension("markdown") is loader
    assert registry.get_loader_for_extension("md") is loader


def test_load_file_uses_registry_for_supported_types(tmp_path: Path) -> None:
    file_path = tmp_path / "notes.md"
    file_path.write_text("# Title\n\nBody", encoding="utf-8")

    docs = load_file(file_path)

    assert len(docs) == 1
    assert docs[0].page_content.startswith("# Title")


def test_load_csv_file_yields_row_documents(tmp_path: Path) -> None:
    file_path = tmp_path / "data.csv"
    file_path.write_text("name,age\nAlice,30\nBob,25\n", encoding="utf-8")

    docs = load_file(file_path)

    assert len(docs) == 2
    assert "name: Alice" in docs[0].page_content
    assert docs[0].metadata["row"] == 1
    assert docs[1].metadata["row"] == 2


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


def test_hybrid_retriever_uses_cache_and_skips_indexer() -> None:
    indexer = Mock()
    retriever = HybridRetriever(indexer)
    retriever._query_cache = Mock()
    retriever._query_cache.get.return_value = [{"content": "cached", "metadata": {}}]

    results = retriever.search("hello", top_k=3, vector_search_weight=0.0)

    assert results[0]["content"] == "cached"
    indexer.search_bm25.assert_not_called()
    retriever._query_cache.get.assert_called_once_with("hello", 3, 0.0)


def test_hybrid_retriever_queries_keyword_indexer_when_cache_is_empty() -> None:
    indexer = Mock()
    indexer.search_bm25.return_value = [{"content": "bm25-hit", "metadata": {"source": "doc.md"}}]
    retriever = HybridRetriever(indexer)
    retriever._query_cache = Mock()
    retriever._query_cache.get.return_value = None

    results = retriever.search("hello", top_k=2, vector_search_weight=0.0)

    assert results[0]["content"] == "bm25-hit"
    indexer.search_bm25.assert_called_once_with("hello", top_k=2)
    retriever._query_cache.set.assert_called_once()


def test_search_model_validates_mode_weight_consistency() -> None:
    with pytest.raises(ValueError):
        SearchModel(query="hello", top_k=3, vector_search_weight=1.5)

    with pytest.raises(ValueError):
        SearchModel(query="hello", top_k=3, vector_search_weight=-0.2)

    assert FileType.is_supported(".PDF") is True
    assert FileType.is_supported(".csv") is True
    assert FileType.is_supported(".unknown") is False


@pytest.mark.asyncio
async def test_admin_probe_endpoint_returns_status_message() -> None:
    router = create_admin_endpoints(api_version="v1")
    probe_route = next(route for route in router.routes if getattr(route, "path", None) == "/admin/probe")

    response = await probe_route.endpoint()

    assert response.message.startswith("OK on")


@pytest.mark.asyncio
async def test_document_list_endpoint_filters_supported_extensions() -> None:
    indexer = Mock()
    indexer.list_files.return_value = ["docs/a.md", "docs/b.txt", "docs/c.pdf"]
    router = create_document_endpoints(api_version="v1", indexer=indexer, retriever=Mock())
    list_route = next(route for route in router.routes if getattr(route, "path", None) == "/documents/list")
    resp = Response()
    response = await list_route.endpoint(resp, file_ext=".md")

    assert isinstance(response, FoundDocuments)
    assert response.doc_paths == [Path("docs/a.md")]


@pytest.mark.asyncio
async def test_search_endpoint_formats_results() -> None:
    indexer = Mock()
    retriever = Mock()
    retriever.search.return_value = [
        {
            "content": "This is the relevant chunk",
            "metadata": {"source": "notes.md", "section": "Overview", "page": 2, "file_type": "md"},
        }
    ]
    router = create_search_endpoints(api_version="v1", indexer=indexer, retriever=retriever)
    search_route = next(route for route in router.routes if getattr(route, "path", None) == "/search/")

    resp = Response()
    response = await search_route.endpoint(resp, SearchModel(query="hello", top_k=5, vector_search_weight=0.5))

    assert response.summary.startswith("Found 1 relevant chunk")
    assert "Source: notes.md" in response.hits[0]
    assert "Section: Overview" in response.hits[0]
    assert "This is the relevant chunk" in response.hits[0]
