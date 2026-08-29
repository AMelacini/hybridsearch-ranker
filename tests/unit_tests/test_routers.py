import asyncio
from pathlib import Path
from typing import Any

import pytest
from fastapi import Response, status
from mock import AsyncMock, Mock, patch

import app.routers as routers_module
from app.models import FileType, FileTypeFilter, FoundDocuments, SearchModel
from app.routers import create_admin_endpoints, create_document_endpoints, create_search_endpoints


@pytest.mark.asyncio
async def test_admin_live_probe() -> None:
    router = create_admin_endpoints(api_version="v1")
    probe_route = next(route for route in router.routes if getattr(route, "path", None) == "/admin/probe")

    response = await probe_route.endpoint()

    assert response.message.startswith("OK on")


@pytest.mark.asyncio
async def test_document_list_by_supported_extensions(mock_indexer: Mock) -> None:
    router = create_document_endpoints(api_version="v1", indexer=mock_indexer, retriever=Mock())
    list_route = next(route for route in router.routes if getattr(route, "path", None) == "/documents/list")
    resp = Response()
    response = await list_route.endpoint(resp, file_ext=".md")

    assert isinstance(response, FoundDocuments)
    assert response.doc_paths == [Path("docs/a.md")]


@pytest.mark.asyncio
async def test_document_content_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "nested").mkdir()
    (docs_dir / "nested" / "notes.md").write_text("# Heading\n\nBody text", encoding="utf-8")
    (docs_dir / "nested" / "notes.bin").write_text("binary-data", encoding="utf-8")

    monkeypatch.setattr(routers_module, "DEFAULT_DOCS_DIR", str(docs_dir))
    indexer = Mock()
    indexer.list_files.return_value = ["nested/notes.md", "../escape.txt", "nested/notes.bin"]
    router = create_document_endpoints(api_version="v1", indexer=indexer, retriever=Mock())
    route = next(
        route for route in router.routes if getattr(route, "path", None) == "/documents/content/{file_path:path}"
    )

    response = await route.endpoint(Response(), "nested/notes.md")
    assert response.textual_content.startswith("[File: nested/notes.md]")

    bad_path_response = await route.endpoint(Response(), "../escape.txt")
    assert bad_path_response.textual_content == ""

    unsupported_response = await route.endpoint(Response(), "nested/notes.bin")
    assert unsupported_response.textual_content == ""


@pytest.mark.asyncio
async def test_search_response(mock_indexer: Mock, sample_search_hits: list[dict[str, Any]]) -> None:
    retriever = Mock()
    retriever.search.return_value = sample_search_hits
    router = create_search_endpoints(api_version="v1", indexer=mock_indexer, retriever=retriever)
    search_route = next(route for route in router.routes if getattr(route, "path", None) == "/search/")

    resp = Response()
    response = await search_route.endpoint(resp, SearchModel(query="hello", top_k=5, vector_search_weight=0.5))

    assert response.summary.startswith("Found 2 relevant chunk")
    assert "Source: notes.md" in response.hits[0]
    assert "Section: Overview" in response.hits[0]
    assert "This is the relevant chunk" in response.hits[0]

    retriever.search.return_value = []
    empty_response = await search_route.endpoint(resp, SearchModel(query="missing", top_k=5, vector_search_weight=0.0))
    assert "No results found" in empty_response.summary


@pytest.mark.asyncio
async def test_search_timeout_and_unexpected_failures(mock_indexer: Mock) -> None:
    retriever = Mock()
    router = create_search_endpoints(api_version="v1", indexer=mock_indexer, retriever=retriever)
    search_route = next(route for route in router.routes if getattr(route, "path", None) == "/search/")

    # NOTE
    # the following Implementation
    """
    with patch("app.routers.asyncio.wait_for", new=AsyncMock(side_effect=asyncio.TimeoutError())):
        with pytest.raises(Exception) as exc_info:
            await search_route.endpoint(Response(), SearchModel(query="timeout", top_k=5, vector_search_weight=0.5))
        assert exc_info.value.status_code == status.HTTP_408_REQUEST_TIMEOUT

    with patch("app.routers.asyncio.wait_for", new=AsyncMock(side_effect=RuntimeError("boom"))):
        with pytest.raises(Exception) as exc_info:
            await search_route.endpoint(Response(), SearchModel(query="fail", top_k=5, vector_search_weight=0.5))
        assert exc_info.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    """

    # Would throw
    # RuntimeWarning: coroutine 'to_thread' was never awaited
    # That's because the mocked wait_for raises before the coroutine created by asyncio.to_thread is awaited
    # This is the correct implementation (thanx to co-pilot ;-) )

    async def fake_wait_for(coro: Any, *args: Any, **kwargs: Any) -> Any:
        # Prevent the unawaited coroutine warning from asyncio.to_thread(...)
        coro.close()
        raise asyncio.TimeoutError()

    with patch("app.routers.asyncio.wait_for", new=AsyncMock(side_effect=fake_wait_for)):
        with pytest.raises(Exception) as exc_info:
            await search_route.endpoint(
                Response(),
                SearchModel(query="timeout", top_k=5, vector_search_weight=0.5),
            )
        assert exc_info.value.status_code == status.HTTP_408_REQUEST_TIMEOUT

    async def fake_wait_for_runtime(coro: Any, *args: Any, **kwargs: Any) -> Any:
        coro.close()
        raise RuntimeError("boom")

    with patch("app.routers.asyncio.wait_for", new=AsyncMock(side_effect=fake_wait_for_runtime)):
        with pytest.raises(Exception) as exc_info:
            await search_route.endpoint(
                Response(),
                SearchModel(query="fail", top_k=5, vector_search_weight=0.5),
            )
        assert exc_info.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR

    # Alternatively, use a fake wait for:
    """
    async def fake_wait_for(coro, *args, **kwargs):
    if hasattr(coro, "close"):
        coro.close()
    raise asyncio.TimeoutError()
    """
