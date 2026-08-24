import asyncio
import logging
import time
import traceback
from datetime import datetime
from getpass import getuser
from pathlib import Path
from typing import Any, cast

from fastapi import APIRouter, HTTPException, Response, status

from app.engine.indexer import DocumentIndexer
from app.engine.retriever import HybridRetriever, get_search_mode
from app.models import (
    FileContent,
    FileType,
    FoundDocuments,
    SearchModel,
    SearchResults,
    TextualResponse,
)
from hsr_config import DEFAULT_DOCS_DIR, SEARCH_REQUEST_TIMEOUT

logger = logging.getLogger(__name__)


def _set_and_log_unsuccesful_execution(
    response: Response,
    http_status_code: int,  # fastapi status
    error_msg: str,
) -> None:
    """Convenience helper"""
    response.status_code = http_status_code
    logger.error(error_msg)


def create_admin_endpoints(api_version: str) -> APIRouter:
    """Admin/internal endpoints"""
    admin = APIRouter(prefix="/admin", tags=["Admin"])

    # --- endpoints ---

    @admin.get("/probe", status_code=status.HTTP_200_OK, summary="Live probe")
    async def get_status() -> TextualResponse:
        """Returns probe time and user ID"""
        return TextualResponse(message=f"OK on {datetime.now().strftime('%A, %d %B, %Y at %X')} for user {getuser()}")

    return admin  # return APIRouter


def create_document_endpoints(api_version: str, indexer: DocumentIndexer, retriever: HybridRetriever) -> APIRouter:
    """
    Retrieve documents content and statistics on the indexed corpus
    """
    documents = APIRouter(prefix="/documents", tags=["Documents"])

    # --- endpoints ---

    @documents.get(
        "/content/{file_path:path}",
        response_description="Retrieve the full textual content of an indexed document",
        response_model=FileContent,
        responses={
            status.HTTP_200_OK: {"description": "Content retrieved"},
            status.HTTP_400_BAD_REQUEST: {"description": "Invalid path"},
            status.HTTP_404_NOT_FOUND: {"description": "File not indexed"},
            status.HTTP_422_UNPROCESSABLE_CONTENT: {"description": "File cannot be read"},
        },
        summary="Retrieve document full content",
    )
    async def get_document(response: Response, file_path: str) -> FileContent:
        """
        Read the full content of a specific document.

        Use this tool when you know exactly which file contains the
        information you need (e.g. after seeing it in search results).

        Supports: CSV, HTML, Markdown, PDF, TXT.  PDF text is returned as extracted
        plain text (page by page).

        Args:
            file_path: Relative path to the file within the docs directory.
                    Example: "endpoints/users.md"

        Returns:
            The full content of the document.

        Note: when client applications call the endpoint, they can pass the full path directly into the URL:
        - Linux Example: GET /documents//my/path/to/file.txt (Note the double slash: one for the route, one for the root Linux path)
        - Windows Example: GET /documents/C:/my/path/to/file.txt
        However, the 'file_path' input parameter takes its argument relative to the reference directory (envar DEFAULT_DOCS_DIR)
        """

        if not file_path in await asyncio.to_thread(indexer.list_files):
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"file {file_path} not indexed")

        full_path = Path(DEFAULT_DOCS_DIR) / file_path

        # Security: ensure the resolved path stays within docs_dir
        try:
            full_path = full_path.resolve()
            docs_resolved = Path(DEFAULT_DOCS_DIR).resolve()
            if not full_path.is_relative_to(docs_resolved):
                _set_and_log_unsuccesful_execution(
                    response=response,
                    http_status_code=status.HTTP_400_BAD_REQUEST,
                    error_msg=f"Error: Path {full_path} is outside the documentation directory.",
                )
                return FileContent(textual_content="")
        except (OSError, ValueError):
            _set_and_log_unsuccesful_execution(
                response=response,
                http_status_code=status.HTTP_400_BAD_REQUEST,
                error_msg=f"Error: Invalid file path {full_path}",
            )
            return FileContent(textual_content="")

        if not full_path.exists():
            available = await asyncio.to_thread(indexer.list_files)
            suggestions = [f for f in available if file_path.lower() in f.lower()]
            if suggestions:
                msg = f"File not found: '{file_path}'. Did you mean: {', '.join(suggestions[:5])}"
            else:
                msg = f"File not found: '{file_path}'. Use list_docs to see available files."
            _set_and_log_unsuccesful_execution(
                response=response, http_status_code=status.HTTP_404_NOT_FOUND, error_msg=msg
            )
            return FileContent(textual_content="")

        ext = full_path.suffix.lower()
        if not FileType.is_supported(ext):
            msg = f"Error: Unsupported file type '{ext}'."
            msg += f" Supported: {', '.join(FileType.supported_extensions())}"
            _set_and_log_unsuccesful_execution(
                response=response, http_status_code=status.HTTP_400_BAD_REQUEST, error_msg=msg
            )
            return FileContent(textual_content="")

        def _read() -> str:
            if ext == ".pdf":
                import pymupdf

                pdf: Any = cast(Any, pymupdf).open(str(full_path))
                pages: list[str] = []

                def _make_page_text(page_text: str, page_num: int) -> str:
                    return f"--- Page {page_num} ---\n{page_text}"

                for i, page in enumerate(cast(Any, pdf), 1):
                    text = cast(Any, page).get_text().strip()
                    if text:
                        pages.append(_make_page_text(text, i))
                cast(Any, pdf).close()
                return f"[File: {file_path}]\n\n" + "\n\n".join(pages)
            else:
                content = full_path.read_text(encoding="utf-8")
                return f"[File: {file_path}]\n\n{content}"

        try:
            textual_content = await asyncio.to_thread(_read)
            return FileContent(textual_content=textual_content)
        except (UnicodeDecodeError, OSError) as e:
            _set_and_log_unsuccesful_execution(
                response=response, http_status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, error_msg=msg
            )
            return FileContent(textual_content="")

    @documents.get(
        "/list",
        responses={
            status.HTTP_200_OK: {"description": "Ok"},
            status.HTTP_400_BAD_REQUEST: {"description": "Invalid or unsupported extension"},
        },
        summary="List documents (possibly filtered by a supported type)",
    )
    async def list_docs(response: Response, file_ext: str = "") -> FoundDocuments:
        """
        List all indexed documents, optionally filtered by file type.

        Use this tool to discover what documents are available or to find
        the right file to look up.

        Args:
            file_ext: Optional filter — e.g. ".csv, .pdf", ".md", ".txt", ".html".
                    Leave empty to list all files.

        Returns:
            A list of all matching files in the document collection.

        Note: although input parameter 'file_ext' takes a plain str,
                a Pydantic validation occurs further down
        """
        files = await asyncio.to_thread(indexer.list_files)

        if file_ext:
            ft = file_ext if file_ext.startswith(".") else f".{file_ext}"  # ensures leading dot '.'
            if not ft[1:].lower() in FileType.supported_extensions():  # Pydantic validation here
                _set_and_log_unsuccesful_execution(
                    response=response,
                    http_status_code=status.HTTP_400_BAD_REQUEST,
                    error_msg=f"{ft} is not a supported/recognized file extension",
                )
                return FoundDocuments()
            files = [f for f in files if f.lower().endswith(ft.lower())]

        return FoundDocuments(doc_paths=[Path(f) for f in files])

    @documents.get("/stats", status_code=status.HTTP_200_OK, summary="Produce statistics on indexed documents")
    async def get_doc_stats() -> TextualResponse:
        """Produce statistics on indexed documents"""
        stats = await asyncio.to_thread(indexer.get_stats)
        files = await asyncio.to_thread(indexer.list_files)

        lines = [
            f"Documents ({stats['total_files']} files, {stats['total_chunks']} chunks):",
            "",
        ]
        for f in files:
            lines.append(f"  - {f}")

        type_summary = stats.get("file_types", {})
        if type_summary:
            lines.append("")
            lines.append("File types: " + ", ".join(f"{ext}: {n}" for ext, n in sorted(type_summary.items())))

        return TextualResponse(message="\n".join(lines))

    return documents  # return APIRouter


def create_search_endpoints(api_version: str, indexer: DocumentIndexer, retriever: HybridRetriever) -> APIRouter:
    """Search for results against a query amongst the indexed documents"""

    search = APIRouter(prefix="/search", tags=["Search"])

    # --- endpoints ---

    @search.post(
        "/",
        responses={
            status.HTTP_200_OK: {"description": "Ok"},
            status.HTTP_408_REQUEST_TIMEOUT: {"description": "Request timed out"},
            status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "Unrecoverable error"},
        },
        summary="Search documents against a query",
    )
    async def search_docs(response: Response, search: SearchModel) -> SearchResults:
        """
        Search the document collection for information relevant to the query.

        Uses hybrid retrieval (BM25 keyword + semantic vector search) by default
        to find the most relevant chunks across all indexed documents.

        Args:

            query: Natural language search query describing what you're looking for.
                Examples: "authentication flow", "error codes", "rate limiting"

            top_k: Maximum number of results to return (default: 5, max: 20).

            vector_search_weight: bias towards keyword (0.0) vs Semantic (1.0 "semantic")

            mode:  Search strategy — "hybrid" (default), "semantic", or "keyword".

            rrf_fold: Multiplier to preserve fuse resolution in RRF (Hybrid ONLY)

            rrf_k: K parameter in RRF canonical formula (Hybrid ONLY)

        Returns:
            Relevant document chunks with source file, section, and page info.
        """
        query = search.query
        top_k = min(max(1, search.top_k), 20)

        try:
            results = await asyncio.wait_for(
                asyncio.to_thread(
                    retriever.search,
                    query,
                    top_k=top_k,
                    vector_search_weight=search.vector_search_weight,
                    rrf_fold=search.rrf_fold,
                    rrf_k=search.rrf_k,
                ),
                SEARCH_REQUEST_TIMEOUT,
            )
        except asyncio.TimeoutError:
            error_msg = f"Search request for '{query}' timed out after {SEARCH_REQUEST_TIMEOUT} seconds."
            error_msg += " Set/Reset envar SEARCH_REQUEST_TIMEOUT to a higher value and re-start the server"
            raise HTTPException(
                status_code=status.HTTP_408_REQUEST_TIMEOUT,
                detail=f"{error_msg}",
            )
        except Exception as e:
            error_msg = f"Unrecoverable search failure: {traceback.format_exc}"
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"{error_msg}",
            )

        search_results: dict[str, Any] = {"summary": "", "hits": []}

        if not results:
            search_results["summary"] = f"No results found for: '{query}'. Try different search parameters."
            return SearchResults(**search_results)

        search_mode = get_search_mode(vector_search_weight=search.vector_search_weight)
        search_results["summary"] = f"Found {len(results)} relevant chunk(s) for '{query}' (mode={search_mode}):\n"
        for i, r in enumerate(results, 1):
            parts: list[str] = []
            meta = r.get("metadata", {})
            source = meta.get("source", "unknown")
            section = meta.get("section", "")
            page = meta.get("page")
            file_type = meta.get("file_type", "")

            header_parts = [f"Source: {source}"]
            if section:
                header_parts.append(f"Section: {section}")
            if page:
                header_parts.append(f"Page: {page}")
            if file_type:
                header_parts.append(f"Type: {file_type}")

            parts.append(f"--- Result {i} ---")
            parts.append(f"[{' | '.join(header_parts)}]")
            parts.append("")
            parts.append(r.get("content", "").strip())
            parts.append("")

            search_results["hits"].append("\n".join(parts))

        return SearchResults(**search_results)

    @search.get("/index", status_code=status.HTTP_200_OK, summary="Search index statistics")
    async def get_index() -> TextualResponse:
        """
        Return statistics about the current search index.

        Returns:
            Index stats including file count, chunk count, file types,
            embedding model, and last build time.
        """
        stats = await asyncio.to_thread(indexer.get_stats)
        lines = [
            "Index Status:",
            f"  Docs directory : {stats['docs_dir']}",
            f"  Total files    : {stats['total_files']}",
            f"  Total chunks   : {stats['total_chunks']}",
            f"  Embedding model: {stats.get('embedding_model', 'N/A')}",
            f"  Chunk size     : {stats.get('chunk_size', 'N/A')}",
            f"  Chunk overlap  : {stats.get('chunk_overlap', 'N/A')}",
        ]
        ft = stats.get("file_types", {})
        if ft:
            lines.append(f"  File types     : {', '.join(f'{e}: {n}' for e, n in sorted(ft.items()))}")
        last = stats.get("last_build")
        if last:
            ago = time.time() - last
            if ago < 60:
                lines.append(f"  Last build     : {ago:.0f}s ago")
            else:
                lines.append(f"  Last build     : {ago / 60:.1f}min ago")
        return TextualResponse(message="\n".join(lines))

    @search.patch(
        "/index",
        status_code=status.HTTP_200_OK,
        summary="Rebuild search index: incremental (default) or full (force: True)",
    )
    async def rebuild_index(force: bool = False) -> TextualResponse:
        """
        Rebuild the search index.

        By default performs an incremental rebuild — only new or changed
        files are re-processed.  Set force=True to fully reindex everything.

        Args:
            force: If True, discard existing index and rebuild from scratch.

        Returns:
            Status message with the number of chunks indexed.
        """
        await asyncio.to_thread(retriever.clear_cache)
        num = await asyncio.to_thread(indexer.build, force=force)
        files = await asyncio.to_thread(indexer.list_files)

        msg = f"Index {'fully rebuilt' if force else 'updated'}: "
        msg += f"{num} chunks from {len(files)} files."
        return TextualResponse(message=msg)

    return search  # return APIRouter
