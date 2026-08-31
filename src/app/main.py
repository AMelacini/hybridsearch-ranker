import asyncio
import logging
import sys
from contextlib import asynccontextmanager, suppress
from typing import AsyncGenerator

import dotenv
import uvicorn
from fastapi import APIRouter, FastAPI
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute

dotenv.load_dotenv()

from typing import Any

from app.engine.indexer import DocumentIndexer
from app.engine.retriever import HybridRetriever
from app.routers import create_admin_endpoints, create_document_endpoints, create_search_endpoints
from hsr_config import (
    DEFAULT_CHROMA_DIR,
    DEFAULT_DOCS_DIR,
    REST_SERVER_PORT,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr,
)

logger = logging.getLogger("hsr-server")


# === Setup FastAPI app ===


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # startup steps - e.g. external DB connections etc...
    port = REST_SERVER_PORT
    logger.info("═" * 60)
    logger.info("✓ Hybrid Search Ranker backend is READY")
    logger.info(f"  Listening on http://0.0.0.0:{port}")
    logger.info(f"  API docs: http://localhost:{port}/docs")
    logger.info("═" * 60)

    try:
        yield
    except asyncio.CancelledError:
        # Uvicorn is already in its own shutdown sequence. Canceling every task in the
        # event loop here includes Uvicorn internals and leads to noisy traceback
        # (spurious errors) on Ctrl+C.
        logger.info("Shutdown signal received; exiting lifespan cleanly")
        return
    finally:
        # Only close resources that this app owns. Do not cancel unrelated tasks from the
        # global event loop; that is what triggers the spurious cancellation errors.
        logger.info("Application shutdown complete")


description = """
## Hybrid Search Ranker (HSR)
Given a set of files of a supported type (currently CSV, HTML, MD, PDF, TXT),
HSR index them and store the extracted information in a local ChromaDB instance

### Indexing
Indexing a file consists of:
- loading and extracting its textual content
- splitting the content in chunks of predefined size and with overalps
- create embedding for seamntic search
- organize all the information to support Keyword Search (BM25), Semantic Search (Vector)
    and a combination of the two - a.k.a. Hybrid Search

All the information is stored in a local ChromaDB instance.

### Searching
Natural Language queries trigger a search through the indexed corpus and the top ranking document parts are retrieved and
can be injected in a LLM context, part of a RAG (Retrieval Augmented Generation) data pipeline.
Three types of searches are supported:
- full Keyword (BM25)
- full Semantic (Vector search)
- Hybrid with RRF (Reciprocal Rank Fusion) rnaked results

### Caching
To improve performance, 3 level of caching is implemented.
- Document Hash: file_path -> (content_hash, mtime) is cached for incremental indexing
- Query-Results: (query, results) info is cached (with TTL or LRU according to configuration)
- BM25 Cache: reconstruct the BM25 search index (Keyword search) from valid cached info only when changes are detected
"""


# === Setup Indexing ===


def _build_backend_dependencies() -> tuple[DocumentIndexer, HybridRetriever]:
    """Create backend runtime objects lazily to avoid import-time side effects."""
    indexer = DocumentIndexer(DEFAULT_DOCS_DIR, DEFAULT_CHROMA_DIR)
    num_chunks = indexer.build()
    logger.info("Index ready: %d chunks at startup", num_chunks)
    retriever = HybridRetriever(indexer)
    return indexer, retriever


app = FastAPI(
    title="Hybrid Search Ranker backend",
    description=description,
    version="0.1.0",
    contact={
        "name": "Alberto Melacini",
        "url": "https://github.com/amelacini",
    },
    lifespan=lifespan,
)

indexer: DocumentIndexer | None = None
retriever: HybridRetriever | None = None


# NOTE: FastAPI creates OpenAPI 3.1.0 specifications.
#       Not all tools for automated language specific APIs generation support 3.1.0 and abort at startup (for no reason!)
#       The following property is to "reassure" ;-) those tools
#       Uncomment the line below if required
# app.openapi_version = "3.0.0"

# === API versioning - PLACEHOLDER

DUMMY_API_VERSION = "v1"  # API versioning is not yet implemented
top_router = APIRouter(prefix=f"/{DUMMY_API_VERSION}")

# --- Register Supported components (routers)


@app.on_event("startup")
async def startup_backend() -> None:
    global indexer, retriever
    if indexer is None or retriever is None:
        indexer, retriever = _build_backend_dependencies()

    sub_router = create_admin_endpoints(api_version=DUMMY_API_VERSION)
    top_router.include_router(sub_router)

    sub_router = create_document_endpoints(
        api_version=DUMMY_API_VERSION,
        indexer=indexer,
        retriever=retriever,
    )
    top_router.include_router(sub_router)

    sub_router = create_search_endpoints(
        api_version=DUMMY_API_VERSION,
        indexer=indexer,
        retriever=retriever,
    )
    top_router.include_router(sub_router)

    app.include_router(top_router)

    for route in app.routes:
        if isinstance(route, APIRoute):
            route.operation_id = route.name


@app.get("/")  # for sanity check
async def home() -> JSONResponse:
    return JSONResponse(content={"message": "Hello from Hybrid Search Ranker tool backend"})


if __name__ == "__main__":
    try:
        uvicorn.run(app, host="0.0.0.0", port=REST_SERVER_PORT)
    except KeyboardInterrupt as e:
        print(f"\nHybrid Search Ranker backend gracefully terminated ({type(e).__name__})")
