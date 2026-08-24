"""
Document indexer.

Loads documents of various types (CSV, HTML, MD, PDF, TXT), chunks them,
embeds with sentence-transformers, and stores vectors in
ChromaDB.
Also maintains an in-memory BM25 index for keyword based retrieval.
"""

import csv
import logging
import re
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, TypeVar, cast

import chromadb

T = TypeVar("T")
from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from rank_bm25 import BM25Okapi

from app.engine.cache import BM25Cache, DocumentHashCache, file_content_hash
from app.engine.loader_registry import LoaderRegistry
from app.models import FileType
from hsr_config import (
    CHROMA_COLLECTION_NAME,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DEFAULT_CHROMA_DIR,
    EMBEDDING_MODEL,
    EMBEDDING_MODEL_SIZE,
)

logger = logging.getLogger(__name__)


def _langchain_document(page_content: str, metadata: dict[str, Any] | None = None) -> Document:
    document_cls: Any = Document
    return cast(Document, document_cls(page_content=page_content, metadata=metadata or {}))


# ---------------------------------------------------------------------------
# BM25 tokenizer (keyword search)
# ---------------------------------------------------------------------------


def tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer for BM25."""
    text = text.lower()
    tokens = re.findall(r"[a-z0-9_\-]+", text)
    return tokens


# ---------------------------------------------------------------------------
# Document loaders — one function per file type
# ---------------------------------------------------------------------------


def _load_markdown(path: Path) -> list[Document]:
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError) as exc:
        logger.warning("Could not read %s: %s", path, exc)
        return []

    # Stage 1: split by markdown headers to preserve structure
    headers_to_split = [
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
        ("####", "h4"),
    ]
    header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split, strip_headers=False)
    header_docs = header_splitter.split_text(text)

    # Carry header metadata forward
    for doc in header_docs:
        parts = [doc.metadata.get(k, "") for k in ("h1", "h2", "h3", "h4") if doc.metadata.get(k)]
        doc.metadata["section"] = " > ".join(parts) if parts else path.stem

    return header_docs


def _load_pdf(path: Path) -> list[Document]:
    try:
        import pymupdf  # noqa: F811

        docs: list[Document] = []
        pdf: Any = cast(Any, pymupdf).open(str(path))

        def _make_doc(page_text: str, page_num: int) -> Document:
            return _langchain_document(page_text, metadata={"page": page_num})

        for page_num, page in enumerate(cast(Any, pdf), 1):
            text = cast(Any, page).get_text()
            if text.strip():
                docs.append(_make_doc(text, page_num))
        cast(Any, pdf).close()
        return docs
    except Exception as exc:
        logger.warning("Could not read PDF %s: %s", path, exc)
        return []


def _load_text(path: Path) -> list[Document]:
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError) as exc:
        logger.warning("Could not read %s: %s", path, exc)
        return []
    if not text.strip():
        return []
    return [_langchain_document(text)]


def _load_html(path: Path) -> list[Document]:
    try:
        from bs4 import BeautifulSoup

        raw = path.read_text(encoding="utf-8")
        soup = BeautifulSoup(raw, "lxml")
        title = soup.title.string if soup.title else path.stem
        text = soup.get_text(separator="\n", strip=True)
        if not text:
            return []
        return [_langchain_document(text, metadata={"title": title})]
    except Exception as exc:
        logger.warning("Could not read HTML %s: %s", path, exc)
        return []


def _load_csv(path: Path) -> list[Document]:
    try:
        with path.open(encoding="utf-8", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            headers = reader.fieldnames or []
            docs: list[Document] = []
            for row_index, row in enumerate(reader, start=1):
                if not row:
                    continue
                row_items = [f"{key}: {value}" for key, value in row.items() if value is not None and value != ""]
                if not row_items:
                    continue
                content = "\n".join(row_items)
                metadata = {
                    "row": row_index,
                    "headers": ", ".join(headers) if headers else path.stem,
                }
                docs.append(_langchain_document(content, metadata=metadata))
            return docs
    except Exception as exc:
        logger.warning("Could not read CSV %s: %s", path, exc)
        return []


LOADER_REGISTRY = LoaderRegistry.from_file_types(
    {
        FileType.CSV: _load_csv,
        FileType.HTML: _load_html,
        FileType.MD: _load_markdown,
        FileType.PDF: _load_pdf,
        FileType.TXT: _load_text,
    }
)


def load_file(path: Path) -> list[Document]:
    """Load a file and return a list of LangChain Documents."""
    loader = LOADER_REGISTRY.get_loader_for_path(path)
    if loader is None:
        logger.debug("Unsupported file type: %s", path.suffix.lower())
        return []
    return loader(path)


# ---------------------------------------------------------------------------
# Embedding helper
# ---------------------------------------------------------------------------

_embedding_fn: Any | None = None


def _get_embedding_fn() -> Any:
    """Lazy-init the HuggingFace embedding function (heavy import)."""
    global _embedding_fn
    if _embedding_fn is None:
        from langchain_huggingface import HuggingFaceEmbeddings

        _embedding_fn = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        logger.info("Loaded embedding model: %s", EMBEDDING_MODEL)
    return _embedding_fn


# ===================================================================
# DocumentIndexer
# ===================================================================


class DocumentIndexer:
    """
    Full document indexing pipeline:
      1. Scan docs_dir for supported files and produce (LangChain-)Documents
      2. Chunk each Document.
      3. Embed and store in ChromaDB (persistent).
      4. Maintain BM25 corpus in memory for keyword search.

    Supports incremental indexing via content-hash caching.

    Usage:
        indexer = DocumentIndexer("/path/to/docs")
        indexer.build()                 # incremental
        indexer.build(force=True)       # full rebuild
    """

    def __init__(self, docs_dir: str, chroma_dir: str = DEFAULT_CHROMA_DIR):
        self.docs_dir = Path(docs_dir)
        self.chroma_dir = Path(chroma_dir)

        # Lock to serialize ChromaDB calls.  ChromaDB's PersistentClient
        # uses SQLite + hnswlib internally, with a background consumer thread
        # that processes the write-ahead log into HNSW segments
        # (Hierarchical Navigable Small World).
        # Concurrent collection operations can collide with the consumer and trigger
        # "resource deadlock would occur" errors.  This lock plus the retry
        # wrapper (_chroma_call) together handle the contention.
        self._lock = threading.Lock()

        # Max retries for transient ChromaDB deadlock errors
        self._max_retries = 5
        self._retry_base_delay = 0.2  # seconds

        # Persistent vector store
        self._chroma_client = chromadb.PersistentClient(path=str(self.chroma_dir))
        self._embedding_fn: Any | None = None  # lazy-loaded on first use

        # hnsw:batch_size controls ChromaDB's internal brute-force buffer that
        # accumulates records before flushing them to the persistent HNSW index.
        # The default of 100 is too small — when the embeddings-queue consumer
        # receives a batch of >100 records it overflows the buffer, causing
        # "Index with capacity 100 … cannot add N records" errors.  Set this
        # high enough that the buffer never fills before a flush.
        #
        # get_or_create_collection does NOT update metadata on an existing
        # collection, so to detect a stale batch_size and recreate.
        desired_metadata: dict[str, Any] = {
            "hnsw:space": "cosine",
            "hnsw:batch_size": 10000,
            "hnsw:sync_threshold": 10000,
        }
        self._collection = self._chroma_client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            metadata=desired_metadata,
        )
        existing_batch_size_value = (self._collection.metadata or {}).get(
            "hnsw:batch_size",
            100,  # ChromaDB default
        )
        existing_batch_size = int(cast(Any, existing_batch_size_value))
        if existing_batch_size < int(desired_metadata["hnsw:batch_size"]):
            logger.warning(
                "Existing collection has hnsw:batch_size=%s (need %s). "
                "Deleting and recreating collection to apply new settings.",
                existing_batch_size,
                desired_metadata["hnsw:batch_size"],
            )
            self._chroma_client.delete_collection(CHROMA_COLLECTION_NAME)
            self._collection = self._chroma_client.create_collection(
                name=CHROMA_COLLECTION_NAME,
                metadata=desired_metadata,
            )

        # Caching
        self._hash_cache: DocumentHashCache = DocumentHashCache()
        self._bm25_cache: BM25Cache = BM25Cache()

        # BM25
        self.bm25: BM25Okapi | None = None
        self._bm25_texts: list[str] = []  # parallel to _bm25_ids
        self._bm25_ids: list[str] = []
        self._bm25_metadatas: list[dict[str, Any]] = []

        # Text splitter for oversized chunks
        self._size_splitter: Any = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
        )

        # Stats
        self._last_build_time: float | None = None
        self._file_type_counts: Counter[str] = Counter()

    def _ensure_embedding_fn(self) -> Any:
        """Lazy-load the embedding model on first use."""
        if self._embedding_fn is None:
            self._embedding_fn = _get_embedding_fn()
        return self._embedding_fn

    def _chroma_call(self, fn: Callable[[], T]) -> T:
        """Execute *fn* under the lock with retry on ChromaDB deadlock errors.

        ChromaDB's internal background consumer thread can collide with
        queries on the shared HNSW index or SQLite database, producing
        "resource deadlock would occur" errors.  Because the consumer runs
        inside ChromaDB (outside our control), our threading.Lock alone
        cannot prevent all collisions.  This wrapper catches transient
        deadlock errors and retries with exponential back-off.
        """
        last_exc: BaseException | None = None
        for attempt in range(self._max_retries):
            try:
                with self._lock:
                    return fn()
            except Exception as exc:
                if "resource deadlock" not in str(exc).lower():
                    raise
                last_exc = exc
                delay = self._retry_base_delay * (2**attempt)
                logger.warning(
                    "ChromaDB deadlock (attempt %d/%d), retrying in %.1fs: %s",
                    attempt + 1,
                    self._max_retries,
                    delay,
                    exc,
                )
                time.sleep(delay)
        raise last_exc  # type: ignore[misc]

    def _stabilize(self) -> None:
        """Wait for ChromaDB's background consumer to finish processing.

        After building/loading the index, the internal consumer thread may
        still be flushing the write-ahead log into HNSW segments.  Querying
        while this is in-flight causes deadlocks.  This method polls
        collection.count() until the value stabilises, then runs a quick
        test query to confirm the HNSW index is ready.
        """
        logger.info("Warming up ChromaDB index (waiting for consumer to settle)...")
        prev_count = -1
        stable_ticks = 0
        for _ in range(30):  # up to ~15 seconds
            try:
                with self._lock:
                    cur_count = self._collection.count()
            except Exception:
                time.sleep(0.5)
                continue
            if cur_count == prev_count:
                stable_ticks += 1
                if stable_ticks >= 3:
                    break
            else:
                stable_ticks = 0
            prev_count = cur_count
            time.sleep(0.5)

        # Run a test query to force the HNSW segment to be fully ready
        if prev_count > 0:
            try:
                dummy_emb = [0.0] * EMBEDDING_MODEL_SIZE  # dimension of all-MiniLM-L6-v2
                self._chroma_call(
                    lambda: self._collection.query(
                        query_embeddings=[cast(Any, dummy_emb)],
                        n_results=1,
                        include=["documents"],
                    )
                )
            except Exception as exc:
                logger.warning("ChormaDB stabilize test query failed (non-fatal): %s", exc)

        logger.info("ChromaDB is now stable (%d chunks)", prev_count)

        # Eagerly load the embedding model so the first real query doesn't
        # block while the model downloads / initializes.
        self._ensure_embedding_fn()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self, force: bool = False) -> int:
        """
        Build or update the index.

        If *force* is True the entire index is rebuilt from scratch.
        Otherwise only new / modified files are processed.
        Deleted files are pruned.

        Returns the total number of chunks in the index.
        """
        start = time.time()

        if not self.docs_dir.exists():
            logger.warning("Docs directory does not exist: %s", self.docs_dir)
            return 0

        if force:
            logger.info("Force rebuild — clearing existing index")
            self._chroma_call(lambda: self._collection.delete(where=cast(Any, {"source": {"$ne": ""}})))
            self._hash_cache.clear()
            self._bm25_cache.invalidate()

        # Discover files
        current_files = self._discover_files()
        logger.info("Discovered %d supported files in %s", len(current_files), self.docs_dir)

        # Bulk-load all cached hashes + mtimes in one query (avoids thousands of
        # individual SQLite lookups).
        cached_entries = self._hash_cache.get_all()  # {rel: (hash, mtime)}
        cached_paths = set(cached_entries.keys())
        current_rel_paths = {self._rel(f) for f in current_files}

        deleted = cached_paths - current_rel_paths
        new_or_changed: list[Path] = []

        for fpath in current_files:
            rel = self._rel(fpath)
            entry = cached_entries.get(rel)
            if entry is None:
                # New file — never indexed
                new_or_changed.append(fpath)
                continue
            old_hash, old_mtime = entry
            # Cheap check: skip SHA-256 if modification time hasn't changed
            try:
                cur_mtime = fpath.stat().st_mtime
            except OSError:
                new_or_changed.append(fpath)
                continue
            if old_mtime is not None and cur_mtime == old_mtime:
                continue  # mtime unchanged → file unchanged
            # mtime changed (or unknown) — fall back to content hash
            new_hash = file_content_hash(fpath)
            if old_hash != new_hash:
                new_or_changed.append(fpath)

        logger.info(
            "Index delta — new/changed: %d, deleted: %d, unchanged: %d",
            len(new_or_changed),
            len(deleted),
            len(current_rel_paths) - len(new_or_changed),
        )

        # Fast path: nothing changed — try loading BM25 from cache
        if not force and not new_or_changed and not deleted:
            # Reuse the already-loaded entries dict instead of re-querying SQLite
            fingerprint = BM25Cache.compute_fingerprint_from_entries(cached_entries)
            if self._load_bm25_from_cache(fingerprint):
                elapsed = time.time() - start
                self._last_build_time = time.time()
                self._file_type_counts = Counter(f.suffix.lower() for f in current_files)
                total = self._chroma_call(self._collection.count)
                self._stabilize()
                logger.info(
                    "Index ready (from cache): %d chunks from %d files (%.1fs)",
                    total,
                    len(current_files),
                    elapsed,
                )
                return total

        # Remove deleted
        for rel in deleted:
            self._remove_file_chunks(rel)
            self._hash_cache.remove(rel)

        # Process new/changed
        total_new_chunks = 0
        for i, fpath in enumerate(new_or_changed, 1):
            rel = self._rel(fpath)
            logger.info("[%d/%d] Processing %s", i, len(new_or_changed), rel)
            # Remove old chunks for this file (if updating)
            self._remove_file_chunks(rel)
            # Load, chunk, embed, store
            n = self._index_file(fpath)
            total_new_chunks += n
            # Update hash cache (store mtime so future startups can skip
            # the expensive SHA-256 when the file hasn't been touched).
            try:
                mtime = fpath.stat().st_mtime
            except OSError:
                mtime = None
            self._hash_cache.put(rel, file_content_hash(fpath), mtime=mtime)

        # Rebuild BM25 from ChromaDB contents and save cache
        self._rebuild_bm25(save_cache=True)

        elapsed = time.time() - start
        self._last_build_time = time.time()
        self._file_type_counts = Counter(f.suffix.lower() for f in current_files)

        total = self._chroma_call(self._collection.count)

        # Let the consumer finish flushing the WAL into HNSW before we
        # accept queries — prevents deadlock between consumer and query
        # threads on the shared HNSW index.
        self._stabilize()

        logger.info(
            "Index ready: %d chunks from %d files (%.1fs, %d new chunks)",
            total,
            len(current_files),
            elapsed,
            total_new_chunks,
        )
        return total

    def search_vector(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        """Semantic search via ChromaDB embeddings."""
        query_embedding = cast(list[float], self._ensure_embedding_fn().embed_query(query))

        def _query() -> Any | None:
            if self._collection.count() == 0:
                return None
            return self._collection.query(
                query_embeddings=[cast(Any, query_embedding)],
                n_results=min(top_k, self._collection.count()),
                include=["documents", "metadatas", "distances"],
            )

        results = self._chroma_call(_query)
        if results is None:
            return []
        out = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            out.append(
                {
                    "content": doc,
                    "metadata": meta,
                    "score": 1 - dist,  # cosine distance → similarity
                }
            )
        return out

    def search_bm25(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        """Keyword search via BM25."""
        if not self.bm25 or not self._bm25_texts:
            return []
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        scored = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        out = []
        for idx, score in scored:
            if score <= 0:
                break
            out.append(
                {
                    "content": self._bm25_texts[idx],
                    "metadata": self._bm25_metadatas[idx],
                    "score": float(score),
                }
            )
        return out

    def list_files(self) -> list[str]:
        """Return sorted list of all indexed file relative paths."""

        def _get_meta() -> list[dict[str, Any]] | None:
            if self._collection.count() == 0:
                return None
            return cast(list[dict[str, Any]], self._collection.get(include=["metadatas"])["metadatas"])

        all_meta = self._chroma_call(_get_meta)
        if all_meta is None:
            return []
        return sorted({m.get("source", "") for m in all_meta if m.get("source")})

    def get_stats(self) -> dict[str, Any]:
        total_chunks = int(self._chroma_call(self._collection.count))
        return {
            "docs_dir": str(self.docs_dir),
            "total_files": len(self.list_files()),
            "total_chunks": total_chunks,
            "file_types": dict(self._file_type_counts),
            "last_build": self._last_build_time,
            "embedding_model": EMBEDDING_MODEL,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rel(self, path: Path) -> str:
        """Relative path string (forward slashes)."""
        return str(path.relative_to(self.docs_dir)).replace("\\", "/")

    def _discover_files(self) -> list[Path]:
        """Find all supported files under docs_dir."""
        files: list[Path] = []
        for ext in FileType.supported_extensions():
            files.extend(self.docs_dir.rglob(f"*.{ext}"))
        return sorted(set(files))

    def _remove_file_chunks(self, rel_path: str) -> None:
        """Delete all ChromaDB documents belonging to a source file."""
        try:

            def _delete_for_path(path_value: str) -> Any:
                return self._collection.delete(where=cast(Any, {"source": path_value}))

            self._chroma_call(lambda: _delete_for_path(rel_path))
        except Exception:
            # Collection may be empty or source may not exist
            pass

    def _index_file(self, path: Path) -> int:
        """Load, chunk, embed, and store a single file. Returns chunk count."""
        rel = self._rel(path)
        ext = path.suffix.lower()

        raw_docs = load_file(path)
        if not raw_docs:
            return 0

        # Stage 2: split oversized chunks
        all_chunks: list[Document] = []
        for doc in raw_docs:
            if len(doc.page_content) > CHUNK_SIZE:
                sub_chunks = self._size_splitter.split_documents([doc])
                all_chunks.extend(sub_chunks)
            else:
                all_chunks.append(doc)

        if not all_chunks:
            return 0

        # Prepare for ChromaDB
        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict[str, Any]] = []
        embeddings: list[list[float]] = []

        for i, chunk in enumerate(all_chunks):
            chunk_id = f"{rel}::{i}"
            meta = {
                "source": rel,
                "file_type": ext,
                "chunk_index": i,
            }
            # Carry forward loader metadata
            if chunk.metadata.get("section"):
                meta["section"] = chunk.metadata["section"]
            if chunk.metadata.get("page"):
                meta["page"] = chunk.metadata["page"]
            if chunk.metadata.get("title"):
                meta["title"] = chunk.metadata["title"]

            ids.append(chunk_id)
            documents.append(chunk.page_content)
            metadatas.append(meta)

        # Batch embed
        embeddings = self._ensure_embedding_fn().embed_documents(documents)

        # Upsert into ChromaDB
        self._chroma_call(
            lambda: self._collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=cast(Any, metadatas),
                embeddings=cast(Any, embeddings),
            )
        )
        return len(ids)

    def _load_bm25_from_cache(self, fingerprint: str) -> bool:
        """Try to restore BM25 from disk cache. Returns True on success."""
        data = self._bm25_cache.load(fingerprint)
        if data is None:
            return False
        self._bm25_texts = cast(list[str], data["texts"])
        self._bm25_ids = cast(list[str], data["ids"])
        self._bm25_metadatas = cast(list[dict[str, Any]], data["metadatas"])
        tokenized = [tokenize(text) for text in self._bm25_texts]
        self.bm25 = BM25Okapi(tokenized)
        return True

    def _rebuild_bm25(self, save_cache: bool = False) -> None:
        """Rebuild the in-memory BM25 index from ChromaDB contents."""
        count = int(self._chroma_call(self._collection.count))
        if count == 0:
            self.bm25 = None
            self._bm25_texts = []
            self._bm25_ids = []
            self._bm25_metadatas = []
            return

        def _load_all_data() -> Any:
            return self._collection.get(include=["documents", "metadatas"])

        all_data = self._chroma_call(_load_all_data)
        self._bm25_ids = cast(list[str], all_data["ids"])
        self._bm25_texts = cast(list[str], all_data["documents"])
        self._bm25_metadatas = cast(list[dict[str, Any]], all_data["metadatas"])

        tokenized = [tokenize(text) for text in self._bm25_texts]
        self.bm25 = BM25Okapi(tokenized)
        logger.info("BM25 index rebuilt with %d documents", len(tokenized))

        if save_cache:
            fingerprint = BM25Cache.compute_fingerprint(self._hash_cache)
            self._bm25_cache.save(self._bm25_texts, self._bm25_ids, self._bm25_metadatas, fingerprint)
