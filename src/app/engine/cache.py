"""
Caching layers.

Provides two caches:
  - DocumentHashCache: tracks file content hashes for incremental indexing.
  - QueryCache: caches search results to avoid redundant vector lookups.
  - BM25Cache: when no changes are detected, reconstruct BM25 index without
               retrieving info from ChromaDB on startup.

Uses Redis when available - falls back to SQLite / in-memory LRU otherwise.
"""

import hashlib
import json
import logging
import sqlite3
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, cast

from hsr_config import CACHE_DIR, QUERY_CACHE_TTL, REDIS_ENABLED, REDIS_URL

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Redis connection helper
# ---------------------------------------------------------------------------

_redis_client: Any | None = None
_redis_checked = False


def _get_redis() -> Any | None:
    """Return a Redis client or None if unavailable."""
    global _redis_client, _redis_checked
    if _redis_checked:
        return _redis_client
    _redis_checked = True

    if not REDIS_ENABLED:
        logger.info("Redis disabled by configuration")
        return None

    try:
        import redis as redis_lib

        client = redis_lib.Redis.from_url(REDIS_URL, decode_responses=True, socket_timeout=2)
        client.ping()
        _redis_client = client
        logger.info("Connected to Redis at %s", REDIS_URL)
    except Exception as exc:
        logger.warning("Redis unavailable (%s) — falling back to local cache", exc)
        _redis_client = None

    return _redis_client


# ---------------------------------------------------------------------------
# File Content hashing
# ---------------------------------------------------------------------------


def file_content_hash(path: Path) -> str:
    """Return a SHA-256 hex digest of a file's contents."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


# ===================================================================
# DocumentHashCache — tracks which files have changed
# ===================================================================


class DocumentHashCache:
    """
    Stores {file_path: (content_hash, mtime)} for incremental indexing.

    On startup the full cache is bulk-loaded into memory via get_all(),
    avoiding thousands of SQLite queries.  File modification time
    (mtime) is stored alongside the content hash so to skip the
    expensive SHA-256 when the mtime hasn't changed.

    Uses Redis hash if available, otherwise SQLite.
    """

    _REDIS_KEY = "rag:doc_hashes"

    def __init__(self) -> None:
        self._redis = _get_redis()
        self._db_path = Path(CACHE_DIR) / "doc_hashes.db"
        if self._redis is None:
            self._init_sqlite()

    # -- SQLite fallback ---------------------------------------------------

    def _init_sqlite(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self._db_path))
        conn.execute("CREATE TABLE IF NOT EXISTS hashes (path TEXT PRIMARY KEY, hash TEXT NOT NULL, mtime REAL)")
        # Migrate: add mtime column if missing (existing installs)
        try:
            conn.execute("SELECT mtime FROM hashes LIMIT 1")
        except sqlite3.OperationalError:
            conn.execute("ALTER TABLE hashes ADD COLUMN mtime REAL")
        conn.commit()
        conn.close()

    def _sqlite_conn(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self._db_path))

    # -- Public API --------------------------------------------------------

    def get(self, path: str) -> str | None:
        """Return stored hash for *path*, or None."""
        if self._redis:
            value = self._redis.hget(self._REDIS_KEY, path)
            return cast(str | None, value)
        conn = self._sqlite_conn()
        row = conn.execute("SELECT hash FROM hashes WHERE path = ?", (path,)).fetchone()
        conn.close()
        return row[0] if row else None

    def get_all(self) -> dict[str, tuple[str, float | None]]:
        """Bulk-load all entries as {path: (hash, mtime)}.

        Much faster than calling get() many times on startup.
        """
        if self._redis:
            # Redis stores hash only (no mtime); return mtime=None
            raw = self._redis.hgetall(self._REDIS_KEY)
            return {k: (v, None) for k, v in raw.items()}
        conn = self._sqlite_conn()
        rows = conn.execute("SELECT path, hash, mtime FROM hashes").fetchall()
        conn.close()
        return {r[0]: (r[1], r[2]) for r in rows}

    def put(self, path: str, content_hash: str, mtime: float | None = None) -> None:
        if self._redis:
            self._redis.hset(self._REDIS_KEY, path, content_hash)
        else:
            conn = self._sqlite_conn()
            conn.execute(
                "INSERT OR REPLACE INTO hashes (path, hash, mtime) VALUES (?, ?, ?)",
                (path, content_hash, mtime),
            )
            conn.commit()
            conn.close()

    def remove(self, path: str) -> None:
        if self._redis:
            self._redis.hdel(self._REDIS_KEY, path)
        else:
            conn = self._sqlite_conn()
            conn.execute("DELETE FROM hashes WHERE path = ?", (path,))
            conn.commit()
            conn.close()

    def all_paths(self) -> set[str]:
        """Return all cached file paths."""
        if self._redis:
            return set(self._redis.hkeys(self._REDIS_KEY))
        conn = self._sqlite_conn()
        rows = conn.execute("SELECT path FROM hashes").fetchall()
        conn.close()
        return {r[0] for r in rows}

    def clear(self) -> None:
        if self._redis:
            self._redis.delete(self._REDIS_KEY)
        else:
            conn = self._sqlite_conn()
            conn.execute("DELETE FROM hashes")
            conn.commit()
            conn.close()


# ===================================================================
# QueryCache — caches search results with TTL
# ===================================================================


class QueryCache:
    """
    Caches serialised search results keyed by (query, top_k, <search parameters>).
    Uses Redis with TTL if available, otherwise an in-memory LRU (OrderedDict).
    """

    _REDIS_PREFIX = "rag:query:"
    _MAX_MEMORY_ENTRIES = 256

    def __init__(self, ttl: int = QUERY_CACHE_TTL) -> None:
        self._redis = _get_redis()
        self._ttl = ttl
        # In-memory fallback: OrderedDict used as LRU
        self._mem: OrderedDict[str, tuple[float, Any]] = OrderedDict()  # LRU

    @staticmethod
    def _cache_key(query: str, top_k: int, vector_search_weight: float, **kwargs: int) -> str:
        if "rrf_fold" in kwargs and "rrf_k" in kwargs:  # Hybrid Search
            raw = f"{query}|{top_k}|{vector_search_weight}|{kwargs['rrf_fold']}|{kwargs['rrf_k']}"
        else:
            raw = f"{query}|{top_k}|{vector_search_weight}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, query: str, top_k: int, vector_search_weight: float, **kwargs: int) -> list[dict[str, Any]] | None:
        key = self._cache_key(query, top_k, vector_search_weight, **kwargs)
        if self._redis:
            raw = self._redis.get(self._REDIS_PREFIX + key)
            if raw:
                return cast(list[dict[str, Any]], json.loads(raw))
            return None

        entry = self._mem.get(key)
        if entry is None:
            return None
        ts, data = entry
        if time.time() - ts > self._ttl:
            del self._mem[key]
            return None
        self._mem.move_to_end(key)  # LRU
        return cast(list[dict[str, Any]], data)

    def set(
        self,
        query: str,
        top_k: int,
        vector_search_weight: float,
        results: list[dict[str, Any]],
        **kwargs: int,
    ) -> None:
        key = self._cache_key(query, top_k, vector_search_weight, **kwargs)
        if self._redis:
            self._redis.setex(self._REDIS_PREFIX + key, self._ttl, json.dumps(results))
        else:
            if len(self._mem) >= self._MAX_MEMORY_ENTRIES:
                self._mem.popitem(last=False)  # evict oldest
            self._mem[key] = (time.time(), results)

    def clear(self) -> None:
        if self._redis:
            # Delete all query cache keys
            cursor = 0
            while True:
                cursor, keys = self._redis.scan(cursor, match=self._REDIS_PREFIX + "*", count=100)
                if keys:
                    self._redis.delete(*keys)
                if cursor == 0:
                    break
        else:
            self._mem.clear()


# ===================================================================
# BM25Cache — persists BM25 corpus to disk to avoid rebuilding
# ===================================================================


class BM25Cache:
    """
    Serialises the BM25 corpus (texts, ids, metadata) and a fingerprint
    so that the BM25 index can be reconstructed without re-reading
    ChromaDB on startup when nothing has changed.

    The fingerprint is a hash of all (path, content_hash) pairs from the
    DocumentHashCache, so any file change invalidates the cache.
    """

    def __init__(self, cache_dir: str = CACHE_DIR):
        self._cache_path = Path(cache_dir) / "bm25_cache.pkl"
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def compute_fingerprint(hash_cache: "DocumentHashCache") -> str:
        """Deterministic fingerprint of all indexed file hashes."""
        all_paths = sorted(hash_cache.all_paths())
        parts = []
        for p in all_paths:
            h = hash_cache.get(p)
            parts.append(f"{p}:{h}")
        return hashlib.sha256("\n".join(parts).encode()).hexdigest()

    @staticmethod
    def compute_fingerprint_from_entries(
        entries: dict[str, tuple[str, float | None]],
    ) -> str:
        """Compute fingerprint from a pre-loaded get_all() result.

        Avoids re-querying SQLite for thousands of rows when the entries are
        already in memory from the change-detection pass.
        """
        parts = [f"{p}:{h}" for p, (h, _mtime) in sorted(entries.items())]
        return hashlib.sha256("\n".join(parts).encode()).hexdigest()

    def load(self, expected_fingerprint: str) -> dict[str, Any] | None:
        """
        Load cached BM25 data if the fingerprint matches.

        Returns dict with keys: texts, ids, metadatas, fingerprint
        or None if the cache is missing / stale.
        """
        if not self._cache_path.exists():
            return None
        try:
            import pickle

            with open(self._cache_path, "rb") as f:
                data = pickle.load(f)
            if not isinstance(data, dict) or data.get("fingerprint") != expected_fingerprint:
                logger.info("BM25 cache fingerprint mismatch — will rebuild")
                return None
            logger.info("BM25 cache loaded (%d documents)", len(data.get("texts", [])))
            return data
        except Exception as exc:
            logger.warning("Could not load BM25 cache: %s", exc)
            return None

    def save(self, texts: list[str], ids: list[str], metadatas: list[dict[str, Any]], fingerprint: str) -> None:
        """Persist BM25 corpus to disk."""
        import pickle

        data = {
            "texts": texts,
            "ids": ids,
            "metadatas": metadatas,
            "fingerprint": fingerprint,
        }
        try:
            with open(self._cache_path, "wb") as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info("BM25 cache saved (%d documents)", len(texts))
        except Exception as exc:
            logger.warning("Could not save BM25 cache: %s", exc)

    def invalidate(self) -> None:
        """Remove the cache file."""
        if self._cache_path.exists():
            self._cache_path.unlink()
            logger.info("BM25 cache invalidated")
