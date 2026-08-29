from pathlib import Path

from app.engine.cache import BM25Cache, DocumentHashCache, QueryCache


def test_query_cache_round_trips_memory_values() -> None:
    cache = QueryCache(ttl=1)

    assert cache.get("hello", 3, 0.5) is None

    cache.set("hello", 3, 0.5, [{"content": "cached-result", "metadata": {"source": "doc.md"}}])

    assert cache.get("hello", 3, 0.5)[0]["content"] == "cached-result"
    cache.clear()
    assert cache.get("hello", 3, 0.5) is None


def test_bm25_cache_persists_and_reloads_state(tmp_path: Path) -> None:
    cache = BM25Cache(cache_dir=str(tmp_path))
    fingerprint = BM25Cache.compute_fingerprint_from_entries({"doc.md": ("abc123", 42.0)})

    cache.save(["alpha beta", "gamma"], ["id-1", "id-2"], [{"source": "doc.md"}, {"source": "doc.md"}], fingerprint)

    loaded = cache.load(fingerprint)
    assert loaded is not None
    assert loaded["texts"][0] == "alpha beta"
    assert cache.load("different-fingerprint") is None


def test_document_hash_cache_tracks_paths() -> None:
    hash_cache = DocumentHashCache()
    hash_cache.clear()

    hash_cache.put("docs/test.md", "hash-123", 10.5)

    assert hash_cache.get("docs/test.md") == "hash-123"
    assert "docs/test.md" in hash_cache.all_paths()
    hash_cache.remove("docs/test.md")
    assert hash_cache.get("docs/test.md") is None
