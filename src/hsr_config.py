"""
Centralized configuration for the Hybrid Search Ranker tool.

All settings can be overridden via environment variables
(specified in .env and sourced at startup).

The project accepts both the legacy names used by the app and the
standardized HSR_ names used by the deployment config.
"""

import os
from pathlib import Path
from tempfile import TemporaryDirectory


def _get_env_value(*names: str, default: str = "") -> str:
    for name in names:
        value = os.environ.get(name)
        if value is not None:
            return value
    return default


def _get_env_int(*names: str, default: int) -> int:
    value = _get_env_value(*names, default=str(default))
    return int(value)


# ======================================
# === REST Server settings (FastAPI) ===
# ======================================

REST_SERVER_PORT = _get_env_int("HSR_REST_SERVER_PORT", "REST_SERVER_PORT", default=8000)
SEARCH_REQUEST_TIMEOUT = _get_env_int("HSR_SEARCH_REQUEST_TIMEOUT", "SEARCH_REQUEST_TIMEOUT", default=180)


# ==================================
# === HSR UI settings (Strealit) ===
# ==================================

UI_PORT = _get_env_int("HSR_UI_PORT", "UI_PORT", default=8501)

# ==============================
# === Search Engine settings ===
# ==============================

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).parent
DEFAULT_DOCS_DIR = os.environ.get("HSR_DOCS_DIR", str(BASE_DIR / "docs"))
CACHE_DIR = os.environ.get("HSR_CACHE_DIR", str(BASE_DIR / "cache"))
DEFAULT_CHROMA_DIR = os.environ.get("HSR_CHROMA_DIR", str(BASE_DIR / "chroma_db"))

# ---------------------------------------------------------------------------
# ChromaDB
# ---------------------------------------------------------------------------

CHROMA_COLLECTION_NAME = os.environ.get("HSR_CHROMA_COLLECTION", "docs_rag")

# ---------------------------------------------------------------------------
# Redis
# ---------------------------------------------------------------------------

REDIS_URL = _get_env_value("HSR_REDIS_URL", "REDIS_URL", default="redis://localhost:6379")
REDIS_ENABLED = _get_env_value("HSR_REDIS_ENABLED", "REDIS_ENABLED", default="true").lower() in (
    "true",
    "1",
    "yes",
)

# TTL for cached query results (seconds). Default: 1 hour.
QUERY_CACHE_TTL = _get_env_int("HSR_QUERY_CACHE_TTL", "QUERY_CACHE_TTL", default=3600)

# ---------------------------------------------------------------------------
# Embedding model
# ---------------------------------------------------------------------------

EMBEDDING_MODEL = os.environ.get("HSR_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_MODEL_SIZE = int(os.environ.get("HSR_EMBEDDING_MODEL_SIZE", "384"))

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

CHUNK_SIZE = int(os.environ.get("HSR_CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.environ.get("HSR_CHUNK_OVERLAP", "200"))

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

# Weight for vector results in hybrid search (keyword weight = 1 - this).
VECTOR_WEIGHT = float(os.environ.get("HSR_VECTOR_WEIGHT", "0.5"))
