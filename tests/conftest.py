from __future__ import annotations

from pathlib import Path
from typing import Any

import dotenv
import pytest
from mock import Mock

env_file = Path(__file__).resolve().parents[1] / ".env"

if env_file.exists():
    dotenv.load_dotenv(env_file, override=False)
else:
    print(f"Warning: {env_file} not found; using existing env/defaults")


@pytest.fixture
def mock_indexer() -> Mock:
    indexer = Mock()
    indexer.list_files.return_value = ["docs/a.md", "docs/b.txt"]
    indexer.get_stats.return_value = {
        "docs_dir": "docs",
        "total_files": 2,
        "total_chunks": 4,
        "file_types": {".md": 2, ".txt": 2},
        "last_build": 123.0,
        "embedding_model": "sentence-transformers/test-model",
        "chunk_size": 1000,
        "chunk_overlap": 200,
    }
    indexer.search_bm25.return_value = [{"content": "bm25-hit", "metadata": {"source": "docs/a.md"}}]
    indexer.search_vector.return_value = [{"content": "vector-hit", "metadata": {"source": "docs/b.txt"}}]
    return indexer


@pytest.fixture
def sample_search_hits() -> list[dict[str, Any]]:
    return [
        {
            "content": "This is the relevant chunk",
            "metadata": {"source": "notes.md", "section": "Overview", "page": 2, "file_type": "md"},
        },
        {
            "content": "Another relevant chunk",
            "metadata": {"source": "guide.txt", "section": "Appendix", "page": 5, "file_type": "txt"},
        },
    ]
