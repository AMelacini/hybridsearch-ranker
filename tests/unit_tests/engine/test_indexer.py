from pathlib import Path

from app.engine.indexer import load_file, tokenize


def test_tokenize_normalizes_and_splits_words() -> None:
    assert tokenize("Hello, world! 123") == ["hello", "world", "123"]


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
