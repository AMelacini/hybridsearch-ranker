import pytest

from app.models import FileType, FileTypeFilter, FoundDocuments, SearchModel


def test_search_model_validates_mode_weight_consistency() -> None:
    with pytest.raises(ValueError):
        SearchModel(query="hello", top_k=3, vector_search_weight=1.5)

    with pytest.raises(ValueError):
        SearchModel(query="hello", top_k=3, vector_search_weight=-0.2)

    assert FileTypeFilter(file_type=FileType.MD).file_type is FileType.MD
    assert FileType.is_supported(".PDF") is True
    assert FileType.is_supported(".csv") is True
    assert FileType.is_supported(".unknown") is False
