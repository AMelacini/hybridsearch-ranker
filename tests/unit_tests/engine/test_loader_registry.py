from pathlib import Path

from mock import AsyncMock, Mock, patch

from app.engine.loader_registry import LoaderRegistry
from app.models import FileType, FileTypeFilter, FoundDocuments, SearchModel


def test_use_filetype_keys() -> None:
    registry = LoaderRegistry()
    loader = Mock()

    registry.set_loader(FileType.MD, loader)

    assert registry.get_loader(FileType.MD) is loader
    assert registry.get_loader(FileType.PDF) is None


def test_build_from_enum_members() -> None:
    registry = LoaderRegistry.from_file_types({FileType.MD: Mock(), FileType.PDF: Mock()})

    assert registry.get_loader(FileType.MD) is not None
    assert registry.get_loader(FileType.PDF) is not None
    assert registry.get_loader(FileType.TXT) is None


def test_resolve_extension_aliases() -> None:
    loader = Mock()
    registry = LoaderRegistry.from_file_types({FileType.MD: loader})

    assert registry.get_loader_for_extension("markdown") is loader
    assert registry.get_loader_for_extension("md") is loader
