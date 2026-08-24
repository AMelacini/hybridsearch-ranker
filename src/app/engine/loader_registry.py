from pathlib import Path
from typing import Callable, Mapping

from langchain_core.documents import Document

from app.models import FileType

LoaderFunction = Callable[[Path], list[Document]]


class LoaderRegistry:
    """Registry mapping supported file types to their document loaders."""

    def __init__(self, loaders: dict[FileType, LoaderFunction] | None = None) -> None:
        self._loaders: dict[FileType, LoaderFunction] = {}
        if loaders is not None:
            self._loaders.update(loaders)

    @classmethod
    def from_file_types(cls, loaders: Mapping[FileType, LoaderFunction]) -> "LoaderRegistry":
        """Create a registry by registering loaders for each enum member that is present."""
        registry = cls()
        for file_type in FileType:
            loader = loaders.get(file_type)
            if loader is not None:
                registry.set_loader(file_type, loader)
        return registry

    def get_loader(self, file_type: FileType) -> LoaderFunction | None:
        """Return the loader for a supported FileType, if registered."""
        return self._loaders.get(file_type)

    def get_loader_for_extension(self, extension: str) -> LoaderFunction | None:
        """Resolve a loader for a known extension or extension alias."""
        file_type = FileType.from_extension(extension)
        return self.get_loader(file_type)

    def get_loader_for_path(self, path: Path) -> LoaderFunction | None:
        """Resolve a loader directly from a file path using its FileType."""
        return self.get_loader_for_extension(path.suffix)

    def set_loader(self, file_type: FileType, loader: LoaderFunction) -> None:
        """Register or replace a loader for a specific FileType."""
        self._loaders[file_type] = loader

    def remove_loader(self, file_type: FileType) -> None:
        """Remove a loader registration for a specific FileType."""
        self._loaders.pop(file_type, None)

    def get_all_loaders(self) -> dict[FileType, LoaderFunction]:
        """Return a copy of the current loader mapping."""
        return dict(self._loaders)
