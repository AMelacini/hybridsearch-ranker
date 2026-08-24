from enum import Enum
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, Field, field_validator, model_validator


class TextualResponse(BaseModel):
    """Plain text message with the option to implement valdation/sanitization"""

    message: str

    @model_validator(mode="after")
    def validate_response(self) -> Self:
        # not implemented
        # Sanity checks, guards, etc... would go here
        return self


class FoundDocuments(BaseModel):
    doc_paths: list[Path] = []


class FileContent(BaseModel):
    textual_content: str


# Supported file type and ETL (Extract-Transform-Load)
class FileType(Enum):
    """Defines supported file types and their associated extensions."""

    CSV = ("csv",)
    HTML = ("htm", "html")
    MD = ("md", "markdown")
    PDF = ("pdf",)
    TXT = ("txt", "text")

    @property
    def extensions(self) -> tuple[str, ...]:
        """Return the supported extensions associated with this file type."""

        # Tell mypy that self.value is always a tuple of strings
        # (By default, the standard library dynamic type for Enum.value is defined as Any.)
        result: tuple[str, ...] = self.value

        return result

    @classmethod
    def from_extension(cls, ext: str) -> "FileType":
        """Finds the FileType matching a given extension string."""

        # Normalize input (lowercase, ensure leading dot)
        cleaned_ext = f"{ext.strip().lower().lstrip('.')}"

        for file_type in cls:
            if cleaned_ext in file_type.value:
                return file_type

        raise ValueError(f"Unsupported extension '{ext}'. Supported: {[e for ft in cls for e in ft.value]}")

    @classmethod
    def is_supported(cls, ext: str) -> bool:
        """Check if 'ext' is a supported extension."""

        # Normalize input (lowercase, ensure leading dot)
        cleaned_ext = f"{ext.strip().lower().lstrip('.')}"

        for file_type in cls:
            if cleaned_ext in file_type.value:
                return True

        return False

    @classmethod
    def supported_extensions(cls) -> list[str]:
        """Return the set of all supported extentions (across types) with no prepended dot '.'."""
        extensions = {e for ft in cls for e in ft.value}
        return sorted(extensions)


class FileTypeFilter(BaseModel):
    """Validates incoming file process requests."""

    # Using the FileType enum ensures type safety
    file_type: FileType

    @field_validator("file_type", mode="before")
    @classmethod
    def validate_extension(cls, ext: str | FileType) -> FileType:
        """Coerces a string input into a valid FileType enum."""
        if isinstance(ext, FileType):
            return ext
        if isinstance(ext, str):
            return FileType.from_extension(ext)
        raise ValueError("Invalid input type. Must be a string or FileType.")


# === Search Models ===

SearchMode = Literal["hybrid", "semantic", "keyword"]


class SearchModel(BaseModel):
    """Collection of parameters guiding a search"""

    query: str  # Textual query
    top_k: int = Field(gt=0, default=5)  # Maximim number of results
    # mode: SearchMode  # "hybrid", "semantic", "keyword"
    vector_search_weight: float = Field(
        ge=0.0, le=1.0, default=0.5
    )  # 0: keyword, 1: semantic, (0, 1): hybrid -> (vector_search_weight, 1-vector_search_weight)

    # Reciprocal Rank Fusion  parameters (applicable for hybrid search)
    rrf_k: int = Field(ge=0, default=60)
    rrf_fold: int = Field(
        gt=0, default=3
    )  # In hybrid search a larger set of results are retrieved from both keyord and semantic search


class SearchResults(BaseModel):
    """Collection of parameters guiding a search"""

    summary: str  # Textual query
    hits: list[str]
