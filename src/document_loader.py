"""
Module 1: Document Loading
WSM Assignment - Web Search & Mining

This module is responsible for loading text documents from files, directories,
or in-memory strings into structured Document objects with metadata.
"""

from dataclasses import dataclass, field
import os
from typing import Dict, List, Optional, Sequence


@dataclass
class Document:
    """Represents an indexed document with identifiers and metadata."""
    doc_id: int
    title: str
    path: str
    content: str
    metadata: Dict[str, str] = field(default_factory=dict)

    @property
    def char_count(self) -> int:
        return len(self.content)

    @property
    def word_count(self) -> int:
        return len(self.content.split())

    def get_preview(self, max_length: int = 150) -> str:
        """Returns a shortened preview of the document content."""
        clean = " ".join(self.content.split())
        if len(clean) <= max_length:
            return clean
        return clean[:max_length].rstrip() + "..."


class DocumentLoader:
    """
    Handles discovery, reading, and storage of documents from directories and files.
    """

    def __init__(self):
        self.documents: Dict[int, Document] = {}
        self._next_id: int = 1

    def clear(self) -> None:
        """Clears all currently loaded documents and resets ID generator."""
        self.documents.clear()
        self._next_id = 1

    def _read_file_content(self, file_path: str) -> str:
        """Attempts to read file content using UTF-8, falling back to latin-1."""
        for encoding in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
            try:
                with open(file_path, "r", encoding=encoding) as f:
                    return f.read()
            except (UnicodeDecodeError, UnicodeError):
                continue
        # Fallback with replacement of un-decodable bytes
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()

    def load_file(self, file_path: str, title: Optional[str] = None) -> Document:
        """
        Loads a single document from disk.

        Args:
            file_path: Absolute or relative path to the file.
            title: Optional custom title (defaults to filename).

        Returns:
            The loaded Document instance.
        """
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Document file not found: {file_path}")

        content = self._read_file_content(file_path)
        doc_title = title or os.path.basename(file_path)
        doc = Document(
            doc_id=self._next_id,
            title=doc_title,
            path=os.path.abspath(file_path),
            content=content,
            metadata={"filename": os.path.basename(file_path)}
        )
        self.documents[self._next_id] = doc
        self._next_id += 1
        return doc

    def load_directory(
        self,
        dir_path: str,
        extensions: Sequence[str] = (".txt", ".text", ".md", ".data"),
        recursive: bool = False
    ) -> List[Document]:
        """
        Loads all matching text files from a directory.

        Args:
            dir_path: Path to the directory containing documents.
            extensions: Allowed file extensions.
            recursive: Whether to search subdirectories recursively.

        Returns:
            List of loaded Document objects.
        """
        if not os.path.isdir(dir_path):
            raise NotADirectoryError(f"Directory not found: {dir_path}")

        loaded: List[Document] = []
        ext_set = {ext.lower() for ext in extensions}

        if recursive:
            walker = os.walk(dir_path)
        else:
            entries = [os.path.join(dir_path, e) for e in os.listdir(dir_path)]
            walker = [(dir_path, [], [os.path.basename(e) for e in entries if os.path.isfile(e)])]

        for root, _, files in walker:
            # Sort files alphabetically for deterministic document ID assignment
            for fname in sorted(files):
                _, ext = os.path.splitext(fname)
                if ext.lower() in ext_set:
                    full_path = os.path.join(root, fname)
                    doc = self.load_file(full_path)
                    loaded.append(doc)

        return loaded

    def add_text_document(self, title: str, content: str) -> Document:
        """
        Loads an in-memory string as a Document without writing to disk.

        Args:
            title: Title of document.
            content: Raw text content.

        Returns:
            The created Document object.
        """
        doc = Document(
            doc_id=self._next_id,
            title=title,
            path="<in-memory>",
            content=content,
            metadata={"source": "in-memory"}
        )
        self.documents[self._next_id] = doc
        self._next_id += 1
        return doc

    def get_document(self, doc_id: int) -> Optional[Document]:
        """Retrieves a document by its integer ID."""
        return self.documents.get(doc_id)

    def get_all_documents(self) -> List[Document]:
        """Returns all currently loaded documents sorted by ID."""
        return [self.documents[doc_id] for doc_id in sorted(self.documents.keys())]

    def __len__(self) -> int:
        return len(self.documents)

