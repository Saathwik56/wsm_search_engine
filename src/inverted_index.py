"""
Module 4: Inverted Index Construction
WSM Assignment - Web Search & Mining

This module builds, serializes, and queries a positional inverted index.
Each vocabulary term maps to a list of Postings containing document IDs,
term frequencies, and exact token positions.
"""

from dataclasses import dataclass, field
import json
import os
from typing import Any, Dict, List, Optional, Set

from src.document_loader import Document
from src.normalizer import TextNormalizer
from src.tokenizer import Tokenizer


@dataclass
class Posting:
    """Represents a term occurrence record in a specific document."""
    doc_id: int
    term_frequency: int
    positions: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "term_frequency": self.term_frequency,
            "positions": self.positions
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Posting":
        return cls(
            doc_id=data["doc_id"],
            term_frequency=data["term_frequency"],
            positions=data.get("positions", [])
        )


class InvertedIndex:
    """
    Positional Inverted Index mapping normalized terms to sorted postings lists.
    """

    def __init__(
        self,
        tokenizer: Optional[Tokenizer] = None,
        normalizer: Optional[TextNormalizer] = None
    ):
        self.tokenizer = tokenizer or Tokenizer()
        self.normalizer = normalizer or TextNormalizer()
        # term -> list of Posting objects sorted by doc_id
        self.index: Dict[str, List[Posting]] = {}
        # doc_id -> Document
        self.documents: Dict[int, Document] = {}
        # doc_id -> total token count
        self.doc_lengths: Dict[int, int] = {}

    @property
    def vocabulary(self) -> Set[str]:
        """Returns the set of all unique terms present in the index."""
        return set(self.index.keys())

    @property
    def total_documents(self) -> int:
        return len(self.documents)

    def build(self, documents: List[Document]) -> None:
        """
        Builds the inverted index from a collection of Document objects.

        Args:
            documents: List of loaded Document objects.
        """
        self.clear()

        for doc in documents:
            self.documents[doc.doc_id] = doc
            raw_tokens = self.tokenizer.tokenize(doc.content)

            # Map normalized_term -> list of token positions in this doc
            doc_term_positions: Dict[str, List[int]] = {}
            valid_tokens_count = 0

            for tok in raw_tokens:
                norm_term = self.normalizer.normalize_token(tok.text)
                if norm_term is None:
                    continue

                valid_tokens_count += 1
                if norm_term not in doc_term_positions:
                    doc_term_positions[norm_term] = []
                doc_term_positions[norm_term].append(tok.position)

            self.doc_lengths[doc.doc_id] = valid_tokens_count

            # Append posting to each term's postings list
            for term, positions in doc_term_positions.items():
                if term not in self.index:
                    self.index[term] = []
                self.index[term].append(
                    Posting(
                        doc_id=doc.doc_id,
                        term_frequency=len(positions),
                        positions=positions
                    )
                )

        # Sort postings lists by doc_id (guaranteed ascending)
        for term in self.index:
            self.index[term].sort(key=lambda p: p.doc_id)

    def get_postings(self, term: str) -> List[Posting]:
        """
        Retrieves the postings list for a normalized term.

        Args:
            term: The term to look up.

        Returns:
            List of Posting objects, or empty list if term is not in vocabulary.
        """
        norm_term = self.normalizer.normalize_token(term)
        if norm_term is None:
            return []
        return self.index.get(norm_term, [])

    def get_doc_ids(self, term: str) -> List[int]:
        """Returns the sorted list of document IDs containing the given term."""
        return [p.doc_id for p in self.get_postings(term)]

    def get_document_frequency(self, term: str) -> int:
        """Returns the number of documents containing the term (DF)."""
        return len(self.get_postings(term))

    def get_document(self, doc_id: int) -> Optional[Document]:
        """Returns Document instance by its doc_id."""
        return self.documents.get(doc_id)

    def get_stats(self) -> Dict[str, Any]:
        """Returns a dictionary summarizing index statistics."""
        vocab_size = len(self.index)
        total_postings = sum(len(p_list) for p_list in self.index.values())
        avg_doc_len = (
            sum(self.doc_lengths.values()) / len(self.doc_lengths)
            if self.doc_lengths else 0.0
        )
        return {
            "total_documents": self.total_documents,
            "vocabulary_size": vocab_size,
            "total_postings_entries": total_postings,
            "average_document_length": round(avg_doc_len, 2)
        }

    def save_to_file(self, filepath: str) -> None:
        """Serializes the index to a JSON file on disk."""
        data = {
            "documents": {
                str(doc_id): {
                    "doc_id": doc.doc_id,
                    "title": doc.title,
                    "path": doc.path,
                    "content": doc.content,
                    "metadata": doc.metadata
                }
                for doc_id, doc in self.documents.items()
            },
            "doc_lengths": {str(k): v for k, v in self.doc_lengths.items()},
            "index": {
                term: [p.to_dict() for p in postings]
                for term, postings in self.index.items()
            }
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load_from_file(self, filepath: str) -> None:
        """Loads an inverted index from a JSON file."""
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"Index file not found: {filepath}")

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.clear()
        for doc_id_str, d_info in data.get("documents", {}).items():
            doc_id = int(doc_id_str)
            doc = Document(
                doc_id=doc_id,
                title=d_info["title"],
                path=d_info["path"],
                content=d_info["content"],
                metadata=d_info.get("metadata", {})
            )
            self.documents[doc_id] = doc

        self.doc_lengths = {int(k): v for k, v in data.get("doc_lengths", {}).items()}

        for term, p_list in data.get("index", {}).items():
            self.index[term] = [Posting.from_dict(p) for p in p_list]

    def clear(self) -> None:
        """Resets the index state."""
        self.index.clear()
        self.documents.clear()
        self.doc_lengths.clear()

