"""
Module 2: Tokenization
WSM Assignment - Web Search & Mining

This module handles splitting unstructured text into atomic tokens, capturing both
word-level token offsets (positions) and character-level boundaries for snippet extraction.
"""

from dataclasses import dataclass
import re
from typing import List, Pattern


@dataclass
class Token:
    """Represents a token extracted from a text document."""
    text: str
    position: int       # Word index in document (0, 1, 2, ...)
    start_char: int     # Character offset in raw text
    end_char: int       # End character offset in raw text


class Tokenizer:
    """
    Splits text streams into lexical tokens while preserving positional metadata.
    """

    # Matches alphanumeric words, numbers, and contractions like don't or user's
    DEFAULT_PATTERN = r"[a-zA-Z0-9]+(?:'[a-zA-Z0-9]+)?"
    # Query pattern retains wildcard symbols * and ?
    QUERY_PATTERN = r"[a-zA-Z0-9*?]+(?:'[a-zA-Z0-9]+)?"

    def __init__(self, pattern: str = DEFAULT_PATTERN):
        self.pattern_str = pattern
        self.pattern: Pattern = re.compile(pattern)
        self.query_pattern: Pattern = re.compile(self.QUERY_PATTERN)

    def tokenize(self, text: str) -> List[Token]:
        """
        Tokenizes text into a list of Token objects with word positions and char offsets.

        Args:
            text: Input string to tokenize.

        Returns:
            List of Token instances.
        """
        if not text:
            return []

        tokens: List[Token] = []
        for position, match in enumerate(self.pattern.finditer(text)):
            tokens.append(
                Token(
                    text=match.group(),
                    position=position,
                    start_char=match.start(),
                    end_char=match.end()
                )
            )
        return tokens

    def tokenize_terms(self, text: str) -> List[str]:
        """
        Extracts plain string tokens without positional wrappers.

        Args:
            text: Input string.

        Returns:
            List of raw string tokens.
        """
        return [tok.text for tok in self.tokenize(text)]

    def tokenize_query_terms(self, text: str) -> List[str]:
        """
        Tokenizes search query terms while preserving wildcards (* and ?).

        Args:
            text: Input query string.

        Returns:
            List of tokens with wildcard characters preserved.
        """
        if not text:
            return []
        return [match.group() for match in self.query_pattern.finditer(text)]
