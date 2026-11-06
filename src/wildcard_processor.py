"""
Module 6: Wildcard Processing
WSM Assignment - Web Search & Mining

This module implements classical Information Retrieval wildcard indexing techniques:
1. Permuterm Index: Rotational index appending '$' for prefix/suffix/middle wildcards (X*, *X, X*Y).
2. K-gram (Trigram) Index: Auxiliary character n-gram inverted index with post-filtering.
3. WildcardProcessor: Unified query expander translating wildcard expressions into vocabulary matches.
"""

from collections import defaultdict
import fnmatch
import re
from typing import Dict, Iterable, List, Optional, Set


class PermutermIndex:
    """
    Implements a Permuterm Index (Manning, Raghavan, Schütze - IIR Chapter 3).
    Appends a special boundary character '$' to each word and stores all cyclic shifts.
    """

    BOUNDARY = "$"

    def __init__(self):
        # Maps cyclic rotation string -> original vocabulary term
        self.rotations: Dict[str, str] = {}
        # Sorted list of keys for efficient binary search prefix lookup
        self._sorted_keys: List[str] = []

    def build(self, vocabulary: Iterable[str]) -> None:
        """
        Builds the permuterm index from an iterable of unique vocabulary terms.

        Args:
            vocabulary: Collection of distinct normalized terms.
        """
        self.rotations.clear()
        for term in vocabulary:
            if not term:
                continue
            padded = term + self.BOUNDARY
            for i in range(len(padded)):
                rotation = padded[i:] + padded[:i]
                self.rotations[rotation] = term

        self._sorted_keys = sorted(self.rotations.keys())

    def _prefix_lookup(self, prefix: str) -> Set[str]:
        """Finds all original terms whose rotation begins with the given prefix."""
        matched_terms: Set[str] = set()
        # Binary search range for prefix
        import bisect
        start_idx = bisect.bisect_left(self._sorted_keys, prefix)
        for i in range(start_idx, len(self._sorted_keys)):
            key = self._sorted_keys[i]
            if key.startswith(prefix):
                matched_terms.add(self.rotations[key])
            else:
                break
        return matched_terms

    def query(self, pattern: str) -> List[str]:
        """
        Evaluates a wildcard pattern using the permuterm rotations.

        Supported forms:
        - X*  (trailing): rotate to $X* -> prefix $X
        - *X  (leading):  rotate to X$* -> prefix X$
        - X*Y (middle):   rotate to Y$X* -> prefix Y$X
        - *X* (general):  rotate to X* -> find rotations containing prefix or fallback

        Args:
            pattern: Wildcard query pattern (e.g. 'mon*', '*mon', 'm*n').

        Returns:
            Sorted list of matching vocabulary terms.
        """
        pattern = pattern.strip()
        if "*" not in pattern:
            # Exact match check
            return [pattern] if (pattern + self.BOUNDARY) in self.rotations else []

        star_count = pattern.count("*")

        if star_count == 1:
            parts = pattern.split("*")
            left, right = parts[0], parts[1]

            if left and not right:
                # Case 1: X* (e.g., 'sea*') -> Lookup '$X'
                prefix = f"{self.BOUNDARY}{left}"
                candidates = self._prefix_lookup(prefix)
            elif not left and right:
                # Case 2: *X (e.g., '*ing') -> Lookup 'X$'
                prefix = f"{right}{self.BOUNDARY}"
                candidates = self._prefix_lookup(prefix)
            elif left and right:
                # Case 3: X*Y (e.g., 's*rch') -> Lookup 'Y$X'
                prefix = f"{right}{self.BOUNDARY}{left}"
                candidates = self._prefix_lookup(prefix)
            else:
                # Case 4: '*' matches all terms
                return sorted(list(set(self.rotations.values())))

            return sorted(list(candidates))

        # For patterns with multiple asterisks (e.g. *search* or a*b*c):
        # We can find prefix of first rotation and then regex post-filter
        regex_pattern = "^" + pattern.replace("*", ".*") + "$"
        compiled = re.compile(regex_pattern)
        matches = {term for term in self.rotations.values() if compiled.match(term)}
        return sorted(list(matches))


class KGramIndex:
    """
    Implements a K-gram (default 3-gram/trigram) index for wildcard processing.
    """

    BOUNDARY = "$"

    def __init__(self, k: int = 3):
        self.k = k
        # k-gram -> set of vocabulary terms
        self.index: Dict[str, Set[str]] = defaultdict(set)
        self.vocabulary: Set[str] = set()

    def build(self, vocabulary: Iterable[str]) -> None:
        """Constructs k-gram index over vocabulary."""
        self.index.clear()
        self.vocabulary = set(vocabulary)

        for term in self.vocabulary:
            padded = f"{self.BOUNDARY}{term}{self.BOUNDARY}"
            if len(padded) < self.k:
                self.index[padded].add(term)
            else:
                for i in range(len(padded) - self.k + 1):
                    kgram = padded[i:i + self.k]
                    self.index[kgram].add(term)

    def query(self, pattern: str) -> List[str]:
        """
        Decomposes query into k-grams, intersects candidate sets, and post-filters.
        """
        pattern = pattern.strip()
        if not pattern:
            return []

        padded_query = f"{self.BOUNDARY}{pattern}{self.BOUNDARY}"
        parts = padded_query.split("*")

        kgrams: List[str] = []
        for part in parts:
            if len(part) >= self.k:
                for i in range(len(part) - self.k + 1):
                    kgrams.append(part[i:i + self.k])

        if not kgrams:
            # Fall back to scan with regex when pattern is shorter than k
            regex_str = "^" + pattern.replace("*", ".*").replace("?", ".") + "$"
            compiled = re.compile(regex_str)
            return sorted([w for w in self.vocabulary if compiled.match(w)])

        # Intersect postings of all extracted k-grams
        candidate_terms = None
        for kg in kgrams:
            matches = self.index.get(kg, set())
            if candidate_terms is None:
                candidate_terms = set(matches)
            else:
                candidate_terms &= matches
            if not candidate_terms:
                break

        if not candidate_terms:
            return []

        # Post-filter step using regex to eliminate false positives
        regex_str = "^" + pattern.replace("*", ".*").replace("?", ".") + "$"
        compiled = re.compile(regex_str)
        return sorted([w for w in candidate_terms if compiled.match(w)])


class WildcardProcessor:
    """
    Unified Wildcard Processor combining Permuterm Index (primary) and
    K-gram Index (fallback/comparison) for expanding wildcard tokens.
    """

    def __init__(self, method: str = "permuterm"):
        self.method = method.lower()
        self.permuterm = PermutermIndex()
        self.kgram = KGramIndex(k=3)
        self.vocabulary: Set[str] = set()

    def build(self, vocabulary: Iterable[str]) -> None:
        """Builds both auxiliary wildcard indices for the vocabulary."""
        self.vocabulary = set(vocabulary)
        self.permuterm.build(self.vocabulary)
        self.kgram.build(self.vocabulary)

    def is_wildcard(self, token: str) -> bool:
        """Checks if a token contains wildcard characters."""
        return "*" in token or "?" in token

    def expand(self, wildcard_token: str, method: Optional[str] = None) -> List[str]:
        """
        Expands a wildcard pattern into a list of matching vocabulary terms.

        Args:
            wildcard_token: Pattern string (e.g. 'sear*', '*ing', 'm*n').
            method: 'permuterm' or 'kgram' (defaults to configured method).

        Returns:
            List of matched vocabulary terms.
        """
        chosen = (method or self.method).lower()
        if chosen == "kgram":
            return self.kgram.query(wildcard_token)
        else:
            return self.permuterm.query(wildcard_token)

