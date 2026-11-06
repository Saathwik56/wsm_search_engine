"""
Module 5: Query Processing
WSM Assignment - Web Search & Mining

This module parses and executes search queries against the Inverted Index and Wildcard Processor.
Supported query modalities:
1. Boolean Retrieval: AND, OR, NOT with operator precedence and parentheses (Shunting-yard algorithm)
2. Positional Phrase Retrieval: Exact word sequences (e.g., "inverted index")
3. Ranked Free-Text Retrieval: Vector space TF-IDF scoring
4. Wildcard Query Integration: Automatic expansion of wildcard tokens via Permuterm/K-gram
5. Contextual Snippet Generation: Context window extraction with highlighted matches
"""

from dataclasses import dataclass, field
import math
import re
from typing import Dict, List, Optional, Set, Tuple

from src.document_loader import Document
from src.inverted_index import InvertedIndex, Posting
from src.normalizer import TextNormalizer
from src.tokenizer import Tokenizer
from src.wildcard_processor import WildcardProcessor


@dataclass
class SearchResult:
    """Represents a matched document with ranking, snippet, and diagnostic metadata."""
    doc_id: int
    title: str
    path: str
    score: float
    matched_terms: List[str] = field(default_factory=list)
    snippet: str = ""
    expanded_terms: Dict[str, List[str]] = field(default_factory=dict)


class QueryProcessor:
    """
    Core search query engine supporting Boolean, Phrase, Wildcard, and Free-Text evaluation.
    """

    BOOLEAN_OPERATORS = {"AND", "OR", "NOT"}

    def __init__(
        self,
        index: InvertedIndex,
        wildcard_processor: Optional[WildcardProcessor] = None,
        tokenizer: Optional[Tokenizer] = None,
        normalizer: Optional[TextNormalizer] = None
    ):
        self.index = index
        self.wildcard_processor = wildcard_processor or WildcardProcessor()
        self.tokenizer = tokenizer or index.tokenizer
        self.normalizer = normalizer or index.normalizer

    # -------------------------------------------------------------------------
    # Snippet & Highlight Generation
    # -------------------------------------------------------------------------

    def generate_snippet(
        self,
        doc: Document,
        matched_terms: Set[str],
        window_chars: int = 180
    ) -> str:
        """
        Extracts a clean snippet centered around the earliest or densest match occurrences.
        """
        content = doc.content
        if not content:
            return ""

        lower_content = content.lower()
        earliest_pos = len(content)

        for term in matched_terms:
            pos = lower_content.find(term.lower())
            if pos != -1 and pos < earliest_pos:
                earliest_pos = pos

        if earliest_pos == len(content):
            # No direct substring match; return beginning
            snippet = " ".join(content[:window_chars].split())
            return snippet + "..." if len(content) > window_chars else snippet

        start = max(0, earliest_pos - 60)
        end = min(len(content), earliest_pos + window_chars)

        # Snap to whole word boundaries
        if start > 0:
            space_before = content.rfind(" ", 0, start)
            if space_before != -1:
                start = space_before + 1
        if end < len(content):
            space_after = content.find(" ", end)
            if space_after != -1:
                end = space_after

        snippet_text = content[start:end].strip()
        prefix = "... " if start > 0 else ""
        suffix = " ..." if end < len(content) else ""

        # Highlight matched terms with brackets for plain text display
        highlighted = snippet_text
        for term in matched_terms:
            pattern = re.compile(rf"\b({re.escape(term)})\b", re.IGNORECASE)
            highlighted = pattern.sub(r"[\1]", highlighted)

        return prefix + highlighted + suffix

    # -------------------------------------------------------------------------
    # Boolean Postings Operations
    # -------------------------------------------------------------------------

    def _intersect_postings(self, list1: List[int], list2: List[int]) -> List[int]:
        """Classical two-pointer linear-time intersection for sorted document ID lists."""
        p1 = 0
        p2 = 0
        result: List[int] = []
        while p1 < len(list1) and p2 < len(list2):
            if list1[p1] == list2[p2]:
                result.append(list1[p1])
                p1 += 1
                p2 += 1
            elif list1[p1] < list2[p2]:
                p1 += 1
            else:
                p2 += 1
        return result

    def _union_postings(self, list1: List[int], list2: List[int]) -> List[int]:
        """Classical two-pointer linear-time union for sorted document ID lists."""
        p1 = 0
        p2 = 0
        result: List[int] = []
        while p1 < len(list1) and p2 < len(list2):
            if list1[p1] == list2[p2]:
                result.append(list1[p1])
                p1 += 1
                p2 += 1
            elif list1[p1] < list2[p2]:
                result.append(list1[p1])
                p1 += 1
            else:
                result.append(list2[p2])
                p2 += 1
        while p1 < len(list1):
            result.append(list1[p1])
            p1 += 1
        while p2 < len(list2):
            result.append(list2[p2])
            p2 += 1
        return result

    def _difference_postings(self, universal: List[int], excluded: List[int]) -> List[int]:
        """Returns sorted elements in universal that are not in excluded."""
        excluded_set = set(excluded)
        return [doc_id for doc_id in universal if doc_id not in excluded_set]

    # -------------------------------------------------------------------------
    # Term Resolution (with Wildcard Expansion)
    # -------------------------------------------------------------------------

    def resolve_term(self, term: str) -> Tuple[List[int], List[str]]:
        """
        Resolves a single token, expanding wildcards if present.

        Returns:
            Tuple of (matched_doc_ids, list_of_expanded_terms)
        """
        if self.wildcard_processor.is_wildcard(term):
            # Wildcard term: expand using WildcardProcessor
            expanded = self.wildcard_processor.expand(term.lower())
            if not expanded:
                return [], []

            doc_ids_set: Set[int] = set()
            for exp_t in expanded:
                doc_ids_set.update(self.index.get_doc_ids(exp_t))
            return sorted(list(doc_ids_set)), expanded
        else:
            norm = self.normalizer.normalize_token(term)
            if not norm:
                return [], []
            return self.index.get_doc_ids(norm), [norm]

    # -------------------------------------------------------------------------
    # Phrase Query Evaluation (Positional Inverted Index)
    # -------------------------------------------------------------------------

    def search_phrase(self, phrase: str) -> List[SearchResult]:
        """
        Evaluates exact phrase queries (e.g., "inverted index" or "web search engines")
        using positional information stored in the inverted index.
        """
        raw_tokens = self.tokenizer.tokenize_terms(phrase)
        terms: List[str] = []
        for t in raw_tokens:
            norm = self.normalizer.normalize_token(t)
            if norm:
                terms.append(norm)

        if not terms:
            return []

        if len(terms) == 1:
            doc_ids, _ = self.resolve_term(terms[0])
            results: List[SearchResult] = []
            for doc_id in doc_ids:
                doc = self.index.get_document(doc_id)
                if doc:
                    snippet = self.generate_snippet(doc, set(terms))
                    results.append(
                        SearchResult(
                            doc_id=doc_id,
                            title=doc.title,
                            path=doc.path,
                            score=1.0,
                            matched_terms=terms,
                            snippet=snippet
                        )
                    )
            return results

        # Phrase search over 2 or more terms
        # Get postings for all terms
        term_postings = [self.index.get_postings(t) for t in terms]
        for p in term_postings:
            if not p:
                return []

        # Find documents containing all terms
        candidate_docs = set(p.doc_id for p in term_postings[0])
        for p_list in term_postings[1:]:
            candidate_docs &= set(p.doc_id for p in p_list)

        matched_doc_ids: List[int] = []

        for doc_id in sorted(list(candidate_docs)):
            # Extract position lists for each term in this doc
            positions_by_term: List[List[int]] = []
            for p_list in term_postings:
                p_item = next(p for p in p_list if p.doc_id == doc_id)
                positions_by_term.append(p_item.positions)

            # Check if there is an unbroken chain pos, pos+1, pos+2 ...
            found_phrase = False
            for start_pos in positions_by_term[0]:
                expected_next = start_pos + 1
                chain_valid = True
                for next_term_positions in positions_by_term[1:]:
                    if expected_next in next_term_positions:
                        expected_next += 1
                    else:
                        chain_valid = False
                        break
                if chain_valid:
                    found_phrase = True
                    break

            if found_phrase:
                matched_doc_ids.append(doc_id)

        results = []
        for doc_id in matched_doc_ids:
            doc = self.index.get_document(doc_id)
            if doc:
                snippet = self.generate_snippet(doc, set(terms))
                results.append(
                    SearchResult(
                        doc_id=doc_id,
                        title=doc.title,
                        path=doc.path,
                        score=float(len(terms)),
                        matched_terms=terms,
                        snippet=snippet
                    )
                )

        return results

    # -------------------------------------------------------------------------
    # Boolean Query Parsing & Evaluation (Shunting-yard Algorithm)
    # -------------------------------------------------------------------------

    def _tokenize_boolean_query(self, query_str: str) -> List[str]:
        """
        Splits boolean expression into tokens preserving operators, parens, and wildcards,
        and inserts implicit AND operators where appropriate (e.g. 'A NOT B' -> 'A AND NOT B',
        'A B' -> 'A AND B').
        """
        pattern = r"\(|\)|\bAND\b|\bOR\b|\bNOT\b|[^\s()]+"
        raw_tokens = re.findall(pattern, query_str)
        if not raw_tokens:
            return []

        def is_operand(t: str) -> bool:
            return t not in ("AND", "OR", "NOT", "(", ")")

        # Insert implicit AND
        normalized: List[str] = []
        for i, tok in enumerate(raw_tokens):
            if i > 0:
                prev = raw_tokens[i - 1]
                # Preceding condition: prev is operand or prev is ')'
                prev_can_end = is_operand(prev) or prev == ")"
                # Current condition: tok is operand, or tok is 'NOT', or tok is '('
                curr_can_start = is_operand(tok) or tok == "NOT" or tok == "("

                if prev_can_end and curr_can_start:
                    normalized.append("AND")

            normalized.append(tok)

        return normalized

    def search_boolean(self, query_str: str) -> Tuple[List[SearchResult], Dict[str, List[str]]]:
        """
        Parses and evaluates a boolean query using Shunting-yard algorithm
        to construct Reverse Polish Notation (RPN), followed by RPN stack evaluation.

        Operators supported: AND, OR, NOT, (, )
        Precedence: NOT (highest) > AND > OR (lowest)
        """
        all_doc_ids = sorted(list(self.index.documents.keys()))
        tokens = self._tokenize_boolean_query(query_str)

        if not tokens:
            return [], {}

        precedence = {"NOT": 3, "AND": 2, "OR": 1}
        output_queue: List[str] = []
        operator_stack: List[str] = []
        expansion_tracker: Dict[str, List[str]] = {}

        for tok in tokens:
            if tok in ("AND", "OR", "NOT"):
                while (
                    operator_stack
                    and operator_stack[-1] in precedence
                    and precedence[operator_stack[-1]] >= precedence[tok]
                ):
                    output_queue.append(operator_stack.pop())
                operator_stack.append(tok)
            elif tok == "(":
                operator_stack.append(tok)
            elif tok == ")":
                while operator_stack and operator_stack[-1] != "(":
                    output_queue.append(operator_stack.pop())
                if operator_stack and operator_stack[-1] == "(":
                    operator_stack.pop()
            else:
                output_queue.append(tok)

        while operator_stack:
            output_queue.append(operator_stack.pop())

        # Evaluate RPN stack
        eval_stack: List[List[int]] = []
        all_matched_terms: Set[str] = set()

        for item in output_queue:
            if item == "NOT":
                op = eval_stack.pop() if eval_stack else []
                eval_stack.append(self._difference_postings(all_doc_ids, op))
            elif item == "AND":
                op2 = eval_stack.pop() if eval_stack else []
                op1 = eval_stack.pop() if eval_stack else []
                eval_stack.append(self._intersect_postings(op1, op2))
            elif item == "OR":
                op2 = eval_stack.pop() if eval_stack else []
                op1 = eval_stack.pop() if eval_stack else []
                eval_stack.append(self._union_postings(op1, op2))
            else:
                doc_ids, expanded = self.resolve_term(item)
                if self.wildcard_processor.is_wildcard(item):
                    expansion_tracker[item] = expanded
                all_matched_terms.update(expanded)
                eval_stack.append(doc_ids)

        final_doc_ids = eval_stack[0] if eval_stack else []

        results: List[SearchResult] = []
        for doc_id in final_doc_ids:
            doc = self.index.get_document(doc_id)
            if doc:
                snippet = self.generate_snippet(doc, all_matched_terms)
                results.append(
                    SearchResult(
                        doc_id=doc_id,
                        title=doc.title,
                        path=doc.path,
                        score=1.0,
                        matched_terms=sorted(list(all_matched_terms)),
                        snippet=snippet,
                        expanded_terms=expansion_tracker
                    )
                )

        return results, expansion_tracker

    # -------------------------------------------------------------------------
    # Ranked Free-Text Search (TF-IDF Scoring)
    # -------------------------------------------------------------------------

    def search_free_text(self, query_str: str) -> Tuple[List[SearchResult], Dict[str, List[str]]]:
        """
        Ranked retrieval scoring documents using Term Frequency - Inverse Document Frequency (TF-IDF).
        """
        raw_tokens = self.tokenizer.tokenize_query_terms(query_str)
        if not raw_tokens:
            return [], {}

        total_docs = self.index.total_documents
        if total_docs == 0:
            return [], {}

        doc_scores: Dict[int, float] = {}
        matched_terms_by_doc: Dict[int, Set[str]] = {}
        expansion_tracker: Dict[str, List[str]] = {}

        for token in raw_tokens:
            doc_ids, expanded = self.resolve_term(token)
            if self.wildcard_processor.is_wildcard(token):
                expansion_tracker[token] = expanded

            for exp_t in expanded:
                postings = self.index.get_postings(exp_t)
                df = len(postings)
                if df == 0:
                    continue
                # Inverse Document Frequency (smoothed)
                idf = math.log(1.0 + (total_docs / df))

                for p in postings:
                    # Term Frequency (sublinear logarithmic scaling)
                    tf = 1.0 + math.log(p.term_frequency)
                    score = tf * idf

                    doc_scores[p.doc_id] = doc_scores.get(p.doc_id, 0.0) + score
                    if p.doc_id not in matched_terms_by_doc:
                        matched_terms_by_doc[p.doc_id] = set()
                    matched_terms_by_doc[p.doc_id].add(exp_t)

        # Normalize scores by document length to prevent long document bias
        ranked_doc_ids = sorted(
            doc_scores.keys(),
            key=lambda d: doc_scores[d] / max(1, math.sqrt(self.index.doc_lengths.get(d, 1))),
            reverse=True
        )

        results: List[SearchResult] = []
        for doc_id in ranked_doc_ids:
            doc = self.index.get_document(doc_id)
            if doc:
                norm_score = doc_scores[doc_id] / max(1, math.sqrt(self.index.doc_lengths.get(doc_id, 1)))
                snippet = self.generate_snippet(doc, matched_terms_by_doc.get(doc_id, set()))
                results.append(
                    SearchResult(
                        doc_id=doc_id,
                        title=doc.title,
                        path=doc.path,
                        score=round(norm_score, 4),
                        matched_terms=sorted(list(matched_terms_by_doc.get(doc_id, set()))),
                        snippet=snippet,
                        expanded_terms=expansion_tracker
                    )
                )

        return results, expansion_tracker

    # -------------------------------------------------------------------------
    # Unified Query Entrypoint
    # -------------------------------------------------------------------------

    def query(
        self,
        query_str: str,
        mode: str = "auto"
    ) -> Tuple[List[SearchResult], Dict[str, List[str]], str]:
        """
        Dispatches search query according to specified mode or auto-detection.

        Args:
            query_str: Raw input query.
            mode: 'auto', 'boolean', 'phrase', 'wildcard', 'free_text'.

        Returns:
            Tuple of (List[SearchResult], expanded_terms_dict, detected_mode_str)
        """
        stripped = query_str.strip()
        if not stripped:
            return [], {}, "empty"

        chosen_mode = mode.lower()

        # Phrase Query: explicitly chosen or enclosed in quotes
        if chosen_mode == "phrase" or (chosen_mode == "auto" and stripped.startswith('"') and stripped.endswith('"') and len(stripped) > 2):
            phrase_content = stripped.strip('"')
            results = self.search_phrase(phrase_content)
            return results, {}, "phrase"

        # Boolean Query: explicitly chosen or contains AND/OR/NOT
        has_boolean_ops = any(op in stripped.split() for op in ("AND", "OR", "NOT"))
        if chosen_mode == "boolean" or (chosen_mode == "auto" and has_boolean_ops):
            results, expansions = self.search_boolean(stripped)
            return results, expansions, "boolean"

        # Wildcard Query: explicitly chosen or contains '*' / '?'
        if chosen_mode == "wildcard" or (chosen_mode == "auto" and ("*" in stripped or "?" in stripped)):
            # If it's a single wildcard or contains multiple words with wildcards
            results, expansions = self.search_free_text(stripped)
            return results, expansions, "wildcard"

        # Default: Free text ranked search
        results, expansions = self.search_free_text(stripped)
        return results, expansions, "free_text"
