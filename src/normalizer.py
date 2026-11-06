"""
Module 3: Text Normalization and Case Folding
WSM Assignment - Web Search & Mining

This module standardizes lexical tokens across documents and queries.
Techniques supported:
1. Case folding (lowercasing)
2. Unicode accent / diacritic stripping
3. Punctuation removal
4. Stopword filtering (optional)
5. Porter Stemming algorithm (optional, self-contained without external dependencies)
"""

import re
import unicodedata
from typing import List, Optional, Set


# Standard English Stopwords for Information Retrieval
DEFAULT_STOPWORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
    "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's",
    "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
    "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
    "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
    "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves"
}


class PorterStemmer:
    """
    Lightweight, self-contained implementation of the classical Porter Stemming algorithm.
    Conforms to Martin Porter's 1980 specification.
    """

    def __init__(self):
        self.b = ""
        self.k = 0
        self.k0 = 0
        self.j = 0

    def _cons(self, i: int) -> bool:
        if self.b[i] in ("a", "e", "i", "o", "u"):
            return False
        if self.b[i] == "y":
            if i == self.k0:
                return True
            else:
                return not self._cons(i - 1)
        return True

    def _m(self) -> int:
        n = 0
        i = self.k0
        while True:
            if i > self.j:
                return n
            if not self._cons(i):
                break
            i += 1
        i += 1
        while True:
            while True:
                if i > self.j:
                    return n
                if self._cons(i):
                    break
                i += 1
            i += 1
            n += 1
            while True:
                if i > self.j:
                    return n
                if not self._cons(i):
                    break
                i += 1
            i += 1

    def _vowelinstem(self) -> bool:
        for i in range(self.k0, self.j + 1):
            if not self._cons(i):
                return True
        return False

    def _doublec(self, i: int) -> bool:
        if i < self.k0 + 1:
            return False
        if self.b[i] != self.b[i - 1]:
            return False
        return self._cons(i)

    def _cvc(self, i: int) -> bool:
        if i < self.k0 + 2 or not self._cons(i) or self._cons(i - 1) or not self._cons(i - 2):
            return False
        ch = self.b[i]
        if ch in ("w", "x", "y"):
            return False
        return True

    def _ends(self, s: str) -> bool:
        length = len(s)
        o = self.k - length + 1
        if o < self.k0:
            return False
        if self.b[o:self.k + 1] != s:
            return False
        self.j = self.k - length
        return True

    def _setto(self, s: str) -> None:
        length = len(s)
        o = self.j + 1
        self.b = self.b[:o] + s + self.b[o + length:]
        self.k = self.j + length

    def _r(self, s: str) -> None:
        if self._m() > 0:
            self._setto(s)

    def _step1ab(self) -> None:
        if self.b[self.k] == "s":
            if self._ends("sses"):
                self.k -= 2
            elif self._ends("ies"):
                self._setto("i")
            elif self.b[self.k - 1] != "s":
                self.k -= 1
        if self._ends("eed"):
            if self._m() > 0:
                self.k -= 1
        elif (self._ends("ed") or self._ends("ing")) and self._vowelinstem():
            self.k = self.j
            if self._ends("at"):
                self._setto("ate")
            elif self._ends("bl"):
                self._setto("ble")
            elif self._ends("iz"):
                self._setto("ize")
            elif self._doublec(self.k):
                self.k -= 1
                ch = self.b[self.k]
                if ch in ("l", "s", "z"):
                    self.k += 1
            elif self._m() == 1 and self._cvc(self.k):
                self._setto("e")

    def _step1c(self) -> None:
        if self._ends("y") and self._vowelinstem():
            self.b = self.b[:self.k] + "i" + self.b[self.k + 1:]

    def _step2(self) -> None:
        rel = {
            "ational": "ate", "tional": "tion", "enci": "ence", "anci": "ance",
            "izer": "ize", "bli": "ble", "alli": "al", "entli": "ent", "eli": "e",
            "ousli": "ous", "ization": "ize", "ation": "ate", "ator": "ate",
            "alism": "al", "iveness": "ive", "fulness": "ful", "ousness": "ous",
            "aliti": "al", "iviti": "ive", "biliti": "ble"
        }
        for k, v in rel.items():
            if self._ends(k):
                if self._m() > 0:
                    self._setto(v)
                break

    def _step3(self) -> None:
        rel = {
            "icate": "ic", "ative": "", "alize": "al", "iciti": "ic",
            "ical": "ic", "ful": "", "ness": ""
        }
        for k, v in rel.items():
            if self._ends(k):
                if self._m() > 0:
                    self._setto(v)
                break

    def _step4(self) -> None:
        suffixes = (
            "al", "ance", "ence", "er", "ic", "able", "ible", "ant", "ement",
            "ment", "ent", "ion", "ou", "ism", "ate", "iti", "ous", "ive", "ize"
        )
        for s in suffixes:
            if self._ends(s):
                if s == "ion":
                    if self.j >= self.k0 and self.b[self.j] in ("s", "t"):
                        if self._m() > 1:
                            self.k = self.j
                else:
                    if self._m() > 1:
                        self.k = self.j
                break

    def _step5(self) -> None:
        self.j = self.k
        if self.b[self.k] == "e":
            a = self._m()
            if a > 1 or (a == 1 and not self._cvc(self.k - 1)):
                self.k -= 1
        if self.b[self.k] == "l" and self._doublec(self.k) and self._m() > 1:
            self.k -= 1

    def stem(self, word: str) -> str:
        """Stems a word string using the Porter Stemmer algorithm."""
        word = word.lower()
        if len(word) <= 2:
            return word
        self.b = word
        self.k = len(word) - 1
        self.k0 = 0
        self._step1ab()
        self._step1c()
        self._step2()
        self._step3()
        self._step4()
        self._step5()
        return self.b[:self.k + 1]


class TextNormalizer:
    """
    Modular text normalizer supporting case folding, diacritic stripping,
    stopword filtering, and stemming.
    """

    def __init__(
        self,
        case_fold: bool = True,
        strip_accents: bool = True,
        remove_stopwords: bool = False,
        use_stemming: bool = False,
        custom_stopwords: Optional[Set[str]] = None
    ):
        self.case_fold = case_fold
        self.strip_accents = strip_accents
        self.remove_stopwords = remove_stopwords
        self.use_stemming = use_stemming
        self.stopwords = custom_stopwords if custom_stopwords is not None else DEFAULT_STOPWORDS
        self.stemmer = PorterStemmer()

    def normalize_token(self, token: str) -> Optional[str]:
        """
        Normalizes a single token. Returns None if the token should be discarded.

        Args:
            token: Input word.

        Returns:
            Normalized token string, or None if filtered out.
        """
        if not token:
            return None

        # 1. Accent / diacritic stripping
        if self.strip_accents:
            token = unicodedata.normalize("NFKD", token)
            token = "".join(c for c in token if not unicodedata.combining(c))

        # 2. Case folding (lowercasing)
        if self.case_fold:
            token = token.lower()

        # 3. Strip surrounding punctuation
        token = token.strip(" \t\n\r`~!@#$%^&*()-_=+[{]}\\|;:'\",<.>/?")
        if not token:
            return None

        # 4. Stopword filtering
        if self.remove_stopwords and token in self.stopwords:
            return None

        # 5. Stemming (optional)
        if self.use_stemming:
            token = self.stemmer.stem(token)

        return token

    def normalize_tokens(self, tokens: List[str]) -> List[str]:
        """Normalizes a list of string tokens, dropping any filtered ones."""
        result: List[str] = []
        for t in tokens:
            norm = self.normalize_token(t)
            if norm is not None:
                result.append(norm)
        return result

