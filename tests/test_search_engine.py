"""
Comprehensive Unit Tests for WSM Search Engine
Tests all 7 required assignment modules.
"""

import os
import shutil
import tempfile
import unittest

from src.document_loader import Document, DocumentLoader
from src.inverted_index import InvertedIndex, Posting
from src.normalizer import PorterStemmer, TextNormalizer
from src.query_processor import QueryProcessor
from src.tokenizer import Tokenizer
from src.wildcard_processor import KGramIndex, PermutermIndex, WildcardProcessor


class TestWSMSearchEngine(unittest.TestCase):

    def setUp(self):
        # Create a temporary directory with test documents
        self.test_dir = tempfile.mkdtemp()
        self.doc1_path = os.path.join(self.test_dir, "test1.txt")
        self.doc2_path = os.path.join(self.test_dir, "test2.txt")
        self.doc3_path = os.path.join(self.test_dir, "test3.txt")

        with open(self.doc1_path, "w", encoding="utf-8") as f:
            f.write("Web search engines crawl web pages and construct an inverted index.")

        with open(self.doc2_path, "w", encoding="utf-8") as f:
            f.write("Information retrieval systems use tokenization and text normalization.")

        with open(self.doc3_path, "w", encoding="utf-8") as f:
            f.write("Permuterm index supports wildcard queries like mon* and *mon.")

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    # -------------------------------------------------------------------------
    # 1. Document Loader Tests
    # -------------------------------------------------------------------------
    def test_document_loader(self):
        loader = DocumentLoader()
        docs = loader.load_directory(self.test_dir)
        self.assertEqual(len(docs), 3)

        doc = loader.get_document(1)
        self.assertIsNotNone(doc)
        self.assertTrue(len(doc.content) > 0)
        self.assertTrue(doc.word_count > 0)
        self.assertIn("inverted index", doc.content)

        # In-memory document test
        mem_doc = loader.add_text_document("Memory Doc", "Pure in-memory content")
        self.assertEqual(mem_doc.doc_id, 4)
        self.assertEqual(len(loader), 4)

    # -------------------------------------------------------------------------
    # 2. Tokenizer Tests
    # -------------------------------------------------------------------------
    def test_tokenizer(self):
        tokenizer = Tokenizer()
        text = "Information Retrieval: 2026 systems aren't simple!"
        tokens = tokenizer.tokenize(text)

        extracted_words = [t.text for t in tokens]
        self.assertIn("Information", extracted_words)
        self.assertIn("Retrieval", extracted_words)
        self.assertIn("2026", extracted_words)
        self.assertIn("aren't", extracted_words)

        # Verify sequential positions
        positions = [t.position for t in tokens]
        self.assertEqual(positions, list(range(len(tokens))))

        # Verify character offsets
        for t in tokens:
            self.assertEqual(text[t.start_char:t.end_char], t.text)

    # -------------------------------------------------------------------------
    # 3. Normalizer & Case Folding Tests
    # -------------------------------------------------------------------------
    def test_normalization_and_case_folding(self):
        normalizer = TextNormalizer(case_fold=True, strip_accents=True, remove_stopwords=False)
        self.assertEqual(normalizer.normalize_token("WebSearch"), "websearch")
        self.assertEqual(normalizer.normalize_token("Café"), "cafe")
        self.assertEqual(normalizer.normalize_token("...mining!"), "mining")

        # Stopword filtering
        normalizer_stop = TextNormalizer(case_fold=True, remove_stopwords=True)
        self.assertIsNone(normalizer_stop.normalize_token("the"))
        self.assertEqual(normalizer_stop.normalize_token("search"), "search")

        # Porter Stemmer verification
        stemmer = PorterStemmer()
        self.assertEqual(stemmer.stem("retrieval"), "retriev")
        self.assertEqual(stemmer.stem("connecting"), "connect")
        self.assertEqual(stemmer.stem("searches"), "search")

    # -------------------------------------------------------------------------
    # 4. Inverted Index Construction Tests
    # -------------------------------------------------------------------------
    def test_inverted_index_construction(self):
        loader = DocumentLoader()
        docs = loader.load_directory(self.test_dir)

        index = InvertedIndex()
        index.build(docs)

        # "inverted" should only be in doc1
        doc_ids_inverted = index.get_doc_ids("inverted")
        self.assertEqual(doc_ids_inverted, [1])

        # "index" is in doc1 ("inverted index") and doc3 ("Permuterm index")
        doc_ids_index = index.get_doc_ids("index")
        self.assertEqual(sorted(doc_ids_index), [1, 3])

        # Verify positional information
        postings = index.get_postings("index")
        for p in postings:
            self.assertTrue(len(p.positions) > 0)
            self.assertEqual(p.term_frequency, len(p.positions))

        # Test index serialization to disk
        save_path = os.path.join(self.test_dir, "index.json")
        index.save_to_file(save_path)
        self.assertTrue(os.path.isfile(save_path))

        new_index = InvertedIndex()
        new_index.load_from_file(save_path)
        self.assertEqual(new_index.get_doc_ids("index"), [1, 3])

    # -------------------------------------------------------------------------
    # 5. Wildcard Processing Tests (Permuterm & K-gram)
    # -------------------------------------------------------------------------
    def test_wildcard_permuterm_and_kgram(self):
        vocab = {"search", "searching", "searches", "retrieve", "retrieval", "mineral", "mining"}

        # Permuterm tests
        perm = PermutermIndex()
        perm.build(vocab)

        # Trailing wildcard X*
        matches = perm.query("sear*")
        self.assertEqual(sorted(matches), ["search", "searches", "searching"])

        # Leading wildcard *X
        matches_ing = perm.query("*ing")
        self.assertEqual(sorted(matches_ing), ["mining", "searching"])

        # Middle wildcard X*Y
        matches_mid = perm.query("m*n")
        # Should not match 'mining' (ends with g) or 'mineral' (ends with l)
        # Add 'moon' or 'man' to test
        perm.build(vocab | {"man", "moon", "main"})
        matches_m_n = perm.query("m*n")
        self.assertEqual(sorted(matches_m_n), ["main", "man", "moon"])

        # K-gram tests
        kgram = KGramIndex(k=3)
        kgram.build(vocab | {"man", "moon", "main"})
        kg_matches = kgram.query("sear*")
        self.assertEqual(sorted(kg_matches), ["search", "searches", "searching"])

        kg_matches_ing = kgram.query("*ing")
        self.assertEqual(sorted(kg_matches_ing), ["mining", "searching"])

    # -------------------------------------------------------------------------
    # 6. Query Processing Tests (Boolean, Phrase, Ranked)
    # -------------------------------------------------------------------------
    def test_query_processing(self):
        loader = DocumentLoader()
        docs = loader.load_directory(self.test_dir)

        tokenizer = Tokenizer()
        normalizer = TextNormalizer(case_fold=True)
        index = InvertedIndex(tokenizer=tokenizer, normalizer=normalizer)
        index.build(docs)

        wildcard_proc = WildcardProcessor()
        wildcard_proc.build(index.vocabulary)

        qp = QueryProcessor(index=index, wildcard_processor=wildcard_proc)

        # 1. Boolean AND
        results_and, _ = qp.search_boolean("web AND crawl")
        doc_ids = [r.doc_id for r in results_and]
        self.assertEqual(doc_ids, [1])

        # 2. Boolean OR
        results_or, _ = qp.search_boolean("crawl OR tokenization")
        doc_ids = sorted([r.doc_id for r in results_or])
        self.assertEqual(doc_ids, [1, 2])

        # 3. Boolean NOT
        results_not, _ = qp.search_boolean("index NOT permuterm")
        doc_ids = [r.doc_id for r in results_not]
        self.assertEqual(doc_ids, [1])

        # 4. Phrase query
        phrase_results = qp.search_phrase("inverted index")
        self.assertEqual(len(phrase_results), 1)
        self.assertEqual(phrase_results[0].doc_id, 1)

        # Non-matching phrase (terms exist in different order or disconnected)
        no_phrase = qp.search_phrase("index inverted")
        self.assertEqual(len(no_phrase), 0)

        # 5. Wildcard query via unified processor
        results_wild, expansions, mode = qp.query("sear*")
        self.assertEqual(mode, "wildcard")
        self.assertTrue(len(results_wild) > 0)
        self.assertIn("sear*", expansions)

    # -------------------------------------------------------------------------
    # 7. Frontend GUI Component Tests
    # -------------------------------------------------------------------------
    def test_frontend_gui_components(self):
        import tkinter as tk
        from src.gui import SearchEngineGUI

        root = tk.Tk()
        # Hide window during test
        root.withdraw()
        try:
            app = SearchEngineGUI(root, default_corpus_path=self.test_dir)
            root.update()

            # Verify required widgets
            self.assertIsNotNone(app.search_entry, "Search input box must exist")
            self.assertIsNotNone(app.search_button, "Search button must exist")
            self.assertIsNotNone(app.results_text, "Read-only result area must exist")

            # Verify read-only state
            self.assertEqual(app.results_text.cget("state"), "disabled", "Result area must be read-only")

            # Perform a search through GUI
            app.search_entry.insert(0, "web")
            app.execute_search()
            root.update()

            # Result area should still be read-only
            self.assertEqual(app.results_text.cget("state"), "disabled")

            # Read content from read-only area
            content = app.results_text.get("1.0", tk.END)
            self.assertIn("SEARCH REPORT", content)
            self.assertIn("test1.txt", content)
        finally:
            root.destroy()


if __name__ == "__main__":
    unittest.main()
