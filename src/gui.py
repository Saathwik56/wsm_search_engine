"""
Module 7: Frontend GUI
WSM Assignment - Web Search & Mining

This module provides a modern, responsive Tkinter graphical user interface (GUI)
featuring:
1. Search input box (with keyboard Return shortcut)
2. Search button
3. Read-only result area with syntax-highlighted snippets, document rankings, and diagnostics
4. Corpus loading and index rebuild tools
5. Query mode selection (Auto, Boolean, Phrase, Wildcard, Free-Text)
"""

import os
import sys
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText
from typing import Dict, List, Optional

from src.document_loader import Document, DocumentLoader
from src.inverted_index import InvertedIndex
from src.normalizer import TextNormalizer
from src.query_processor import QueryProcessor, SearchResult
from src.tokenizer import Tokenizer
from src.wildcard_processor import WildcardProcessor


class SearchEngineGUI:
    """
    Main Tkinter Desktop application for the WSM Search Engine.
    """

    def __init__(self, root: tk.Tk, default_corpus_path: Optional[str] = None):
        self.root = root
        self.root.title("WSM Search Engine - Web Search & Mining")
        self.root.geometry("1000x750")
        self.root.minsize(800, 600)

        # Initialize IR components
        self.loader = DocumentLoader()
        self.tokenizer = Tokenizer()
        self.stemming_var = tk.BooleanVar(value=False)
        self.stopwords_var = tk.BooleanVar(value=False)
        self.normalizer = self._create_normalizer()
        self.index = InvertedIndex(tokenizer=self.tokenizer, normalizer=self.normalizer)
        self.wildcard_processor = WildcardProcessor(method="permuterm")
        self.query_processor = QueryProcessor(
            index=self.index,
            wildcard_processor=self.wildcard_processor,
            tokenizer=self.tokenizer,
            normalizer=self.normalizer
        )

        # Path tracking
        self.corpus_path_var = tk.StringVar(value=default_corpus_path or "")
        self.query_mode_var = tk.StringVar(value="Auto (Smart Detection)")
        self.status_var = tk.StringVar(value="Initializing...")

        # Setup Tk styles
        self._setup_styles()

        # Build UI layout
        self._build_header()
        self._build_corpus_bar()
        self._build_search_bar()
        self._build_results_area()
        self._build_status_bar()

        # Auto-load default corpus if present
        if default_corpus_path and os.path.isdir(default_corpus_path):
            self.load_corpus(default_corpus_path)
        else:
            self.status_var.set("Ready. Please select a document corpus to index.")
            self._display_welcome_message()

    def _setup_styles(self) -> None:
        """Configures ttk styles for a clean, modern aesthetic."""
        style = ttk.Style()
        # Use available theme
        if "clam" in style.theme_names():
            style.theme_use("clam")

        # Fonts and colors
        style.configure(".", font=("Segoe UI", 10))
        style.configure("Header.TLabel", font=("Segoe UI", 16, "bold"), foreground="#1e293b")
        style.configure("SubHeader.TLabel", font=("Segoe UI", 9), foreground="#64748b")
        style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"), background="#2563eb", foreground="white")
        style.map("Accent.TButton", background=[("active", "#1d4ed8"), ("pressed", "#1e40af")])
        style.configure("Secondary.TButton", font=("Segoe UI", 9))
        style.configure("Card.TFrame", background="#f8fafc")

    def _create_normalizer(self) -> TextNormalizer:
        """Creates a normalizer instance based on current checkbox states."""
        return TextNormalizer(
            case_fold=True,
            strip_accents=True,
            remove_stopwords=self.stopwords_var.get(),
            use_stemming=self.stemming_var.get()
        )

    # -------------------------------------------------------------------------
    # UI Layout Construction
    # -------------------------------------------------------------------------

    def _build_header(self) -> None:
        """Constructs header brand and description."""
        header_frame = ttk.Frame(self.root, padding="16 12 16 8")
        header_frame.pack(fill=tk.X)

        title_lbl = ttk.Label(header_frame, text="WSM Information Retrieval System", style="Header.TLabel")
        title_lbl.pack(anchor=tk.W)

        desc_lbl = ttk.Label(
            header_frame,
            text="Modular Search Engine: Document Loading • Tokenization • Normalization • Inverted Index • Boolean • Wildcard",
            style="SubHeader.TLabel"
        )
        desc_lbl.pack(anchor=tk.W, pady=(2, 0))

    def _build_corpus_bar(self) -> None:
        """Constructs the corpus directory management and configuration toolbar."""
        bar_frame = ttk.LabelFrame(self.root, text=" 1. Corpus & Index Management ", padding="10 8")
        bar_frame.pack(fill=tk.X, padx=16, pady=4)

        # Corpus Path selector
        path_frame = ttk.Frame(bar_frame)
        path_frame.pack(fill=tk.X, pady=(0, 6))

        ttk.Label(path_frame, text="Corpus Directory:").pack(side=tk.LEFT, padx=(0, 6))
        self.path_entry = ttk.Entry(path_frame, textvariable=self.corpus_path_var)
        self.path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))

        browse_btn = ttk.Button(path_frame, text="Browse...", command=self._browse_directory, style="Secondary.TButton")
        browse_btn.pack(side=tk.LEFT, padx=(0, 6))

        index_btn = ttk.Button(path_frame, text="Rebuild Index", command=self._rebuild_index, style="Secondary.TButton")
        index_btn.pack(side=tk.LEFT)

        # Preprocessing settings
        opts_frame = ttk.Frame(bar_frame)
        opts_frame.pack(fill=tk.X)

        stem_cb = ttk.Checkbutton(
            opts_frame,
            text="Porter Stemming",
            variable=self.stemming_var,
            command=self._on_options_changed
        )
        stem_cb.pack(side=tk.LEFT, padx=(0, 16))

        stop_cb = ttk.Checkbutton(
            opts_frame,
            text="Remove Stopwords",
            variable=self.stopwords_var,
            command=self._on_options_changed
        )
        stop_cb.pack(side=tk.LEFT, padx=(0, 16))

        self.index_badge_lbl = ttk.Label(
            opts_frame,
            text="Index: No documents loaded",
            foreground="#64748b",
            font=("Segoe UI", 9, "italic")
        )
        self.index_badge_lbl.pack(side=tk.RIGHT)

    def _build_search_bar(self) -> None:
        """Constructs search input box, query mode selector, and Search button."""
        search_frame = ttk.LabelFrame(self.root, text=" 2. Search Query ", padding="10 8")
        search_frame.pack(fill=tk.X, padx=16, pady=6)

        # Top row: Search input box, Search button, Clear button
        input_row = ttk.Frame(search_frame)
        input_row.pack(fill=tk.X, pady=(0, 6))

        # Search Entry Box
        self.search_entry = ttk.Entry(input_row, font=("Segoe UI", 12))
        self.search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8), ipady=3)
        self.search_entry.bind("<Return>", lambda event: self.execute_search())
        self.search_entry.focus()

        # Search Button (Requested)
        self.search_button = ttk.Button(
            input_row,
            text="🔍 Search",
            command=self.execute_search,
            style="Accent.TButton",
            width=12
        )
        self.search_button.pack(side=tk.LEFT, padx=(0, 6))

        # Clear Button
        clear_btn = ttk.Button(input_row, text="Clear", command=self._clear_search, width=8)
        clear_btn.pack(side=tk.LEFT)

        # Mode row: Query paradigm selector
        mode_row = ttk.Frame(search_frame)
        mode_row.pack(fill=tk.X)

        ttk.Label(mode_row, text="Query Mode:").pack(side=tk.LEFT, padx=(0, 6))
        mode_combo = ttk.Combobox(
            mode_row,
            textvariable=self.query_mode_var,
            values=[
                "Auto (Smart Detection)",
                "Boolean Query (AND / OR / NOT)",
                "Phrase Query (\"exact words\")",
                "Wildcard Query (mon*, *ing, m*n)",
                "Ranked Free Text (TF-IDF)"
            ],
            state="readonly",
            width=32
        )
        mode_combo.pack(side=tk.LEFT, padx=(0, 16))

        # Example helper shortcuts
        ttk.Label(mode_row, text="Examples: ", foreground="#64748b").pack(side=tk.LEFT)
        ex_bool = ttk.Label(mode_row, text="web AND mining", foreground="#2563eb", cursor="hand2")
        ex_bool.pack(side=tk.LEFT, padx=4)
        ex_bool.bind("<Button-1>", lambda e: self._set_query("web AND mining"))

        ex_wild = ttk.Label(mode_row, text="sear*", foreground="#2563eb", cursor="hand2")
        ex_wild.pack(side=tk.LEFT, padx=4)
        ex_wild.bind("<Button-1>", lambda e: self._set_query("sear*"))

        ex_phrase = ttk.Label(mode_row, text='"inverted index"', foreground="#2563eb", cursor="hand2")
        ex_phrase.pack(side=tk.LEFT, padx=4)
        ex_phrase.bind("<Button-1>", lambda e: self._set_query('"inverted index"'))

    def _build_results_area(self) -> None:
        """Constructs the read-only result area (Text widget) with scrolling and text styling."""
        results_frame = ttk.LabelFrame(self.root, text=" 3. Search Results (Read-Only) ", padding="10 8")
        results_frame.pack(fill=tk.BOTH, expand=True, padx=16, pady=4)

        # ScrolledText widget configured as read-only (state='disabled')
        self.results_text = ScrolledText(
            results_frame,
            wrap=tk.WORD,
            font=("Consolas", 10),
            background="#ffffff",
            foreground="#1e293b",
            relief=tk.FLAT,
            padx=12,
            pady=10
        )
        self.results_text.pack(fill=tk.BOTH, expand=True)

        # Configure visual tags for rich read-only formatting
        self.results_text.tag_config("title", font=("Segoe UI", 12, "bold"), foreground="#1e3a8a")
        self.results_text.tag_config("meta", font=("Segoe UI", 9, "italic"), foreground="#64748b")
        self.results_text.tag_config("badge", font=("Segoe UI", 9, "bold"), foreground="#0369a1", background="#e0f2fe")
        self.results_text.tag_config("snippet", font=("Consolas", 10), foreground="#334155")
        self.results_text.tag_config("highlight", font=("Consolas", 10, "bold"), foreground="#0f172a", background="#fef08a")
        self.results_text.tag_config("expansion", font=("Segoe UI", 9, "bold"), foreground="#7c3aed", background="#ede9fe")
        self.results_text.tag_config("separator", foreground="#cbd5e1")
        self.results_text.tag_config("header_banner", font=("Segoe UI", 11, "bold"), foreground="#0f766e", background="#ccfbf1")
        self.results_text.tag_config("info", font=("Segoe UI", 10), foreground="#475569")
        self.results_text.tag_config("empty", font=("Segoe UI", 11, "italic"), foreground="#94a3b8")

        # Set to read-only initially
        self.results_text.config(state=tk.DISABLED)

    def _build_status_bar(self) -> None:
        """Bottom status bar."""
        status_bar = ttk.Frame(self.root, padding="16 4 16 6")
        status_bar.pack(fill=tk.X)

        self.status_lbl = ttk.Label(status_bar, textvariable=self.status_var, font=("Segoe UI", 9))
        self.status_lbl.pack(side=tk.LEFT)

    # -------------------------------------------------------------------------
    # Corpus & Index Management Logic
    # -------------------------------------------------------------------------

    def _browse_directory(self) -> None:
        """Opens file dialog for user to select corpus folder."""
        current = self.corpus_path_var.get()
        initial = current if os.path.isdir(current) else os.getcwd()
        selected = filedialog.askdirectory(initialdir=initial, title="Select Corpus Directory")
        if selected:
            self.corpus_path_var.set(selected)
            self.load_corpus(selected)

    def _on_options_changed(self) -> None:
        """Re-indexes documents when stemming or stopword options are toggled."""
        if self.loader.documents:
            self._rebuild_index()

    def _rebuild_index(self) -> None:
        """Re-indexes currently loaded documents with active normalizer settings."""
        path = self.corpus_path_var.get()
        if not path or not os.path.isdir(path):
            messagebox.showwarning("Warning", "Please provide a valid corpus directory path first.")
            return
        self.load_corpus(path)

    def load_corpus(self, directory_path: str) -> None:
        """Loads all documents from directory, builds inverted index and wildcard indexes."""
        try:
            self.loader.clear()
            documents = self.loader.load_directory(directory_path, extensions=[".txt", ".text", ".md", ".data"])

            if not documents:
                self.index_badge_lbl.config(text="Warning: 0 documents found in directory.")
                self.status_var.set(f"No text files found in {directory_path}")
                return

            # Re-initialize normalizer with current settings
            self.normalizer = self._create_normalizer()
            self.index = InvertedIndex(tokenizer=self.tokenizer, normalizer=self.normalizer)
            self.index.build(documents)

            # Build wildcard index
            self.wildcard_processor.build(self.index.vocabulary)

            # Re-link QueryProcessor
            self.query_processor = QueryProcessor(
                index=self.index,
                wildcard_processor=self.wildcard_processor,
                tokenizer=self.tokenizer,
                normalizer=self.normalizer
            )

            stats = self.index.get_stats()
            badge_text = (
                f"✓ Indexed {stats['total_documents']} docs | "
                f"Vocab: {stats['vocabulary_size']} terms | "
                f"Postings: {stats['total_postings_entries']}"
            )
            self.index_badge_lbl.config(text=badge_text, foreground="#15803d")
            self.status_var.set(f"Successfully loaded and indexed {stats['total_documents']} documents.")

            self._display_corpus_summary(stats)

        except Exception as e:
            messagebox.showerror("Indexing Error", f"Failed to index corpus:\n{str(e)}")
            self.status_var.set(f"Error loading corpus: {str(e)}")

    # -------------------------------------------------------------------------
    # Search Execution & Result Rendering
    # -------------------------------------------------------------------------

    def _set_query(self, query_text: str) -> None:
        """Sets the query entry text and triggers search."""
        self.search_entry.delete(0, tk.END)
        self.search_entry.insert(0, query_text)
        self.execute_search()

    def _clear_search(self) -> None:
        """Clears search input and result display."""
        self.search_entry.delete(0, tk.END)
        self._write_results_readonly("")
        self.status_var.set("Search cleared.")
        self.search_entry.focus()

    def execute_search(self) -> None:
        """
        Executes query against the inverted index, measures runtime,
        and renders formatted results to the read-only result area.
        """
        query_str = self.search_entry.get().strip()
        if not query_str:
            messagebox.showinfo("Input", "Please enter a search query.")
            return

        if self.index.total_documents == 0:
            messagebox.showwarning("Index Empty", "No documents have been indexed yet. Please load a corpus first.")
            return

        # Determine mode
        mode_selection = self.query_mode_var.get()
        mode_map = {
            "Auto (Smart Detection)": "auto",
            "Boolean Query (AND / OR / NOT)": "boolean",
            "Phrase Query (\"exact words\")": "phrase",
            "Wildcard Query (mon*, *ing, m*n)": "wildcard",
            "Ranked Free Text (TF-IDF)": "free_text"
        }
        chosen_mode = mode_map.get(mode_selection, "auto")

        t_start = time.perf_counter()
        try:
            results, expansions, detected_mode = self.query_processor.query(query_str, mode=chosen_mode)
            elapsed_ms = (time.perf_counter() - t_start) * 1000

            self._display_search_results(query_str, results, expansions, detected_mode, elapsed_ms)
            self.status_var.set(
                f"Completed {detected_mode} search in {elapsed_ms:.2f} ms. Found {len(results)} match(es)."
            )
        except Exception as ex:
            self._display_error(str(ex))
            self.status_var.set(f"Search failed: {str(ex)}")

    def _write_results_readonly(self, content_builder) -> None:
        """
        Safely writes to the read-only result area by toggling state=NORMAL,
        inserting tagged content, and restoring state=DISABLED.
        """
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete("1.0", tk.END)

        if callable(content_builder):
            content_builder(self.results_text)
        elif isinstance(content_builder, str):
            self.results_text.insert(tk.END, content_builder)

        self.results_text.config(state=tk.DISABLED)

    def _display_search_results(
        self,
        query: str,
        results: List[SearchResult],
        expansions: Dict[str, List[str]],
        mode: str,
        elapsed_ms: float
    ) -> None:
        """Renders rich, structured search results into the read-only text box."""

        def builder(txt: tk.Text):
            # Header summary banner
            banner_text = (
                f"  SEARCH REPORT  |  Query: \"{query}\"  |  Mode: {mode.upper()}  |  "
                f"Results: {len(results)}  |  Time: {elapsed_ms:.2f} ms\n"
            )
            txt.insert(tk.END, banner_text, "header_banner")
            txt.insert(tk.END, "\n")

            # Wildcard expansion notice if applicable
            if expansions:
                txt.insert(tk.END, "  ⚡ Wildcard Expansions:\n", "expansion")
                for wild_token, matched_words in expansions.items():
                    expansion_line = f"    • '{wild_token}' -> [{', '.join(matched_words) if matched_words else 'No matches in vocabulary'}]\n"
                    txt.insert(tk.END, expansion_line, "expansion")
                txt.insert(tk.END, "\n")

            # Check if no results were found
            if not results:
                txt.insert(tk.END, "  [!] No documents matched your query.\n\n", "empty")
                txt.insert(tk.END, "  Suggestions:\n", "info")
                txt.insert(tk.END, "  • Check spelling of terms\n", "info")
                txt.insert(tk.END, "  • Try wildcard variations (e.g. sear* or *ing)\n", "info")
                txt.insert(tk.END, "  • For boolean queries, verify operators are uppercase: AND, OR, NOT\n", "info")
                txt.insert(tk.END, "  • Try broader terms without quotation marks\n", "info")
                return

            # Render each matching document result
            separator_line = "─" * 78 + "\n\n"

            for rank, r in enumerate(results, start=1):
                # Document title
                txt.insert(tk.END, f"  #{rank}. ", "title")
                txt.insert(tk.END, f"{r.title}\n", "title")

                # Metadata badge
                doc = self.index.get_document(r.doc_id)
                char_len = doc.char_count if doc else 0
                word_len = doc.word_count if doc else 0
                meta_str = (
                    f"     Doc ID: {r.doc_id}  |  Score / TF: {r.score:.4f}  |  "
                    f"Words: {word_len}  |  File: {os.path.basename(r.path)}\n"
                )
                txt.insert(tk.END, meta_str, "meta")

                # Matched terms badge
                if r.matched_terms:
                    matched_str = f"     Matched Terms: {', '.join(r.matched_terms[:12])}\n"
                    txt.insert(tk.END, matched_str, "badge")

                # Snippet with highlights
                txt.insert(tk.END, "     Snippet: ", "meta")
                self._insert_highlighted_snippet(txt, r.snippet)
                txt.insert(tk.END, "\n\n")

                # Separator
                txt.insert(tk.END, "  " + separator_line, "separator")

        self._write_results_readonly(builder)

    def _insert_highlighted_snippet(self, txt: tk.Text, snippet: str) -> None:
        """Parses bracket markers [matched_word] in snippet and applies highlight tag."""
        import re
        tokens = re.split(r"(\[[^\]]+\])", snippet)
        for part in tokens:
            if part.startswith("[") and part.endswith("]"):
                txt.insert(tk.END, f" {part[1:-1]} ", "highlight")
            else:
                txt.insert(tk.END, part, "snippet")

    def _display_corpus_summary(self, stats: Dict) -> None:
        """Displays index summary in result area after loading."""
        def builder(txt: tk.Text):
            txt.insert(tk.END, "  INDEX INITIALIZED SUCCESSFULLY\n", "header_banner")
            txt.insert(tk.END, "\n")
            txt.insert(tk.END, f"  • Total Documents:          {stats['total_documents']}\n", "info")
            txt.insert(tk.END, f"  • Vocabulary Size:          {stats['vocabulary_size']} unique terms\n", "info")
            txt.insert(tk.END, f"  • Total Postings Records:   {stats['total_postings_entries']}\n", "info")
            txt.insert(tk.END, f"  • Average Document Length:  {stats['average_document_length']} tokens\n", "info")
            txt.insert(tk.END, f"  • Porter Stemming:          {'Enabled' if self.stemming_var.get() else 'Disabled'}\n", "info")
            txt.insert(tk.END, f"  • Stopwords Removal:        {'Enabled' if self.stopwords_var.get() else 'Disabled'}\n", "info")
            txt.insert(tk.END, "\n")
            txt.insert(tk.END, "  Ready to search! Type your query above or click one of the example queries.\n", "title")

        self._write_results_readonly(builder)

    def _display_welcome_message(self) -> None:
        """Initial welcome screen explaining how to use the search engine."""
        def builder(txt: tk.Text):
            txt.insert(tk.END, "  WELCOME TO WSM SEARCH ENGINE\n", "header_banner")
            txt.insert(tk.END, "\n")
            txt.insert(tk.END, "  This IR system provides full evaluation for:\n\n", "info")
            txt.insert(tk.END, "  1. Boolean Retrieval:   Supports AND, OR, NOT and nested parentheses\n", "info")
            txt.insert(tk.END, "                          Example: web AND mining NOT machine\n\n", "meta")
            txt.insert(tk.END, "  2. Wildcard Processing: Supported via Permuterm & K-gram indices\n", "info")
            txt.insert(tk.END, "                          Example: sear*, *ing, m*n, p*rmut*rm\n\n", "meta")
            txt.insert(tk.END, "  3. Phrase Retrieval:    Exact word sequence via Positional Inverted Index\n", "info")
            txt.insert(tk.END, "                          Example: \"inverted index\", \"web search\"\n\n", "meta")
            txt.insert(tk.END, "  4. Free-Text Search:    Ranked scoring via TF-IDF\n", "info")
            txt.insert(tk.END, "                          Example: information retrieval crawler\n\n", "meta")
            txt.insert(tk.END, "  To begin, select a corpus folder using 'Browse...' or click 'Rebuild Index'.\n", "badge")

        self._write_results_readonly(builder)

    def _display_error(self, message: str) -> None:
        """Displays error message inside the read-only result area."""
        def builder(txt: tk.Text):
            txt.insert(tk.END, "  SEARCH ENGINE ERROR\n", "header_banner")
            txt.insert(tk.END, f"\n  An error occurred while evaluating your query:\n\n  {message}\n", "empty")
        self._write_results_readonly(builder)


def launch_gui(corpus_dir: Optional[str] = None) -> None:
    """Entrypoint function to run the Tkinter desktop GUI."""
    root = tk.Tk()
    app = SearchEngineGUI(root, default_corpus_path=corpus_dir)
    root.mainloop()


if __name__ == "__main__":
    launch_gui()

