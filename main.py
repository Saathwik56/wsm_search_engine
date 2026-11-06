"""
WSM Assignment: Web Search and Mining Information Retrieval System
Main Entry Point

Usage:
    python main.py                  # Launches the Desktop GUI (Tkinter)
    python main.py --cli            # Launches Interactive Terminal Mode
    python main.py --query "web AND mining"  # Runs a single search from CLI
"""

import argparse
import os
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.document_loader import DocumentLoader
from src.inverted_index import InvertedIndex
from src.normalizer import TextNormalizer
from src.query_processor import QueryProcessor
from src.tokenizer import Tokenizer
from src.wildcard_processor import WildcardProcessor


DEFAULT_CORPUS_DIR = os.path.join(PROJECT_ROOT, "data", "sample_corpus")


def run_cli_interactive(corpus_dir: str):
    """Runs interactive command line session for testing and evaluation."""
    print("=" * 70)
    print("  WSM Information Retrieval System - Interactive CLI Mode")
    print("=" * 70)
    print(f"Loading corpus from: {corpus_dir}")

    loader = DocumentLoader()
    docs = loader.load_directory(corpus_dir)
    print(f"Loaded {len(docs)} documents.")

    tokenizer = Tokenizer()
    normalizer = TextNormalizer(case_fold=True, strip_accents=True)
    index = InvertedIndex(tokenizer=tokenizer, normalizer=normalizer)
    index.build(docs)

    wildcard_proc = WildcardProcessor()
    wildcard_proc.build(index.vocabulary)

    query_proc = QueryProcessor(
        index=index,
        wildcard_processor=wildcard_proc,
        tokenizer=tokenizer,
        normalizer=normalizer
    )

    stats = index.get_stats()
    print(f"Index built: {stats['vocabulary_size']} terms in vocabulary.")
    print("-" * 70)
    print("Query Types Supported:")
    print("  • Boolean:   web AND mining NOT structure")
    print("  • Wildcard:  sear*, *ing, m*n")
    print("  • Phrase:    \"inverted index\", \"web search\"")
    print("  • Free-text: information retrieval crawler")
    print("Type 'exit' or 'quit' to stop.")
    print("-" * 70)

    while True:
        try:
            query = input("\nEnter Search Query > ").strip()
            if not query:
                continue
            if query.lower() in ("exit", "quit", "q"):
                print("Exiting search engine.")
                break

            results, expansions, mode = query_proc.query(query)
            print(f"\n[Result] Mode: {mode.upper()} | Matches: {len(results)}")
            if expansions:
                for w, terms in expansions.items():
                    print(f"  • Wildcard '{w}' expanded to: {terms}")

            if not results:
                print("  No documents matched.")
                continue

            for rank, r in enumerate(results, 1):
                print(f"\n  #{rank} Doc {r.doc_id}: {r.title} (Score: {r.score:.4f})")
                print(f"     Matched Terms: {', '.join(r.matched_terms)}")
                print(f"     Snippet: {r.snippet}")
        except KeyboardInterrupt:
            print("\nExiting search engine.")
            break
        except Exception as e:
            print(f"Error processing query: {e}")


def run_single_query(corpus_dir: str, query: str):
    """Executes a single query and prints formatted results to stdout."""
    loader = DocumentLoader()
    docs = loader.load_directory(corpus_dir)

    tokenizer = Tokenizer()
    normalizer = TextNormalizer(case_fold=True, strip_accents=True)
    index = InvertedIndex(tokenizer=tokenizer, normalizer=normalizer)
    index.build(docs)

    wildcard_proc = WildcardProcessor()
    wildcard_proc.build(index.vocabulary)

    query_proc = QueryProcessor(
        index=index,
        wildcard_processor=wildcard_proc,
        tokenizer=tokenizer,
        normalizer=normalizer
    )

    results, expansions, mode = query_proc.query(query)
    print(f"\nQuery: '{query}' [Mode: {mode.upper()}] - {len(results)} hit(s)")
    if expansions:
        print(f"Wildcard expansions: {expansions}")
    for rank, r in enumerate(results, 1):
        print(f"  [{rank}] Doc {r.doc_id}: {r.title} (Score: {r.score:.4f})")
        print(f"      Snippet: {r.snippet}")


def main():
    parser = argparse.ArgumentParser(description="WSM Information Retrieval Search Engine")
    parser.add_argument(
        "--corpus",
        type=str,
        default=DEFAULT_CORPUS_DIR,
        help="Path to documents directory (default: sample corpus)"
    )
    parser.add_argument(
        "--cli",
        action="store_true",
        help="Run in interactive CLI mode instead of GUI"
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Execute a single query and print results"
    )

    args = parser.parse_args()

    corpus_path = os.path.abspath(args.corpus)

    if args.query:
        run_single_query(corpus_path, args.query)
    elif args.cli:
        run_cli_interactive(corpus_path)
    else:
        # Launch Frontend GUI (Default)
        from src.gui import launch_gui
        launch_gui(corpus_dir=corpus_path)


if __name__ == "__main__":
    main()

