# Web Search and Mining (WSM) Information Retrieval System

A modular Information Retrieval (IR) and search engine project implemented in pure Python for the Web Search and Mining (WSM) assignment.

---

## 📁 Project Structure

```
wsm_search_engine/
│
├── data/
│   └── sample_corpus/           # Sample text corpus covering IR & WSM topics
│       ├── doc1.txt             # Web search engines & crawling
│       ├── doc2.txt             # Inverted index & tokenization
│       ├── doc3.txt             # Query processing & boolean retrieval
│       ├── doc4.txt             # Wildcard queries & permuterm index
│       ├── doc5.txt             # Web mining & machine learning in search
│       └── doc6.txt             # NLP & text normalization
│
├── src/
│   ├── __init__.py              # Package initializer
│   ├── document_loader.py       # Module 1: Document Loading & Metadata
│   ├── tokenizer.py             # Module 2: Tokenization & Positional Offsets
│   ├── normalizer.py            # Module 3: Case Folding, Accents, Stopwords, Stemming
│   ├── inverted_index.py        # Module 4: Positional Inverted Index Construction
│   ├── query_processor.py       # Module 5: Boolean, Phrase, & Ranked Query Processing
│   ├── wildcard_processor.py    # Module 6: Permuterm Index & K-gram Index
│   └── gui.py                   # Module 7: Frontend Desktop GUI (Tkinter)
│
├── tests/
│   ├── __init__.py
│   └── test_search_engine.py    # Comprehensive test suite covering all 7 modules
│
├── main.py                      # Application launcher (Desktop GUI / CLI)
├── requirements.txt             # Standard library only (zero external dependencies)
└── README.md                    # Project documentation & reference
```

---

## 🧩 Architectural Modules

### 1. Document Loading (`src/document_loader.py`)
- Reads documents from disk (`.txt`, `.md`, `.data`) or in-memory strings.
- Graceful multi-encoding fallback (`utf-8`, `utf-8-sig`, `latin-1`, `cp1252`).
- Encapsulates documents into a `Document` dataclass preserving `doc_id`, `title`, `path`, raw `content`, word count, and character count.

### 2. Tokenization (`src/tokenizer.py`)
- Splits text into atomic `Token` objects.
- Preserves **word positions** ($0, 1, 2, \dots$) required for positional phrase search.
- Preserves **character start/end offsets** in raw text for snippet highlighting.
- Supports query-aware tokenization to preserve wildcard characters (`*`, `?`).

### 3. Text Normalization / Case Folding (`src/normalizer.py`)
- **Case Folding**: Converts all uppercase characters to lowercase.
- **Diacritic/Accent Stripping**: Standardizes Unicode characters via NFKD decomposition (e.g. `café` $\rightarrow$ `cafe`).
- **Punctuation Stripping**: Cleans peripheral symbols while preserving alphanumeric data.
- **Stopword Filtering**: Configurable removal of standard English stopwords.
- **Porter Stemmer**: Built-in, self-contained implementation of Martin Porter's suffix-stripping stemming algorithm (no external NLTK dependency required).

### 4. Inverted Index Construction (`src/inverted_index.py`)
- Constructs a **Positional Inverted Index** mapping normalized vocabulary terms to sorted `Posting` records.
- Each `Posting` records:
  - `doc_id`: Target document identifier
  - `term_frequency`: Frequency of term in document
  - `positions`: Exact ordered word indices $[p_1, p_2, \dots]$
- Includes index statistics reporting and JSON serialization (`save_to_file` / `load_from_file`).

### 5. Query Processing (`src/query_processor.py`)
- **Boolean Retrieval**: Evaluates logical `AND`, `OR`, `NOT` queries with nested parentheses using the **Shunting-yard algorithm** to generate and evaluate Reverse Polish Notation (RPN). Employs two-pointer linear merge intersection ($O(M+N)$).
- **Phrase Retrieval**: Evaluates exact sequence phrases (`"inverted index"`) by verifying adjacent positional offsets (`pos2 == pos1 + 1`).
- **Ranked Free-Text Retrieval**: Uses logarithmic Term Frequency - Inverse Document Frequency (**TF-IDF**) scoring with document length normalization:
  $$\text{Score}(D, Q) = \frac{\sum_{t \in Q} (1 + \ln \text{TF}_{t, D}) \cdot \ln(1 + \frac{N}{\text{DF}_t})}{\sqrt{\text{Length}_D}}$$
- **Contextual Snippets**: Automatically extracts matching sentence windows and highlights matching terms.

### 6. Wildcard Processing (`src/wildcard_processor.py`)
Implements the classical Information Retrieval algorithms (Manning, Raghavan, Schütze - *Introduction to Information Retrieval*, Ch. 3):
- **Permuterm Index**:
  - Appends boundary symbol `$` to each vocabulary term (e.g., `apple$`).
  - Stores all cyclic rotations: `apple$`, `pple$a`, `ple$ap`, `le$app`, `e$appl`, `$apple`.
  - Translates queries:
    - Trailing $X*$ (e.g., `sear*`) $\rightarrow$ lookup prefix `$sear`
    - Leading $*X$ (e.g., `*ing`) $\rightarrow$ lookup prefix `ing$`
    - Middle $X*Y$ (e.g., `m*n`) $\rightarrow$ lookup prefix `n$m`
  - Uses binary search range prefix matching over sorted keys.
- **K-gram Index**: Auxiliary character 3-gram index with post-filtering regex validation.

### 7. Frontend GUI (`src/gui.py`)
Built using Python's standard `tkinter` and `ttk` library:
- **Search Input Box**: Clean styled entry widget with `<Return>` keyboard trigger and Clear button.
- **Search Button**: Primary accent button (`🔍 Search`) for executing queries.
- **Read-Only Result Area**: Scrollable text view (`state='disabled'`) formatted with:
  - Tagged document headers and filenames
  - Matched terms badges and relevance scores
  - Colored text highlighting in context snippets
  - Wildcard expansion diagnostic panel
- **Corpus Management Bar**: Allows selecting any directory of text files and re-indexing dynamically.
- **Settings & Mode Selector**: Checkboxes for Porter Stemming and Stopwords, plus dropdown for query modes.

---

## 🚀 How to Run

### Requirements
- **Python 3.8+** (Tested on Python 3.11)
- Zero external libraries required.

### 1. Launch the Desktop GUI (Default)
```powershell
cd C:\Users\rathi\.gemini\antigravity\scratch\wsm_search_engine
python main.py
```
*The GUI opens pre-loaded with the sample corpus.*

### 2. Run in Interactive CLI Mode
```powershell
python main.py --cli
```

### 3. Run a Single CLI Query
```powershell
python main.py --query "web AND mining"
python main.py --query "sear*"
```

### 4. Run the Unit Test Suite
```powershell
python -m unittest discover tests
```

---

## 🔍 Example Queries to Try in the GUI

| Query Type | Query String | What It Tests |
|---|---|---|
| **Boolean AND** | `web AND mining` | Two-pointer intersection between documents containing both terms |
| **Boolean NOT** | `web NOT machine` | Postings difference (documents containing `web` but excluding `machine`) |
| **Boolean Compound** | `(web OR search) AND mining` | Operator precedence and nested parentheses evaluation |
| **Wildcard Trailing** | `sear*` | Permuterm lookup for terms beginning with `sear` (`search`, `searches`) |
| **Wildcard Suffix** | `*ing` | Permuterm lookup for terms ending in `ing` (`crawling`, `mining`, etc.) |
| **Wildcard Middle** | `m*n` | Middle rotation lookup in Permuterm index |
| **Phrase Query** | `"inverted index"` | Positional postings check for exact adjacent words |
| **Free-Text Ranked** | `information retrieval crawler` | TF-IDF ranked vector space search |
