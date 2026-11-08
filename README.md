# Hallelujah — RAG Hallucination Checker

**Hallelujah** is an agentic AI application designed to combat factual hallucinations in AI-generated content. The system processes an AI-generated text along with trusted source documents and verifies each factual statement against provided evidence.

It operates via a multi-agent pipeline: a **Claim Extraction Agent** breaks the generated text into atomic statements, a **RAG-based Retrieval Layer** fetches matching passage chunks from a vector database, and a **Verification Agent** classifies each claim as **Supported**, **Contradicted**, or **Not Found**.

---

## 📌 Key Highlights

- **Granular Claim Extraction:** Breaks down multi-sentence AI responses into standalone, testable factual assertions.
- **RAG-Powered Grounding:** Chunks `.pdf` and `.txt` reference documents and supports Gemini embeddings with a persistent FAISS similarity index.
- **Tri-State Verdict Engine:** Evaluates statements as **Supported**, **Contradicted**, or **Not Found** with direct source quote attribution.
- **Type-Safe Multi-Agent Pipeline:** Enforces Pydantic schema validation on `gemini-2.5-flash` for deterministic inter-agent data flow.
- **Interactive Streamlit UI:** Features real-time character bounds checking, color-coded verdict displays, and an interactive evidence inspector.


## 📂 Repository Structure

Hallelujah/
│
├── .env                         # Local keys (GEMINI_API_KEY, GOOGLE_API_KEY) — Excluded from Git
├── .env.example                 # Template configuration file for onboarding
├── .gitignore                   # Version control exclusions (.env, __pycache__, vector_db/)
├── README.md                    # Project documentation & setup instructions
├── requirements.txt             # Managed dependencies
├── main.py                      # Streamlit application entry point (SCRUM-40, SCRUM-43)
│
├── docs/                        # Project architecture & documentation (SCRUM-44)
│   └── architecture.md
│
├── prompts/                     # Prompt templates & system instructions (SCRUM-44)
│   ├── claim_extraction_prompt.txt
│   └── verification_prompt.txt
│
├── tests/                       # Automated test suite (SCRUM-44)
│   ├── __init__.py
│   ├── test_client.py
│   └── test_parser.py
│
└── src/                         # Core Source Package
    ├── __init__.py
    ├── config.py                # App configuration, model IDs, character bounds
    ├── ui/                      # Epic: UI Package
    ├── utils/                   # Epic: Utils Package
    ├── rag/                     # Epic: RAG Package
    │   ├── embeddings.py        # Gemini embedding API integration
    │   ├── vector_store.py      # Persistent FAISS index and cosine search
    │   └── index_documents.py   # Document indexing/search CLI
    └── agents/                  # Epic: Agents Package

## Semantic indexing and search

Install the project dependencies:

```powershell
python -m pip install -r requirements.txt
```

Copy `.env.example` to `.env` and set `GEMINI_API_KEY` to a key from
[Google AI Studio](https://aistudio.google.com). The `.env` file is ignored by
Git. The embedding pipeline uses this key with Google's official Gen AI SDK.

Index one or more PDF or TXT documents. The existing `chunk_documents`
implementation is used, and the index is saved in `vector_db/`:

```powershell
python -m src.rag.index_documents .\documents\guide.pdf .\documents\notes.txt
```

Re-running the command skips chunks already in the index. To search the saved
index (the default result limit is five):

```powershell
python -m src.rag.index_documents --search "What does the guide say about setup?"
```

The same indexing operation is available in the Streamlit app after chunks
have been created with **Run Hallucination Verification**, using the
**Build / update semantic search index** button.
