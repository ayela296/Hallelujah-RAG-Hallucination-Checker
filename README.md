# 🔍 Hallelujah — RAG Hallucination Checker

**Hallelujah** is an agentic AI application designed to combat factual hallucinations in AI-generated content. The system processes an AI-generated text along with trusted source documents and verifies each factual statement against provided evidence.

It operates via a multi-agent pipeline: a **Claim Extraction Agent** breaks the generated text into atomic statements, a **RAG-based Retrieval Layer** fetches matching passage chunks from a vector database, and a **Verification Agent** classifies each claim as **Supported**, **Contradicted**, or **Not Found**.

---

## 📌 Key Highlights

- **Granular Claim Extraction:** Breaks down multi-sentence AI responses into standalone, testable factual assertions.
- **RAG-Powered Grounding:** Indexes `.pdf` and `.txt` reference documents using `ChromaDB` vector embeddings to locate matching passage evidence.
- **Tri-State Verdict Engine:** Evaluates statements as **Supported**, **Contradicted**, or **Not Found** with direct source quote attribution.
- **Type-Safe Multi-Agent Pipeline:** Enforces Pydantic schema validation on `gemini-2.5-flash` for deterministic inter-agent data flow.
- **Interactive Streamlit UI:** Features real-time character bounds checking, color-coded verdict displays, and an interactive evidence inspector.


## 📂 Repository Structure

Hallelujah/
│
├── .env                         # Local environment variables (GOOGLE_API_KEY) — Excluded from Git
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
    └── agents/                  # Epic: Agents Package