# Source document chunking

Uploaded PDF/TXT files and pasted text are kept in `source_documents` as
original documents with their names and full extracted content. When the user
starts verification, `chunk_documents` splits each source independently into
word-based passages (250 words by default, with 40 words of overlap). Each
passage contains `text`, `source`, and a zero-based `chunk_index`; the source
name is carried through unchanged.

The resulting passages are stored separately in
`st.session_state["source_chunks"]` so retrieval can consume them without
replacing or modifying the original documents. Each passage also has a
deterministic ID and can retain optional document metadata.

Semantic search is provided by `src.rag.embeddings` and
`src.rag.vector_store`: the official Google Gen AI SDK generates
`gemini-embedding-001` document/query vectors, and a normalized FAISS
inner-product index ranks cosine similarity. The default index and its source
text/metadata are persisted under `vector_db/`. Re-indexing skips existing
chunk IDs. See the README for CLI commands and API-key setup.
