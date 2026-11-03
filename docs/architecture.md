# Source document chunking

Uploaded PDF/TXT files and pasted text are kept in `source_documents` as
original documents with their names and full extracted content. When the user
starts verification, `chunk_documents` splits each source independently into
word-based passages (250 words by default, with 40 words of overlap). Each
passage contains `text`, `source`, and a zero-based `chunk_index`; the source
name is carried through unchanged.

The resulting passages are stored separately in
`st.session_state["source_chunks"]` so retrieval can consume them without
replacing or modifying the original documents. The retrieval layer accepts
extracted claim strings and chunk mappings, ranks passages with
inverse-document-frequency-weighted token overlap, and returns each claim with
up to three matching passages by default. Each passage includes its relevance
score, source name, and chunk index; claims with no matching terms receive an
empty evidence list. Retrieval operates on supplied chunks and does not create
or manage a vector index.