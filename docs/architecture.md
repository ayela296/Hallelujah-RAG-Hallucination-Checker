# Source document chunking

Uploaded PDF/TXT files and pasted text are kept in `source_documents` as
original documents with their names and full extracted content. When the user
starts verification, `chunk_documents` splits each source independently into
word-based passages (250 words by default, with 40 words of overlap). Each
passage contains `text`, `source`, and a zero-based `chunk_index`; the source
name is carried through unchanged.

The resulting passages are stored separately in
`st.session_state["source_chunks"]` so a future retrieval/indexing step can
consume them without replacing or modifying the original documents. This
project does not currently implement retrieval or vector-store indexing.