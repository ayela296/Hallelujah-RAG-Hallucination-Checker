import streamlit as st
import pypdf


def extract_text_from_file(uploaded_file) -> str:
    """Extracts raw text from uploaded .txt or .pdf files."""
    if uploaded_file.type == "text/plain":
        return uploaded_file.read().decode("utf-8", errors="ignore")
    elif uploaded_file.type == "application/pdf":
        pdf_reader = pypdf.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
        return text
    return ""


def render_source_input() -> list[dict]:
    """Renders the UI component for uploading or pasting source documents."""
    # Ensure session state tracking for multi-source accumulation
    if "source_documents" not in st.session_state:
        st.session_state["source_documents"] = []

    st.subheader("Source Documents / Evidence")

    tab_upload, tab_paste = st.tabs(["Upload Files (.txt, .pdf)", "Paste Text"])

    # Tab 1: File Upload
    with tab_upload:
        uploaded_files = st.file_uploader(
            "Upload reference documents",
            type=["txt", "pdf"],
            accept_multiple_files=True,
            help="Upload one or more .txt or .pdf files.",
        )
        if st.button("Add Uploaded File(s)", key="btn_add_files"):
            if not uploaded_files:
                st.warning("Please select a file to upload.")
            else:
                for file in uploaded_files:
                    content = extract_text_from_file(file)
                    if content.strip():
                        doc_entry = {"name": file.name, "content": content.strip()}
                        if doc_entry not in st.session_state["source_documents"]:
                            st.session_state["source_documents"].append(doc_entry)
                st.success(f"Added {len(uploaded_files)} file(s) successfully.")

    # Tab 2: Pasted Text
    with tab_paste:
        doc_name = st.text_input("Document Name / Title", value="Pasted Source 1")
        pasted_text = st.text_area("Paste source reference text below:", height=150)
        if st.button("Add Pasted Text", key="btn_add_text"):
            if not pasted_text.strip():
                st.warning("Pasted text cannot be empty.")
            else:
                doc_entry = {"name": doc_name.strip(), "content": pasted_text.strip()}
                st.session_state["source_documents"].append(doc_entry)
                st.success(f"Added '{doc_name}' successfully.")

    # Display accumulated sources
    sources = st.session_state["source_documents"]
    if sources:
        st.write(f"**Loaded Sources ({len(sources)}):**")
        for idx, doc in enumerate(sources, 1):
            st.caption(f"{idx}. **{doc['name']}** ({len(doc['content']):,} chars)")

        if st.button("Clear All Sources", type="secondary"):
            st.session_state["source_documents"] = []
            st.rerun()

    return st.session_state["source_documents"]


def validate_sources() -> bool:
    """AC: Checks if at least one source document is provided."""
    sources = st.session_state.get("source_documents", [])
    if not sources:
        st.error("Please provide at least one source document.")
        return False
    return True