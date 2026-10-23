import streamlit as st
from src.ui.answer_input import render_answer_input
from src.ui.source_input import render_source_input, validate_sources

# 1. Page Configuration
st.set_page_config(
    page_title="Hallelujah - RAG Hallucination Checker",
    layout="wide",
)

st.title("Hallelujah: RAG Hallucination Checker")
st.markdown(
    "Verify the factual accuracy of AI-generated responses against trusted source evidence."
)
st.divider()

# 2. Main Layout (Two-column interface)
col_answer, col_sources = st.columns([1, 1], gap="large")

# Left Column: AI Answer Input (SCRUM-34)
with col_answer:
    submitted_answer = render_answer_input()

# Right Column: Source Documents Input (SCRUM-35)
with col_sources:
    source_documents = render_source_input()

st.divider()

# 3. Overall Verification Trigger
if st.button("Run Hallucination Verification", type="primary", use_container_width=True):
    # Validate AI Answer Input
    if not submitted_answer:
        st.error("Please enter an AI-generated answer to check.")
        st.stop()

    # Validate Source Documents
    if not validate_sources():
        st.stop()

    # Success State: Input Data Ready for Claim Extraction
    st.success(
        f"Ready for Claim Extraction! Processing 1 AI Answer "
        f"({len(submitted_answer):,} chars) against {len(source_documents)} source document(s)."
    )
    
    # Next Pipeline Step: Pass submitted_answer & source_documents to Claim Extraction Agent