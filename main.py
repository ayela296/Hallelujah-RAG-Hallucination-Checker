import streamlit as st
import logging
from google.genai.errors import APIError
from src.agents.claim_extractor import extract_claims
from src.rag.claims import extract_and_store_claims
from src.ui.answer_input import render_answer_input
from src.ui.source_input import render_source_input, validate_sources

logger = logging.getLogger(__name__)

# 1. Page Configuration
st.set_page_config(
    page_title="Hallelujah - RAG Hallucination Checker",
    page_icon="🔍",
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
        st.session_state["claims"] = []
        st.session_state.pop("claims_source_answer", None)
        st.error("Please enter an AI-generated answer to check.")
        st.stop()

    # Validate Source Documents
    if not validate_sources():
        st.stop()

    try:
        with st.spinner("Extracting factual claims..."):
            claims = extract_and_store_claims(
                submitted_answer,
                st.session_state,
                extractor=extract_claims,
            )

        st.success(
            f"Extracted {len(claims)} claim(s) from the submitted answer and "
            f"prepared them for retrieval against {len(source_documents)} source document(s)."
        )
        if claims:
            st.subheader("Extracted Claims")
            for claim in claims:
                st.markdown(f"- {claim['claim_text']}")
        else:
            st.info("No factual claims were found in the submitted answer.")
    except (APIError, OSError, RuntimeError, TypeError, ValueError) as error:
        logger.exception("Could not extract claims from the submitted answer")
        st.error(f"Could not extract claims: {error}")