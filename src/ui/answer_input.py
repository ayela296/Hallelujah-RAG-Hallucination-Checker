

import streamlit as st
MAX_ANSWER_LENGTH = 5000


def render_answer_input(max_length: int = MAX_ANSWER_LENGTH) -> str | None:
    """Renders the UI component for typing or pasting an AI-generated answer."""
    # Ensure session state persistence
    if "submitted_answer" not in st.session_state:
        st.session_state["submitted_answer"] = ""

    st.subheader("Input AI-Generated Answer")

    # Multi-line text input field
    answer_text = st.text_area(
        label="Paste or type the AI-generated answer below:",
        placeholder="Paste response text here...",
        height=200,
        max_chars=max_length,
        help=f"Maximum allowed length: {max_length:,} characters.",
    )

    # Character counter
    current_length = len(answer_text) if answer_text else 0
    st.caption(f"Character count: {current_length:,} / {max_length:,}")

    # Process submission
    if st.button("Submit for Verification", type="primary"):
        # AC: Empty answer validation
        if not answer_text or not answer_text.strip():
            st.error("Please enter an answer to check.")
            return None

        # AC: Max length validation
        if current_length > max_length:
            st.error(
                f"Input exceeds maximum allowed length of {max_length:,} characters."
            )
            return None

        # AC: Valid answer submission
        cleaned_answer = answer_text.strip()
        st.session_state["submitted_answer"] = cleaned_answer
        st.success("Answer accepted! Passing to Claim Extraction Agent...")
        return cleaned_answer

    return st.session_state.get("submitted_answer") or None