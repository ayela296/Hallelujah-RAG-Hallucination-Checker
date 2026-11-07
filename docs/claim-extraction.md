# Claim extraction handoff

The submitted Sprint 1 answer is persisted as
`st.session_state["submitted_answer"]`. Clicking **Run Hallucination
Verification** sends that answer to `src.agents.claim_extractor.extract_claims`.

The resulting claims are stored in the shared
`st.session_state["claims"]` list. Each record contains a unique `claim_id`,
the original `claim_text`, its `source`, and a `pending` status for the
retrieval stage. The associated `claims_source_answer` prevents repeating the
Gemini extraction when the same answer is checked again. A new answer replaces
the previous claim list.
