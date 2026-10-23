

import pytest
from unittest.mock import patch, MagicMock
from src.ui.answer_input import render_answer_input, MAX_ANSWER_LENGTH


# Print header when pytest loads this test module
def setup_module(module):
    print("\n" + "=" * 60)
    print(" [SUITE] RUNNING UI TESTS: AI-Generated Answer Input (SCRUM-34)")
    print("=" * 60)


@pytest.fixture(autouse=True)
def mock_streamlit():
    """Mocks Streamlit session_state and UI components for testing."""
    with patch("src.ui.answer_input.st") as mock_st:
        mock_st.session_state = {}
        yield mock_st


def test_scenario_valid_answer_submitted(mock_streamlit):
    """AC: Valid AI answer submitted -> System accepts answer and stores in session_state."""
    print("\n -> Testing Scenario: Valid AI Answer Submission...")
    valid_text = "This is a valid AI-generated answer about quantum computing."
    mock_streamlit.text_area.return_value = valid_text
    mock_streamlit.button.return_value = True

    result = render_answer_input()

    assert result == valid_text
    assert mock_streamlit.session_state["submitted_answer"] == valid_text
    mock_streamlit.success.assert_called_once_with(
        "Answer accepted! Passing to Claim Extraction Agent..."
    )


def test_scenario_empty_answer_submitted(mock_streamlit):
    """AC: Empty answer submitted -> System blocks processing and shows error message."""
    print("\n -> Testing Scenario: Empty AI Answer Submission...")
    mock_streamlit.text_area.return_value = "   "
    mock_streamlit.button.return_value = True

    result = render_answer_input()

    assert result is None
    mock_streamlit.error.assert_called_once_with(
        "Please enter an answer to check."
    )


def test_scenario_answer_exceeds_max_length(mock_streamlit):
    """AC: Exceeds max length -> System rejects input and shows max length error."""
    print("\n -> Testing Scenario: AI Answer Exceeds 5,000 Chars Limit...")
    long_text = "A" * (MAX_ANSWER_LENGTH + 10)
    mock_streamlit.text_area.return_value = long_text
    mock_streamlit.button.return_value = True

    result = render_answer_input(max_length=MAX_ANSWER_LENGTH)

    assert result is None
    mock_streamlit.error.assert_called_once_with(
        f"Input exceeds maximum allowed length of {MAX_ANSWER_LENGTH:,} characters."
    )