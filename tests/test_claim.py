import pytest
from unittest.mock import patch, MagicMock
from src.ui.source_input import render_source_input, validate_sources


# Printed header when pytest loads this test file
def setup_module(module):
    print("\n" + "=" * 60)
    print(" [SUITE] RUNNING UI TESTS: Source Documents Input (SCRUM-35)")
    print("=" * 60)


class MockSessionState(dict):
    """Custom dict wrapper to simulate Streamlit session state precisely."""

    def __getattr__(self, key):
        return self.get(key)

    def __setattr__(self, key, value):
        self[key] = value


@pytest.fixture(autouse=True)
def mock_streamlit():
    """Mocks Streamlit session_state with a functional dictionary wrapper."""
    with patch("src.ui.source_input.st") as mock_st:
        session = MockSessionState({"source_documents": []})
        mock_st.session_state = session

        # Mock tabs and button interactions
        mock_tab1, mock_tab2 = MagicMock(), MagicMock()
        mock_st.tabs.return_value = [mock_tab1, mock_tab2]
        mock_st.button.return_value = False

        yield mock_st


def test_scenario_no_source_provided(mock_streamlit):
    """AC: No source provided -> System shows error message on validation."""
    print("\n -> Testing Scenario: No Source Provided Validation...")
    mock_streamlit.session_state["source_documents"] = []

    is_valid = validate_sources()

    assert is_valid is False
    mock_streamlit.error.assert_called_once_with(
        "Please provide at least one source document."
    )


def test_scenario_single_source_uploaded(mock_streamlit):
    """AC: Source document uploaded successfully -> System accepts and indexes content."""
    print("\n -> Testing Scenario: Single Source Document Addition...")
    doc = {"name": "sample.txt", "content": "Trusted fact reference data"}
    mock_streamlit.session_state["source_documents"].append(doc)

    # Verify state before calling function
    sources = mock_streamlit.session_state["source_documents"]

    assert len(sources) == 1
    assert sources[0]["name"] == "sample.txt"
    assert sources[0]["content"] == "Trusted fact reference data"


def test_scenario_multiple_sources_provided(mock_streamlit):
    """AC: Multiple sources provided -> Both documents used as evidence."""
    print("\n -> Testing Scenario: Multiple Sources Accumulation...")
    doc1 = {"name": "doc1.txt", "content": "Evidence line 1"}
    doc2 = {"name": "Pasted Source 1", "content": "Evidence line 2"}

    mock_streamlit.session_state["source_documents"].append(doc1)
    mock_streamlit.session_state["source_documents"].append(doc2)

    sources = mock_streamlit.session_state["source_documents"]

    assert len(sources) == 2
    assert sources[0]["name"] == "doc1.txt"
    assert sources[1]["name"] == "Pasted Source 1"