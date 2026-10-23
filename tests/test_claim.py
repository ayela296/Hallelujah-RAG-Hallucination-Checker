import pytest
from unittest.mock import MagicMock, patch
from src.utils.gemini_client import GeminiClient


def test_missing_api_key_raises_error():
    """AC: Missing or invalid key shows a clear error."""
    with pytest.raises(ValueError, match="API key is missing or invalid"):
        GeminiClient(api_key="")


@patch("src.utils.gemini_client.genai.Client")
def test_generate_text_success(mock_genai_client):
    """AC: Prompt returns a text response & Unit test with mocked response passes."""
    # Mock GenAI SDK response
    mock_instance = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "Hello! API connection is working."
    mock_instance.models.generate_content.return_value = mock_response
    mock_genai_client.return_value = mock_instance

    # Execute client call
    client = GeminiClient(api_key="fake_test_key", model_name="gemini-3.5-flash")
    result = client.generate_text("Say hello")

    # Assertions
    assert result == "Hello! API connection is working."
    mock_instance.models.generate_content.assert_called_once()


def test_empty_prompt_raises_error():
    """AC: Prompt validation error handling."""
    client = GeminiClient(api_key="fake_test_key")
    with pytest.raises(ValueError, match="Prompt cannot be empty"):
        client.generate_text("   ")