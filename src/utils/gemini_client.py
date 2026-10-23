import logging
from google import genai
from google.genai import types
from google.genai.errors import APIError

from src.config import Config

logger = logging.getLogger(__name__)

class GeminiClient:
    """Reusable LLM client for Hallelujah agent pipelines."""

    def __init__(self, api_key: str = None, model_name: str = None):
        if api_key is None:
            self.api_key = Config.GOOGLE_API_KEY
        else:
            self.api_key = api_key

        if not self.api_key or not str(self.api_key).strip():
            raise ValueError("Cannot initialize GeminiClient: API key is missing or invalid.")
          
        self.model_name = model_name or Config.GEMINI_MODEL

        # Acceptance Criterion: Missing or invalid key raises a clear error
        if not self.api_key:
            raise ValueError("Cannot initialize GeminiClient: API key is missing or invalid.")

        self.client = genai.Client(api_key=self.api_key)
        
        # Suppress automatic function calling console warnings
        self.config = types.GenerateContentConfig(
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)
        )

    def generate_text(self, prompt: str) -> str:
        """Sends a prompt to Gemini and returns the model's text response."""
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        try:
            # Acceptance Criterion: Prompt returns a text response
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=self.config
            )
            
            if hasattr(response, 'text') and response.text:
                return response.text
            
            raise RuntimeError("Gemini API returned an empty response.")

        except APIError as e:
            logger.error(f"Gemini API Error: {e}")
            raise RuntimeError(f"Gemini API call failed: {e.message}") from e
        except TimeoutError as e:
            # Acceptance Criterion: Timeout handled with a clear message
            logger.error("Gemini API call timed out.")
            raise TimeoutError("The request to Gemini API timed out. Please try again.") from e
        except Exception as e:
            logger.error(f"Unexpected error calling Gemini API: {e}")
            raise RuntimeError(f"An unexpected error occurred: {str(e)}") from e