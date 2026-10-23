import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    # Confirmed model from SCRUM-11
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
    DEFAULT_TIMEOUT: int = 15

    @classmethod
    def validate(cls):
        """Validates that critical environment variables exist."""
        if not cls.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY is missing or invalid in .env file.")