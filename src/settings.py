"""Centralized settings and configuration management.

Loads API keys and service URLs from environment variables with sensible defaults.
"""
import os


class Settings:
    """Application settings loaded from environment variables."""
    
    # API Keys (optional - not all may be needed for every test)
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    SCALEDOWN_API_KEY: str = os.getenv("SCALEDOWN_API_KEY", "")
    LANGSMITH_API_KEY: str = os.getenv(
        "LANGSMITH_API_KEY", os.getenv("LANGCHAIN_API_KEY", "")
    )
    LANGSMITH_PROJECT: str = os.getenv("LANGSMITH_PROJECT", "")
    
    # Service URLs
    SCALEDOWN_API_URL: str = os.getenv("SCALEDOWN_API_URL", "")
    SCALEDOWN_SUMMARIZE_URL: str = os.getenv("SCALEDOWN_SUMMARIZE_URL", "")


# Export singleton instance
settings = Settings()
