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
    
    # Service URLs
    SCALEDOWN_API_URL: str = os.getenv(
        "SCALEDOWN_API_URL",
        "https://api.scaledown.xyz/compress/raw"
    )


# Export singleton instance
settings = Settings()
