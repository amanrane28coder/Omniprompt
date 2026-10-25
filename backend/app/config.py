"""Server-side configuration only. Provider keys never leave this process."""
import os

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./omniprompt.db")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "")
MOCK_PROVIDER_ONLY = os.environ.get("MOCK_PROVIDER_ONLY", "false").lower() == "true"
