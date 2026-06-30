import os
from functools import lru_cache
from pathlib import Path


def _load_env_file() -> None:
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue

        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


class Settings:
    """Application settings loaded from environment variables."""

    def __init__(self) -> None:
        self.APP_NAME: str = os.getenv("APP_NAME", "Hybrid Token Router API")
        self.APP_VERSION: str = os.getenv("APP_VERSION", "1.0.0")

        # TODO:
        # Replace with your frontend URL in backend/.env, for example:
        # FRONTEND_URL=http://localhost:5173
        self.FRONTEND_URL: str = os.getenv("FRONTEND_URL", "http://localhost:5173")

        # TODO:
        # Add your local Ollama URL and model in backend/.env.
        # Example:
        # OLLAMA_BASE_URL=http://localhost:11434
        # OLLAMA_MODEL=llama3.1
        self.OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "")

        # TODO:
        # Add your free remote provider settings in backend/.env.
        # For Groq, REMOTE_BASE_URL is usually https://api.groq.com/openai/v1
        # and REMOTE_API_KEY should be your Groq API key.
        self.REMOTE_API_KEY: str = os.getenv("REMOTE_API_KEY", "")
        self.REMOTE_BASE_URL: str = os.getenv("REMOTE_BASE_URL", "")
        self.REMOTE_MODEL: str = os.getenv("REMOTE_MODEL", "")

    @property
    def ALLOWED_ORIGINS(self) -> list[str]:
        return [self.FRONTEND_URL]


@lru_cache
def get_settings() -> Settings:
    _load_env_file()
    return Settings()
