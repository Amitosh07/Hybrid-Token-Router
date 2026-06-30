import os
from functools import lru_cache


class Settings:
    """Application settings loaded from environment variables."""

    APP_NAME: str = os.getenv("APP_NAME", "Hybrid Token Router API")
    APP_VERSION: str = os.getenv("APP_VERSION", "1.0.0")

    # TODO:
    # Replace with your frontend URL in .env, for example:
    # ALLOWED_ORIGINS=http://localhost:5173
    ALLOWED_ORIGINS: list[str] = [
        origin.strip()
        for origin in os.getenv("ALLOWED_ORIGINS", "http://localhost:5173").split(",")
        if origin.strip()
    ]


@lru_cache
def get_settings() -> Settings:
    return Settings()
