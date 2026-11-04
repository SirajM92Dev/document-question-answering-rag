from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    openai_api_key: str | None = None
    openai_chat_model: str = "gpt-4o-mini"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chroma_persist_directory: Path = Path("data/chroma")
    upload_directory: Path = Path("data/uploads")
    top_k: int = 5
    chunk_size: int = 900
    chunk_overlap: int = 150
    max_upload_mb: int = 25

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.chroma_persist_directory.mkdir(parents=True, exist_ok=True)
    settings.upload_directory.mkdir(parents=True, exist_ok=True)
    return settings
