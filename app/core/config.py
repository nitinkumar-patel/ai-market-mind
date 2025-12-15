from functools import lru_cache
from typing import Literal, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Strict logical environment; keep to a small, explicit set
    environment: Literal["local", "dev", "prod"] = "local"

    # Postgres / pgvector – required from env (.env or Docker env)
    postgres_host: str
    postgres_port: int
    postgres_db: str
    postgres_user: str
    postgres_password: str

    # LLM provider: "openai" or "ollama"
    llm_provider: Literal["openai", "ollama"] = "openai"

    # OpenAI
    openai_embedding_model: str = "text-embedding-3-small"
    openai_api_key: Optional[str] = None

    # Ollama
    ollama_base_url: str = "http://host.docker.internal:11434"
    ollama_model: str = "llama3.1"

    # Tools
    tavily_api_key: Optional[str] = None

    # App
    debug: bool = True


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()


