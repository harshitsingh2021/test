"""Runtime configuration. Every secret comes from the environment — nothing is hardcoded."""
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Postgres. Overridden by DATABASE_URL in .env / docker-compose.
    database_url: str = "postgresql+psycopg://aster:aster@localhost:5433/aster_oak"

    # LLM. The key is read from the environment and never written to the repo.
    anthropic_api_key: str | None = None
    llm_model: str = "claude-opus-5"
    llm_max_tokens: int = 2000
    llm_max_attempts: int = 3            # 1 initial call + 2 retries
    llm_backoff_base_seconds: float = 0.75
    llm_concurrency: int = 5             # bounded fan-out across campaign rows

    # Ingest
    raw_data_path: Path = PROJECT_ROOT / "campaigns_raw.json"

    # Self-check: flag when the LLM's health_score drifts this far from the
    # deterministic ROAS-band score.
    health_score_disagreement_threshold: int = 30

    @property
    def llm_enabled(self) -> bool:
        return bool(self.anthropic_api_key)


@lru_cache
def get_settings() -> Settings:
    return Settings()
