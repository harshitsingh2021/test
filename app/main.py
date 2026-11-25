"""Aster & Oak campaign intelligence service."""
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import get_settings
from app.db import init_db
from app.routers import campaigns

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s | %(message)s"
)

@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    init_db()
    settings = get_settings()
    logging.getLogger(__name__).info(
        "ready | model=%s | llm=%s",
        settings.llm_model,
        "enabled"
        if settings.llm_enabled
        else "DISABLED (no ANTHROPIC_API_KEY — ingest falls back to deterministic enrichment)",
    )
    yield


app = FastAPI(
    lifespan=lifespan,
    title="Aster & Oak — Campaign Intelligence",
    version="1.0.0",
    description=(
        "Ingests a messy ad-ops campaign export, normalizes and enriches it with an LLM "
        "via strict tool calling, stores it in Postgres, and serves it back."
    ),
)
app.include_router(campaigns.router)



@app.get("/healthz", tags=["meta"])
def healthz() -> dict:
    settings = get_settings()
    return {
        "status": "ok",
        "llm_enabled": settings.llm_enabled,
        "model": settings.llm_model,
    }
