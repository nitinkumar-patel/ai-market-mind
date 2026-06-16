from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from app.api.routes import router as api_router
from app.core.config import settings
from app.db import ensure_schema

_UI_HTML = Path(__file__).parent / "static" / "index.html"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    ensure_schema()
    yield
    # Shutdown – nothing for now


def create_app() -> FastAPI:
    app = FastAPI(
        title="MarketMind – Agentic Market Research Assistant",
        version="0.1.0",
        description="FastAPI backend orchestrating an autonomous market research agent with LangGraph + Postgres/pgvector.",
        lifespan=lifespan,
    )

    # Basic CORS – adjust for production
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_router, prefix="/api")

    return app


app = create_app()


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def serve_ui() -> str:
    return _UI_HTML.read_text()


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "environment": settings.environment}

