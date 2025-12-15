from fastapi import APIRouter

from app.api.v1 import research

router = APIRouter()

router.include_router(research.router, prefix="/v1", tags=["research"])


