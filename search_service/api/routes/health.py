# api/routes/health.py
from typing import Dict
from fastapi import APIRouter

router = APIRouter(tags=["Health"])


@router.get("/health")
async def health() -> Dict:
    """
    GET - метод для проверки статуса сервиса
    Returns:
        Dict: Статус сервиса
    """
    return {"status": "ok"}
