# api/routes/summarize_routes.py
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from search_service.api.routes.validate_params import validate_params
from search_service.service.clients.summarization_orchestrator import SummarizationOrchestrator
from search_service.text_processing.text_preparation import transforms_nn, transforms_comments
import logging

log = logging.getLogger(__name__)


def create_summarize_router(summarization_orchestrator: SummarizationOrchestrator) -> APIRouter:
    router = APIRouter(prefix="/summarization", tags=["Summarize"])

    @router.post("/")
    async def summarize(request: Request):
        """
        POST - метод для суммаризации текста
        :input:
            text: строка, которую нужно суммаризировать
            comments: Комментарии запроса
        """
        data = await request.json()

        validate_params(data, ["text"])

        # Применяем очистку текста и комментариев
        def clean_input(dirty_text: str, dirty_comments: str = None):
            clean_text = transforms_nn(text=dirty_text)["text"]  # общий pipeline для текста
            clean_comments_text = transforms_comments(text=dirty_comments)["text"] if dirty_comments else None
            return clean_text, clean_comments_text

        text, comments = clean_input(data.get("text"), data.get("comments"))

        log.info(f"Request on summarization {text}")

        summary = await summarization_orchestrator.summarize(
            problem=text,
            comments=comments,
            max_concurrent=1,
        )

        return JSONResponse({"summary": summary})

    return router
