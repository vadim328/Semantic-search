# api/routes/summarize_routes.py
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from search_service.api.routes.validate_params import validate_params
from search_service.service.di import container
import logging

log = logging.getLogger(__name__)


def create_summarize_router() -> APIRouter:
    router = APIRouter(prefix="/summarization", tags=["Summarize"])

    @router.post("/")
    async def summarize(request: Request):
        """
        POST - метод для суммаризации текста
        :input:
            text: строка, которую нужно суммаризировать
            text: Комментарии запроса
        """
        data = await request.json()
        log.info(f"Request on summarization, data {data}")

        validate_params(data, ["text"])

        text = data.get("text")
        comments = data.get("comments", None)

        log.info(f"Request on summarization {text}")

        summary = container.model_client.make_summarize(
            problem=text,
            comments=comments
        )

        log.info(f"Summarized text: {summary}")

        return JSONResponse({"summary": summary})

    return router
