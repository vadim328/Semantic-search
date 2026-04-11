# api/routes/summarize.py
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from search_service.infrastructure.clients.summarization_orchestrator import SummarizationOrchestrator
from search_service.text_processing.text_preparation import transforms_llm, transforms_comments
from search_service.api.schemas.summarization import SummarizeRequest
from search_service.api.deps.orchestrator import get_orchestrator
import logging

log = logging.getLogger(__name__)

router = APIRouter(prefix="/summarization", tags=["Summarize"])


@router.post("/")
async def summarize(
        request: SummarizeRequest,
        orchestrator: SummarizationOrchestrator = Depends(get_orchestrator)
):
    """
    POST - метод для суммаризации текста
    Args:
        request (SummarizeRequest): pydantic класс, содержащий валидируемые данные для суммаризации.
            Ожидается, что тело запроса содержит JSON с полями:
                text: строка, которую нужно суммаризировать
                comments: Комментарии запроса
        orchestrator (SummarizationOrchestrator): оркестратор для суммаризации
    """

    # Применяем очистку текста и комментариев (при наличии)
    text = transforms_llm(text=request.text)["text"]

    comments = None
    if request.comments:
        comments = transforms_comments(text=request.comments)["text"]

    log.info(f"Request on summarization {text}")

    summary = await orchestrator.summarize(
        problem=text,
        comments=comments,
        max_concurrent=1,
    )

    return JSONResponse({"summary": summary})


