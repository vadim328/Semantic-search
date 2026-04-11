from fastapi import Request
from search_service.service.clients.summarization_orchestrator import SummarizationOrchestrator


def get_orchestrator(request: Request) -> SummarizationOrchestrator:
    return request.app.state.container.summarization_orchestrator

