from fastapi import Request
from search_service.service.core.search_engine import SemanticSearchEngine


def get_searcher(request: Request) -> SemanticSearchEngine:
    return SemanticSearchEngine(request.app.state.container)
