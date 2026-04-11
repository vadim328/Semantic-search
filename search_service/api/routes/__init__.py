from .search_routes import create_search_routes
from .summarize_routes import create_summarize_router
from search_service.container.di import Container
from search_service.service.core.search_engine import SemanticSearchEngine


def register_routes(
        searcher: SemanticSearchEngine,
        container: Container
):

    routers = [
        create_search_routes(searcher),
        create_summarize_router(container.summarization_orchestrator),
        # сюда можно просто добавлять новые роутеры
    ]

    return routers
