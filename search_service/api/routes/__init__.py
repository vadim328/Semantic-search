from fastapi import FastAPI
from .search_routes import create_search_routes
from .summarize_routes import create_summarize_router
from service.search_engine import SemanticSearchEngine


def register_routes(
        search_engine: SemanticSearchEngine
):

    routers = [
        create_search_routes(search_engine),
        create_summarize_router(),
        # сюда можно просто добавлять новые роутеры
    ]

    return routers
