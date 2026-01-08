# routes/search_routes.py
from fastapi import APIRouter, Request
from service.pipeline import SemanticSearchEngine
from fastapi.responses import JSONResponse
import logging

log = logging.getLogger(__name__)


def create_search_router(searcher: SemanticSearchEngine) -> APIRouter:

    router = APIRouter(prefix="/search", tags=["Search"])

    @router.get("/options")
    def get_products():
        """
            GET - метод для получения списка проудктов и клиентов
        """
        return searcher.get_metadata()

    @router.post("/")
    async def search(request: Request):
        """
            POST - метод для поиска схожих запросов
                :input:
                    query: Текст, по которобу будут искаться схожие запросы
                    limit: Ограничение на количество найденых совпадений в порядке убывания
                    alpha: коэффициент балансировки, принимающий значения в диапазоне от 0 до 1
                            - При α = 0 полностью используется поиск через алгоритм BM25
                            - При α = 1 полностью используется поиск по косинусной схожести
                    exact: Включение быстрого поиска по индексированным векторам
                    filter: Фильтры по датам, продукту и клиенту для сужения поиска
        """
        data = await request.json()

        query = data.get("query")
        limit = data.get("limit", 5)
        alpha = data.get("alpha", 0.5)
        exact = data.get("exact", False)
        filters = data.get("filter", {})

        log.info(f"Request: {query}, limit: {limit}, alpha: {alpha}, exact: {exact}")

        result = await searcher.search(query, limit, alpha, exact, filters)
        log.info(f"Request result: {result}")

        return JSONResponse(result)

    return router
