# api/routes/search_routes.py
from fastapi import APIRouter, Request
from search_service.service.core.search_engine import SemanticSearchEngine
from fastapi.responses import JSONResponse
from search_service.api.routes.validate_params import validate_params
from search_service.config import Config
import logging

log = logging.getLogger(__name__)
cfg = Config().data


def create_search_routes(searcher: SemanticSearchEngine) -> APIRouter:

    router = APIRouter(prefix="/search", tags=["Search"])

    @router.get("/options/products")
    def get_products():
        """
            GET - метод для получения списка проудктов и клиентов
        """
        return cfg["service"]["products"]

    @router.get("/options/metadata")
    def get_metadata(product):
        """
            GET - метод для получения списка проудктов и клиентов
        """
        return searcher.get_metadata(product)

    @router.post("/")
    async def search(request: Request):
        """
            POST - метод для поиска схожих запросов
                :input:
                    query: Текст, по которобу будут искаться схожие запросы
                    limit: Ограничение на количество найденых совпадений в порядке убывания
                    alpha: коэффициент балансировки, принимающий значения в диапазоне от 0 до 1
                    mode: Режим поиска - base, full, comments
                    product: Название продукта для осуществления поиска
                            - При α = 0 полностью используется поиск по косинусной схожести
                            - При α = 1 полностью используется поиск через алгоритм BM25
                    exact: Включение быстрого поиска по индексированным векторам
                    filter: Фильтры по датам и клиенту для сужения поиска
        """

        data = await request.json()

        validate_params(data, ["query", "product"])

        query = data.get("query")
        product = data.get("product")
        limit = data.get("limit", 5)
        alpha = data.get("alpha", 0.5)
        search_mode = data.get("mode", "base")
        exact = data.get("exact", False)
        filters = data.get("filter", {})

        log.info(
            f"Request: {query}, "
            f"product: {product}, "
            f"limit: {limit}, "
            f"alpha: {alpha}, "
            f"search mode: {search_mode}, "
            f"exact: {exact}",
        )

        result = await searcher.search(query, product, search_mode, limit, alpha, exact, filters)
        log.info(f"Result search request : {result}")

        return JSONResponse(result)

    return router
