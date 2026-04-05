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
        Выполняет поиск схожих запросов по заданным параметрам.

        POST-метод для поиска по тексту запроса с возможностью настройки
        количества результатов, режима поиска и фильтров.

        Args:
            request (Request): Объект запроса FastAPI, содержащий данные поиска.
                Ожидается, что тело запроса содержит JSON с полями:
                    query (str): Текст, по которому выполняется поиск схожих запросов.
                    limit (int, optional): Максимальное количество результатов (по убыванию). По умолчанию без ограничения.
                    alpha (float, optional): Коэффициент балансировки между косинусной схожестью и BM25 (0 ≤ α ≤ 1).
                        - α = 0: полностью используется поиск по косинусной схожести.
                        - α = 1: полностью используется поиск через алгоритм BM25.
                    mode (str, optional): Режим поиска. Возможные значения: "base", "full", "comments".
                    product (str, optional): Название продукта для поиска.
                    exact (bool, optional): Включение быстрого поиска по индексированным векторам.
                    filter (dict, optional): Фильтры для сужения поиска, например, по дате или клиенту.

        Returns:
            JSON: Список найденных схожих запросов с соответствующими метаданными.
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
            f"exact: {exact}, "
            f"filters: {filters}"
        )

        result = await searcher.search(query, product, search_mode, limit, alpha, exact, filters)
        log.info(f"Result search request : {result}")

        return JSONResponse(result)

    return router
