# routes/search_routes.py
from typing import List
from fastapi import APIRouter, Request, HTTPException
from service.search_engine import SemanticSearchEngine
from fastapi.responses import JSONResponse
from config import Config
import logging

log = logging.getLogger(__name__)
cfg = Config().data


def validate_params(params: dict, req_params: List):
    for req_param in req_params:
        if not params.get(req_param):
            raise HTTPException(
                status_code=400,
                detail="Missing required parameters"
            )


def create_search_router(searcher: SemanticSearchEngine) -> APIRouter:

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
                    product: Название продукта для осуществления поиска
                            - При α = 0 полностью используется поиск по косинусной схожести
                            - При α = 1 полностью используется поиск через алгоритм BM25
                    exact: Включение быстрого поиска по индексированным векторам
                    filter: Фильтры по датам и клиенту для сужения поиска
        """

        data = await request.json()

        try:
            validate_params(data, ["query", "product"])
        except Exception as e:
            return e

        query = data.get("query")
        product = data.get("product")
        limit = data.get("limit", 5)
        alpha = data.get("alpha", 0.5)
        exact = data.get("exact", False)
        filters = data.get("filter", {})

        log.info(
            f"Request: {query}, "
            f"product: {product}, "
            f"limit: {limit}, "
            f"alpha: {alpha}, "
            f"exact: {exact}",
            f"filters: {filters}"
        )

        result = await searcher.search(query, product, limit, alpha, exact, filters)
        log.info(f"Request result: {result}")

        return JSONResponse(result)

    return router
