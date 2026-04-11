# api/routes/search.py
from typing import List, Dict
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from search_service.service.core.search_engine import SemanticSearchEngine
from search_service.api.schemas.search import SearchRequest
from search_service.api.deps.searcher import get_searcher
from search_service.config import Config
import logging

log = logging.getLogger(__name__)
cfg = Config().data


router = APIRouter(prefix="/search", tags=["Search"])


@router.get("/options/products")
def get_products(
        searcher: SemanticSearchEngine = Depends(get_searcher)
) -> List:
    """
        GET - метод для получения списка проудктов
    """
    return searcher.get_products()


@router.get("/options/metadata")
def get_metadata(
        product,
        searcher: SemanticSearchEngine = Depends(get_searcher)
) -> Dict:
    """
        GET - метод для получения метаданных
    """
    return searcher.get_metadata(product)


@router.post("/")
async def search(
        request: SearchRequest,
        searcher: SemanticSearchEngine = Depends(get_searcher)
) -> JSONResponse:
    """
    Выполняет поиск схожих запросов по заданным параметрам.

    POST-метод для поиска по тексту запроса с возможностью настройки
    количества результатов, режима поиска и фильтров.

    Args:
        request (SearchRequest): pydantic класс, содержащий валидируемые данные поиска.
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
        searcher (SemanticSearchEngine): Движок для поиска

    Returns:
        JSON: Список найденных схожих запросов с соответствующими метаданными.
    """

    log.info(
        f"Search request: query={request.query}, "
        f"product={request.product}, "
        f"limit={request.limit}, "
        f"alpha={request.alpha}, "
        f"mode={request.mode}, "
        f"exact={request.exact}, "
        f"filters={request.filter}"
    )

    result = await searcher.search(
        request.query,
        request.product,
        request.mode,
        limit=request.limit,
        alpha=request.alpha,
        exact=request.exact,
        filters=request.filter,
    )
    log.info(f"Result search request : {result}")

    return JSONResponse(result)
