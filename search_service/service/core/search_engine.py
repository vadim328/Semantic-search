from typing import List, Dict, Any, Union, Optional
from search_service.service.core.scorer import HybridScorer
from search_service.service.core.search_mode import SearchMode
from search_service.text_processing.text_preparation import \
    transforms_embed, \
    transforms_llm, \
    transforms_comments
from qdrant_client.models import PointStruct
from search_service.service.utils.utils import timestamp_to_date
import asyncio
from search_service.config import Config
import logging


log = logging.getLogger(__name__)
cfg = Config().data["service"]["searcher"]


class SemanticSearchEngine:
    """Поисковой движок"""
    def __init__(self, container):
        self.container = container
        self.scorer = HybridScorer()
        self.threshold = cfg["threshold"]

        self.SEARCH_MODES = SearchMode.get_vector_names

    async def generate_result(self, calc_result: list[dict]) -> List[Dict]:
        """
        Формирует итоговый результат поиска
        Args:
            list[dict]: Результаты поиска
        Returns:
            list[dict]: Результат поиска с дополнительной информацией
        """

        additional_data = await self.container.relational_db.fetch_additional_data(
            {
                "numbers": [cr["id"] for cr in calc_result]
            }
        )

        result = []
        for cr, ad in zip(calc_result, additional_data):
            if cr["score"] < self.threshold:
                continue
            result.append({
                "id": str(cr["id"]),
                "score": str(round(cr["score"] * 100)) + "%",
                "responsible": ad["fio"],
                "priority": ad["admission_prority"],
                "registry_date": str(timestamp_to_date(cr["registry_date"])),
                "url": "https://support.naumen.ru/sd/operator/#uuid:%s" % ad["servicecall"]
            })

        return result

    @staticmethod
    def merge_hits(results: List[PointStruct]) -> Dict:
        """
        Объединение полученных результатов и их группировка по максимальному score
        Args:
            results (List[PointStruct]): Результаты по векторам
        Returns:
            Dict: Объединенные результаты
        """
        hits = {}

        for res in results:
            for point in res.points:
                pid = point.id
                score = point.score

                if pid not in hits or hits[pid]["score"] < score:
                    hits[pid] = {
                        "score": score,
                        "registry_date": point.payload.get("registry_date"),
                        "text": point.payload.get("text"),
                        "comments": point.payload.get("comments"),
                    }

        return hits

    async def _get_embedding(
            self,
            query: str,
    ) -> Any:
        """
        Метод для получения эмбеддинга запроса
        Args:
            query (str): Искомый текст или номер запроса
        Returns:
            np.ndarray: эмбеддинг запроса
        """
        # Если введен номер запроса, а не текст для поиска
        if query.isdigit():
            req_data = await self.container.relational_db.fetch_request_data(
                {"number": int(query)}
            )
            req_data = req_data[0]  # Берем первую и единствуенную строку

            query = await self.container.model_client.make_summarize(
                problem=transforms_llm(text=req_data["problem"])["text"],
                comments=transforms_comments(text=req_data["comments"])["text"]
            )

            return await self.container.model_client.embed(
                texts=query,
                prefix="query"
            )

        text = transforms_embed(text=query)["text"]
        return await self.container.model_client.embed(
            texts=text,
            prefix="query"
        )

    async def search(
            self,
            query: str,
            product: str,
            search_mode: SearchMode,
            *,
            limit: int = 5,
            alpha: float = 0.5,
            exact: bool = True,
            filters: Optional[Dict[str, Any]] = None
    ) -> Union[Dict, List[Dict]]:
        """
        Поиск информации в векторной БД по введенному тексту

        Args:
            query (str): Искомый текст
            product (str): Название продукта для поиска в нужной коллекции
            search_mode (str): Режим поиска
            limit (int): Количество лучших совпадений
            alpha (float): Баланс между значением косинусного расстояния и
                    алгоритма BM25
            exact (bool): Включение/отключение поиска по всем точкам коллекции
            filters (dict, None): Фильтры для сужения поиска

        :Returns:
            Union[Dict, List[Dict]]:: Отсортированный словарь с ИД запроса и точности сходства
        """
        if filters is None:
            filters = {}

        embedding = await self._get_embedding(query)
        try:
            # Берем коллекцию для продукта
            vector_db_collection = self.container.vector_db.collection(product)

            # Получаем результаты по векторам в коллекции, в зависимости от режима
            vector_names = search_mode.get_vector_names()

            log.info(f"Search text - {query} in product collection - {product}, vector names - {vector_names}")

            search_tasks = [
                vector_db_collection.fetch_embeddings(
                    vector_name=name,
                    vector=embedding,
                    exact=exact,
                    filters=filters
                )
                for name in vector_names
            ]

            # асинхронно делаем запросы для поиска
            results = await asyncio.gather(*search_tasks)

            hits = self.merge_hits(results)

            log.info(f"Result searching in vector db, found {len(hits)} points")

            ranked = self.scorer(
                hits=hits,
                query_text=query,
                search_mode=search_mode,
                alpha=alpha,
            )

            log.info(f'Result searching: {ranked}')

            return await self.generate_result(ranked[:limit])
        except ZeroDivisionError:
            return {"result": "data not found"}
        except Exception as e:
            log.error(f"Error: {e}")

    def get_metadata(self, product: str) -> Dict:
        """
        Формирует и передает метаданные
        Args:
            product (str): Название продукта
        Returns:
            Dict: метаданные
        """
        log.debug(f"Metadata for the '{product}' product was requested")
        res = self.container.vector_db.collection(product).metadata()
        log.debug(f"Metadata {res}")
        return res
