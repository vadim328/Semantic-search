from search_service.service.core.scorer import HybridScorer
from search_service.text_processing.text_preparation import transforms_bert, transforms_nn, transforms_comments
from search_service.service.utils.utils import timestamp_to_date
import asyncio
from search_service.config import Config
import logging


log = logging.getLogger(__name__)
cfg = Config().data["service"]["searcher"]


class SemanticSearchEngine:
    def __init__(self, container):
        self.container = container
        self.scorer = HybridScorer()
        self.threshold = cfg["threshold"]

        # TODO Можно перенести в конфиг
        self.SEARCH_MODES = {
            "full": ["original", "summary", "comments"],
            "base": ["original", "summary"],
            "comments": ["comments"],
        }

    async def generate_result(self, calc_result: list[dict]):
        """
            Формирует итоговый результат поиска
            :input:
                list[dict]: Результаты поиска
            :output:
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
    def merge_hits(results):
        """
            Объединение полученных результатов и
                их группировка по максимальному score

            input:
                list[list] - Результаты по векторам
            output:
                list - Объединенные результаты
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
                        "text": point.payload.get("text")
                    }

        return hits

    async def _get_embedding(
            self,
            query: str,
    ):
        """
            Метод для получения эмбеддинга запроса
                input:
                    query - Искомый текст или номер запроса
                output:
                    vector - эмбеддинг запроса
        """
        # Если введен номер запроса, а не текст для поиска
        if query.isdigit():
            req_data = await self.container.relational_db.fetch_request_data(
                {"number": int(query)}
            )
            req_data = req_data[0]  # Берем первую и единствуенную строку

            query = await self.container.model_client.make_summarize(
                problem=transforms_nn(text=req_data["problem"])["text"],
                comments=transforms_comments(text=req_data["comments"])["text"]
            )

            return await self.container.model_client.embed(query)

        text = transforms_bert(text=query)["text"]
        return await self.container.model_client.embed(text)

    async def search(
            self,
            query: str,
            product: str,
            search_mode: str,
            limit=5,
            alpha=0.5,
            exact=True,
            filters=None
    ):
        """
            Поиск информации в векторной БД по введенному тексту

            :input:
                query: Искомый текст
                product: Название продукта для поиска в нужной коллекции
                limit: Количество лучших совпадений
                alpha: Баланс между значением косинусного расстояния и
                        алгоритма BM25
                exact: Включение/отключение поиска по всем точкам коллекции
                filters: Фильтры для сужения поиска

            :output:
                dict: Отсортированный словарь с ИД запроса и точности сходства
        """
        if filters is None:
            filters = {}

        embedding = await self._get_embedding(query)
        try:
            # Берем коллекцию для продукта
            vector_db_collection = self.container.vector_db.collection(product)

            # Получаем результаты по векторам в коллекции, в зависимости от режима
            vector_names = self.SEARCH_MODES[search_mode]

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
                alpha=alpha
            )

            log.info(f'Result searching: {ranked}')

            return await self.generate_result(ranked[:limit])
        except ZeroDivisionError:
            return {"result": "data not found"}
        except Exception as e:
            log.error(f"Error: {e}")

    def get_metadata(self, product):
        """
            Формирует и передает метаданные
                :input:
                    str: Название продукта
                :output:
                    dict: метаданные
        """
        log.debug(f"Metadata for the '{product}' product was requested")
        res = self.container.vector_db.collection(product).metadata()
        log.debug(f"Metadata {res}")
        return res
