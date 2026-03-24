from search_service.service.scorer import HybridScorer
from search_service.text_processing.text_preparation import transforms_bert
from search_service.service.utils import timestamp_to_date
import asyncio
import logging
from search_service.config import Config

log = logging.getLogger(__name__)
cfg = Config().data


class SemanticSearchEngine:
    def __init__(self, container):
        self.container = container
        self.scorer = HybridScorer()
        self.threshold = cfg["service"]["threshold"]

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
            product: str,
            problem: str,
            comments=None,
    ):
        # Временная функция
        """
            Метод для получения эмбеддинга запроса
                input:
                    row - запись с полями
                output:
                    vector - эмбеддинг запроса
        """
        if product == "Naumen":

            problem_summary = await self.container.model_client.make_summarize(
                problem=problem,
                comments=comments
            )
            return await self.container.model_client.embed(problem_summary)

        text = transforms_bert(text=problem)["text"]
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

        # Если введен номер запроса, а не текст для поиска
        if query.isdigit():
            req_data = await self.container.relational_db.fetch_request_data(
                {"number": int(query)}
            )
            req_data = req_data[0]  # Берем первую и единствуенную строку
            embedding = await self._get_embedding(
                product,
                req_data["problem"],
                req_data["comments"]
            )
        else:
            embedding = await self._get_embedding(product, query)
        try:
            # Берем коллекцию для продукта
            vector_db_collection = self.container.vector_db.collection(product)

            # Получаем результаты по векторам в коллекции, в зависимости от режима
            vector_names = self.SEARCH_MODES[search_mode]

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
                :output:
                    dict: метаданные
        """
        log.debug(f"Metadata for the '{product}' product was requested")
        res = self.container.vector_db.collection(product).metadata()
        log.debug(f"Metadata {res}")
        return res
