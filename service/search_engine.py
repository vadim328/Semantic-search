import asyncio
#from datetime import datetime, timedelta
from service.di import container
from models.embedding_request import fetch_embedding
from service.scorer import HybridScorer
from text_processing.text_preparation import transforms_bm25, transforms_bert
from service.utils import timestamp_to_date
import logging
from config import Config

log = logging.getLogger(__name__)
cfg = Config().data


class SemanticSearchEngine:
    def __init__(self):
        self.container = container
        self.scorer = HybridScorer()
        self.threshold = cfg["service"]["threshold"]

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
                "date_end": str(timestamp_to_date(cr["date_end"])),
                "url": "https://support.naumen.ru/sd/operator/#uuid:%s" % ad["servicecall"]
            })

        return result

    def _get_embedding(self, product: str, problem: str):
        # Временная функция
        """
            Метод для получения эмбеддинга запроса
                input:
                    row - запись с полями
                output:
                    vector - эмбеддинг запроса
        """
        if product == "Naumen Erudite":
            return fetch_embedding(
                self.container.llm_models[product],
                self.container.embedding_model,
                problem,
            )

        text = transforms_bert(text=problem)["text"]

        return self.container.embedding_model.encode(text)

    async def search(
            self,
            query: str,
            product: str,
            limit=5,
            alpha=0.5,
            exact=True,
            filters=None
    ):
        """
            Поиск информации по векторной БД

            :input:
                str: Искомый текст
                int: Количество лучших совпадений
                float: Определяет баланс между значением косинусного расстояния и
                        алгоритма BM25

            :output:
                dict: Отсортированный словарь с ИД запроса и значению комбинированного сходства
        """
        if filters is None:
            filters = {}
        # Для поиска по ключевым словам лучше увеличить альфу
        tokenized_query = transforms_bm25(text=query)["text"].split()
        log.debug(f'transforms text for bm25: {tokenized_query}')

        embedding = self._get_embedding(product, query)
        try:
            # Получаем эмбеддинги из нужной коллекции в взависимости от продукта
            hits = self.container.vector_dbs[product].fetch_embeddings(
                embedding,
                exact,
                filters
            )

            ranked = self.scorer(
                hits=hits.points,
                query_text=query,
                alpha=alpha
            )

            log.info(f'Result qdrant fetching: {ranked}')

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
        res = self.container.vector_dbs[product].get_metadata()
        log.debug(f"Metadata {res}")
        return res
