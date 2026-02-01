import asyncio
from datetime import datetime, timedelta
from models.model import OnnxSentenseTransformer
from service.di import model, relational_db, vector_db
from db.database import RelationalDatabaseTouch, VectorDatabaseTouch
from service.scorer import HybridScorer
from text_processing.text_preparation import transforms_bm25, transforms_bert
from service.utils import timestamp_to_date
import logging
from config import Config

log = logging.getLogger(__name__)
cfg = Config().data


class SemanticSearchEngine:
    def __init__(self):
        self.model = model
        self.relational_db = relational_db
        self.vector_db = vector_db

        self.scorer = HybridScorer()

    async def generate_result(self, calc_result: list[dict]):
        """
            Формирует итоговый результат поиска
            :input:
                list[dict]: Результаты поиска
            :output:
                list[dict]: Результат поиска с дополнительной информацией
        """

        additional_data = await self.relational_db.fetch_additional_data(
            {
                "numbers": [cr["id"] for cr in calc_result]
            }
        )

        result = []
        for cr, ad in zip(calc_result, additional_data):
            if cr["score"] < cfg["service"]["threshold"]:
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

    async def search(
            self,
            query: str,
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

        query_bert = transforms_bert(text=query)["text"]
        log.debug(f'transforms text for NN: {query_bert}')

        embedding = self.model.encode(query_bert)[0]
        try:
            hits = self.vector_db.fetch_embeddings(embedding, exact, filters)

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

    def get_metadata(self):
        """
            Формирует и передает метаданные
                :output:
                    dict: метаданные
        """
        return self.vector_db.get_metadata()
