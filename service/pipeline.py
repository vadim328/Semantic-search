import numpy as np
import asyncio
from datetime import datetime, timedelta
from models.model import OnnxSentenseTransformer
from db.database import RelationalDatabaseTouch, VectorDatabaseTouch
from rank_bm25 import BM25Okapi
from text_processing.text_preparation import transforms_bm25, transforms_bert
import logging

log = logging.getLogger(__name__)


class SemanticSearchEngine:
    def __init__(self):
        self.model = OnnxSentenseTransformer(
            'models/onnx/optim/',
            'model_optimized.onnx'
        )

        self.relational_db = RelationalDatabaseTouch(
            "[CHANGE]"
        )

        # self.vector_db = VectorDatabaseTouch("http://localhost:6333")
        self.vector_db = VectorDatabaseTouch(":memory:")

    def calculation(self, data_calculation: dict):
        """
            Рассчет косинусного расстояния в совокупности
            с рассчетом значения BM25
            :input:
                dict: Данные для вычисления
            :output:
                list[tuple]: кортеж с двумя списками, score и № запроса
        """
        cosine_scores = np.array(data_calculation["cosine_scores"])  # преобразуем в numpy массив для дальнейших рассчетов
        bm25 = BM25Okapi(data_calculation["tokenized_querys"])

        # --- Считаем BM25 ---
        bm25_scores = bm25.get_scores(data_calculation["tokenized_query"])

        # 4. Нормализация и объединение
        bm25_norm = bm25_scores / (bm25_scores.max() + 1e-9)
        cosine_norm = cosine_scores / (cosine_scores.max() + 1e-9)
        hybrid_scores = data_calculation["alpha"] * bm25_norm + (1 - data_calculation["alpha"]) * cosine_norm

        # --- Ранжируем ---
        return sorted(zip(data_calculation["numbers"], hybrid_scores), key=lambda x: x[1], reverse=True)

    def search(self, query: str, limit=5, alpha=0.5, filters=None):
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
        hits = self.vector_db.fetch_embeddings(embedding, filters)

        cosine_scores, tokenized_querys = [], []
        numbers = []
        log.info(f"{len(hits.points)}")
        for hit in hits.points:
            log.info(type(hit.payload["registry_date"]))
            numbers.append(hit.id)
            cosine_scores.append(hit.score)
            tokens = transforms_bm25(text=hit.payload["text"])["text"].split()
            log.debug(f"Text tokens: {tokens}")
            tokenized_querys.append(tokens)

        calc = self.calculation(
            {
                "cosine_scores": cosine_scores,
                "tokenized_querys": tokenized_querys,
                "tokenized_query": tokenized_query,
                "alpha": alpha,
                "numbers": numbers,
            }
        )
        log.info(f'Result fetching')
        calc = calc[:limit]
        ranked = {}
        for i in calc:
            ranked[str(i[0])] = str(round(i[1] * 100)) + "%"

        return ranked

    async def update(self):
        """
            Получение новых данных и сохранение их в векторную БД
        """

        log.info("Request for data ...")
        date_last_record = self.vector_db.get_date_last_record()
        if date_last_record != datetime.now().date():
            first_fetch = False
            if date_last_record is None:
                first_fetch = True
                date_last_record = datetime.now().date()
            first_fetch = False
            await self.relational_db.fetch_data(first_fetch, date_last_record)
            rows = self.relational_db.get_data()

            for row in rows:
                log.debug(f"Text for preparation {row['problem']}")
                text_bert = transforms_bert(text=row["problem"])["text"]
                row["embedding"] = self.model.encode(text_bert)[0]
                row["registry_date"] = row["registry_date"].timestamp()

            self.vector_db.save_embeddings(rows)
        else:
            log.info("Update skip, there is already data for the current date")

    async def background_updater(self):
        """
            Фоновая задача для обновления данных
            Засыпает  до 3 часов ночи каждого дня
            По истечении таймера запсукает функцию на получение данных
        """
        try:
            while True:
                now = datetime.now()
                # Цель — 3:00 следующего дня
                target_time = (now + timedelta(days=1)).replace(hour=3, minute=0, second=0, microsecond=0)
                wait_seconds = (target_time - now).total_seconds()
                log.info(f'Waiting until {target_time} {int(wait_seconds)} sec.')
                await asyncio.sleep(wait_seconds)
                try:
                    await self.update()
                except Exception as e:
                    log.info(f'Error during update: {e}')
        except asyncio.CancelledError:
            log.info("Background updater was cancelled.")
            raise
