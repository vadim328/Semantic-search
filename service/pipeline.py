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

    def search(self, query: str, limit=5, alpha=0.5):
        """
            Поиск информации по векторной БД и
            рассчет косинусного расстояния в совокупности
            с рассчетом значения BM25

            :input:
                str: Искомый текст
                int: Количество лучших совпадений
                float: Определяет баланс между значением косинусного расстояния и
                        алгоритма BM25

            :output:
                tuple(list, list): кортеж с двумя списками, score и № запроса
        """

        log.info(f'Request params: query - {query},\nlimit - {limit},\nalpha - {alpha}')

        # Для поиска по ключевым словам лучше увеличить альфу
        tokenized_query = transforms_bm25(text=query)["text"].split()
        log.debug(f'transforms text for bm25: {tokenized_query}')

        query_bert = transforms_bert(text=query)["text"]
        log.debug(f'transforms text for NN: {query_bert}')

        embedding = self.model.encode(query_bert)[0]
        hits = self.vector_db.fetch_embeddings(embedding)

        cosine_scores, tokenized_querys = [], []
        numbers = []
        for hit in hits.points:
            numbers.append(hit.id)
            cosine_scores.append(hit.score)
            tokens = transforms_bm25(text=hit.payload["text"])["text"].split()
            tokenized_querys.append(tokens)

        cosine_scores = np.array(cosine_scores)  # преобразуем в numpy массив для дальнейших рассчетов
        bm25 = BM25Okapi(tokenized_querys)

        # --- Считаем BM25 ---
        bm25_scores = bm25.get_scores(tokenized_query)

        # 4. Нормализация и объединение
        bm25_norm = bm25_scores / (bm25_scores.max() + 1e-9)
        cosine_norm = cosine_scores / (cosine_scores.max() + 1e-9)
        hybrid_scores = alpha * bm25_norm + (1 - alpha) * cosine_norm
        """
        TODO
        вернуть N-ое количество
        """
        # --- Ранжируем ---
        ranked = sorted(zip(numbers, hybrid_scores), key=lambda x: x[1], reverse=True)[:limit]
        log.info(f'Result fetching')

        return ranked

    async def update(self, first_fetch=False):
        """
            Получение новых данных и сохранение их в векторную БД
            :input:
                bool: Определяет первичное или вторичное получение данных
        """

        log.info("Request for data ...")
        date_last_record = self.vector_db.get_date_last_record()
        if date_last_record is None:
            first_fetch = True
            date_last_record = str(datetime.now().date())
        await self.relational_db.fetch_data(first_fetch, date_last_record)
        rows = self.relational_db.get_data()

        for row in rows:
            text_bert = transforms_bert(text=row["problem"])["text"]
            row["embedding"] = self.model.encode(text_bert)[0]

        await self.vector_db.save_embeddings(rows)

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
