# service/updater.py
import asyncio
from typing import List, Dict
from search_service.text_processing.text_preparation import \
    transforms_embed, \
    transforms_llm, \
    transforms_comments
from qdrant_client.models import PointStruct
from datetime import datetime, timedelta
from collections import defaultdict
from search_service.config import Config
import logging

log = logging.getLogger(__name__)
cfg = Config().data["service"]["updater"]


class DataUpdater:
    """Периодическое добавление новых данных"""
    def __init__(self, container):

        self.container = container
        self.max_concurrent = cfg["max_concurrent"]
        self.time_window = cfg["time_window"] * 86_400

        # Берем последнюю запись среди всех коллекций
        self.date_from = datetime.fromtimestamp(
            max(
                collection.metadata()["date_last_record"]
                for collection in self.container.vector_db.collections().values()
            )
        )

    async def run(self):
        # Первый запуск — сразу
        try:
            log.info("Initial update started")
            await self.update()
            log.info("Initial update finished")
        except Exception:
            log.exception("Initial update failed")
            raise

    def _build_intervals(self, start_interval: datetime) -> List:
        """
        Вычисление временных интервалов для их использования в запросах на получение данных
            Конец последнего интервала - текущая дата и время
        Args:
            start_interval (datetime): Начальная дата
        Returns:
            list: Список кортежей с начальной и конечной датами

        """
        log.info("Calculation data intervals")
        intervals = []
        start_interval = start_interval.timestamp()
        end_interval = datetime.now().timestamp()
        while True:
            batch_end = start_interval + self.time_window
            if batch_end >= end_interval:
                intervals.append(
                    {
                        "from_date": datetime.fromtimestamp(start_interval),
                        "to_date": datetime.fromtimestamp(end_interval)
                    }
                )
                break
            intervals.append(
                {
                    "from_date": datetime.fromtimestamp(start_interval),
                    "to_date": datetime.fromtimestamp(batch_end)
                }
            )
            # Переводим начало в конец предыдущего интервала
            start_interval = batch_end

        log.info(f"Intervals fetching, only {len(intervals)} intervals")
        return intervals

    async def _save_points(self, product_points: dict):
        """
        Сохранение точек в коллекции
            Args:
                product_points (dict): Cловарь с данными в формате PointStruct по продуктам,
        """
        for product, points in product_points.items():
            vector_db_collection = self.container.vector_db.collection(product)

            await vector_db_collection.save_embeddings(points)

    async def _get_embedding(self, row: dict) -> Dict:
        """
        Получение эмбеддингов запроса по трем составляющим:
            1) Оригинальный текст
            2) Суммаризированный текст по описанию и комментариям
            3) Комментарии
        Args:
            row (dict): запись с полями
        Returns:
            dict - эмбеддинги запроса
        """
        vectors = {}

        try:
            vectors["original"] = await self.container.model_client.embed(
                texts=transforms_embed(text=row["problem"])["text"],
                prefix="passage",
            )
        except Exception as e:
            log.exception(f"Embedding failed for original text: {e}")

        if row["comments"]:
            comments = transforms_comments(text=row["comments"])["text"]
            comments = transforms_embed(text=comments)["text"],
            try:
                vectors["comments"] = await self.container.model_client.embed(
                    texts=comments,
                    prefix="passage",
                )
            except Exception as e:
                log.exception(f"Embedding failed for comments: {e}")
        else:
            comments = None

        try:
            problem_summary = await self.container.summarization_orchestrator.summarize(
                problem=transforms_llm(text=row["problem"])["text"],
                comments=comments,
                max_concurrent=self.max_concurrent,
            )
            vectors["summary"] = await self.container.model_client.embed(
                texts=problem_summary,
                prefix="passage",
            )
        except Exception as e:
            log.exception(f"Embedding failed for comments: {e}")

        if not vectors:
            raise ValueError("Failed to get any embedding")

        return vectors

    async def _build_points(self, rows: List[dict]) -> dict:
        """
        Асинхронное преобразование записей в PointStruct с параллельным получением эмбеддингов.
        Args:
            rows (List[dict]): список записей из БД
        Returns:
            dict: ключ — product, значение — список PointStruct
        """
        product_points = defaultdict(list)

        count_missing_rows = 0
        for row in rows:
            try:
                log.info(f"Summarize and fetch embedding for request - {row['number']}")
                vectors = await self._get_embedding(row)

                product_points[row["product"]].append(
                    PointStruct(
                        id=int(row["number"]),
                        vector=vectors,
                        payload={
                            "text": row["problem"],
                            "comments": row["comments"],
                            "client": row["client"],
                            "registry_date": row["registry_date"].timestamp(),
                            "date_end": row["date_end"].timestamp(),
                        }
                    )
                )
            except Exception as e:
                log.error(f"Error processing string {row.get('number')}: {e}")
                count_missing_rows += 1
                if count_missing_rows >= 10:
                    raise ValueError("A lot of missing lines")
                continue

        return product_points

    async def _process_interval(self, interval: dict):

        """
        Метод для работы с интервалами, обрабатывает строки батчами,
            с сохранением в БД
        Args:
            interval (dict) - содержит дату начала и конца выборки
        """

        log.info(f"Work with interval: {interval['from_date'].strftime('%Y-%m-%d %H:%M')} - "
                 f"{interval['to_date'].strftime('%Y-%m-%d %H:%M')}")

        await self.container.relational_db.fetch_data(interval)

        rows = self.container.relational_db.get_data()

        product_points = await self._build_points(rows)

        await self._save_points(product_points)

        log.info("Interval work completed")

    async def update(self):
        """Получение, обработка и сохранение данных в векторной БД"""

        date_intervals = self._build_intervals(self.date_from)

        for interval in date_intervals:
            await self._process_interval(interval)

    async def background_updater(self):
        """
        Фоновая задача для обновления данных. Засыпает  до 3 часов ночи каждого дня.
            По истечении таймера запсукает функцию на получение данных
        """
        try:
            while True:
                now = datetime.now()
                # Цель — 3:00 следующего дня
                target_time = (now + timedelta(days=1)).replace(hour=3, minute=0, second=0, microsecond=0)
                wait_seconds = (target_time - now).total_seconds()
                log.info(f"Next update at {target_time}, sleeping {int(wait_seconds)} sec")
                await asyncio.sleep(wait_seconds)
                try:
                    await self.update()
                except Exception as e:
                    log.info(f"Error during update: {e}")
        except asyncio.CancelledError:
            log.info("Background updater was cancelled.")
            raise
