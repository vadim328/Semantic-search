# service/updater.py
import asyncio
from typing import List
from search_service.text_processing.text_preparation import transforms_bert, transforms_nn, transforms_comments
from qdrant_client.models import PointStruct
from datetime import datetime, timedelta
from collections import defaultdict
from search_service.config import Config
import logging

log = logging.getLogger(__name__)
cfg = Config().data["service"]["updater"]


class DataUpdater:
    def __init__(self, container):

        self.container = container
        self.max_concurrent = cfg["max_concurrent"]

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

        # Дальше — по расписанию
        await self.background_updater()

    @staticmethod
    def _build_intervals(start_interval: datetime):
        """
            Вычисление временных интервалов для их использования в запросах на получение данных
                Конец последнего интервала - текущая дата и время
            :input:
                datetime: Начальная дата
            :return:
                list: Список кортежей с начальной и конечной датами

        """
        log.info("Calculation data intervals ...")
        intervals = []
        start_interval = start_interval.timestamp()
        end_interval = datetime.now().timestamp()
        while True:
            batch_end = start_interval + 2592000  # Берем месяц
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
            Поочередно для кадого продукта сохраняем эмбеддинги
            в их коллекции
                input: product_points - словарь с данными в формате PointStruct по продуктам,
        """
        for product, points in product_points.items():
            vector_db_collection = self.container.vector_db.collection(product)

            await vector_db_collection.save_embeddings(points)

    async def _get_embedding(self, row: dict):
        """
            Метод для получения эмбеддинга запроса по трем составляющим:
                1) Оригинальный текст
                2) Суммаризированный текст по описанию и комментариям
                3) Комментарии
                input:
                    row - запись с полями
                output:
                    vectors - эмбеддинги запроса
        """
        vectors = {}

        # Очищенные комментарии сохраняем чтоб не чистить 2 раза
        comments = transforms_comments(text=row["comments"])["text"]

        vectors["original"] = await self.container.model_client.embed(
            transforms_bert(text=row["problem"])["text"]
        )

        problem_summary = await self.container.summarization_orchestrator.summarize(
            problem=transforms_nn(text=row["problem"])["text"],
            comments=comments,
            max_concurrent=self.max_concurrent,
        )
        vectors["summary"] = await self.container.model_client.embed(problem_summary)

        vectors["comments"] = await self.container.model_client.embed(comments)

        return vectors

    async def _build_points(self, rows: List[dict]) -> dict:
        """
            Асинхронное преобразование записей в PointStruct с параллельным получением эмбеддингов.

            Args:
                rows: список записей из БД

            Returns:
                dict: ключ — product, значение — список PointStruct
        """
        product_points = defaultdict(list)

        for row in rows:
            log.info(f"Summarize and fetch embedding for request - {row['number']}")
            vectors = await self._get_embedding(row)

            product_points[row["product"]].append(
                PointStruct(
                    id=int(row["number"]),
                    vector=vectors,
                    payload={
                        "text": row["problem"],
                        "client": row["client"],
                        "registry_date": row["registry_date"].timestamp(),
                        "date_end": row["date_end"].timestamp(),
                    }
                )
            )

        return product_points

    @classmethod
    def chunked(cls, iterable, size):
        for i in range(0, len(iterable), size):
            yield iterable[i:i + size]

    async def _process_interval(self, interval: dict):

        """
            Метод для работы с интервалами, обрабатывает строки батчами,
                с сохранением в БД
                input:
                    dict - содержит дату начала и конца выборки
        """

        log.info(f"Work with interval: {interval['from_date'].strftime('%Y-%m-%d, %H:%M')} - "
                 f"{interval['to_date'].strftime('%Y-%m-%d, %H:%M')} ...")

        await self.container.relational_db.fetch_data(interval)

        rows = self.container.relational_db.get_data()

        batch_size = 5
        max_retries = 3

        log.info(f"Batches for processing - {int(len(rows)/batch_size + 1)}")

        for i, batch in enumerate(self.chunked(rows, batch_size), start=1):

            # три раза пытаемся обработать батч, иначе пропускаем его
            for attempt in range(1, max_retries + 1):
                log.info(f"Processing batch {i} ({len(batch)} rows)")
                try:
                    product_points = await self._build_points(batch)
                    await self._save_points(product_points)

                    log.info(f"Batch {i} completed on attempt {attempt}")
                    break

                except Exception:
                    log.exception(f"Batch {i} failed on attempt {attempt}")

                    if attempt == max_retries:
                        log.error(f"Batch {i} could not processed, skipping")
                    else:
                        await asyncio.sleep(10 ** attempt)

        log.info("Interval work completed")

    async def update(self):
        """
            Получение, обработка и сохранение данных в векторной БД
        """
        log.info("Request for data ...")

        date_intervals = self._build_intervals(self.date_from)

        for interval in date_intervals:
            await self._process_interval(interval)

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
                log.info(f"Next update at {target_time}, sleeping {int(wait_seconds)} sec")
                await asyncio.sleep(wait_seconds)
                try:
                    await self.update()
                except Exception as e:
                    log.info(f"Error during update: {e}")
        except asyncio.CancelledError:
            log.info("Background updater was cancelled.")
            raise
