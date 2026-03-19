# service/updater.py
import asyncio
from typing import List
from search_service.service.di import container
from search_service.text_processing.text_preparation import transforms_bert
from qdrant_client.models import PointStruct
from datetime import datetime, timedelta
from collections import defaultdict
import logging

log = logging.getLogger(__name__)


class DataUpdater:
    def __init__(self):

        self.container = container

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
            vector_db = self.container.vector_db.collection(product)

            vector_db.save_embeddings(points)

            log.info(f"Saved {len(points)} points to product '{product}'")

    def _get_embedding(self, row: dict):
        """
            Метод для получения эмбеддинга запроса
                input:
                    row - запись с полями
                output:
                    vector - эмбеддинг запроса
        """

        if row["product"] == "Naumen":
            problem_summary = self.container.model_client.make_summarize(
                problem=row["problem"],
                comments=row["comments"]
            )
            return self.container.model_client.embed(problem_summary)

        text = transforms_bert(text=row["problem"])["text"]
        return self.container.model_client.embed(text)

    def _build_points(self, rows: List[dict]) -> dict:
        """
            Преобразование полученных строк из реляционной БД
            в нужный формат для сохранения в векторную БД
                input:
                    rows (List) - записи, полученные из реляционной БД
                output:
                    dict - словарь, где ключями являются названия продуктов,
                        а значениями, список объектов PointStruct
        """
        product_points = defaultdict(list)

        for row in rows:
            embedding = self._get_embedding(row)

            product_points[row["product"]].append(
                PointStruct(
                    id=int(row["number"]),
                    vector=embedding,
                    payload={
                        "text": row["problem"],
                        "client": row["client"],
                        "date_end": row["date_end"].timestamp()
                    }
                )
            )

        return product_points

    async def _process_interval(self, interval: dict):

        """
            Метод для работы с интервалами
                input:
                    dict - содержит дату начала и конца выборки
        """

        log.info(f"Work with interval: {interval['from_date'].strftime('%Y-%m-%d, %H:%M')} - "
                 f"{interval['to_date'].strftime('%Y-%m-%d, %H:%M')} ...")

        await self.container.relational_db.fetch_data(interval)

        rows = self.container.relational_db.get_data()

        product_points = self._build_points(rows)

        await self._save_points(product_points)

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
