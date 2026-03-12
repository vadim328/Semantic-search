# service/updater.py
import asyncio
from service.di import Container
from text_processing.text_preparation import transforms_bert
from qdrant_client.models import PointStruct
from datetime import datetime, timedelta
from collections import defaultdict
import logging

log = logging.getLogger(__name__)


class DataUpdater:
    def __init__(self):

        self.container = Container()

        # Берем последнюю запись среди всех коллекций
        self.date_from = max(
            vector_db.date_last_record
            for vector_db in self.container.vector_dbs.values()
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
                    {"from_date": datetime.fromtimestamp(start_interval),
                     "to_date": datetime.fromtimestamp(end_interval)}
                )
                break
            intervals.append(
                {"from_date": datetime.fromtimestamp(start_interval),
                 "to_date": datetime.fromtimestamp(batch_end)}
            )
            # Переводим начало в конец предыдущего интервала
            start_interval = batch_end

        log.info(f"Intervals fetching, only {len(intervals)} intervals")
        return intervals

    async def update(self):
        """
            Получение новых данных и сохранение их в векторную БД
        """

        log.info("Request for data ...")

        date_intervals = self._build_intervals(self.date_from)
        for date_interval in date_intervals:

            log.info(f"Work with interval: {date_interval['from_date'].strftime('%Y-%m-%d, %H:%M')} - "
                     f"{date_interval['to_date'].strftime('%Y-%m-%d, %H:%M')} ...")
            await self.container.relational_db.fetch_data(date_interval)

            # TODO  Можно вынести в отдельную функцию формирование точек
            product_points = defaultdict(list)
            for row in self.container.relational_db.get_data():
                log.debug(f"Problem {row['problem']}\nComments: {row['comments']}")

                # Формируем точки для сохранения в каждой коллекции
                product_points[row["product"]].append(
                    PointStruct(
                        id=int(row["number"]),
                        vector=self.container.request_embedding.fetch_embedding(
                            row['problem'],
                            row['comments']
                        ),
                        payload={
                            "text": row["problem"],
                            "client": row["client"],
                            "product": row["product"],
                            "registry_date": row["registry_date"].timestamp()
                        }
                    )
                )

            for product, points_for_product in product_points.items():

                # получаем объект VectorDatabaseTouch для этой коллекции
                vector_db = self.container.vector_dbs[product]
                vector_db.save_embeddings(points_for_product)
                log.info(f"Saved {len(points_for_product)} points to collection '{product}'")

            log.info("Interval work completed")

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
