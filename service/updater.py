# service/updater.py
import asyncio
from service.di import model, relational_db, vector_db
from text_processing.text_preparation import transforms_bert
from datetime import datetime, timedelta
import logging

log = logging.getLogger(__name__)


class DataUpdater:
    def __init__(self):
        self.model = model
        self.relational_db = relational_db
        self.vector_db = vector_db

    async def run(self):
        # Первый запуск — сразу
        try:
            log.info("Initial update started")
            await self.update()
            log.info("Initial update finished")
        except Exception:
            log.exception("Initial update failed")

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

        from_date = self.vector_db.get_date_last_record()
        date_intervals = self._build_intervals(from_date)
        for date_interval in date_intervals:

            log.info(f"Work with interval: {date_interval['from_date'].strftime('%Y-%m-%d, %H:%M')} - "
                     f"{date_interval['to_date'].strftime('%Y-%m-%d, %H:%M')} ...")
            await self.relational_db.fetch_data(date_interval)
            rows = self.relational_db.get_data()

            for row in rows:
                log.debug(f"Text for preparation {row['problem']}")
                text_bert = transforms_bert(text=row["problem"])["text"]
                row["embedding"] = self.model.encode(text_bert)[0]
                row["registry_date"] = row["registry_date"].timestamp()

            self.vector_db.save_embeddings(rows)

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
