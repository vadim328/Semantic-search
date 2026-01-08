import numpy as np
import asyncio
from datetime import datetime, timedelta
from models.model import OnnxSentenseTransformer
from db.database import RelationalDatabaseTouch, VectorDatabaseTouch
from rank_bm25 import BM25Okapi
from text_processing.text_preparation import transforms_bm25, transforms_bert
from service.utils import timestamp_to_date
import logging
from config import Config

log = logging.getLogger(__name__)
cfg = Config().data


class SemanticSearchEngine:
    def __init__(self):
        self.model = OnnxSentenseTransformer(
            cfg["model"]["path"],
            cfg["model"]["model_name"]
        )

        self.relational_db = RelationalDatabaseTouch()
        self.vector_db = VectorDatabaseTouch()

    async def generate_result(self, calc_result: list):
        """
            Формирует итоговый результат поиска
            :input:
                list: Результаты поиска
            :output:
                dict: Результат поиска с дополнительной информацией
        """
        numbers = [i[0] for i in calc_result]
        additional_data = await self.relational_db.fetch_additional_data({"numbers": numbers})

        result = []
        for cr, ad in zip(calc_result, additional_data):
            if cr[1] < cfg["service"]["threshold"]:
                continue
            result.append({
                "id": str(cr[0]),
                "score": str(round(cr[1] * 100)) + "%",
                "responsible": ad["fio"],
                "priority": ad["admission_prority"],
                "registry_date": str(timestamp_to_date(cr[2])),
                "url": "https://support.naumen.ru/sd/operator/#uuid:%s" % ad["servicecall"]
            })

        return result

    @staticmethod
    def calculation(data_calculation: dict):
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
        return sorted(zip(data_calculation["numbers"], hybrid_scores, data_calculation["reg_dates"]),
                      key=lambda x: x[1], reverse=True)

    async def search(self, query: str, limit=5, alpha=0.5, exact=True, filters=None):
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

            calculation_data = {
                "numbers": [],
                "cosine_scores": [],
                "tokenized_querys": [],
                "tokenized_query": tokenized_query,
                "reg_dates": [],
                "alpha": alpha,
            }

            # Формируем данные для вычисления
            for hit in hits.points:
                tokens = transforms_bm25(text=hit.payload["text"])["text"].split()
                log.debug(f"Text tokens: {tokens}")

                calculation_data["numbers"].append(hit.id)
                calculation_data["cosine_scores"].append(hit.score)
                calculation_data["tokenized_querys"].append(tokens)
                calculation_data["reg_dates"].append(hit.payload["registry_date"])

            calc = self.calculation(calculation_data)
            log.info(f'Result fetching')

            return await self.generate_result(calc[:limit])
        except ZeroDivisionError:
            return {"result": "data not found"}
        except Exception as e:
            log.error(f"Error: {e}")

    @staticmethod
    def _extract_date_interval(start_interval: datetime):
        """
            Вычисление временных интервалов для их использования в запросах
                Конец интервала - текущая дата и время
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

        log.info(f"Intervals fetching: {intervals}")
        return intervals

    async def update(self):
        """
            Получение новых данных и сохранение их в векторную БД
        """

        log.info("Request for data ...")

        from_date = self.vector_db.get_date_last_record()
        date_intervals = self._extract_date_interval(from_date)
        for date_interval in date_intervals:

            log.info(f"Work at intervals of {date_interval} ...")
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
                log.info(f'Waiting until {target_time} {int(wait_seconds)} sec.')
                await asyncio.sleep(wait_seconds)
                try:
                    await self.update()
                except Exception as e:
                    log.info(f'Error during update: {e}')
        except asyncio.CancelledError:
            log.info("Background updater was cancelled.")
            raise

    def get_metadata(self):
        """
            Формирует и передает метаданные
                :output:
                    dict: метаданные
        """
        return self.vector_db.get_metadata()
