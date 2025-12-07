from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from datetime import datetime

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, PointStruct, Distance, \
    HnswConfigDiff, SearchParams
from qdrant_client.http import models
import logging
from config import Config

log = logging.getLogger(__name__)
cfg = Config().data


# Загружаем SQL из файла
def load_query(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


class RelationalDatabaseTouch:
    def __init__(self):
        engine = create_async_engine(cfg["database"]["relational_db"]["url"])
        self.Session = async_sessionmaker(bind=engine)
        self.additional_data_query = text(load_query("db/queries/additional_data.sql"))
        self.requests = {}

    async def make_request(self, query, params=None):
        """
            Получение данных из БД, сохраняет в переменную
            :input:
                bool: Определяет первичную загрузку или вторичную
                datetime: Дата последней записи в векторной БД
                    или дата последнего успешного сохранения
        """
        async with self.Session() as session:
            try:
                response = await session.execute(query, params)
                response = [dict(row) for row in response.mappings().all()]
                return response
            except Exception as e:
                log.error(f"Error retrieving data from relational db: {e}")

    async def fetch_data(self, params: dict):
        """
            Формирование запроса на получение данных
            :input:
                dict: Параметры запроса
        """
        query = text(load_query("db/queries/fetch_requests.sql"))
        self.requests = await self.make_request(query, params)
        log.info(f"Data received from relational db, count rows - {len(self.requests)}")

    async def fetch_additional_data(self, params: dict):
        """
            Формирование запроса на получение дополнительных данных
            :input:
                dict: Параметры запроса
            :output:
                dict: Результат запроса
        """
        additional_data = await self.make_request(self.additional_data_query, params)
        log.info(f"Additional data received from relational db: {additional_data}")
        return additional_data

    def get_data(self):
        """
            Отдает запросы и очищает кэш
            :output:
                dict: Запросы полученные из БД
        """
        requests = self.requests
        self.requests = {}
        return requests


class VectorDatabaseTouch:
    def __init__(self):
        self.qdrant_config = cfg["database"]["vector_db"]
        # Подключаемся к Qdrant
        self.qdrant_client = QdrantClient(self.qdrant_config["main"]["url"])
        self.collection_name = self.qdrant_config["main"]["collection_name"]
        self.vector_size = 312  # размер эмбеддинга
        self.distance = Distance.COSINE  # метрика
        self.points_count = 0
        self.metadata = {}
        self.date_last_record = datetime.strptime(
            self.qdrant_config["main"]["date_from"],
            "%Y-%m-%d").timestamp()
        self.initialize()

    def init_db(self):
        """Создание новой коллекции"""

        log.info(f"create new collection, collection name - {self.collection_name}")
        self.qdrant_client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=self.vector_size, distance=self.distance),
            hnsw_config=HnswConfigDiff(
                m=self.qdrant_config["indexing"]["m_value"],
                ef_construct=self.qdrant_config["indexing"]["ef_construct"],
                full_scan_threshold=self.qdrant_config["indexing"]["full_scan_threshold"],
                max_indexing_threads=self.qdrant_config["indexing"]["max_indexing_threads"],
                on_disk=self.qdrant_config["indexing"]["on_disk"],
            )
        )

    def initialize(self):
        # Проверяем, существует ли коллекция
        if not self.qdrant_client.collection_exists(self.collection_name):
            log.info(f"Collection '{self.collection_name}' not found")
            self.init_db()
        else:
            self.points_count = self._fetch_existing_points_count()
            self._update_metadata()
            log.info(f"Collection '{self.collection_name}' found, "
                     f"count points - {self.points_count}, "
                     f"date last point - {self.get_date_last_record()}")

    def save_embeddings(self, rows: dict):
        """
            Формируем записи для Qdrant и сохраняем
            :input:
                dict: Словарь с данными из реляционной БД
        """
        log.info("Save data in vector db ...")
        try:
            points = [
                PointStruct(
                    id=int(row["number"]),
                    vector=row["embedding"],
                    payload={
                        "text": row["problem"],
                        "client": row["client"],
                        "product": row["product"],
                        "registry_date": row["registry_date"]
                    }
                )
                for row in rows
            ]

            # Сохраняем в Qdrant
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            self.points_count = self.points_count + len(rows)  # Актуализируем количество точек
            self._update_metadata()  # Актуализируем метаданные
            log.info(f"✅ Embedding successfully saved in vector db. Date for next update {self.get_date_last_record()}")
        except Exception as e:
            log.info(f"Embedding unsuccessfully saved in vector db. Error: {e}")
            log.info(f"Date for next update {self.get_date_last_record()}")

    @staticmethod
    def _build_filter(filters: dict):
        """
            Создаёт объект Filter для Qdrant из словаря фильтров.
            Автоматически поддерживает:
                - точное совпадение по любому ключу
                - диапазоны дат: date_from, date_to
            :input:
                dict: массив фильтров
        """
        log.info("Building filters...")
        conditions = []

        for key, value in filters.items():
            if value is None:
                continue

            # Специальная обработка диапазонов дат
            if key == "date_from":
                if not isinstance(value, float):
                    value = datetime.strptime(value, "%Y-%m-%d").timestamp()
                conditions.append(
                    models.FieldCondition(
                        key="registry_date",
                        range=models.Range(gte=value)
                    )
                )
            elif key == "date_to":
                if not isinstance(value, float):
                    value = datetime.strptime(value, "%Y-%m-%d").timestamp()
                conditions.append(
                    models.FieldCondition(
                        key="registry_date",
                        range=models.Range(lte=value)
                    )
                )
            else:
                # Любые другие поля считаем точным совпадением
                conditions.append(
                    models.FieldCondition(
                        key=key,
                        match=models.MatchValue(value=value)
                    )
                )

        log.info(f"Filters added: {len(conditions)} conditions")
        return models.Filter(must=conditions) if conditions else None

    def fetch_embeddings(self, embedding, exact, filters):
        """
            Получаем все эмбеддинги из Qdrant
            :input:
                np.array: Исходный эмбеддинг
                bool: Требуется ли использовать поиск по графу
                dict: Фильтры для поиска
            return:
                custom qdrant class: результат поиска по векторной БД
        """
        log.info("Fetch embeddings from vector db ...")

        # Добавляем фильтр
        query_filter = self._build_filter(filters)
        hits = self.qdrant_client.query_points(
            collection_name=self.collection_name,
            query=embedding,
            limit=self.points_count if exact else 500,  # Находим все точки, если не быстрый поиск
            query_filter=query_filter,
            search_params=SearchParams(
                exact=exact,  # False - используем индексы
                hnsw_ef=512   # Количество кандидатов для рассмотерния
            ),
        )
        log.info("✅ Embedding received successfully")
        return hits

    def _fetch_existing_points_count(self):
        """
            Получение количества точек в коллекции
            :output:
                int: Количество точек в коллекции
        """
        info = self.qdrant_client.get_collection(collection_name=self.collection_name)
        return info.points_count or 0  # актуальное количество точек

    def _update_metadata(self):
        """
            Обновление метаданных коллекции.
            Для ускорения добавлен фильтр даты последней записи,
            чтоб не проходить по всем точкам каждый раз.
        """
        offset = None
        clients = self.metadata.get("clients", set())
        products = self.metadata.get("products", set())
        scroll_filter = self._build_filter({
            "date_from": self.date_last_record
        })
        # Получаем все записи коллекции по 1000, чтоб не нагружать RAM
        while True:
            scroll_result = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                limit=1000,
                scroll_filter=scroll_filter,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            points = scroll_result[0]

            # Проходимся по каждой записи
            for p in points:
                clients.add(p.payload["client"])
                products.add(p.payload["product"])

                # Ищем последнюю дату
                if p.payload["registry_date"] > self.date_last_record:
                    self.date_last_record = p.payload["registry_date"]

            offset = scroll_result[1]
            if offset is None:
                break

        self.metadata = {
            "clients": clients,
            "products": products,
        }

    def get_date_last_record(self):
        """
            Получение даты последней записи из переменной
            :output:
                datetime: дата последней записи в переменной
        """
        return datetime.fromtimestamp(self.date_last_record)

    def get_metadata(self):
        """
            Получение метаданных векторной БД
            :output:
                dict: метаданные
        """
        return self.metadata
