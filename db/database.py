from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from datetime import datetime

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, PointStruct, Distance
from qdrant_client.http import models
import logging

log = logging.getLogger(__name__)


# Загружаем SQL из файла
def load_query(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


class RelationalDatabaseTouch:
    def __init__(self, url):
        engine = create_async_engine(url)
        self.Session = async_sessionmaker(bind=engine)
        self.first_fetch_query = load_query("db/queries/fetch_all_requests.sql")
        self.next_fetch_query = load_query("db/queries/fetch_requests_last.sql")
        self.requests = {}

    async def fetch_data(self, first_fetch: bool, last_fetch_time: datetime):
        """
            Получение данных из БД, сохраняет в переменную
            :input:
                bool: Определяет первичную загрузку или вторичную
                datetime: Дата последней записи в векторной БД
                    или дата последнего успешного сохранения
        """
        if first_fetch:
            query = text(self.first_fetch_query)
        else:
            query = text(self.next_fetch_query)
        params = {"last_fetch_time": last_fetch_time}
        date_str = '2025-11-10' # test
        date_obj = datetime.strptime(date_str, "%Y-%m-%d").date() # test
        params = {"last_fetch_time": date_obj} # test
        async with self.Session() as session:
            try:
                requests = await session.execute(query, params)
                self.requests = [dict(row) for row in requests.mappings().all()]
                log.info(f"Data received from relational db, count rows - {len(self.requests)}")
            except Exception as e:
                log.error(f"Error retrieving data from relational db: {e}")

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
    def __init__(self, url):
        # Подключаемся к Qdrant
        self.client = QdrantClient(url)
        self.collection_name = "support_tickets"  # Название коллекции
        self.vector_size = 312  # размер эмбеддинга
        self.distance = Distance.COSINE  # метрика
        self.points_count = 0
        self.date_last_record = None
        self.initialize()

    def init_db(self):
        """Создание новой коллекции"""

        log.info(f"create new collection, collection name - {self.collection_name}")
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=self.vector_size, distance=self.distance),
        )

    def initialize(self):
        # Проверяем, существует ли коллекция
        if not self.client.collection_exists(self.collection_name):
            log.info(f"Collection '{self.collection_name}' not found")
            self.init_db()
            self.points_count = 0
            self.date_last_record = None
        else:
            self.points_count = self._fetch_existing_points_count()
            self.date_last_record = self._fetch_date_last_record()
            log.info(f"Collection '{self.collection_name}' found, "
                     f"count points - {self.points_count}, "
                     f"date last point - {self.date_last_record}")

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
                        "registry_date": row["registry_date"]
                    }
                )
                for row in rows
            ]

            # Сохраняем в Qdrant
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            self.points_count = self.points_count + len(rows)  # Актуализируем количество точек
            self.date_last_record = str(datetime.now().date())  # Актуализируем дату последнего сохранения
            log.info(f"✅ Embedding successfully saved in vector db. Date last points {self.date_last_record}")
        except Exception as e:
            log.info(f"Embedding unsuccessfully saved in vector db. Date last points {self.date_last_record}")

    def _build_filter(self, filters):

        log.info("Add filters ...")
        conditions = []
        if filters.get("date_from"):
            date_from = datetime.strptime(filters.get("date_from"), "%Y-%m-%d").timestamp()
            conditions.append(
                models.FieldCondition(
                    key="registry_date",
                    range=models.Range(gte=date_from)
                )
            )
        if filters.get("date_to"):
            date_to = datetime.strptime(filters.get("date_to"), "%Y-%m-%d").timestamp()
            conditions.append(
                models.FieldCondition(
                    key="registry_date",
                    range=models.Range(lte=date_to)
                )
            )
        log.info("Filters added")
        return models.Filter(must=conditions) if conditions else None

    def fetch_embeddings(self, embedding, filters):
        """
            Получаем все эмбеддинги из Qdrant
            :input:
                np.array: Исходный эмбеддинг
        """
        log.info("Fetch embeddings from vector db ...")
        # Добавляем фильтр
        query_filter = self._build_filter(filters)
        hits = self.client.query_points(
            collection_name=self.collection_name,
            query=embedding,
            limit=self.points_count,  # Находим все точки
            query_filter=query_filter
        )
        log.info("✅ Embedding received successfully")
        return hits

    def _fetch_existing_points_count(self):
        """
            Получение количества точек в коллекции
            :output:
                int: Количество точек в коллекции
        """
        info = self.client.get_collection(collection_name=self.collection_name)
        return info.result.points_count or 0  # актуальное количество точек

    def _fetch_date_last_record(self):
        """
            Получение даты последней точки в коллекции
            :output:
                str: Дата в формате YY-mm-dd
        """
        # Получаем все записи коллекции
        scroll_result = self.client.scroll(
            collection_name=self.collection_name,
            limit=self.points_count,
            with_payload=True,
            with_vectors=False,
        )

        # Сортируем по registry_date
        last_point = max(
            scroll_result[0],
            key=lambda p: p.payload["registry_date"]
        )

        # Извлекаем только дату
        last_date = last_point.payload["registry_date"]

        return datetime.fromtimestamp(last_date).date()

    def get_date_last_record(self):
        """
            Получение даты последней записи из переменной
            :output:
                datetime: дата последней записи в переменной
        """
        return self.date_last_record





