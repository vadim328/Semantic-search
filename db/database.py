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
        self.fetch_query = text(load_query("db/queries/fetch_requests.sql"))
        self.requests = {}

    async def fetch_data(self, from_date: datetime):
        """
            Получение данных из БД, сохраняет в переменную
            :input:
                bool: Определяет первичную загрузку или вторичную
                datetime: Дата последней записи в векторной БД
                    или дата последнего успешного сохранения
        """
        params = {"from_date": from_date}
        async with self.Session() as session:
            try:
                requests = await session.execute(self.fetch_query, params)
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
        self.qdrant_client = QdrantClient(url)
        self.collection_name = "support_tickets"  # Название коллекции
        self.vector_size = 312  # размер эмбеддинга
        self.distance = Distance.COSINE  # метрика
        self.points_count = 0
        self.metadata = {}
        self.date_last_record = datetime.strptime("2025-11-14", "%Y-%m-%d").timestamp()
        self.initialize()

    def init_db(self):
        """Создание новой коллекции"""

        log.info(f"create new collection, collection name - {self.collection_name}")
        self.qdrant_client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=self.vector_size, distance=self.distance),
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
            log.info(f"✅ Embedding successfully saved in vector db. Date for next update {self.date_last_record}")
        except Exception as e:
            log.info(f"Embedding unsuccessfully saved in vector db. Date for next update {self.date_last_record}")

    def _build_filter(self, filters):

        log.info("Add filters ...")
        conditions = []
        if filters.get("client"):
            conditions.append(
                models.FieldCondition(
                    key="client",
                    match=models.MatchValue(value=filters.get("client"))
                )
            )
        if filters.get("product"):
            conditions.append(
                models.FieldCondition(
                    key="product",
                    match=models.MatchValue(value=filters.get("product"))
                )
            )
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
        hits = self.qdrant_client.query_points(
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
        info = self.qdrant_client.get_collection(collection_name=self.collection_name)
        return info.result.points_count or 0  # актуальное количество точек

    def _update_metadata(self):
        """
            Обновление метаданных коллекции
        """
        offset = None
        clients = set()
        products = set()
        # TODO добавить фильтр, чтоб проходиться по последним записям, а не по всем

        # Получаем все записи коллекции по 1000, чтоб не тратить RAM
        while True:
            scroll_result = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                limit=1000,
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





