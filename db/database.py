from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from datetime import datetime, timedelta

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, PointStruct, Distance
from qdrant_client.http.models import Filter, FieldCondition, Range


# Загружаем SQL из файла
def load_query(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


class RelationalDatabaseTouch:
    def __init__(self, url):
        engine = create_engine(url)
        self.Session = sessionmaker(bind=engine)
        self.first_fetch_query = load_query("db/queries/fetch_all_requests.sql")
        self.next_fetch_query = load_query("db/queries/fetch_requests_last.sql")
        self.requests = {}

    def fetch_data(self, first_fetch: bool, last_fetch_time: str):
        """Получение данных из БД"""
        if first_fetch:
            query = text(self.first_fetch_query)
        else:
            query = text(self.next_fetch_query)
        params = {"last_fetch_time": last_fetch_time}

        try:
            session = self.Session()
            requests = session.execute(query, params)
            self.requests = [dict(row) for row in requests.mappings().all()]  # Преобразуем в словарь
            session.close()
            print("Данные получены")
        except Exception as e:
            print(f"Ошибка получения данных")

    def get_data(self):
        """Отдает запросы и очищает кэш"""
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

        # Проверяем, существует ли коллекция
        if not self.client.collection_exists(self.collection_name):
            self.init_db()
            self.points_count = 0
            self.date_last_record = None
            print(f"Коллекция '{self.collection_name}' не найдена, создана новая коллекция")
        else:
            self.points_count = self._get_existing_points_count()
            self.date_last_record = self._fetch_date_last_record()
            print(f"Коллекция '{self.collection_name}' найдена, записей: {self.points_count}")

    def init_db(self):
        # Создаём коллекцию
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=self.vector_size, distance=self.distance),
        )

    def save_embeddings(self, rows: dict):
        # Формируем записи для Qdrant
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
        print(f"✅ Эмбеддинги успешно сохранены в Qdrant. Дата последней записи {self.date_last_record}")

    def fetch_embeddings(self, embedding):
        # Получаем все эмбеддинги из Qdrant
        hits = self.client.query_points(
            collection_name=self.collection_name,
            query=embedding,
            limit=self.points_count # Находим все
        )
        print("✅ Эмбеддинги получены")
        return hits

    def _get_existing_points_count(self):
        # Получаем количество записей и сохраняем в переменную, для оптимизации
        info = self.client.get_collection(collection_name=self.collection_name)
        return info.result.points_count or 0 # актуальное количество точек

    def _fetch_date_last_record(self):
        """Возвращаем дату последней записи в коллекции"""
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
        last_date = last_point.payload["registry_date"].split("T")[0]

        return last_date

    def get_date_last_record(self):
        return self.date_last_record





