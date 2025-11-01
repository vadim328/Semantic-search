from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from datetime import datetime, timedelta

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, PointStruct, Distance


# Загружаем SQL из файла
def load_query(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


class RelationalDatabaseTouch:
    def __init__(self, url):
        engine = create_engine(url)
        self.Session = sessionmaker(bind=engine)
        self.first_fetch_query = load_query('queries/fetch_all_requests.sql')
        self.last_day_query = load_query('queries/fetch_requests_last.sql')
        self.last_fetch_time = str(datetime.now().date())
        self.requests = None

    def fetch_requests(self, first_fetch=False):
        """Получение данных из БД"""
        if first_fetch:
            query = text(self.first_fetch_query)
        else:
            query = text(self.last_day_query)
        params = {'last_fetch_time': self.last_fetch_time}

        try:
            session = self.Session()
            self.requests = session.execute(query, params)
            session.close()
            print("Данные получены")
            self.last_fetch_time = str(datetime.now().date())
        except Exception as e:
            print(f"Ошибка получения данных, дата последнего успешного получения {self.last_fetch_time}")

    def get_requests(self):
        """Отдает запросы и очищает кэш"""
        requests = self.requests
        self.requests = None
        return requests


class VectorDatabaseTouch:
    def __init__(self, url):
        # Подключаемся к Qdrant
        self.client = QdrantClient(url)
        self.collection_name = "support_tickets"  # Название коллекции
        self.vector_size = 312  # размер эмбеддинга
        self.distance = Distance.COSINE  # метрика
        self.points_count = 0

    def init_db(self):
        # Создаём коллекцию
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=self.vector_size, distance=self.distance),
        )

    def save_embeddings(self, embeddings, texts, bm25_tokens, registry_date):
        # Формируем записи для Qdrant
        points = [
            PointStruct(
                id=i,
                vector=embeddings[i],
                payload={
                    "text": texts[i],
                    "bm25_token": bm25_tokens[i],
                    "registry_date": registry_date[i]
                }
            )
            for i in range(len(texts))
        ]

        # Сохраняем в Qdrant
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        print("✅ Эмбеддинги успешно сохранены в Qdrant!")

    def fetch_embeddings(self, embedding):
        # Получаем все эмбеддинги из Qdrant
        hits = self.client.query_points(
            collection_name=self.collection_name,
            query_vector=embedding,
            limit=self.points_count  # Находим все
        )
        print("✅ Эмбеддинги получены")
        return hits

    def get_points_count(self):
        # Получаем количество записей и сохраняем в переменную, для оптимизации
        info = self.client.get_collection(collection_name=self.collection_name)
        self.points_count = info.result.points_count  # актуальное количество точек
        print("Записей в коллекции:", self.points_count)



