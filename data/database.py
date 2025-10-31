from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from datetime import datetime, timedelta

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, PointStruct, Distance


# Загружаем SQL из файла
def load_query(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


class RequestsDatabaseTouch:
    def __init__(self, postgresql_url):
        engine = create_engine(postgresql_url)
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
            self.last_fetch_time = str(datetime.now().date())
        except Exception as e:
            self.last_fetch_time = str(datetime.now().date() - timedelta(days=1))

    def get_requests(self):
        """Отдает запросы и очищает кэш"""
        requests = self.requests
        self.requests = None
        return requests

class VectorDatabaseTouch:
    def __init__(self, postgresql_url):
        engine = create_engine(postgresql_url)
        self.Session = sessionmaker(bind=engine)
        self.first_fetch_query = load_query('queries/fetch_all_requests.sql')
        self.last_day_query = load_query('queries/fetch_requests_last.sql')
        self.last_fetch_time = str(datetime.now().date())
        self.requests = None

