from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from search_service.service.utils.utils import load_file
import logging

log = logging.getLogger(__name__)


class RelationalDatabaseTouch:
    def __init__(self, url):
        engine = create_async_engine(url)
        self.Session = async_sessionmaker(bind=engine)
        self.request_data_query = text(load_file("search_service/db/relational_db/queries/fetch_data_request.sql"))
        self.requests = {}

    async def make_request(self, query, params=None):
        """
            Получение данных из БД, сохраняет в переменную
            :input:
                bool: Определяет первичную загрузку или вторичную
                datetime: Дата последней записи в векторной БД
                    или дата последнего успешного сохранения
        """
        log.debug(f"Request params: {params}")
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
        query = text(load_file("search_service/db/relational_db/queries/fetch_requests.sql"))
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
        query = text(load_file("search_service/db/relational_db/queries/additional_data.sql"))
        additional_data = await self.make_request(query, params)
        log.info(f"Additional data received from relational db: {additional_data}")
        return additional_data

    async def fetch_request_data(self, params: dict):
        """
            Формирование запроса на получение описания и
                комментариев по определенному запросу
            :input:
                dict: Параметры запроса
            :output:
                dict: Результат запроса
        """
        request_data = await self.make_request(self.request_data_query, params)
        log.debug(f"Request data received from relational db: {request_data}")
        return request_data

    def get_data(self):
        """
            Отдает запросы и очищает кэш
            :output:
                dict: Запросы полученные из БД
        """
        requests = self.requests
        self.requests = {}
        return requests
