from typing import List, Dict
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from search_service.service.utils.utils import load_file
import logging

log = logging.getLogger(__name__)


class RelationalDatabaseTouch:
    def __init__(self, url):
        engine = create_async_engine(url)
        self.Session = async_sessionmaker(bind=engine)
        self.request_data_query = text(load_file("search_service/db/relational_db/queries/fetch_data_request.sql"))
        self.requests = []

    @retry(
        stop=stop_after_attempt(3),  # максимум 3 попытки
        wait=wait_exponential(multiplier=1, min=1, max=10),  # exponential backoff
        retry=retry_if_exception_type(Exception),
        reraise=True,  # пробрасывает финальную ошибку
    )
    async def make_request(self,
                           query: text,
                           params=None) -> List[Dict]:
        """
        Получение данных из БД и возврат результата в виде списка словарей.
        
        Args:
            query (str): SQL-запрос
            params (dict, optional): Параметры для использования в запросе. По умолчанию None.
        
        Returns:
            List[dict]: Список строк результата запроса, каждая строка — словарь.
        """
        log.debug(f"Request params: {params}")
        async with self.Session() as session:
            try:
                response = await session.execute(query, params)
                response = [dict(row) for row in response.mappings().all()]
                return response
            except Exception as e:
                log.error(f"Error retrieving data from relational db: {e}")
                raise

    async def fetch_data(self, params: dict):
        """
        Формирование запроса на получение данных
        Args:
            params (dict): Параметры запроса
        """
        query = text(load_file("search_service/db/relational_db/queries/fetch_requests.sql"))
        self.requests.extend(await self.make_request(query, params))
        log.info(f"Data received from relational db, count rows - {len(self.requests)}")

    async def fetch_additional_data(self, params: dict) -> List[Dict]:
        """
        Формирование запроса на получение дополнительных данных
        Args:
            params (dict): Параметры запроса
        Returns:
            additional_data (List[dict]): Результат запроса
        """
        query = text(load_file("search_service/db/relational_db/queries/additional_data.sql"))
        additional_data = await self.make_request(query, params)
        log.info(f"Additional data received from relational db: {additional_data}")
        return additional_data

    async def fetch_request_data(self, params: dict) -> List[Dict]:
        """
        Формирование запроса на получение описания и комментариев по определенному запросу
        Args:
            params (dict): Параметры запроса
        Returns:
            request_data (List[dict]): Результат запроса
        """
        request_data = await self.make_request(self.request_data_query, params)
        log.debug(f"Request data received from relational db: {request_data}")
        return request_data

    def get_data(self) -> List[dict]:
        """
        Отдает запросы и очищает кэш
        Returns:
            requests (List[dict]): Запросы полученные из БД
        """
        requests = self.requests
        self.requests.clear()
        return requests
