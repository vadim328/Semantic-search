from typing import List, Dict
from qdrant_client import AsyncQdrantClient
from search_service.infrastructure.db.vector_db.collection import CollectionStore


class VectorDB:

    def __init__(self, url: str):
        self.client = AsyncQdrantClient(
            url,
            timeout=120,
        )
        self._collections = {}

    async def make_collection(

        self,
        date_from: str,
        collection_name: str,
        vectors_param: List[Dict],
        qdrant_config: dict
    ):
        """
        Инициирует создание коллекции и сохраняет ее в словарь
        Args:
            date_from (str): Дата крайней записи в коллекции
            collection_name (str): Название коллекции
            vectors_param: List[dict]: Параметры векторов (названия и размер)
            qdrant_config (dict): Параметры индексирования коллекции
        """

        store = await CollectionStore.create(
            client=self.client,
            collection=collection_name,
            vectors_param=vectors_param,
            qdrant_config=qdrant_config,
            date_from=date_from
        )

        self._collections[collection_name] = store

    def collection(self, name: str) -> CollectionStore:
        """
        Получение коллекции по ее наименованию
        Args:
            name (str): Название коллекции
        Returns:
            CollectionStore: Коллекция
        """
        return self._collections[name]

    def collections(self) -> dict[CollectionStore]:
        """
        Получение всех коллекций
        Returns:
            dict[CollectionStore]: Словарь коллекций
        """
        return self._collections
