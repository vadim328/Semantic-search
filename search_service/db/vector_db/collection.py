from typing import List, Optional
from qdrant_client.models import (
    PointStruct,
    SearchParams,
    VectorParams,
    HnswConfigDiff,
    Distance,
    QueryResponse
)
from qdrant_client import AsyncQdrantClient
from search_service.db.vector_db.filters import _build_filter
from search_service.db.vector_db.metadata import CollectionMetadata
from dataclasses import asdict
from datetime import datetime
from search_service.infrastructure.retry.qdrant import qdrant_retry
import logging

log = logging.getLogger(__name__)


class CollectionStore:

    def __init__(
        self,
        client: AsyncQdrantClient,
        collection: str,
        date_from: str
    ):
        # только присвоения — никакого I/O
        self._client = client
        self._collection = collection

        self._metadata = CollectionMetadata()
        self._metadata.date_last_record = datetime.strptime(
            date_from,
            "%Y-%m-%d"
        ).timestamp()

    @classmethod
    async def create(
        cls,
        client: AsyncQdrantClient,
        collection: str,
        vectors_param: List[dict],
        qdrant_config: dict,
        date_from: str
    ) -> "CollectionStore":
        """
        Асинхронная фабрика для инициализации CollectionStore
        Args:
            client (AsyncQdrantClient): Асинхронный клиент Qdrant
            collection (str): Название коллекции
            vectors_param: List[dict]: Параметры векторов (названия и размер)
            qdrant_config (dict): Параметры индексирования коллекции
            date_from (str): Дата крайней записи в коллекции
        Returns:
            CollectionStore: Коллекция
        """

        self = cls(client, collection, date_from)

        await self._init_collection(
            vectors_param=vectors_param,
            qdrant_config=qdrant_config
        )

        return self

    async def _init_collection(
            self,
            vectors_param: List[dict],
            qdrant_config: dict
    ):
        """
        Создание коллекции (без гонок)
        Args:
            vectors_param: List[dict]: Параметры векторов (названия и размер)
            qdrant_config (dict): Параметры индексирования коллекции
        """

        exists = await self._client.collection_exists(self._collection)

        if exists:
            log.info(f"Collection '{self._collection}' already exists")
            await self._refresh_metadata()
            return

        vectors_config = {}

        for param in vectors_param:
            vectors_config[param["name"]] = VectorParams(
                size=param["size"],
                distance=Distance.COSINE
            )

        log.info(f"Creating collection '{self._collection}'")

        try:
            await self._client.create_collection(
                collection_name=self._collection,
                vectors_config=vectors_config,
                hnsw_config=HnswConfigDiff(
                    m=qdrant_config["m_value"],
                    ef_construct=qdrant_config["ef_construct"],
                    full_scan_threshold=qdrant_config["full_scan_threshold"],
                    max_indexing_threads=qdrant_config["max_indexing_threads"],
                    on_disk=qdrant_config["on_disk"],
                )
            )
        except Exception as e:
            # защита от race condition (другая корутина могла создать)
            log.warning(f"Collection creation race: {e}")

        # после — гарантированно читаем актуальное состояние
        await self._refresh_metadata()

    async def _refresh_metadata(self):
        """
        Полное обновление metadata при первом запуске приложения (дорогое)
        """

        offset: Optional[int] = None

        clients = set()
        last_date = self._metadata.date_last_record

        while True:
            points, offset = await self._client.scroll(
                collection_name=self._collection,
                limit=1000,
                offset=offset,
                with_payload=True,
                with_vectors=False
            )

            for p in points:
                payload = p.payload or {}

                client = payload.get("client")
                if client:
                    clients.add(client)

                date_end = payload.get("date_end")
                if date_end and date_end > last_date:
                    last_date = date_end

            if offset is None:
                break

        collection_info = await self._client.get_collection(self._collection)

        self._metadata.clients = clients
        self._metadata.date_last_record = last_date
        self._metadata.points_count = collection_info.points_count

    def _update_metadata_fast(self, points: List[PointStruct]):
        """
        Быстрое обновление metadata (без похода в БД)
        Args:
            points: (List[PointStruct]): Точки для обновления метаданных
        """

        for p in points:
            payload = p.payload or {}

            client = payload.get("client")
            if client:
                self._metadata.clients.add(client)

            date_end = payload.get("date_end")
            if date_end and date_end > self._metadata.date_last_record:
                self._metadata.date_last_record = date_end

        self._metadata.points_count += len(points)

    @qdrant_retry()
    async def _upsert_with_retry(self, points: List[PointStruct]):
        """
        Отдельный метод для сохранения точек с retry-декоратором
        Args:
            points [List[PointStruct]]: точки для сохранения
        """
        await self._client.upsert(
            collection_name=self._collection,
            points=points
        )

    async def save_embeddings(self, points: List[PointStruct]):
        """
        Сохранение точек в коллекцию
        Args:
            points: (List[PointStruct]): Точки которые необходимо сохранить
        """
        log.info(f"Saving {len(points)} points to collection - '{self._collection}'")

        try:
            await self._upsert_with_retry(points)
        except Exception:
            log.exception(
                f"Final failure after retries saving points "
                f"to collection '{self._collection}'"
            )
            raise

        try:
            self._update_metadata_fast(points)
        except Exception as e:
            log.error(f"Metadata update failed: {repr(e)}")
            log.warning("Metadata is out of sync with Qdrant")

        log.info(f"Points successfully saved to collection - '{self._collection}'")

    async def fetch_embeddings(
        self,
        vector_name: str,
        vector: list[float],
        exact: bool,
        filters: dict
    ) -> QueryResponse:
        """
        Получение релевантных точек из коллекции
        Args:
            vector_name (str): Название вектора для поиска
            vector (list[float]): Вектор для поиска
            exact (bool): Параметр, определяющий необходимость полного перебора точек
            filters (dict): Фильтры для сужения поиска
        Returns:
            hits (QueryResponse): Список найденных точек
        """
        query_filter = _build_filter(filters)

        hits = await self._client.query_points(
            collection_name=self._collection,
            query=vector,
            using=vector_name,
            limit=self._metadata.points_count if exact else 500,
            query_filter=query_filter,
            search_params=SearchParams(
                exact=exact,
                hnsw_ef=512
            ),
        )

        return hits

    def metadata(self) -> dict:
        """
        Получение метаданных колелкции
        Returns:
            dict: Метаданные коллекции
        """
        return asdict(self._metadata)
