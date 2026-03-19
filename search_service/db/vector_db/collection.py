from typing import List
from qdrant_client.models import PointStruct, SearchParams, VectorParams, HnswConfigDiff, Distance
from qdrant_client import QdrantClient
from db.vector_db.filters import _build_filter
from db.vector_db.metadata import CollectionMetadata
from dataclasses import asdict
from datetime import datetime
import logging

log = logging.getLogger(__name__)


class CollectionStore:

    def __init__(
            self,
            client: QdrantClient,
            collection: str,
            qdrant_config: dict,
            date_from: str
    ):

        self._client = client
        self._collection = collection

        self._metadata = CollectionMetadata()
        self._metadata.date_last_record = datetime.strptime(
            date_from,
            "%Y-%m-%d"
        ).timestamp()

        self.init_collection(
            qdrant_config=qdrant_config
        )

    def refresh_metadata(self):
        """
            Получение метаданных коллекции при первичном подключении,
            когда коллекция существует
        """
        offset = None

        clients = set()
        last_date = self._metadata.date_last_record

        # Батчами размером в 1000 точек проходимся по всем точкам коллекции
        while True:

            points, offset = self._client.scroll(
                collection_name=self._collection,
                limit=1000,
                offset=offset,
                with_payload=True,
                with_vectors=False
            )

            for p in points:

                payload = p.payload

                if "client" in payload:
                    clients.add(payload["client"])

                if "date_end" in payload:
                    if payload["date_end"] > last_date:
                        last_date = payload["date_end"]

            if offset is None:
                break

        self._metadata.clients = clients
        self._metadata.date_last_record = last_date
        self._metadata.points_count = self._client.get_collection(self._collection).points_count

    def init_collection(self, qdrant_config: dict):
        """
            Инициализация коллекции
                input:
                    qdrant_config - параметры для создания коллекции
        """
        if self._client.collection_exists(self._collection):
            log.info(f"Collection '{self._collection}' already exists")
            self.refresh_metadata()  # Получаем метаданные коллекции
            return

        log.info(f"Creating collection '{self._collection}'")

        try:

            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(
                    size=qdrant_config["vector_size"],
                    distance=Distance.COSINE
                ),
                hnsw_config=HnswConfigDiff(
                    m=qdrant_config["m_value"],
                    ef_construct=qdrant_config["ef_construct"],
                    full_scan_threshold=qdrant_config["full_scan_threshold"],
                    max_indexing_threads=qdrant_config["max_indexing_threads"],
                    on_disk=qdrant_config["on_disk"],
                )
            )

        except ValueError:
            log.error("Qdrant config not specified")

    def _update_metadata_fast(self, points: List[PointStruct]):
        """
            Обновление метаданных после добавления точек
                input:
                    points - точки для извлечения метаданных
        """

        for p in points:

            payload = p.payload

            if "client" in payload:
                self._metadata.clients.add(payload["client"])

            if "date_end" in payload:
                if payload["date_end"] > self._metadata.date_last_record:
                    self._metadata.date_last_record = payload["date_end"]

        self._metadata.points_count += len(points)

    def save_embeddings(self, points: List[PointStruct]):
        """
            Сохранение эмбеддингов в коллекцию
                input:
                    points - точки для сохранению
        """
        log.info(f"Saving {len(points)} embeddings")

        try:
            self._client.upsert(
                collection_name=self._collection,
                points=points
            )

            # быстро обновляем metadata
            self._update_metadata_fast(points)

        except Exception as e:
            log.exception(f"Embedding unsuccessfully saved. Error - {e}")

    def fetch_embeddings(
        self,
        embedding,
        exact: bool,
        filters: dict,
    ):
        """
            Получение близких эмбеддингов по косинусному расстоянию
                input:
                    embedding - вектор для рассчета
                    exact - требуется ли использовать полный перебор точек в коллекции
                    filters - фильтры для сужения поиска
        """
        query_filter = _build_filter(filters)

        hits = self._client.query_points(
            collection_name=self._collection,
            query=embedding,
            limit=self._metadata.points_count if exact else 500,  # Находим все точки, если не быстрый поиск,
            query_filter=query_filter,
            search_params=SearchParams(
                exact=exact,
                hnsw_ef=512
            ),
        )

        return hits

    def metadata(self) -> dict:
        return asdict(self._metadata)
