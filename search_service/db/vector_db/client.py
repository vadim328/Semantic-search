from typing import List
from qdrant_client import AsyncQdrantClient
from search_service.db.vector_db.collection import CollectionStore


class VectorDB:

    def __init__(self, url: str):
        self.client = AsyncQdrantClient(url)
        self._collections = {}

    async def make_collection(
        self,
        date_from: str,
        collection_name: str,
        vectors_param: List[dict],
        qdrant_config: dict
    ):

        store = await CollectionStore.create(
            client=self.client,
            collection=collection_name,
            vectors_param=vectors_param,
            qdrant_config=qdrant_config,
            date_from=date_from
        )

        self._collections[collection_name] = store

    def collection(self, name: str):
        return self._collections[name]

    def collections(self) -> dict[CollectionStore]:
        return self._collections
