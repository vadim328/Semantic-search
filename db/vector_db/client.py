from qdrant_client import QdrantClient
from db.vector_db.collection import CollectionStore


class VectorDB:

    def __init__(self, url: str):
        self.client = QdrantClient(url)
        self._collections = {}

    def make_collection(
            self,
            date_from: str,
            name: str,
            qdrant_config: dict
    ):

        self._collections[name] = CollectionStore(
            client=self.client,
            collection=name,
            qdrant_config=qdrant_config,
            date_from=date_from
        )

    def collection(self, name: str):
        return self._collections[name]

    def collections(self) -> dict[CollectionStore]:
        return self._collections
