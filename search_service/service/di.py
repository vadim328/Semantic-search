# services/di.py

import asyncio
import logging

from search_service.db.relational_db.relational_db import RelationalDatabaseTouch
from search_service.db.vector_db.client import VectorDB
from search_service.service.model_client import ModelServiceClient
from search_service.config import Config

log = logging.getLogger(__name__)
cfg = Config()


class Container:
    """
    Класс для инициализации объектов,
    необходимых для работы сервиса
    """

    def __init__(self):
        # ❗ только синхронная инициализация (без I/O)
        log.info("Init model service client")
        self.model_client = ModelServiceClient(cfg.model["url"])

        log.info("Init relational db client")
        self.relational_db = RelationalDatabaseTouch(
            cfg.database["relational_db"]["url"]
        )

        log.info("Init vector db client")
        self.vector_db = VectorDB(
            url=cfg.database["vector_db"]["url"]
        )

    @classmethod
    async def create(cls) -> "Container":
        self = cls()

        await self._build_collections()

        return self

    async def _build_collections(self):

        log.info("Initializing vector DB collections")

        tasks = [
            self.vector_db.make_collection(
                collection_name=name,
                vectors_param=cfg.database["vector_db"]["vector_params"],
                date_from=cfg.database["vector_db"]["date_from"],
                qdrant_config=cfg.database["vector_db"]["params"]
            )
            for name in cfg.service["products"]
        ]

        await asyncio.gather(*tasks)

        log.info("Collections initialized")


container: Container | None = None


async def init_container() -> Container:
    global container

    if container is None:
        log.info("Creating DI container")
        container = await Container.create()

    return container
