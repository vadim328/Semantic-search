import asyncio
import logging

from search_service.infrastructure.db.relational_db.relational_db import RelationalDatabaseTouch
from search_service.infrastructure.db.vector_db.client import VectorDB
from search_service.infrastructure.clients.model_client import ModelServiceClient
from search_service.infrastructure.clients.summarization_orchestrator import SummarizationOrchestrator
from search_service.config import Config

log = logging.getLogger(__name__)
cfg = Config()


class Container:
    """
    Класс для инициализации объектов, необходимых для работы сервиса
    """

    def __init__(self):
        # Cинхронная инициализация (без I/O)
        log.info("Init model service client")
        self.model_client = ModelServiceClient(
            cfg.model["url"],
            cfg.model["timeouts"]["timeout_generate"],
            cfg.model["timeouts"]["timeout_embed"]
        )

        log.info("Init relational db client")
        self.relational_db = RelationalDatabaseTouch(
            cfg.database["relational_db"]["url"]
        )

        log.info("Init vector db client")
        self.vector_db = VectorDB(
            url=cfg.database["vector_db"]["url"]
        )

        self.summarization_orchestrator = SummarizationOrchestrator(
            self.model_client
        )

    @classmethod
    async def create(cls) -> "Container":
        """
        Асинхронная фабрика для инициализации класса
        (НЕ хранит global state, полностью stateless)
        """
        self = cls()

        # Асинхронная инициализация (I/O operations)
        await self._init_async()

        return self

    async def _init_async(self):
        """
        Асинхронная инициализация внешних ресурсов
        """

        log.info("Initializing vector DB collections")

        await self._build_collections()

        log.info("Vector DB initialization completed")

    async def _build_collections(self):
        """
        Асинхронная инициализация коллекций
        """

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
