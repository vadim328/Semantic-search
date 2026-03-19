# services/di.py
from db.relational_db.relational_db import RelationalDatabaseTouch
from db.vector_db.client import VectorDB
from service.model_client import ModelServiceClient
from config import Config
import logging

log = logging.getLogger(__name__)
cfg = Config().data


class Container:
    """
        Класс для инициализации объектов,
        необходимых для работы сервиса
    """
    def __init__(self):

        log.info("Init model service client")
        self.model_client = ModelServiceClient(cfg["model"]["url"])

        log.info("Init relational db client")
        self.relational_db = RelationalDatabaseTouch(
            cfg["database"]["relational_db"]["url"]
        )

        log.info("Init vector db client")
        self.vector_db = VectorDB(url=cfg["database"]["vector_db"]["url"])
        self._build_collections()

    def _build_collections(self):

        for collection in cfg["service"]["products"]:
            self.vector_db.make_collection(
                name=collection,
                qdrant_config=cfg["database"]["vector_db"]["params"],
                date_from=cfg["database"]["vector_db"]["date_from"]
            )


container = Container()
