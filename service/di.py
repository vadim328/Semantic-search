# services/di.py
from db.relational_db.relational_db import RelationalDatabaseTouch
from db.vector_db.client import VectorDB
from models.inference_models import EmbeddingModel, LLMModel
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

        log.info("Load LLM model")
        self.llm_model = LLMModel(model_path=cfg["models"]["llm"]["path"])

        log.info("Load embedding model")
        self.embedding_model = EmbeddingModel(
            cfg["models"]["embedding"]["path"],
            cfg["models"]["embedding"]["model_name"]
        )

        log.info("Init relational db client")
        self.relational_db = RelationalDatabaseTouch(
            cfg["database"]["relational_db"]["url"]
        )

        log.info("Init vector db client")
        self.vector_db = VectorDB(url=cfg["database"]["vector_db"]["url"])
        self._build_collections()

    @staticmethod
    def _build_llm_models():

        llm_cfg = cfg["models"]["llm"]

        return {
            name: LLMModel(model_path=path["path"])
            for name, path in llm_cfg.items()
        }

    def _build_collections(self):

        for collection in cfg["service"]["products"]:
            self.vector_db.make_collection(
                name=collection,
                qdrant_config=cfg["database"]["vector_db"]["params"],
                date_from=cfg["database"]["vector_db"]["date_from"]
            )


container = Container()
