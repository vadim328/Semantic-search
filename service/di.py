# services/di.py
from db.database import RelationalDatabaseTouch, VectorDatabaseTouch
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

        log.info("Load LLM models")
        self.llm_models = self._build_llm_models()
        log.info("LLM models is load")

        log.info("Load embedding model")
        self.embedding_model = EmbeddingModel(
            cfg["models"]["embedding"]["path"],
            cfg["models"]["embedding"]["model_name"]
        )
        log.info("Embedding model is load")

        self.relational_db = RelationalDatabaseTouch(
            cfg["database"]["relational_db"]["url"]
        )

        log.info("Init vector db")
        self.vector_dbs = self._build_vector_dbs()

    @staticmethod
    def _build_vector_dbs():

        vector_cfg = cfg["database"]["vector_db"]

        return {
            name: VectorDatabaseTouch(
                url=vector_cfg["url"],
                collection_name=collection_cfg["name"],
                date_from=vector_cfg["date_from"],
                qdrant_config=collection_cfg["indexing"]
            )
            for name, collection_cfg in vector_cfg["collections"].items()
        }

    @staticmethod
    def _build_llm_models():

        llm_cfg = cfg["models"]["llm"]

        return {
            name: LLMModel(model_path=path["path"])
            for name, path in llm_cfg.items()
        }


container = Container()
