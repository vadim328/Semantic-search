# services/di.py
from db.database import RelationalDatabaseTouch, VectorDatabaseTouch
from models.inference_models import EmbeddingModel, LLMModel
from config import Config

cfg = Config().data


class Container:
    """
        Класс для инициализации объектов,
        необходимых для работы сервиса
    """
    def __init__(self):

        self.llm_models = self._build_llm_models()

        self.embedding_model = EmbeddingModel(
            cfg["models"]["embedding"]["path"],
            cfg["models"]["embedding"]["model_name"]
        )

        self.relational_db = RelationalDatabaseTouch(
            cfg["database"]["relational_db"]["url"]
        )

        self.vector_dbs = self._build_vector_dbs()

    @staticmethod
    def _build_vector_dbs():

        vector_cfg = cfg["database"]["vector_db"]

        return {
            name: VectorDatabaseTouch(
                vector_cfg["url"],
                vector_cfg["date_from"],
                collection_cfg["name"],
                collection_cfg["indexing"]
            )
            for name, collection_cfg in vector_cfg["collections"].items()
        }

    @staticmethod
    def _build_llm_models():

        llm_cfg = cfg["models"]["llm"]

        return {
            name: LLMModel(path["path"])
            for name, path in llm_cfg.items()
        }


container = Container()
