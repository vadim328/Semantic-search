# services/di.py
from db.database import RelationalDatabaseTouch, VectorDatabaseTouch
from models.request_embedding_service import RequestEmbedding
from models.inference_models import EmbeddingModel, LLMModel
from config import Config

cfg = Config().data


class Container:

    def __init__(self):

        self.llm_model = LLMModel(cfg["models"]["llm"]["path"])
        self.embedding_model = EmbeddingModel(
            cfg["models"]["embedding"]["path"],
            cfg["models"]["embedding"]["model_name"]
        )

        self.request_embedding = RequestEmbedding(self.llm_model, self.embedding_model)

        self.relational_db = RelationalDatabaseTouch(
            cfg["database"]["relational_db"]["url"]
        )

        self.vector_dbs = self._build_vector_dbs()

    def _build_vector_dbs(self):

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
