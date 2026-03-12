# services/di.py
from db.database import RelationalDatabaseTouch, VectorDatabaseTouch
from models.request_embedding_service import RequestEmbedding
from models.inference_models import EmbeddingModel, LLMModel
from config import Config

cfg = Config().data

llm_model = LLMModel(cfg["models"]["llm"]["path"])
embedding_model = EmbeddingModel(
    cfg["models"]["embedding"]["path"],
    cfg["models"]["embedding"]["model_name"]
)
request_embedding = RequestEmbedding(llm_model, embedding_model)
model = RequestEmbedding(cfg["model"]["path"], cfg["model"]["model_name"])
relational_db = RelationalDatabaseTouch()
vector_db = VectorDatabaseTouch()
