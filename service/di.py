# services/di.py
from db.database import RelationalDatabaseTouch, VectorDatabaseTouch
from models.model import OnnxSentenseTransformer
from config import Config

cfg = Config().data

model = OnnxSentenseTransformer(cfg["model"]["path"], cfg["model"]["model_name"])
relational_db = RelationalDatabaseTouch()
vector_db = VectorDatabaseTouch()
