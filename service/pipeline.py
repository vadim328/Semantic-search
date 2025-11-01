import numpy as np
from models.model import OnnxSentenseTransformer
from db.database import RelationalDatabaseTouch, VectorDatabaseTouch
from rank_bm25 import BM25Okapi
from text_processing.text_preparation import transforms_bm25, transforms_bert


class SemanticSearchEngine:
    def __init__(self):
        self.model = OnnxSentenseTransformer(
            'models/onnx/',
            'model_optimized.onnx'
        )
        self.relational_db = RelationalDatabaseTouch(
            "REMOVED"
        )
        self.vector_db = VectorDatabaseTouch("http://localhost:6333")

    def search(self, query: str, alpha=0.5):

        tokenized_query = transforms_bm25(text=query)["text"].split()
        query_bert = transforms_bert(text=query)["text"]

        embedding = self.model.encode(query_bert)[0]
        hits = self.vector_db.fetch_embeddings(embedding)

        cosine_scores, tokenized_querys = [], []
        numbers = []

        for hit in hits:
            numbers.append(hit.id)
            cosine_scores.append(hit.score)
            tokens = transforms_bm25(text=hit.payload.text)["text"].split()
            tokenized_querys.append(tokens)

        cosine_scores = np.array(cosine_scores)  # преобразуем в numpy массив для дальнейших рассчетов
        bm25 = BM25Okapi(tokenized_querys)

        # --- Считаем BM25 ---
        bm25_scores = bm25.get_scores(tokenized_query)

        # 4. Нормализация и объединение
        bm25_norm = bm25_scores / (bm25_scores.max() + 1e-9)
        cosine_norm = cosine_scores / (cosine_scores.max() + 1e-9)
        hybrid_scores = alpha * bm25_norm + (1 - alpha) * cosine_norm

        # --- Ранжируем ---
        ranked = sorted(zip(numbers, hybrid_scores), key=lambda x: x[1])

        return ranked






