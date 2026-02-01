# service/scorer.py

import numpy as np
from rank_bm25 import BM25Okapi
from text_processing.text_preparation import transforms_bm25


class HybridScorer:
    def __call__(
        self,
        hits,
        query_text: str,
        *,
        alpha: float = 0.5
    ):
        """
        :param hits: list[ScoredPoint] из Qdrant
        :param query_text: текст запроса
        :param alpha: вес BM25 (0..1)
        """
        if not 0 <= alpha <= 1:
            raise ValueError("alpha must be between 0 and 1")

        if not hits:
            return []

        # --- cosine ---
        cosine_scores = np.array([h.score for h in hits])

        # --- BM25 ---
        tokenized_docs = [
            transforms_bm25(h.payload["text"])["text"].split()
            for h in hits
        ]
        tokenized_query = transforms_bm25(query_text)["text"].split()

        bm25 = BM25Okapi(tokenized_docs)
        bm25_scores = bm25.get_scores(tokenized_query)

        # --- normalize ---
        cosine_norm = cosine_scores / (cosine_scores.max() + 1e-9)
        bm25_norm = bm25_scores / (bm25_scores.max() + 1e-9)

        # --- hybrid ---
        hybrid_scores = (
            alpha * bm25_norm +
            (1 - alpha) * cosine_norm
        )

        # --- pack ---
        results = []
        for hit, score in zip(hits, hybrid_scores):
            results.append({
                "id": hit.id,
                "score": float(score),
                "registry_date": hit.payload.get("registry_date"),
            })

        return sorted(results, key=lambda x: x["score"], reverse=True)
