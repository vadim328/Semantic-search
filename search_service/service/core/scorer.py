# service/scorer.py

import numpy as np
from rank_bm25 import BM25Okapi
from search_service.text_processing.text_preparation import transforms_bm25
import logging

log = logging.getLogger(__name__)


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
            :param alpha: коэффициент для гибридного поиска BM25 (0..1)
        """
        if not 0 <= alpha <= 1:
            raise ValueError("alpha must be between 0 and 1")

        if not hits:
            return []

        # --- cosine ---
        cosine_scores = np.array([h["score"] for h in hits.values()])

        # --- BM25 ---
        tokenized_docs = [
            transforms_bm25(h["text"])["text"].split()
            for h in hits.values()
        ]

        tokenized_query = transforms_bm25(query_text)["text"].split()
        log.debug(f'transforms text for bm25: {tokenized_query}')

        bm25 = BM25Okapi(tokenized_docs)
        bm25_scores = bm25.get_scores(tokenized_query)

        # --- normalize ---
        cosine_norm = cosine_scores / (cosine_scores.max() + 1e-9)
        bm25_norm = bm25_scores / (bm25_scores.max() + 1e-9)

        log.debug(f"Cosine score - {cosine_norm}, BM25 score - {bm25_norm}")

        # --- hybrid ---
        # Для поиска по ключевым словам лучше увеличить альфу
        hybrid_scores = (
            alpha * bm25_norm +
            (1 - alpha) * cosine_norm
        )

        # --- pack ---
        results = []
        for (hit_id, data), score in zip(hits.items(), hybrid_scores):
            results.append({
                "id": hit_id,
                "score": float(score),
                "registry_date": data["registry_date"]
            })

        return sorted(results, key=lambda x: x["score"], reverse=True)
