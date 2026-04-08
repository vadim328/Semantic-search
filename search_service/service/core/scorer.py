# service/scorer.py
from typing import List, Dict
import numpy as np
from rank_bm25 import BM25Okapi
from search_service.text_processing.text_preparation import transforms_bm25
from search_service.service.core.search_mode import SearchMode
import logging

log = logging.getLogger(__name__)


class HybridScorer:
    """Класс для вычисления гибридного результата поиска"""
    def __call__(
        self,
        hits: Dict,
        query_text: str,
        search_mode: SearchMode,
        alpha: float = 0.5
    ) -> List[Dict]:
        """
        Args
            hits (dict): Результаты поиска в векторной БД
            query_text (str): текст запроса
            alpha (float): коэффициент для гибридного поиска BM25 (0..1)
        Returns:
            List[Dict]: Отсортированыый список результатов поиска
        """
        if not 0 <= alpha <= 1:
            raise ValueError("alpha must be between 0 and 1")

        if not hits:
            return []

        # --- cosine ---
        cosine_scores = np.array([h["score"] for h in hits.values()])

        log.info(f"cosine_scores - {cosine_scores}")

        # Получаем текст проблемы в зависимости от режима поиска
        tokenized_docs = [
            transforms_bm25(search_mode.extract_text(h))["text"].split()
            for h in hits.values()
        ]

        tokenized_query = transforms_bm25(query_text)["text"].split()
        log.debug(f'transforms text for bm25: {tokenized_query}')

        bm25 = BM25Okapi(tokenized_docs)
        bm25_scores = bm25.get_scores(tokenized_query)

        # Нормализация для BM25
        bm25_norm = bm25_scores / (bm25_scores.max() + 1e-9)

        log.debug(f"Cosine score - {cosine_scores}, BM25 score - {bm25_norm}")

        # --- hybrid ---
        # Для поиска по ключевым словам лучше увеличить альфу
        hybrid_scores = (
            alpha * bm25_norm +
            (1 - alpha) * cosine_scores
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
