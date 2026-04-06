from typing import List
import torch
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer
import numpy as np
import logging

log = logging.getLogger(__name__)


class EmbeddingModel:
    """Инференс для Эмбеддинг моделей"""
    def __init__(self,
                 model_path: str,
                 file_name=None
                 ):
        """
        Инициализация энкодера и токенайзера модели
        Args:
            model_path (str): Путь до модели
            file_name (str | None): Название модели
        """

        self.encoder = ORTModelForFeatureExtraction.from_pretrained(
            model_path,
            file_name=file_name
            )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def mean_pooling(last_hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        return (last_hidden_state * mask).sum(1) / mask.sum(1)

    def encode(
            self,
            texts: List,
            batch_size=8,
            normalize=True
    ):
        """
        Получение эмбеддинга для текстов
        Args:
            texts (List): Текст
            batch_size (int): Размер батча
            normalize (bool): Определяет необходимость нормализация вектора
        Returns:
            List: список полученных эмбеддингов
        """
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True)

            # отключаем вычисление градиентов для инференса
            with torch.no_grad():
                outputs = self.encoder(**inputs)
                embeddings = self.mean_pooling(outputs.last_hidden_state, inputs["attention_mask"])

            # Нормализуем для адекватного вычисления коминусного расстояния
            if normalize:
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings.cpu().numpy())
        return np.vstack(all_embeddings)
