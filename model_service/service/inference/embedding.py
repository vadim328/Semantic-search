from typing import List
import torch
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer
import numpy as np
import logging

log = logging.getLogger(__name__)
device = "cuda" if torch.cuda.is_available() else "cpu"


class EmbeddingModel:
    """Класс эмбеддинг модели"""
    def __init__(self, model_path, file_name=None):
        self.encoder = ORTModelForFeatureExtraction.from_pretrained(
            model_path,
            file_name=file_name
            )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def encode(
            self,
            texts: List,
            batch_size=8,
            normalize=True
    ):
        """
            Получение эмбеддинга для текстов
            :input:
                Any (str/list): Текст
                int: Размер батча
                bool: Определяет необходимость нормализация вектора

            :output:
                list: список полученных эмбеддингов
        """
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True)

            # отключаем вычисление градиентов для инференса
            with torch.no_grad():
                outputs = self.encoder(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1)

            # Нормализуем для адекватного вычисления коминусного расстояния
            if normalize:
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings.cpu().numpy())
        return np.vstack(all_embeddings)
