from typing import List
import torch
from torch import Tensor
import torch.nn.functional as F
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer
import numpy as np
from numpy import ndarray
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

        self.max_length = 512

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def mean_pooling(last_hidden_states: Tensor,
                     attention_mask: Tensor) -> Tensor:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    @staticmethod
    def chunk_tokens(
            tokens: List[int],
            max_length: int = 512,
            overlap: int = 50
    ) -> List[List[int]]:
        """
        Разбивает текст на чанки по токенам с перекрытием
        Args:
            tokens (List[int]): Токены
            max_length (int): Максимальное количество токенов в чанке (ограничение модели)
            overlap (int): Перекрытие, часть токенов из предыдущего чанка
        Returns:
            chunks (List[List[int]])
        """

        chunks = []
        start = 0

        while start < len(tokens):
            end = start + max_length
            chunk_tokens = tokens[start:end]
            chunks.append(chunk_tokens)
            # шаг с overlap
            start += max_length - overlap
        return chunks

    def _encode(
            self,
            chunks: List,
            batch_size: int,
            normalize=True
    ) -> Tensor:
        """
        Получение эмбеддинга для текстов
        Args:
            chunks (List):
            batch_size (int): Размер батча
            normalize (bool): Определяет необходимость нормализация вектора
        Returns:
            Tensor: Полученный эмбеддинг
        """

        # создаем батчи чанков
        chunk_embeddings = []
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            # конвертируем обратно в тензор с паддингом
            batch_encodings = self.tokenizer.pad(
                {"input_ids": batch_chunks},
                return_tensors="pt",
            )
            batch_encodings = {k: v.to(self.device) for k, v in batch_encodings.items()}

            with torch.no_grad():
                outputs = self.encoder(**batch_encodings)
                embeddings = self.mean_pooling(outputs.last_hidden_state, batch_encodings["attention_mask"])
                chunk_embeddings.append(embeddings)

        # усредняем эмбеддинги всех чанков для одного текста
        final_embedding = torch.cat(chunk_embeddings, dim=0).mean(dim=0, keepdim=True)
        if normalize:
            final_embedding = F.normalize(final_embedding, p=2, dim=-1)

        return final_embedding

    def embed(
            self,
            texts: List,
            prefix: str,
            batch_size=8,
    ) -> ndarray[list[Tensor]]:

        """
        Получение эмбеддинга для текстов
        Args:
            texts (List): Список текстов
            prefix (str): query/passage. query - для поиска passage - для сохранения в БД
            batch_size (int): Размер батча
        Returns:
            ndarray[list[Tensor]]: array полученных эмбеддингов
        """

        all_embeddings = []

        for text in texts:
            text = f"{prefix}: {text}"
            tokens = self.tokenizer(
                text,
                add_special_tokens=False,
            )["input_ids"]
            if len(tokens) <= self.max_length:
                chunks = [tokens]
            else:
                chunks = self.chunk_tokens(tokens, max_length=self.max_length)

            all_embeddings.append(self._encode(chunks, batch_size=batch_size))

        return np.vstack(all_embeddings)
