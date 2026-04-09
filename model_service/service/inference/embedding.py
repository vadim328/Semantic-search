from typing import List, Any
import torch
from torch import Tensor
import torch.nn.functional as F
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer
import numpy as np
import logging

log = logging.getLogger(__name__)


class EmbeddingModel:
    """Инференс для Эмбеддинг моделей"""
    def __init__(self,
                 model_path: str,
                 file_name: str,
                 max_length=512,
                 ):
        """
        Инициализация энкодера и токенайзера модели
        Args:
            model_path (str): Путь до модели
            file_name (str): Название модели
            max_length (int): Максимальная длина входа модели в токенах
        """

        log.info(f"Initializing EmbeddingModel: model_path={model_path}, file_name={file_name}")

        self.encoder = ORTModelForFeatureExtraction.from_pretrained(
            model_path,
            file_name=file_name
            )
        log.debug("Encoder model loaded successfully")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        log.debug("Tokenizer loaded successfully")

        self.max_length = max_length

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info(f"Using device: {self.device}")

    @staticmethod
    def mean_pooling(last_hidden_states: Tensor,
                     attention_mask: Tensor) -> Tensor:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def weighted_pooling(
            self,
            chunks: List[str],
            device: Any) -> Tensor:
        """
        Вычисление весов чанков пропорционально их длине
        Args:
            chunks (List[str]): Список текстовых чанков
            device (Any): Device
        Returns:
            weights (Tensor): Веса чанков
        """

        log.debug(f"Computing weights for {len(chunks)} chunks")

        lengths = []
        for chunk in chunks:
            tokens = self.tokenizer(
                chunk,
                add_special_tokens=False
            )["input_ids"]
            lengths.append(len(tokens))

        lengths = torch.tensor(lengths, device=device, dtype=torch.float32)

        if lengths.sum() == 0:
            log.warning("Sum of chunk lengths is zero — possible issue with input data")

        # нормализуем веса
        weights = lengths / (lengths.sum() + 1e-8)

        log.debug(f"Chunk weights: {weights.tolist()}")

        return weights

    def chunk_text(
            self,
            text: str,
            overlap: int = 50
    ) -> List[str]:
        """
        Разбивает текст на чанки по токенам с перекрытием
        Args:
            text (str): Полный текст проблемы
            overlap (int): Перекрытие, часть токенов из предыдущего чанка
        Returns:
            chunks (List[str]): Чанки текста
        """

        log.debug(f"Chunking text with length {len(text)} characters")

        tokens = self.tokenizer(
            text,
            add_special_tokens=False,
            return_attention_mask=False,
            return_tensors=None
        )["input_ids"]

        chunks = []

        if len(tokens) <= self.max_length:
            log.debug("Text fits into a single chunk")
            return [text]

        start = 0

        while start < len(tokens):
            end = start + self.max_length
            chunk_tokens = tokens[start:end]
            chunk_text = self.tokenizer.decode(
                chunk_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
            chunks.append(chunk_text)
            start += self.max_length - overlap

        log.info(f"Text split into {len(chunks)} chunks")

        return chunks

    def _encode(
            self,
            chunks: List[str],
            batch_size: int,
    ) -> Tensor:
        """
        Получение эмбеддинга для текстов
        Args:
            chunks (List[str]):
            batch_size (int): Размер батча
        Returns:
            Tensor: Полученный эмбеддинг
        """

        log.info(f"Encoding {len(chunks)} chunks (batch_size={batch_size})")

        all_embeddings = []

        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            log.debug(f"Processing batch {i // batch_size + 1} with size {len(batch_chunks)}")

            batch = self.tokenizer(
                batch_chunks,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )

            batch = {k: v.to(self.device) for k, v in batch.items()}

            with torch.no_grad():
                outputs = self.encoder(**batch)
                embeddings = self.mean_pooling(
                    outputs.last_hidden_state,
                    batch["attention_mask"]
                )

                # Нормализация каждого чанка
                embeddings = F.normalize(embeddings, p=2, dim=-1)

                all_embeddings.append(embeddings)

        result = torch.cat(all_embeddings, dim=0)
        log.info(f"Encoding complete. Output shape: {result.shape}")

        return result

    def embed(
            self,
            texts: List[str],
            batch_size=8,
    ) -> np.ndarray:

        """
        Получение эмбеддинга для текстов
        Args:
            texts (List): Список текстов
            batch_size (int): Размер батча
        Returns:
            ndarray[list[Tensor]]: array полученных эмбеддингов
        """

        log.info(f"Starting embedding for {len(texts)} texts")

        all_embeddings = []

        for idx, text in enumerate(texts):
            log.debug(f"Processing text {idx + 1}/{len(texts)}")

            # делим текст на чанки
            chunks = self.chunk_text(text)
            log.debug(f"Number of chunks: {len(chunks)}")

            # получаем эмбеддинги чанков
            chunk_embeddings = self._encode(chunks, batch_size)

            # weighted pooling
            weights = self.weighted_pooling(
                chunks,
                chunk_embeddings.device
            )
            final_embedding = (chunk_embeddings * weights.unsqueeze(1)).sum(dim=0, keepdim=True)

            # финальная нормализация
            final_embedding = F.normalize(final_embedding, p=2, dim=-1)

            all_embeddings.append(final_embedding)

        result: np.ndarray = np.vstack([emb.cpu().numpy() for emb in all_embeddings])

        log.info(f"Embedding complete. Final shape: {result.shape}")

        return result
