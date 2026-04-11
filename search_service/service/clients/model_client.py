from typing import List, Union

import grpc.aio
import numpy as np

from contracts.generated import model_pb2, model_pb2_grpc
from search_service.service.clients.llm_settings import LLMSettings as settings
from search_service.infrastructure.retry.grpc import grpc_retry
import logging

log = logging.getLogger(__name__)


class ModelServiceClient:
    """
    Клиент для взаимодействия с сервисом моделей
    """
    def __init__(self,
                 url: str,
                 timeout_generate=90,
                 timeout_embed=90
                 ):
        self._channel = grpc.aio.insecure_channel(url)
        self.stub = model_pb2_grpc.ModelServiceStub(self._channel)

        self.timeout_generate = timeout_generate
        self.timeout_embed = timeout_embed

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self._channel.close()

    @grpc_retry()
    async def generate(self, prompt: str) -> str:
        """
        Выполняет gRPC запрос к LLM модели
        Args:
            prompt (str): Промпт для LLM модели
        Returns:
            str: Ответ LLM модели
        """
        log.debug(f"Prompt for summarization:\n{prompt}")

        response = await self.stub.Generate(
            model_pb2.GenerateRequest(                                     # type: ignore
                prompt=prompt,
                max_tokens=settings.generation_tokens,
            ),
            timeout=self.timeout_generate,
        )
        return response.text

    @grpc_retry()
    async def embed(
            self,
            texts: Union[str, List[str]],
            prefix: str) -> np.ndarray:
        """
        Выполняет gRPC запрос к Embedding модели
        Args:
            texts (Union[str, List[str]]): Строки(а) для получения эмбеддинга
            prefix (str): query/passage. query - для поиска passage - для сохранения в БД
        Returns:
            np.ndarray: Эмбеддинг текста
        """
        if isinstance(texts, str):
            texts = [texts]

        response = await self.stub.Embed(
            model_pb2.EmbeddingRequest(                                    # type: ignore
                texts=texts,
                prefix=prefix,
            ),
            timeout=self.timeout_embed,
        )

        if not response.embeddings:
            raise ValueError("Empty embeddings response")

        return np.array(
            response.embeddings[0].vector,
            dtype=np.float32,
        )
