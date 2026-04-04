import grpc.aio
import numpy as np
import logging

from typing import List, Union

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception,
    before_sleep_log,
)

from contracts.generated import model_pb2, model_pb2_grpc
from search_service.service.clients.chunk_settings import ChunkSettings as settings

log = logging.getLogger(__name__)


def is_retryable_grpc_error(exception):
    import grpc

    if isinstance(exception, grpc.aio.AioRpcError):
        return exception.code() in {
            grpc.StatusCode.UNAVAILABLE,
            grpc.StatusCode.DEADLINE_EXCEEDED,
            grpc.StatusCode.RESOURCE_EXHAUSTED,
        }

    if isinstance(exception, (ConnectionRefusedError, OSError)):
        return True

    return False


class ModelServiceClient:

    def __init__(self, url: str):
        self._channel = grpc.aio.insecure_channel(url)
        self.stub = model_pb2_grpc.ModelServiceStub(self._channel)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self._channel.close()

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_random_exponential(multiplier=1, max=10),
        retry=retry_if_exception(is_retryable_grpc_error),
        before_sleep=before_sleep_log(log, logging.WARNING),
        reraise=True,
    )
    async def generate(self, prompt: str) -> str:

        log.debug(f"Prompt for summarization:\n{prompt}")

        response = await self.stub.Generate(
            model_pb2.GenerateRequest(
                prompt=prompt,
                max_tokens=settings.generation_tokens,
            ),
            timeout=90.0,
        )
        return response.text

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_random_exponential(multiplier=1, max=10),
        retry=retry_if_exception(is_retryable_grpc_error),
        before_sleep=before_sleep_log(log, logging.WARNING),
        reraise=True,
    )
    async def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]

        response = await self.stub.Embed(
            model_pb2.EmbeddingRequest(texts=texts),
            timeout=10.0,
        )

        if not response.embeddings:
            raise ValueError("Empty embeddings response")

        return np.array(
            response.embeddings[0].vector,
            dtype=np.float32,
        )
