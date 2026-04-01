import grpc.aio
import numpy as np
import json
import logging
from typing import List, Union
from search_service.service.clients.chunk_settings import ChunkSettings as settings

from contracts.generated import model_pb2
from contracts.generated import model_pb2_grpc

log = logging.getLogger(__name__)


def split_text_into_chunks(
    text: str,
    max_chars: int = 10_000,
    overlap: int = 1_000,
) -> List[str]:

    if overlap >= max_chars:
        raise ValueError("overlap must be smaller than max_chars")

    start = 0
    text_len = len(text)

    log.debug(f"len comments - {text_len}")

    while start < text_len - overlap:
        end = min(start + max_chars, text_len)
        log.debug(f"start point - {start}, end point - {end}")
        chunk = text[start:end]

        if end < text_len:
            last_newline = chunk.rfind("\n")
            if last_newline > max_chars * 0.7:
                end = start + last_newline
                chunk = text[start:end]

        yield chunk.strip()

        # Гарантируем, что start продвигается вперёд
        start = end - overlap


def build_prompt(problem: str, comments: str = settings.default_empty_comments) -> str:
    return settings.prompt_template.format(
        problem=problem,
        comments=comments
    )


class ModelServiceClient:

    def __init__(self, url: str):
        self._channel = grpc.aio.insecure_channel(url)
        self.stub = model_pb2_grpc.ModelServiceStub(self._channel)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self._channel.close()

    async def generate(self, prompt: str) -> str:
        log.debug("Prompt for LLM: %s", prompt)

        response = await self.stub.Generate(
            model_pb2.GenerateRequest(
                prompt=prompt,
                max_tokens=settings.generation_tokens
            )
        )
        return response.text

    async def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]

        response = await self.stub.Embed(
            model_pb2.EmbeddingRequest(texts=texts)
        )

        if not response.embeddings:
            raise ValueError("Empty embeddings response")

        return np.array(
            response.embeddings[0].vector,
            dtype=np.float32
        )

    def _estimate_tokens(self, text: str) -> int:
        return len(text) // settings.chars_per_token

    def _needs_chunking(self, prompt: str) -> bool:
        safe_limit = int(settings.max_context_tokens * settings.token_safety_ratio)
        return self._estimate_tokens(prompt) > safe_limit

    async def _generate_json(self, problem: str, comments: str) -> str:
        prompt = build_prompt(problem, comments)
        result = await self.generate(prompt)

        try:
            sum_result = json.loads(result)["Сценарий проблемы"]
            log.info(f"Result summaries - {sum_result}")
            return sum_result
        except json.JSONDecodeError:
            log.warning("Invalid JSON from model: %s", result)
            raise

    async def _summarize_chunks(
        self,
        problem: str,
        comments: str
    ) -> List[str]:

        available_tokens = settings.max_context_tokens - self._estimate_tokens(problem)
        max_chars = available_tokens * settings.chars_per_token

        summaries = []
        # Берем каждый чанк отдельно, через генератор, чтоб не нагружать ОЗУ
        for chunk in split_text_into_chunks(comments, max_chars=max_chars):
            try:
                summary = await self._generate_json(problem, chunk)
                summaries.append(summary)
            except json.JSONDecodeError:
                continue
            except Exception:
                log.exception("Unexpected error during chunk processing")

        return summaries

    async def _reduce_summaries(self, summaries: List[str]) -> str:
        summaries_text = "\n\n".join(
            json.dumps(s, ensure_ascii=False)
            for s in summaries
        )

        return await self._generate_json(
            problem=summaries_text,
            comments=settings.default_empty_comments
        )

    async def make_summarize(
        self,
        problem: str,
        comments: str
    ) -> str:

        prompt = build_prompt(problem, comments)

        # простой кейс
        if not self._needs_chunking(prompt):
            result = await self._generate_json(problem, comments)
            return result

        log.debug("Using chunked summarization")

        summaries = await self._summarize_chunks(problem, comments)

        if not summaries:
            raise RuntimeError("Failed to summarize any chunk")

        final = await self._reduce_summaries(summaries)

        return final
