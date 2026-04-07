import json
import logging
import asyncio
from typing import List

from search_service.service.clients.summarization_builder import build_summarization_prompts
from search_service.service.clients.llm_settings import LLMSettings as settings

log = logging.getLogger(__name__)


class SummarizationOrchestrator:
    """
    Оркестратор для суммаризации запроса
    """
    def __init__(self, client):
        self.client = client

    async def _generate(self, prompt: str) -> str:
        """
        Внутренний метод для генерации
        Args:
            prompt (str): Промпт для LLM
        Returns:
            result (str): Результат генерации (суммаризации)
        """
        result = await self.client.generate(prompt)
        log.info(f"Result summarization:\n{result}")
        return result

    async def _map_phase(self,
                         prompts: List[str],
                         max_concurrent: int) -> List[str]:
        """
        Поочередная суммаризация чанков одного запроса
        Args:
            prompts (List[str]): Промпты для LLM
            max_concurrent (int): Количество одновременно (почти) выполняющихся запросов
        Returns:
            summaries (List[str]): Список суммаризированных чанков
        """

        semaphore = asyncio.Semaphore(max_concurrent)  # Ограничиваем одновременное количество запросов

        async def limited_generate(prompt: str, idx: int) -> str:
            async with semaphore:
                log.info(f"Start summarization for chunk {idx + 1}/{len(prompts)}")
                result = await self._generate(prompt)
                log.info(f"Summarization for chunk {idx + 1}/{len(prompts)} - finish")
                return result

        tasks = [limited_generate(p, i) for i, p in enumerate(prompts)]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        summaries = []
        for r in results:
            if isinstance(r, Exception):
                log.warning("Chunk failed: %s", r)
                continue
            summaries.append(r)

        return summaries

    async def _reduce_phase(self, summaries: List[str]) -> str:
        """
        Суммаризация суммаризированных чанков
        Args:
            summaries (List[str]): Суммаризированные чанки
        Returns:
            str: Результат суммаризации
        """
        summaries_text = "\n\n".join(
            json.dumps(s, ensure_ascii=False)
            for s in summaries
        )

        final_prompt = settings.prompt_template.format(
            problem=summaries_text,
            comments=settings.default_empty_comments,
        )

        log.info("Result for chunked summarization:")
        return await self._generate(final_prompt)

    async def summarize(self,
                        problem: str,
                        comments: str,
                        max_concurrent: int) -> str:
        """
        Суммаризация запроса по проблеме и комментариям
        Args:
            problem (str): Описание проблемы
            comments (str): Комментарии
            max_concurrent (int): Количество одновременно (почти) выполняющихся запросов
        Returns:
            str: Результат суммаризации
        """

        prompts = build_summarization_prompts(
            problem=problem,
            comments=comments,
            max_context_tokens=settings.max_context_tokens,
            chars_per_token=settings.chars_per_token,
            token_safety_ratio=settings.token_safety_ratio,
        )

        # простой кейс
        if len(prompts) == 1:
            return await self._generate(prompts[0])

        log.info("Using chunked summarization")

        summaries = await self._map_phase(prompts, max_concurrent)

        if not summaries:
            raise RuntimeError("Failed to summarize any chunk")

        return await self._reduce_phase(summaries)
