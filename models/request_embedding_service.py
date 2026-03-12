from models.inference_models import EmbeddingModel, LLMModel
from config import Config
from text_processing.text_preparation import clean_comments
import logging
from typing import List

cfg = Config().data
log = logging.getLogger(__name__)
PROMPT_TEMPLATE = (
    "Сформируй структурированное техническое резюме проблемы.\n\n"
    "Описание проблемы:\n{problem}\n\n"
    "Комментарии:\n{comments}"
)


class RequestEmbedding:
    """
        Класс для получению эмбеддинга запроса
        1. LLM модель суммаризирует запрос в нужном формате
        2. Embedding модель отдает вектор
    """
    def __init__(self, llm: LLMModel, embedding: EmbeddingModel):
        self.llm = llm
        self.embedding = embedding

    def summarize(self, problem: str, comments: str) -> str:
        """
            Отправляем запрос с комментариями на суммаризацию
            input:
                str: problem - Описание запроса
                str: comments - Комментарии запроса
            :return:
                str: Результат суммаризации от LLM модели

        """
        if comments:
            comments = clean_comments(comments)
        else:
            comments = "отсутствуют"

        prompt = PROMPT_TEMPLATE.format(
            problem=problem,
            comments=comments
        )

        log.debug("Running LLM inference")

        return self.llm.infer(prompt)

    def fetch_embedding(self, problem: str, comments=None) -> List[float]:
        try:
            summary = self.summarize(problem, comments)
            log.info(f"Summary request: {summary}")

            log.debug("Encoding summary")

            return self.embedding.encode(summary)[0]

        except Exception:
            log.exception("Embedding generation failed")
            raise
