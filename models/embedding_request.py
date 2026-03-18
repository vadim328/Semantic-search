from models.inference_models import EmbeddingModel, LLMModel
from config import Config
from text_processing.text_preparation import clean_comments
import logging
from typing import List
import re

cfg = Config().data
log = logging.getLogger(__name__)
PROMPT_TEMPLATE = (
    "Сформируй структурированное техническое резюме проблемы.\n\n"
    "Описание проблемы:\n{problem}\n\n"
    "Комментарии:\n{comments}"
)


def make_summarize(
        llm_model: LLMModel,
        problem: str,
        comments: str
) -> str:
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
    match = re.search(r"Сценарий проблемы:\s*(.*)", llm_model.infer(prompt))
    scenario = match.group(1)
    return scenario


def fetch_embedding(
        llm_model: LLMModel,
        embedding_model: EmbeddingModel,
        problem: str,
        comments=None
) -> List[float]:

    try:
        summary = make_summarize(llm_model, problem, comments)
        log.debug(f"Result of summarization:\n{summary}")

        log.debug("Encoding summary")
        return embedding_model.encode(summary)[0]

    except Exception:
        log.exception("Embedding generation failed")
        raise
