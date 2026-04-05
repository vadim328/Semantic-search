# config.py
from dataclasses import dataclass
from search_service.config import Config

cfg = Config().data["model"]["chunking"]


@dataclass
class LLMSettings:
    """
    Настройки для LLM и чанкинга
    """
    max_context_tokens: int = cfg["max_content_tokens"]
    generation_tokens: int = cfg["generation_tokens"]
    token_safety_ratio: float = cfg["token_safety_ratio"]
    chars_per_token: int = cfg["chars_per_token"]

    default_empty_comments: str = "отсутствуют"

    prompt_template: str = (
        "Сформируй структурированное техническое резюме проблемы.\n\n"
        "Описание проблемы:\n{problem}\n\n"
        "Комментарии:\n{comments}\n"
        "Ответ:\n"
    )
