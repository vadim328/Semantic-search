from typing import List, Union
from search_service.service.clients.chunk_settings import ChunkSettings as settings
import logging


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