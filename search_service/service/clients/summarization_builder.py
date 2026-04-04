from typing import List, Generator
from search_service.service.clients.chunk_settings import ChunkSettings as settings


def split_text_into_chunks(
    text: str,
    max_chars: int,
    overlap: int = 1000,
) -> Generator[str, None, None]:

    if overlap >= max_chars:
        raise ValueError("overlap must be smaller than max_chars")

    start = 0
    text_len = len(text)

    while start < text_len - overlap:
        end = min(start + max_chars, text_len)
        chunk = text[start:end]

        if end < text_len:
            last_newline = chunk.rfind("\n")
            if last_newline > max_chars * 0.7:
                end = start + last_newline
                chunk = text[start:end]

        yield chunk.strip()
        start = end - overlap


def build_prompt(problem: str, comments: str) -> str:
    return settings.prompt_template.format(
        problem=problem,
        comments=comments or settings.default_empty_comments
    )


def build_summarization_prompts(
    problem: str,
    comments: str,
    max_context_tokens: int,
    chars_per_token: int,
    token_safety_ratio: float,
) -> List[str]:
    """
    Возвращает список готовых prompt'ов:
    - 1 элемент → если всё влезает
    - несколько → если нужен chunking
    """

    def estimate_tokens(text: str) -> int:
        return len(text) // chars_per_token

    safe_limit = int(max_context_tokens * token_safety_ratio)

    full_prompt = build_prompt(problem, comments)

    if estimate_tokens(full_prompt) <= safe_limit:
        return [full_prompt]

    available_tokens = max_context_tokens - estimate_tokens(problem)
    max_chars = available_tokens * chars_per_token

    prompts = []
    for chunk in split_text_into_chunks(comments, max_chars=max_chars):
        prompts.append(build_prompt(problem, chunk))

    return prompts
