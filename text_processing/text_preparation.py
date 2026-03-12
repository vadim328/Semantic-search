import typing
import text_processing.TransformsText as TT
from bs4 import BeautifulSoup
import re
from typing import List


transforms_bm25 = TT.TextCompose([
    TT.RemoveFirstWords([r'Erudite']),
    TT.CleanText(
        no_emoji=True,
        no_urls=True,
        no_punct=False,
        lower=True,
        no_numbers=True,
        no_currency_symbols=True,
        to_ascii=False,
        replace_with_url='веб-интерфейс',
        replace_with_number='',
        replace_with_currency_symbol=''
    ),
    TT.ReplaceText([
                    (r'erudite', 'система'),
                    (r'[^а-яА-Яa-zA-Z0-9\s\-]', ''),
    ]),
    TT.TextLemmatization(),
    TT.RemoveStopWords(),
    # Убираем длинные слова, на английском, которые слиплись
    TT.ReplaceText([
                    (r'\b[A-Za-z]{8,}\b', ''),
                    (r'\s{2,}', ' '),
    ]),
])

transforms_bert = TT.TextCompose([
    TT.RemoveFirstWords([r'Erudite']),
    TT.CleanText(
        no_emoji=True,
        no_urls=True,
        no_punct=False,
        lower=True,
        no_numbers=True,
        no_currency_symbols=True,
        to_ascii=False,
        replace_with_url='веб-интерфейс',
        replace_with_number='',
        replace_with_currency_symbol=''
    ),
    TT.ReplaceText([
                    (r'erudite', 'система'),
    ]),
    # Убираем длинные слова, на английском, которые слиплись
    TT.ReplaceText([
                    (r'\b[A-Za-z]{8,}\b', ''),
                    (r'\s{2,}', ' '),
    ]),
])

transforms_comments = TT.TextCompose([
    TT.ReplaceText([
                    (r'в работе', ''),
                    (r'запрос в работе', ''),
                    (r'отложен до.*', ''),
                    (r'т\.к\.', ''),
                    (r'отложенная заявка автоматически.*', ''),
                    (r'<b>#### статус изменился.*', ''),
                    (r'####.*', ''),
                    (r'данный запрос находится в состоянии.*', ''),
                    (r'необходимо классифицировать запрос', ''),
                    (r'категория назначена', ''),
    ])
])

def preparation_list(texts: list[str]) -> tuple[list[list[str]], list[str]]:
    """Подготавливает тексты для BM25 и BERT-моделей.

    Args:
        texts: Список исходных текстов.

    Returns:
        Кортеж из двух списков строк:
        - первый — тексты, подготовленные для BM25;
        - второй — тексты, подготовленные для BERT.
    """
    tokens_bm25 = [transforms_bm25(text=text)["text"].split() for text in texts]
    texts_bert = [transforms_bert(text=text)["text"] for text in texts]
    return tokens_bm25, texts_bert


def preparation_str(text: str) -> tuple[list[str], str]:
    """Подготавливает тексты для BM25 и BERT-моделей.

    Args:
        text: Список исходных текстов.

    Returns:
        Кортеж из двух списков строк:
        - первый — тексты, подготовленные для BM25;
        - второй — тексты, подготовленные для BERT.
    """
    tokens_bm25 = transforms_bm25(text=text)["text"].split()
    text_bert = transforms_bert(text=text)["text"]

    return tokens_bm25, text_bert


def strip_html(text: str) -> str:
    return re.compile(r'<[^>]+>').sub(' ', text)


def normalize_whitespace(text: str) -> str:
    return re.compile(r'\s+').sub(' ', text).strip()


def clean_comment_block(text: str) -> str:
    text = text.lower()

    text = transforms_comments(text=text)["text"]

    if "<" in text and ">" in text:
        text = strip_html(text)

    text = normalize_whitespace(text)

    return text


def clean_comments(raw_text: str) -> str:

    if not raw_text:
        return ""

    blocks: List[str] = [block.strip() for block in raw_text.split("|||")]

    cleaned_blocks = []

    for block in blocks:

        if not block:
            continue

        cleaned = clean_comment_block(block)

        if cleaned:
            cleaned_blocks.append(cleaned)

    return "\n".join(cleaned_blocks)
