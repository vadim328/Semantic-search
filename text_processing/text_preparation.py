import typing
import text_processing.TransformsText as TT


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
