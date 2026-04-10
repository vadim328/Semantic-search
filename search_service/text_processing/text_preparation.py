from search_service.text_processing import TransformsText as TT


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
    TT.ReplaceText(TT.transcript),
    TT.ReplaceText([
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

transforms_embed = TT.TextCompose([
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
    TT.RemoveLogs(),
    TT.ReplaceText(TT.transcript),
    # Убираем длинные слова, на английском, которые слиплись
    TT.ReplaceText([
                    (r'\b[A-Za-z]{8,}\b', ''),
                    (r'\s{2,}', ' '),
    ]),
])

transforms_llm = TT.TextCompose([
    TT.RemoveLogs(),
    TT.ReplaceText([
        (r'\b[A-Za-z]{8,}\b', ''),
        (r'\s{2,}', ' '),
    ]),
])

comment_block_cleaner = TT.TextCompose([
    TT.RemoveFirstWords([r'Erudite']),
    TT.StripHTML(),
    TT.LowerText(),
    TT.ReplaceText([
        (r'добрый день', ''),
        (r'в работе', ''),
        (r'ознакомлен', ''),
        (r'запрос в работе', ''),
        (r'отложен до.*', ''),
        (r'т\.к\.', ''),
        (r'отложенная заявка автоматически.*', ''),
        (r'<b>#### статус изменился.*', ''),
        (r'####.*', ''),
        (r'данный запрос находится в состоянии.*', ''),
        (r'необходимо классифицировать запрос', ''),
        (r'категория назначена', ''),
    ]),
    TT.RemoveLogs(),
    TT.NormalizeWhitespace(),
])

transforms_comments = TT.TextCompose([
    TT.SplitBlocks(),                         # строка → список блоков
    TT.MapBlocks(comment_block_cleaner),      # чистим каждый блок
    TT.JoinBlocks("\n"),                      # обратно в текст
])


def preparation_list(texts: list[str]) -> tuple[list[list[str]], list[str]]:
    """Подготавливает тексты для BM25 и BERT-моделей.

    Args:
        texts: Список исходных текстов.

    Returns:
        tuple[list[list[str]], list[str]]: Кортеж из двух списков строк:
            - первый — токены, подготовленные для BM25;
            - второй — тексты, подготовленные для BERT.
    """
    tokens_bm25 = [transforms_bm25(text=text)["text"].split() for text in texts]
    texts_bert = [transforms_embed(text=text)["text"] for text in texts]
    return tokens_bm25, texts_bert


def preparation_str(text: str) -> tuple[list[str], str]:
    """Подготавливает тексты для BM25 и BERT-моделей.

    Args:
        text: Список исходных текстов.

    Returns:
        tuple[list[list[str]], str]: Кортеж из списка и строки:
            - первый — токены, подготовленные для BM25;
            - второй — текст, подготовленный для BERT.
    """
    tokens_bm25 = transforms_bm25(text=text)["text"].split()
    text_bert = transforms_embed(text=text)["text"]

    return tokens_bm25, text_bert
