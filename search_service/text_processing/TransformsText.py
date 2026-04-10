import nltk
from nltk.corpus import stopwords
from cleantext import clean
import re
from natasha import Doc, Segmenter, NewsEmbedding, NewsMorphTagger, MorphVocab
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List


# Проверка и тихая загрузка ресурсов
def ensure_nltk_resources():
    resources = [
        ("corpora/stopwords", "stopwords"),
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
    ]
    for path, name in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name, quiet=True)


ensure_nltk_resources()

# Стоп-слова
sw = set(stopwords.words("russian"))
sw.update(["добрый", "день", "вечер", "привет", "здравствуйте", "запрос", "оригинальный", "и", "в", "у", "с", "к"])
keep_words = {"не"}
sw = sw - keep_words

transcript = [
    # Erudite
    (r'erudite', 'эрудит'),
    (r'levitan', 'левитан'),
    (r'threducon', 'тредукон'),
    (r'reporter', 'репортер'),
    (r'naumen speech ai', 'наумен спич'),
    (r'consul', 'консул'),
    (r'patroni', 'патрони'),
    (r'postgres', 'постгрес'),
    (r'grafana', 'графана'),
    (r'amd', 'амд'),
    (r'asr', 'распознавание'),
    (r'tts', 'синтез'),
    (r'crt', 'црт'),
    (r'erudite-web', 'эрудит-веб'),
    (r'erudite-python', 'эрудит-пайтон'),
    (r'levitan-python', 'левитан-пайтон'),
    (r'docker', 'докер'),
    (r'kubernetes', 'кубер'),
    (r'kerberos', 'керберос'),

    # NCC
    (r'ncc', 'нцц'),
    (r'pms', 'пмс'),
    (r'naumen contact center', 'нцц'),
    (r'dialer', 'диалер'),
    (r'buddy', 'бадди'),
    (r'naucore', 'наукор'),
    (r'naumuddy', 'бадди'),
    (r'naumb', 'мб'),
    (r'naucore', 'наукор'),
    (r'snitch', 'снитч'),
    (r'balancer', 'балансер'),
    (r'ncc chat', 'нцц чат'),
    (r'tel', 'тел'),
    (r'nautel', 'тел'),

    # Общие
    (r'rest api', 'рест апи'),
    (r'api', 'апи'),
]

segmenter = Segmenter()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)
morph_vocab = MorphVocab()

vectorizer = TfidfVectorizer()


class TextCompose:
    """Основной класс для трансформаций"""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, text: str):
        for t in self.transforms:
            text = t(text)["text"]
        return {"text": text}


class TextLemmatization:
    """Приведение текста к одной лемме"""

    def __call__(self, text: str):
        doc = Doc(text)
        doc.segment(segmenter)
        doc.tag_morph(morph_tagger)
        for token in doc.tokens:
            token.lemmatize(morph_vocab)
        lemmas_text = [_.lemma for _ in doc.tokens]
        text = ' '.join(lemmas_text)
        return {"text": text}


class RemoveStopWords:
    """Удаление стоп-слов из текста"""

    def __call__(self, text: str):
        tokens = text.split(' ')
        tokens_without_sw = [token for token in tokens if token not in sw]
        text_without_sw = ' '.join(tokens_without_sw)
        return {"text": text_without_sw}


class ReplaceText:
    """Замена слов и символов"""

    def __init__(self, replacements: List):
        self.replacements = replacements

    def __call__(self, text: str):
        for replacement in self.replacements:
            text = re.sub(replacement[0], replacement[1], text)
        return {"text": text}


class CleanText:
    """Очистка текста с использованием библиотеки clean-text"""

    def __init__(self,
                 no_emoji=True,
                 no_urls=True,
                 no_punct=True,
                 lower=True,
                 no_numbers=True,
                 no_currency_symbols=True,
                 to_ascii=False,
                 replace_with_url='',
                 replace_with_number='',
                 replace_with_currency_symbol=''
                 ):
        self.no_emoji = no_emoji
        self.no_urls = no_urls
        self.no_punct = no_punct
        self.lower = lower
        self.no_numbers = no_numbers
        self.no_currency_symbols = no_currency_symbols
        self.to_ascii = to_ascii
        self.replace_with_url = replace_with_url
        self.replace_with_number = replace_with_number
        self.replace_with_currency_symbol = replace_with_currency_symbol

    def __call__(self, text: str):
        text = clean(text,
                     no_emoji=self.no_emoji,
                     no_urls=self.no_urls,
                     no_punct=self.no_punct,
                     lower=self.lower,
                     no_numbers=self.no_numbers,
                     no_currency_symbols=self.no_currency_symbols,
                     to_ascii=self.to_ascii,
                     replace_with_url=self.replace_with_url,
                     replace_with_number=self.replace_with_number,
                     replace_with_currency_symbol=self.replace_with_currency_symbol
                     )
        return {"text": text}


class RemoveFirstWords:
    """Удаление определенного текста в начале строки"""

    def __init__(self, words):
        self.words = words

    def __call__(self, text: str):
        for word in self.words:
            match = re.match(word, text)
            if match:
                text = text[match.end():]
        return {"text": text}


class LowerText:
    """Приведение текста к нижнему регистру"""
    def __call__(self, text):
        if not isinstance(text, str):
            text = str(text or "")
        return {"text": text.lower()}


class StripHTML:
    def __init__(self):
        self.pattern = re.compile(r'<[^>]+>')

    def __call__(self, text: str):
        return {"text": self.pattern.sub(' ', text)}


class NormalizeWhitespace:
    def __call__(self, text: str):
        return {"text": re.sub(r'\s+', ' ', text).strip()}


class SplitBlocks:
    def __init__(self, separator="|||"):
        self.separator = separator

    def __call__(self, text: str):
        blocks = [b.strip() for b in text.split(self.separator) if b.strip()]
        return {"text": blocks}


class JoinBlocks:
    def __init__(self, separator="\n"):
        self.separator = separator

    def __call__(self, blocks: List[str]):
        return {"text": self.separator.join(blocks)}


class FilterEmpty:
    def __call__(self, blocks: List[str]):
        return {"text": [b for b in blocks if b.strip()]}


class MapBlocks:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, blocks: List[str]):
        result = []
        for b in blocks:
            cleaned = self.transform(b)["text"]
            if cleaned:
                result.append(cleaned)
        return {"text": result}


class RemoveLogs:
    def __init__(self, min_seq_len: int = 10):
        self.min_seq_len = min_seq_len

    @staticmethod
    def is_log_token(token: str) -> bool:

        # есть латиница
        if re.search(r'[a-zA-Z]', token):
            return True

        # сильные признаки логов
        log_signs = [
            ':', '(', ')', '[', ']', '::'
        ]

        if any(s in token for s in log_signs):
            return True

        # версии / системные строки
        if re.search(r'\d+\.\d+', token):
            return True

        # чисто короткие тех слова
        if re.fullmatch(r'[a-zA-Z0-9_-]+', token):
            return True

        return False

    def __call__(self, text: str):
        tokens = re.findall(r'\S+', text)

        result = []
        buffer = []

        def flush():
            nonlocal buffer

            if not buffer:
                return

            # удаляем если длинная последовательность
            if len(buffer) < self.min_seq_len:
                result.extend(buffer)

            buffer = []

        for token in tokens:
            if self.is_log_token(token):
                buffer.append(token)
            else:
                flush()
                result.append(token)

        flush()

        cleaned = ' '.join(result)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()

        return {"text": cleaned}
