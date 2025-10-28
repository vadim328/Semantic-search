import nltk
from nltk.corpus import stopwords
from cleantext import clean
import re
from natasha import Doc, Segmenter, NewsEmbedding, NewsMorphTagger, MorphVocab
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

nltk.download('stopwords')
sw = set(stopwords.words("russian"))
sw.update(["добрый", "день", "вечер", "привет", "здравствуйте", "запрос", "оригинальный", "и", "в", "у", "с", "к"])
keep_words = {"не"}
sw = sw - keep_words
nltk.download("punkt_tab")

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

    def __init__(self, replacements):
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
