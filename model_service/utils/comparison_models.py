import time
import numpy as np
import torch
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForFeatureExtraction


def mean_pooling(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    return (last_hidden_state * mask).sum(1) / mask.sum(1)


def get_embeddings(model, tokenizer, texts):
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = mean_pooling(outputs.last_hidden_state, inputs["attention_mask"])
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    return embeddings.cpu().numpy()


def cosine_similarity(vecs):
    """Косинусная близость между первым и вторым вектором"""
    a, b = vecs[0], vecs[1]
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def compare_two_models(model_paths, texts):
    """
    model_paths: dict[str, str] - {"model_name": "путь до модели"}
    texts: list[str] - два текста
    """
    # Загружаем токенизаторы для каждой модели
    tokenizers = {name: AutoTokenizer.from_pretrained(path) for name, path in model_paths.items()}

    results = {}
    for name, path in model_paths.items():
        model = ORTModelForFeatureExtraction.from_pretrained(path)
        tokenizer = tokenizers[name]  # используем токенизатор этой модели
        start_time = time.time()  # старт таймера
        embeddings = get_embeddings(model, tokenizer, texts)
        print(f"Fetch embedding time, model {name} - {time.time() - start_time}")
        sim = cosine_similarity(embeddings)
        results[name] = sim

    # Вывод
    for name, sim in results.items():
        print(f"Модель '{name}': косинусная близость между текстами = {sim:.6f}")

    # Разница между моделями
    sims = list(results.values())
    if len(sims) == 2:
        diff = abs(sims[0] - sims[1])
        print(f"\nРазница в косинусной близости между моделями = {diff:.6f}")


if __name__ == "__main__":
    texts_to_compare = [
        "Не удается извлечь атрибут",
        "Ошибка Read timed out говорит, что закончился таймаут ожидания ответа от сервиса. Но как мы видим Эрудит обрабатывает часть диалогов, значит с ним проблем нет, скорее всего есть проблема в маршруте по которому идет проблемный запрос. Проверил на трех нодах, на всех есть диалоги, которые обрабатываются."
    ]

    models = {
        "Bert": "../models/embedding/optim",
        "E5": "../models/embedding/e5/quant"
    }

    compare_two_models(models, texts_to_compare)
