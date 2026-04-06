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

    # нормализация как в sentence-transformers
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

    return embeddings.cpu().numpy()


def compare_models(original_path, optimized_path):
    texts = [
        "Привет, как дела?",
        "Как работает семантический поиск?",
        "Машинное обучение и нейронные сети",
        "Погода сегодня хорошая"
    ]

    print("🔹 Загружаем модели...")

    tokenizer = AutoTokenizer.from_pretrained(original_path)

    original_model = ORTModelForFeatureExtraction.from_pretrained(original_path)
    optimized_model = ORTModelForFeatureExtraction.from_pretrained(optimized_path)

    print("🔹 Считаем эмбеддинги...")

    emb_orig = get_embeddings(original_model, tokenizer, texts)
    emb_opt = get_embeddings(optimized_model, tokenizer, texts)

    print("\n🔹 Сравнение:")

    # cosine similarity по каждой паре
    cosine_sim = np.sum(emb_orig * emb_opt, axis=1)

    for i, text in enumerate(texts):
        print(f"\nТекст: {text}")
        print(f"Cosine similarity: {cosine_sim[i]:.6f}")

    # глобальная проверка
    is_close = np.allclose(emb_orig, emb_opt, rtol=1e-3, atol=1e-3)

    print("\n🔹 Итог:")
    print("Эмбеддинги почти одинаковые:", is_close)


if __name__ == "__main__":
    compare_models(
        original_path="./onnx",
        optimized_path="./onnx/optim"
    )
