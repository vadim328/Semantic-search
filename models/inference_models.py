import torch
from abc import ABC, abstractmethod
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import logging

log = logging.getLogger(__name__)
device = "cuda" if torch.cuda.is_available() else "cpu"


class EmbeddingModel:
    """Класс эмбеддинг модели"""
    def __init__(self, model_path, file_name=None):
        self.encoder = ORTModelForFeatureExtraction.from_pretrained(
            model_path,
            file_name=file_name
            )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def encode(self, texts, batch_size=8, normalize=True):
        """
            Получение эмбеддинга для текстов
            :input:
                Any (str/list): Текст
                int: Размер батча
                bool: Определяет необходимость нормализация вектора

            :output:
                list: список полученных эмбеддингов
        """
        log.debug(f"Data: {texts}")
        if isinstance(texts, str):
            texts = [texts]
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True)

            # отключаем вычисление градиентов для инференса
            with torch.no_grad():
                outputs = self.encoder(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1)

            # Нормализуем для адекватного вычисления коминусного расстояния
            if normalize:
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings.cpu().numpy())
        return np.vstack(all_embeddings)


class LLMModel:
    """Класс LLM модели для суммаризации диалога
        и извлечения признаков"""
    def __init__(self, model_path):

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map={"": "cpu"},
            torch_dtype="bfloat16",
        )

        self.tokenizer = AutoTokenizer.from_pretrained("LiquidAI/LFM2-350M")

    def generate(self, prompt, max_tokens=512):

        input_ids = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_tensors="pt",
            tokenize=True,
        ).to(device)

        output = self.model.generate(
            input_ids,
            do_sample=True,
            temperature=0.3,
            min_p=0.15,
            repetition_penalty=1.05,
            max_new_tokens=max_tokens,
        )
        #return self.tokenizer.decode(output[0], skip_special_tokens=True)
        generated_tokens = output[0][input_ids.shape[1]:]
        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

    def infer(self, prompt):
        return self.generate(prompt)
