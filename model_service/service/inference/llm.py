import torch
from llama_cpp import Llama
import logging

log = logging.getLogger(__name__)
device = "cuda" if torch.cuda.is_available() else "cpu"


class LLMModel:
    """Класс LLM модели для суммаризации диалога
        и извлечения признаков"""
    def __init__(
            self,
            model_path: str,
            n_ctx: int,
            threads: int
    ):

        self.model = Llama(
            model_path=model_path,  # путь к квантованной модели
            n_ctx=n_ctx,            # длина контекста
            n_threads=threads,      # сколько потоков CPU использовать
            n_gpu_layers=0          # 0 если без GPU
        )

    def generate(
            self,
            prompt: str,
            max_tokens=512,
    ):
        output = self.model(
            prompt,
            max_tokens=max_tokens,
            temperature=0.3,
            min_p=0.15,
            repeat_penalty=1.05,
            top_p=1.0,
            top_k=0
        )
        self.model.reset()  # вопрос-ответ, не чат

        # 3. Вывод результата
        return output['choices'][0]['text']

    def infer(self, prompt):
        return self.generate(prompt)
