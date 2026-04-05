import torch
from llama_cpp import Llama
import logging

log = logging.getLogger(__name__)
device = "cuda" if torch.cuda.is_available() else "cpu"


class LLMModel:
    """
    Класс LLM модели для суммаризации диалога
        и извлечения признаков
    """
    def __init__(
            self,
            model_path: str,
            n_ctx: int,
            threads: int,
            generate_params: dict,
    ):
        """
        Инициализация энкодера и токенайзера модели
        Args:
            model_path (str): Путь до модели
            n_ctx (int): длина контекста
            threads (int): Количество потоков CPU
            generate_params (dict): Параметры для генерации
        """

        self.model = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=threads,
            n_gpu_layers=0  # 0 если без GPU
        )

        self.generate_params = generate_params

    def generate(
            self,
            prompt: str,
    ):
        """
        Обращение к модели для генерации
        Args:
            prompt (str): Промпт для генерации
        Returns:
            str: Результат генерации
        """
        output = self.model(
            prompt,
            max_tokens=self.generate_params["max_tokens"],
            temperature=self.generate_params["temperature"],
            top_p=self.generate_params["top_p"],
            top_k=self.generate_params["top_k"],
            repeat_penalty=self.generate_params["repeat_penalty"],
        )
        self.model.reset()  # вопрос-ответ, не чат. Сбрасываем контекст

        return output['choices'][0]['text']

    def infer(self, prompt):
        return self.generate(prompt)
