import torch
from llama_cpp import Llama
import logging

log = logging.getLogger(__name__)


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

        log.info(
            f"Initializing LLMModel: model_path={model_path}, "
            f"n_ctx={n_ctx}, threads={threads}"
        )

        self.model = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=threads,
            n_gpu_layers=0  # 0 если без GPU
        )

        log.debug("LLM model loaded successfully")

        self.generate_params = generate_params
        log.debug(f"Generation parameters: {self.generate_params}")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info(f"Using device: {self.device}")

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

        if not prompt:
            log.warning("Empty prompt received")

        log.info(f"Starting generation (prompt length={len(prompt)})")
        log.debug(f"Prompt preview: {prompt[:200]}")

        try:
            output = self.model(
                prompt,
                max_tokens=self.generate_params["max_tokens"],
                temperature=self.generate_params["temperature"],
                top_p=self.generate_params["top_p"],
                top_k=self.generate_params["top_k"],
                repeat_penalty=self.generate_params["repeat_penalty"],
            )

            text = output['choices'][0]['text']

            log.debug(f"Raw model output: {text[:200]}")
            log.info(f"Generation complete (output length={len(text)})")

        except Exception as e:
            log.exception("Error during generation")
            raise

        finally:
            # reset context for non-chat usage
            self.model.reset()
            log.debug("Model context reset")

        return text

    def infer(self, prompt):
        log.debug("Infer called")
        return self.generate(prompt)
