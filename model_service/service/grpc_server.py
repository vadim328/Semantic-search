import grpc
from concurrent import futures

from contracts.generated import model_pb2, model_pb2_grpc

from model_service.service.inference.embedding import EmbeddingModel
from model_service.service.inference.llm import LLMModel

from model_service.service.logging_config import setup_logging
from model_service.service.config import Config
import logging

setup_logging()  # настройка логирования
log = logging.getLogger(__name__)
config = Config()


class ModelService(model_pb2_grpc.ModelServiceServicer):
    """
    Сервис для взаимодействия с инференсом моделей.

    Предоставляет методы для:
    - генерации текста с помощью LLM
    - получения эмбеддингов текста
    """

    def __init__(self):
        """Инициализация моделей эмбеддингов и LLM."""
        self.embedding_model = EmbeddingModel(
            config.embedding["path"],
            config.embedding["model_name"]
        )

        self.llm_model = LLMModel(
            model_path=config.llm["path"],
            n_ctx=config.llm["n_ctx"],
            threads=config.llm["n_threads"],
            generate_params=config.llm["generate"]
        )

    def Generate(self,
                 request: model_pb2.GenerateRequest,
                 context: grpc.ServicerContext
                 ) -> model_pb2.GenerateResponse:
        """
        Генерация текста на основе входного промпта.

        Args:
            request: Объект запроса gRPC, содержащий:
                - prompt (str): входной текст для генерации
            context: grpc.ServicerContext — служебный объект gRPC,
                предоставляющий доступ к метаданным запроса,
                управлению статусами, таймаутами и отменой вызова.

        Returns:
            model_pb2.GenerateResponse: объект с сгенерированным текстом.
        """
        result = self.llm_model.generate(
            request.prompt,
        )

        return model_pb2.GenerateResponse(text=result)

    def Embed(self,
              request: model_pb2.EmbeddingRequest,
              context: grpc.ServicerContext
              ) -> model_pb2.EmbeddingResponse:
        """
        Получение эмбеддингов для списка текстов.

        Args:
            request: Объект запроса gRPC, содержащий:
                - texts (Iterable[str]): список текстов
            context: grpc.ServicerContext — служебный объект gRPC
                (может использоваться для обработки ошибок, таймаутов и метаданных).

        Returns:
            model_pb2.EmbeddingResponse: объект со списком эмбеддингов,
                где каждый эмбеддинг представлен как список чисел.
        """
        embeddings = self.embedding_model.encode(list(request.texts))

        response_embeddings = []

        for emb in embeddings:
            response_embeddings.append(
                model_pb2.Embedding(vector=emb.tolist())
            )

        return model_pb2.EmbeddingResponse(
            embeddings=response_embeddings
        )


def serve():
    """
    Запуск gRPC сервера.

    Создаёт пул потоков, регистрирует сервис ModelService
    и начинает прослушивание на порту 50051.
    """
    server = grpc.server(
        futures.ThreadPoolExecutor(
            max_workers=config.service["max_workers"]
        )
    )

    model_pb2_grpc.add_ModelServiceServicer_to_server(
        ModelService(), server
    )

    server.add_insecure_port("[::]:50051")

    server.start()
    print("gRPC server started on 50051")

    server.wait_for_termination()


if __name__ == "__main__":
    serve()
