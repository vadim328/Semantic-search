import grpc
from concurrent import futures

from contracts.generated import model_pb2, model_pb2_grpc

from model_service.service.inference.embedding import EmbeddingModel
from model_service.service.inference.llm import LLMModel

import logging
from model_service.service.logging_config import setup_logging
from model_service.service.config import Config

setup_logging()  # настройка логирования
log = logging.getLogger(__name__)
config = Config()


class ModelService(model_pb2_grpc.ModelServiceServicer):

    def __init__(self):

        self.embedding_model = EmbeddingModel(
            config.embedding["path"],
            config.embedding["model_name"]
        )

        self.llm_model = LLMModel(
            model_path=config.llm["path"],
            n_ctx=config.llm["n_ctx"],
            threads=config.llm["n_threads"]
        )

    def Generate(self, request, context):

        result = self.llm_model.generate(
            request.prompt,
            max_tokens=request.max_tokens
        )

        return model_pb2.GenerateResponse(text=result)

    def Embed(self, request, context):

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
