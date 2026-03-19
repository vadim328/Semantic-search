import grpc
from concurrent import futures

from contracts.generated import model_pb2, model_pb2_grpc

from service.inference.embedding import EmbeddingModel
from service.inference.llm import LLMModel

import logging
from logging_config import setup_logging
from config import Config

setup_logging()  # настройка логирования
log = logging.getLogger(__name__)


class ModelService(model_pb2_grpc.ModelServiceServicer):

    def __init__(self):

        config = Config()

        self.embedding_model = EmbeddingModel(
            config.embedding["path"],
            config.embedding["model_name"]
        )

        self.llm_model = LLMModel(
            config.llm["path"]
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
        futures.ThreadPoolExecutor(max_workers=4)
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
