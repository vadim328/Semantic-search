import re
import grpc.aio
import numpy as np
import json

from contracts.generated import model_pb2
from contracts.generated import model_pb2_grpc

from search_service.text_processing.text_preparation import clean_comments

import logging


log = logging.getLogger(__name__)


PROMPT_TEMPLATE = (
    "Сформируй структурированное техническое резюме проблемы.\n\n"
    "Описание проблемы:\n{problem}\n\n"
    "Комментарии:\n{comments}\n"
    "Ответ:\n"
)


class ModelServiceClient:

    def __init__(self, url):
        channel = grpc.aio.insecure_channel(url)
        self.stub = model_pb2_grpc.ModelServiceStub(channel)

    async def generate(self, prompt):

        log.debug(f"Prompt for LLM - {prompt}")
        response = await self.stub.Generate(
            model_pb2.GenerateRequest(
                prompt=prompt,
                max_tokens=512
            )
        )
        return response.text

    async def embed(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        response = await self.stub.Embed(
            model_pb2.EmbeddingRequest(
                texts=texts
            )
        )
        # меняем на нужный тип данных
        query_vector = np.array(response.embeddings[0].vector, dtype=np.float32)
        return query_vector

    async def make_summarize(self, problem, comments):

        if comments:
            comments = clean_comments(comments)
        else:
            comments = "отсутствуют"

        prompt = PROMPT_TEMPLATE.format(
            problem=problem,
            comments=comments
        )

        result = await self.generate(prompt)
        log.debug(f"Result sum - {result}")
        scenario = json.loads(result)["Сценарий проблемы"]
        return scenario
