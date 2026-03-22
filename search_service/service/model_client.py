import re
import grpc
import numpy as np

from contracts.generated import model_pb2
from contracts.generated import model_pb2_grpc

from search_service.text_processing.text_preparation import clean_comments

import logging


log = logging.getLogger(__name__)


PROMPT_TEMPLATE = (
    "<|im_start|>user\n"
    "Сформируй структурированное техническое резюме проблемы.\n\n"
    "Описание проблемы:\n{problem}\n\n"
    "Комментарии:\n{comments}<|im_end|>\n"
    "<|im_start|>assistant\n"
)


class ModelServiceClient:

    def __init__(self, url):
        channel = grpc.insecure_channel(url)
        self.stub = model_pb2_grpc.ModelServiceStub(channel)

    def generate(self, prompt):

        log.debug(f"Prompt for LLM - {prompt}")
        response = self.stub.Generate(
            model_pb2.GenerateRequest(
                prompt=prompt,
                max_tokens=512
            )
        )
        return response.text

    def embed(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        response = self.stub.Embed(
            model_pb2.EmbeddingRequest(
                texts=texts
            )
        )
        # меняем на нужный тип данных
        query_vector = np.array(response.embeddings[0].vector, dtype=np.float32)
        return query_vector

    def make_summarize(self, problem, comments):

        if comments:
            comments = clean_comments(comments)
        else:
            comments = "отсутствуют"

        prompt = PROMPT_TEMPLATE.format(
            problem=problem,
            comments=comments
        )

        result = self.generate(prompt)
        log.debug(f"Result sum - {result}")
        match = re.search(r"Сценарий проблемы:\s*(.*)", result)
        scenario = match.group(1)
        return scenario
