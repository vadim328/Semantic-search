from tenacity import retry_if_exception
from .base import base_retry
from .conditions import is_retryable_grpc_error


def grpc_retry(*, retry_condition=None, attempts=4):
    retry_condition = retry_condition or is_retryable_grpc_error
    return base_retry(
        attempts=attempts,
        multiplier=10,
        min_wait=10,
        max_wait=60,
        retry_condition=retry_if_exception(retry_condition),
    )
