import asyncio
import httpx
from tenacity import retry_if_exception_type
from .base import base_retry


def qdrant_retry(attempts=3):
    return base_retry(
        attempts=attempts,
        multiplier=1,
        min_wait=5,
        max_wait=15,
        retry_condition=retry_if_exception_type((
            httpx.ReadTimeout,
            httpx.ConnectError,
            asyncio.TimeoutError,
        )),
    )
