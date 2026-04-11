from tenacity import retry, stop_after_attempt, wait_random_exponential, before_sleep_log
import logging

log = logging.getLogger(__name__)


def base_retry(
    *,
    attempts=3,
    multiplier=1,
    min_wait=1,
    max_wait=10,
    retry_condition=None,
):
    return retry(
        stop=stop_after_attempt(attempts),
        wait=wait_random_exponential(
            multiplier=multiplier,
            min=min_wait,
            max=max_wait,
        ),
        retry=retry_condition,
        before_sleep=before_sleep_log(log, logging.WARNING),
        reraise=True,
    )
