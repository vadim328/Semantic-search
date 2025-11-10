import logging
import sys


def setup_logging():
    """Настройка логирования сервиса
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Проверка, чтобы не добавлять повторно обработчики
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(pathname)s] %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
