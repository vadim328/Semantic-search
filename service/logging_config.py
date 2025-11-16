import logging
import sys
from config import Config


cfg = Config().data


def setup_logging():
    """
        Настройка логирования сервиса
    """
    log_level = logging.INFO if cfg["logging"]["level"] == "INFO" else logging.DEBUG
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # Проверка, чтобы не добавлять повторно обработчики
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(pathname)s] %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
