from pathlib import Path
from datetime import datetime


# Читаем файл
def load_file(path: str | Path) -> str:
    """
    Чтение файла
    Args:
        path (str): Путь до файла
    Returns:
        str: Текст из файла
    """
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def timestamp_to_date(timestamp_date: int) -> datetime:
    """
    Преобразование timestamp в дату
    Args:
        timestamp_date (float): Путь до файла
    Returns:
        datetime: Дата
    """
    return datetime.fromtimestamp(timestamp_date)
