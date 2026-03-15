from datetime import datetime


# Читаем файл
def load_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def timestamp_to_date(timestamp_date: float) -> datetime:
    return datetime.fromtimestamp(timestamp_date)
