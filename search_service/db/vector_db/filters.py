from datetime import datetime
from qdrant_client.http import models
import logging

log = logging.getLogger(__name__)


def _build_filter(filters: dict):
    """
        Создаёт объект Filter для Qdrant из словаря фильтров.
        Автоматически поддерживает:
            - точное совпадение по любому ключу
            - диапазоны дат: date_from, date_to
        :input:
            dict: массив фильтров
    """
    log.info("Building filters...")
    conditions = []

    for key, value in filters.items():
        if not value:
            continue

        # Специальная обработка диапазонов дат
        if key == "date_from":
            if not isinstance(value, float):
                value = datetime.strptime(value, "%Y-%m-%d").timestamp()
            conditions.append(
                models.FieldCondition(
                    key="date_end",
                    range=models.Range(gte=value)
                )
            )
        elif key == "date_to":
            if not isinstance(value, float):
                value = datetime.strptime(value, "%Y-%m-%d").timestamp()
            conditions.append(
                models.FieldCondition(
                    key="date_end",
                    range=models.Range(lte=value)
                )
            )
        else:
            # Любые другие поля считаем точным совпадением
            conditions.append(
                models.FieldCondition(
                    key=key,
                    match=models.MatchValue(value=value)
                )
            )

    log.info(f"Filters added: {len(conditions)} conditions")
    return models.Filter(must=conditions) if conditions else None