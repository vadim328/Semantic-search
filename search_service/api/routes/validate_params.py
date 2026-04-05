from typing import List
from fastapi import HTTPException


def validate_params(params: dict, req_params: List):
    """
    Валидация параметров. проверяет наличие обязательных параметров
    Args:
        params (dict): Переданные параметры запроса
        req_params (List): Обязательные параметры для проверки их наличия
    """
    for req_param in req_params:
        if not params.get(req_param):
            raise HTTPException(
                status_code=400,
                detail=f"Missing required parameters - {req_param}"
            )
