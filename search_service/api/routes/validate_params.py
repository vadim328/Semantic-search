from typing import List
from fastapi import HTTPException


def validate_params(params: dict, req_params: List):
    for req_param in req_params:
        if not params.get(req_param):
            raise HTTPException(
                status_code=400,
                detail="Missing required parameters"
            )
