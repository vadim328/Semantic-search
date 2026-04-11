# api/schemas/summarization.py

from pydantic import BaseModel
from typing import Optional


class SummarizeRequest(BaseModel):
    text: str
    comments: Optional[str] = None
