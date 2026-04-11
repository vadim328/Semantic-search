# api/schemas/search.py

from pydantic import BaseModel, Field
from typing import Dict, Any, Annotated
from search_service.service.core.search_mode import SearchMode


class SearchRequest(BaseModel):
    query: str
    product: str
    limit: Annotated[int, Field(ge=1, le=20)] = 5
    alpha: Annotated[float, Field(ge=0.0, le=1.0)] = 0.5
    mode: SearchMode = SearchMode.BASE
    exact: bool = False
    filter: Dict[str, Any] = Field(default_factory=dict)
