from dataclasses import dataclass, field
from typing import Set


@dataclass
class CollectionMetadata:
    """Датакласс для метаданных"""

    points_count: int = 0
    clients: Set[str] = field(default_factory=set)
    date_last_record: float = 0
