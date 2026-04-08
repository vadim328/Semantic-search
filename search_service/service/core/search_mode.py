from enum import Enum


class SearchMode(Enum):
    FULL = "full"
    BASE = "base"
    COMMENTS = "comments"

    def get_vector_names(self):
        """Возвращает список векторов в зависимости от режима поиска"""
        if self is SearchMode.FULL:
            return ["original", "summary", "comments"]
        if self is SearchMode.BASE:
            return ["original", "summary"]
        if self is SearchMode.COMMENTS:
            return ["comments"]

    def extract_text(self, hit):
        if self == SearchMode.BASE:
            return hit["text"]
        elif self == SearchMode.COMMENTS:
            return hit["comments"]
        else:
            return hit["text"] + " " + hit["comments"]
