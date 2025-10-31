class EmbeddingPipeline:
    def __init__(self, postgresql_url, qdrant_url):
        self.postgresql_url = postgresql_url
        self.qdrant_url = qdrant_url
