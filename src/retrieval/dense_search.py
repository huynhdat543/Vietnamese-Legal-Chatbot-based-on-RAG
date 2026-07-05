from sentence_transformers import SentenceTransformer
from pyvi import ViTokenizer
from config.settings import settings


class DenseSearcher:
    def __init__(self):
        self._model = SentenceTransformer(
            settings.EMBEDDING_MODEL,
            device=settings.DEVICE
        )

    def embed_query(self, query: str) -> list[float]:
        tokenized = ViTokenizer.tokenize(query)
        vector = self._model.encode(tokenized, normalize_embeddings=True)
        return vector.tolist()

dense_searcher = DenseSearcher()
