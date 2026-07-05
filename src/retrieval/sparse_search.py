import pickle
from rank_bm25 import BM25Okapi
from pyvi import ViTokenizer
from config.settings import settings


class SparseSearcher:
    def __init__(self):
        if not settings.BM25_PATH.exists():
            raise FileNotFoundError(
                f"BM25 index không tồn tại tại: {settings.BM25_PATH}"
            )

        with open(settings.BM25_PATH, "rb") as f:
            data = pickle.load(f)

        self._bm25: BM25Okapi = data["bm25"]
        self._ids: list[str]  = data["ids"]

    def tokenize_query(self, query: str) -> list[str]:
        return ViTokenizer.tokenize(query).lower().split()

    def get_scores(self, query: str) -> tuple[list[float], list[str]]:
        tokenized = self.tokenize_query(query)
        scores = self._bm25.get_scores(tokenized)
        return scores.tolist(), self._ids


sparse_searcher = SparseSearcher()
