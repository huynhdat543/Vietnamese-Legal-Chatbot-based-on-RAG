import os
import torch
from pathlib import Path
from typing import ClassVar


class Settings:
    # Root directory
    BASE_DIR: ClassVar[Path] = Path(__file__).resolve().parent.parent

    # Device
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Qdrant Cloud
    QDRANT_URL = ""
    QDRANT_API_KEY = ""
    COLLECTION_NAME = ""

    # Gemini
    GOOGLE_API_KEY = ""
    # BM25
    BM25_PATH: Path = BASE_DIR/"data"/"bm25_index_ver2.pkl"
    VECTOR_SIZE: int = 768

    # Retrival
    TOP_K: int = 50
    DENSE_WEIGHT: float = 0.7
    SPARSE_WEIGHT: float = 0.3
    RRF_K: int = 60

    # Model
    EMBEDDING_MODEL: str = "bqbbao6/vietnamese-legal-embedding"


settings = Settings()