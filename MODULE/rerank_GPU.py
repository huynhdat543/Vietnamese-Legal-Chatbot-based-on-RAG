from typing import List, Dict
from sentence_transformers import CrossEncoder
import torch

class RerankModel:

    def __init__(self, model_path: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CrossEncoder(model_path, trust_remote_code=True,
                                  device=self.device)

        if self.device == "cuda":
            self.model.model.half()
            print("[RERANK] FP16 enabled on GPU")
        else:
            print("[RERANK] Running on CPU (FP32)")

    def rerank(
        self,
        query: str,
        candidates: List[Dict],
        top_k: int = 5,
        batch_size: int = 8
    ) -> List[Dict]:

        pairs = [(query, c["text"]) for c in candidates]

        scores = self.model.predict(
            pairs,
            batch_size=batch_size,
            show_progress_bar=False
        )

        for item, s in zip(candidates, scores):
            item["rerank_score"] = float(s)

        candidates.sort(
            key=lambda x: x["rerank_score"],
            reverse=True
        )

        return candidates[:top_k]
