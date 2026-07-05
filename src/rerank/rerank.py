from typing import List, Dict
from sentence_transformers import CrossEncoder
import torch

class RerankModel:

    def __init__(self, model_path: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = CrossEncoder(model_path, trust_remote_code=True,
                                  device=self.device)
        
        # chuyển sang FP16 để tăng tốc
        if self.device == "cuda":
            self.model.model.half()
            print("[RERANK] FP16 enabled on GPU")
        else:
            print("[RERANK] Running on CPU (FP32)")
        # print(f"[RERANK] Running on {self.device}")

    def rerank(
        self,
        query: str,
        candidates: List[Dict],
        top_k: int = 5,
        batch_size: int = 8,
        score_field: str = "rerank_score"
    ) -> List[Dict]:

        pairs = [(query, c["text_tok"]) for c in candidates]

        scores = self.model.predict(
            pairs,
            batch_size=batch_size,
            show_progress_bar=False
        )

        for item, s in zip(candidates, scores):
            item[score_field] = float(s)

        candidates.sort(
            key=lambda x: x[score_field],
            reverse=True
        )

        return candidates[:top_k]
