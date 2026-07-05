from typing import List, Dict
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering


class ModelExtractor:
    def __init__(self, model_id: str, subfolder_name: str):

        self.device = (0 if torch.cuda.is_available() else -1)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            subfolder=subfolder_name
        )

        self.model = AutoModelForQuestionAnswering.from_pretrained(
            model_id,
            subfolder=subfolder_name
        )

        self.qa_pipeline = pipeline(
            "question-answering",
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device
        )

        print("[EXTRACT] Model loaded")


    def extract(
        self,
        questions: List[str],
        candidates: List[Dict],
        score_field: str = "extract_score",
        text_field: str = "text_extract"
    ) -> List[Dict]:

        for item in candidates:
            context = item.get("text_tok", "").strip()

            if not context:
                item[text_field] = ""
                item[score_field] = 0.0
                continue

            best_answer = ""
            best_score = -1

            for question in questions:
                try:
                    result = self.qa_pipeline(
                        question=question,
                        context=context,
                        top_k=1,
                        handle_impossible_answer=True,
                        max_answer_len=256,
                        max_seq_len=512,
                        doc_stride=128,
                        truncation="only_second"
                    )

                    if isinstance(result, list):
                        result = result[0]

                    answer = result["answer"].strip()
                    score = float(result["score"])

                    if score > best_score:
                        best_score = score
                        best_answer = answer

                except Exception as e:
                    print(f"[EXTRACT ERROR] {e}")

            item[text_field] = best_answer
            item[score_field] = best_score

        return candidates