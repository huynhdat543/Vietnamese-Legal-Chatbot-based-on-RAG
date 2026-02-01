import torch
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
import google.generativeai as genai

model_id = "lmka05/Model_finetune_bert"
subfolder_name = "checkpoint-6795" 

class Model_Extractor:
    def __init__(self, model_id, subfolder_name):

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, subfolder=subfolder_name)

        self.model = AutoModelForQuestionAnswering.from_pretrained(model_id, subfolder=subfolder_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.qa_pipeline = pipeline(
                "question-answering",
                model=self.model,
                tokenizer=self.tokenizer,
                device= self.device
            )

    def extract(self,list_questions, list_documents):
        unique_answers = set()
        list_score = list()
        for doc_idx, doc_content in enumerate(list_documents):
            if not doc_content or len(doc_content.strip()) == 0:
                print(f"[Cảnh báo] Doc {doc_idx} bị rỗng!")
                continue

            for q_idx, question in enumerate(list_questions):
                try:
                    result = self.qa_pipeline(
                        question=question,
                        context=doc_content,
                        top_k=1,
                        handle_impossible_answer=False,
                        max_answer_len=512,
                        max_seq_len=512,
                        doc_stride=128,
                        truncation="only_second"
                    )

                    if isinstance(result, list): result = result[0]

                    answer_text = result['answer']
                    score = result['score']

                    if len(answer_text.strip()) > 0:
                        unique_answers.add(answer_text)
                        list_score.append(score)

                except Exception as e:
                    print(f"   [LỖI] Q{q_idx}: {e}")
                    continue

        return list(unique_answers), list_score




