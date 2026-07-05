import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List


# INIT MODEL
def init_model(model_name: str = "Qwen/Qwen2.5-7B-Instruct"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )

    return tokenizer, model


PROMPT = """
Bạn đang hỗ trợ xây dựng hệ thống hỏi đáp pháp luật Việt Nam.

Nhiệm vụ:
- Nhận vào MỘT câu hỏi của người dân
- Phân tích ngữ nghĩa của câu hỏi và sinh ra CHÍNH XÁC {num_queries} câu hỏi khác nhưng chuyển sang các cách diễn đạt khác
- Tuyệt đối GIỮ NGUYÊN Ý NGHĨA của câu hỏi gốc

Yêu cầu định dạng:
- Mỗi câu hỏi là MỘT câu hoàn chỉnh, không viết dở dang, không thêm giải thích
- Mỗi câu trên một dòng riêng, không đánh số, không gạch đầu dòng

Ví dụ:
Câu hỏi gốc: "Vượt đèn đỏ bị phạt như thế nào?"
Các câu được sinh ra:
Hành vi không chấp hành hiệu lệnh của đèn tín hiệu giao thông bị xử phạt như thế nào?
Mức xử phạt vi phạm hành chính đối với hành vi không tuân thủ tín hiệu đèn giao thông là bao nhiêu?
Người điều khiển phương tiện vượt đèn tín hiệu giao thông sẽ phải chịu chế tài gì theo quy định pháp luật?


Câu hỏi gốc:
"{query}"
"""

# GENERATE
def generate_similar_queries(
    query: str,
    tokenizer,
    model,
    num_queries: int = 3,
    max_new_tokens: int = 1024,
    temperature: float = 0.6,
) -> List[str]:

    if not query or not query.strip():
        raise ValueError("Query đầu vào không hợp lệ")

    prompt = PROMPT.format(query=query, num_queries=num_queries)
    messages = [
        {
            "role": "system",
            "content": "Bạn là chuyên gia ngôn ngữ pháp luật Việt Nam."
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.7,
            do_sample=True
        )

    generated_ids = outputs[0][model_inputs.input_ids.shape[-1]:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    queries: List[str] = []

    for line in generated_text.split("\n"):
        line = line.strip()
        if not line:
            continue

        line = re.sub(r"^[0-9\-\*\•\)]\s*", "", line)
        queries.append(line)

    return queries[:num_queries]


def rewrite_with_original(query: str, tokenizer, model, num_queries: int = 3) -> List[str]:
    expanded = generate_similar_queries(
        query,
        tokenizer,
        model,
        num_queries=num_queries
    )
    return [query] + expanded