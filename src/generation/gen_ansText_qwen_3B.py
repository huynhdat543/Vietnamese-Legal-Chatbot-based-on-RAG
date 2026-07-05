from typing import List, Dict, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# INIT MODEL
def init_model(model_name: str = "Qwen/Qwen2.5-3B-Instruct"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )

    return tokenizer, model


# PROMPT
ANSWER_PROMPT = """
Bạn là trợ lý AI hỗ trợ hỏi đáp pháp luật Việt Nam.

Nhiệm vụ:
Hãy phân tích, chọn lọc và tổng hợp tất cả các thông tin có liên quan nhất từ các đoạn luật để trả lời cho câu hỏi người dùng chính xác nhất.
Câu trả lời cuối cùng phải trả lời đúng trọng tâm câu hỏi người dùng dựa trên những thông tin đã tổng hợp được,
không giải thích dài dòng, có đầy đủ chủ ngữ vị ngữ và không trích lại toàn bộ các đoạn luật.

Câu hỏi:
{query}

Các đoạn luật:
{contexts}

Hãy sinh câu trả lời cuối cùng:
"""



# FORMAT CONTEXT
def build_context_text(contexts: List[Dict]):
    formatted_contexts = []

    for idx, ctx in enumerate(contexts, start=1):

        if isinstance(ctx, dict):
            text = ctx.get("text", "").strip()
        else:
            text = str(ctx).strip()

        if not text:
            continue

        formatted_contexts.append(f"[Đoạn luật {idx}]:\n{text}")

    return "\n\n".join(formatted_contexts)


# GENERATE ANSWER
def generate_answer(
    query: str,
    contexts: List[Dict],
    tokenizer,
    model,
    max_new_tokens: int = 256,
) -> str:
    if not query or not query.strip():
        raise ValueError("Query không hợp lệ")

    if not contexts:
        return "Không tìm thấy thông tin pháp luật liên quan để trả lời câu hỏi."

    context_text = build_context_text(contexts)

    prompt = ANSWER_PROMPT.format(query=query, contexts=context_text)

    messages = [
        {
            "role": "system",
            "content": (
                "Bạn là chuyên gia tư vấn pháp luật Việt Nam. "
                "Chỉ trả lời dựa trên context được cung cấp."
            )
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

    model_inputs = tokenizer(
        [text],
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():

        outputs = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.6,
            top_p=0.7,
            do_sample=False
        )

    generated_ids = outputs[0][model_inputs.input_ids.shape[-1]:]

    answer = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True
    ).strip()

    return answer


# WRAPPER
def answer_with_context(
    query: str,
    contexts: List[Dict],
    tokenizer,
    model
) -> Dict:

    answer = generate_answer(
        query=query,
        contexts=contexts,
        tokenizer=tokenizer,
        model=model
    )

    return {
        "query": query,
        "answer": answer,
        "contexts": contexts
    }