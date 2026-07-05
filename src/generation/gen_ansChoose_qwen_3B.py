from typing import List, Dict, Tuple

import torch
import ast
import re
from transformers import AutoTokenizer, AutoModelForCausalLM


# =====================
# 1. INIT MODEL
# =====================

def init_model(model_name: str = "Qwen/Qwen2.5-3B-Instruct"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )

    return tokenizer, model


# =====================
# 2. PROMPT
# =====================

ANSWER_PROMPT = """
Bạn là trợ lý AI hỗ trợ hỏi đáp pháp luật Việt Nam.

Nhiệm vụ:
Hãy phân tích, chọn lọc và tổng hợp tất cả các thông tin có liên quan nhất từ các đoạn luật để chọn câu trả lời cho câu hỏi người dùng chính xác nhất trong các lựa chọn.
Câu trả lời cuối cùng phải chỉ là một số từ 0 đến {max_index} trong các các lựa chọn,
không giải thích bất cứ gì thêm.

Các đoạn luật:
{contexts}

Câu hỏi:
{query}

Các lựa chọn:
{answers}

Đáp án:
"""


# ANSWER_PROMPT = """
# Bạn là trợ lý AI hỗ trợ hỏi đáp pháp luật Việt Nam.

# Nhiệm vụ:
# Dựa vào các đoạn luật để trả lời cho câu hỏi người dùng chính xác nhất.
# Câu trả lời cuối cùng phải chỉ là một số từ 0 đến {max_index} trong các các lựa chọn,
# không giải thích bất cứ gì thêm.

# Các đoạn luật:
# {contexts}

# Câu hỏi:
# {query}

# Các lựa chọn:
# {answers}

# Đáp án:
# """


# =====================
# 3. FORMAT CONTEXT
# =====================

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


def format_answer(answers):
    answers = ast.literal_eval(answers)
    text = ""
    for i, ans in enumerate(answers):
        text += f"{i}. {ans}\n"

    return text.strip()


# =====================
# 4. GENERATE ANSWER
# =====================

def generate_answer(
    query: str,
    answers: str,
    contexts: List[Dict],
    tokenizer,
    model,
    max_new_tokens: int = 5,
) -> str:
    if not query or not query.strip():
        raise ValueError("Query không hợp lệ")

    if not contexts:
        return -1

    context_text = build_context_text(contexts)
    # answers = format_answer(answers)
    # num_answers = len(ast.literal_eval(answers))
    raw_answers = ast.literal_eval(answers)
    num_answers = len(raw_answers)-1
    answers = format_answer(answers)
    
    prompt = ANSWER_PROMPT.format(query=query, answers=answers, contexts=context_text, max_index=num_answers)

    messages = [
        {
            "role": "system",
            "content": (
                "Bạn là chuyên gia tư vấn pháp luật Việt Nam. "
                "Chỉ trả lời cho câu hỏi người dùng bằng một số tương ứng trong các lựa chọn."
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

    # lấy số trong output
    match = re.search(r"\d+", answer)

    if match:
        return int(match.group())
    else:
        return -1

    # return answer


# =====================
# 5. WRAPPER
# =====================

def answer_with_context(
    query: str,
    answers: str,
    contexts: List[Dict],
    tokenizer,
    model
) -> Dict:

    answer = generate_answer(
        query=query,
        answers=answers,
        contexts=contexts,
        tokenizer=tokenizer,
        model=model
    )

    return {
        "query": query,
        "answer": answer,
        "contexts": contexts
    }