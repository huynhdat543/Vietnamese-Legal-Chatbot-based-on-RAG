from typing import List, Dict, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# INIT MODEL
def init_model(model_name: str = "Qwen/Qwen2.5-7B-Instruct"):
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
Dựa vào các đoạn luật liên quan, hãy phân tích và chọn ra DUY NHẤT 1 đoạn luật có nội dung phù hợp và đúng nhất để trả lời cho câu hỏi người dùng.
Sau khi đã chọn 1 đoạn luật phù hợp, tóm tắt nội dung của đoạn luật đó để trả lời cho câu hỏi người dùng và KHÔNG GIẢI THÍCH gì thêm.
Câu trả lời cuối cùng chỉ là nội dung đoạn luật bạn đã tóm tắt, TUYỆT ĐỐI KHÔNG TRÍCH LẠI TOÀN BỘ ĐOẠN LUẬT và không có thêm thông tin nào khác như đoạn luật số mấy,...
Nếu bạn không thấy đoạn luật nào phù hợp để trả lời thì hãy nói "Chưa đủ năng lực để trả lời câu hỏi đó".

Ví dụ:
Câu hỏi:
Người lao động có được đơn phương chấm dứt hợp đồng không?

Các đoạn luật liên quan:
[Đoạn luật 1]:
Mỗi bên đều có quyền đơn phương chấm dứt hợp đồng lao động trước khi hết thời hạn báo trước nhưng phải thông báo bằng văn bản và phải được bên kia đồng ý.
[Đoạn luật 2]:
Đơn phương chấm dứt hợp đồng lao động trái pháp luật là trường hợp chấm dứt hợp đồng lao động không đúng quy định tại các điều 35, 36 và 37 của Bộ luật này.
[Đoạn luật 3]:
3. Hai bên thỏa thuận trong hợp đồng lao động về hình thức trả lương, kỳ hạn trả lương, thời gian làm việc hằng ngày, chỗ ở.
[Đoạn luật 4]:
Lao động nữ mang thai nếu có xác nhận của cơ sở khám bệnh, chữa bệnh có thẩm quyền về việc tiếp tục làm việc sẽ ảnh hưởng xấu tới thai nhi thì có quyền đơn phương chấm dứt hợp đồng lao động hoặc tạm hoãn thực hiện hợp đồng lao động.
[Đoạn luật 5]:
Đơn vị sử dụng lao động không được đơn phương chấm dứt hợp đồng lao động, hợp đồng làm việc, sa thải, buộc thôi việc hoặc thuyên chuyển công tác đối với cán bộ công đoàn không chuyên trách nếu không có ý kiến thỏa thuận bằng văn bản của Ban chấp hành công đoàn cơ sở hoặc Ban chấp hành công đoàn cấp trên trực tiếp cơ sở.

Câu trả lời cuối cùng:
Người lao động được quyền đơn phương chấm dứt hợp đồng lao động trước khi hết thời hạn nhưng phải thông báo bằng văn bản và phải được bên kia đồng ý


Bây giờ hãy dựa vào câu hỏi và các đoạn luật sau đây để trả lời câu hỏi:
Câu hỏi:
{query}

Các đoạn luật liên quan:
{contexts}

Hãy sinh câu trả lời cuối cùng:
"""

# FORMAT CONTEXT
def build_context_text(contexts: List[Dict]):
    """
    Chuyển list context -> text prompt

    Expected format mỗi phần tử:
    {
        "text": "...",
        ...
    }
    """

    formatted_contexts = []

    for idx, ctx in enumerate(contexts, start=1):

        if isinstance(ctx, dict):
            text = ctx.get("text", "").strip()
        else:
            text = str(ctx).strip()

        if not text:
            continue

        formatted_contexts.append(
            f"[Đoạn luật {idx}]:\n{text}"
        )

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

    prompt = ANSWER_PROMPT.format(
        query=query,
        contexts=context_text
    )

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

    with torch.inference_mode():

        outputs = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
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