import re
from typing import List, Optional
import google.generativeai as genai


def init_gemini(api_key: str):
    api_key = api_key

    if not api_key:
        raise ValueError("Gemini API key không được để trống")

    genai.configure(api_key=api_key)

QUERY_EXPANSION_PROMPT = """
Bạn đang hỗ trợ xây dựng hệ thống hỏi đáp pháp luật Việt Nam.

Nhiệm vụ:
- Nhận vào MỘT câu hỏi của người dân
- Sinh CHÍNH XÁC {num_queries} câu hỏi khác nhau
- Giữ NGUYÊN ý nghĩa pháp lý của hành vi
- Nhưng chuyển sang các cách diễn đạt thường gặp trong văn bản pháp luật Việt Nam

Hướng dẫn sinh câu hỏi:
- Ưu tiên chuyển cách nói đời thường → cách diễn đạt pháp lý
- Có thể mở rộng sang các khái niệm tương đương (cùng hành vi)
- Có thể dùng các thuật ngữ pháp luật
- Tránh chỉ thay từ đồng nghĩa bề mặt
- Không suy đoán thêm tình tiết không có trong câu hỏi gốc

Yêu cầu định dạng:
- Mỗi câu hỏi là MỘT câu hoàn chỉnh
- Không viết dở dang
- Không thêm giải thích
- Không đánh số, không gạch đầu dòng
- Mỗi câu trên một dòng riêng

Câu hỏi gốc:
"{query}"
"""

def generate_similar_queries(
    query: str,
    num_queries: int = 3,
    model_name: str = "gemini-2.5-flash",
    temperature: float = 0.6,
    max_output_tokens: int = 3000,
) -> List[str]:
    """
    Sinh các câu query tương tự từ Gemini

    Returns:
        List[str]: danh sách câu query tương tự
    """

    if not query or not query.strip():
        raise ValueError("Query đầu vào không hợp lệ")

    model = genai.GenerativeModel(
        model_name=model_name,
        generation_config={
            "temperature": temperature,
            "top_p": 0.6,
            "max_output_tokens": max_output_tokens,
        },
    )

    prompt = QUERY_EXPANSION_PROMPT.format(
        query=query,
        num_queries=num_queries
    )

    response = model.generate_content(prompt)

    if not response or not response.text:
        return []

    queries: List[str] = []

    for line in response.text.split("\n"):
        line = line.strip()
        if not line:
            continue

        line = re.sub(r"^[0-9\-\*\•\)]\s*", "", line)

        queries.append(line)

    return queries[:num_queries]


def rewrite_with_original(query: str, num_queries: int = 3) -> List[str]:
    """
    Trả về:
    - query gốc
    - + các query sinh thêm
    (để truyền sang bước retrieval)
    """
    expanded = generate_similar_queries(query, num_queries=num_queries)
    return [query] + expanded
