import google.generativeai as genai

def generate_final_answer(gemini_model, user_query, extracted_contexts):
    if not extracted_contexts:
        return "Xin lỗi, hệ thống không tìm thấy thông tin trích xuất."
    
    context_str = "\n- ".join(extracted_contexts)
    
    prompt = f"""
    Bạn là trợ lý pháp lý AI. Dưới đây là các đoạn thông tin được trích xuất từ văn bản luật dựa trên nhiều cách hỏi khác nhau.
    Hãy tổng hợp chúng để trả lời câu hỏi gốc của người dùng.

    CÂU HỎI GỐC CỦA NGƯỜI DÙNG:
    "{user_query}"

    THÔNG TIN TRÍCH XUẤT ĐƯỢC:
    {context_str}

    YÊU CẦU:
    1. Trả lời trực tiếp vào câu hỏi.
    2. Nếu thông tin trích xuất có vẻ mâu thuẫn, hãy nêu rõ.
    3. Chỉ dùng thông tin được cung cấp.
    4. Nếu trong thông tin được trích xuất không chứa thông tin để trả lời, hãy xuất ra câu " Đoạn trích xuất không chứa thông tin để tôi trả lời"
    """
    
    print("--- Đang gửi context cho Gemini ---")
    response = gemini_model.generate_content(prompt)
    return response