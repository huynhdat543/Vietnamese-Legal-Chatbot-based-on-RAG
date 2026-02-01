import streamlit as st
import sys
import os
import google.generativeai as genai

MODULE_PATH = "/content/drive/MyDrive/DS310/MODULE"
if MODULE_PATH not in sys.path:
    sys.path.append(MODULE_PATH)
try:
    from rewrite_best import rewrite_with_original, init_gemini
    from retrieval_chroma import load_chroma_db, search_topk_by_text
    from rrf import rrf_fusion
    from rerank_GPU import RerankModel
    from extract_model import Model_Extractor
    from call_api import generate_final_answer
except ImportError as e:
    st.error(f"Lỗi không tìm thấy module: {e}. Hãy kiểm tra lại đường dẫn Drive.")
    st.stop()

st.set_page_config(page_title="Legal AI Chatbot", layout="wide")
st.title("⚖️ Trợ Lý Pháp Lý ")

with st.sidebar:
    st.header("Cấu hình")
    api_key = st.text_input("Gemini API Key", type="password")
    
    st.subheader("Tham số")
    top_k_retrieval = st.number_input("Retrieval K", value=100)
    top_n_rrf = st.number_input("RRF Top N", value=50)
    top_k_rerank = st.slider("Rerank Top K", 1, 10, 5)

@st.cache_resource
def load_global_resources():
    db_path = "/content/drive/MyDrive/DS310/Vector Database"
    print("Đang load ChromaDB...")
    load_chroma_db(db_path, collection_name="langchain")

    print("Đang load Rerank Model...")
    rerank_path = "huynhdat543/VietNamese_law_rerank"
    reranker = RerankModel(rerank_path)

    print("Đang load Extract Model...")
    extract_path = "lmka05/Model_finetune_bert"
    subfolder = "checkpoint-6795"
    extractor = Model_Extractor(extract_path, subfolder)
    
    return reranker, extractor

try:
    with st.spinner("Đang tải Models & Database từ Drive..."):
        reranker, extractor = load_global_resources()
    st.success("Hệ thống đã sẵn sàng!")
except Exception as e:
    st.error(f"Lỗi khởi tạo: {e}")
    st.stop()

if api_key:
    genai.configure(api_key=api_key)
    init_gemini(api_key)
    gemini_model = genai.GenerativeModel('gemini-2.5-flash')

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "details" in msg:
             with st.expander("Chi tiết xử lý"):
                st.json(msg["details"])

if prompt := st.chat_input("Nhập câu hỏi luật..."):
    if not api_key:
        st.warning("Vui lòng nhập API Key trước.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        with st.spinner("Đang suy luận..."):
            queries = rewrite_with_original(prompt, num_queries=3)
            
            retrieval_lists = []
            for q in queries:
                results = search_topk_by_text(q, top_k=top_k_retrieval)
                retrieval_lists.append(results)
            
            topN = rrf_fusion(retrieval_lists, top_n=top_n_rrf)
            
            topK = reranker.rerank(prompt, topN, top_k=top_k_rerank)
            
            context_texts = [doc['text'] for doc in topK]
            extract_info = extractor.extract([prompt], context_texts)
            final_info = extract_info[0] if extract_info else ""
            
            answer = generate_final_answer(gemini_model, prompt, final_info)
            
            message_placeholder.markdown(answer)
            
            details = {
                "queries": queries,
                "top_docs": [d['text'][:100] for d in topK],
                "extracted": final_info
            }
            st.session_state.messages.append({
                "role": "assistant", 
                "content": answer,
                "details": details
            })