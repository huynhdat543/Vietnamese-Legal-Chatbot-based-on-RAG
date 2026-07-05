import streamlit as st
import sys
import json
import os

# Setup module path
module_path = "/content/drive/MyDrive/se365/flow_SE365/module"

sys.path.insert(0, module_path)
os.chdir(module_path)


# import pipeline
from pipeline import qa_pipeline


# Streamlit UI
st.set_page_config(
    page_title="Legal RAG QA",
    page_icon="⚖️",
    layout="wide"
)


st.title("⚖️ Hệ thống hỏi đáp pháp luật sử dụng RAG")


st.write(
    """
    Nhập câu hỏi pháp luật và các lựa chọn đáp án.
    Hệ thống sẽ truy hồi văn bản pháp luật và đưa ra đáp án.
    """
)


# Input câu hỏi
question = st.text_area(
    "Câu hỏi:",
    height=150,
    placeholder="Nhập câu hỏi pháp luật..."
)



# # Input answers

# answers = st.text_area(
#     "Các lựa chọn:",
#     height=120,
#     placeholder="""
# Ví dụ:
# ['1 năm đến 5 năm tù',
#  '5 năm đến 10 năm tù',
#  '10 năm đến 15 năm tù',
#  'Chung thân']
# """
# )



# button
if st.button("🔍 Tìm đáp án"):


    # if question.strip()=="" or answers.strip()=="":
    #     st.warning("Vui lòng nhập đầy đủ câu hỏi và đáp án")


    # else:

        with st.spinner("Đang chạy RAG pipeline..."):


            result = qa_pipeline(

                user_query=question,

                # answers=answers,


                # rewrite
                use_rewrite=True,
                num_rewrites=3,


                # retrieval
                retrieval_top_k=100,


                # RRF
                use_rewrite_rrf=True,
                rewrite_rrf_top_n=50,


                # rerank
                use_rerank=True,


                # dual rerank
                use_dual_rerank=True,
                rerank_top_k=10,


                # extract
                use_extract=False,


                print_result=False
            )


        st.success("Hoàn thành")


        st.subheader("Kết quả")


        if isinstance(result, dict):
            # Hiển thị từng field rõ ràng
            for key, value in result.items():
                st.markdown(f"**{key}**")
                st.write(value)
        else:
            # Nếu result là string thuần
            st.write(result)