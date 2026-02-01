from typing import List, Dict, Optional
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


VECTOR_DB = None

def load_chroma_db(
    db_path: str,
    collection_name: str = "langchain",
    device: str = "cpu",
):
    global VECTOR_DB

    embeddings_model = HuggingFaceEmbeddings(
        model_name="bkai-foundation-models/vietnamese-bi-encoder",
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )

    VECTOR_DB = Chroma(
        persist_directory=db_path,
        embedding_function=embeddings_model,
        collection_name=collection_name,
    )

    print(f"[OK] Loaded ChromaDB — {VECTOR_DB._collection.count()} chunks")


def search_topk_by_text(
    query: str,
    top_k: int = 100,
) -> List[Dict]:

    if VECTOR_DB is None:
        raise RuntimeError("ChromaDB chưa được load. Hãy gọi load_chroma_db() trước.")

    docs_and_scores = VECTOR_DB.similarity_search_with_score(query, k=top_k)

    results: List[Dict] = []

    for rank, (doc, score) in enumerate(docs_and_scores):

        payload = doc.metadata or {}

        results.append({
            "id": str(payload.get("id", rank)), 
            "text": doc.page_content,
            "metadata": payload,
            "rank": rank,
            "sim_score": float(-score),
        })

    return results
