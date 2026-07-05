from .hybrid_search import hybrid_search
from pyvi import ViTokenizer


def search_topk_hybrid(
    query: str,
    top_k: int
):
    """
    Adapter: Hybrid Search -> pipeline cũ
    """

    results = hybrid_search(query=query, top_k=top_k)

    converted = []

    for r in results:

        payload = r.get("payload", {})

        if payload is None:
          continue

        metadata = payload.get("metadata", {})
        context = payload.get("context", "")
        article_title = metadata.get("article_title", "")
        content = payload.get("content", "")
        # nối text
        full_text = f"{context} {article_title} {content}".strip()

        # Thêm tách từ của pyvi
        full_text_tok = ViTokenizer.tokenize(full_text)

        converted.append({
            "id": r.get("node_id"),
            "metadata": metadata,
            "text": full_text,
            "text_tok": full_text_tok,
            # retrieval info
            "retrieval_rrf_score": r.get("rrf_score", 0),
            # optional
            "level": payload.get("level"),
            # giữ raw payload để debug
            "payload": payload
        })


    return converted