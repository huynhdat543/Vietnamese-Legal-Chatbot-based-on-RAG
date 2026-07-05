from typing import List, Dict


def rewrite_rrf_fusion(
    results_per_query: List[List[Dict]],
    k: int = 60,
    top_n: int = 50,
) -> List[Dict]:

    fused = {}

    for results in results_per_query:

        for rank, item in enumerate(results):

            key = item["id"]

            # công thức tính RRF score
            score = 1 / (k + rank + 1)
            
            # chưa có thì add nguyên document
            if key not in fused:
                fused[key] = item
                fused[key]["rewrite_rrf_score"] = 0.0

            # cộng score
            fused[key]["rewrite_rrf_score"] += score

    fused_list = list(fused.values())

    # sort theo rewrite_rrf_score
    fused_list.sort(
        key=lambda x: x["rewrite_rrf_score"],
        reverse=True
    )

    return fused_list[:top_n]



def rerank_rrf_fusion(
    rerank_results: List[List[Dict]],
    k: int = 60,
    top_n: int = 10,
) -> List[Dict]:

    fused = {}

    for results in rerank_results:

        for rank, item in enumerate(results):

            key = item["id"]

            # công thức RRF
            score = 1 / (k + rank + 1)

            # chưa có thì add nguyên document
            if key not in fused:
                fused[key] = item
                fused[key]["rerank_rrf_score"] = 0.0

            # cộng score
            fused[key]["rerank_rrf_score"] += score

    fused_list = list(fused.values())

    # sort theo rerank_rrf_score
    fused_list.sort(
        key=lambda x: x["rerank_rrf_score"],
        reverse=True
    )

    return fused_list[:top_n]
