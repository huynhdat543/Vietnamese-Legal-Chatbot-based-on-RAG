from typing import List, Dict

def rrf_fusion(
    results_per_query: List[List[Dict]],
    k: int = 60,
    top_n: int = 50,
) -> List[Dict]:

    fused = {}

    for results in results_per_query:

        for rank, item in enumerate(results):

            key = item["id"]

            score = 1 / (k + rank + 1)

            if key not in fused:
                fused[key] = {
                    "id": item["id"],
                    "text": item.get("text", ""),
                    "metadata": item.get("metadata", {}),
                    "rrf_score": 0.0,
                    "ranks": []
                }

            fused[key]["rrf_score"] += score
            fused[key]["ranks"].append(rank)

    fused_list = list(fused.values())

    fused_list.sort(key=lambda x: x["rrf_score"], reverse=True)

    return fused_list[:top_n]
