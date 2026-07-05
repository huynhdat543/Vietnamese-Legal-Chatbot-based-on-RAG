from qdrant_client.models import Filter
import uuid
from config.settings import settings
from database import db_manager
from .dense_search import dense_searcher
from .sparse_search import sparse_searcher


def _dense_search(query: str,top_k: int,query_filter: Filter | None) -> list[dict]:
    vector = dense_searcher.embed_query(query)
    results = db_manager.get_client().query_points(
        collection_name=settings.COLLECTION_NAME,
        query=vector,
        limit=top_k,
        with_payload=True,
        query_filter=query_filter
    )

    return [
        {
            "node_id": r.payload.get("node_id"),
            "score":   r.score
        }
        for r in results.points
    ]


def _sparse_search(query: str,top_k: int) -> list[dict]:
    scores, ids = sparse_searcher.get_scores(query)
    top_indices = sorted(range(len(scores)),key=lambda i: scores[i],reverse=True)[:top_k]

    return [
        {
            "node_id": ids[i],
            "score":   float(scores[i])
        }
        for i in top_indices if scores[i] > 0
    ]


def _rrf(
    dense_results: list[dict],
    sparse_results: list[dict],
    k: int = settings.RRF_K,
    dense_weight: float = settings.DENSE_WEIGHT,
    sparse_weight: float = settings.SPARSE_WEIGHT) -> list[tuple]:

    scores: dict[str, float] = {}
    for rank, item in enumerate(dense_results):
        node_id = item["node_id"]
        scores[node_id] = scores.get(node_id, 0) + dense_weight * (1 / (k + rank + 1))

    for rank, item in enumerate(sparse_results):
        node_id = item["node_id"]
        scores[node_id] = scores.get(node_id, 0) + sparse_weight * (1 / (k + rank + 1))

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def _fetch_payloads(node_ids: list[str]) -> dict[str, dict]:
    uuid_ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, nid)) for nid in node_ids]
 
    points = db_manager.get_client().retrieve(
        collection_name=settings.COLLECTION_NAME,
        ids=uuid_ids,
        with_payload=True
    )

    return {p.payload["node_id"]: p.payload for p in points}

def hybrid_search(
    query: str,
    top_k: int = settings.TOP_K,
    rrf_k: int = settings.RRF_K,
    dense_weight: float = settings.DENSE_WEIGHT,
    sparse_weight: float = settings.SPARSE_WEIGHT,
    query_filter: Filter | None = None) -> list[dict]:

    dense_results  = _dense_search(query, top_k=top_k * 2, query_filter=query_filter)
    sparse_results = _sparse_search(query, top_k=top_k * 2)
 
    ranked = _rrf(dense_results, sparse_results, k=rrf_k,dense_weight=dense_weight, sparse_weight=sparse_weight)
 
    top_ranked = ranked[:top_k]
    top_node_ids = [node_id for node_id, _ in top_ranked]
 
    payload_map = _fetch_payloads(top_node_ids)
 
    return [
        {
            "node_id":   node_id,
            "rrf_score": rrf_score,
            "payload":   payload_map.get(node_id)
        }
        for node_id, rrf_score in top_ranked
    ]