from setup import *
import torch
import gc

def qa_pipeline(
    user_query: str,

    # rewrite
    use_rewrite: bool = True,
    num_rewrites: int = 3,

    # retrieval
    retrieval_top_k: int = 100,

    # rewrite rrf
    use_rewrite_rrf: bool = True,
    rewrite_rrf_top_n: int = 50,

    # rerank
    use_rerank: bool = True,

    # dual rerank
    use_dual_rerank: bool = True,

    rerank_top_k: int = 10,

    # extract
    use_extract: bool = True,

    # print result
    print_result: bool = True
):

    # 1. REWRITE
    if use_rewrite:
        queries = rewrite_with_original(
            query=user_query,
            tokenizer=tokenizer,
            model=model,
            num_queries=num_rewrites
        )
    else:
        queries = [user_query]
    print(f"[REWRITE] {len(queries)} queries")


    # 2. HYBRID SEARCH
    retrieval_list = []
    for q in queries:
        docs = search_topk_hybrid(
            q,
            top_k=retrieval_top_k
        )
        retrieval_list.append(docs)

    print(f"[HYBRID SEARCH] {len(retrieval_list)} done")


    # 3. REWRITE RRF
    if use_rewrite_rrf:
        topN = rewrite_rrf_fusion(
            results_per_query=retrieval_list,
            top_n=rewrite_rrf_top_n
        )
    else:
        # flatten list
        topN = []
        for docs in retrieval_list:
            topN.extend(docs)

    print(f"[REWRITE RRF] {len(topN)} docs")


    # 4. RERANK
    final_results = topN

    if use_rerank:
        # SINGLE RERANK
        if not use_dual_rerank:
            final_results = reranker_ViNLI.rerank(
                query=user_query,
                candidates=topN,
                top_k=rerank_top_k,
                score_field="rerank_score"
            )
            print(f"[RERANK] single done {len(final_results)}")

        # DUAL RERANK
        else:
            rerank_vinli = reranker_ViNLI.rerank(
                query=user_query,
                candidates=topN.copy(),
                top_k=None,
                score_field="rerank_score_vinli"
            )

            rerank_combine = reranker_combine.rerank(
                query=user_query,
                candidates=topN.copy(),
                top_k=None,
                score_field="rerank_score_combine"
            )

            final_results = rerank_rrf_fusion(
                rerank_results=[rerank_vinli, rerank_combine],
                top_n=rerank_top_k
            )
            print(f"[RERANK] dual done {len(final_results)}")


    # 5. EXTRACT
    if use_extract:
        final_results = extractor.extract(
            questions=queries,
            candidates=final_results
        )
        print("[EXTRACT] done")


    # 6. GENERATE ANSWER
    llm_answer = answer_with_context(
        query=user_query,
        contexts=final_results,
        tokenizer=tokenizer,
        model=model
    )
    print("[LLM] done")


    # PRINT RESULT
    if print_result:
        print("\n")
        print("="*50)
        print("\n")
        print("QUERY:", user_query)
        print("REWRITES:", queries)
        print("RESULT:")
        for i, d in enumerate(final_results, 1):
            print("-"*30)
            print("Rank:", i)
            print("ID:", d["id"])
            print("Hybrid_RRF:", d.get("retrieval_rrf_score"))
            print("Rewrite_RRF:", d.get("rewrite_rrf_score"))
            print("Rerank:", d.get("rerank_score"))
            print("Rerank_ViNLI:", d.get("rerank_score_vinli"))
            print("Rerank_Combine:", d.get("rerank_score_combine"))
            print("Rerank_RRF:", d.get("rerank_rrf_score"))
            print("Extract:", d.get("extract_score"))
            print("Text:", d["text"])
            print("Text_tokenized:", d["text_tok"])
            print("Extract Text:", d.get("text_extract"))


    # return {
    #     "query": user_query,
    #     "queries": queries,
    #     "final_results": final_results,
    #     "answer": llm_answer["answer"]
    # }

    result = llm_answer["answer"]
    
    queries = None
    retrieval_list = None
    topN = None
    final_results = None
    llm_answer = None

    gc.collect()
    torch.cuda.empty_cache()

    return result