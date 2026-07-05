from rewrite_qwen_3B import init_model, rewrite_with_original
from retrieval_hybrid import search_topk_hybrid
from rrf import rewrite_rrf_fusion, rerank_rrf_fusion
from rerank import RerankModel
from extract import ModelExtractor
from gen_ansText_qwen_3B import answer_with_context
# from gen_ansChoose_qwen_3B import answer_with_context


RERANK_MODEL_ViNLI = "huynhdat543/VietNamese_law_rerank_v1"
RERANK_MODEL_COMBINE = "huynhdat543/VietNamese_law_rerank_v3"
reranker_ViNLI = RerankModel(RERANK_MODEL_ViNLI)
reranker_combine = RerankModel(RERANK_MODEL_COMBINE)

extractor = ModelExtractor(model_id="lmka05/Model_finetune_bert", subfolder_name="checkpoint-6795")

tokenizer, model = init_model()