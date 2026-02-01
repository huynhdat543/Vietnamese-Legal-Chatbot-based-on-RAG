from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings

def create_embedding_model(
    model_name: str = "bkai-foundation-models/vietnamese-bi-encoder",
    device: str = "cuda",
    normalize: bool = True
):
    embeddings_model = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": normalize}
    )
    return embeddings_model


def create_semantic_chunker(
    embeddings_model,
    buffer_size: int = 3,
    breakpoint_threshold_type: str = "gradient",
    breakpoint_threshold_amount: float = 0.85,
    sentence_split_regex: str = r"(?<=[.?!])\s+(?!\d+\.\s)",
    min_chunk_size: int = 500
):

    chunker = SemanticChunker(
        embeddings=embeddings_model,
        buffer_size=buffer_size,
        breakpoint_threshold_type=breakpoint_threshold_type,
        breakpoint_threshold_amount=breakpoint_threshold_amount,
        sentence_split_regex=sentence_split_regex,
        min_chunk_size=min_chunk_size
    )
    return chunker