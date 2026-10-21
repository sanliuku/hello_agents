"""本地 RAG 模块（Faiss + ollama bge）"""

from .pipeline import (
    create_rag_pipeline,
    index_chunks,
    search_vectors,
    search_vectors_expanded,
)

__all__ = [
    "create_rag_pipeline",
    "index_chunks",
    "search_vectors",
    "search_vectors_expanded",
]
