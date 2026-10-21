"""本地 RAG 管道（ollama bge 嵌入 + Faiss 向量库）

与 hello_agents.memory.rag 的接口保持一致，仅替换底层：
- 嵌入：复用 memory.embedding 的统一嵌入（默认 ollama bge-m3）
- 向量库：memory.storage.faiss_store.FaissVectorStore（本地落盘）
- 文档解析与分块：复用 hello_agents 的 MarkItDown 解析逻辑（与存储无关）

对外暴露 create_rag_pipeline()，返回的字典包含：
store / namespace / add_documents / search / search_advanced / get_stats
"""

from typing import Any, Dict, List, Optional
import os
import time

from ..embedding import get_text_embedder, get_dimension
from ..storage.faiss_store import FaissConnectionManager

# 文档解析与分块是纯文本处理，与向量库无关，直接复用原实现
from hello_agents.memory.rag.pipeline import (
    load_and_chunk_texts,
    _preprocess_markdown_for_embedding,
)


# ==================
# 索引
# ==================

def index_chunks(
    store,
    chunks: List[Dict],
    rag_namespace: str = "default",
    batch_size: int = 16,
) -> int:
    """把分块向量化并写入 Faiss

    Returns:
        成功写入的向量条数
    """
    if not chunks:
        return 0

    embedder = get_text_embedder()
    texts = [_preprocess_markdown_for_embedding(c["content"]) for c in chunks]

    vectors: List[List[float]] = []
    for i in range(0, len(texts), batch_size):
        part = texts[i:i + batch_size]
        try:
            part_vecs = embedder.encode(part)
        except Exception as e:
            print(f"[RAG] 批次 {i} 嵌入失败: {e}")
            continue
        for v in part_vecs:
            if hasattr(v, "tolist"):
                v = v.tolist()
            vectors.append([float(x) for x in v])

    if len(vectors) != len(chunks):
        # 有批次失败时按实际成功数截断，避免向量与元数据错位
        chunks = chunks[:len(vectors)]
    if not vectors:
        return 0

    metas: List[Dict[str, Any]] = []
    ids: List[str] = []
    for ch in chunks:
        meta = {
            "memory_id": ch["id"],
            "user_id": "rag_user",
            "memory_type": "rag_chunk",
            "content": ch["content"],
            "data_source": "rag_pipeline",
            "rag_namespace": rag_namespace,
            "is_rag_data": True,
        }
        meta.update(ch.get("metadata", {}))
        metas.append(meta)
        ids.append(ch["id"])

    ok = store.add_vectors(vectors=vectors, metadata=metas, ids=ids)
    return len(vectors) if ok else 0


# ==================
# 检索
# ==================

def embed_query(query: str) -> List[float]:
    """把查询文本转为向量"""
    vec = get_text_embedder().encode(query)
    if hasattr(vec, "tolist"):
        vec = vec.tolist()
    if isinstance(vec, list) and vec and isinstance(vec[0], (list, tuple)):
        vec = vec[0]
    return [float(x) for x in vec]


def _rag_where(rag_namespace: Optional[str]) -> Dict[str, Any]:
    where: Dict[str, Any] = {
        "memory_type": "rag_chunk",
        "is_rag_data": True,
        "data_source": "rag_pipeline",
    }
    if rag_namespace:
        where["rag_namespace"] = rag_namespace
    return where


def _prompt_mqe(llm, query: str, n: int) -> List[str]:
    """多查询扩展：让 LLM 给出若干等价表述"""
    if llm is None:
        return [query]
    try:
        prompt = [
            {"role": "system", "content": "你是检索查询扩展助手。生成语义等价或互补的多样化查询。使用中文，简短，避免标点。"},
            {"role": "user", "content": f"原始查询：{query}\n请给出{n}个不同表述的查询，每行一个。"},
        ]
        text = llm.invoke(prompt) or ""
        outs = [ln.strip("- \t") for ln in text.splitlines() if ln.strip()]
        return outs[:n] or [query]
    except Exception:
        return [query]


def _prompt_hyde(llm, query: str) -> Optional[str]:
    """HyDE：让 LLM 先写一段假想答案，用其向量检索"""
    if llm is None:
        return None
    try:
        prompt = [
            {"role": "system", "content": "根据用户问题，先写一段可能的答案性段落，用于向量检索的查询文档（不要分析过程）。"},
            {"role": "user", "content": f"问题：{query}\n请直接写一段中等长度、客观、包含关键术语的段落。"},
        ]
        return llm.invoke(prompt)
    except Exception:
        return None


def search_vectors(
    store,
    query: str,
    top_k: int = 8,
    rag_namespace: Optional[str] = None,
    score_threshold: Optional[float] = None,
) -> List[Dict]:
    """基础向量检索"""
    if not query:
        return []
    return store.search_similar(
        query_vector=embed_query(query),
        limit=top_k,
        score_threshold=score_threshold,
        where=_rag_where(rag_namespace),
    )


def search_vectors_expanded(
    store,
    query: str,
    top_k: int = 8,
    rag_namespace: Optional[str] = None,
    score_threshold: Optional[float] = None,
    llm=None,
    enable_mqe: bool = False,
    mqe_expansions: int = 2,
    enable_hyde: bool = False,
    candidate_pool_multiplier: int = 4,
) -> List[Dict]:
    """带查询扩展的检索（MQE + HyDE），多路结果按最高分去重合并"""
    if not query:
        return []

    expansions: List[str] = [query]
    if enable_mqe and mqe_expansions > 0:
        expansions.extend(_prompt_mqe(llm, query, mqe_expansions))
    if enable_hyde:
        hyde_text = _prompt_hyde(llm, query)
        if hyde_text:
            expansions.append(hyde_text)

    uniq: List[str] = []
    for e in expansions:
        if e and e not in uniq:
            uniq.append(e)

    pool = max(top_k * candidate_pool_multiplier, 20)
    per = max(1, pool // max(1, len(uniq)))

    where = _rag_where(rag_namespace)
    agg: Dict[str, Dict] = {}
    for q in uniq:
        hits = store.search_similar(
            query_vector=embed_query(q),
            limit=per,
            score_threshold=score_threshold,
            where=where,
        )
        for h in hits:
            mid = h.get("metadata", {}).get("memory_id", h.get("id"))
            score = float(h.get("score", 0.0))
            if mid not in agg or score > float(agg[mid].get("score", 0.0)):
                agg[mid] = h

    merged = sorted(agg.values(), key=lambda x: float(x.get("score", 0.0)), reverse=True)
    return merged[:top_k]


# ==================
# 工厂
# ==================

def create_rag_pipeline(
    collection_name: str = "hello_agents_rag_vectors",
    rag_namespace: str = "default",
    persist_dir: Optional[str] = None,
    llm=None,
) -> Dict[str, Any]:
    """创建本地 RAG 管道（Faiss + 统一嵌入）

    Args:
        collection_name: Faiss 集合名
        rag_namespace: 知识库命名空间（写入元数据，用于隔离检索）
        persist_dir: Faiss 落盘根目录，默认 FAISS_PERSIST_DIR 或 ./memory_data/faiss
        llm: 可选的 LLM 实例，供 MQE/HyDE 使用
    """
    dimension = get_dimension(1024)
    store = FaissConnectionManager.get_instance(
        collection_name=collection_name,
        vector_size=dimension,
        distance="cosine",
        persist_dir=persist_dir,
    )

    def add_documents(file_paths: List[str], chunk_size: int = 800, chunk_overlap: int = 100) -> int:
        chunks = load_and_chunk_texts(
            paths=file_paths,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            namespace=rag_namespace,
            source_label="rag",
        )
        return index_chunks(store=store, chunks=chunks, rag_namespace=rag_namespace)

    def search(query: str, top_k: int = 8, score_threshold: Optional[float] = None):
        return search_vectors(
            store=store,
            query=query,
            top_k=top_k,
            rag_namespace=rag_namespace,
            score_threshold=score_threshold,
        )

    def search_advanced(
        query: str,
        top_k: int = 8,
        enable_mqe: bool = False,
        enable_hyde: bool = False,
        score_threshold: Optional[float] = None,
    ):
        return search_vectors_expanded(
            store=store,
            query=query,
            top_k=top_k,
            rag_namespace=rag_namespace,
            score_threshold=score_threshold,
            llm=llm,
            enable_mqe=enable_mqe,
            enable_hyde=enable_hyde,
        )

    def get_stats():
        stats = store.get_collection_stats()
        # 只统计当前命名空间的分块数
        try:
            stats["namespace_chunks"] = len([
                sid for sid, meta in store._meta.items()
                if meta.get("rag_namespace") == rag_namespace and meta.get("is_rag_data")
            ])
        except Exception:
            pass
        return stats

    return {
        "store": store,
        "namespace": rag_namespace,
        "add_documents": add_documents,
        "search": search,
        "search_advanced": search_advanced,
        "get_stats": get_stats,
    }
