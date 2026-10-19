"""本地版记忆模块测试

底层后端全部换成本地实现：
- 嵌入     : 本地 ollama 的 bge-m3（1024 维）
- 向量库   : faiss（本地文件持久化）
- 图库     : 本地 neo4j（未启动时自动降级为空实现）

运行方式（在 hello_agents_learn 目录下）：
    python test_memory.py

覆盖内容：
1. ollama bge 嵌入（打印维度）
2. faiss 向量写入 / 检索 / 落盘重载
3. MemoryManager 端到端：semantic + episodic 写入 -> 检索 -> 打印命中
4. 本地 neo4j 图库（可用则演示实体/关系，不可用则降级）
"""

import os
import sys

# 让本目录优先被导入，确保使用本地 memory 包（而不是 site-packages 里的包）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ---- 底层后端配置：全部指向本地 ----
os.environ.setdefault("EMBED_MODEL_TYPE", "ollama")
os.environ.setdefault("EMBED_MODEL_NAME", "bge-m3")
os.environ.setdefault("EMBED_BASE_URL", "http://localhost:11434")

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(_HERE, "memory_test_data")
os.environ.setdefault("FAISS_PERSIST_DIR", os.path.join(_DATA_DIR, "faiss"))

import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

from memory import MemoryManager, MemoryConfig
from memory.embedding import get_text_embedder
from memory.storage.faiss_store import FaissVectorStore


def section(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def test_embedding():
    """1) 本地 ollama bge 嵌入"""
    section("1) 嵌入模型（本地 ollama bge）")
    embedder = get_text_embedder()
    vec = embedder.encode("记忆系统的本地向量化测试")
    print(f"嵌入后端 : {type(embedder).__name__}  模型: {getattr(embedder, 'model_name', '?')}")
    print(f"向量维度 : {len(vec)}")
    assert len(vec) > 0, "嵌入向量为空"
    return embedder


def test_faiss_reload():
    """2) faiss 写入 / 检索 / 落盘重载"""
    section("2) Faiss 落盘与重载")

    persist_dir = os.path.join(_DATA_DIR, "faiss_reload_demo")
    store = FaissVectorStore(
        collection_name="reload_demo", vector_size=4,
        persist_dir=persist_dir, auto_persist=True,
    )
    store.clear_collection()
    store.add_vectors(
        vectors=[[1, 0, 0, 0], [0, 1, 0, 0], [0.9, 0.1, 0, 0]],
        metadata=[{"memory_id": "a"}, {"memory_id": "b"}, {"memory_id": "c"}],
        ids=["a", "b", "c"],
    )
    print(f"写入条数 : {store.get_collection_info().get('vectors_count')}")
    print(f"落盘文件 : {store._index_path} 存在={os.path.exists(store._index_path)}")
    assert store.get_collection_info().get("vectors_count") == 3

    # 新建实例，从磁盘自动重载
    reloaded = FaissVectorStore(
        collection_name="reload_demo", vector_size=4,
        persist_dir=persist_dir, auto_load=True,
    )
    hits = reloaded.search_similar([1, 0, 0, 0], limit=2)
    print(f"重载条数 : {reloaded.get_collection_info().get('vectors_count')}")
    print(f"重载检索 : {[(h['id'], round(h['score'], 4)) for h in hits]}")
    assert reloaded.get_collection_info().get("vectors_count") == 3, "重载后条数不一致"
    assert hits and hits[0]["id"] == "a", "重载检索结果不正确"


def test_memory_end_to_end():
    """3) MemoryManager 端到端：semantic + episodic"""
    section("3) MemoryManager 端到端（semantic + episodic）")

    config = MemoryConfig(storage_path=os.path.join(_DATA_DIR, "sqlite"))
    manager = MemoryManager(
        config=config,
        user_id="test_user",
        enable_working=True,
        enable_episodic=True,
        enable_semantic=True,
        enable_perceptual=False,
    )

    # 清空历史数据，保证测试可重复
    manager.memory_types["semantic"].clear()
    manager.memory_types["episodic"].clear()

    # 语义记忆（概念 / 知识类）
    semantic_texts = [
        "向量数据库是一种用于存储和检索高维向量的系统，常用于语义检索。",
        "Faiss 是 Facebook 开源的高效相似度检索库，支持十亿级向量。",
        "知识图谱用实体和关系描述现实世界中的概念及其联系。",
        "Ollama 可以在本地运行大语言模型和嵌入模型，例如 bge-m3。",
    ]
    for text in semantic_texts:
        manager.add_memory(text, memory_type="semantic", auto_classify=False)
    print(f"已写入 semantic: {len(semantic_texts)} 条")

    # 情景记忆（交互事件）
    episodic_texts = [
        "今天我运行了 test_memory.py，验证了本地 faiss 向量库检索。",
        "昨天我把 embedding 从 dashscope 换成了本地 ollama 的 bge-m3。",
    ]
    for i, text in enumerate(episodic_texts):
        manager.add_memory(
            text, memory_type="episodic", auto_classify=False,
            metadata={"session_id": f"s{i}", "importance": 0.6},
        )
    print(f"已写入 episodic: {len(episodic_texts)} 条")

    # 语义检索
    query = "本地向量检索库是什么？"
    print(f"\n[语义检索] query = {query}")
    hits = manager.retrieve_memories(query, memory_types=["semantic"], limit=3)
    for i, m in enumerate(hits, 1):
        score = m.metadata.get("combined_score", m.metadata.get("vector_score", 0.0))
        print(f"  #{i} score={float(score):.4f} | {m.content}")
    assert hits, "语义检索无结果"

    # 情景检索
    query2 = "昨天我换了什么 embedding？"
    print(f"\n[情景检索] query = {query2}")
    hits2 = manager.retrieve_memories(query2, memory_types=["episodic"], limit=3)
    for i, m in enumerate(hits2, 1):
        score = m.metadata.get("relevance_score", m.metadata.get("vector_score", 0.0))
        print(f"  #{i} score={float(score):.4f} | {m.content}")
    assert hits2, "情景检索无结果"

    # 统计
    stats = manager.get_memory_stats()
    print(f"\n[统计] total_memories = {stats['total_memories']}")
    for mt, s in stats["memories_by_type"].items():
        store_type = (s.get("vector_store") or {}).get("store_type", "n/a")
        print(f"  - {mt}: count={s.get('count')} store={store_type}")

    # 本地图库状态
    semantic = manager.memory_types["semantic"]
    available = getattr(semantic.graph_store, "available", False)
    print(f"\n[图库] 本地Neo4j可用: {available}")
    if available:
        semantic.graph_store.add_entity("e_demo", "本地向量库", "CONCEPT", {"note": "demo"})
        rels = semantic.graph_store.get_entity_relationships("e_demo")
        print(f"  实体关系数: {len(rels)}")
    else:
        print("  （未启动本地 neo4j，图功能已自动降级为空实现，向量检索不受影响）")


if __name__ == "__main__":
    test_embedding()
    test_faiss_reload()
    test_memory_end_to_end()
    print("\n✅ 全部测试完成")
