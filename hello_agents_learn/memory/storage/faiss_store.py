"""
Faiss向量库存储实现
使用本地 faiss 向量库替代 Qdrant，接口与 QdrantVectorStore 完全对齐

距离语义：IndexFlatIP + 向量 L2 归一化 => 内积 == 余弦相似度（分数越大越相似）
持久化：index.faiss + meta.json 落盘，支持进程重启后重载继续检索
"""

import logging
import os
import json
import time
import threading
import uuid
from typing import Dict, List, Optional, Any
import numpy as np
from datetime import datetime

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    faiss = None

logger = logging.getLogger(__name__)

class FaissConnectionManager:
    """Faiss连接管理器 - 防止重复连接和初始化"""
    _instances = {}  # key: (collection_name, persist_dir) -> FaissVectorStore instance
    _lock = threading.Lock()

    @classmethod
    def get_instance(
        cls,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        collection_name: str = "hello_agents_vectors",
        vector_size: int = 1024,
        distance: str = "cosine",
        timeout: int = 30,
        persist_dir: Optional[str] = None,
        **kwargs
    ) -> 'FaissVectorStore':
        """获取或创建Faiss实例（单例模式）"""
        base_dir = persist_dir or os.getenv("FAISS_PERSIST_DIR") or os.path.join("./memory_data", "faiss")
        collection_dir = os.path.join(base_dir, collection_name)
        key = (collection_name, collection_dir)

        if key not in cls._instances:
            with cls._lock:
                # 双重检查锁定
                if key not in cls._instances:
                    logger.debug(f"🔄 创建新的Faiss连接: {collection_name}")
                    cls._instances[key] = FaissVectorStore(
                        url=url,
                        api_key=api_key,
                        collection_name=collection_name,
                        vector_size=vector_size,
                        distance=distance,
                        timeout=timeout,
                        persist_dir=collection_dir,
                        **kwargs
                    )
                else:
                    logger.debug(f"♻️ 复用现有Faiss连接: {collection_name}")
        else:
            logger.debug(f"♻️ 复用现有Faiss连接: {collection_name}")

        return cls._instances[key]

class FaissVectorStore:
    """Faiss向量库存储实现"""

    def __init__(
        self,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        collection_name: str = "hello_agents_vectors",
        vector_size: int = 1024,
        distance: str = "cosine",
        timeout: int = 30,
        persist_dir: Optional[str] = None,
        auto_persist: bool = True,
        auto_load: bool = True,
        **kwargs
    ):
        """
        初始化Faiss向量存储

        Args:
            url: 兼容参数（本地faiss不使用）
            api_key: 兼容参数（本地faiss不使用）
            collection_name: 集合名称
            vector_size: 向量维度
            distance: 距离度量方式 (cosine, dot, euclidean) —— 本地实现以 cosine 为准
            timeout: 兼容参数（本地faiss不使用）
            persist_dir: 落盘目录
            auto_persist: 写入/删除后是否自动落盘
            auto_load: 初始化时是否尝试从磁盘重载
        """
        if not FAISS_AVAILABLE:
            raise ImportError(
                "faiss未安装。请运行: pip install faiss-cpu"
            )

        self.url = url
        self.api_key = api_key
        self.collection_name = collection_name
        self.vector_size = int(vector_size) if vector_size else None
        self.distance = (distance or "cosine").lower()
        self.timeout = timeout
        self.auto_persist = auto_persist

        self.persist_dir = persist_dir or os.getenv("FAISS_PERSIST_DIR") or os.path.join(
            "./memory_data", "faiss", collection_name
        )
        self._index_path = os.path.join(self.persist_dir, "index.faiss")
        self._meta_path = os.path.join(self.persist_dir, "meta.json")

        # 索引与元数据
        self.index = None
        self._meta: Dict[str, Dict[str, Any]] = {}   # str_id -> metadata
        self._id_map: Dict[int, str] = {}             # faiss 内部整数 id -> str_id
        self._str_to_int: Dict[str, int] = {}         # str_id -> faiss 内部整数 id
        self._next_int = 0
        self._lock = threading.RLock()

        if self.vector_size:
            self._new_index(self.vector_size)

        # 尝试从磁盘重载
        if auto_load and os.path.exists(self._index_path) and os.path.exists(self._meta_path):
            try:
                self.load()
                logger.info(f"✅ Faiss从磁盘重载: {self._index_path} (共 {self.index.ntotal} 条)")
            except Exception as e:
                logger.warning(f"⚠️ Faiss重载失败，使用空索引: {e}")

    def _new_index(self, dim: int):
        """创建内积索引（配合L2归一化等价余弦）"""
        base = faiss.IndexFlatIP(int(dim))
        self.index = faiss.IndexIDMap2(base)

    def _ensure_index(self, dim: int):
        """确保索引存在（首次写入时按实际维度惰性创建）"""
        if self.index is not None:
            return
        if not self.vector_size:
            self.vector_size = int(dim)
        self._new_index(self.vector_size)

    def _remove_sid(self, sid: str):
        """按字符串ID移除向量与元数据"""
        int_id = self._str_to_int.pop(sid, None)
        if int_id is not None:
            try:
                self.index.remove_ids(np.asarray([int_id], dtype=np.int64))
            except Exception:
                pass
            self._id_map.pop(int_id, None)
        self._meta.pop(sid, None)

    def add_vectors(
        self,
        vectors: List[List[float]],
        metadata: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ) -> bool:
        """
        添加向量到Faiss

        Args:
            vectors: 向量列表
            metadata: 元数据列表
            ids: 可选的ID列表

        Returns:
            bool: 是否成功
        """
        try:
            if not vectors:
                logger.warning("⚠️ 向量列表为空")
                return False

            # 生成ID（如果未提供）
            if ids is None:
                ids = [f"vec_{i}_{int(datetime.now().timestamp() * 1000000)}"
                       for i in range(len(vectors))]

            # 构建向量数据
            logger.info(f"[Faiss] add_vectors start: n_vectors={len(vectors)} n_meta={len(metadata)} collection={self.collection_name}")
            added = 0
            with self._lock:
                for vector, meta, point_id in zip(vectors, metadata, ids):
                    arr = np.asarray(vector, dtype=np.float32).reshape(-1)
                    self._ensure_index(arr.shape[0])
                    if arr.shape[0] != self.vector_size:
                        logger.warning(f"⚠️ 向量维度不匹配: 期望{self.vector_size}, 实际{arr.shape[0]}")
                        continue

                    # L2 归一化 => 内积退化为余弦相似度
                    norm = float(np.linalg.norm(arr))
                    if norm > 0:
                        arr = arr / norm

                    sid = str(point_id)
                    if sid in self._str_to_int:
                        # 幂等覆盖：同一ID重新写入
                        self._remove_sid(sid)
                    int_id = self._next_int
                    self._next_int += 1
                    self.index.add_with_ids(arr.reshape(1, -1), np.asarray([int_id], dtype=np.int64))
                    self._id_map[int_id] = sid
                    self._str_to_int[sid] = int_id

                    m = dict(meta or {})
                    now = int(datetime.now().timestamp())
                    m.setdefault("timestamp", now)
                    m.setdefault("added_at", now)
                    self._meta[sid] = m
                    added += 1

                if added and self.auto_persist:
                    self.persist()

            if added == 0:
                logger.warning("⚠️ 没有有效的向量点")
                return False

            logger.info(f"✅ 成功添加 {added} 个向量到Faiss")
            return True

        except Exception as e:
            logger.error(f"❌ 添加向量失败: {e}")
            return False

    def _allowed_ids(self, where: Optional[Dict[str, Any]]) -> Optional[set]:
        """按元数据标量字段过滤（等价 Qdrant 的 where 精确匹配）"""
        if not where:
            return None
        allowed = set()
        for sid, meta in self._meta.items():
            ok = True
            for key, val in where.items():
                if val is None:
                    continue
                if not isinstance(val, (str, int, float, bool)):
                    continue
                if meta.get(key) != val:
                    ok = False
                    break
            if ok:
                allowed.add(sid)
        return allowed

    def search_similar(
        self,
        query_vector: List[float],
        limit: int = 10,
        score_threshold: Optional[float] = None,
        where: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        搜索相似向量

        Args:
            query_vector: 查询向量
            limit: 返回结果数量限制
            score_threshold: 相似度阈值
            where: 过滤条件

        Returns:
            List[Dict]: 搜索结果
        """
        try:
            with self._lock:
                if self.index is None or self.index.ntotal == 0:
                    return []

                q = np.asarray(query_vector, dtype=np.float32).reshape(-1)
                if self.vector_size and q.shape[0] != self.vector_size:
                    logger.error(f"❌ 查询向量维度错误: 期望{self.vector_size}, 实际{q.shape[0]}")
                    return []
                norm = float(np.linalg.norm(q))
                if norm > 0:
                    q = q / norm

                allowed = self._allowed_ids(where)
                k = int(self.index.ntotal)
                scores, int_ids = self.index.search(q.reshape(1, -1), k)

                results = []
                for score, iid in zip(scores[0], int_ids[0]):
                    if iid < 0:
                        continue
                    sid = self._id_map.get(int(iid))
                    if sid is None:
                        continue
                    if allowed is not None and sid not in allowed:
                        continue
                    if score_threshold is not None and float(score) < score_threshold:
                        continue

                    results.append({
                        "id": sid,
                        "score": float(score),
                        "metadata": self._meta.get(sid, {}),
                    })
                    if len(results) >= limit:
                        break

                logger.debug(f"🔍 Faiss搜索返回 {len(results)} 个结果")
                return results

        except Exception as e:
            logger.error(f"❌ 向量搜索失败: {e}")
            return []

    def delete_vectors(self, ids: List[str]) -> bool:
        """
        删除向量

        Args:
            ids: 要删除的向量ID列表

        Returns:
            bool: 是否成功
        """
        try:
            if not ids:
                return True

            with self._lock:
                for sid in list(ids):
                    if str(sid) in self._str_to_int:
                        self._remove_sid(str(sid))
                if self.auto_persist:
                    self.persist()

            logger.info(f"✅ 成功删除 {len(ids)} 个向量")
            return True

        except Exception as e:
            logger.error(f"❌ 删除向量失败: {e}")
            return False

    def clear_collection(self) -> bool:
        """
        清空集合

        Returns:
            bool: 是否成功
        """
        try:
            with self._lock:
                self._meta.clear()
                self._id_map.clear()
                self._str_to_int.clear()
                self._next_int = 0
                if self.vector_size:
                    self._new_index(self.vector_size)
                else:
                    self.index = None
                # 同时清除落盘文件，避免下次重载“复活”
                for p in (self._index_path, self._meta_path):
                    try:
                        if os.path.exists(p):
                            os.remove(p)
                    except Exception:
                        pass

            logger.info(f"✅ 成功清空Faiss集合: {self.collection_name}")
            return True

        except Exception as e:
            logger.error(f"❌ 清空集合失败: {e}")
            return False

    def delete_memories(self, memory_ids: List[str]):
        """
        删除指定记忆（通过payload中的 memory_id 过滤删除）
        """
        try:
            if not memory_ids:
                return
            targets = {str(m) for m in memory_ids}
            with self._lock:
                to_remove = [sid for sid, meta in self._meta.items()
                             if str(meta.get("memory_id")) in targets]
                for sid in to_remove:
                    self._remove_sid(sid)
                if to_remove and self.auto_persist:
                    self.persist()
            logger.info(f"✅ 成功按memory_id删除 {len(to_remove)} 个Faiss向量")
        except Exception as e:
            logger.error(f"❌ 删除记忆失败: {e}")
            raise

    def get_collection_info(self) -> Dict[str, Any]:
        """
        获取集合信息

        Returns:
            Dict: 集合信息
        """
        try:
            n = int(self.index.ntotal) if self.index is not None else 0
            info = {
                "name": self.collection_name,
                "vectors_count": n,
                "indexed_vectors_count": n,
                "points_count": n,
                "config": {
                    "vector_size": self.vector_size,
                    "distance": self.distance,
                },
                "persist_dir": self.persist_dir,
            }
            return info

        except Exception as e:
            logger.error(f"❌ 获取集合信息失败: {e}")
            return {}

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        获取集合统计信息（兼容抽象接口）
        """
        info = self.get_collection_info()
        if not info:
            return {"store_type": "faiss", "name": self.collection_name}
        info["store_type"] = "faiss"
        return info

    def health_check(self) -> bool:
        """
        健康检查

        Returns:
            bool: 本地索引是否可用
        """
        try:
            return self.index is not None
        except Exception as e:
            logger.error(f"❌ Faiss健康检查失败: {e}")
            return False

    def persist(self, path: Optional[str] = None) -> bool:
        """
        落盘：index.faiss + meta.json

        Args:
            path: 可选，自定义落盘目录

        Returns:
            bool: 是否成功
        """
        try:
            with self._lock:
                if self.index is None:
                    return False
                index_path = os.path.join(path, "index.faiss") if path else self._index_path
                meta_path = os.path.join(path, "meta.json") if path else self._meta_path
                os.makedirs(os.path.dirname(os.path.abspath(index_path)), exist_ok=True)
                faiss.write_index(self.index, index_path)
                payload = {
                    "collection_name": self.collection_name,
                    "vector_size": self.vector_size,
                    "distance": self.distance,
                    "next_int": self._next_int,
                    "id_map": {str(k): v for k, v in self._id_map.items()},
                    "meta": self._meta,
                }
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False)
                logger.debug(f"💾 Faiss已落盘: {index_path} (共 {self.index.ntotal} 条)")
                return True
        except Exception as e:
            logger.error(f"❌ Faiss落盘失败: {e}")
            return False

    def load(self, path: Optional[str] = None) -> bool:
        """
        从磁盘重载

        Args:
            path: 可选，自定义重载目录

        Returns:
            bool: 是否成功
        """
        try:
            with self._lock:
                index_path = os.path.join(path, "index.faiss") if path else self._index_path
                meta_path = os.path.join(path, "meta.json") if path else self._meta_path
                self.index = faiss.read_index(index_path)
                with open(meta_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                self.vector_size = payload.get("vector_size") or self.vector_size
                self._next_int = int(payload.get("next_int", 0))
                self._id_map = {int(k): v for k, v in payload.get("id_map", {}).items()}
                self._str_to_int = {v: int(k) for k, v in self._id_map.items()}
                self._meta = payload.get("meta", {}) or {}
                return True
        except Exception as e:
            logger.error(f"❌ Faiss重载失败: {e}")
            return False

    def __del__(self):
        """析构函数：数据已在写入/删除时通过 auto_persist 落盘，此处无需再操作"""
        pass
